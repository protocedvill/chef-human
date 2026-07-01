from __future__ import annotations

import pytest

from chef_human.agent.retry import RetryAction, RetryManager, RetryState


class TestRetryState:
    def test_defaults(self):
        s = RetryState()
        assert s.consecutive_failures == 0
        assert s.replan_count == 0
        assert s.tool_results == []


class TestRetryManagerInit:
    def test_defaults(self):
        mgr = RetryManager()
        assert mgr.consecutive_failures == 0
        assert mgr.replan_count == 0
        assert mgr.tool_results == []

    def test_custom_values(self):
        mgr = RetryManager(max_retries_per_step=5, max_replans=2)
        assert mgr._max_retries == 5
        assert mgr._max_replans == 2

    def test_invalid_max_retries(self):
        with pytest.raises(ValueError, match="max_retries_per_step"):
            RetryManager(max_retries_per_step=0)

    def test_invalid_max_replans(self):
        with pytest.raises(ValueError, match="max_replans"):
            RetryManager(max_replans=-1)

    def test_zero_replans_disables_escalation(self):
        mgr = RetryManager(max_replans=0)
        assert mgr._max_replans == 0


class TestRetryAction:
    def test_constants_are_strings(self):
        assert RetryAction.RETRY == "retry"
        assert RetryAction.REPLAN == "replan"
        assert RetryAction.ESCALATE == "escalate"
        assert RetryAction.STEP_COMPLETED == "step_completed"


class TestRecordIteration:
    def test_step_completed_when_all_success(self):
        mgr = RetryManager(max_retries_per_step=3)
        action = mgr.record_iteration(True, ["ok"])
        assert action == RetryAction.STEP_COMPLETED
        assert mgr.consecutive_failures == 0

    def test_retry_after_first_failure(self):
        mgr = RetryManager(max_retries_per_step=3)
        action = mgr.record_iteration(False, ["error: x"])
        assert action == RetryAction.RETRY
        assert mgr.consecutive_failures == 1
        assert mgr.tool_results == ["error: x"]

    def test_retry_after_second_failure(self):
        mgr = RetryManager(max_retries_per_step=3)
        mgr.record_iteration(False, ["err1"])
        action = mgr.record_iteration(False, ["err2"])
        assert action == RetryAction.RETRY
        assert mgr.consecutive_failures == 2

    def test_replan_after_max_retries(self):
        mgr = RetryManager(max_retries_per_step=3)
        mgr.record_iteration(False, ["e1"])
        mgr.record_iteration(False, ["e2"])
        action = mgr.record_iteration(False, ["e3"])
        assert action == RetryAction.REPLAN
        assert mgr.consecutive_failures == 3
        assert mgr.replan_count == 0  # not incremented until on_replan

    def test_escalate_after_replan_fails(self):
        mgr = RetryManager(max_retries_per_step=2, max_replans=1)
        # First round: 2 failures → replan
        mgr.record_iteration(False, ["e1"])
        action = mgr.record_iteration(False, ["e2"])
        assert action == RetryAction.REPLAN
        mgr.on_replan()
        assert mgr.replan_count == 1
        assert mgr.consecutive_failures == 0

        # Second round (new plan): 2 more failures → escalate
        mgr.record_iteration(False, ["e3"])
        action = mgr.record_iteration(False, ["e4"])
        assert action == RetryAction.ESCALATE
        assert mgr.replan_count == 1

    def test_success_resets_failures_before_max(self):
        mgr = RetryManager(max_retries_per_step=3)
        mgr.record_iteration(False, ["e1"])
        mgr.record_iteration(False, ["e2"])
        mgr.record_iteration(True, ["ok"])
        assert mgr.consecutive_failures == 0
        assert mgr.tool_results == []

    def test_success_after_one_failure(self):
        mgr = RetryManager(max_retries_per_step=3)
        mgr.record_iteration(False, ["e1"])
        action = mgr.record_iteration(True, ["ok"])
        assert action == RetryAction.STEP_COMPLETED

    def test_tool_results_accumulate_across_failures(self):
        mgr = RetryManager(max_retries_per_step=3)
        mgr.record_iteration(False, ["err: read failed"])
        mgr.record_iteration(False, ["err: write failed"])
        results = mgr.tool_results
        assert len(results) == 2
        assert results[0] == "err: read failed"
        assert results[1] == "err: write failed"

    def test_tool_results_cleared_on_success(self):
        mgr = RetryManager(max_retries_per_step=3)
        mgr.record_iteration(False, ["err"])
        mgr.record_iteration(True, ["ok"])
        assert mgr.tool_results == []

    def test_escalate_with_zero_max_replans(self):
        """max_replans=0 means escalate immediately after max_retries."""
        mgr = RetryManager(max_retries_per_step=2, max_replans=0)
        mgr.record_iteration(False, ["e1"])
        action = mgr.record_iteration(False, ["e2"])
        assert action == RetryAction.ESCALATE
        assert mgr.replan_count == 0

    def test_multiple_replans_allowed_with_higher_max(self):
        mgr = RetryManager(max_retries_per_step=2, max_replans=2)
        # Round 1: 2 fails → replan
        mgr.record_iteration(False, ["e1"])
        mgr.record_iteration(False, ["e2"])
        mgr.on_replan()
        # Round 2: 2 fails → replan again
        mgr.record_iteration(False, ["e3"])
        mgr.record_iteration(False, ["e4"])
        mgr.on_replan()
        assert mgr.replan_count == 2
        # Round 3: 2 fails → escalate
        mgr.record_iteration(False, ["e5"])
        action = mgr.record_iteration(False, ["e6"])
        assert action == RetryAction.ESCALATE


class TestOnReplan:
    def test_resets_state(self):
        mgr = RetryManager(max_retries_per_step=3)
        mgr.record_iteration(False, ["err"])
        mgr.record_iteration(False, ["err"])
        mgr.record_iteration(False, ["err"])
        assert mgr.consecutive_failures == 3
        mgr.on_replan()
        assert mgr.consecutive_failures == 0
        assert mgr.tool_results == []
        assert mgr.replan_count == 1

    def test_increments_replan_count(self):
        mgr = RetryManager()
        assert mgr.replan_count == 0
        mgr.on_replan()
        assert mgr.replan_count == 1
        mgr.on_replan()
        assert mgr.replan_count == 2
