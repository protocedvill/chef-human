from __future__ import annotations

import pytest

from chef_human.agent.retry import RetryAction, RetryManager, RetryState

# Helper: all-success shortcut (0 failed out of 0 calls)
_OK = (0, 0, ["ok"])
# Helper: all-fail shortcut (1 failed out of 1 call)
_FAIL = (1, 1, ["err"])


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
        action = mgr.record_iteration(*_OK)
        assert action == RetryAction.STEP_COMPLETED
        assert mgr.consecutive_failures == 0

    def test_retry_after_first_failure(self):
        mgr = RetryManager(max_retries_per_step=3)
        action = mgr.record_iteration(*_FAIL)
        assert action == RetryAction.RETRY
        assert mgr.consecutive_failures == 1
        assert mgr.tool_results == ["err"]

    def test_retry_after_second_failure(self):
        mgr = RetryManager(max_retries_per_step=3)
        mgr.record_iteration(*_FAIL)
        action = mgr.record_iteration(*_FAIL)
        assert action == RetryAction.RETRY
        assert mgr.consecutive_failures == 2

    def test_replan_after_max_retries(self):
        mgr = RetryManager(max_retries_per_step=3)
        mgr.record_iteration(*_FAIL)
        mgr.record_iteration(*_FAIL)
        action = mgr.record_iteration(*_FAIL)
        assert action == RetryAction.REPLAN
        assert mgr.consecutive_failures == 3
        assert mgr.replan_count == 0  # not incremented until on_replan

    def test_escalate_after_replan_fails(self):
        mgr = RetryManager(max_retries_per_step=2, max_replans=1)
        # First round: 2 failures → replan
        mgr.record_iteration(*_FAIL)
        action = mgr.record_iteration(*_FAIL)
        assert action == RetryAction.REPLAN
        mgr.on_replan()
        assert mgr.replan_count == 1
        assert mgr.consecutive_failures == 0

        # Second round (new plan): 2 more failures → escalate
        mgr.record_iteration(*_FAIL)
        action = mgr.record_iteration(*_FAIL)
        assert action == RetryAction.ESCALATE
        assert mgr.replan_count == 1

    def test_success_resets_failures_before_max(self):
        mgr = RetryManager(max_retries_per_step=3)
        mgr.record_iteration(*_FAIL)
        mgr.record_iteration(*_FAIL)
        mgr.record_iteration(*_OK)
        assert mgr.consecutive_failures == 0
        assert mgr.tool_results == []

    def test_success_after_one_failure(self):
        mgr = RetryManager(max_retries_per_step=3)
        mgr.record_iteration(*_FAIL)
        action = mgr.record_iteration(*_OK)
        assert action == RetryAction.STEP_COMPLETED

    def test_tool_results_accumulate_across_failures(self):
        mgr = RetryManager(max_retries_per_step=3)
        mgr.record_iteration(1, 1, ["err: read failed"])
        mgr.record_iteration(1, 1, ["err: write failed"])
        results = mgr.tool_results
        assert len(results) == 2
        assert results[0] == "err: read failed"
        assert results[1] == "err: write failed"

    def test_tool_results_cleared_on_success(self):
        mgr = RetryManager(max_retries_per_step=3)
        mgr.record_iteration(*_FAIL)
        mgr.record_iteration(*_OK)
        assert mgr.tool_results == []

    def test_escalate_with_zero_max_replans(self):
        """max_replans=0 means escalate immediately after max_retries."""
        mgr = RetryManager(max_retries_per_step=2, max_replans=0)
        mgr.record_iteration(*_FAIL)
        action = mgr.record_iteration(*_FAIL)
        assert action == RetryAction.ESCALATE
        assert mgr.replan_count == 0

    def test_multiple_replans_allowed_with_higher_max(self):
        mgr = RetryManager(max_retries_per_step=2, max_replans=2)
        # Round 1: 2 fails → replan
        mgr.record_iteration(*_FAIL)
        mgr.record_iteration(*_FAIL)
        mgr.on_replan()
        # Round 2: 2 fails → replan again
        mgr.record_iteration(*_FAIL)
        mgr.record_iteration(*_FAIL)
        mgr.on_replan()
        assert mgr.replan_count == 2
        # Round 3: 2 fails → escalate
        mgr.record_iteration(*_FAIL)
        action = mgr.record_iteration(*_FAIL)
        assert action == RetryAction.ESCALATE

    def test_partial_success_returns_partial_action(self):
        """When some calls succeed and some fail, returns PARTIAL_SUCCESS."""
        mgr = RetryManager(max_retries_per_step=3)
        action = mgr.record_iteration(3, 1, ["err: one failed"])
        assert action == RetryAction.PARTIAL_SUCCESS
        assert mgr.consecutive_failures == 1

    def test_partial_success_preserves_tool_results(self):
        """PARTIAL_SUCCESS keeps tool_results for the LLM to see."""
        mgr = RetryManager(max_retries_per_step=3)
        mgr.record_iteration(3, 1, ["ok1", "err: write failed", "ok2"])
        assert mgr.tool_results == ["ok1", "err: write failed", "ok2"]

    def test_partial_success_counts_as_failure_for_retry_limit(self):
        """PARTIAL_SUCCESS increments the failure counter."""
        mgr = RetryManager(max_retries_per_step=2)
        mgr.record_iteration(2, 1, ["ok", "err"])
        mgr.record_iteration(2, 1, ["ok", "err"])
        action = mgr.record_iteration(2, 1, ["ok", "err"])
        assert action == RetryAction.REPLAN
        assert mgr.consecutive_failures == 3

    def test_partial_success_runs_after_max_retries_then_replan(self):
        """Partial success triggers replan after max_retries."""
        mgr = RetryManager(max_retries_per_step=2, max_replans=1)
        mgr.record_iteration(2, 1, ["ok", "err"])
        mgr.record_iteration(2, 1, ["ok", "err"])
        action = mgr.record_iteration(2, 1, ["ok", "err"])
        assert action == RetryAction.REPLAN

    def test_partial_success_accepts_partial_success_as_new_action(self):
        """PARTIAL_SUCCESS is a valid RetryAction enum value."""
        assert RetryAction.PARTIAL_SUCCESS == "partial_success"


class TestOnReplan:
    def test_resets_state(self):
        mgr = RetryManager(max_retries_per_step=3)
        mgr.record_iteration(*_FAIL)
        mgr.record_iteration(*_FAIL)
        mgr.record_iteration(*_FAIL)
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
