from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

logger = logging.getLogger(__name__)


@dataclass
class RetryState:
    consecutive_failures: int = 0
    replan_count: int = 0
    tool_results: list[str] = field(default_factory=list)


class RetryManager:
    """Tracks retry/replan pressure per plan node (`node_id`), so one
    struggling branch's failures can't leak into an unrelated branch and a
    sibling's replan can't reset a different node's accumulated count.

    Deliberately knows nothing about `Plan`/`PlanNode` shape -- callers pass
    a bare `node_id` string, and `ReActLoop` is the only place that maps the
    returned `RetryAction` onto actual tree mutations (e.g. marking a node
    `failed`)."""

    def __init__(self, max_retries_per_step: int = 3, max_replans: int = 1) -> None:
        if max_retries_per_step < 1:
            raise ValueError("max_retries_per_step must be >= 1")
        if max_replans < 0:
            raise ValueError("max_replans must be >= 0")
        self._max_retries = max_retries_per_step
        self._max_replans = max_replans
        self._states: dict[str, RetryState] = {}

    def _state_for(self, node_id: str) -> RetryState:
        state = self._states.get(node_id)
        if state is None:
            state = RetryState()
            self._states[node_id] = state
        return state

    def consecutive_failures(self, node_id: str) -> int:
        return self._state_for(node_id).consecutive_failures

    def replan_count(self, node_id: str) -> int:
        return self._state_for(node_id).replan_count

    def tool_results(self, node_id: str) -> list[str]:
        return self._state_for(node_id).tool_results

    def record_iteration(
        self, node_id: str, total_calls: int, failed_calls: int, tool_results: list[str]
    ) -> RetryAction:
        state = self._state_for(node_id)

        if failed_calls == 0:
            state.consecutive_failures = 0
            state.tool_results = []
            return RetryAction.STEP_COMPLETED

        state.tool_results.extend(tool_results)
        state.consecutive_failures += 1

        if state.consecutive_failures >= self._max_retries:
            if state.replan_count >= self._max_replans:
                logger.warning(
                    "Escalating node %s after %d replans and %d consecutive failures",
                    node_id,
                    state.replan_count,
                    state.consecutive_failures,
                )
                return RetryAction.ESCALATE
            return RetryAction.REPLAN

        if failed_calls < total_calls:
            return RetryAction.PARTIAL_SUCCESS

        return RetryAction.RETRY

    def on_replan(self, node_id: str) -> None:
        state = self._state_for(node_id)
        state.replan_count += 1
        state.consecutive_failures = 0
        state.tool_results = []


class RetryAction(StrEnum):
    RETRY = "retry"
    PARTIAL_SUCCESS = "partial_success"
    REPLAN = "replan"
    ESCALATE = "escalate"
    STEP_COMPLETED = "step_completed"
