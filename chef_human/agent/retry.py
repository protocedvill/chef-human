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
    def __init__(self, max_retries_per_step: int = 3, max_replans: int = 1) -> None:
        if max_retries_per_step < 1:
            raise ValueError("max_retries_per_step must be >= 1")
        if max_replans < 0:
            raise ValueError("max_replans must be >= 0")
        self._max_retries = max_retries_per_step
        self._max_replans = max_replans
        self._state = RetryState()

    @property
    def consecutive_failures(self) -> int:
        return self._state.consecutive_failures

    @property
    def replan_count(self) -> int:
        return self._state.replan_count

    @property
    def tool_results(self) -> list[str]:
        return self._state.tool_results

    def record_iteration(
        self, total_calls: int, failed_calls: int, tool_results: list[str]
    ) -> RetryAction:
        if failed_calls == 0:
            self._state.consecutive_failures = 0
            self._state.tool_results = []
            return RetryAction.STEP_COMPLETED

        self._state.tool_results.extend(tool_results)
        self._state.consecutive_failures += 1

        if self._state.consecutive_failures >= self._max_retries:
            if self._state.replan_count >= self._max_replans:
                logger.warning(
                    "Escalating after %d replans and %d consecutive failures",
                    self._state.replan_count,
                    self._state.consecutive_failures,
                )
                return RetryAction.ESCALATE
            return RetryAction.REPLAN

        if failed_calls < total_calls:
            return RetryAction.PARTIAL_SUCCESS

        return RetryAction.RETRY

    def on_replan(self) -> None:
        self._state.replan_count += 1
        self._state.consecutive_failures = 0
        self._state.tool_results = []


class RetryAction(StrEnum):
    RETRY = "retry"
    PARTIAL_SUCCESS = "partial_success"
    REPLAN = "replan"
    ESCALATE = "escalate"
    STEP_COMPLETED = "step_completed"
