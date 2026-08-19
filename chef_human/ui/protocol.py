from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from chef_human.agent.planner import Plan, PlanNode
    from chef_human.agent.parser import ParsedToolCall


@dataclass
class PlanReviewAction:
    """One action from the one-time pre-execution whole-tree review pass
    (ticket 07). `kind` is one of "approve", "reject", "edit", "mark_leaf",
    "redecompose". `node_id` addresses the target node for every kind
    except "approve"/"reject" (whole-tree actions). `description` is used
    by "edit", `guidance` by "redecompose"."""

    kind: str
    node_id: str | None = None
    description: str | None = None
    guidance: str | None = None


PLAN_REVIEW_HELP_TEXT = (
    "Flagged nodes (⚑) are ones the planner was unsure how to decompose.\n"
    "  [a] approve as-is\n"
    "  [e <id> <new description>] edit a node's description\n"
    "  [l <id>] mark node as leaf (discard its children)\n"
    "  [r <id> <guidance>] redecompose a branch with guidance\n"
    "  [q] reject, abandon task"
)


def parse_plan_review_command(line: str) -> PlanReviewAction:
    """Parses one line of the shared plan-review command grammar (`a`,
    `e <id> <desc>`, `l <id>`, `r <id> <guidance>`, `q`) into a
    `PlanReviewAction`. Shared by every UI's `on_plan_review` (stdin-based
    and TUI-input-based alike) so the grammar only has one implementation to
    keep in sync. Returns `kind="noop"` for anything unrecognized, leaving
    the plan unchanged and letting the caller re-prompt."""
    line = line.strip()
    if not line or line.lower() == "a":
        return PlanReviewAction(kind="approve")
    if line.lower() == "q":
        return PlanReviewAction(kind="reject")

    parts = line.split(maxsplit=2)
    cmd = parts[0].lower()
    if cmd == "e" and len(parts) == 3:
        return PlanReviewAction(kind="edit", node_id=parts[1], description=parts[2])
    if cmd == "l" and len(parts) >= 2:
        return PlanReviewAction(kind="mark_leaf", node_id=parts[1])
    if cmd == "r" and len(parts) == 3:
        return PlanReviewAction(kind="redecompose", node_id=parts[1], guidance=parts[2])

    return PlanReviewAction(kind="noop")


async def review_plan_via_stdin(plan: "Plan") -> PlanReviewAction:
    """Blocking print()+stdin.readline() whole-tree review, for UIs that run
    in a plain terminal -- mirrors `ask_via_stdin`'s no-tty short-circuit so
    headless/benchmark runs (no attached tty) approve immediately and never
    block on a human that isn't there."""
    if not sys.stdin.isatty():
        return PlanReviewAction(kind="approve")

    from chef_human.agent.planner import Planner

    print("\n" + Planner.format_full_tree(plan))
    print(f"\nReview the plan above before execution starts. {PLAN_REVIEW_HELP_TEXT}")
    print("> ", end="", flush=True)
    try:
        line = sys.stdin.readline().strip()
    except (EOFError, KeyboardInterrupt):
        line = "a"

    action = parse_plan_review_command(line)
    if action.kind == "noop":
        print(f"Unrecognized command: {line!r}; leaving the plan unchanged.")
    return action


async def ask_via_stdin(question: str) -> str:
    """Blocking print()+stdin.readline() question/answer, for UIs that run
    in a plain terminal (not one a TUI framework has put into raw/alternate-
    screen mode, where a synchronous stdin read can't receive real input and
    would block the whole event loop -- see TuiUI.on_ask_user for that
    case)."""
    if not sys.stdin.isatty():
        return (
            "[no-tty] Cannot ask user in non-interactive mode. No answer is "
            "coming; do not ask again. Proceed using the current plan and your "
            "best judgment."
        )
    print(f"\n[Agent asks]: {question}")
    print("[Type your response, or 'skip' to continue without answering]: ", end="", flush=True)
    try:
        response = sys.stdin.readline().strip()
    except (EOFError, KeyboardInterrupt):
        response = ""
    if not response or response.lower() == "skip":
        return "User skipped the question"
    return response


class ReActUI(Protocol):
    def on_start(self, task: str) -> None: ...
    def on_planning_start(self) -> None: ...
    def on_plan(self, plan: Plan) -> None: ...
    def on_reasoning_start(self) -> None: ...
    def on_stream(self, chunk: str) -> None: ...
    def on_reasoning(self, content: str) -> None: ...
    def on_tool_call(self, tool_call: ParsedToolCall) -> None: ...
    def on_tool_result(self, name: str, result: str) -> None: ...
    def on_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None: ...
    def on_replan(self) -> None: ...
    def on_error(self, message: str) -> None: ...
    def on_llm_start(self, activity: str) -> None: ...
    def on_llm_end(self) -> None: ...
    def on_plan_progress(self, plan: Plan) -> None: ...
    def on_escalation(self, node: "PlanNode", message: str) -> None: ...

    async def on_ask_user(self, question: str) -> str:
        return await ask_via_stdin(question)

    async def on_approval_request(
        self, tool_call: ParsedToolCall
    ) -> bool | None:
        return None

    async def on_plan_review(self, plan: Plan) -> PlanReviewAction:
        return await review_plan_via_stdin(plan)


class NoopUI:
    def on_start(self, task: str) -> None: ...
    def on_planning_start(self) -> None: ...
    def on_plan(self, plan: Plan) -> None: ...
    def on_reasoning_start(self) -> None: ...
    def on_stream(self, chunk: str) -> None: ...
    def on_reasoning(self, content: str) -> None: ...
    def on_tool_call(self, tool_call: ParsedToolCall) -> None: ...
    def on_tool_result(self, name: str, result: str) -> None: ...
    def on_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None: ...
    def on_replan(self) -> None: ...
    def on_error(self, message: str) -> None: ...
    def on_llm_start(self, activity: str) -> None: ...
    def on_llm_end(self) -> None: ...
    def on_plan_progress(self, plan: Plan) -> None: ...
    def on_escalation(self, node: "PlanNode", message: str) -> None: ...

    async def on_ask_user(self, question: str) -> str:
        return await ask_via_stdin(question)

    async def on_approval_request(
        self, tool_call: ParsedToolCall
    ) -> bool | None:
        return None

    async def on_plan_review(self, plan: Plan) -> PlanReviewAction:
        return await review_plan_via_stdin(plan)
