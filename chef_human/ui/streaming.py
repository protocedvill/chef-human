from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from rich.console import Console

from chef_human.agent.planner import StepStatus

if TYPE_CHECKING:
    from chef_human.agent.parser import ParsedToolCall
    from chef_human.agent.planner import Plan

_STATUS_ICONS = {
    StepStatus.pending: "○",
    StepStatus.in_progress: "◷",
    StepStatus.completed: "✓",
    StepStatus.failed: "✗",
    StepStatus.skipped: "–",
}


class StreamingUI:
    def __init__(self, quiet: bool = False) -> None:
        self._console = Console()
        self._quiet = quiet
        self._reasoning_started = False

    def _get_icon(self, status: StepStatus) -> str:
        return _STATUS_ICONS.get(status, "○")

    def on_start(self, task: str) -> None:
        if not self._quiet:
            self._console.print(f"\n[bold]Task:[/] {task}\n")

    def on_planning_start(self) -> None:
        if not self._quiet:
            self._console.print("[dim]Planning...[/]")

    def on_plan(self, plan: Plan) -> None:
        if self._quiet:
            return
        self._console.print("[bold yellow]Plan:[/]")
        for step in plan.steps:
            icon = self._get_icon(step.status)
            self._console.print(f"  {icon} {step.description}")
        self._console.print()

    def on_reasoning_start(self) -> None:
        if self._quiet:
            return
        self._reasoning_started = True
        self._console.print("[dim]Thinking...[/]")

    def on_stream(self, chunk: str) -> None:
        if self._quiet:
            return
        if self._reasoning_started:
            self._reasoning_started = False
        sys.stdout.write(chunk)
        sys.stdout.flush()

    def on_reasoning(self, content: str) -> None:
        if self._quiet or not content:
            return
        self._reasoning_started = False
        self._console.print(f"\n[dim]{content}[/]\n")

    def on_tool_call(self, tool_call: ParsedToolCall) -> None:
        if self._quiet:
            return
        args = ", ".join(
            f"{k}={v}" for k, v in tool_call.arguments.items()
        )[:120]
        self._console.print(f"  [bold cyan]▶[/] [green]{tool_call.name}[/]({args})")

    def on_tool_result(self, name: str, result: str) -> None:
        if self._quiet:
            return
        icon = "✗" if result.startswith("Error:") else "✓"
        summary = result[:100].replace("\n", " ").strip()
        self._console.print(f"  {icon} [bold]{name}[/] → {summary}")

    def on_replan(self) -> None:
        if not self._quiet:
            self._console.print("[bold yellow]↻ Re-planning...[/]")

    def on_error(self, message: str) -> None:
        self._console.print(f"[bold red]Error:[/] {message}")

    async def on_approval_request(
        self, tool_call: ParsedToolCall
    ) -> bool | None:
        return None
