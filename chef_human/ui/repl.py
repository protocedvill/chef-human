from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from rich.console import Console
from rich.prompt import Prompt

from chef_human.ui.protocol import ask_via_stdin

if TYPE_CHECKING:
    from chef_human.agent.parser import ParsedToolCall
    from chef_human.agent.planner import Plan
    from chef_human.agent.react_loop import AgentResult


class ReplUI:
    def __init__(self) -> None:
        self._console = Console()
        self._current_task: str = ""

    def _status_icon(self, status: str) -> str:
        icons = {
            "pending": "○",
            "in_progress": "◷",
            "completed": "✓",
            "failed": "✗",
            "skipped": "–",
        }
        return icons.get(status, "○")

    def on_start(self, task: str) -> None:
        self._current_task = task
        self._console.print(f"\n[bold cyan]You:[/] {task}\n")

    def on_planning_start(self) -> None:
        self._console.print("[dim]Planning...[/]")

    def on_plan(self, plan: Plan) -> None:
        self._console.print(f"[bold yellow]Plan:[/] {plan.goal}")
        for step in plan.steps:
            icon = self._status_icon(step.status.value if hasattr(step.status, 'value') else step.status)
            self._console.print(f"  {icon} {step.description}")

    def on_reasoning_start(self) -> None:
        pass

    def on_stream(self, chunk: str) -> None:
        sys.stdout.write(chunk)
        sys.stdout.flush()

    def on_reasoning(self, content: str) -> None:
        if content:
            self._console.print(f"\n[dim]{content}[/]")

    def on_tool_call(self, tool_call: ParsedToolCall) -> None:
        args = ", ".join(
            f"{k}={v}" for k, v in tool_call.arguments.items()
        )[:120]
        self._console.print(f"  [bold green]▶[/] [cyan]{tool_call.name}[/]({args})")

    def on_tool_result(self, name: str, result: str) -> None:
        icon = "✗" if result.startswith("Error:") else "✓"
        summary = result[:100].replace("\n", " ").strip()
        self._console.print(f"  {icon} [bold]{name}[/] → {summary}")

    def on_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        pass

    async def on_ask_user(self, question: str) -> str:
        return await ask_via_stdin(question)

    def on_replan(self) -> None:
        self._console.print("[bold yellow]↻ Re-planning...[/]")

    def on_error(self, message: str) -> None:
        self._console.print(f"[bold red]Error:[/] {message}")

    async def on_approval_request(self, tool_call: ParsedToolCall) -> bool | None:
        return None

    def display_result(self, result: AgentResult) -> None:
        status = "[bold green]✓ Success[/]" if result.success else "[bold red]✗ Failed[/]"
        self._console.print(f"\n[bold]Result:[/] {status}")
        self._console.print(f"  [dim]Steps:[/] {result.steps_taken}")
        if result.message:
            self._console.print(f"  [dim]Message:[/] {result.message[:200]}")
        if result.total_prompt_tokens or result.total_completion_tokens:
            self._console.print(
                f"  [dim]Tokens:[/] {result.total_prompt_tokens:,} prompt / "
                f"{result.total_completion_tokens:,} completion"
            )

    def read_input(self) -> str | None:
        try:
            text = Prompt.ask("\n[bold cyan]You[/]")
        except (EOFError, KeyboardInterrupt):
            return None

        if not text:
            return ""

        if text.startswith("/"):
            cmd = text[1:].lower().strip()
            if cmd in ("exit", "quit", "q"):
                return None
            elif cmd == "help":
                self._print_help()
                return ""
            elif cmd in ("clear", "save", "tokens", "history", "undo", "redo"):
                return text
            else:
                self._console.print(f"[yellow]Unknown command: {text}. Type /help for available commands.[/]")
            return ""

        return text

    def _print_help(self) -> None:
        self._console.print("""
[bold cyan]Available commands:[/]
  /exit, /quit, /q    Exit the REPL
  /help               Show this message
  /save               Save the current session
  /clear              Clear conversation history
  /undo               Undo the last change
  /redo               Redo the last undone change
  /tokens             Show token usage
  /history            Show recent messages

Any other input is sent to the agent as a task.
""")
