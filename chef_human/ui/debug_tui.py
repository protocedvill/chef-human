from __future__ import annotations

import signal
import sys
import time
from datetime import datetime
from typing import TYPE_CHECKING

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.tree import Tree

from chef_human.agent.planner import StepStatus

if TYPE_CHECKING:
    from chef_human.agent.parser import ParsedToolCall
    from chef_human.agent.planner import Plan
    from chef_human.agent.react_loop import AgentResult

_STATUS_STYLES = {
    StepStatus.pending: "white",
    StepStatus.in_progress: "bold yellow",
    StepStatus.completed: "green",
    StepStatus.failed: "bold red",
    StepStatus.skipped: "dim white",
}


class DebugTUI:
    """Rich-based debug terminal UI for the ReAct loop."""

    def __init__(self, max_reasoning_lines: int = 50) -> None:
        self.console = Console()
        self.layout = Layout()
        self._setup_layout()
        self._live = Live(self.layout, refresh_per_second=4, screen=True)
        self._started = False
        self._plan: Plan | None = None
        self._reasoning_text = ""
        self._reasoning_collapsed = False
        self._max_reasoning_lines = max_reasoning_lines
        self._tool_calls: list[tuple[str, str, str]] = []
        self._log_entries: list[str] = []
        self._log_search: str | None = None
        self._step_count = 0
        self._max_steps = 0
        self._start_time = 0.0
        self._last_key_check = 0.0
        self._task = ""
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._setup_signal_handler()

    def _setup_signal_handler(self) -> None:
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, sig: int, frame: object) -> None:
        self._stop_live()
        self.console.print("\n[bold yellow]Task interrupted by user.[/]")
        sys.exit(130)

    def _setup_layout(self) -> None:
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        self.layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1),
        )
        self.layout["left"].split_column(
            Layout(name="plan_panel"),
            Layout(name="tool_panel"),
        )
        self.layout["right"].split_column(
            Layout(name="reasoning_panel"),
            Layout(name="log_panel"),
        )

    def _ensure_live(self) -> None:
        if not self._started:
            self._live.__enter__()
            self._started = True

    def _stop_live(self) -> None:
        if self._started:
            self._live.__exit__(None, None, None)
            self._started = False

    def _check_keys(self) -> None:
        if not sys.stdin.isatty():
            return
        now = time.time()
        if now - self._last_key_check < 0.1:
            return
        self._last_key_check = now
        import select
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key == "r":
                    self._reasoning_collapsed = not self._reasoning_collapsed
                    self._refresh_reasoning()
                elif key == "/":
                    self._prompt_search()
        except (ValueError, TypeError):
            pass

    def _prompt_search(self) -> None:
        self._stop_live()
        try:
            query = Prompt.ask("Search log", default="")
        except (EOFError, KeyboardInterrupt):
            query = ""
        self._log_search = query if query else None
        self._ensure_live()
        self._refresh_log()

    def _render_plan(self) -> Tree:
        tree = Tree("📋 Plan", guide_style="dim")
        if self._plan is None:
            return tree
        for step in self._plan.steps:
            style = _STATUS_STYLES.get(step.status, "white")
            label = f"[{style}]{step.index}. {step.description}[/{style}]"
            tree.add(label)
        return tree

    def _render_reasoning(self) -> Panel:
        if self._reasoning_collapsed:
            lines = self._reasoning_text.splitlines()
            if len(lines) > 5:
                display = "\n".join(lines[-5:])
                display = f"... ({len(lines) - 5} more lines) ...\n" + display
            else:
                display = self._reasoning_text
        else:
            display = self._reasoning_text
        return Panel(display, title="Reasoning")

    def _render_log(self) -> Panel:
        from rich.text import Text
        text = Text()
        for entry in self._log_entries[-20:]:
            if self._log_search and self._log_search.lower() in entry.lower():
                text.append(entry + "\n", style="reverse")
            else:
                text.append(entry + "\n")
        title = "Log"
        if self._log_search:
            title += f" (search: {self._log_search})"
        return Panel(text, title=title, border_style="dim")

    def _render_footer(self) -> Panel:
        elapsed = time.time() - self._start_time
        token_str = ""
        if self._total_prompt_tokens or self._total_completion_tokens:
            token_str = (
                f" | Tokens: {self._total_prompt_tokens:,}↑ {self._total_completion_tokens:,}↓"
            )
        text = (
            f"Steps: {self._step_count}/{self._max_steps or '?'} | "
            f"Elapsed: {elapsed:.0f}s{token_str} | "
            f"[dim]r[/dim] toggle reasoning  [dim]/[/dim] search  [dim]Ctrl+C[/dim] quit"
        )
        return Panel(text, border_style="dim")

    def _refresh_all(self) -> None:
        if self._plan is not None:
            self.layout["plan_panel"].update(Panel(self._render_plan(), title="Plan"))
        self._refresh_reasoning()
        self._refresh_log()
        self.layout["footer"].update(self._render_footer())

    def _refresh_reasoning(self) -> None:
        self.layout["reasoning_panel"].update(self._render_reasoning())

    def _refresh_log(self) -> None:
        self.layout["log_panel"].update(self._render_log())

    def on_start(self, task: str) -> None:
        self._ensure_live()
        self._start_time = time.time()
        self._task = task
        self._log(f"Task started: {task}")
        header = Panel(
            f"[bold cyan]chef-human[/] | Step {self._step_count}/{self._max_steps} | Task: {task[:60]}",
            style="white on dark_blue",
        )
        self.layout["header"].update(header)
        self.layout["footer"].update(self._render_footer())

    def on_planning_start(self) -> None:
        self._ensure_live()
        self._log("Planning...")

    def on_plan(self, plan: Plan) -> None:
        self._ensure_live()
        self._plan = plan
        self._max_steps = len(plan.steps)
        self.layout["plan_panel"].update(Panel(self._render_plan(), title="Plan"))
        self._log(f"Plan generated: {len(plan.steps)} steps")
        self.layout["footer"].update(self._render_footer())

    def on_reasoning_start(self) -> None:
        self._ensure_live()
        self._check_keys()
        self._reasoning_text = ""
        self.layout["reasoning_panel"].update(
            Panel("Thinking...", title="Reasoning")
        )

    def on_stream(self, chunk: str) -> None:
        self._ensure_live()
        self._check_keys()
        self._reasoning_text += chunk
        if not self._reasoning_collapsed:
            display = self._reasoning_text[-500:]
            if len(self._reasoning_text) > 500:
                display = "... " + display
            self.layout["reasoning_panel"].update(
                Panel(display, title="Reasoning")
            )
        self.layout["footer"].update(self._render_footer())

    def on_reasoning(self, content: str) -> None:
        self._ensure_live()
        self._check_keys()
        self._reasoning_text = content
        self._refresh_reasoning()
        self._log("Model reasoning received")
        self.layout["footer"].update(self._render_footer())

    def on_tool_call(self, tool_call: ParsedToolCall) -> None:
        self._ensure_live()
        self._check_keys()
        args_str = ", ".join(f"{k}={v}" for k, v in tool_call.arguments.items())
        self._tool_calls.append(("▶", tool_call.name, args_str))
        self._refresh_tool_panel()
        self._log(f"Tool call: {tool_call.name}({args_str})")
        self.layout["footer"].update(self._render_footer())

    def on_tool_result(self, name: str, result: str) -> None:
        self._ensure_live()
        status = "✓" if not result.startswith("Error") else "✗"
        result_preview = result[:80] + ("..." if len(result) > 80 else "")
        self._tool_calls.append((status, name, result_preview))
        self._refresh_tool_panel()
        self._log(f"Tool result: {name} -> {result_preview}")
        self.layout["footer"].update(self._render_footer())

    def on_replan(self) -> None:
        self._ensure_live()
        self._check_keys()
        self._log("[yellow]Re-planning...[/]")

    def on_error(self, message: str) -> None:
        self._ensure_live()
        self._check_keys()
        self._log(f"[red]Error: {message}[/]")

    async def on_approval_request(self, tool_call: ParsedToolCall) -> bool:
        self._stop_live()
        cmd = tool_call.arguments.get("command", "")
        approved = Confirm.ask(
            f"\n[bold yellow]Destructive operation requested:[/]\n"
            f"  [red]{cmd}[/]\n"
            f"Approve?"
        )
        self._ensure_live()
        return approved

    def _refresh_tool_panel(self) -> None:
        table = Table(show_header=False, box=None)
        table.add_column("icon", width=2)
        table.add_column("name", width=15)
        table.add_column("detail", width=60)
        for icon, name, detail in self._tool_calls[-20:]:
            table.add_row(icon, name, detail)
        self.layout["tool_panel"].update(Panel(table, title="Tool Calls"))

    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._log_entries.append(f"[{timestamp}] {message}")
        self._refresh_log()

    def display_final(self, result: AgentResult) -> None:
        self._stop_live()
        self.console.print("\n[bold]=== Task Complete ===[/]")
        self.console.print(f"Steps taken: {result.steps_taken}")
        self.console.print(f"Success: {result.success}")
        self.console.print(f"Message: {result.message}")
