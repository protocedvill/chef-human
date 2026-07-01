from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table
from rich.tree import Tree

if TYPE_CHECKING:
    from chef_human.agent.parser import ParsedToolCall
    from chef_human.agent.planner import Plan
    from chef_human.agent.react_loop import AgentResult


class DebugTUI:
    """Rich-based debug terminal UI for the ReAct loop."""

    def __init__(self) -> None:
        self.console = Console()
        self.layout = Layout()
        self._setup_layout()
        self._live = Live(self.layout, refresh_per_second=4, screen=True)
        self._started = False
        self._plan_tree: Tree | None = None
        self._reasoning_text = ""
        self._tool_calls: list[tuple[str, str, str]] = []
        self._log_entries: list[str] = []
        self._step_count = 0
        self._max_steps = 0

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

    def _update_header(self, task: str) -> None:
        header = Panel(
            f"[bold cyan]chef-human[/] | Step {self._step_count}/{self._max_steps} | Task: {task[:60]}",
            style="white on dark_blue",
        )
        self.layout["header"].update(header)

    def on_start(self, task: str) -> None:
        self._ensure_live()
        self._log(f"Task started: {task}")
        self._update_header(task)

    def on_planning_start(self) -> None:
        self._ensure_live()
        self._log("Planning...")

    def on_plan(self, plan: Plan) -> None:
        self._ensure_live()
        self._max_steps = len(plan.steps)
        self._plan_tree = Tree("[bold]Plan[/]")
        for step in plan.steps:
            self._plan_tree.add(f"[ ] Step {step.index}: {step.description}")
        self.layout["plan_panel"].update(Panel(self._plan_tree, title="Plan"))
        self._log(f"Plan generated: {len(plan.steps)} steps")

    def on_reasoning_start(self) -> None:
        self._ensure_live()
        self._reasoning_text = ""
        self.layout["reasoning_panel"].update(
            Panel("Thinking...", title="Reasoning")
        )

    def on_reasoning(self, content: str) -> None:
        self._ensure_live()
        self._reasoning_text = content
        display = content[:500] + ("..." if len(content) > 500 else "")
        self.layout["reasoning_panel"].update(
            Panel(display, title="Reasoning")
        )
        self._log("Model reasoning received")

    def on_tool_call(self, tool_call: ParsedToolCall) -> None:
        self._ensure_live()
        args_str = ", ".join(f"{k}={v}" for k, v in tool_call.arguments.items())
        self._tool_calls.append(("▶", tool_call.name, args_str))
        self._refresh_tool_panel()
        self._log(f"Tool call: {tool_call.name}({args_str})")

    def on_tool_result(self, name: str, result: str) -> None:
        self._ensure_live()
        status = "✓" if not result.startswith("Error") else "✗"
        result_preview = result[:80] + ("..." if len(result) > 80 else "")
        self._tool_calls.append((status, name, result_preview))
        self._refresh_tool_panel()
        self._log(f"Tool result: {name} -> {result_preview}")

    def on_replan(self) -> None:
        self._ensure_live()
        self._log("[yellow]Re-planning...[/]")

    def on_error(self, message: str) -> None:
        self._ensure_live()
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
        log_text = "\n".join(self._log_entries[-10:])
        self.layout["log_panel"].update(Panel(log_text, title="Log"))

    def display_final(self, result: AgentResult) -> None:
        self._stop_live()
        self.console.print("\n[bold]=== Task Complete ===[/]")
        self.console.print(f"Steps taken: {result.steps_taken}")
        self.console.print(f"Success: {result.success}")
        self.console.print(f"Message: {result.message}")
