from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Footer, Header, Input, Label, RichLog

if TYPE_CHECKING:
    from chef_human.agent.parser import ParsedToolCall
    from chef_human.agent.planner import Plan
    from chef_human.agent.react_loop import AgentResult

_DIFF_BLOCK_RE = re.compile(r"```diff\n(.*?)\n```", re.DOTALL)

_PREVIEW_BYTES_LIMIT = 20_000


def extract_diff_block(text: str) -> str | None:
    """Pull the first fenced ```diff block out of a tool result, if present."""
    match = _DIFF_BLOCK_RE.search(text)
    return match.group(1) if match else None


class ApprovalModal(ModalScreen[bool]):
    """Blocking-style Yes/No prompt for destructive command approval."""

    DEFAULT_CSS = """
    ApprovalModal {
        align: center middle;
    }
    #approval-dialog {
        width: 70%;
        height: auto;
        border: thick $error;
        padding: 1 2;
        background: $surface;
    }
    #approval-buttons {
        height: auto;
        padding-top: 1;
    }
    """

    def __init__(self, command: str) -> None:
        super().__init__()
        self._command = command

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-dialog"):
            yield Label("[bold yellow]Destructive operation requested:[/]")
            yield Label(f"[red]{self._command}[/]")
            with Horizontal(id="approval-buttons"):
                yield Button("Approve", id="approve", variant="error")
                yield Button("Reject", id="reject", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "approve")


class TuiUI:
    """`ReActUI` implementation that renders into a `ChefHumanTUI`'s widgets."""

    def __init__(self, app: "ChefHumanTUI") -> None:
        self._app = app

    def _chat(self) -> RichLog:
        return self._app.query_one("#chat-log", RichLog)

    def _preview(self) -> RichLog:
        return self._app.query_one("#preview-log", RichLog)

    def on_start(self, task: str) -> None:
        self._chat().write(f"\n[bold cyan]You:[/] {task}")

    def on_planning_start(self) -> None:
        self._chat().write("[dim]Planning...[/]")

    def on_plan(self, plan: "Plan") -> None:
        self._chat().write(f"[bold yellow]Plan:[/] {plan.goal}")
        for step in plan.steps:
            self._chat().write(f"  {step.index}. {step.description}")

    def on_reasoning_start(self) -> None:
        pass

    def on_stream(self, chunk: str) -> None:
        # Token-by-token writes would flood the log; full reasoning is shown
        # once complete via on_reasoning instead.
        pass

    def on_reasoning(self, content: str) -> None:
        if content:
            self._chat().write(f"[dim]{content}[/]")

    def on_tool_call(self, tool_call: "ParsedToolCall") -> None:
        args = ", ".join(f"{k}={v}" for k, v in tool_call.arguments.items())[:120]
        self._chat().write(f"  [bold green]▶[/] [cyan]{tool_call.name}[/]({args})")

    def on_tool_result(self, name: str, result: str) -> None:
        icon = "✗" if result.startswith("Error") else "✓"
        summary = result[:200].replace("\n", " ").strip()
        self._chat().write(f"  {icon} [bold]{name}[/] → {summary}")

        diff = extract_diff_block(result)
        if diff is not None:
            self._show_diff(name, diff)

    def _show_diff(self, tool_name: str, diff_text: str) -> None:
        preview = self._preview()
        preview.clear()
        preview.write(f"[bold]diff — {tool_name}[/]\n")
        for line in diff_text.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                preview.write(f"[green]{line}[/]")
            elif line.startswith("-") and not line.startswith("---"):
                preview.write(f"[red]{line}[/]")
            else:
                preview.write(line)

    def on_replan(self) -> None:
        self._chat().write("[bold yellow]↻ Re-planning...[/]")

    def on_error(self, message: str) -> None:
        self._chat().write(f"[bold red]Error:[/] {message}")

    async def on_approval_request(self, tool_call: "ParsedToolCall") -> bool:
        command = tool_call.arguments.get("command", "")
        approved = await self._app.push_screen_wait(ApprovalModal(command))
        return bool(approved)

    def display_result(self, result: "AgentResult") -> None:
        status = "[bold green]✓ Success[/]" if result.success else "[bold red]✗ Failed[/]"
        self._chat().write(f"\n[bold]Result:[/] {status}  (steps: {result.steps_taken})")
        if result.message:
            self._chat().write(f"  {result.message[:300]}")


class ChefHumanTUI(App):
    """Split-pane TUI: file tree (left), chat/log + diff preview (right)."""

    CSS = """
    #body {
        height: 1fr;
    }
    #tree-pane {
        width: 30%;
        border: solid $accent;
    }
    #right-pane {
        width: 70%;
    }
    #chat-log {
        height: 65%;
        border: solid $accent;
    }
    #preview-log {
        height: 35%;
        border: solid $accent;
    }
    """

    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(
        self,
        workspace_root: Path,
        on_submit: Callable[[str], Awaitable[None]],
    ) -> None:
        super().__init__()
        self._workspace_root = workspace_root
        self._on_submit = on_submit
        self.tui_ui = TuiUI(self)

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            yield DirectoryTree(str(self._workspace_root), id="tree-pane")
            with Vertical(id="right-pane"):
                yield RichLog(id="chat-log", wrap=True, markup=True, highlight=False)
                yield RichLog(id="preview-log", wrap=True, markup=True, highlight=False)
        yield Input(placeholder="Type a task and press Enter...", id="task-input")
        yield Footer()

    def on_mount(self) -> None:
        self._app_log("[bold cyan]chef-human[/] — type a task below. Ctrl+C to quit.")
        self.query_one("#task-input", Input).focus()

    def _app_log(self, text: str) -> None:
        self.query_one("#chat-log", RichLog).write(text)

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        preview = self.query_one("#preview-log", RichLog)
        preview.clear()
        try:
            content = Path(event.path).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            preview.write(f"[red]Cannot read {event.path}: {exc}[/]")
            return
        preview.write(f"[bold]{event.path}[/]\n")
        preview.write(content[:_PREVIEW_BYTES_LIMIT])

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        self.run_worker(self._on_submit(text), exclusive=False)
