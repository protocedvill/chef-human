from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Footer, Header, Input, Label, RichLog

from chef_human.agent.planner import StepStatus

if TYPE_CHECKING:
    from chef_human.agent.parser import ParsedToolCall
    from chef_human.agent.planner import Plan
    from chef_human.agent.react_loop import AgentResult

_DIFF_BLOCK_RE = re.compile(r"```diff\n(.*?)\n```", re.DOTALL)

_PREVIEW_BYTES_LIMIT = 20_000

_MAX_WARNINGS_KEPT = 5

_LOG_PANE_IDS = ("chat-log", "preview-log", "stats-panel")


@dataclass
class SessionStats:
    """Running totals and current-state snapshot for the whole TUI session
    (persists across every task submitted, not just the current one)."""

    tasks_run: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    replans: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    status: str = "Idle"
    current_step: str | None = None
    steps_done: int = 0
    steps_total: int = 0
    warnings: list[str] = field(default_factory=list)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)
        del self.warnings[: -_MAX_WARNINGS_KEPT]


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
            yield Label(f"[red]{escape(self._command)}[/]")
            with Horizontal(id="approval-buttons"):
                yield Button("Approve", id="approve", variant="error")
                yield Button("Reject", id="reject", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "approve")


class AskUserModal(ModalScreen[str]):
    """Blocking-style free-text prompt for the agent's `ask_user` tool.

    Textual owns the terminal in raw/alternate-screen mode while the TUI is
    running, so a tool that tries to read an answer via plain
    sys.stdin.readline() (as AskUserTool.run() does for terminal-based UIs)
    both hangs the whole asyncio event loop and can never actually receive
    the answer -- there's no visible prompt and nothing forwarded to that
    file descriptor. This modal collects the answer the same way
    ApprovalModal collects a yes/no, through Textual's own event loop via
    push_screen_wait."""

    DEFAULT_CSS = """
    AskUserModal {
        align: center middle;
    }
    #ask-user-dialog {
        width: 70%;
        height: auto;
        border: thick $warning;
        padding: 1 2;
        background: $surface;
    }
    #ask-user-input {
        margin-top: 1;
    }
    #ask-user-buttons {
        height: auto;
        padding-top: 1;
    }
    """

    def __init__(self, question: str) -> None:
        super().__init__()
        self._question = question

    def compose(self) -> ComposeResult:
        with Vertical(id="ask-user-dialog"):
            yield Label("[bold yellow]Agent asks:[/]")
            yield Label(escape(self._question))
            yield Input(placeholder="Type your answer...", id="ask-user-input")
            with Horizontal(id="ask-user-buttons"):
                yield Button("Submit", id="submit", variant="success")
                yield Button("Skip", id="skip", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#ask-user-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit":
            self._submit(self.query_one("#ask-user-input", Input).value)
        else:
            self._submit("")

    def _submit(self, value: str) -> None:
        text = value.strip()
        self.dismiss(text or "User skipped the question")


class TuiUI:
    """`ReActUI` implementation that renders into a `ChefHumanTUI`'s widgets."""

    def __init__(self, app: "ChefHumanTUI") -> None:
        self._app = app
        self.stats = SessionStats()

    def _chat(self) -> RichLog:
        return self._app.query_one("#chat-log", RichLog)

    def _preview(self) -> RichLog:
        return self._app.query_one("#preview-log", RichLog)

    def _stats_panel(self) -> RichLog:
        return self._app.query_one("#stats-panel", RichLog)

    def on_start(self, task: str) -> None:
        self._chat().write(f"\n[bold cyan]You:[/] {escape(task)}")
        self.stats.tasks_run += 1
        self.stats.status = "Running"
        self.render_stats()

    def on_planning_start(self) -> None:
        self._chat().write("[dim]Planning...[/]")
        self.stats.status = "Planning..."
        self.render_stats()

    def on_plan(self, plan: "Plan") -> None:
        self._chat().write(f"[bold yellow]Plan:[/] {escape(plan.goal)}")
        for step in plan.steps:
            self._chat().write(f"  {step.index}. {escape(step.description)}")
        self.stats.status = "Running"
        self._update_plan_stats(plan)
        self.render_stats()

    def _update_plan_stats(self, plan: "Plan") -> None:
        current = plan.current_step()
        self.stats.current_step = current.description if current else None
        self.stats.steps_total = len(plan.steps)
        self.stats.steps_done = sum(
            1 for s in plan.steps if s.status == StepStatus.completed
        )

    def on_reasoning_start(self) -> None:
        pass

    def on_stream(self, chunk: str) -> None:
        # Token-by-token writes would flood the log; full reasoning is shown
        # once complete via on_reasoning instead.
        pass

    def on_reasoning(self, content: str) -> None:
        if content:
            self._chat().write(f"[dim]{escape(content)}[/]")

    def on_tool_call(self, tool_call: "ParsedToolCall") -> None:
        args = ", ".join(f"{k}={v}" for k, v in tool_call.arguments.items())[:120]
        self._chat().write(
            f"  [bold green]▶[/] [cyan]{escape(tool_call.name)}[/]({escape(args)})"
        )
        self.stats.tool_calls += 1
        self.render_stats()

    def on_tool_result(self, name: str, result: str) -> None:
        icon = "✗" if result.startswith("Error") else "✓"
        summary = result[:200].replace("\n", " ").strip()
        self._chat().write(f"  {icon} [bold]{escape(name)}[/] → {escape(summary)}")

        # "plan-check" / "repeat-guard" are the synthetic corrective
        # messages react_loop injects (step-not-done feedback, repeated-call
        # nudges) -- surface those as warnings even though they aren't
        # literal tool errors.
        if name in ("plan-check", "repeat-guard") or result.startswith("Error"):
            self.stats.tool_errors += 1
            self.stats.add_warning(f"{name}: {summary[:150]}")

        diff = extract_diff_block(result)
        if diff is not None:
            self._show_diff(name, diff)

        self.render_stats()

    def on_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.stats.total_prompt_tokens += prompt_tokens
        self.stats.total_completion_tokens += completion_tokens
        self.render_stats()

    def _show_diff(self, tool_name: str, diff_text: str) -> None:
        preview = self._preview()
        preview.clear()
        preview.write(f"[bold]diff — {escape(tool_name)}[/]\n")
        for line in diff_text.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                preview.write(f"[green]{escape(line)}[/]")
            elif line.startswith("-") and not line.startswith("---"):
                preview.write(f"[red]{escape(line)}[/]")
            else:
                preview.write(escape(line))

    def on_replan(self) -> None:
        self._chat().write("[bold yellow]↻ Re-planning...[/]")
        self.stats.replans += 1
        self.stats.status = "Re-planning..."
        self.stats.add_warning("Re-planning triggered after repeated failures")
        self.render_stats()

    def on_error(self, message: str) -> None:
        self._chat().write(f"[bold red]Error:[/] {escape(message)}")
        self.stats.add_warning(message[:150])
        self.render_stats()

    async def on_approval_request(self, tool_call: "ParsedToolCall") -> bool:
        command = tool_call.arguments.get("command", "")
        approved = await self._app.push_screen_wait(ApprovalModal(command))
        return bool(approved)

    async def on_ask_user(self, question: str) -> str:
        self._chat().write(f"[bold yellow]Agent asks:[/] {escape(question)}")
        answer = await self._app.push_screen_wait(AskUserModal(question))
        self._chat().write(f"[bold cyan]You:[/] {escape(answer)}")
        return answer

    def display_result(self, result: "AgentResult") -> None:
        status = "[bold green]✓ Success[/]" if result.success else "[bold red]✗ Failed[/]"
        self._chat().write(f"\n[bold]Result:[/] {status}  (steps: {result.steps_taken})")
        if result.message:
            self._chat().write(f"  {escape(result.message[:300])}")

        # Token totals are NOT added here -- they're accumulated live,
        # per-LLM-call, via on_token_usage() as the task runs (see
        # react_loop.py). Adding result.total_prompt_tokens/
        # total_completion_tokens here too would double-count every token.
        self.stats.status = "Idle" if result.success else "Failed"
        self._update_plan_stats(result.plan)
        self.render_stats()

    def render_stats(self) -> None:
        s = self.stats
        panel = self._stats_panel()
        panel.clear()
        panel.write("[bold]Session[/]")
        panel.write(f"  Tasks run: {s.tasks_run}")
        panel.write(f"  Tool calls: {s.tool_calls}  Errors: {s.tool_errors}  Replans: {s.replans}")
        panel.write(f"  Tokens: {s.total_prompt_tokens:,}↑ / {s.total_completion_tokens:,}↓")
        panel.write("")
        panel.write(f"[bold]Status:[/] {escape(s.status)}")
        if s.current_step:
            panel.write(
                f"[bold]Step:[/] {escape(s.current_step)} ({s.steps_done}/{s.steps_total})"
            )
        elif s.steps_total:
            panel.write(f"[bold]Steps:[/] {s.steps_done}/{s.steps_total} complete")
        panel.write("")
        if s.warnings:
            panel.write(f"[bold yellow]Warnings ({len(s.warnings)}):[/]")
            for w in s.warnings:
                panel.write(f"  • {escape(w)}")
        else:
            panel.write("[dim]No warnings[/]")


class ChefHumanTUI(App):
    """Split-pane TUI: file tree + session stats (left), chat/log + diff
    preview (right)."""

    CSS = """
    #body {
        height: 1fr;
    }
    #left-pane {
        width: 30%;
    }
    #tree-pane {
        height: 60%;
        border: solid $accent;
    }
    #stats-panel {
        height: 40%;
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

    # Mouse-drag selection inside a scrolled RichLog requires reconstructing
    # Textual's internal viewport/selection coordinate mapping, which proved
    # fragile in practice (see plan_5.2.md 5.2.8/5.2.10). Instead, ctrl+c
    # copies the *entire* content of whichever log pane last had focus
    # (click a pane to focus it) to the system clipboard -- no drag needed,
    # and it can't desync from scroll position since it copies everything.
    # priority=True so this overrides Screen's default ctrl+c -> copy_text
    # binding (which only copies drag-selected text and would otherwise
    # shadow this, since screen-level bindings resolve before app-level
    # ones).
    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        Binding("ctrl+c", "copy_focused_pane", "Copy pane", priority=True),
    ]

    def __init__(
        self,
        workspace_root: Path,
        on_submit: Callable[[str], Awaitable[None]],
        initial_task: str | None = None,
        auto_exit_after_initial_task: bool = False,
    ) -> None:
        super().__init__()
        self._workspace_root = workspace_root
        self._on_submit = on_submit
        self._initial_task = initial_task
        self._auto_exit_after_initial_task = auto_exit_after_initial_task
        self.tui_ui = TuiUI(self)

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            with Vertical(id="left-pane"):
                yield DirectoryTree(str(self._workspace_root), id="tree-pane")
                yield RichLog(id="stats-panel", wrap=True, markup=True, highlight=False)
            with Vertical(id="right-pane"):
                yield RichLog(id="chat-log", wrap=True, markup=True, highlight=False)
                yield RichLog(id="preview-log", wrap=True, markup=True, highlight=False)
        yield Input(placeholder="Type a task and press Enter...", id="task-input")
        yield Footer()

    def on_mount(self) -> None:
        self._app_log(
            "[bold cyan]chef-human[/] — type a task below. Ctrl+Q to quit. "
            "Click a log pane to focus it, Ctrl+C to copy its full contents."
        )
        self.query_one("#task-input", Input).focus()
        self.tui_ui.render_stats()
        if self._initial_task:
            self.run_worker(self._run_initial_task())

    def action_copy_focused_pane(self) -> None:
        if self.focused is None or self.focused.id not in _LOG_PANE_IDS:
            return
        log = self.focused
        assert isinstance(log, RichLog)
        text = "\n".join(strip.text for strip in log.lines)
        if text:
            self.copy_to_clipboard(text)

    async def _run_initial_task(self) -> None:
        await self._on_submit(self._initial_task)
        if self._auto_exit_after_initial_task:
            self.exit()

    def _app_log(self, text: str) -> None:
        self.query_one("#chat-log", RichLog).write(text)

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        preview = self.query_one("#preview-log", RichLog)
        preview.clear()
        try:
            content = Path(event.path).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            preview.write(f"[red]Cannot read {escape(str(event.path))}: {escape(str(exc))}[/]")
            return
        preview.write(f"[bold]{escape(str(event.path))}[/]\n")
        preview.write(escape(content[:_PREVIEW_BYTES_LIMIT]))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        self.run_worker(self._on_submit(text), exclusive=False)
