from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Footer, Header, Input, Label, RichLog, Static

from chef_human.agent.planner import StepStatus
from chef_human.agent.react_loop import FILE_MUTATING_TOOLS

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
    ollama_activity: str | None = None
    auto_mode: bool = False

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
        event.stop()
        self.dismiss(event.button.id == "approve")




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

    def _tree(self) -> DirectoryTree:
        return self._app.query_one("#tree-pane", DirectoryTree)

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

    def on_plan_progress(self, plan: "Plan") -> None:
        # Unlike on_plan() (announced once at task start, and again on
        # replan), this fires every turn so the stats panel's "Step:"
        # line actually tracks which step is current as steps complete --
        # without it, current_step/steps_done are only ever refreshed at
        # on_plan() and at the very end in display_result(), so the panel
        # looks frozen for the whole run.
        self._update_plan_stats(plan)
        self.render_stats()

    def _update_plan_stats(self, plan: "Plan") -> None:
        current = plan.current_leaf()
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

        # The #tree-pane DirectoryTree is only populated once, at compose
        # time -- it never re-scans disk on its own, so it goes stale the
        # moment the agent creates/deletes/renames a file. Reload it after
        # any tool that can change the file layout actually succeeds
        # (same tool set react_loop.py uses to invalidate the repo-map
        # cache, for the same reason).
        if name in FILE_MUTATING_TOOLS and not result.startswith("Error"):
            self._tree().reload()

        diff = extract_diff_block(result)
        if diff is not None:
            self._show_diff(name, diff)

        self.render_stats()

    def on_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.stats.total_prompt_tokens += prompt_tokens
        self.stats.total_completion_tokens += completion_tokens
        self.render_stats()

    def on_llm_start(self, activity: str) -> None:
        self.stats.ollama_activity = activity
        self.render_stats()

    def on_llm_end(self) -> None:
        self.stats.ollama_activity = None
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

    def on_escalation(self, node, message: str) -> None:
        self._chat().write(
            f"[bold red]Escalation:[/] {escape(node.description[:80])!r} was "
            f"marked failed and skipped -- {escape(message)}"
        )
        self.stats.add_warning(f"Escalated: {node.description[:80]}")
        self.render_stats()

    async def on_approval_request(self, tool_call: "ParsedToolCall") -> bool:
        command = tool_call.arguments.get("command", "")
        approved = await self._app.push_screen_wait(ApprovalModal(command))
        return bool(approved)

    async def on_ask_user(self, question: str) -> str:
        self._chat().write(f"[bold yellow]Agent asks:[/] {escape(question)}")
        answer = await self._app.ask_inline(question)
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
        panel.write(
            f"[bold]Auto-mode:[/] {'[green]ON[/]' if s.auto_mode else '[dim]OFF[/]'} (Ctrl+G)"
        )
        if s.ollama_activity:
            panel.write(f"[bold]Ollama:[/] [green]● processing[/] ({escape(s.ollama_activity)})")
        else:
            panel.write("[bold]Ollama:[/] [dim]○ idle[/]")
        if s.current_step:
            panel.write(
                f"[bold]Step:[/] {escape(s.current_step)} ({s.steps_done}/{s.steps_total})"
            )
        elif s.steps_total:
            panel.write(f"[bold]Steps:[/] {s.steps_done}/{s.steps_total} complete")
        panel.write("")
        panel.write(f"[bold yellow]Warnings: {len(s.warnings)}[/]")


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
    #question-banner {
        background: $warning 30%;
        color: $text;
        padding: 0 1;
        height: auto;
    }
    #question-banner.hidden {
        display: none;
    }
    #input-bar {
        height: auto;
    }
    #task-input {
        width: 1fr;
    }
    #skip-question-btn {
        width: auto;
    }
    #skip-question-btn.hidden {
        display: none;
    }
    """

    # Mouse-drag selection inside a scrolled RichLog requires reconstructing
    # Textual's internal viewport/selection coordinate mapping, which proved
    # fragile in practice (see docs/archive/plans/plan_5.2.md 5.2.8/5.2.10).
    # Instead, ctrl+c
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
        Binding("ctrl+g", "toggle_auto_mode", "Toggle auto-mode"),
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
        # Set while an ask_user answer is pending -- routes the next
        # #task-input submission to _pending_answer instead of treating it
        # as a new task. See ask_inline().
        self._pending_answer: asyncio.Future[str] | None = None
        # When on, ask_inline() resolves immediately instead of taking over
        # the task pane -- read by main.py when building each task's
        # ReActConfig.disable_ask_user.
        self.auto_mode: bool = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            with Vertical(id="left-pane"):
                yield DirectoryTree(str(self._workspace_root), id="tree-pane")
                yield RichLog(id="stats-panel", wrap=True, markup=True, highlight=False)
            with Vertical(id="right-pane"):
                yield RichLog(id="chat-log", wrap=True, markup=True, highlight=False)
                yield RichLog(id="preview-log", wrap=True, markup=True, highlight=False)
        yield Static("", id="question-banner", classes="hidden")
        with Horizontal(id="input-bar"):
            yield Input(placeholder="Type a task and press Enter...", id="task-input")
            yield Button("Skip", id="skip-question-btn", variant="warning", classes="hidden")
        yield Footer()

    def on_mount(self) -> None:
        self._app_log(
            "[bold cyan]chef-human[/] — type a task below. Ctrl+Q to quit. "
            "Ctrl+G to toggle auto-mode. Click a log pane to focus it, "
            "Ctrl+C to copy its full contents."
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

    def action_toggle_auto_mode(self) -> None:
        self.auto_mode = not self.auto_mode
        self.tui_ui.stats.auto_mode = self.auto_mode
        self._app_log(
            f"[bold]Auto-mode {'enabled' if self.auto_mode else 'disabled'}[/] -- "
            + (
                "the agent will no longer ask questions; it makes its own "
                "best-guess decisions instead."
                if self.auto_mode
                else "the agent can ask questions again."
            )
        )
        self.tui_ui.render_stats()

    async def ask_inline(self, question: str) -> str:
        """Collects an ask_user answer by taking over the task input bar
        (showing the question above it, swapping the placeholder, and
        revealing a Skip button) instead of a modal popup -- a popup is
        obtrusive for something that happens on every task, and a Label in
        a fixed-width dialog was prone to clipping long questions. Static
        wraps within #question-banner's full-width, scrollable-height box
        instead.

        Not called at all when auto-mode is on: ReActConfig.disable_ask_user
        (set from self.auto_mode when each task's config is built -- see
        main.py) makes react_loop.py answer ask_user itself without ever
        invoking the UI, so this method doesn't need to know about the
        toggle."""
        banner = self.query_one("#question-banner", Static)
        banner.update(f"[bold yellow]Agent asks:[/] {escape(question)}")
        banner.remove_class("hidden")
        skip_btn = self.query_one("#skip-question-btn", Button)
        skip_btn.remove_class("hidden")
        task_input = self.query_one("#task-input", Input)
        old_placeholder = task_input.placeholder
        task_input.placeholder = "Type your answer, or click Skip..."

        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending_answer = future
        try:
            return await future
        finally:
            self._pending_answer = None
            banner.add_class("hidden")
            skip_btn.add_class("hidden")
            task_input.placeholder = old_placeholder

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "skip-question-btn":
            return
        event.stop()
        if self._pending_answer is not None and not self._pending_answer.done():
            self._pending_answer.set_result("User skipped the question")

    async def _run_initial_task(self) -> None:
        if self._initial_task is None:
            return
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
        if self._pending_answer is not None and not self._pending_answer.done():
            self._pending_answer.set_result(text or "User skipped the question")
            return
        if not text:
            return
        self.run_worker(self._on_submit(text), exclusive=False)
