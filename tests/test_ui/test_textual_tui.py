from __future__ import annotations


from chef_human.agent.parser import ParsedToolCall
from chef_human.agent.planner import Plan, PlanStep, StepStatus
from chef_human.agent.react_loop import AgentResult
from chef_human.ui.textual_tui import (
    ApprovalModal,
    ChefHumanTUI,
    SessionStats,
    TuiUI,
    extract_diff_block,
)


class TestExtractDiffBlock:
    def test_returns_diff_when_present(self):
        text = 'Wrote file.\n```diff\n-old line\n+new line\n```\nDone.'
        diff = extract_diff_block(text)
        assert diff == "-old line\n+new line"

    def test_returns_none_when_absent(self):
        assert extract_diff_block("no diff here") is None

    def test_returns_first_block_when_multiple(self):
        text = "```diff\nfirst\n```\nmore\n```diff\nsecond\n```"
        assert extract_diff_block(text) == "first"


async def _make_app(tmp_path):
    calls: list[str] = []

    async def on_submit(text: str) -> None:
        calls.append(text)

    app = ChefHumanTUI(workspace_root=tmp_path, on_submit=on_submit)
    return app, calls


class TestChefHumanTUIWidgets:
    async def test_composes_expected_panes(self, tmp_path):
        app, _ = await _make_app(tmp_path)
        async with app.run_test():
            assert app.query_one("#tree-pane") is not None
            assert app.query_one("#stats-panel") is not None
            assert app.query_one("#chat-log") is not None
            assert app.query_one("#preview-log") is not None
            assert app.query_one("#task-input") is not None

    async def test_stats_panel_rendered_on_mount(self, tmp_path):
        from textual.widgets import RichLog

        app, _ = await _make_app(tmp_path)
        async with app.run_test():
            panel = app.query_one("#stats-panel", RichLog)
            assert len(panel.lines) >= 1

    async def test_mount_writes_welcome_message(self, tmp_path):
        app, _ = await _make_app(tmp_path)
        async with app.run_test():
            from textual.widgets import RichLog
            log = app.query_one("#chat-log", RichLog)
            assert len(log.lines) >= 1

    async def test_input_submit_invokes_callback(self, tmp_path):
        app, calls = await _make_app(tmp_path)
        async with app.run_test() as pilot:
            await pilot.click("#task-input")
            for ch in "add a function":
                await pilot.press(ch)
            await pilot.press("enter")
            await pilot.pause()
            assert calls == ["add a function"]

    async def test_empty_input_does_not_invoke_callback(self, tmp_path):
        app, calls = await _make_app(tmp_path)
        async with app.run_test() as pilot:
            await pilot.click("#task-input")
            await pilot.press("enter")
            await pilot.pause()
            assert calls == []

    async def test_file_selected_shows_preview(self, tmp_path):
        target = tmp_path / "hello.py"
        target.write_text("print('hi')\n")
        app, _ = await _make_app(tmp_path)
        async with app.run_test():
            from textual.widgets import DirectoryTree, RichLog
            tree = app.query_one("#tree-pane", DirectoryTree)
            event = DirectoryTree.FileSelected(tree.cursor_node, target)
            app.on_directory_tree_file_selected(event)
            preview = app.query_one("#preview-log", RichLog)
            assert len(preview.lines) >= 1

    async def test_file_selected_escapes_rich_markup_in_content(self, tmp_path):
        # Regression test: file content containing Rich markup-like brackets
        # (type hints, slices, scratchpad-style tags) must not crash the
        # markup=True RichLog or silently vanish from the preview.
        target = tmp_path / "typed.py"
        target.write_text(
            "def foo(x: Dict[str, int]) -> Optional[List[bool]]:\n"
            "    return arr[i:j]\n"
            "## Scratchpad: [decision|file|assumption|question] note\n"
        )
        app, _ = await _make_app(tmp_path)
        async with app.run_test():
            from textual.widgets import DirectoryTree, RichLog
            tree = app.query_one("#tree-pane", DirectoryTree)
            event = DirectoryTree.FileSelected(tree.cursor_node, target)
            app.on_directory_tree_file_selected(event)
            preview = app.query_one("#preview-log", RichLog)
            text = "\n".join(str(line) for line in preview.lines)
            assert "Dict[str, int]" in text
            assert "List[bool]" in text
            assert "arr[i:j]" in text


class TestCopyPane:
    """Copy-to-clipboard is implemented as an explicit ctrl+c action that
    copies the full contents of whichever log pane last had focus, rather
    than relying on Textual's built-in mouse-drag selection -- reconstructing
    RichLog's internal viewport/scroll coordinate mapping for drag-select
    proved fragile in practice (see plan_5.2.md 5.2.8/5.2.10, both of which
    fixed real bugs in that approach but still didn't make it reliable)."""

    async def test_ctrl_q_quits(self, tmp_path):
        app, _ = await _make_app(tmp_path)
        async with app.run_test():
            active = app.active_bindings
            assert active["ctrl+q"].binding.action == "quit"

    async def test_ctrl_c_resolves_to_copy_focused_pane(self, tmp_path):
        app, _ = await _make_app(tmp_path)
        async with app.run_test():
            active = app.active_bindings
            assert active["ctrl+c"].binding.action == "copy_focused_pane"

    async def test_log_panes_are_plain_rich_log(self, tmp_path):
        from textual.widgets import RichLog

        app, _ = await _make_app(tmp_path)
        async with app.run_test():
            assert type(app.query_one("#chat-log")) is RichLog
            assert type(app.query_one("#preview-log")) is RichLog
            assert type(app.query_one("#stats-panel")) is RichLog

    async def test_ctrl_c_copies_focused_pane_contents(self, tmp_path):
        from textual.widgets import RichLog

        app, _ = await _make_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            panel = app.query_one("#stats-panel", RichLog)
            panel.clear()
            panel.write("hello from stats")
            await pilot.pause()

            panel.focus()
            await pilot.pause()
            await pilot.press("ctrl+c")
            await pilot.pause()

            assert "hello from stats" in app.clipboard

    async def test_ctrl_c_copies_full_scrollback_even_when_scrolled(self, tmp_path):
        """Since the whole pane is copied (not just the visible viewport),
        scroll position must not affect what gets copied -- content from the
        very top of a long log must still be included."""
        from textual.widgets import RichLog

        app, _ = await _make_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            panel = app.query_one("#stats-panel", RichLog)
            panel.clear()
            panel.write("FIRST-LINE-MARKER")
            for i in range(100):
                panel.write(f"line-{i:03d}")
            await pilot.pause()
            assert panel.scroll_offset.y > 0  # sanity: content did scroll

            panel.focus()
            await pilot.pause()
            await pilot.press("ctrl+c")
            await pilot.pause()

            assert "FIRST-LINE-MARKER" in app.clipboard
            assert "line-099" in app.clipboard

    async def test_ctrl_c_does_nothing_when_input_focused(self, tmp_path):
        app, _ = await _make_app(tmp_path)
        async with app.run_test() as pilot:
            app.query_one("#task-input").focus()
            await pilot.pause()
            app._clipboard = ""
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert app.clipboard == ""

    async def test_ctrl_c_does_nothing_when_pane_empty(self, tmp_path):
        from textual.widgets import RichLog

        app, _ = await _make_app(tmp_path)
        async with app.run_test() as pilot:
            panel = app.query_one("#preview-log", RichLog)
            panel.clear()
            panel.focus()
            await pilot.pause()
            app._clipboard = ""
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert app.clipboard == ""


class TestInitialTaskAutoSubmit:
    """`chef-human run` (the new default, non-headless path) drives the TUI
    with a task provided on the command line rather than typed interactively
    -- it must auto-submit on mount and, when configured, exit once done."""

    async def test_initial_task_auto_submitted_on_mount(self, tmp_path):
        calls: list[str] = []

        async def on_submit(text: str) -> None:
            calls.append(text)

        app = ChefHumanTUI(
            workspace_root=tmp_path,
            on_submit=on_submit,
            initial_task="fix the bug",
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            assert calls == ["fix the bug"]

    async def test_no_initial_task_does_not_auto_submit(self, tmp_path):
        calls: list[str] = []

        async def on_submit(text: str) -> None:
            calls.append(text)

        app = ChefHumanTUI(workspace_root=tmp_path, on_submit=on_submit)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert calls == []

    async def test_auto_exit_after_initial_task_completes(self, tmp_path):
        async def on_submit(text: str) -> None:
            return None

        app = ChefHumanTUI(
            workspace_root=tmp_path,
            on_submit=on_submit,
            initial_task="fix the bug",
            auto_exit_after_initial_task=True,
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._exit is True

    async def test_stays_open_when_auto_exit_disabled(self, tmp_path):
        async def on_submit(text: str) -> None:
            return None

        app = ChefHumanTUI(
            workspace_root=tmp_path,
            on_submit=on_submit,
            initial_task="fix the bug",
            auto_exit_after_initial_task=False,
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._exit is False

    async def test_user_can_still_type_after_initial_task(self, tmp_path):
        calls: list[str] = []

        async def on_submit(text: str) -> None:
            calls.append(text)

        app = ChefHumanTUI(
            workspace_root=tmp_path,
            on_submit=on_submit,
            initial_task="first task",
            auto_exit_after_initial_task=False,
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.click("#task-input")
            for ch in "second task":
                await pilot.press(ch)
            await pilot.press("enter")
            await pilot.pause()
            assert calls == ["first task", "second task"]


class TestTuiUIProtocol:
    PROTOCOL_METHODS = [
        "on_start",
        "on_planning_start",
        "on_plan",
        "on_reasoning_start",
        "on_stream",
        "on_reasoning",
        "on_tool_call",
        "on_tool_result",
        "on_token_usage",
        "on_replan",
        "on_error",
        "on_ask_user",
        "on_approval_request",
    ]

    async def _ui(self, tmp_path):
        app, _ = await _make_app(tmp_path)
        return app, app.tui_ui

    def test_has_all_protocol_methods(self):
        # TuiUI methods are checked directly on the class to avoid needing a
        # mounted app for a pure attribute-presence check.
        for method in self.PROTOCOL_METHODS:
            assert hasattr(TuiUI, method), f"TuiUI missing {method}"
            assert callable(getattr(TuiUI, method))

    async def test_on_start_writes_task(self, tmp_path):
        app, ui = await self._ui(tmp_path)
        async with app.run_test():
            from textual.widgets import RichLog
            log = app.query_one("#chat-log", RichLog)
            before = len(log.lines)
            ui.on_start("fix the bug")
            assert len(log.lines) > before

    async def test_on_plan_writes_steps(self, tmp_path):
        app, ui = await self._ui(tmp_path)
        async with app.run_test():
            from textual.widgets import RichLog
            log = app.query_one("#chat-log", RichLog)
            before = len(log.lines)
            plan = Plan(goal="do a thing", steps=[
                PlanStep(index=1, description="step one", status=StepStatus.pending),
                PlanStep(index=2, description="step two", status=StepStatus.pending),
            ])
            ui.on_plan(plan)
            assert len(log.lines) >= before + 3  # goal + 2 steps

    async def test_on_tool_result_shows_diff_in_preview(self, tmp_path):
        app, ui = await self._ui(tmp_path)
        async with app.run_test():
            from textual.widgets import RichLog
            preview = app.query_one("#preview-log", RichLog)
            result = "Wrote file.\n```diff\n-old\n+new\n```"
            ui.on_tool_result("write", result)
            text = "\n".join(str(line) for line in preview.lines)
            assert "new" in text or len(preview.lines) > 0

    async def test_on_tool_result_without_diff_does_not_touch_preview(self, tmp_path):
        app, ui = await self._ui(tmp_path)
        async with app.run_test():
            from textual.widgets import RichLog
            preview = app.query_one("#preview-log", RichLog)
            before = len(preview.lines)
            ui.on_tool_result("read", "some file contents, no diff block")
            assert len(preview.lines) == before

    async def test_on_error_writes_message(self, tmp_path):
        app, ui = await self._ui(tmp_path)
        async with app.run_test():
            from textual.widgets import RichLog
            log = app.query_one("#chat-log", RichLog)
            before = len(log.lines)
            ui.on_error("something broke")
            assert len(log.lines) > before

    async def test_on_tool_result_does_not_crash_on_malformed_markup(self, tmp_path):
        # Regression test: goto_definition and similar tools can return text
        # with brackets that read as malformed Rich markup (e.g. an unmatched
        # closing tag), which previously raised MarkupError and silently
        # killed the Textual worker running the task (reported as the UI
        # "freezing" after the goto_definition step).
        app, ui = await self._ui(tmp_path)
        async with app.run_test():
            from textual.widgets import RichLog
            log = app.query_one("#chat-log", RichLog)
            before = len(log.lines)
            ui.on_tool_result("goto_definition", "foo[/bar] and Dict[str, int]")
            assert len(log.lines) > before
            text = "\n".join(str(line) for line in log.lines)
            assert "foo[/bar]" in text
            assert "Dict[str, int]" in text

    async def test_display_result_success(self, tmp_path):
        app, ui = await self._ui(tmp_path)
        async with app.run_test():
            from textual.widgets import RichLog
            log = app.query_one("#chat-log", RichLog)
            before = len(log.lines)
            result = AgentResult(
                plan=Plan(goal="test", steps=[]),
                steps_taken=2,
                message="All done",
                success=True,
            )
            ui.display_result(result)
            assert len(log.lines) > before

    async def test_on_approval_request_approve(self, tmp_path):
        # push_screen_wait requires running inside a Textual worker, matching
        # how on_approval_request is actually invoked in production (from
        # within the ReActLoop task started via app.run_worker in
        # on_input_submitted).
        app, ui = await self._ui(tmp_path)
        async with app.run_test() as pilot:
            tc = ParsedToolCall(name="bash", arguments={"command": "rm -rf /tmp/x"}, raw="")
            results: list[bool] = []

            async def request() -> None:
                results.append(await ui.on_approval_request(tc))

            app.run_worker(request())
            await pilot.pause()
            await pilot.click("#approve")
            await pilot.pause()
            assert results == [True]

    async def test_on_approval_request_reject(self, tmp_path):
        app, ui = await self._ui(tmp_path)
        async with app.run_test() as pilot:
            tc = ParsedToolCall(name="bash", arguments={"command": "rm -rf /tmp/x"}, raw="")
            results: list[bool] = []

            async def request() -> None:
                results.append(await ui.on_approval_request(tc))

            app.run_worker(request())
            await pilot.pause()
            await pilot.click("#reject")
            await pilot.pause()
            assert results == [False]

    async def test_on_ask_user_submits_typed_answer(self, tmp_path):
        """Regression test: AskUserTool.run()'s own sys.stdin.readline()
        deadlocks under the Textual TUI (it owns the terminal in raw mode,
        so a blocking synchronous stdin read both hangs the event loop and
        can never receive real input). on_ask_user must collect the answer
        through Textual's own event loop via a modal instead."""
        app, ui = await self._ui(tmp_path)
        async with app.run_test() as pilot:
            results: list[str] = []

            async def request() -> None:
                results.append(await ui.on_ask_user("Which database should I use?"))

            app.run_worker(request())
            await pilot.pause()
            await pilot.click("#ask-user-input")
            for ch in "SQLite":
                await pilot.press(ch)
            await pilot.click("#submit")
            await pilot.pause()
            assert results == ["SQLite"]

    async def test_on_ask_user_skip_returns_default_message(self, tmp_path):
        app, ui = await self._ui(tmp_path)
        async with app.run_test() as pilot:
            results: list[str] = []

            async def request() -> None:
                results.append(await ui.on_ask_user("Which database should I use?"))

            app.run_worker(request())
            await pilot.pause()
            await pilot.click("#skip")
            await pilot.pause()
            assert results == ["User skipped the question"]


class TestSessionStats:
    def test_defaults(self):
        s = SessionStats()
        assert s.tasks_run == 0
        assert s.status == "Idle"
        assert s.warnings == []

    def test_add_warning_appends(self):
        s = SessionStats()
        s.add_warning("first warning")
        assert s.warnings == ["first warning"]

    def test_add_warning_caps_to_max_kept(self):
        s = SessionStats()
        for i in range(10):
            s.add_warning(f"warning {i}")
        assert len(s.warnings) == 5
        # Keeps the most recent ones, drops the oldest.
        assert s.warnings == [f"warning {i}" for i in range(5, 10)]


class TestStatsPanelUpdates:
    async def _ui(self, tmp_path):
        app, _ = await _make_app(tmp_path)
        return app, app.tui_ui

    def _panel_text(self, app) -> str:
        from textual.widgets import RichLog
        panel = app.query_one("#stats-panel", RichLog)
        return "\n".join(str(line) for line in panel.lines)

    async def test_on_start_increments_tasks_run(self, tmp_path):
        app, ui = await self._ui(tmp_path)
        async with app.run_test():
            ui.on_start("fix the bug")
            assert ui.stats.tasks_run == 1
            assert ui.stats.status == "Running"
            assert "Tasks run: 1" in self._panel_text(app)

    async def test_on_tool_call_increments_tool_calls(self, tmp_path):
        app, ui = await self._ui(tmp_path)
        async with app.run_test():
            tc = ParsedToolCall(name="read", arguments={"path": "a.py"}, raw="")
            ui.on_tool_call(tc)
            ui.on_tool_call(tc)
            assert ui.stats.tool_calls == 2

    async def test_on_tool_result_error_adds_warning(self, tmp_path):
        app, ui = await self._ui(tmp_path)
        async with app.run_test():
            ui.on_tool_result("read", "Error: file not found")
            assert ui.stats.tool_errors == 1
            assert len(ui.stats.warnings) == 1
            assert "read" in self._panel_text(app)

    async def test_on_tool_result_plan_check_adds_warning(self, tmp_path):
        """plan-check / repeat-guard are the synthetic corrective messages
        from react_loop.py -- they don't start with "Error" but should
        still surface as warnings, since they signal the agent got
        redirected (repeated a call, tried to finish early, etc.)."""
        app, ui = await self._ui(tmp_path)
        async with app.run_test():
            ui.on_tool_result("plan-check", "Step 2 is not fully done yet (partial): ...")
            assert ui.stats.tool_errors == 1
            assert any("plan-check" in w for w in ui.stats.warnings)

    async def test_on_tool_result_success_does_not_add_warning(self, tmp_path):
        app, ui = await self._ui(tmp_path)
        async with app.run_test():
            ui.on_tool_result("read", "file contents here")
            assert ui.stats.tool_errors == 0
            assert ui.stats.warnings == []

    async def test_on_replan_increments_and_warns(self, tmp_path):
        app, ui = await self._ui(tmp_path)
        async with app.run_test():
            ui.on_replan()
            assert ui.stats.replans == 1
            assert len(ui.stats.warnings) == 1
            assert ui.stats.status == "Re-planning..."

    async def test_on_error_adds_warning(self, tmp_path):
        app, ui = await self._ui(tmp_path)
        async with app.run_test():
            ui.on_error("something broke")
            assert ui.stats.warnings == ["something broke"]

    async def test_on_plan_updates_current_step_and_progress(self, tmp_path):
        app, ui = await self._ui(tmp_path)
        async with app.run_test():
            plan = Plan(goal="g", steps=[
                PlanStep(index=1, description="done step", status=StepStatus.completed),
                PlanStep(index=2, description="active step", status=StepStatus.pending),
                PlanStep(index=3, description="later step", status=StepStatus.pending),
            ])
            ui.on_plan(plan)
            assert ui.stats.current_step == "active step"
            assert ui.stats.steps_done == 1
            assert ui.stats.steps_total == 3
            text = self._panel_text(app)
            assert "active step" in text
            assert "1/3" in text

    async def test_on_token_usage_accumulates_live(self, tmp_path):
        """Tokens accumulate live, per-LLM-call, via on_token_usage() as a
        task runs -- not just once at the end via display_result() -- so the
        counter updates during a long-running task instead of staying at 0
        until it finishes."""
        app, ui = await self._ui(tmp_path)
        async with app.run_test():
            ui.on_token_usage(100, 20)
            assert ui.stats.total_prompt_tokens == 100
            assert ui.stats.total_completion_tokens == 20
            ui.on_token_usage(50, 10)
            assert ui.stats.total_prompt_tokens == 150
            assert ui.stats.total_completion_tokens == 30

    async def test_display_result_does_not_double_count_tokens(self, tmp_path):
        """Tokens are already accumulated live via on_token_usage() during
        the run; display_result() must not add result.total_prompt_tokens/
        total_completion_tokens again on top, or every token gets counted
        twice."""
        app, ui = await self._ui(tmp_path)
        async with app.run_test():
            ui.on_token_usage(100, 20)
            result = AgentResult(
                plan=Plan(goal="g", steps=[]), steps_taken=1, message="done",
                success=True, total_prompt_tokens=100, total_completion_tokens=20,
            )
            ui.display_result(result)
            assert ui.stats.total_prompt_tokens == 100
            assert ui.stats.total_completion_tokens == 20

    async def test_display_result_sets_status_idle_on_success(self, tmp_path):
        app, ui = await self._ui(tmp_path)
        async with app.run_test():
            result = AgentResult(
                plan=Plan(goal="g", steps=[]), steps_taken=1, message="done", success=True,
            )
            ui.display_result(result)
            assert ui.stats.status == "Idle"

    async def test_display_result_sets_status_failed_on_failure(self, tmp_path):
        app, ui = await self._ui(tmp_path)
        async with app.run_test():
            result = AgentResult(
                plan=Plan(goal="g", steps=[]), steps_taken=1, message="failed", success=False,
            )
            ui.display_result(result)
            assert ui.stats.status == "Failed"

    async def test_no_warnings_message_when_clean(self, tmp_path):
        app, ui = await self._ui(tmp_path)
        async with app.run_test():
            ui.render_stats()
            assert "No warnings" in self._panel_text(app)


class TestApprovalModal:
    def test_dismiss_on_approve(self):
        modal = ApprovalModal("rm -rf /tmp/x")
        results = []
        modal.dismiss = lambda v: results.append(v)
        from textual.widgets import Button
        approve_btn = Button("Approve", id="approve")
        modal.on_button_pressed(Button.Pressed(approve_btn))
        assert results == [True]

    def test_dismiss_on_reject(self):
        modal = ApprovalModal("rm -rf /tmp/x")
        results = []
        modal.dismiss = lambda v: results.append(v)
        from textual.widgets import Button
        reject_btn = Button("Reject", id="reject")
        modal.on_button_pressed(Button.Pressed(reject_btn))
        assert results == [False]
