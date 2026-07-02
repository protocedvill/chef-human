from __future__ import annotations


from chef_human.agent.parser import ParsedToolCall
from chef_human.agent.planner import Plan, PlanStep, StepStatus
from chef_human.agent.react_loop import AgentResult
from chef_human.ui.textual_tui import (
    ApprovalModal,
    ChefHumanTUI,
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
            assert app.query_one("#chat-log") is not None
            assert app.query_one("#preview-log") is not None
            assert app.query_one("#task-input") is not None

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
        "on_replan",
        "on_error",
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
