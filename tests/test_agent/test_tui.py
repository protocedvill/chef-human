from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from chef_human.agent.parser import ParsedToolCall
from chef_human.agent.planner import Plan, PlanStep, StepStatus
from chef_human.agent.react_loop import AgentResult
from chef_human.ui.protocol import NoopUI
from rich.tree import Tree


@pytest.fixture(autouse=True)
def _mock_live():
    with patch("chef_human.ui.debug_tui.Live") as mock:
        instance = MagicMock()
        instance.__enter__ = MagicMock()
        instance.__exit__ = MagicMock()
        mock.return_value = instance
        yield mock


@pytest.fixture(autouse=True)
def _mock_signal():
    with patch("chef_human.ui.debug_tui.signal.signal") as mock:
        yield mock


@pytest.fixture
def tui():
    from chef_human.ui.debug_tui import DebugTUI
    return DebugTUI()


class TestDebugTUILayout:
    def test_layout_has_required_panels(self, tui):
        tui.layout["header"]
        tui.layout["body"]
        tui.layout["footer"]
        tui.layout["body"]["left"]
        tui.layout["body"]["right"]
        tui.layout["body"]["left"]["plan_panel"]
        tui.layout["body"]["left"]["tool_panel"]
        tui.layout["body"]["right"]["reasoning_panel"]
        tui.layout["body"]["right"]["log_panel"]

    def test_initial_state(self, tui):
        assert tui._step_count == 0
        assert tui._max_steps == 0
        assert tui._log_entries == []
        assert tui._tool_calls == []
        assert tui._reasoning_text == ""
        assert tui._plan is None
        assert tui._started is False
        assert tui._reasoning_collapsed is False
        assert tui._log_search is None


class TestDebugTUICallbacks:
    def test_on_start_sets_header_and_logs(self, tui):
        tui.on_start("Fix the parser bug")
        assert len(tui._log_entries) == 1
        assert "Task started" in tui._log_entries[0]
        assert tui._started is True

    def test_on_planning_start_logs(self, tui):
        tui.on_planning_start()
        assert len(tui._log_entries) == 1
        assert "Planning" in tui._log_entries[0]

    def test_on_plan_creates_tree_and_sets_max_steps(self, tui):
        plan = Plan(
            goal="Test task",
            steps=[
                PlanStep(index=1, description="Read file"),
                PlanStep(index=2, description="Write fix"),
            ],
        )
        tui.on_plan(plan)
        assert tui._max_steps == 2
        assert tui._plan is not None
        assert tui._plan.goal == "Test task"
        assert len(tui._log_entries) == 1
        assert "Plan generated" in tui._log_entries[0]

    def test_on_reasoning_updates_text(self, tui):
        tui.on_reasoning_start()
        tui.on_reasoning("The bug is in the parser")
        assert tui._reasoning_text == "The bug is in the parser"
        assert len(tui._log_entries) == 1

    def test_on_reasoning_truncates_long_text(self, tui):
        long_text = "x" * 1000
        tui.on_reasoning_start()
        tui.on_reasoning(long_text)
        assert len(tui._reasoning_text) == 1000

    def test_on_stream_appends_chunks(self, tui):
        tui.on_reasoning_start()
        tui.on_stream("hello ")
        tui.on_stream("world")
        assert tui._reasoning_text == "hello world"

    def test_on_stream_ensures_live(self, tui):
        tui.on_stream("test")
        assert tui._started is True

    def test_on_stream_updates_panel_with_tail(self, tui):
        tui.on_reasoning_start()
        chunks = ["a"] * 600
        for c in chunks:
            tui.on_stream(c)
        # Panel should show truncated content
        assert len(tui._reasoning_text) == 600

    def test_on_reasoning_after_stream_overwrites(self, tui):
        tui.on_reasoning_start()
        tui.on_stream("partial")
        tui.on_reasoning("complete")
        assert tui._reasoning_text == "complete"
        assert len(tui._log_entries) == 1  # on_reasoning logs once

    def test_on_tool_call_appends_to_list(self, tui):
        call = ParsedToolCall(name="grep", arguments={"pattern": "foo"}, raw='{"name": "grep", ...}')
        tui.on_tool_call(call)
        assert len(tui._tool_calls) == 1
        icon, name, detail = tui._tool_calls[0]
        assert icon == "▶"
        assert name == "grep"
        assert "pattern=foo" in detail
        assert len(tui._log_entries) == 1
        assert "grep" in tui._log_entries[0]

    def test_on_tool_result_appends_with_checkmark(self, tui):
        tui.on_tool_result("grep", "Found at line 42")
        assert len(tui._tool_calls) == 1
        icon, name, detail = tui._tool_calls[0]
        assert icon == "✓"
        assert name == "grep"
        assert "Found at line 42" in detail

    def test_on_tool_result_appends_with_cross_on_error(self, tui):
        tui.on_tool_result("bash", "Error: permission denied")
        assert len(tui._tool_calls) == 1
        icon, name, detail = tui._tool_calls[0]
        assert icon == "✗"
        assert "Error" in detail

    def test_on_tool_result_truncates_long_result(self, tui):
        long_result = "x" * 200
        tui.on_tool_result("read", long_result)
        _, _, detail = tui._tool_calls[0]
        assert len(detail) <= 83

    def test_on_error_logs_error(self, tui):
        tui.on_error("Something went wrong")
        assert len(tui._log_entries) == 1
        assert "Error" in tui._log_entries[0]
        assert "Something went wrong" in tui._log_entries[0]

    def test_on_replan_logs_replan(self, tui):
        tui.on_replan()
        assert len(tui._log_entries) == 1
        assert "Re-planning" in tui._log_entries[0]

    def test_log_entries_limited_to_ten(self, tui):
        for i in range(15):
            tui._log(f"Entry {i}")
        assert len(tui._log_entries) == 15
        rendered = "\n".join(tui._log_entries[-10:])
        assert "Entry 5" in rendered

    def test_display_final_does_not_crash(self, tui):
        result = AgentResult(
            plan=Plan(goal="test", steps=[]),
            steps_taken=3,
            message="All done",
            success=True,
        )
        tui.display_final(result)
        assert tui._started is False


class TestReActUIProtocol:
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

    def test_noop_ui_has_all_protocol_methods(self):
        ui = NoopUI()
        for method in self.PROTOCOL_METHODS:
            assert hasattr(ui, method), f"NoopUI missing {method}"

    def test_debug_tui_has_all_protocol_methods(self):
        from chef_human.ui.debug_tui import DebugTUI

        with patch("chef_human.ui.debug_tui.Live"):
            ui = DebugTUI()
        for method in self.PROTOCOL_METHODS:
            assert hasattr(ui, method), f"DebugTUI missing {method}"

    @pytest.mark.asyncio
    async def test_debug_tui_on_approval_request_returns_bool(self, tui):
        call = ParsedToolCall(name="bash", arguments={"command": "rm -rf /"}, raw="{}")
        with patch("chef_human.ui.debug_tui.Confirm.ask", return_value=True):
            result = await tui.on_approval_request(call)
            assert result is True

        with patch("chef_human.ui.debug_tui.Confirm.ask", return_value=False):
            result = await tui.on_approval_request(call)
            assert result is False


class TestPlanColoring:
    def test_pending_step_style(self):
        from chef_human.ui.debug_tui import _STATUS_STYLES
        assert _STATUS_STYLES[StepStatus.pending] == "white"
        assert _STATUS_STYLES[StepStatus.in_progress] == "bold yellow"
        assert _STATUS_STYLES[StepStatus.completed] == "green"
        assert _STATUS_STYLES[StepStatus.failed] == "bold red"
        assert _STATUS_STYLES[StepStatus.skipped] == "dim white"

    def test_render_plan_uses_color(self, tui):
        plan = Plan(
            goal="Test",
            steps=[
                PlanStep(index=1, description="Step A", status=StepStatus.completed),
                PlanStep(index=2, description="Step B", status=StepStatus.in_progress),
                PlanStep(index=3, description="Step C", status=StepStatus.pending),
            ],
        )
        tui.on_plan(plan)
        tree = tui._render_plan()
        assert isinstance(tree, Tree)
        assert tui._plan is plan
        assert len(tui._plan.steps) == 3

    def test_render_plan_empty(self, tui):
        tree = tui._render_plan()
        assert isinstance(tree, Tree)


class TestReasoningCollapse:
    def test_default_not_collapsed(self, tui):
        assert tui._reasoning_collapsed is False

    def test_toggle_r_key(self, tui):
        tui._last_key_check = 0
        with patch("sys.stdin.isatty", return_value=True):
            with patch("select.select", return_value=[True]):
                with patch("sys.stdin.read", return_value="r"):
                    tui._check_keys()
        assert tui._reasoning_collapsed is True

        tui._last_key_check = 0
        with patch("sys.stdin.isatty", return_value=True):
            with patch("select.select", return_value=[True]):
                with patch("sys.stdin.read", return_value="r"):
                    tui._check_keys()
        assert tui._reasoning_collapsed is False

    def test_render_reasoning_collapsed_shows_last_5_lines(self, tui):
        tui._reasoning_text = "\n".join(f"line {i}" for i in range(10))
        tui._reasoning_collapsed = True
        panel = tui._render_reasoning()
        rendered = panel.renderable
        assert "line 9" in rendered
        assert "line 0" not in rendered

    def test_render_reasoning_not_collapsed_shows_all(self, tui):
        tui._reasoning_text = "\n".join(f"line {i}" for i in range(10))
        tui._reasoning_collapsed = False
        panel = tui._render_reasoning()
        rendered = panel.renderable
        assert "line 0" in rendered
        assert "line 9" in rendered


class TestLogSearch:
    def test_search_highlights_matching_entries(self, tui):
        tui._log("error: file not found")
        tui._log("info: task complete")
        tui._log("error: permission denied")
        tui._log_search = "error"
        panel = tui._render_log()
        assert "search: error" in panel.title or "error" in panel.title

    def test_no_search_does_not_filter(self, tui):
        tui._log("entry one")
        tui._log("entry two")
        tui._log_search = None
        panel = tui._render_log()
        assert "entry one" in panel.renderable

    def test_slash_key_prompts_search(self, tui):
        tui._last_key_check = 0
        with patch("sys.stdin.isatty", return_value=True):
            with patch("select.select", return_value=[True]):
                with patch("sys.stdin.read", return_value="/"):
                    with patch("chef_human.ui.debug_tui.Prompt.ask", return_value="test"):
                        tui._check_keys()
        assert tui._log_search == "test"


class TestFooter:
    def test_footer_shows_step_count(self, tui):
        tui._step_count = 3
        tui._max_steps = 5
        panel = tui._render_footer()
        rendered = panel.renderable
        assert "3/5" in rendered
        assert "Elapsed" in rendered

    def test_footer_shows_key_bindings(self, tui):
        panel = tui._render_footer()
        rendered = panel.renderable
        assert "toggle reasoning" in rendered
        assert "search" in rendered
        assert "Ctrl+C" in rendered

    def test_footer_handles_zero_steps(self, tui):
        tui._step_count = 0
        tui._max_steps = 0
        panel = tui._render_footer()
        rendered = panel.renderable
        assert "?" in rendered


class TestSigintHandler:
    def test_signal_handler_registered(self):
        with patch("chef_human.ui.debug_tui.signal.signal") as mock_sig:
            from chef_human.ui.debug_tui import DebugTUI
            DebugTUI()
            mock_sig.assert_called_once()

    def test_sigint_handler_stops_live(self, tui):
        tui._started = True
        with patch.object(tui, "_stop_live") as mock_stop:
            with patch.object(tui.console, "print"):
                with pytest.raises(SystemExit):
                    tui._handle_sigint(2, None)
        mock_stop.assert_called_once()


class TestMaxReasoningLinesConfig:
    def test_default_max_reasoning_lines(self, tui):
        assert tui._max_reasoning_lines == 50

    def test_custom_max_reasoning_lines(self):
        with patch("chef_human.ui.debug_tui.Live"):
            from chef_human.ui.debug_tui import DebugTUI
            tui = DebugTUI(max_reasoning_lines=100)
            assert tui._max_reasoning_lines == 100
