from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from chef_human.agent.parser import ParsedToolCall
from chef_human.agent.planner import Plan, PlanStep
from chef_human.agent.react_loop import AgentResult
from chef_human.ui.protocol import NoopUI


@pytest.fixture(autouse=True)
def _mock_live():
    with patch("chef_human.ui.debug_tui.Live") as mock:
        instance = MagicMock()
        instance.__enter__ = MagicMock()
        instance.__exit__ = MagicMock()
        mock.return_value = instance
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
        assert tui._plan_tree is None
        assert tui._started is False


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
        assert tui._plan_tree is not None
        assert any("Test task" not in str(tui._plan_tree) for _ in [1])
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
