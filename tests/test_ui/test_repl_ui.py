from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from chef_human.agent.parser import ParsedToolCall
from chef_human.agent.planner import Plan, PlanStep, StepStatus
from chef_human.agent.react_loop import AgentResult


@pytest.fixture
def ui():
    from chef_human.ui.repl import ReplUI
    return ReplUI()


class TestReplUIProtocol:
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

    def test_has_all_protocol_methods(self, ui):
        for method in self.PROTOCOL_METHODS:
            assert hasattr(ui, method), f"ReplUI missing {method}"
            assert callable(getattr(ui, method))


class TestReplUIDisplay:
    def test_display_result_success(self, ui):
        result = AgentResult(
            plan=Plan(goal="test", steps=[]),
            steps_taken=3,
            message="Done well",
            success=True,
        )
        with patch.object(ui._console, "print") as mock_print:
            ui.display_result(result)
            texts = [str(c) for c in mock_print.call_args_list]
            assert any("✓ Success" in t for t in texts)
            assert any("3" in t for t in texts)

    def test_display_result_failure(self, ui):
        result = AgentResult(
            plan=Plan(goal="test", steps=[]),
            steps_taken=0,
            message="Failed",
            success=False,
        )
        with patch.object(ui._console, "print") as mock_print:
            ui.display_result(result)
            texts = [str(c) for c in mock_print.call_args_list]
            assert any("✗ Failed" in t for t in texts)

    def test_display_result_shows_message(self, ui):
        result = AgentResult(
            plan=Plan(goal="test", steps=[]),
            steps_taken=1,
            message="Added feature X",
            success=True,
        )
        with patch.object(ui._console, "print") as mock_print:
            ui.display_result(result)
            texts = [str(c) for c in mock_print.call_args_list]
            assert any("Added feature X" in t for t in texts)

    def test_display_result_shows_tokens(self, ui):
        result = AgentResult(
            plan=Plan(goal="test", steps=[]),
            steps_taken=1,
            message="Done",
            success=True,
            total_prompt_tokens=100,
            total_completion_tokens=50,
        )
        with patch.object(ui._console, "print") as mock_print:
            ui.display_result(result)
            texts = [str(c) for c in mock_print.call_args_list]
            assert any("100" in t and "50" in t for t in texts)

    def test_display_result_no_tokens(self, ui):
        result = AgentResult(
            plan=Plan(goal="test", steps=[]),
            steps_taken=1,
            message="Done",
            success=True,
            total_prompt_tokens=0,
            total_completion_tokens=0,
        )
        with patch.object(ui._console, "print") as mock_print:
            ui.display_result(result)
            texts = [str(c) for c in mock_print.call_args_list]
            assert not any("Tokens" in t for t in texts)

    def test_display_result_no_message(self, ui):
        result = AgentResult(
            plan=Plan(goal="test", steps=[]),
            steps_taken=1,
            message="",
            success=True,
        )
        with patch.object(ui._console, "print") as mock_print:
            ui.display_result(result)
            texts = [str(c) for c in mock_print.call_args_list]
            assert not any("Message:" in t for t in texts)


class TestReplUIInput:
    def test_read_input_returns_text(self, ui):
        with patch("chef_human.ui.repl.Prompt.ask", return_value="hello world"):
            result = ui.read_input()
            assert result == "hello world"

    def test_read_input_empty_returns_empty_string(self, ui):
        with patch("chef_human.ui.repl.Prompt.ask", return_value=""):
            result = ui.read_input()
            assert result == ""

    def test_read_input_eof_returns_none(self, ui):
        with patch("chef_human.ui.repl.Prompt.ask", side_effect=EOFError):
            result = ui.read_input()
            assert result is None

    def test_read_input_keyboard_interrupt_returns_none(self, ui):
        with patch("chef_human.ui.repl.Prompt.ask", side_effect=KeyboardInterrupt):
            result = ui.read_input()
            assert result is None

    def test_read_input_exit_returns_none(self, ui):
        with patch("chef_human.ui.repl.Prompt.ask", return_value="/exit"):
            result = ui.read_input()
            assert result is None

    def test_read_input_quit_returns_none(self, ui):
        with patch("chef_human.ui.repl.Prompt.ask", return_value="/quit"):
            result = ui.read_input()
            assert result is None

    def test_read_input_q_returns_none(self, ui):
        with patch("chef_human.ui.repl.Prompt.ask", return_value="/q"):
            result = ui.read_input()
            assert result is None

    def test_read_input_help_prints_and_returns_empty(self, ui):
        with (
            patch("chef_human.ui.repl.Prompt.ask", return_value="/help"),
            patch.object(ui._console, "print") as mock_print,
        ):
            result = ui.read_input()
            assert result == ""
            mock_print.assert_called_once()
            assert "/exit" in str(mock_print.call_args[0][0])

    def test_read_input_unknown_command_prints_warning(self, ui):
        with (
            patch("chef_human.ui.repl.Prompt.ask", return_value="/unknown"),
            patch.object(ui._console, "print") as mock_print,
        ):
            result = ui.read_input()
            assert result == ""
            texts = [str(c) for c in mock_print.call_args_list]
            assert any("Unknown" in t for t in texts)

    def test_read_input_slash_commands_returned_for_handling(self, ui):
        for cmd in ("/clear", "/save", "/tokens", "/history", "/undo", "/redo"):
            with patch("chef_human.ui.repl.Prompt.ask", return_value=cmd):
                result = ui.read_input()
                assert result == cmd


class TestReplUIEvents:
    def test_on_start_sets_current_task(self, ui):
        ui.on_start("my task")
        assert ui._current_task == "my task"

    def test_on_planning_start_prints(self, ui):
        with patch.object(ui._console, "print") as mock_print:
            ui.on_planning_start()
            mock_print.assert_called_once()

    def test_on_plan_prints_goal_and_steps(self, ui):
        plan = Plan(
            goal="test goal",
            steps=[
                PlanStep(index=1, description="step A"),
                PlanStep(index=2, description="step B", status=StepStatus.completed),
            ],
        )
        with patch.object(ui._console, "print") as mock_print:
            ui.on_plan(plan)
            texts = [str(c) for c in mock_print.call_args_list]
            assert any("test goal" in t for t in texts)
            assert any("step A" in t for t in texts)

    def test_on_stream_writes_to_stdout(self, ui):
        with (
            patch("sys.stdout.write") as mock_write,
            patch("sys.stdout.flush") as mock_flush,
        ):
            ui.on_stream("chunk")
            mock_write.assert_called_once_with("chunk")
            mock_flush.assert_called_once()

    def test_on_reasoning_prints_content(self, ui):
        with patch.object(ui._console, "print") as mock_print:
            ui.on_reasoning("the reasoning")
            mock_print.assert_called_once()
            assert "the reasoning" in str(mock_print.call_args[0][0])

    def test_on_reasoning_skips_empty(self, ui):
        with patch.object(ui._console, "print") as mock_print:
            ui.on_reasoning("")
            mock_print.assert_not_called()

    def test_on_tool_call_prints(self, ui):
        call = ParsedToolCall(name="bash", arguments={"command": "ls"}, raw='{}')
        with patch.object(ui._console, "print") as mock_print:
            ui.on_tool_call(call)
            mock_print.assert_called_once()

    def test_on_tool_result_prints_success(self, ui):
        with patch.object(ui._console, "print") as mock_print:
            ui.on_tool_result("read", "file content")
            mock_print.assert_called_once()
            assert "✓" in str(mock_print.call_args[0][0])

    def test_on_tool_result_prints_error(self, ui):
        with patch.object(ui._console, "print") as mock_print:
            ui.on_tool_result("bash", "Error: not found")
            mock_print.assert_called_once()
            assert "✗" in str(mock_print.call_args[0][0])

    def test_on_replan_prints(self, ui):
        with patch.object(ui._console, "print") as mock_print:
            ui.on_replan()
            mock_print.assert_called_once()

    def test_on_error_prints(self, ui):
        with patch.object(ui._console, "print") as mock_print:
            ui.on_error("something broke")
            mock_print.assert_called_once()
            assert "something broke" in str(mock_print.call_args[0][0])

    def test_on_approval_request_returns_none(self, ui):
        import asyncio
        call = ParsedToolCall(name="bash", arguments={"command": "ls"}, raw='{}')
        result = asyncio.run(ui.on_approval_request(call))
        assert result is None


class TestReplUIStatusIcons:
    def test_status_icons(self, ui):
        assert ui._status_icon("pending") == "○"
        assert ui._status_icon("in_progress") == "◷"
        assert ui._status_icon("completed") == "✓"
        assert ui._status_icon("failed") == "✗"
        assert ui._status_icon("skipped") == "–"
        assert ui._status_icon("unknown") == "○"
