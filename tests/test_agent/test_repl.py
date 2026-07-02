from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from chef_human.agent.parser import ParsedToolCall
from chef_human.agent.planner import Plan, PlanStep, StepStatus
from chef_human.agent.react_loop import AgentResult


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def repl_ui():
    from chef_human.ui.repl import ReplUI
    return ReplUI()


class TestReplCLI:
    def test_repl_command_in_help(self, runner):
        from chef_human.main import cli
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "repl" in result.output

    def test_repl_command_help(self, runner):
        from chef_human.main import cli
        result = runner.invoke(cli, ["repl", "--help"])
        assert result.exit_code == 0
        assert "--max-steps" in result.output
        assert "--workspace" in result.output
        assert "--resume" in result.output
        assert "--save-dir" in result.output


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

    def test_has_all_protocol_methods(self, repl_ui):
        for method in self.PROTOCOL_METHODS:
            assert hasattr(repl_ui, method), f"ReplUI missing {method}"


class TestReplUIReadInput:
    def test_exit_command_returns_none(self, repl_ui):
        with patch("rich.prompt.Prompt.ask", return_value="/exit"):
            assert repl_ui.read_input() is None

    def test_quit_command_returns_none(self, repl_ui):
        with patch("rich.prompt.Prompt.ask", return_value="/quit"):
            assert repl_ui.read_input() is None

    def test_q_command_returns_none(self, repl_ui):
        with patch("rich.prompt.Prompt.ask", return_value="/q"):
            assert repl_ui.read_input() is None

    def test_help_command_prints_and_returns_empty(self, repl_ui):
        with (
            patch("rich.prompt.Prompt.ask", return_value="/help"),
            patch.object(repl_ui._console, "print") as mock_print,
        ):
            result = repl_ui.read_input()
            assert result == ""
            mock_print.assert_called_once()

    def test_unknown_command_prints_warning(self, repl_ui):
        with (
            patch("rich.prompt.Prompt.ask", return_value="/bogus"),
            patch.object(repl_ui._console, "print") as mock_print,
        ):
            result = repl_ui.read_input()
            assert result == ""
            mock_print.assert_called_once()
            assert "Unknown command" in mock_print.call_args[0][0]

    def test_empty_input_returns_empty(self, repl_ui):
        with patch("rich.prompt.Prompt.ask", return_value=""):
            result = repl_ui.read_input()
            assert result == ""

    def test_normal_text_returns_text(self, repl_ui):
        with patch("rich.prompt.Prompt.ask", return_value="add a test"):
            result = repl_ui.read_input()
            assert result == "add a test"

    def test_eof_error_returns_none(self, repl_ui):
        with patch("rich.prompt.Prompt.ask", side_effect=EOFError()):
            assert repl_ui.read_input() is None

    def test_keyboard_interrupt_returns_none(self, repl_ui):
        with patch("rich.prompt.Prompt.ask", side_effect=KeyboardInterrupt()):
            assert repl_ui.read_input() is None

    def test_clear_command(self, repl_ui):
        with patch("rich.prompt.Prompt.ask", return_value="/clear"):
            assert repl_ui.read_input() == "/clear"

    def test_save_command(self, repl_ui):
        with patch("rich.prompt.Prompt.ask", return_value="/save"):
            assert repl_ui.read_input() == "/save"

    def test_tokens_command(self, repl_ui):
        with patch("rich.prompt.Prompt.ask", return_value="/tokens"):
            assert repl_ui.read_input() == "/tokens"

    def test_history_command(self, repl_ui):
        with patch("rich.prompt.Prompt.ask", return_value="/history"):
            assert repl_ui.read_input() == "/history"

    def test_undo_command(self, repl_ui):
        with patch("rich.prompt.Prompt.ask", return_value="/undo"):
            assert repl_ui.read_input() == "/undo"

    def test_redo_command(self, repl_ui):
        with patch("rich.prompt.Prompt.ask", return_value="/redo"):
            assert repl_ui.read_input() == "/redo"


class TestReplUICallbacks:
    def test_on_start_prints_task(self, repl_ui):
        with patch.object(repl_ui._console, "print") as mock_print:
            repl_ui.on_start("fix the bug")
            mock_print.assert_called_once()
            assert "fix the bug" in mock_print.call_args[0][0]

    def test_on_planning_start_prints(self, repl_ui):
        with patch.object(repl_ui._console, "print") as mock_print:
            repl_ui.on_planning_start()
            mock_print.assert_called_once()

    def test_on_plan_prints_goal_and_steps(self, repl_ui):
        plan = Plan(
            goal="Test task",
            steps=[
                PlanStep(index=1, description="Read file"),
                PlanStep(index=2, description="Write fix"),
            ],
        )
        with patch.object(repl_ui._console, "print") as mock_print:
            repl_ui.on_plan(plan)
            assert mock_print.call_count >= 2

    def test_on_stream_writes_to_stdout(self, repl_ui):
        with patch("sys.stdout.write") as mock_write:
            repl_ui.on_stream("hello")
            mock_write.assert_called_once_with("hello")

    def test_on_reasoning_prints_when_content_exists(self, repl_ui):
        with patch.object(repl_ui._console, "print") as mock_print:
            repl_ui.on_reasoning("some reasoning")
            mock_print.assert_called_once()
            assert "some reasoning" in mock_print.call_args[0][0]

    def test_on_reasoning_skips_empty_content(self, repl_ui):
        with patch.object(repl_ui._console, "print") as mock_print:
            repl_ui.on_reasoning("")
            mock_print.assert_not_called()

    def test_on_tool_call_prints_formatted(self, repl_ui):
        call = ParsedToolCall(
            name="grep",
            arguments={"pattern": "foo"},
            raw='{"name": "grep", "arguments": {"pattern": "foo"}}',
        )
        with patch.object(repl_ui._console, "print") as mock_print:
            repl_ui.on_tool_call(call)
            mock_print.assert_called_once()
            assert "grep" in mock_print.call_args[0][0]
            assert "pattern=foo" in mock_print.call_args[0][0]

    def test_on_tool_result_prints_success(self, repl_ui):
        with patch.object(repl_ui._console, "print") as mock_print:
            repl_ui.on_tool_result("grep", "Found at line 42")
            mock_print.assert_called_once()
            assert "Found at line 42" in mock_print.call_args[0][0]

    def test_on_tool_result_prints_error(self, repl_ui):
        with patch.object(repl_ui._console, "print") as mock_print:
            repl_ui.on_tool_result("bash", "Error: permission denied")
            mock_print.assert_called_once()
            assert "Error" in mock_print.call_args[0][0]

    def test_on_replan_prints(self, repl_ui):
        with patch.object(repl_ui._console, "print") as mock_print:
            repl_ui.on_replan()
            mock_print.assert_called_once()

    def test_on_error_prints(self, repl_ui):
        with patch.object(repl_ui._console, "print") as mock_print:
            repl_ui.on_error("Something went wrong")
            mock_print.assert_called_once()
            assert "Something went wrong" in mock_print.call_args[0][0]

    def test_on_approval_request_returns_none(self, repl_ui):
        import asyncio
        call = ParsedToolCall(name="bash", arguments={"command": "ls"}, raw='{}')
        result = asyncio.run(repl_ui.on_approval_request(call))
        assert result is None

    def test_display_result_success(self, repl_ui):
        result = AgentResult(
            plan=Plan(goal="test", steps=[]),
            steps_taken=3,
            message="All done",
            success=True,
            total_prompt_tokens=100,
            total_completion_tokens=50,
        )
        with patch.object(repl_ui._console, "print") as mock_print:
            repl_ui.display_result(result)
            assert mock_print.call_count >= 1
            all_text = " ".join(str(c) for c in mock_print.call_args_list)
            assert "Success" in all_text
            assert "3" in all_text

    def test_display_result_failure(self, repl_ui):
        result = AgentResult(
            plan=Plan(goal="test", steps=[]),
            steps_taken=0,
            message="Failed",
            success=False,
        )
        with patch.object(repl_ui._console, "print") as mock_print:
            repl_ui.display_result(result)
            all_text = " ".join(str(c) for c in mock_print.call_args_list)
            assert "Failed" in all_text


class TestReplUIStatusIcon:
    def test_status_icons(self, repl_ui):
        assert repl_ui._status_icon("pending") == "○"
        assert repl_ui._status_icon("in_progress") == "◷"
        assert repl_ui._status_icon("completed") == "✓"
        assert repl_ui._status_icon("failed") == "✗"
        assert repl_ui._status_icon("skipped") == "–"
        assert repl_ui._status_icon("unknown") == "○"


class TestReplPrintHelp:
    def test_help_contains_commands(self, repl_ui):
        with patch.object(repl_ui._console, "print") as mock_print:
            repl_ui._print_help()
            mock_print.assert_called_once()
            text = mock_print.call_args[0][0]
            assert "/exit" in text
            assert "/help" in text
            assert "/save" in text
            assert "/clear" in text
            assert "/undo" in text
            assert "/redo" in text
            assert "/tokens" in text
            assert "/history" in text
