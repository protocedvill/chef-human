from __future__ import annotations

from unittest.mock import patch

import pytest

from chef_human.agent.parser import ParsedToolCall
from chef_human.agent.planner import Plan, PlanStep, StepStatus
from chef_human.ui.protocol import NoopUI


@pytest.fixture
def ui():
    from chef_human.ui.streaming import StreamingUI
    return StreamingUI(quiet=False)


@pytest.fixture
def quiet_ui():
    from chef_human.ui.streaming import StreamingUI
    return StreamingUI(quiet=True)


class TestStreamingUIProtocol:
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
            assert hasattr(ui, method), f"StreamingUI missing {method}"

    def test_protocol_compliance(self):
        """StreamingUI must satisfy the ReActUI protocol (same methods as NoopUI)."""
        from chef_human.ui.streaming import StreamingUI
        ui = StreamingUI()
        for method in self.PROTOCOL_METHODS:
            assert callable(getattr(ui, method))


class TestStreamingUIOutput:
    def test_on_start_prints_task(self, ui):
        with patch.object(ui._console, "print") as mock_print:
            ui.on_start("do something")
            mock_print.assert_called_once()
            assert "do something" in mock_print.call_args[0][0]

    def test_on_start_quiet_does_not_print(self, quiet_ui):
        with patch.object(quiet_ui._console, "print") as mock_print:
            quiet_ui.on_start("do something")
            mock_print.assert_not_called()

    def test_on_planning_start_prints(self, ui):
        with patch.object(ui._console, "print") as mock_print:
            ui.on_planning_start()
            mock_print.assert_called_once()

    def test_on_planning_start_quiet_does_not_print(self, quiet_ui):
        with patch.object(quiet_ui._console, "print") as mock_print:
            quiet_ui.on_planning_start()
            mock_print.assert_not_called()

    def test_on_plan_prints_steps(self, ui):
        plan = Plan(
            goal="Test task",
            steps=[
                PlanStep(index=1, description="Read file"),
                PlanStep(index=2, description="Write fix", status=StepStatus.completed),
            ],
        )
        with patch.object(ui._console, "print") as mock_print:
            ui.on_plan(plan)
            assert mock_print.call_count >= 1
            all_text = " ".join(str(c) for c in mock_print.call_args_list)
            assert "Plan" in all_text
            assert "Read file" in all_text
            assert "Write fix" in all_text

    def test_on_plan_quiet_does_not_print(self, quiet_ui):
        plan = Plan(goal="Test", steps=[])
        with patch.object(quiet_ui._console, "print") as mock_print:
            quiet_ui.on_plan(plan)
            mock_print.assert_not_called()

    def test_on_reasoning_start_prints(self, ui):
        with patch.object(ui._console, "print") as mock_print:
            ui.on_reasoning_start()
            mock_print.assert_called_once()
            assert "Thinking" in mock_print.call_args[0][0]

    def test_on_reasoning_start_quiet_does_not_print(self, quiet_ui):
        with patch.object(quiet_ui._console, "print") as mock_print:
            quiet_ui.on_reasoning_start()
            mock_print.assert_not_called()

    def test_on_stream_writes_to_stdout(self, ui):
        with (
            patch("sys.stdout.write") as mock_write,
            patch("sys.stdout.flush") as mock_flush,
        ):
            ui.on_stream("thinking step")
            mock_write.assert_called_once_with("thinking step")
            mock_flush.assert_called_once()

    def test_on_stream_quiet_does_not_write(self, quiet_ui):
        with patch("sys.stdout.write") as mock_write:
            quiet_ui.on_stream("thinking")
            mock_write.assert_not_called()

    def test_on_reasoning_prints_content(self, ui):
        with patch.object(ui._console, "print") as mock_print:
            ui.on_reasoning("the answer is 42")
            mock_print.assert_called_once()
            assert "42" in mock_print.call_args[0][0]

    def test_on_reasoning_skips_empty(self, ui):
        with patch.object(ui._console, "print") as mock_print:
            ui.on_reasoning("")
            mock_print.assert_not_called()

    def test_on_reasoning_quiet_does_not_print(self, quiet_ui):
        with patch.object(quiet_ui._console, "print") as mock_print:
            quiet_ui.on_reasoning("content")
            mock_print.assert_not_called()

    def test_on_tool_call_prints_formatted(self, ui):
        call = ParsedToolCall(
            name="grep",
            arguments={"pattern": "foo", "path": "."},
            raw='{}',
        )
        with patch.object(ui._console, "print") as mock_print:
            ui.on_tool_call(call)
            mock_print.assert_called_once()
            text = mock_print.call_args[0][0]
            assert "grep" in text
            assert "pattern=foo" in text

    def test_on_tool_call_quiet_does_not_print(self, quiet_ui):
        call = ParsedToolCall(name="read", arguments={"path": "f.py"}, raw='{}')
        with patch.object(quiet_ui._console, "print") as mock_print:
            quiet_ui.on_tool_call(call)
            mock_print.assert_not_called()

    def test_on_tool_result_prints_success(self, ui):
        with patch.object(ui._console, "print") as mock_print:
            ui.on_tool_result("read", "file content here")
            mock_print.assert_called_once()
            assert "file content here" in mock_print.call_args[0][0]

    def test_on_tool_result_prints_error(self, ui):
        with patch.object(ui._console, "print") as mock_print:
            ui.on_tool_result("bash", "Error: not found")
            mock_print.assert_called_once()
            assert "Error" in mock_print.call_args[0][0]

    def test_on_tool_result_quiet_does_not_print(self, quiet_ui):
        with patch.object(quiet_ui._console, "print") as mock_print:
            quiet_ui.on_tool_result("read", "content")
            mock_print.assert_not_called()

    def test_on_replan_prints(self, ui):
        with patch.object(ui._console, "print") as mock_print:
            ui.on_replan()
            mock_print.assert_called_once()

    def test_on_replan_quiet_does_not_print(self, quiet_ui):
        with patch.object(quiet_ui._console, "print") as mock_print:
            quiet_ui.on_replan()
            mock_print.assert_not_called()

    def test_on_error_prints(self, ui):
        with patch.object(ui._console, "print") as mock_print:
            ui.on_error("something broke")
            mock_print.assert_called_once()
            assert "something broke" in mock_print.call_args[0][0]

    def test_on_approval_request_returns_none(self, ui):
        import asyncio
        call = ParsedToolCall(name="bash", arguments={"command": "ls"}, raw='{}')
        result = asyncio.run(ui.on_approval_request(call))
        assert result is None


class TestStreamingUIClearReasoningFlag:
    def test_reasoning_started_set_on_start(self, ui):
        assert ui._reasoning_started is False
        ui.on_reasoning_start()
        assert ui._reasoning_started is True

    def test_on_stream_clears_flag(self, ui):
        ui.on_reasoning_start()
        assert ui._reasoning_started is True
        with patch("sys.stdout.write"):
            ui.on_stream("chunk")
        assert ui._reasoning_started is False

    def test_on_reasoning_clears_flag(self, ui):
        ui.on_reasoning_start()
        assert ui._reasoning_started is True
        with patch.object(ui._console, "print"):
            ui.on_reasoning("done")
        assert ui._reasoning_started is False


class TestStreamingUIStatusIcons:
    def test_icons(self, ui):
        assert ui._get_icon(StepStatus.pending) == "○"
        assert ui._get_icon(StepStatus.in_progress) == "◷"
        assert ui._get_icon(StepStatus.completed) == "✓"
        assert ui._get_icon(StepStatus.failed) == "✗"
        assert ui._get_icon(StepStatus.skipped) == "–"
