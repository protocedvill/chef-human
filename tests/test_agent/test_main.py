from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from chef_human.agent.planner import Plan, PlanStep, StepStatus
from chef_human.agent.react_loop import AgentResult


@pytest.fixture
def runner():
    return CliRunner()


class TestCLIStructure:
    def test_cli_group_exists(self, runner):
        from chef_human.main import cli

        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output

    def test_run_command_help(self, runner):
        from chef_human.main import cli

        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--max-steps" in result.output
        assert "--workspace" in result.output
        assert "--debug-tui" in result.output
        assert "--no-stream" in result.output

    def test_run_with_task_argument(self, runner):
        from chef_human.main import cli

        with patch("chef_human.main._execute_task", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = AgentResult(
                plan=Plan(goal="test", steps=[]),
                steps_taken=1,
                message="Done",
                success=True,
            )
            result = runner.invoke(cli, ["run", "list files"])
            assert result.exit_code == 0
            mock_exec.assert_awaited_once()
            assert mock_exec.call_args[1]["task"] == "list files"

    def test_interactive_mode_prompts_for_task(self, runner):
        from chef_human.main import cli

        with patch("chef_human.main._execute_task", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = AgentResult(
                plan=Plan(goal="test", steps=[]),
                steps_taken=1,
                message="Done",
                success=True,
            )
            result = runner.invoke(cli, ["run"], input="hello\n")
            assert result.exit_code == 0
            mock_exec.assert_awaited_once()
            assert mock_exec.call_args[1]["task"] == "hello"

    def test_interactive_mode_exits_on_empty_task(self, runner):
        from chef_human.main import cli

        result = runner.invoke(cli, ["run"], input="\n")
        assert result.exit_code == 0
        assert "No task provided" in result.output

    def test_non_zero_exit_on_failure(self, runner):
        from chef_human.main import cli

        with patch("chef_human.main._execute_task", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = AgentResult(
                plan=Plan(goal="test", steps=[]),
                steps_taken=0,
                message="Failed",
                success=False,
            )
            result = runner.invoke(cli, ["run", "fail"])
            assert result.exit_code == 1
            assert "FAILURE" in result.output

    def test_run_with_options(self, runner):
        from chef_human.main import cli

        with patch("chef_human.main._execute_task", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = AgentResult(
                plan=Plan(goal="test", steps=[]),
                steps_taken=1,
                message="Done",
                success=True,
            )
            result = runner.invoke(
                cli,
                [
                    "run",
                    "test task",
                    "--max-steps",
                    "10",
                    "--no-debug-tui",
                    "--no-stream",
                    "--workspace",
                    "/tmp",
                ],
            )
            assert result.exit_code == 0
            mock_exec.assert_awaited_once()
            kwargs = mock_exec.call_args[1]
            assert kwargs["max_steps"] == 10
            assert kwargs["debug_tui"] is False
            assert kwargs["stream"] is False
            assert kwargs["workspace"] == "/tmp"

    # ── 2.2.3 Headless mode ──────────────────────────────────────────

    def test_headless_flag_in_help(self, runner):
        from chef_human.main import cli

        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--headless" in result.output

    def test_headless_forces_no_debug_tui(self, runner):
        from chef_human.main import cli

        with patch("chef_human.main._execute_task", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = AgentResult(
                plan=Plan(goal="test", steps=[]),
                steps_taken=1,
                message="Done",
                success=True,
            )
            result = runner.invoke(cli, ["run", "do it", "--headless"])
            assert result.exit_code == 0
            kwargs = mock_exec.call_args[1]
            assert kwargs["headless"] is True
            assert kwargs["debug_tui"] is False

    def test_headless_outputs_json_to_stdout(self, runner):
        from chef_human.main import cli

        with patch("chef_human.main._execute_task", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = AgentResult(
                plan=Plan(goal="test task", steps=[]),
                steps_taken=2,
                message="All done",
                success=True,
            )
            result = runner.invoke(cli, ["run", "test task", "--headless"])
            assert result.exit_code == 0
            import json

            data = json.loads(result.output)
            assert data["success"] is True
            assert data["steps_taken"] == 2
            assert data["message"] == "All done"
            assert data["plan"]["goal"] == "test task"
            assert data["plan"]["steps"] == []

    def test_headless_non_zero_exit_on_failure(self, runner):
        from chef_human.main import cli

        with patch("chef_human.main._execute_task", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = AgentResult(
                plan=Plan(goal="fail", steps=[]),
                steps_taken=0,
                message="Failed",
                success=False,
            )
            result = runner.invoke(cli, ["run", "fail", "--headless"])
            assert result.exit_code == 1
            import json

            data = json.loads(result.output)
            assert data["success"] is False
            assert data["message"] == "Failed"




class TestToDict:
    def test_agent_result_to_dict(self):
        plan = Plan(
            goal="test goal",
            steps=[
                PlanStep(index=1, description="step one", status=StepStatus.completed),
                PlanStep(index=2, description="step two", status=StepStatus.pending),
            ],
        )
        result = AgentResult(
            plan=plan,
            steps_taken=2,
            message="Done",
            success=True,
        )
        d = result.to_dict()
        assert d == {
            "success": True,
            "steps_taken": 2,
            "message": "Done",
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "plan": {
                "goal": "test goal",
                "steps": [
                    {"index": 1, "description": "step one", "status": "completed"},
                    {"index": 2, "description": "step two", "status": "pending"},
                ],
            },
        }

    def test_plan_to_dict(self):
        plan = Plan(
            goal="my goal",
            steps=[
                PlanStep(index=1, description="first", status=StepStatus.completed),
                PlanStep(index=2, description="second", status=StepStatus.pending),
            ],
        )
        d = plan.to_dict()
        assert d == {
            "goal": "my goal",
            "steps": [
                {"index": 1, "description": "first", "status": "completed"},
                {"index": 2, "description": "second", "status": "pending"},
            ],
        }

    def test_plan_step_to_dict(self):
        step = PlanStep(index=1, description="do something", status=StepStatus.in_progress)
        d = step.to_dict()
        assert d == {
            "index": 1,
            "description": "do something",
            "status": "in_progress",
        }

    def test_agent_result_to_dict_no_steps(self):
        plan = Plan(goal="empty", steps=[])
        result = AgentResult(plan=plan, steps_taken=0, message="", success=False)
        d = result.to_dict()
        assert d["success"] is False
        assert d["steps_taken"] == 0
        assert d["plan"]["steps"] == []
        assert d["total_prompt_tokens"] == 0
        assert d["total_completion_tokens"] == 0

    def test_agent_result_to_dict_with_tokens(self):
        plan = Plan(goal="test", steps=[])
        result = AgentResult(
            plan=plan,
            steps_taken=1,
            message="done",
            success=True,
            total_prompt_tokens=100,
            total_completion_tokens=50,
        )
        d = result.to_dict()
        assert d["total_prompt_tokens"] == 100
        assert d["total_completion_tokens"] == 50


class TestExecuteTask:
    @pytest.mark.asyncio
    async def test_wires_components_and_runs(self):
        mock_loop = MagicMock()
        mock_loop.run = AsyncMock(
            return_value=AgentResult(
                plan=Plan(goal="test", steps=[]),
                steps_taken=1,
                message="Done",
                success=True,
            )
        )
        mock_ctx = MagicMock()
        mock_ctx.conversation = MagicMock()

        with (
            patch("chef_human.main.create_agent", return_value=(mock_loop, mock_ctx)),
        ):
            from chef_human.main import _execute_task

            result = await _execute_task("test task", max_steps=5, stream=False)

            assert result.success is True
            assert result.steps_taken == 1


class TestSessionCLI:
    def test_session_group_in_help(self, runner):
        from chef_human.main import cli

        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "session" in result.output

    def test_session_list_empty(self, runner):
        from chef_human.main import cli

        with patch("chef_human.main.list_sessions", return_value=[]):
            result = runner.invoke(cli, ["session", "list"])
            assert result.exit_code == 0
            assert "No sessions found" in result.output

    def test_session_list_shows_sessions(self, runner):
        from chef_human.main import cli

        sessions = [
            {"session_id": "abc123", "task": "do something"},
            {"session_id": "def456", "task": "fix bug"},
        ]
        with patch("chef_human.main.list_sessions", return_value=sessions):
            result = runner.invoke(cli, ["session", "list"])
            assert result.exit_code == 0
            assert "abc123" in result.output
            assert "def456" in result.output
            assert "do something" in result.output

    def test_session_show_not_found(self, runner):
        from chef_human.main import cli

        with patch("chef_human.main.load_session_data", return_value=None):
            result = runner.invoke(cli, ["session", "show", "nonexistent"])
            assert result.exit_code == 1
            assert "not found" in result.output

    def test_session_show_displays_details(self, runner):
        from chef_human.main import cli

        data = {
            "session_id": "abc123",
            "task": "test task",
            "conversation": {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi there"},
                ]
            },
        }
        with patch("chef_human.main.load_session_data", return_value=data):
            result = runner.invoke(cli, ["session", "show", "abc123"])
            assert result.exit_code == 0
            assert "abc123" in result.output
            assert "test task" in result.output
            assert "hello" in result.output

    def test_session_delete_not_found(self, runner):
        from chef_human.main import cli

        with patch("chef_human.main.delete_session", return_value=False):
            result = runner.invoke(cli, ["session", "delete", "nonexistent"])
            assert result.exit_code == 1
            assert "not found" in result.output

    def test_session_delete_success(self, runner):
        from chef_human.main import cli

        with patch("chef_human.main.delete_session", return_value=True):
            result = runner.invoke(cli, ["session", "delete", "abc123"])
            assert result.exit_code == 0
            assert "deleted" in result.output

    def test_session_export_json(self, runner):
        from chef_human.main import cli

        data = {
            "session_id": "abc123",
            "task": "test",
            "conversation": {"messages": []},
        }
        with patch("chef_human.main.load_session_data", return_value=data):
            result = runner.invoke(cli, ["session", "export", "abc123", "--format", "json"])
            assert result.exit_code == 0
            assert '"session_id": "abc123"' in result.output

    def test_session_export_markdown(self, runner):
        from chef_human.main import cli

        data = {
            "session_id": "abc123",
            "task": "test",
            "conversation": {
                "messages": [
                    {"role": "user", "content": "Hello"},
                ]
            },
        }
        with patch("chef_human.main.load_session_data", return_value=data):
            result = runner.invoke(cli, ["session", "export", "abc123", "--format", "md"])
            assert result.exit_code == 0
            assert "# Session: abc123" in result.output
            assert "Hello" in result.output
