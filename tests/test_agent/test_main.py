from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from chef_human.agent.planner import Plan
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


class TestExecuteTask:
    @pytest.mark.asyncio
    async def test_wires_components_and_runs(self):
        mock_repl = MagicMock()
        mock_repl.run = AsyncMock(
            return_value=AgentResult(
                plan=Plan(goal="test", steps=[]),
                steps_taken=1,
                message="Done",
                success=True,
            )
        )

        with (
            patch("chef_human.main.create_context_assembler") as mock_ctx_factory,
            patch("chef_human.main.create_backend") as mock_backend_factory,
            patch("chef_human.main.create_tool_registry"),
            patch("chef_human.main.Planner"),
            patch("chef_human.main.ReActLoop", return_value=mock_repl),
            patch("chef_human.main.DebugTUI"),
        ):
            from chef_human.main import _execute_task

            result = await _execute_task("test task", max_steps=5, stream=False)

            mock_ctx_factory.assert_called_once_with(workspace_root=None)
            mock_backend_factory.assert_called_once()
            assert result.success is True
            assert result.steps_taken == 1
