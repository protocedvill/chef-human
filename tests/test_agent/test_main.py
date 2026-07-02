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

    def test_json_flag_in_help(self, runner):
        from chef_human.main import cli

        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--json" in result.output

    def test_json_flag_outputs_json_after_rich(self, runner):
        from chef_human.main import cli

        with patch("chef_human.main._execute_task", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = AgentResult(
                plan=Plan(goal="test", steps=[]),
                steps_taken=1,
                message="Done",
                success=True,
            )
            result = runner.invoke(cli, ["run", "test", "--json"])
            assert result.exit_code == 0
            import json
            json_start = result.output.index("{")
            data = json.loads(result.output[json_start:])
            assert data["success"] is True
            assert data["message"] == "Done"

    def test_json_headless_precedence(self, runner):
        from chef_human.main import cli

        with patch("chef_human.main._execute_task", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = AgentResult(
                plan=Plan(goal="test", steps=[]),
                steps_taken=1,
                message="Done",
                success=True,
            )
            result = runner.invoke(cli, ["run", "test", "--headless", "--json"])
            assert result.exit_code == 0
            import json
            data = json.loads(result.output)
            assert data["success"] is True

    def test_stdin_pipe_reads_task(self, runner):
        from chef_human.main import cli

        with (
            patch("chef_human.main._execute_task", new_callable=AsyncMock) as mock_exec,
            patch("sys.stdin.isatty", return_value=False),
        ):
            mock_exec.return_value = AgentResult(
                plan=Plan(goal="test", steps=[]),
                steps_taken=1,
                message="Done",
                success=True,
            )
            result = runner.invoke(cli, ["run"], input="task from stdin\n")
            assert result.exit_code == 0
            mock_exec.assert_awaited_once()
            assert mock_exec.call_args[1]["task"] == "task from stdin"

    def test_interactive_mode_prompts_for_task(self, runner):
        from chef_human.main import cli

        with (
            patch("chef_human.main._execute_task", new_callable=AsyncMock) as mock_exec,
            patch("sys.stdin.isatty", return_value=True),
        ):
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

    def test_no_task_exits(self, runner):
        from chef_human.main import cli

        with patch("sys.stdin.isatty", return_value=False):
            result = runner.invoke(cli, ["run"], input="\n")
            assert result.exit_code == 0
            assert "No task provided" in result.output

    def test_main_routes_inline_task(self):
        with (
            patch("chef_human.main.cli") as mock_cli,
        ):
            import sys
            from chef_human.main import main
            with patch.object(sys, "argv", ["chef-human", "fix this bug"]):
                main()
                mock_cli.assert_called_once()

    def test_main_does_not_route_known_subcommands(self):
        import sys
        from chef_human.main import main

        with patch("chef_human.main.cli") as mock_cli:
            with patch.object(sys, "argv", ["chef-human", "run", "--help"]):
                main()
            with patch.object(sys, "argv", ["chef-human", "repl"]):
                main()
            with patch.object(sys, "argv", ["chef-human", "session", "list"]):
                main()
        assert mock_cli.call_count == 3

    def test_main_routes_help_without_insertion(self):
        import sys
        from chef_human.main import main

        with (
            patch("chef_human.main.cli") as mock_cli,
        ):
            with patch.object(sys, "argv", ["chef-human", "--help"]):
                main()
            mock_cli.assert_called_once()

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

    # ── 5.1.5 Configuration Overrides ─────────────────────────────────

    def test_run_model_flag_in_help(self, runner):
        from chef_human.main import cli

        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output

    def test_run_temperature_flag_in_help(self, runner):
        from chef_human.main import cli

        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--temperature" in result.output

    def test_run_config_flag_in_help(self, runner):
        from chef_human.main import cli

        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output

    def test_show_config_command_in_help(self, runner):
        from chef_human.main import cli

        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "show-config" in result.output

    def test_show_config_displays_settings(self, runner):
        from chef_human.main import cli

        result = runner.invoke(cli, ["show-config"])
        assert result.exit_code == 0
        assert "ollama_model" in result.output
        assert "temperature" in result.output

    def test_show_config_shows_model_tip_when_upgrade_available(self, runner):
        from chef_human.agent.hardware import HardwareInfo
        from chef_human.config import Settings
        from chef_human.main import cli

        with (
            patch(
                "chef_human.main._resolve_settings",
                return_value=Settings(ollama_model="qwen2.5-coder:7b"),
            ),
            patch(
                "chef_human.agent.model_advisor.detect_hardware",
                return_value=HardwareInfo(ram_gb=64.0, vram_gb=None),
            ),
        ):
            result = runner.invoke(cli, ["show-config"])
        assert result.exit_code == 0
        assert "Tip:" in result.output
        assert "recommend-model" in result.output

    def test_show_config_no_tip_when_no_upgrade_available(self, runner):
        from chef_human.agent.hardware import HardwareInfo
        from chef_human.main import cli

        with patch(
            "chef_human.agent.model_advisor.detect_hardware",
            return_value=HardwareInfo(ram_gb=None, vram_gb=None),
        ):
            result = runner.invoke(cli, ["show-config"])
        assert result.exit_code == 0
        assert "Tip:" not in result.output


class TestRecommendModelCLI:
    def test_command_in_help(self, runner):
        from chef_human.main import cli

        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "recommend-model" in result.output

    def test_shows_detected_hardware(self, runner):
        from chef_human.agent.hardware import HardwareInfo
        from chef_human.main import cli

        with patch(
            "chef_human.agent.hardware.detect_hardware",
            return_value=HardwareInfo(ram_gb=32.0, vram_gb=8.0),
        ):
            result = runner.invoke(cli, ["recommend-model"])
        assert result.exit_code == 0
        assert "32.0 GB" in result.output
        assert "8.0 GB" in result.output

    def test_shows_recommendation_when_upgrade_available(self, runner):
        from chef_human.agent.hardware import HardwareInfo
        from chef_human.config import Settings
        from chef_human.main import cli

        with (
            patch(
                "chef_human.main._resolve_settings",
                return_value=Settings(ollama_model="qwen2.5-coder:7b"),
            ),
            patch(
                "chef_human.agent.hardware.detect_hardware",
                return_value=HardwareInfo(ram_gb=64.0, vram_gb=None),
            ),
        ):
            result = runner.invoke(cli, ["recommend-model"])
        assert result.exit_code == 0
        assert "Recommendation:" in result.output
        assert "ollama pull" in result.output

    def test_shows_already_best_fit_message(self, runner):
        from chef_human.agent.hardware import HardwareInfo
        from chef_human.config import Settings
        from chef_human.main import cli

        with (
            patch(
                "chef_human.main._resolve_settings",
                return_value=Settings(ollama_model="qwen2.5-coder:14b"),
            ),
            patch(
                "chef_human.agent.hardware.detect_hardware",
                return_value=HardwareInfo(ram_gb=26.0, vram_gb=None),
            ),
        ):
            result = runner.invoke(cli, ["recommend-model"])
        assert result.exit_code == 0
        assert "already using the best-fit model" in result.output

    def test_no_hardware_detected_message(self, runner):
        from chef_human.agent.hardware import HardwareInfo
        from chef_human.main import cli

        with patch(
            "chef_human.agent.hardware.detect_hardware",
            return_value=HardwareInfo(ram_gb=None, vram_gb=None),
        ):
            result = runner.invoke(cli, ["recommend-model"])
        assert result.exit_code == 0
        assert "Could not detect" in result.output


class TestMainSubcommandDispatch:
    """Regression coverage for the bug where `chef-human <command-name>` for
    any command not in a hand-maintained set got misrouted into `run` with
    the command name treated as the task string."""

    def test_show_config_is_not_treated_as_a_task(self):
        import sys
        from unittest.mock import patch as mock_patch

        from chef_human.main import main

        with mock_patch.object(sys, "argv", ["chef-human", "show-config"]):
            with mock_patch("chef_human.main._execute_task") as mock_exec:
                try:
                    main()
                except SystemExit:
                    pass
        mock_exec.assert_not_called()

    def test_recommend_model_is_not_treated_as_a_task(self):
        import sys
        from unittest.mock import patch as mock_patch

        from chef_human.main import main

        with mock_patch.object(sys, "argv", ["chef-human", "recommend-model"]):
            with mock_patch("chef_human.main._execute_task") as mock_exec:
                try:
                    main()
                except SystemExit:
                    pass
        mock_exec.assert_not_called()

    def test_all_registered_commands_are_dispatch_safe(self):
        """Every top-level command name must route to itself, not `run`."""
        from chef_human.main import cli

        for name in cli.commands:
            assert not name.startswith("-")

    def test_model_flag_passed_to_execute_task(self, runner):
        from chef_human.main import cli

        with patch("chef_human.main._execute_task", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = AgentResult(
                plan=Plan(goal="test", steps=[]),
                steps_taken=1,
                message="Done",
                success=True,
            )
            result = runner.invoke(cli, ["run", "test", "--model", "llama3"])
            assert result.exit_code == 0
            kwargs = mock_exec.call_args[1]
            assert kwargs["model"] == "llama3"

    def test_temperature_flag_passed_to_execute_task(self, runner):
        from chef_human.main import cli

        with patch("chef_human.main._execute_task", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = AgentResult(
                plan=Plan(goal="test", steps=[]),
                steps_taken=1,
                message="Done",
                success=True,
            )
            result = runner.invoke(cli, ["run", "test", "--temperature", "0.5"])
            assert result.exit_code == 0
            kwargs = mock_exec.call_args[1]
            assert kwargs["temperature"] == 0.5

    def test_config_flag_passed_to_execute_task(self, runner):
        from chef_human.main import cli
        import tempfile, os

        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            f.write(b"[chef_human]\n")
            cfg_path = f.name
        try:
            with patch("chef_human.main._execute_task", new_callable=AsyncMock) as mock_exec:
                mock_exec.return_value = AgentResult(
                    plan=Plan(goal="test", steps=[]),
                    steps_taken=1,
                    message="Done",
                    success=True,
                )
                result = runner.invoke(cli, ["run", "test", "--config", cfg_path])
                assert result.exit_code == 0
                kwargs = mock_exec.call_args[1]
                assert kwargs["config_path"] == cfg_path
        finally:
            os.unlink(cfg_path)

    def test_repl_model_flag_in_help(self, runner):
        from chef_human.main import cli

        result = runner.invoke(cli, ["repl", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output


class TestResolveSettings:
    def test_returns_global_settings_without_overrides(self):
        from chef_human.main import _resolve_settings

        result = _resolve_settings(model=None, temperature=None, config_path=None)
        from chef_human.config import settings
        assert result is settings

    def test_overrides_model(self):
        from chef_human.main import _resolve_settings

        result = _resolve_settings(model="llama3", temperature=None, config_path=None)
        from chef_human.config import Settings
        assert isinstance(result, Settings)
        assert result.ollama_model == "llama3"

    def test_overrides_temperature(self):
        from chef_human.main import _resolve_settings

        result = _resolve_settings(model=None, temperature=0.5, config_path=None)
        assert result.temperature == 0.5

    def test_overrides_both(self):
        from chef_human.main import _resolve_settings

        result = _resolve_settings(model="llama3", temperature=0.5, config_path=None)
        assert result.ollama_model == "llama3"
        assert result.temperature == 0.5

    def test_does_not_mutate_global_settings(self):
        from chef_human.main import _resolve_settings
        from chef_human.config import settings

        original_model = settings.ollama_model
        _resolve_settings(model="llama3", temperature=None, config_path=None)
        assert settings.ollama_model == original_model


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

            # debug_tui=False (or headless) is required to exercise the
            # create_agent() wiring path -- debug_tui=True (the default)
            # now routes through _run_task_in_tui / the split-pane TUI.
            result = await _execute_task(
                "test task", max_steps=5, stream=False, debug_tui=False
            )

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

    # ── 5.1.6 Session Management Improvements ─────────────────────────

    def test_session_list_shows_dates(self, runner):
        from chef_human.main import cli

        sessions = [
            {"session_id": "abc123", "task": "do something", "created": 1000000},
        ]
        with patch("chef_human.main.list_sessions", return_value=sessions):
            result = runner.invoke(cli, ["session", "list"])
            assert result.exit_code == 0
            assert "abc123" in result.output

    def test_session_list_sorts_by_mtime(self, runner):
        from chef_human.main import cli

        sessions = [
            {"session_id": "newer", "task": "b", "created": 2000},
            {"session_id": "older", "task": "a", "created": 1000},
        ]
        with patch("chef_human.main.list_sessions", return_value=sessions):
            result = runner.invoke(cli, ["session", "list"])
            assert result.exit_code == 0
            newer_pos = result.output.index("newer")
            older_pos = result.output.index("older")
            assert newer_pos < older_pos

    def test_default_save_dir_is_project_relative(self):
        from chef_human.agent.persistence import DEFAULT_SAVE_DIR
        assert ".chef-human" in str(DEFAULT_SAVE_DIR)
        assert "sessions" in str(DEFAULT_SAVE_DIR)

    def test_save_conversation_includes_created_timestamp(self):
        from chef_human.agent.persistence import save_conversation, load_session_data
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_conversation(
                {"messages": []}, task="test", save_dir=tmpdir
            )
            data = load_session_data(path.stem.split("_", 1)[1], save_dir=tmpdir)
            assert data is not None
            assert "created" in data
            assert isinstance(data["created"], (int, float))
            assert data["created"] > 0

    def test_continue_alias_in_run_help(self, runner):
        from chef_human.main import cli

        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--continue" in result.output

    def test_continue_alias_in_repl_help(self, runner):
        from chef_human.main import cli

        result = runner.invoke(cli, ["repl", "--help"])
        assert result.exit_code == 0
        assert "--continue" in result.output

    def test_continue_passed_as_resume(self, runner):
        from chef_human.main import cli

        with patch("chef_human.main._execute_task", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = AgentResult(
                plan=Plan(goal="test", steps=[]),
                steps_taken=1,
                message="Done",
                success=True,
            )
            with patch("chef_human.main.load_session_data", return_value=None):
                result = runner.invoke(cli, ["run", "test", "--continue", "abc123"])
                assert result.exit_code == 1
                assert "not found" in result.output
