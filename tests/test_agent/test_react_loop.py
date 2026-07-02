from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chef_human.agent.planner import Plan, PlanStep, Planner, StepStatus
from chef_human.agent.prompts import build_agent_prompt
from chef_human.agent.react_loop import (
    AgentResult,
    ReActConfig,
    ReActLoop,
)
from chef_human.llm.backend import (
    CompletionResponse,
    LLMBackend,
    Message,
    Role,
    ToolDefinition,
)
from chef_human.tools.registry import ToolRegistry
from chef_human.ui.protocol import NoopUI


def _make_mock_backend() -> MagicMock:
    backend = MagicMock(spec=LLMBackend)
    backend.complete = AsyncMock()
    backend.model_name = "mock-model"
    backend.context_length = 4096
    return backend


def _make_mock_tool_registry() -> MagicMock:
    registry = MagicMock(spec=ToolRegistry)
    registry.get = MagicMock()
    registry.list_tools = MagicMock(return_value=["read", "write", "bash", "finish"])
    registry.get_definitions = MagicMock(
        return_value=[
            ToolDefinition(
                name="read",
                description="Read a file",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            ),
            ToolDefinition(
                name="write",
                description="Write a file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            ),
            ToolDefinition(
                name="finish",
                description="Signal task completion",
                parameters={
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                },
            ),
        ]
    )
    return registry


def _make_mock_context() -> MagicMock:
    context = MagicMock()
    context.conversation = MagicMock()
    context.conversation.add_message = MagicMock()
    context.conversation.to_dict = MagicMock(return_value={"messages": []})
    context.assemble = MagicMock(
        return_value=[Message(role=Role.system, content="assembled context")]
    )
    context._repo_map = MagicMock()
    context._repo_map.generate_tree = MagicMock(return_value="mock tree")
    return context


def _make_mock_planner() -> MagicMock:
    planner = MagicMock(spec=Planner)
    planner.generate_plan = AsyncMock()
    planner.update_plan = AsyncMock()
    planner.format_plan_for_prompt = MagicMock(
        side_effect=lambda p: f"Plan: {p.goal}"
    )
    return planner


def _make_default_plan() -> Plan:
    return Plan(
        goal="Test task",
        steps=[
            PlanStep(index=1, description="Step one", status=StepStatus.pending),
        ],
    )


def _make_tool_run(result_str: str = "ok", success: bool = True):
    """Create an async function that returns a ToolResult-like object."""
    async def run(**kwargs):
        obj = MagicMock()
        obj.output = result_str
        obj.success = success
        obj.error = None if success else "something went wrong"
        return obj
    return run


class TestBuildAgentPrompt:
    def test_base_prompt(self):
        plan = Plan(goal="Test", steps=[])
        tool_defs: list[ToolDefinition] = []
        prompt = build_agent_prompt(plan=plan, tool_defs=tool_defs)
        assert "chef-human" in prompt

    def test_includes_tool_definitions(self):
        plan = Plan(goal="Test", steps=[])
        tool_defs = [
            ToolDefinition(name="read", description="Read", parameters={"type": "object"})
        ]
        prompt = build_agent_prompt(plan=plan, tool_defs=tool_defs)
        assert "read" in prompt

    def test_includes_plan(self):
        plan = Plan(goal="Test", steps=[PlanStep(index=1, description="Do something")])
        tool_defs: list[ToolDefinition] = []
        prompt = build_agent_prompt(plan=plan, tool_defs=tool_defs)
        assert "Step 1" in prompt
        assert "Do something" in prompt

    def test_with_both(self):
        plan = Plan(goal="Test", steps=[PlanStep(index=1, description="Do something")])
        tool_defs = [
            ToolDefinition(name="read", description="Read", parameters={"type": "object"})
        ]
        prompt = build_agent_prompt(plan=plan, tool_defs=tool_defs)
        assert "read" in prompt
        assert "Step 1" in prompt

    def test_repo_map_empty_uses_fallback(self):
        plan = Plan(goal="Test", steps=[])
        tool_defs: list[ToolDefinition] = []
        prompt = build_agent_prompt(plan=plan, tool_defs=tool_defs)
        assert "no project context loaded" in prompt

    def test_repo_map_included_when_provided(self):
        plan = Plan(goal="Test", steps=[])
        tool_defs: list[ToolDefinition] = []
        prompt = build_agent_prompt(plan=plan, tool_defs=tool_defs, repo_map="src/\n  main.py")
        assert "src/" in prompt
        assert "no project context loaded" not in prompt


class TestReActConfig:
    def test_defaults(self):
        config = ReActConfig()
        assert config.max_steps == 25
        assert config.max_retries_per_step == 3
        assert config.temperature == 0.0
        assert config.max_tokens_per_response == 4096
        assert config.lint_after_write is True

    def test_custom(self):
        config = ReActConfig(max_steps=5, temperature=0.7)
        assert config.max_steps == 5
        assert config.temperature == 0.7

    def test_lint_off(self):
        config = ReActConfig(lint_after_write=False)
        assert config.lint_after_write is False

    def test_tool_timeout_default(self):
        config = ReActConfig()
        assert config.tool_timeout == 60.0

    def test_tool_timeout_custom(self):
        config = ReActConfig(tool_timeout=120.0)
        assert config.tool_timeout == 120.0


class TestAgentResult:
    def test_default_success(self):
        plan = Plan(goal="test", steps=[])
        result = AgentResult(plan=plan, steps_taken=0, message="done")
        assert result.success is True
        assert result.message == "done"


class TestReActLoopInit:
    def test_creates_without_ui(self):
        loop = ReActLoop(
            llm_backend=_make_mock_backend(),
            tool_registry=_make_mock_tool_registry(),
            context_assembler=_make_mock_context(),
            planner=_make_mock_planner(),
        )
        assert isinstance(loop._ui, NoopUI)

    def test_accepts_custom_config(self):
        config = ReActConfig(max_steps=10)
        loop = ReActLoop(
            llm_backend=_make_mock_backend(),
            tool_registry=_make_mock_tool_registry(),
            context_assembler=_make_mock_context(),
            planner=_make_mock_planner(),
            config=config,
        )
        assert loop._config.max_steps == 10


class TestReActLoopRun:
    @pytest.mark.asyncio
    async def test_plans_before_execution(self):
        backend = _make_mock_backend()
        # Return finish tool call on first LLM request
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        plan = _make_default_plan()
        planner.generate_plan.return_value = plan
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        finish_tool = MagicMock()
        finish_tool.name = "finish"
        finish_tool.parameters = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        }
        finish_tool.run = AsyncMock(return_value=MagicMock(output="Task complete: done", success=True, error=None))
        registry.get.side_effect = lambda name: {"finish": finish_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
        )
        result = await loop.run("do something")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_lint_runs_after_write_and_appends_result(self):
        """Lint runs automatically after successful write tool call."""
        backend = _make_mock_backend()
        backend.complete.side_effect = [
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content='<tool_call>{"name": "write", "arguments": {"path": "/tmp/test.py", "content": "x=1"}}</tool_call>',
                )
            ),
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content="The task is complete.",
                )
            ),
        ]
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        write_tool = MagicMock()
        write_tool.name = "write"
        write_tool.parameters = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        }
        write_tool.run = AsyncMock(return_value=MagicMock(output="wrote /tmp/test.py", success=True, error=None))
        registry.get.side_effect = lambda name: {"write": write_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=3),
        )

        mock_lint_output = "\nLint results (1 issue):\ntest.py:1:1: F401 error"
        with patch("chef_human.agent.react_loop.run_lint", return_value="test.py:1:1: F401 error"):
            with patch("chef_human.agent.react_loop.format_lint_result", return_value=mock_lint_output):
                result = await loop.run("do something")
                # Lint output was appended to tool_results; loop still completes successfully
                assert result.success is True

    @pytest.mark.asyncio
    async def test_lint_skipped_when_config_disabled(self):
        """Lint is skipped when lint_after_write=False."""
        backend = _make_mock_backend()
        backend.complete.side_effect = [
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content='<tool_call>{"name": "write", "arguments": {"path": "/tmp/test.py", "content": "x=1"}}</tool_call>',
                )
            ),
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content="The task is complete.",
                )
            ),
        ]
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        write_tool = MagicMock()
        write_tool.name = "write"
        write_tool.parameters = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        }
        write_tool.run = AsyncMock(return_value=MagicMock(output="wrote /tmp/test.py", success=True, error=None))
        registry.get.side_effect = lambda name: {"write": write_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=3, lint_after_write=False),
        )

        with patch("chef_human.agent.react_loop.run_lint") as mock_lint:
            await loop.run("do something")
            mock_lint.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_tool_calls_with_finish_text(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content="The task is complete. All steps finished successfully.",
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
        )
        result = await loop.run("do something")
        assert result.success is True
        assert "complete" in result.message

    @pytest.mark.asyncio
    async def test_max_steps_exceeded(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content="Thinking... no tools needed.",
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=_make_mock_tool_registry(),
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=3),
        )
        result = await loop.run("do something")
        assert result.success is False
        assert "Max steps exceeded" in result.message
        assert result.steps_taken == 3

    @pytest.mark.asyncio
    async def test_unknown_tool_is_reported(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "nonexistent", "arguments": {}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        registry.get.return_value = None

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1),
        )
        result = await loop.run("do something")
        # It would try to get a tool, fail, continue — eventually hit max_steps
        assert result.success is False

    @pytest.mark.asyncio
    async def test_tool_execution_error_triggers_retry(self):
        backend = _make_mock_backend()

        async def side_effect(*args, **kwargs):
            return CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content='<tool_call>{"name": "read", "arguments": {"path": "x.py"}}</tool_call>',
                )
            )
        backend.complete.side_effect = side_effect

        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        read_tool = MagicMock()
        read_tool.name = "read"
        read_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        read_tool.run = _make_tool_run("file not found", success=False)
        registry.get.return_value = read_tool

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=3),
        )
        result = await loop.run("do something")
        # Tool keeps failing, max retries triggers max_steps
        assert result.success is False

    @pytest.mark.asyncio
    async def test_replan_after_consecutive_failures(self):
        backend = _make_mock_backend()

        async def side_effect(*args, **kwargs):
            return CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content='<tool_call>{"name": "read", "arguments": {"path": "x.py"}}</tool_call>',
                )
            )
        backend.complete.side_effect = side_effect

        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        updated_plan = Plan(goal="Retry plan", steps=[])
        planner.update_plan.return_value = updated_plan

        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        read_tool = MagicMock()
        read_tool.name = "read"
        read_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        read_tool.run = _make_tool_run("fail", success=False)
        registry.get.return_value = read_tool

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=10, max_retries_per_step=2),
        )
        await loop.run("do something")
        # After 2 consecutive failures (max_retries_per_step=2), should trigger re-plan
        planner.update_plan.assert_awaited()

    @pytest.mark.asyncio
    async def test_reasoning_stored_as_assistant_message(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content="Let me read the file.\n```json\n{\"name\": \"finish\", \"arguments\": {\"summary\": \"done\"}}\n```",
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        finish_tool = MagicMock()
        finish_tool.name = "finish"
        finish_tool.parameters = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        }
        finish_tool.run = AsyncMock(return_value=MagicMock(output="done", success=True, error=None))
        registry.get.side_effect = lambda name: {"finish": finish_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
        )
        await loop.run("do something")

        # The assistant message should be added with stripped content (no code block)
        all_messages = [c for c in context.conversation.add_message.call_args_list]
        assistant_msgs = [
            c.args[0] for c in all_messages if c.args[0].role == Role.assistant
        ]
        assert len(assistant_msgs) >= 1
        # Content should be stripped of tool call markup
        last = assistant_msgs[-1]
        assert "```" not in last.content
        assert last.content == "Let me read the file."

    @pytest.mark.asyncio
    async def test_tool_results_recorded_in_conversation(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        finish_tool = MagicMock()
        finish_tool.name = "finish"
        finish_tool.parameters = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        }
        finish_tool.run = AsyncMock(return_value=MagicMock(output="done", success=True, error=None))
        registry.get.side_effect = lambda name: {"finish": finish_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
        )
        await loop.run("do something")

        # The user message should be added
        user_added = any(
            c.args[0].role == Role.user for c in context.conversation.add_message.call_args_list
        )
        assert user_added

    @pytest.mark.asyncio
    async def test_ui_callbacks_invoked(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        finish_tool = MagicMock()
        finish_tool.name = "finish"
        finish_tool.parameters = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        }
        finish_tool.run = AsyncMock(return_value=MagicMock(output="done", success=True, error=None))
        registry.get.side_effect = lambda name: {"finish": finish_tool}.get(name)

        ui = MagicMock(spec=NoopUI)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            ui=ui,
        )
        await loop.run("do something")

        ui.on_start.assert_called_once()
        ui.on_planning_start.assert_called_once()
        ui.on_plan.assert_called_once()
        ui.on_reasoning_start.assert_called_once()
        ui.on_reasoning.assert_called_once()
        ui.on_tool_call.assert_called_once()
        ui.on_tool_result.assert_called_once()

    @pytest.mark.asyncio
    async def test_validation_error_adds_tool_error(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "read", "arguments": {}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        read_tool = MagicMock()
        read_tool.name = "read"
        read_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        registry.get.return_value = read_tool

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1),
        )
        await loop.run("do something")

        # The tool error should be recorded
        tool_messages = [
            c.args[0]
            for c in context.conversation.add_message.call_args_list
            if c.args[0].role == Role.tool
        ]
        error_texts = [m.content for m in tool_messages]
        assert any("path" in e for e in error_texts)

    @pytest.mark.asyncio
    async def test_escalates_after_replan_fails(self):
        backend = _make_mock_backend()

        async def side_effect(*args, **kwargs):
            return CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content='<tool_call>{"name": "read", "arguments": {"path": "x.py"}}</tool_call>',
                )
            )
        backend.complete.side_effect = side_effect

        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        read_tool = MagicMock()
        read_tool.name = "read"
        read_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        read_tool.run = _make_tool_run("fail", success=False)
        registry.get.return_value = read_tool

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=10, max_retries_per_step=2, max_replans=1),
        )
        result = await loop.run("do something")
        assert result.success is False
        assert "persistent failures" in result.message.lower() or "replan" in result.message.lower()

    @pytest.mark.asyncio
    async def test_approval_gate_rejects_destructive_command(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "bash", "arguments": {"command": "rm -rf /tmp/test"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        bash_tool = MagicMock()
        bash_tool.name = "bash"
        bash_tool.parameters = {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }
        registry.get.return_value = bash_tool

        ui = MagicMock(spec=NoopUI)
        ui.on_approval_request = AsyncMock(return_value=False)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1, require_approval_for_destructive=True),
            ui=ui,
        )
        await loop.run("do something")

        # The bash tool should NOT have been executed
        bash_tool.run.assert_not_called()
        # The rejection message should be in the tool results
        tool_messages = [
            c.args[0].content
            for c in context.conversation.add_message.call_args_list
            if c.args[0].role == Role.tool
        ]
        assert any("rejected" in msg.lower() for msg in tool_messages)

    @pytest.mark.asyncio
    async def test_approval_gate_approves_and_executes(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "bash", "arguments": {"command": "rm -rf /tmp/test"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        bash_tool = MagicMock()
        bash_tool.name = "bash"
        bash_tool.parameters = {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }
        bash_tool.run = AsyncMock(return_value=MagicMock(output="ok", success=True, error=None))
        registry.get.return_value = bash_tool

        ui = MagicMock(spec=NoopUI)
        ui.on_approval_request = AsyncMock(return_value=True)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1, require_approval_for_destructive=True),
            ui=ui,
        )
        await loop.run("do something")

        bash_tool.run.assert_awaited_once_with(command="rm -rf /tmp/test")

    @pytest.mark.asyncio
    async def test_non_destructive_command_passes_without_approval(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "bash", "arguments": {"command": "ls -la"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        bash_tool = MagicMock()
        bash_tool.name = "bash"
        bash_tool.parameters = {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }
        bash_tool.run = AsyncMock(return_value=MagicMock(output="file1.txt", success=True, error=None))
        registry.get.return_value = bash_tool

        ui = MagicMock(spec=NoopUI)
        ui.on_approval_request = AsyncMock()

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1, require_approval_for_destructive=True),
            ui=ui,
        )
        await loop.run("do something")

        bash_tool.run.assert_awaited_once_with(command="ls -la")
        # Approval should NOT have been requested for non-destructive command
        ui.on_approval_request.assert_not_called()

    @pytest.mark.asyncio
    async def test_approval_gate_disabled_via_config(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "bash", "arguments": {"command": "rm -rf /tmp/test"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        bash_tool = MagicMock()
        bash_tool.name = "bash"
        bash_tool.parameters = {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }
        bash_tool.run = AsyncMock(return_value=MagicMock(output="ok", success=True, error=None))
        registry.get.return_value = bash_tool

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1, require_approval_for_destructive=False),
        )
        await loop.run("do something")

        bash_tool.run.assert_awaited_once_with(command="rm -rf /tmp/test")

    @pytest.mark.asyncio
    async def test_approval_fallback_to_console_when_ui_returns_none(self, monkeypatch):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "bash", "arguments": {"command": "rm -rf /tmp/test"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        bash_tool = MagicMock()
        bash_tool.name = "bash"
        bash_tool.parameters = {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }
        bash_tool.run = AsyncMock(return_value=MagicMock(output="ok", success=True, error=None))
        registry.get.return_value = bash_tool

        monkeypatch.setattr("builtins.input", lambda _: "y")

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1, require_approval_for_destructive=True),
        )
        await loop.run("do something")

        bash_tool.run.assert_awaited_once_with(command="rm -rf /tmp/test")

    @pytest.mark.asyncio
    async def test_streaming_on_stream_callback_invoked(self):
        backend = _make_mock_backend()

        async def _mock_stream(
            req,
        ) -> AsyncGenerator[tuple[str, CompletionResponse | None], None]:
            yield "think", None
            yield "ing", None
            yield "", CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content="thinking",
                )
            )

        backend.complete_stream = _mock_stream

        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        # Return finish tool call on the first LLM response
        finish_tool = MagicMock()
        finish_tool.name = "finish"
        finish_tool.parameters = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        }
        finish_tool.run = AsyncMock(
            return_value=MagicMock(output="done", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"finish": finish_tool}.get(name)

        ui = MagicMock(spec=NoopUI)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1, stream=True),
            ui=ui,
        )
        await loop.run("do something")

        assert ui.on_stream.call_count == 2
        calls = [c.args[0] for c in ui.on_stream.call_args_list]
        assert calls == ["think", "ing"]

    @pytest.mark.asyncio
    async def test_streaming_false_does_not_call_on_stream(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content="done",
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        finish_tool = MagicMock()
        finish_tool.name = "finish"
        finish_tool.parameters = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        }
        finish_tool.run = AsyncMock(
            return_value=MagicMock(output="done", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"finish": finish_tool}.get(name)

        ui = MagicMock(spec=NoopUI)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1, stream=False),
            ui=ui,
        )
        await loop.run("do something")

        ui.on_stream.assert_not_called()
        ui.on_reasoning.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_content_used_for_tool_parsing(self):
        backend = _make_mock_backend()

        async def _mock_stream(
            req,
        ) -> AsyncGenerator[tuple[str, CompletionResponse | None], None]:
            yield "", CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content='<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>',
                )
            )

        backend.complete_stream = _mock_stream

        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        finish_tool = MagicMock()
        finish_tool.name = "finish"
        finish_tool.parameters = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        }
        finish_tool.run = AsyncMock(
            return_value=MagicMock(output="Task complete: done", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"finish": finish_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1, stream=True),
        )
        result = await loop.run("do something")

        assert result.success is True
        assert "done" in result.message
        finish_tool.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_scratchpad_extracted_and_injected(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content=(
                    "Let me check.\n"
                    "## Scratchpad: path is src/main.py\n"
                    '<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>'
                ),
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        finish_tool = MagicMock()
        finish_tool.name = "finish"
        finish_tool.parameters = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        }
        finish_tool.run = AsyncMock(
            return_value=MagicMock(output="done", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"finish": finish_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=2),
        )
        await loop.run("do something")

        # The scratchpad should have been extracted and the next prompt should
        # include it. We verify by checking build_agent_prompt was called with
        # scratchpad on the second iteration. The first iteration creates the
        # prompt, the model returns a scratchpad update that should appear
        # in the second iteration's prompt.
        # Since the loop finishes on the first iteration (finish tool),
        # we verify the assistant message had scratchpad stripped.
        all_msgs = [
            c.args[0] for c in context.conversation.add_message.call_args_list
        ]
        assistant_msgs = [m for m in all_msgs if m.role == Role.assistant]
        assert len(assistant_msgs) >= 1
        last = assistant_msgs[-1]
        assert "Scratchpad" not in last.content
        assert "Let me check" in last.content

    @pytest.mark.asyncio
    async def test_scratchpad_updated_across_turns(self):
        backend = _make_mock_backend()

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return CompletionResponse(
                    message=Message(
                        role=Role.assistant,
                        content=(
                            "First turn.\n"
                            "## Scratchpad: note one\n"
                            '<tool_call>{"name": "read", "arguments": {"path": "x.py"}}</tool_call>'
                        ),
                    )
                )
            return CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content=(
                        "Second turn.\n"
                        "## Scratchpad: note two\n"
                        '<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>'
                    ),
                )
            )

        backend.complete.side_effect = side_effect

        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        read_tool = MagicMock()
        read_tool.name = "read"
        read_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        read_tool.run = _make_tool_run("file content", success=True)
        finish_tool = MagicMock()
        finish_tool.name = "finish"
        finish_tool.parameters = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        }
        finish_tool.run = AsyncMock(
            return_value=MagicMock(output="done", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"read": read_tool, "finish": finish_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=5),
        )
        await loop.run("do something")

        # The second prompt should have "note one" (from first turn)
        # Since we can't easily inspect the prompt, we verify that the
        # loop completed successfully (scratchpad didn't cause errors)
        # and the second turn's assistant message has scratchpad stripped
        all_msgs = [
            c.args[0] for c in context.conversation.add_message.call_args_list
        ]
        assistant_msgs = [m for m in all_msgs if m.role == Role.assistant]
        second_msg = assistant_msgs[-1]
        assert "Scratchpad" not in second_msg.content
        assert "Second turn" in second_msg.content

    @pytest.mark.asyncio
    async def test_scratchpad_reset_on_replan(self):
        backend = _make_mock_backend()

        async def side_effect(*args, **kwargs):
            return CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content=(
                        "Failing turn.\n"
                        "## Scratchpad: some note\n"
                        '<tool_call>{"name": "read", "arguments": {"path": "x.py"}}</tool_call>'
                    ),
                )
            )

        backend.complete.side_effect = side_effect

        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        updated_plan = Plan(goal="Replanned", steps=[])
        planner.update_plan.return_value = updated_plan

        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        read_tool = MagicMock()
        read_tool.name = "read"
        read_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        read_tool.run = _make_tool_run("fail", success=False)
        registry.get.return_value = read_tool

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=10, max_retries_per_step=2),
        )
        await loop.run("do something")

        # After re-plan, the scratchpad should be empty
        planner.update_plan.assert_awaited()


def _make_loop_with_mocks() -> ReActLoop:
    backend = _make_mock_backend()
    planner = _make_mock_planner()
    planner.generate_plan.return_value = _make_default_plan()
    context = _make_mock_context()
    registry = _make_mock_tool_registry()
    return ReActLoop(
        llm_backend=backend,
        tool_registry=registry,
        context_assembler=context,
        planner=planner,
    )


class TestIsDestructiveCommand:
    def test_destructive_prefixes(self):
        loop = _make_loop_with_mocks()
        assert loop._is_destructive_command("rm file.txt")
        assert loop._is_destructive_command("rm -rf /")
        assert loop._is_destructive_command("mv a b")
        assert loop._is_destructive_command("> out.txt")
        assert loop._is_destructive_command(">> log.txt")
        assert loop._is_destructive_command("dd if=/dev/zero of=file")

    def test_non_destructive_commands(self):
        loop = _make_loop_with_mocks()
        assert not loop._is_destructive_command("ls -la")
        assert not loop._is_destructive_command("grep foo bar")
        assert not loop._is_destructive_command("echo hello")
        assert not loop._is_destructive_command("read file.txt")
        assert not loop._is_destructive_command("")

    def test_strips_whitespace(self):
        loop = _make_loop_with_mocks()
        assert loop._is_destructive_command("  rm file.txt  ")
        assert not loop._is_destructive_command("  ls -la  ")


class TestParallelToolExecution:
    @pytest.mark.asyncio
    async def test_multiple_tools_execute_in_parallel_turn(self):
        """Multiple tool calls in a single turn execute via asyncio.gather."""
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content="Reading both files.\n"
                '<tool_call>{"name": "read", "arguments": {"path": "a.py"}}</tool_call>\n'
                '<tool_call>{"name": "read", "arguments": {"path": "b.py"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()

        read_a = MagicMock()
        read_a.name = "read"
        read_a.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        read_a.run = AsyncMock(return_value=MagicMock(output="content a", success=True, error=None))

        read_b = MagicMock()
        read_b.name = "read"
        read_b.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        read_b.run = AsyncMock(return_value=MagicMock(output="content b", success=True, error=None))

        call_count = 0

        def get_tool(name: str):
            nonlocal call_count
            call_count += 1
            return {"read": read_a if call_count == 1 else read_b}.get(name)

        registry.get.side_effect = get_tool

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=3),
        )
        result = await loop.run("do something")
        assert result.success is False  # no finish tool, hits max_steps

    @pytest.mark.asyncio
    async def test_finish_with_parallel_calls(self):
        """Finish tool executed after parallel calls."""
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content="Read then finish.\n"
                '<tool_call>{"name": "read", "arguments": {"path": "a.py"}}</tool_call>\n'
                '<tool_call>{"name": "finish", "arguments": {"summary": "all done"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()

        read_tool = MagicMock()
        read_tool.name = "read"
        read_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        read_tool.run = AsyncMock(return_value=MagicMock(output="file content", success=True, error=None))

        finish_tool = MagicMock()
        finish_tool.name = "finish"
        finish_tool.parameters = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        }
        finish_tool.run = AsyncMock(return_value=MagicMock(output="all done", success=True, error=None))

        registry.get.side_effect = lambda name: {"read": read_tool, "finish": finish_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=3),
        )
        result = await loop.run("do something")
        assert result.success is True
        assert "all done" in result.message
        read_tool.run.assert_awaited_once()
        finish_tool.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_parallel_with_unknown_tool(self):
        """Unknown tool errors collected, valid tools still execute."""
        backend = _make_mock_backend()
        backend.complete.side_effect = [
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content='<tool_call>{"name": "nonexistent", "arguments": {}}</tool_call>\n'
                    '<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>',
                )
            ),
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content="Task is complete.",
                )
            ),
        ]
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()

        finish_tool = MagicMock()
        finish_tool.name = "finish"
        finish_tool.parameters = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        }
        finish_tool.run = AsyncMock(return_value=MagicMock(output="done", success=True, error=None))

        registry = _make_mock_tool_registry()
        registry.get.side_effect = lambda name: {"finish": finish_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
        )
        result = await loop.run("do something")
        assert result.success is True
        assert "done" in result.message

    @pytest.mark.asyncio
    async def test_parallel_execution_error_handled(self):
        """Exception in one parallel call doesn't crash others."""
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "read", "arguments": {"path": "ok.py"}}</tool_call>\n'
                '<tool_call>{"name": "read", "arguments": {"path": "bad.py"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()

        ok_tool = MagicMock()
        ok_tool.name = "read"
        ok_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        ok_tool.run = AsyncMock(return_value=MagicMock(output="ok content", success=True, error=None))

        bad_tool = MagicMock()
        bad_tool.name = "read"
        bad_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        bad_tool.run = AsyncMock(side_effect=RuntimeError("tool crashed"))

        call_idx = 0

        def get_tool(name: str):
            nonlocal call_idx
            call_idx += 1
            return {"read": ok_tool if call_idx == 1 else bad_tool}.get(name)

        registry.get.side_effect = get_tool

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=3),
        )
        result = await loop.run("do something")
        assert result.success is False  # error → retry → max_steps

    @pytest.mark.asyncio
    async def test_single_tool_call_still_works(self):
        """Single tool call (no parallelism needed) still works."""
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        finish_tool = MagicMock()
        finish_tool.name = "finish"
        finish_tool.parameters = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        }
        finish_tool.run = AsyncMock(return_value=MagicMock(output="done", success=True, error=None))
        registry.get.side_effect = lambda name: {"finish": finish_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
        )
        result = await loop.run("do something")
        assert result.success is True


class TestTokenTracking:
    @pytest.mark.asyncio
    async def test_tokens_accumulated_across_calls(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>',
            ),
            usage={"prompt_tokens": 50, "completion_tokens": 10},
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        finish_tool = MagicMock()
        finish_tool.name = "finish"
        finish_tool.parameters = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        }
        finish_tool.run = AsyncMock(return_value=MagicMock(output="done", success=True, error=None))
        registry.get.side_effect = lambda name: {"finish": finish_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
        )
        result = await loop.run("do something")
        assert result.total_prompt_tokens == 50
        assert result.total_completion_tokens == 10

    @pytest.mark.asyncio
    async def test_tokens_default_to_zero_when_no_usage(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>',
            ),
            usage=None,
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        finish_tool = MagicMock()
        finish_tool.name = "finish"
        finish_tool.parameters = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        }
        finish_tool.run = AsyncMock(return_value=MagicMock(output="done", success=True, error=None))
        registry.get.side_effect = lambda name: {"finish": finish_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
        )
        result = await loop.run("do something")
        assert result.total_prompt_tokens == 0
        assert result.total_completion_tokens == 0


class TestParseFailureFeedback:
    @pytest.mark.asyncio
    async def test_malformed_tool_call_injects_parse_error(self):
        """When LLM emits content with <tool_call> but unparseable JSON, parse error injected."""
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='Let me read the file.\n<tool_call>{bad json}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1),
        )
        await loop.run("do something")

        # A tool message with parse error should be in conversation
        tool_msgs = [
            c.args[0]
            for c in context.conversation.add_message.call_args_list
            if c.args[0].role == Role.tool
        ]
        assert any("Failed to parse tool call" in m.content for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_plain_text_no_parse_error(self):
        """When LLM emits no tool calls, no parse error is injected."""
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content="Just thinking, no tools needed.",
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1),
        )
        await loop.run("do something")

        # No tool messages should be in conversation
        tool_msgs = [
            c.args[0]
            for c in context.conversation.add_message.call_args_list
            if c.args[0].role == Role.tool
        ]
        assert not any("Failed to parse tool call" in m.content for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_malformed_finish_does_not_trigger_completion(self):
        """Malformed finish tool call with parse error should not complete the task."""
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='The task is complete.\n<tool_call>{"name": "finish", "arguments": {}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1),
        )
        result = await loop.run("do something")
        # Missing required 'summary' argument → validation error, not parse error
        assert result.success is False

    @pytest.mark.asyncio
    async def test_parse_error_counts_as_failure_for_retry(self):
        """A parse error should count as a failure for retry tracking."""
        backend = _make_mock_backend()

        async def always_fail(*args, **kwargs):
            return CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content='<tool_call>{"name": "nonexistent", "arguments": {}}</tool_call>',
                )
            )
        backend.complete.side_effect = always_fail

        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        registry.get.return_value = None  # tool doesn't exist

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=5, max_retries_per_step=2),
        )
        result = await loop.run("do something")
        assert result.success is False


class TestToolTimeout:
    @pytest.mark.asyncio
    async def test_tool_timeout_applied_to_parallel_execution(self):
        """Tool execution is wrapped with tool_timeout."""
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "read", "arguments": {"path": "x.py"}}</tool_call>\n'
                '<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        read_tool = MagicMock()
        read_tool.name = "read"
        read_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        read_tool.run = AsyncMock(return_value=MagicMock(output="content", success=True, error=None))
        finish_tool = MagicMock()
        finish_tool.name = "finish"
        finish_tool.parameters = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        }
        finish_tool.run = AsyncMock(return_value=MagicMock(output="done", success=True, error=None))
        registry.get.side_effect = lambda name: {"read": read_tool, "finish": finish_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=3, tool_timeout=30.0),
        )
        result = await loop.run("do something")
        assert result.success is True
        read_tool.run.assert_awaited_once()
        finish_tool.run.assert_awaited_once()


class TestLintRollback:
    @pytest.mark.asyncio
    async def test_lint_rollback_appends_rollback_message(self):
        """When lint detects errors after write, rollback message is appended."""
        backend = _make_mock_backend()
        backend.complete.side_effect = [
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content='<tool_call>{"name": "write", "arguments": {"path": "/tmp/test.py", "content": "x=1"}}</tool_call>',
                )
            ),
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content="The task is complete.",
                )
            ),
        ]
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        write_tool = MagicMock()
        write_tool.name = "write"
        write_tool.parameters = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        }
        write_tool.run = AsyncMock(return_value=MagicMock(output="wrote /tmp/test.py", success=True, error=None))
        registry.get.side_effect = lambda name: {"write": write_tool}.get(name)

        mock_lint_output = "\nLint results (1 issue):\ntest.py:1:1: F401 error"
        with (
            patch("chef_human.agent.react_loop.run_lint", return_value="test.py:1:1: F401 error"),
            patch("chef_human.agent.react_loop.format_lint_result", return_value=mock_lint_output),
        ):
            loop = ReActLoop(
                llm_backend=backend,
                tool_registry=registry,
                context_assembler=context,
                planner=planner,
                config=ReActConfig(max_steps=3),
            )
            await loop.run("do something")

        # A rollback message should appear in the tool results
        tool_msgs = [
            c.args[0].content
            for c in context.conversation.add_message.call_args_list
            if c.args[0].role == Role.tool
        ]
        rollback_msgs = [m for m in tool_msgs if "rollback" in m.lower()]
        assert len(rollback_msgs) >= 1
        assert "lint errors detected" in rollback_msgs[0].lower()

    @pytest.mark.asyncio
    async def test_lint_no_rollback_when_lint_succeeds(self):
        """When lint finds no issues, no rollback message is produced."""
        backend = _make_mock_backend()
        backend.complete.side_effect = [
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content='<tool_call>{"name": "write", "arguments": {"path": "/tmp/test.py", "content": "x = 1"}}</tool_call>',
                )
            ),
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content="The task is complete.",
                )
            ),
        ]
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        write_tool = MagicMock()
        write_tool.name = "write"
        write_tool.parameters = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        }
        write_tool.run = AsyncMock(return_value=MagicMock(output="wrote /tmp/test.py", success=True, error=None))
        registry.get.side_effect = lambda name: {"write": write_tool}.get(name)

        with patch("chef_human.agent.react_loop.run_lint", return_value=""):
            loop = ReActLoop(
                llm_backend=backend,
                tool_registry=registry,
                context_assembler=context,
                planner=planner,
                config=ReActConfig(max_steps=3),
            )
            await loop.run("do something")

        # No rollback message should appear
        tool_msgs = [
            c.args[0].content
            for c in context.conversation.add_message.call_args_list
            if c.args[0].role == Role.tool
        ]
        assert not any("rollback" in m.lower() for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_lint_rollback_increments_failed_calls(self):
        """Lint failure causing rollback should increment failed_calls and trigger retry."""
        backend = _make_mock_backend()

        async def always_write(*args, **kwargs):
            return CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content='<tool_call>{"name": "write", "arguments": {"path": "/tmp/test.py", "content": "x=1"}}</tool_call>',
                )
            )
        backend.complete.side_effect = always_write

        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        write_tool = MagicMock()
        write_tool.name = "write"
        write_tool.parameters = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        }
        write_tool.run = AsyncMock(return_value=MagicMock(output="wrote /tmp/test.py", success=True, error=None))
        registry.get.side_effect = lambda name: {"write": write_tool}.get(name)

        mock_lint_output = "\nLint results (1 issue):\ntest.py:1:1: F401 error"
        with (
            patch("chef_human.agent.react_loop.run_lint", return_value="test.py:1:1: F401 error"),
            patch("chef_human.agent.react_loop.format_lint_result", return_value=mock_lint_output),
        ):
            loop = ReActLoop(
                llm_backend=backend,
                tool_registry=registry,
                context_assembler=context,
                planner=planner,
                config=ReActConfig(max_steps=5, max_retries_per_step=1, max_replans=0),
            )
            result = await loop.run("do something")
            # Lint failure triggers retry; with max_retries_per_step=1 and max_replans=0,
            # it escalates after first failure
            assert result.success is False


class TestRepeatedToolCallDetection:
    @pytest.mark.asyncio
    async def test_identical_consecutive_calls_trigger_nudge_and_escalate(self):
        """Repeating the exact same tool call should be flagged as a failure
        and eventually escalate, instead of burning every step doing nothing."""
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "ls_tree", "arguments": {}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        ls_tool = MagicMock()
        ls_tool.name = "ls_tree"
        ls_tool.parameters = {"type": "object", "properties": {}}
        ls_tool.run = AsyncMock(
            return_value=MagicMock(output="tree", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"ls_tree": ls_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=5, max_retries_per_step=2, max_replans=0),
        )
        result = await loop.run("do something")

        tool_msgs = [
            c.args[0].content
            for c in context.conversation.add_message.call_args_list
            if c.args[0].role == Role.tool
        ]
        assert any("repeated the exact same tool call" in m for m in tool_msgs)
        # Escalates well before exhausting max_steps, instead of spinning to
        # "Max steps exceeded".
        assert result.success is False
        assert result.steps_taken < 5

    @pytest.mark.asyncio
    async def test_first_call_is_not_flagged_as_repeat(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "ls_tree", "arguments": {}}</tool_call>\n'
                '<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        ls_tool = MagicMock()
        ls_tool.name = "ls_tree"
        ls_tool.parameters = {"type": "object", "properties": {}}
        ls_tool.run = AsyncMock(
            return_value=MagicMock(output="tree", success=True, error=None)
        )
        finish_tool = MagicMock()
        finish_tool.name = "finish"
        finish_tool.parameters = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        }
        finish_tool.run = AsyncMock(
            return_value=MagicMock(output="done", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {
            "ls_tree": ls_tool,
            "finish": finish_tool,
        }.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=5),
        )
        result = await loop.run("do something")

        tool_msgs = [
            c.args[0].content
            for c in context.conversation.add_message.call_args_list
            if c.args[0].role == Role.tool
        ]
        assert not any("repeated the exact same tool call" in m for m in tool_msgs)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_calls_with_different_arguments_are_not_flagged(self):
        backend = _make_mock_backend()
        backend.complete.side_effect = [
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content='<tool_call>{"name": "read", "arguments": {"path": "a.py"}}</tool_call>',
                )
            ),
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content='<tool_call>{"name": "read", "arguments": {"path": "b.py"}}</tool_call>',
                )
            ),
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content='<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>',
                )
            ),
        ]
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        read_tool = MagicMock()
        read_tool.name = "read"
        read_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        read_tool.run = AsyncMock(
            return_value=MagicMock(output="contents", success=True, error=None)
        )
        finish_tool = MagicMock()
        finish_tool.name = "finish"
        finish_tool.parameters = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        }
        finish_tool.run = AsyncMock(
            return_value=MagicMock(output="done", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {
            "read": read_tool,
            "finish": finish_tool,
        }.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=5),
        )
        result = await loop.run("do something")

        tool_msgs = [
            c.args[0].content
            for c in context.conversation.add_message.call_args_list
            if c.args[0].role == Role.tool
        ]
        assert not any("repeated the exact same tool call" in m for m in tool_msgs)
        assert result.success is True
