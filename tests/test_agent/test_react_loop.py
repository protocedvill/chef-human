from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

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

    def test_custom(self):
        config = ReActConfig(max_steps=5, temperature=0.7)
        assert config.max_steps == 5
        assert config.temperature == 0.7


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

        planner.generate_plan.assert_awaited_once()
        assert result.steps_taken == 0  # no reasoning steps needed

    @pytest.mark.asyncio
    async def test_finish_tool_ends_loop(self):
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
        assert "done" in result.message

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
