from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chef_human.agent.planner import Plan, PlanNode, Planner, StepStatus, StepVerdict
from chef_human.agent.retry import RetryManager
from chef_human.agent.prompts import build_agent_prompt
from chef_human.agent.react_loop import (
    AgentResult,
    EscalationRecord,
    ReActConfig,
    ReActLoop,
    StepEvidence,
    _step_file_contents,
    _step_python_syntax_errors,
)
from chef_human.agent.workspace import WorkspaceManager
from chef_human.llm.backend import (
    CompletionResponse,
    LLMBackend,
    Message,
    Role,
    ToolDefinition,
)
from chef_human.tools.registry import ToolRegistry
from chef_human.ui.protocol import NoopUI, PlanReviewAction


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


class _FakeResolvedPath:
    """Stand-in for WorkspaceManager.resolve()'s return value in tests.
    Defaults to reporting as non-existent so the read-before-edit guard
    treats every path as "new file, nothing to read" unless a test
    explicitly wants to exercise the guard (see TestReadBeforeEditGuard),
    which passes exists=True."""

    def __init__(self, path: object, exists: bool = False) -> None:
        self._path = str(path)
        self._exists = exists

    def exists(self) -> bool:
        return self._exists

    def read_text(self) -> str:
        raise OSError("test path has no backing file")

    def __str__(self) -> str:
        return self._path

    def __eq__(self, other: object) -> bool:
        return str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))


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
    context.workspace = MagicMock()
    context.workspace.resolve = MagicMock(side_effect=lambda p: _FakeResolvedPath(p))
    return context


def _make_mock_planner() -> MagicMock:
    planner = MagicMock(spec=Planner)
    planner.generate_plan = AsyncMock()
    # Default to an empty replanned Plan (a real Plan instance, not an
    # unconfigured mock) so tests that trigger REPLAN without explicitly
    # setting update_plan.return_value still get something build_agent_prompt
    # can safely call plan.current_leaf() / iterate plan.steps on.
    planner.update_plan = AsyncMock(return_value=Plan(goal="Replanned", steps=[]))
    # Subtree-scoped replan (used by REPLAN handling instead of update_plan
    # whenever a specific failing node can be identified) -- mutates the
    # target node's children in place and returns None, matching the real
    # Planner.replan_subtree contract.
    planner.replan_subtree = AsyncMock(return_value=None)
    # Default to "complete" so existing tests that don't care about step
    # verification keep their old behavior (any non-failing turn advances
    # the plan). Tests that specifically exercise verification override this.
    planner.verify_step = AsyncMock(return_value=(StepVerdict.complete, "done"))
    # Same default-to-complete rationale as verify_step above: only tests
    # exercising rollup verification specifically override this.
    planner.verify_rollup = AsyncMock(return_value=(StepVerdict.complete, "covered"))
    planner.format_plan_for_prompt = MagicMock(
        side_effect=lambda p: f"Plan: {p.goal}"
    )
    return planner


def _make_default_plan() -> Plan:
    return Plan(
        goal="Test task",
        steps=[
            PlanNode(index=1, description="Step one", status=StepStatus.pending),
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
        plan = Plan(goal="Test", steps=[PlanNode(index=1, description="Do something")])
        tool_defs: list[ToolDefinition] = []
        prompt = build_agent_prompt(plan=plan, tool_defs=tool_defs)
        assert "Step 1" in prompt
        assert "Do something" in prompt

    def test_with_both(self):
        plan = Plan(goal="Test", steps=[PlanNode(index=1, description="Do something")])
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
        assert config.max_retries_per_step == 5
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
        planner.replan_subtree.assert_awaited()

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
            config=ReActConfig(require_plan_complete_to_finish=False),
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
        planner.update_plan.return_value = Plan(goal="Replanned", steps=[])
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
    async def test_streaming_without_final_response_fails_clearly(self):
        backend = _make_mock_backend()

        async def _incomplete_stream(
            req,
        ) -> AsyncGenerator[tuple[str, CompletionResponse | None], None]:
            yield "partial", None

        backend.complete_stream = _incomplete_stream
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        ui = MagicMock(spec=NoopUI)
        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=_make_mock_tool_registry(),
            context_assembler=_make_mock_context(),
            planner=planner,
            config=ReActConfig(max_steps=1, stream=True),
            ui=ui,
        )

        with pytest.raises(RuntimeError, match="stream ended without a final response"):
            await loop.run("do something")

        ui.on_stream.assert_called_once_with("partial")
        ui.on_llm_end.assert_called_once()

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
            config=ReActConfig(max_steps=1, stream=True, require_plan_complete_to_finish=False),
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
    async def test_scratchpad_persists_across_replan(self):
        """The scratchpad is the agent's accumulated working memory -- it
        must survive a re-plan instead of being wiped, since a failed
        attempt is exactly when that context matters most for the retry."""
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
                            "Failing turn.\n"
                            "## Scratchpad: [decision] use SQLite\n"
                            '<tool_call>{"name": "read", "arguments": {"path": "x.py"}}</tool_call>'
                        ),
                    )
                )
            return CompletionResponse(
                message=Message(role=Role.assistant, content="Just thinking, no tools."),
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
            config=ReActConfig(max_steps=3, max_retries_per_step=1, max_replans=1),
        )

        with patch(
            "chef_human.agent.react_loop.build_agent_prompt", wraps=build_agent_prompt
        ) as mock_build:
            await loop.run("do something")

        planner.replan_subtree.assert_awaited()
        scratchpad_args = [c.kwargs["scratchpad"] for c in mock_build.call_args_list]
        assert len(scratchpad_args) >= 2
        # First prompt is built before the model has written anything.
        assert "use SQLite" not in scratchpad_args[0]
        # Every prompt built after the note (including post-replan) still has it.
        assert all("use SQLite" in s for s in scratchpad_args[1:])


class TestStepVerification:
    @pytest.mark.asyncio
    async def test_reasoning_only_verifier_exception_is_bounded_failure(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(role=Role.assistant, content="I believe this is done.")
        )
        planner = _make_mock_planner()
        plan = _make_default_plan()
        planner.generate_plan.return_value = plan
        planner.verify_step = AsyncMock(side_effect=ConnectionError("planner offline"))

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=_make_mock_tool_registry(),
            context_assembler=_make_mock_context(),
            planner=planner,
            config=ReActConfig(
                max_steps=10,
                max_retries_per_step=1,
                max_replans=0,
            ),
        )
        result = await loop.run("do something")

        # Ticket 06: escalation marks the node failed and continues (same
        # default as headless) instead of terminating the run on the spot --
        # with nothing left to work on, the loop idles out to max_steps
        # rather than stopping the instant the one node escalates.
        assert not result.success
        assert plan.steps[0].status == StepStatus.failed
        assert len(result.escalations) == 1
        assert result.escalations[0].node_id == plan.steps[0].node_id

    @pytest.mark.asyncio
    async def test_reasoning_only_verifier_rejections_accumulate(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(role=Role.assistant, content="I think this is done.")
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        planner.verify_step = AsyncMock(
            return_value=(StepVerdict.not_complete, "no evidence")
        )

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=_make_mock_tool_registry(),
            context_assembler=_make_mock_context(),
            planner=planner,
            config=ReActConfig(max_steps=2, max_retries_per_step=2),
        )
        await loop.run("do something")

        planner.replan_subtree.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_successful_tools_do_not_reset_verification_failures(self):
        backend = _make_mock_backend()
        backend.complete.side_effect = [
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content=(
                        '<tool_call>{"name": "bash", "arguments": '
                        '{"command": "python check_one.py"}}</tool_call>'
                    ),
                )
            ),
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content=(
                        '<tool_call>{"name": "bash", "arguments": '
                        '{"command": "python check_two.py"}}</tool_call>'
                    ),
                )
            ),
        ]
        planner = _make_mock_planner()
        plan = _make_default_plan()
        planner.generate_plan.return_value = plan
        planner.verify_step = AsyncMock(
            return_value=(StepVerdict.not_complete, "no useful evidence")
        )
        registry = _make_mock_tool_registry()
        bash_tool = MagicMock()
        bash_tool.name = "bash"
        bash_tool.parameters = {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }
        bash_tool.run = AsyncMock(
            return_value=MagicMock(output="ok", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"bash": bash_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=_make_mock_context(),
            planner=planner,
            config=ReActConfig(
                max_steps=2,
                max_retries_per_step=2,
                lint_after_write=False,
            ),
        )
        await loop.run("do something")

        planner.replan_subtree.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_partial_verdict_does_not_advance_step(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "write", "arguments": {"path": "a.py", "content": "x"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        plan = _make_default_plan()
        planner.generate_plan.return_value = plan
        planner.verify_step = AsyncMock(
            return_value=(StepVerdict.partial, "only wrote a stub")
        )
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        write_tool = MagicMock()
        write_tool.name = "write"
        write_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        }
        write_tool.run = AsyncMock(
            return_value=MagicMock(output="wrote a.py", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"write": write_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=2, lint_after_write=False),
        )
        await loop.run("do something")

        assert plan.steps[0].status == StepStatus.pending
        tool_msgs = [
            c.args[0].content
            for c in context.conversation.add_message.call_args_list
            if c.args[0].role == Role.tool
        ]
        assert any("not fully done yet" in m for m in tool_msgs)
        assert any("only wrote a stub" in m for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_not_complete_verdict_does_not_advance_step(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "write", "arguments": {"path": "a.py", "content": "x"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        plan = _make_default_plan()
        planner.generate_plan.return_value = plan
        planner.verify_step = AsyncMock(
            return_value=(StepVerdict.not_complete, "no evidence of progress")
        )
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        write_tool = MagicMock()
        write_tool.name = "write"
        write_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        }
        write_tool.run = AsyncMock(
            return_value=MagicMock(output="wrote a.py", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"write": write_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=2, lint_after_write=False),
        )
        await loop.run("do something")

        assert plan.steps[0].status == StepStatus.pending

    @pytest.mark.asyncio
    async def test_complete_verdict_advances_step(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "write", "arguments": {"path": "a.py", "content": "x"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        plan = _make_default_plan()
        planner.generate_plan.return_value = plan
        planner.verify_step = AsyncMock(
            return_value=(StepVerdict.complete, "file fully written")
        )
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        write_tool = MagicMock()
        write_tool.name = "write"
        write_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        }
        write_tool.run = AsyncMock(
            return_value=MagicMock(output="wrote a.py", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"write": write_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=2, lint_after_write=False),
        )
        await loop.run("do something")

        assert plan.steps[0].status == StepStatus.completed

    @pytest.mark.asyncio
    async def test_successful_bash_auto_completes_execution_step(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "bash", "arguments": {"command": "python hello.py"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        plan = Plan(goal="g", steps=[
            PlanNode(
                index=1,
                description="Run hello.py using Python and verify the output",
                status=StepStatus.pending,
            ),
        ])
        planner.generate_plan.return_value = plan
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        bash_tool = MagicMock()
        bash_tool.name = "bash"
        bash_tool.parameters = {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }
        bash_tool.run = AsyncMock(
            return_value=MagicMock(output="Hello, world!\n", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"bash": bash_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1, lint_after_write=False),
        )
        await loop.run("do something")

        planner.verify_step.assert_not_awaited()
        assert plan.steps[0].status == StepStatus.completed

    @pytest.mark.asyncio
    async def test_no_pending_steps_skips_verification(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "read", "arguments": {"path": "a.py"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        plan = Plan(
            goal="g", steps=[PlanNode(index=1, description="s", status=StepStatus.completed)]
        )
        planner.generate_plan.return_value = plan
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
        registry.get.side_effect = lambda name: {"read": read_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1),
        )
        await loop.run("do something")

        planner.verify_step.assert_not_awaited()


class TestPlanningPreflight:
    def _context_rooted_at(self, tmp_path):
        context = _make_mock_context()
        context.workspace.resolve = MagicMock(side_effect=lambda p: tmp_path / p)
        return context

    @pytest.mark.asyncio
    async def test_plan_task_passes_file_existence_facts_to_planner(self, tmp_path):
        (tmp_path / "SPEC.md").write_text(
            "Implement `slugify(value: str) -> str` in `slugify.py`.\n"
        )
        (tmp_path / "test_slugify.py").write_text("from slugify import slugify\n")

        planner = _make_mock_planner()
        planner.generate_plan.return_value = Plan(goal="g", steps=[])
        context = self._context_rooted_at(tmp_path)
        registry = _make_mock_tool_registry()

        loop = ReActLoop(
            llm_backend=_make_mock_backend(),
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(),
        )

        await loop._plan_task(
            "Read SPEC.md and test_slugify.py, then implement slugify.py. "
            "Do not modify the tests."
        )

        planner.generate_plan.assert_awaited_once()
        call = planner.generate_plan.await_args
        assert call is not None
        assert call.kwargs["planning_facts"] == {
            "SPEC.md": True,
            "test_slugify.py": True,
            "slugify.py": False,
        }
        assert "### Planning Facts" in call.kwargs["repo_context"]
        assert "- slugify.py: missing" in call.kwargs["repo_context"]


class TestObjectiveFileVerification:
    """_verify_and_mark_step's objective-file-creation bypass: for a step
    that names a specific file and reads like "create/write X", check the
    file directly on disk instead of relying on an LLM reading tool-output
    wording. Regression coverage for the bug where `edit`'s "Replaced
    entire contents of X" message made the LLM judge conclude a "create a
    new file named X" step wasn't done, even though X's content was
    already correct -- causing the agent to alternate write/edit forever."""

    def _context_rooted_at(self, tmp_path):
        context = _make_mock_context()
        context.workspace.resolve = MagicMock(side_effect=lambda p: tmp_path / p)
        return context

    @pytest.mark.asyncio
    async def test_creation_step_auto_completes_when_file_exists_on_disk(self, tmp_path):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "write", "arguments": {"path": "hello_world.py", "content": "print(1)"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        plan = Plan(goal="g", steps=[
            PlanNode(index=1, description="Create a new file named 'hello_world.py'", status=StepStatus.pending),
        ])
        planner.generate_plan.return_value = plan
        context = self._context_rooted_at(tmp_path)
        registry = _make_mock_tool_registry()
        write_tool = MagicMock()
        write_tool.name = "write"
        write_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        }

        async def do_write(path: str, content: str):
            (tmp_path / path).write_text(content)
            return MagicMock(output=f"Wrote {content.count(chr(10)) + 1} lines to {path}", success=True, error=None)

        write_tool.run = AsyncMock(side_effect=do_write)
        registry.get.side_effect = lambda name: {"write": write_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1, lint_after_write=False),
        )
        await loop.run("do something")

        assert plan.steps[0].status == StepStatus.completed
        planner.verify_step.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_creation_step_completes_from_plan_time_missing_to_now_existing(self, tmp_path):
        planner = _make_mock_planner()
        plan = Plan(goal="g", steps=[
            PlanNode(index=1, description="Create a new file named slugify.py", status=StepStatus.pending),
        ])
        context = self._context_rooted_at(tmp_path)
        loop = ReActLoop(
            llm_backend=_make_mock_backend(),
            tool_registry=_make_mock_tool_registry(),
            context_assembler=context,
            planner=planner,
            config=ReActConfig(),
        )
        loop._planning_facts = {"slugify.py": False}
        (tmp_path / "slugify.py").write_text("def slugify(value):\n    return value\n")

        feedback = await loop._verify_and_mark_step(
            plan,
            evidence="The file now exists.",
            has_tool_evidence=False,
        )

        assert feedback is None
        assert plan.steps[0].status == StepStatus.completed
        planner.verify_step.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_mutation_step_reuses_prior_turn_write_evidence(self, tmp_path):
        planner = _make_mock_planner()
        planner.verify_step = AsyncMock(side_effect=[
            (StepVerdict.partial, "tests have not been run yet"),
            (StepVerdict.complete, "implementation verified"),
        ])
        plan = Plan(goal="g", steps=[
            PlanNode(
                index=1,
                description="Implement the slugify function in slugify.py according to the specifications.",
                status=StepStatus.pending,
            ),
        ])
        context = self._context_rooted_at(tmp_path)
        loop = ReActLoop(
            llm_backend=_make_mock_backend(),
            tool_registry=_make_mock_tool_registry(),
            context_assembler=context,
            planner=planner,
            config=ReActConfig(),
        )
        slugify_path = str(tmp_path / "slugify.py")
        (tmp_path / "slugify.py").write_text("def slugify(value):\n    return value\n")

        first_feedback = await loop._verify_and_mark_step(
            plan,
            evidence="Wrote slugify.py",
            files_written_this_turn={slugify_path: False},
        )
        second_feedback = await loop._verify_and_mark_step(
            plan,
            evidence="....\nOK",
            successful_commands_this_turn=["python test_slugify.py"],
        )

        assert first_feedback is not None
        assert second_feedback is None
        assert plan.steps[0].status == StepStatus.completed
        assert planner.verify_step.await_count == 2
        second_evidence = planner.verify_step.await_args_list[1].args[2]
        assert "slugify.py: exists=True" in second_evidence
        assert "created_during_step=True" in second_evidence
        assert "python test_slugify.py" in second_evidence

    @pytest.mark.asyncio
    async def test_bypass_does_not_fire_for_a_different_file(self, tmp_path):
        """The step names hello_world.py, but the file actually touched
        this turn is something else -- must not auto-complete."""
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "write", "arguments": {"path": "notes.txt", "content": "unrelated"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        plan = Plan(goal="g", steps=[
            PlanNode(index=1, description="Create a new file named 'hello_world.py'", status=StepStatus.pending),
        ])
        planner.generate_plan.return_value = plan
        context = self._context_rooted_at(tmp_path)
        registry = _make_mock_tool_registry()
        write_tool = MagicMock()
        write_tool.name = "write"
        write_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        }

        async def do_write(path: str, content: str):
            (tmp_path / path).write_text(content)
            return MagicMock(output=f"Wrote 1 lines to {path}", success=True, error=None)

        write_tool.run = AsyncMock(side_effect=do_write)
        registry.get.side_effect = lambda name: {"write": write_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1, lint_after_write=False),
        )
        await loop.run("do something")

        planner.verify_step.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_bypass_does_not_fire_without_a_creation_verb(self, tmp_path):
        """Mentioning a filename isn't enough -- the step must also read
        like a creation/write step, otherwise it needs real judgment (e.g.
        "test that hello_world.py runs correctly" isn't satisfied just
        because the file exists)."""
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "write", "arguments": {"path": "hello_world.py", "content": "print(1)"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        plan = Plan(goal="g", steps=[
            PlanNode(index=1, description="Test that hello_world.py runs correctly", status=StepStatus.pending),
        ])
        planner.generate_plan.return_value = plan
        context = self._context_rooted_at(tmp_path)
        registry = _make_mock_tool_registry()
        write_tool = MagicMock()
        write_tool.name = "write"
        write_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        }

        async def do_write(path: str, content: str):
            (tmp_path / path).write_text(content)
            return MagicMock(output=f"Wrote 1 lines to {path}", success=True, error=None)

        write_tool.run = AsyncMock(side_effect=do_write)
        registry.get.side_effect = lambda name: {"write": write_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1, lint_after_write=False),
        )
        await loop.run("do something")

        planner.verify_step.assert_not_awaited()
        tool_msgs = [
            c.args[0].content
            for c in context.conversation.add_message.call_args_list
            if c.args[0].role == Role.tool
        ]
        assert any("still needs real execution evidence" in m for m in tool_msgs)
        assert any("`bash`" in m for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_objective_facts_included_in_evidence_for_llm_judge(self, tmp_path):
        """When the bypass doesn't apply, the evidence passed to
        verify_step must still include objectively-checked facts about
        files touched this turn, not just raw tool-output text."""
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "write", "arguments": {"path": "hello_world.py", "content": "print(1)\\nprint(2)"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        plan = Plan(goal="g", steps=[
            PlanNode(index=1, description="Implement proper error handling", status=StepStatus.pending),
        ])
        planner.generate_plan.return_value = plan
        context = self._context_rooted_at(tmp_path)
        registry = _make_mock_tool_registry()
        write_tool = MagicMock()
        write_tool.name = "write"
        write_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        }

        async def do_write(path: str, content: str):
            (tmp_path / path).write_text(content)
            return MagicMock(output=f"Wrote 2 lines to {path}", success=True, error=None)

        write_tool.run = AsyncMock(side_effect=do_write)
        registry.get.side_effect = lambda name: {"write": write_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1, lint_after_write=False),
        )
        await loop.run("do something")

        planner.verify_step.assert_awaited_once()
        _, _, evidence_arg = planner.verify_step.await_args.args
        assert "Objective facts" in evidence_arg
        assert "hello_world.py: exists=True, lines=2, created_during_step=True" in evidence_arg


class TestRollupVerification:
    """A branch node, once all its children are marked complete, must pass
    its own rollup verification call before the branch itself is marked
    complete -- a decomposition can leave the sub-goal uncovered even
    though every child step individually succeeded."""

    def _context_rooted_at(self, tmp_path):
        context = _make_mock_context()
        context.workspace.resolve = MagicMock(side_effect=lambda p: tmp_path / p)
        return context

    def _branch_plan(self):
        child_a = PlanNode(index=1, description="Write utils.py", status=StepStatus.completed)
        child_b = PlanNode(index=2, description="Verify main.py runs", status=StepStatus.pending)
        branch = PlanNode(description="Implement the module")
        branch.set_children([child_a, child_b])
        plan = Plan(goal="Build the module", steps=[branch])
        return plan, branch, child_a, child_b

    @pytest.mark.asyncio
    async def test_rollup_runs_only_once_all_children_complete(self, tmp_path):
        planner = _make_mock_planner()
        plan, branch, child_a, child_b = self._branch_plan()
        context = self._context_rooted_at(tmp_path)
        loop = ReActLoop(
            llm_backend=_make_mock_backend(),
            tool_registry=_make_mock_tool_registry(),
            context_assembler=context,
            planner=planner,
            config=ReActConfig(),
        )

        # child_b still pending: completing it should trigger the rollup,
        # since it's the last of branch's children.
        feedback = await loop._verify_and_mark_step(
            plan,
            evidence="ran python main.py successfully",
            successful_commands_this_turn=["python main.py"],
            step_override=child_b,
        )

        assert feedback is None
        assert child_b.status == StepStatus.completed
        planner.verify_rollup.assert_awaited_once()
        assert branch.status == StepStatus.completed
        assert plan.is_complete()

    @pytest.mark.asyncio
    async def test_rollup_rejection_leaves_branch_incomplete_with_feedback(self, tmp_path):
        planner = _make_mock_planner()
        planner.verify_rollup = AsyncMock(
            return_value=(StepVerdict.not_complete, "main.py never calls utils.slugify")
        )
        plan, branch, child_a, child_b = self._branch_plan()
        context = self._context_rooted_at(tmp_path)
        loop = ReActLoop(
            llm_backend=_make_mock_backend(),
            tool_registry=_make_mock_tool_registry(),
            context_assembler=context,
            planner=planner,
            config=ReActConfig(),
        )

        feedback = await loop._verify_and_mark_step(
            plan,
            evidence="ran python main.py successfully",
            successful_commands_this_turn=["python main.py"],
            step_override=child_b,
        )

        # child_b itself still verified complete on its own terms --
        # rollup rejection is about the branch's sub-goal, not the leaf.
        assert child_b.status == StepStatus.completed
        assert branch.status != StepStatus.completed
        assert feedback is not None
        assert "Implement the module" in feedback
        assert "main.py never calls utils.slugify" in feedback
        assert not plan.is_complete()

    @pytest.mark.asyncio
    async def test_rollup_receives_children_verdicts_as_supporting_context(self, tmp_path):
        planner = _make_mock_planner()
        plan, branch, child_a, child_b = self._branch_plan()
        child_a.last_verdict_reason = "created utils.py with slugify()"
        context = self._context_rooted_at(tmp_path)
        loop = ReActLoop(
            llm_backend=_make_mock_backend(),
            tool_registry=_make_mock_tool_registry(),
            context_assembler=context,
            planner=planner,
            config=ReActConfig(),
        )

        await loop._verify_and_mark_step(
            plan,
            evidence="ran python main.py successfully",
            successful_commands_this_turn=["python main.py"],
            step_override=child_b,
        )

        planner.verify_rollup.assert_awaited_once()
        call = planner.verify_rollup.await_args
        assert call.args[1] is branch
        children_summary = call.kwargs["children_summary"]
        assert "Write utils.py: completed -- created utils.py with slugify()" in children_summary
        assert "Verify main.py runs: completed" in children_summary

    @pytest.mark.asyncio
    async def test_rollup_evidence_is_current_disk_contents_not_children_say_so(self, tmp_path):
        planner = _make_mock_planner()
        child_a = PlanNode(index=1, description="Write module.py", status=StepStatus.completed)
        branch = PlanNode(description="Implement module.py correctly")
        branch.set_children([child_a])
        plan = Plan(goal="Build the module", steps=[branch])
        context = self._context_rooted_at(tmp_path)
        (tmp_path / "module.py").write_text("def real_impl():\n    return 42\n")
        loop = ReActLoop(
            llm_backend=_make_mock_backend(),
            tool_registry=_make_mock_tool_registry(),
            context_assembler=context,
            planner=planner,
            config=ReActConfig(),
        )

        await loop._verify_and_mark_step(
            plan,
            evidence="wrote module.py",
            files_written_this_turn={str(tmp_path / "module.py"): False},
            step_override=child_a,
        )

        planner.verify_rollup.assert_awaited_once()
        evidence_arg = planner.verify_rollup.await_args.args[2]
        assert "def real_impl():" in evidence_arg

    @pytest.mark.asyncio
    async def test_no_rollup_call_while_siblings_still_pending(self, tmp_path):
        planner = _make_mock_planner()
        child_a = PlanNode(index=1, description="Run the utils tests", status=StepStatus.completed)
        child_b = PlanNode(index=2, description="Write main.py", status=StepStatus.pending)
        branch = PlanNode(description="Implement the module")
        branch.set_children([child_a, child_b])
        plan = Plan(goal="Build the module", steps=[branch])
        context = self._context_rooted_at(tmp_path)
        loop = ReActLoop(
            llm_backend=_make_mock_backend(),
            tool_registry=_make_mock_tool_registry(),
            context_assembler=context,
            planner=planner,
            config=ReActConfig(),
        )

        # Re-verify child_a (already complete) via step_override -- branch
        # still has child_b pending, so no rollup should fire.
        feedback = await loop._verify_and_mark_step(
            plan,
            evidence="ran tests",
            successful_commands_this_turn=["python -m pytest"],
            step_override=child_a,
        )

        assert feedback is None
        assert child_a.status == StepStatus.completed
        planner.verify_rollup.assert_not_awaited()


class TestVerifierSeesFileContents:
    @pytest.mark.asyncio
    async def test_verifier_evidence_includes_current_file_contents(self, tmp_path):
        """The step verifier must see the verbatim current content of the
        step's files (read from disk at verification time), not just the
        turn's tool-result diffs -- the vague "duplicate code" feedback that
        sent the agent chasing a phantom problem came from the verifier
        having to guess at the file's actual state."""
        (tmp_path / "report.py").write_text(
            "def inventory_report(inventory):\n"
            "    if not inventory.items:\n"
            "        return '(empty)'\n"
            "    return inventory.items\n"
            "    return inventory.items\n"
        )
        workspace = WorkspaceManager(root=str(tmp_path))

        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "read", "arguments": {"path": "report.py"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        plan = Plan(goal="g", steps=[
            PlanNode(index=1, description="Fix the duplicated return line in report.py", status=StepStatus.pending),
        ])
        planner.generate_plan.return_value = plan
        context = _make_mock_context()
        context.workspace = workspace
        registry = _make_mock_tool_registry()
        read_tool = MagicMock()
        read_tool.name = "read"
        read_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        read_tool.run = AsyncMock(
            return_value=MagicMock(output="(contents)", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"read": read_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1),
        )
        await loop.run("do something")

        planner.verify_step.assert_awaited_once()
        _, _, evidence_arg = planner.verify_step.await_args.args
        assert "Current file contents (read directly from disk for verification):" in evidence_arg
        assert "--- report.py ---" in evidence_arg
        assert "return inventory.items" in evidence_arg


class TestVerifierSyntaxGuard:
    @pytest.mark.asyncio
    async def test_broken_file_cannot_complete_even_if_verifier_says_complete(self, tmp_path):
        """A step whose .py file does not even parse must not be marked
        complete, even when the LLM verifier (which previously returned a
        false "complete" verdict for a column-0 IndentationError, observed in
        the inventory benchmark run 4) says it is -- the deterministic syntax
        check overrides the verdict."""
        (tmp_path / "inventory.py").write_text(
            "class Inventory:\n"
            "    def remove(self, name, quantity):\n"
            "        if name not in self.stock:\n"
            " raise KeyError()\n"
        )
        workspace = WorkspaceManager(root=str(tmp_path))

        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "edit", "arguments": {"path": "inventory.py", "old_string": "raise KeyError()", "new_string": "        raise KeyError()"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        # Default verify_step mock returns complete -- the regression this
        # guards against.
        plan = Plan(goal="g", steps=[
            PlanNode(index=1, description="Implement the remove method in inventory.py", status=StepStatus.pending),
        ])
        planner.generate_plan.return_value = plan
        context = _make_mock_context()
        context.workspace = workspace
        registry = _make_mock_tool_registry()
        edit_tool = MagicMock()
        edit_tool.name = "edit"
        edit_tool.parameters = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["path", "old_string", "new_string"],
        }
        edit_tool.run = AsyncMock(
            return_value=MagicMock(output="edited inventory.py", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"edit": edit_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(
                max_steps=1,
                require_read_before_edit=False,
                lint_after_write=False,
            ),
        )
        await loop.run("do something")

        assert plan.steps[0].status == StepStatus.pending
        planner.verify_step.assert_awaited_once()
        added = [
            c.args[0].content
            for c in context.conversation.add_message.call_args_list
            if hasattr(c.args[0], "content")
        ]
        assert any("syntax error" in text for text in added if isinstance(text, str))
        assert any("E999 SyntaxError" in text for text in added if isinstance(text, str))


class TestRolledBackContentVerification:
    def test_step_file_contents_shows_attempted_content(self, tmp_path):
        """After a lint-rollback the on-disk file is the clean pre-write
        state; the verifier evidence must instead show the *attempted*
        content the model produced, so it can cite the exact offending lines."""
        path = tmp_path / "inventory.py"
        path.write_text("class Inventory:\n    def remove(self, name, quantity):\n        pass\n")
        workspace = WorkspaceManager(root=str(tmp_path))
        attempted = (
            "class Inventory:\n"
            "    def remove(self, name, quantity):\n"
            "        if name not in self.stock:\n"
            " raise KeyError()\n"
        )
        content = _step_file_contents(
            workspace,
            written_paths=[str(path.resolve())],
            named_files=[],
            rolled_back_content={str(path.resolve()): attempted},
        )
        assert "attempted write that was rolled back after lint errors" in content
        assert "raise KeyError()" in content
        assert "pass" not in content

    def test_step_python_syntax_errors_checks_attempted_content(self, tmp_path):
        """The deterministic syntax check must look at the attempted content
        even though the disk file was rolled back to a valid state -- a
        rolled-back broken write would otherwise hide from the guard."""
        path = tmp_path / "inventory.py"
        path.write_text("class Inventory:\n    def remove(self, name, quantity):\n        pass\n")
        workspace = WorkspaceManager(root=str(tmp_path))
        attempted = (
            "class Inventory:\n"
            "    def remove(self, name, quantity):\n"
            "        if name not in self.stock:\n"
            " raise KeyError()\n"
        )
        errors = _step_python_syntax_errors(
            workspace,
            written_paths=[str(path.resolve())],
            named_files=[],
            rolled_back_content={str(path.resolve()): attempted},
        )
        assert len(errors) == 1
        assert "E999 SyntaxError" in errors[0]

    def test_step_python_syntax_errors_clean_attempted_content(self, tmp_path):
        path = tmp_path / "inventory.py"
        path.write_text("class Inventory:\n    def remove(self, name, quantity):\n        pass\n")
        workspace = WorkspaceManager(root=str(tmp_path))
        attempted = (
            "class Inventory:\n"
            "    def remove(self, name, quantity):\n"
            "        if name not in self.stock:\n"
            "            raise KeyError()\n"
        )
        errors = _step_python_syntax_errors(
            workspace,
            written_paths=[str(path.resolve())],
            named_files=[],
            rolled_back_content={str(path.resolve()): attempted},
        )
        assert errors == []


class TestInvestigativeStepBypassesVerification:
    @pytest.mark.asyncio
    async def test_named_file_read_step_runs_through_verifier(self):
        """A 'read the file' step that names a specific file must be
        validated through the LLM verifier (with objective read facts in the
        evidence) rather than auto-completed on any tool evidence -- a turn
        that read a *different* file used to mark "read plan.md" done."""
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "read", "arguments": {"path": "plan.md"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        plan = Plan(goal="g", steps=[
            PlanNode(index=1, description="Read the content of plan.md", status=StepStatus.pending),
        ])
        planner.generate_plan.return_value = plan
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
            return_value=MagicMock(output="# Plan\nSome real content", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"read": read_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1),
        )
        await loop.run("do something")

        assert plan.steps[0].status == StepStatus.completed
        planner.verify_step.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_wrong_file_read_does_not_complete_named_explore_step(self):
        """Regression: an explore step that names specific files must not be
        auto-completed by a turn that only read a *different* file -- the
        benchmark showed 'Explore ... inventory.py and test_inventory.py'
        being marked done because the model read SPEC.md instead."""
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "read", "arguments": {"path": "SPEC.md"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        plan = Plan(goal="g", steps=[
            PlanNode(
                index=1,
                description=(
                    "Explore the existing codebase to understand the current "
                    "implementation of inventory.py and test_inventory.py"
                ),
                status=StepStatus.pending,
            ),
        ])
        planner.generate_plan.return_value = plan
        context = _make_mock_context()
        context.workspace.resolve = MagicMock(
            side_effect=lambda p: _FakeResolvedPath(p, exists=str(p) in {"inventory.py", "test_inventory.py"})
        )
        registry = _make_mock_tool_registry()
        read_tool = MagicMock()
        read_tool.name = "read"
        read_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        read_tool.run = AsyncMock(
            return_value=MagicMock(output="# SPEC\ncontract text", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"read": read_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1),
        )
        await loop.run("do something")

        # The step must NOT be auto-completed by reading a different file,
        # and it must not be silently handed to the LLM verifier either --
        # the objective check should reject it outright with guidance to
        # read the named files.
        assert plan.steps[0].status == StepStatus.pending
        planner.verify_step.assert_not_awaited()
        tool_msgs = [
            c.args[0].content
            for c in context.conversation.add_message.call_args_list
            if c.args[0].role == Role.tool
        ]
        assert any("`inventory.py`" in m for m in tool_msgs)
        assert any("`test_inventory.py`" in m for m in tool_msgs)
        assert any("not been read yet this session" in m for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_non_investigative_step_still_verified(self):
        """A step like 'write the implementation' isn't read/identify/
        check-style, so it should still go through normal LLM verification."""
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "write", "arguments": {"path": "a.py", "content": "x"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        plan = Plan(goal="g", steps=[
            PlanNode(index=1, description="Write the implementation code", status=StepStatus.pending),
        ])
        planner.generate_plan.return_value = plan
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        write_tool = MagicMock()
        write_tool.name = "write"
        write_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        }
        write_tool.run = AsyncMock(
            return_value=MagicMock(output="wrote a.py", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"write": write_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1, lint_after_write=False),
        )
        await loop.run("do something")

        planner.verify_step.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_investigative_step_without_tool_evidence_still_verified(self):
        """The no-tool-calls branch (pure reasoning, no actual tool ran)
        must not auto-complete an investigative step either -- reasoning
        alone isn't evidence the file was actually read."""
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(role=Role.assistant, content="I think I've read enough."),
        )
        planner = _make_mock_planner()
        plan = Plan(goal="g", steps=[
            PlanNode(index=1, description="Read the content of plan.md", status=StepStatus.pending),
        ])
        planner.generate_plan.return_value = plan
        context = _make_mock_context()
        context.workspace.resolve = MagicMock(
            side_effect=lambda p: _FakeResolvedPath(p, exists=str(p) == "plan.md")
        )
        registry = _make_mock_tool_registry()

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1),
        )
        await loop.run("do something")

        planner.verify_step.assert_not_awaited()
        tool_msgs = [
            c.args[0].content
            for c in context.conversation.add_message.call_args_list
            if c.args[0].role == Role.tool
        ]
        assert any("still needs real tool evidence" in m for m in tool_msgs)
        assert any("`plan.md`" in m for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_execution_step_without_tool_evidence_demands_bash(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(role=Role.assistant, content="Let's run hello.py using Python."),
        )
        planner = _make_mock_planner()
        plan = Plan(goal="g", steps=[
            PlanNode(
                index=1,
                description="Run hello.py using Python and verify the output",
                status=StepStatus.pending,
            ),
        ])
        planner.generate_plan.return_value = plan
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

        planner.verify_step.assert_not_awaited()
        tool_msgs = [
            c.args[0].content
            for c in context.conversation.add_message.call_args_list
            if c.args[0].role == Role.tool
        ]
        assert any("still needs real execution evidence" in m for m in tool_msgs)
        assert any("`bash`" in m for m in tool_msgs)
        assert any("`python hello.py`" in m for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_reasoning_only_direct_read_step_auto_reads_named_file(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(role=Role.assistant, content="I should inspect the file."),
        )
        planner = _make_mock_planner()
        plan = Plan(goal="g", steps=[
            PlanNode(index=1, description="Read the contents of test_slugify.py", status=StepStatus.pending),
        ])
        planner.generate_plan.return_value = plan
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
            return_value=MagicMock(output="file contents", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"read": read_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1),
        )
        await loop.run("do something")

        read_tool.run.assert_awaited_once_with(path="test_slugify.py")
        planner.verify_step.assert_awaited_once()
        assert plan.steps[0].status == StepStatus.completed

    @pytest.mark.asyncio
    async def test_reasoning_only_file_creation_step_demands_mutating_tool_evidence(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(role=Role.assistant, content="I created slugify.py and can move on."),
        )
        planner = _make_mock_planner()
        plan = Plan(goal="g", steps=[
            PlanNode(
                index=1,
                description="Create a new file named slugify.py",
                status=StepStatus.pending,
            ),
        ])
        planner.generate_plan.return_value = plan
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

        planner.verify_step.assert_not_awaited()
        tool_msgs = [
            c.args[0].content
            for c in context.conversation.add_message.call_args_list
            if c.args[0].role == Role.tool
        ]
        assert any("still needs real file-change evidence" in m for m in tool_msgs)
        assert any("`write`/`edit`" in m for m in tool_msgs)
        assert any("`slugify.py`" in m for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_reasoning_only_implementation_step_demands_mutating_tool_evidence(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(role=Role.assistant, content="The implementation is complete."),
        )
        planner = _make_mock_planner()
        plan = Plan(goal="g", steps=[
            PlanNode(
                index=1,
                description="Implement the slugify function in slugify.py according to SPEC.md",
                status=StepStatus.pending,
            ),
        ])
        planner.generate_plan.return_value = plan
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

        planner.verify_step.assert_not_awaited()
        tool_msgs = [
            c.args[0].content
            for c in context.conversation.add_message.call_args_list
            if c.args[0].role == Role.tool
        ]
        assert any("still needs real file-change evidence" in m for m in tool_msgs)
        assert any("`write`/`edit`" in m for m in tool_msgs)
        assert any("`slugify.py`" in m for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_repeated_reasoning_only_mutation_step_gets_stronger_guard(self):
        backend = _make_mock_backend()
        backend.complete = AsyncMock(side_effect=[
            CompletionResponse(
                message=Message(role=Role.assistant, content="I should create the file."),
            ),
            CompletionResponse(
                message=Message(role=Role.assistant, content="I'll create it next."),
            ),
        ])
        planner = _make_mock_planner()
        plan = Plan(goal="g", steps=[
            PlanNode(
                index=1,
                description="Create a new file named slugify.py",
                status=StepStatus.pending,
            ),
        ])
        planner.generate_plan.return_value = plan
        context = _make_mock_context()
        registry = _make_mock_tool_registry()

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=2),
        )
        await loop.run("do something")

        planner.verify_step.assert_not_awaited()
        tool_msgs = [
            c.args[0].content
            for c in context.conversation.add_message.call_args_list
            if c.args[0].role == Role.tool
        ]
        assert any("Respond by calling `write`/`edit` on `slugify.py` now." in m for m in tool_msgs)
        assert any("must contain a single `write`/`edit` tool call" in m for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_unrelated_successful_tool_call_does_not_auto_complete(self):
        """Regression test: a turn whose only tool call is unrelated to the
        step (e.g. ask_user) still "succeeds" from RetryManager's
        perspective (no failed calls), but that must not be treated as
        "has_tool_evidence" for the investigative bypass -- otherwise a
        step like "Read the existing codebase" can get silently
        auto-completed by a turn that never actually read anything."""
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "ask_user", "arguments": {"question": "Should this use SQLite or an in-memory dict?"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        plan = Plan(goal="g", steps=[
            PlanNode(
                index=1,
                description="Read the existing codebase to understand the project",
                status=StepStatus.pending,
            ),
        ])
        planner.generate_plan.return_value = plan
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        ask_tool = MagicMock()
        ask_tool.name = "ask_user"
        ask_tool.parameters = {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        }
        registry.get.side_effect = lambda name: {"ask_user": ask_tool}.get(name)

        ui = MagicMock(spec=NoopUI)
        ui.on_ask_user = AsyncMock(return_value="SQLite")

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1),
            ui=ui,
        )
        await loop.run("do something")

        planner.verify_step.assert_not_awaited()
        tool_msgs = [
            c.args[0].content
            for c in context.conversation.add_message.call_args_list
            if c.args[0].role == Role.tool
        ]
        assert any("still needs real tool evidence" in m for m in tool_msgs)


class TestAskUserVagueQuestionGuard:
    @pytest.mark.asyncio
    async def test_vague_question_blocked_with_active_step(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "ask_user", "arguments": {"question": "What would you like to do next?"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        ask_tool = MagicMock()
        ask_tool.name = "ask_user"
        ask_tool.parameters = {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        }
        ask_tool.run = AsyncMock(
            return_value=MagicMock(output="some answer", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"ask_user": ask_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1),
        )
        await loop.run("do something")

        ask_tool.run.assert_not_awaited()
        tool_msgs = [
            c.args[0].content
            for c in context.conversation.add_message.call_args_list
            if c.args[0].role == Role.tool
        ]
        assert any("active plan step" in m for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_specific_question_allowed_with_active_step(self):
        """ask_user is intercepted and routed through the UI directly
        (ui.on_ask_user), not dispatched via AskUserTool.run() -- see
        docs/archive/plans/plan_5.2.md 5.2.16: AskUserTool.run()'s own
        sys.stdin.readline()
        deadlocks under the Textual TUI, which owns the terminal."""
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "ask_user", "arguments": {"question": "Should the ledger use SQLite or an in-memory dict?"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        ask_tool = MagicMock()
        ask_tool.name = "ask_user"
        ask_tool.parameters = {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        }
        ask_tool.run = AsyncMock(
            return_value=MagicMock(output="SQLite", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"ask_user": ask_tool}.get(name)

        ui = MagicMock(spec=NoopUI)
        ui.on_ask_user = AsyncMock(return_value="SQLite")

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1),
            ui=ui,
        )
        await loop.run("do something")

        ask_tool.run.assert_not_awaited()
        ui.on_ask_user.assert_awaited_once_with(
            "Should the ledger use SQLite or an in-memory dict?"
        )

    @pytest.mark.asyncio
    async def test_disable_ask_user_answers_without_prompting_ui(self):
        """Auto-mode (ReActConfig.disable_ask_user) must answer ask_user
        itself -- even for a genuine design question -- without ever
        invoking the UI, so a "hands off" run never blocks on a prompt."""
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "ask_user", "arguments": {"question": "Should the ledger use SQLite or an in-memory dict?"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        ask_tool = MagicMock()
        ask_tool.name = "ask_user"
        ask_tool.parameters = {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        }
        ask_tool.run = AsyncMock(
            return_value=MagicMock(output="SQLite", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"ask_user": ask_tool}.get(name)

        ui = MagicMock(spec=NoopUI)
        ui.on_ask_user = AsyncMock(return_value="SQLite")

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1, disable_ask_user=True),
            ui=ui,
        )
        await loop.run("do something")

        ask_tool.run.assert_not_awaited()
        ui.on_ask_user.assert_not_awaited()
        tool_msgs = [
            c.args[0].content
            for c in context.conversation.add_message.call_args_list
            if c.args[0].role == Role.tool
        ]
        assert any("auto mode" in m.lower() for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_repeated_disabled_questions_trigger_replan(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content=(
                    '<tool_call>{"name": "ask_user", "arguments": '
                    '{"question": "Have you run the tests?"}}</tool_call>'
                ),
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        ask_tool = MagicMock()
        ask_tool.name = "ask_user"
        ask_tool.parameters = {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        }
        registry.get.side_effect = lambda name: {"ask_user": ask_tool}.get(name)
        ui = MagicMock(spec=NoopUI)
        ui.on_ask_user = AsyncMock(return_value="yes")

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(
                max_steps=2,
                max_retries_per_step=2,
                disable_ask_user=True,
            ),
            ui=ui,
        )
        await loop.run("do something")

        ui.on_ask_user.assert_not_awaited()
        planner.verify_step.assert_not_awaited()
        planner.replan_subtree.assert_awaited_once()
        assert any(
            call.args[0] == "repeat-guard"
            for call in ui.on_tool_result.call_args_list
        )

    @pytest.mark.asyncio
    async def test_vague_question_allowed_when_no_active_step(self):
        """Once every plan step is complete, there's nothing to redirect
        back to -- a vague question is fine (and arguably expected) then."""
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "ask_user", "arguments": {"question": "What would you like to do next?"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        plan = Plan(goal="g", steps=[
            PlanNode(index=1, description="Done already", status=StepStatus.completed),
        ])
        planner.generate_plan.return_value = plan
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        ask_tool = MagicMock()
        ask_tool.name = "ask_user"
        ask_tool.parameters = {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        }
        ask_tool.run = AsyncMock(
            return_value=MagicMock(output="some answer", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"ask_user": ask_tool}.get(name)

        ui = MagicMock(spec=NoopUI)
        ui.on_ask_user = AsyncMock(return_value="some answer")

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1),
            ui=ui,
        )
        await loop.run("do something")

        ask_tool.run.assert_not_awaited()
        ui.on_ask_user.assert_awaited_once_with("What would you like to do next?")

    @pytest.mark.asyncio
    async def test_guard_disabled_via_config(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "ask_user", "arguments": {"question": "What would you like to do next?"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        ask_tool = MagicMock()
        ask_tool.name = "ask_user"
        ask_tool.parameters = {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        }
        ask_tool.run = AsyncMock(
            return_value=MagicMock(output="some answer", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"ask_user": ask_tool}.get(name)

        ui = MagicMock(spec=NoopUI)
        ui.on_ask_user = AsyncMock(return_value="some answer")

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1, block_vague_ask_user=False),
            ui=ui,
        )
        await loop.run("do something")

        ask_tool.run.assert_not_awaited()
        ui.on_ask_user.assert_awaited_once_with("What would you like to do next?")


class TestPrematureFinishGuard:
    @pytest.mark.asyncio
    async def test_finish_is_rechecked_after_bundled_work_completes_plan(self, tmp_path):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content=(
                    '<tool_call>{"name": "write", "arguments": '
                    '{"path": "hello.py", "content": "print(1)"}}</tool_call>\n'
                    '<tool_call>{"name": "finish", "arguments": '
                    '{"summary": "Done!"}}</tool_call>'
                ),
            )
        )
        planner = _make_mock_planner()
        plan = Plan(
            goal="g",
            steps=[PlanNode(index=1, description="Create hello.py")],
        )
        planner.generate_plan.return_value = plan
        context = _make_mock_context()
        context.workspace.resolve = MagicMock(side_effect=lambda path: tmp_path / path)
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

        async def do_write(path: str, content: str):
            (tmp_path / path).write_text(content)
            return MagicMock(output=f"Wrote {path}", success=True, error=None)

        write_tool.run = AsyncMock(side_effect=do_write)
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
            "write": write_tool,
            "finish": finish_tool,
        }.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1, lint_after_write=False),
        )
        result = await loop.run("do something")

        assert result.success is True
        assert plan.is_complete()
        write_tool.run.assert_awaited_once()
        finish_tool.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_bundled_work_advances_step_before_finish_remains_blocked(self, tmp_path):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content=(
                    '<tool_call>{"name": "write", "arguments": '
                    '{"path": "hello.py", "content": "print(1)"}}</tool_call>\n'
                    '<tool_call>{"name": "bash", "arguments": '
                    '{"command": "python hello.py"}}</tool_call>\n'
                    '<tool_call>{"name": "finish", "arguments": '
                    '{"summary": "Done!"}}</tool_call>'
                ),
            )
        )
        planner = _make_mock_planner()
        plan = Plan(
            goal="g",
            steps=[
                PlanNode(index=1, description="Create hello.py"),
                PlanNode(index=2, description="Run hello.py and verify its output"),
            ],
        )
        planner.generate_plan.return_value = plan
        context = _make_mock_context()
        context.workspace.resolve = MagicMock(side_effect=lambda path: tmp_path / path)
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

        async def do_write(path: str, content: str):
            (tmp_path / path).write_text(content)
            return MagicMock(output=f"Wrote {path}", success=True, error=None)

        write_tool.run = AsyncMock(side_effect=do_write)
        bash_tool = MagicMock()
        bash_tool.name = "bash"
        bash_tool.parameters = {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }
        bash_tool.run = AsyncMock(
            return_value=MagicMock(output="Hello, world!", success=True, error=None)
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
            "write": write_tool,
            "bash": bash_tool,
            "finish": finish_tool,
        }.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1, lint_after_write=False),
        )
        result = await loop.run("do something")

        assert result.success is False
        assert plan.steps[0].status == StepStatus.completed
        assert plan.steps[1].status == StepStatus.pending
        bash_tool.run.assert_awaited_once()
        finish_tool.run.assert_not_awaited()
        tool_messages = [
            call.args[0].content
            for call in context.conversation.add_message.call_args_list
            if call.args[0].role == Role.tool
        ]
        assert any("Run hello.py" in message for message in tool_messages)

    @pytest.mark.parametrize(
        "unresolved_status",
        [StepStatus.pending, StepStatus.failed, StepStatus.skipped],
    )
    @pytest.mark.asyncio
    async def test_finish_blocked_with_unresolved_steps(self, unresolved_status):
        """Reproduces the bug: agent reads the plan, then immediately calls
        finish with a summary claiming work was done, without ever writing
        anything. finish must be rejected while steps remain pending."""
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "finish", "arguments": {"summary": "Done!"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        plan = Plan(goal="g", steps=[
            PlanNode(index=1, description="Read plan.md", status=StepStatus.completed),
            PlanNode(
                index=2,
                description="Implement the feature",
                status=unresolved_status,
            ),
        ])
        planner.generate_plan.return_value = plan
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
            config=ReActConfig(max_steps=1),
        )
        result = await loop.run("do something")

        finish_tool.run.assert_not_awaited()
        assert result.success is False
        tool_msgs = [
            c.args[0].content
            for c in context.conversation.add_message.call_args_list
            if c.args[0].role == Role.tool
        ]
        assert any("cannot finish yet" in m for m in tool_msgs)
        assert any("Implement the feature" in m for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_finish_allowed_when_all_steps_complete(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "finish", "arguments": {"summary": "Done!"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        plan = Plan(goal="g", steps=[
            PlanNode(index=1, description="Read plan.md", status=StepStatus.completed),
            PlanNode(index=2, description="Implement the feature", status=StepStatus.completed),
        ])
        planner.generate_plan.return_value = plan
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
            config=ReActConfig(max_steps=1),
        )
        result = await loop.run("do something")

        finish_tool.run.assert_awaited_once()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_finish_allowed_for_plan_with_no_steps(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "finish", "arguments": {"summary": "Done!"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        plan = Plan(goal="g", steps=[])
        planner.generate_plan.return_value = plan
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
            config=ReActConfig(max_steps=1),
        )
        result = await loop.run("do something")

        finish_tool.run.assert_awaited_once()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_guard_disabled_via_config(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "finish", "arguments": {"summary": "Done!"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        plan = Plan(goal="g", steps=[
            PlanNode(index=1, description="Implement the feature", status=StepStatus.pending),
        ])
        planner.generate_plan.return_value = plan
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
            config=ReActConfig(max_steps=1, require_plan_complete_to_finish=False),
        )
        result = await loop.run("do something")

        finish_tool.run.assert_awaited_once()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_blocked_finish_does_not_advance_call_signature_repeat_detection(self):
        """A blocked finish followed by real progress shouldn't be mistaken
        for a repeated call -- different tool calls, so no false nudge."""
        backend = _make_mock_backend()
        backend.complete.side_effect = [
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content='<tool_call>{"name": "finish", "arguments": {"summary": "Done!"}}</tool_call>',
                )
            ),
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content='<tool_call>{"name": "write", "arguments": {"path": "a.py", "content": "x"}}</tool_call>',
                )
            ),
        ]
        planner = _make_mock_planner()
        plan = Plan(goal="g", steps=[
            PlanNode(index=1, description="Implement the feature", status=StepStatus.pending),
        ])
        planner.generate_plan.return_value = plan
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        finish_tool = MagicMock()
        finish_tool.name = "finish"
        finish_tool.parameters = {"type": "object", "properties": {"summary": {"type": "string"}}}
        finish_tool.run = AsyncMock(return_value=MagicMock(output="done", success=True, error=None))
        write_tool = MagicMock()
        write_tool.name = "write"
        write_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        }
        write_tool.run = AsyncMock(
            return_value=MagicMock(output="wrote a.py", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"finish": finish_tool, "write": write_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=2, lint_after_write=False),
        )
        await loop.run("do something")

        write_tool.run.assert_awaited_once()
        finish_tool.run.assert_not_awaited()


class TestFeedbackVisibleInUI:
    @pytest.mark.asyncio
    async def test_verify_feedback_shown_via_on_tool_result(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "write", "arguments": {"path": "a.py", "content": "x"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        plan = _make_default_plan()
        planner.generate_plan.return_value = plan
        planner.verify_step = AsyncMock(
            return_value=(StepVerdict.partial, "only wrote a stub")
        )
        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        write_tool = MagicMock()
        write_tool.name = "write"
        write_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        }
        write_tool.run = AsyncMock(
            return_value=MagicMock(output="wrote a.py", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"write": write_tool}.get(name)

        ui = MagicMock()
        ui.on_tool_result = MagicMock()
        ui.on_approval_request = AsyncMock(return_value=None)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1, lint_after_write=False),
            ui=ui,
        )
        await loop.run("do something")

        calls = [c.args for c in ui.on_tool_result.call_args_list]
        assert any(name == "plan-check" and "not fully done yet" in msg for name, msg in calls)

    @pytest.mark.asyncio
    async def test_repeat_nudge_shown_via_on_tool_result(self):
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

        ui = MagicMock()
        ui.on_tool_result = MagicMock()
        ui.on_approval_request = AsyncMock(return_value=None)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=2),
            ui=ui,
        )
        await loop.run("do something")

        calls = [c.args for c in ui.on_tool_result.call_args_list]
        assert any(name == "repeat-guard" and "repeated the exact same tool call" in msg for name, msg in calls)


class TestReadBeforeEditGuard:
    @pytest.mark.asyncio
    async def test_edit_on_unread_existing_file_is_blocked(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "edit", "arguments": {"path": "a.py", "old_string": "x", "new_string": "y"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        context.workspace.resolve = MagicMock(
            side_effect=lambda p: _FakeResolvedPath(p, exists=True)
        )
        registry = _make_mock_tool_registry()
        edit_tool = MagicMock()
        edit_tool.name = "edit"
        edit_tool.parameters = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["path", "old_string", "new_string"],
        }
        edit_tool.run = AsyncMock(
            return_value=MagicMock(output="edited a.py", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"edit": edit_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1),
        )
        await loop.run("do something")

        edit_tool.run.assert_not_awaited()
        tool_msgs = [
            c.args[0].content
            for c in context.conversation.add_message.call_args_list
            if c.args[0].role == Role.tool
        ]
        assert any("haven't read" in m for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_edit_on_new_file_is_allowed(self):
        """A file that doesn't exist yet has nothing to read -- editing
        (creating) it should proceed without a prior read."""
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "edit", "arguments": {"path": "new.py", "old_string": "", "new_string": "x = 1"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()  # default: resolve().exists() is False
        registry = _make_mock_tool_registry()
        edit_tool = MagicMock()
        edit_tool.name = "edit"
        edit_tool.parameters = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["path", "old_string", "new_string"],
        }
        edit_tool.run = AsyncMock(
            return_value=MagicMock(output="edited new.py", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"edit": edit_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1),
        )
        await loop.run("do something")

        edit_tool.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_edit_after_read_is_allowed(self):
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
                    content='<tool_call>{"name": "edit", "arguments": {"path": "a.py", "old_string": "x", "new_string": "y"}}</tool_call>',
                )
            ),
        ]
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        context.workspace.resolve = MagicMock(
            side_effect=lambda p: _FakeResolvedPath(p, exists=True)
        )
        registry = _make_mock_tool_registry()
        read_tool = MagicMock()
        read_tool.name = "read"
        read_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        read_tool.run = AsyncMock(
            return_value=MagicMock(output="file contents", success=True, error=None)
        )
        edit_tool = MagicMock()
        edit_tool.name = "edit"
        edit_tool.parameters = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["path", "old_string", "new_string"],
        }
        edit_tool.run = AsyncMock(
            return_value=MagicMock(output="edited a.py", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"read": read_tool, "edit": edit_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=2),
        )
        await loop.run("do something")

        edit_tool.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_read_then_edit_in_same_turn_runs_in_order(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content=(
                    '<tool_call>{"name": "read", "arguments": {"path": "a.py"}}</tool_call>'
                    '<tool_call>{"name": "edit", "arguments": {"path": "a.py", '
                    '"old_string": "x", "new_string": "y"}}</tool_call>'
                ),
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        context.workspace.resolve = MagicMock(
            side_effect=lambda p: _FakeResolvedPath(p, exists=True)
        )
        registry = _make_mock_tool_registry()
        order: list[str] = []

        async def read_run(**_kwargs):
            order.append("read")
            return MagicMock(output="x", success=True, error=None)

        async def edit_run(**_kwargs):
            order.append("edit")
            return MagicMock(output="edited", success=True, error=None)

        read_tool = MagicMock(
            name="read",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        read_tool.run = AsyncMock(side_effect=read_run)
        edit_tool = MagicMock(
            name="edit",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        )
        edit_tool.run = AsyncMock(side_effect=edit_run)
        registry.get.side_effect = lambda name: {
            "read": read_tool,
            "edit": edit_tool,
        }.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1, lint_after_write=False),
        )
        await loop.run("do something")

        assert order == ["read", "edit"]

    @pytest.mark.asyncio
    async def test_guard_disabled_via_config(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "edit", "arguments": {"path": "a.py", "old_string": "x", "new_string": "y"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = _make_default_plan()
        context = _make_mock_context()
        context.workspace.resolve = MagicMock(
            side_effect=lambda p: _FakeResolvedPath(p, exists=True)
        )
        registry = _make_mock_tool_registry()
        edit_tool = MagicMock()
        edit_tool.name = "edit"
        edit_tool.parameters = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["path", "old_string", "new_string"],
        }
        edit_tool.run = AsyncMock(
            return_value=MagicMock(output="edited a.py", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {"edit": edit_tool}.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=1, require_read_before_edit=False),
        )
        await loop.run("do something")

        edit_tool.run.assert_awaited_once()


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


class TestLooksInvestigative:
    def test_read_step(self):
        from chef_human.agent.react_loop import _looks_investigative
        assert _looks_investigative("Read the content of plan.md") is True

    def test_identify_step(self):
        from chef_human.agent.react_loop import _looks_investigative
        assert _looks_investigative("Identify the tasks for part 1") is True

    def test_check_step(self):
        from chef_human.agent.react_loop import _looks_investigative
        assert _looks_investigative("Check that the config is valid") is True

    def test_case_insensitive(self):
        from chef_human.agent.react_loop import _looks_investigative
        assert _looks_investigative("REVIEW the existing tests") is True

    def test_write_step_not_investigative(self):
        from chef_human.agent.react_loop import _looks_investigative
        assert _looks_investigative("Write the implementation code") is False

    def test_create_step_not_investigative(self):
        from chef_human.agent.react_loop import _looks_investigative
        assert _looks_investigative("Create a new file named foo.py") is False

    def test_test_step_not_investigative(self):
        from chef_human.agent.react_loop import _looks_investigative
        assert _looks_investigative("Test the implementation by running it") is False


class TestIsVagueNextStepQuestion:
    def test_exact_phrase_from_bug_report(self):
        from chef_human.agent.react_loop import _is_vague_next_step_question
        assert _is_vague_next_step_question("What would you like to do next?") is True

    def test_variant_phrasing(self):
        from chef_human.agent.react_loop import _is_vague_next_step_question
        assert _is_vague_next_step_question("What should I do next?") is True
        assert _is_vague_next_step_question("What do you want me to do?") is True

    def test_case_insensitive(self):
        from chef_human.agent.react_loop import _is_vague_next_step_question
        assert _is_vague_next_step_question("WHAT'S NEXT?") is True

    def test_specific_question_not_vague(self):
        from chef_human.agent.react_loop import _is_vague_next_step_question
        assert _is_vague_next_step_question(
            "Should the ledger use SQLite or an in-memory dict?"
        ) is False

    def test_specific_confirmation_not_vague(self):
        from chef_human.agent.react_loop import _is_vague_next_step_question
        assert _is_vague_next_step_question(
            "Do you want to overwrite the existing hello_world.py?"
        ) is False


class TestIsLowValueAskUserQuestion:
    def test_permission_seeking_do_you_want(self):
        from chef_human.agent.react_loop import _is_low_value_ask_user_question
        assert _is_low_value_ask_user_question(
            "Do you want to proceed with grepping through the source files?"
        ) is True

    def test_permission_seeking_should_i(self):
        from chef_human.agent.react_loop import _is_low_value_ask_user_question
        assert _is_low_value_ask_user_question("Should I proceed with this step?") is True

    def test_permission_seeking_can_i(self):
        from chef_human.agent.react_loop import _is_low_value_ask_user_question
        assert _is_low_value_ask_user_question("Can I go ahead and edit the file?") is True

    def test_permission_seeking_is_it_ok(self):
        from chef_human.agent.react_loop import _is_low_value_ask_user_question
        assert _is_low_value_ask_user_question("Is it ok if I overwrite hello_world.py?") is True

    @pytest.mark.parametrize(
        "question",
        [
            "Have you run the tests?",
            "Have you executed hello.py?",
            "Have you tested the program?",
            "Have you verified the output?",
            "Have you checked the file?",
            "Have you created the module?",
            "Have you configured the project?",
        ],
    )
    def test_status_confirmation_questions_are_low_value(self, question):
        from chef_human.agent.react_loop import _is_low_value_ask_user_question

        assert _is_low_value_ask_user_question(question) is True

    def test_vague_next_step_still_caught(self):
        from chef_human.agent.react_loop import _is_low_value_ask_user_question
        assert _is_low_value_ask_user_question("What would you like to do next?") is True

    def test_genuine_design_question_not_blocked(self):
        from chef_human.agent.react_loop import _is_low_value_ask_user_question
        assert _is_low_value_ask_user_question(
            "Should the ledger use SQLite or an in-memory dict?"
        ) is False

    def test_genuine_design_question_naming_not_blocked(self):
        from chef_human.agent.react_loop import _is_low_value_ask_user_question
        assert _is_low_value_ask_user_question(
            "What should the primary key column be named?"
        ) is False


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
            config=ReActConfig(max_steps=3, require_plan_complete_to_finish=False),
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
            config=ReActConfig(require_plan_complete_to_finish=False),
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
            config=ReActConfig(require_plan_complete_to_finish=False),
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

    @pytest.mark.asyncio
    async def test_planner_usage_is_included_in_totals(self):
        """Planner's plan-building call runs on a separate LLM call path
        from the main reasoning loop -- ReActLoop must wire itself up as
        planner.on_usage so those tokens count towards the same total
        (previously they were silently dropped)."""
        # Real Planner (not a mock) so on_usage actually gets exercised via
        # its _complete() helper.
        planner_backend = MagicMock(spec=LLMBackend)
        planner_backend.complete = AsyncMock(return_value=CompletionResponse(
            message=Message(role=Role.assistant, content='["Do the thing"]'),
            usage={"prompt_tokens": 40, "completion_tokens": 8},
        ))
        planner = Planner(planner_backend)

        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>',
            ),
            usage={"prompt_tokens": 50, "completion_tokens": 10},
        )
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
            config=ReActConfig(require_plan_complete_to_finish=False),
        )
        result = await loop.run("do something")

        # 50 (main loop) + 40 (planner's expansion call) + 40 (atomicity
        # check on the single leaf child) = 130
        assert result.total_prompt_tokens == 130
        assert result.total_completion_tokens == 26

    @pytest.mark.asyncio
    async def test_planner_usage_reported_to_ui_live(self):
        """Planner usage must reach the UI via on_token_usage as it
        happens, not just get folded into the end-of-task total."""
        planner_backend = MagicMock(spec=LLMBackend)
        planner_backend.complete = AsyncMock(return_value=CompletionResponse(
            message=Message(role=Role.assistant, content='["Do the thing"]'),
            usage={"prompt_tokens": 40, "completion_tokens": 8},
        ))
        planner = Planner(planner_backend)

        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>',
            ),
            usage=None,
        )
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
        ui.on_approval_request = AsyncMock(return_value=True)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(require_plan_complete_to_finish=False),
            ui=ui,
        )
        await loop.run("do something")

        ui.on_token_usage.assert_any_call(40, 8)


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
        planner.update_plan.return_value = Plan(goal="Replanned", steps=[])
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
            config=ReActConfig(
                max_steps=3, tool_timeout=30.0, require_plan_complete_to_finish=False
            ),
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
            config=ReActConfig(max_steps=5, require_plan_complete_to_finish=False),
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


class TestAutoFinishWhenPlanComplete:
    @pytest.mark.asyncio
    async def test_finishes_after_two_pointless_turns_post_completion(self):
        """Regression test: once plan.current_leaf() is None, the prompt
        tells the model to call `finish`, but a weak local model can just
        keep calling arbitrary (successful, non-repeating -- so the
        repeat-guard doesn't catch it) tools instead of finishing. After
        two such wasted turns in a row, the loop must finish on the
        model's behalf instead of grinding on until max_steps."""
        backend = _make_mock_backend()
        backend.complete.side_effect = [
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content='<tool_call>{"name": "ls_tree", "arguments": {}}</tool_call>',
                )
            ),
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content='<tool_call>{"name": "grep", "arguments": {"pattern": "foo"}}</tool_call>',
                )
            ),
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content='<tool_call>{"name": "read", "arguments": {"path": "bar.py"}}</tool_call>',
                )
            ),
        ]
        planner = _make_mock_planner()
        plan = Plan(goal="g", steps=[
            PlanNode(index=1, description="Explore the project tree", status=StepStatus.pending),
        ])
        planner.generate_plan.return_value = plan
        context = _make_mock_context()
        registry = _make_mock_tool_registry()

        def make_tool(name: str, output: str, properties: dict) -> MagicMock:
            tool = MagicMock()
            tool.name = name
            tool.parameters = {"type": "object", "properties": properties}
            tool.run = AsyncMock(return_value=MagicMock(output=output, success=True, error=None))
            return tool

        tools = {
            "ls_tree": make_tool("ls_tree", "tree", {}),
            "grep": make_tool("grep", "no matches", {"pattern": {"type": "string"}}),
            "read": make_tool("read", "file contents", {"path": {"type": "string"}}),
        }
        registry.get.side_effect = lambda name: tools.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=25),
        )
        result = await loop.run("do something")

        # Should stop well short of max_steps=25: turn 1 completes the
        # (investigative) step, turns 2-3 are the "pointless but
        # successful" turns that trigger auto-finish.
        assert result.steps_taken <= 3
        assert result.success is True
        assert plan.steps[0].status == StepStatus.completed
        planner.verify_step.assert_not_awaited()


def _evidence(files_written: dict, commands: list) -> StepEvidence:
    evidence = StepEvidence()
    evidence.merge_turn(files_written, commands)
    return evidence


class TestSubtreeReplanAndEvidence:
    """Ticket 04: evidence propagation + subtree-scoped replan (see
    .scratch/planning-tree/issues/04-evidence-propagation-and-subtree-replan.md)."""

    def _make_loop(self, planner: MagicMock) -> ReActLoop:
        return ReActLoop(
            llm_backend=_make_mock_backend(),
            tool_registry=_make_mock_tool_registry(),
            context_assembler=_make_mock_context(),
            planner=planner,
            config=ReActConfig(),
        )

    @pytest.mark.asyncio
    async def test_evidence_propagates_to_every_ancestor_at_write_time(self):
        planner = _make_mock_planner()
        loop = self._make_loop(planner)

        branch = PlanNode(index=1, description="Build the subsystem")
        leaf = PlanNode(index=1, description="Write feature.py")
        branch.set_children([leaf])
        plan = Plan(goal="Task", steps=[branch])

        leaf.status = StepStatus.pending
        # Drive the leaf through verification with real write evidence so
        # merge_turn actually records something to propagate.
        loop._planner.verify_step = AsyncMock(return_value=(StepVerdict.complete, "done"))
        await loop._verify_and_mark_step(
            plan,
            "wrote feature.py",
            files_written_this_turn={"/repo/feature.py": False},
            successful_commands_this_turn=["python feature.py"],
        )

        leaf_evidence = loop._step_evidence[leaf.node_id]
        branch_evidence = loop._step_evidence[branch.node_id]
        assert leaf_evidence.files_written == branch_evidence.files_written
        assert leaf_evidence.successful_commands == branch_evidence.successful_commands
        assert "/repo/feature.py" in branch_evidence.files_written

    @pytest.mark.asyncio
    async def test_leaf_failure_replan_is_scoped_to_that_node(self):
        planner = _make_mock_planner()
        loop = self._make_loop(planner)

        sibling = PlanNode(index=1, description="Untouched sibling")
        failing_leaf = PlanNode(index=2, description="Failing leaf")
        plan = Plan(goal="Task", steps=[sibling, failing_leaf])

        loop._last_failed_node = failing_leaf
        loop._step_evidence[failing_leaf.node_id] = StepEvidence()

        await loop._replan_failing_node(plan, "it failed")

        planner.replan_subtree.assert_awaited_once_with(plan, failing_leaf, "it failed")
        assert loop._last_failed_node is None

    @pytest.mark.asyncio
    async def test_branch_rollup_rejection_targets_the_branch_not_current_leaf(self):
        planner = _make_mock_planner()
        loop = self._make_loop(planner)

        branch = PlanNode(index=1, description="Build the subsystem")
        leaf = PlanNode(index=1, description="Write feature.py", status=StepStatus.completed)
        branch.set_children([leaf])
        next_leaf = PlanNode(index=2, description="Unrelated next step")
        plan = Plan(goal="Task", steps=[branch, next_leaf])

        planner.verify_rollup = AsyncMock(
            return_value=(StepVerdict.not_complete, "coverage gap")
        )
        feedback = await loop._process_rollups(plan)

        assert feedback is not None
        # current_leaf() has already moved on to next_leaf, but the failing
        # node the replan should target is the rejected branch.
        assert plan.current_leaf() is next_leaf
        assert loop._last_failed_node is branch

        await loop._replan_failing_node(plan, "coverage gap")
        planner.replan_subtree.assert_awaited_once_with(plan, branch, "coverage gap")

    @pytest.mark.asyncio
    async def test_rollup_exception_targets_the_branch_not_a_stale_leaf(self):
        """Regression: an exception from verify_rollup used to leave
        _last_failed_node pointing at whatever leaf/branch was last recorded
        (e.g. the leaf that just completed and triggered this rollup),
        so a subsequent replan would reset an already-succeeded leaf back to
        pending instead of retrying the branch whose rollup actually failed."""
        planner = _make_mock_planner()
        loop = self._make_loop(planner)

        branch = PlanNode(index=1, description="Build the subsystem")
        leaf = PlanNode(index=1, description="Write feature.py", status=StepStatus.completed)
        branch.set_children([leaf])
        plan = Plan(goal="Task", steps=[branch])

        loop._last_failed_node = leaf  # stale, as if leaf just completed
        planner.verify_rollup = AsyncMock(side_effect=RuntimeError("backend hiccup"))

        feedback = await loop._process_rollups(plan)

        assert feedback is not None
        assert loop._last_failed_node is branch

    @pytest.mark.asyncio
    async def test_completed_rollup_clears_last_failed_node(self):
        """Regression: a branch passing its own rollup used to leave
        _last_failed_node unchanged, so a later rollup failure elsewhere in
        the tree (via the exception path) could get mis-attributed back to
        this now-completed branch instead of the branch that actually needs
        a replan."""
        planner = _make_mock_planner()
        loop = self._make_loop(planner)

        branch = PlanNode(index=1, description="Build the subsystem")
        leaf = PlanNode(index=1, description="Write feature.py", status=StepStatus.completed)
        branch.set_children([leaf])
        plan = Plan(goal="Task", steps=[branch])

        loop._last_failed_node = leaf  # stale, as if leaf just completed
        planner.verify_rollup = AsyncMock(return_value=(StepVerdict.complete, "covered"))

        feedback = await loop._process_rollups(plan)

        assert feedback is None
        assert loop._last_failed_node is None

    def test_subtree_replan_discards_descendant_evidence_and_scrubs_ancestors(self):
        planner = _make_mock_planner()
        loop = self._make_loop(planner)

        root_leaf_sibling = PlanNode(index=1, description="Sibling branch")
        branch = PlanNode(index=2, description="Failing branch")
        child_a = PlanNode(index=1, description="child a")
        child_b = PlanNode(index=2, description="child b")
        branch.set_children([child_a, child_b])
        _ = Plan(goal="Task", steps=[root_leaf_sibling, branch])

        loop._step_evidence[child_a.node_id] = _evidence({"a.py": True}, ["cmd-a"])
        loop._step_evidence[child_b.node_id] = _evidence({"b.py": True}, ["cmd-b"])
        loop._step_evidence[branch.node_id] = _evidence(
            {"a.py": True, "b.py": True}, ["cmd-a", "cmd-b"]
        )

        loop._discard_subtree_evidence(branch)

        assert child_a.node_id not in loop._step_evidence
        assert child_b.node_id not in loop._step_evidence
        assert loop._step_evidence[branch.node_id].files_written == {}
        assert loop._step_evidence[branch.node_id].successful_commands == []

        # New children replace the old ones (as a real replan_subtree call
        # would do), then ancestor buckets are rebuilt from what's actually
        # left in the tree.
        new_child = PlanNode(index=1, description="fresh child")
        branch.set_children([new_child])
        loop._step_evidence[new_child.node_id] = _evidence({"c.py": True}, ["cmd-c"])
        loop._rebuild_ancestor_evidence(new_child)

        branch_evidence = loop._step_evidence[branch.node_id]
        assert branch_evidence.files_written == {"c.py": True}
        assert branch_evidence.successful_commands == ["cmd-c"]
        assert "a.py" not in branch_evidence.files_written
        assert "b.py" not in branch_evidence.files_written


class TestPerNodeRetryEscalation:
    """RetryManager tracks retry/replan pressure per node_id (ticket 05) --
    ReActLoop is where that turns into tree mutations: in headless mode
    (disable_ask_user) a node that exhausts its own budget is marked failed
    and execution continues onto the rest of the tree instead of the whole
    run terminating."""

    @pytest.mark.asyncio
    async def test_headless_marks_node_failed_and_continues_to_next_sibling(self):
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "read", "arguments": {"path": "x.py"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        step_one = PlanNode(index=1, description="Step one", status=StepStatus.pending)
        step_two = PlanNode(index=2, description="Step two", status=StepStatus.pending)
        plan = Plan(goal="Test task", steps=[step_one, step_two])
        planner.generate_plan.return_value = plan

        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        read_tool = MagicMock()
        read_tool.name = "read"
        read_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        underlying_run = _make_tool_run("fail", success=False)
        call_count = 0

        async def counting_run(**kwargs):
            nonlocal call_count
            call_count += 1
            return await underlying_run(**kwargs)

        read_tool.run = counting_run
        registry.get.return_value = read_tool

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(
                max_steps=4,
                max_retries_per_step=1,
                max_replans=0,
                disable_ask_user=True,
            ),
        )
        result = await loop.run("do something")

        # Both leaves exhausted their own (independent) budget and were
        # marked failed -- the run kept going past step one's escalation
        # instead of terminating on the first one.
        assert step_one.status == StepStatus.failed
        assert step_two.status == StepStatus.failed
        assert call_count >= 2
        # Neither leaf ever reached `completed`, so the run can't report
        # success -- it should exhaust max_steps rather than escalate-abort
        # immediately after the first node fails.
        assert result.success is False

    @pytest.mark.asyncio
    async def test_interactive_mode_also_marks_failed_and_continues(self):
        """Ticket 06: interactive mode applies the identical mark-failed-and-
        continue default as headless -- the run never blocks waiting on a
        human mid-execution. It differs from headless only in that a
        notification is recorded (`AgentResult.escalations`) for the human to
        act on retroactively."""
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(
                role=Role.assistant,
                content='<tool_call>{"name": "read", "arguments": {"path": "x.py"}}</tool_call>',
            )
        )
        planner = _make_mock_planner()
        step_one = PlanNode(index=1, description="Step one", status=StepStatus.pending)
        step_two = PlanNode(index=2, description="Step two", status=StepStatus.pending)
        plan = Plan(goal="Test task", steps=[step_one, step_two])
        planner.generate_plan.return_value = plan

        context = _make_mock_context()
        registry = _make_mock_tool_registry()
        read_tool = MagicMock()
        read_tool.name = "read"
        read_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        underlying_run = _make_tool_run("fail", success=False)
        call_count = 0

        async def counting_run(**kwargs):
            nonlocal call_count
            call_count += 1
            return await underlying_run(**kwargs)

        read_tool.run = counting_run
        registry.get.return_value = read_tool

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(
                max_steps=4,
                max_retries_per_step=1,
                max_replans=0,
            ),
        )
        result = await loop.run("do something")

        # Same fallback as headless: both leaves get their own turn and are
        # marked failed individually rather than the run terminating on the
        # first escalation.
        assert step_one.status == StepStatus.failed
        assert step_two.status == StepStatus.failed
        assert call_count >= 2
        assert result.success is False
        assert len(result.escalations) == 2
        assert {e.node_id for e in result.escalations} == {
            step_one.node_id,
            step_two.node_id,
        }


class TestResolveEscalation:
    """Ticket 06: after a node escalates and gets auto-marked failed, the
    human can retroactively edit/redecompose/reject via
    `ReActLoop.resolve_escalation`, reversing the applied default and
    resuming execution."""

    @pytest.mark.asyncio
    async def test_edit_unmarks_failed_and_retries_node(self):
        read_call = '<tool_call>{"name": "read", "arguments": {"path": "x.py"}}</tool_call>'
        finish_call = '<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>'
        backend = _make_mock_backend()
        backend.complete = AsyncMock(
            side_effect=[
                CompletionResponse(message=Message(role=Role.assistant, content=read_call)),
                CompletionResponse(message=Message(role=Role.assistant, content=read_call)),
                CompletionResponse(message=Message(role=Role.assistant, content=finish_call)),
            ]
        )
        planner = _make_mock_planner()
        step_one = PlanNode(index=1, description="Step one", status=StepStatus.pending)
        plan = Plan(goal="Test task", steps=[step_one])
        planner.generate_plan.return_value = plan

        registry = _make_mock_tool_registry()
        read_tool = MagicMock()
        read_tool.name = "read"
        read_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        call_count = 0

        async def flaky_run(**kwargs):
            nonlocal call_count
            call_count += 1
            obj = MagicMock()
            obj.success = call_count > 1
            obj.output = "ok" if obj.success else "fail"
            obj.error = None if obj.success else "fail"
            return obj

        read_tool.run = flaky_run
        finish_tool = MagicMock()
        finish_tool.name = "finish"
        finish_tool.parameters = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        }
        finish_tool.run = AsyncMock(
            return_value=MagicMock(output="Task complete: done", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {
            "read": read_tool, "finish": finish_tool
        }.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=_make_mock_context(),
            planner=planner,
            config=ReActConfig(max_steps=4, max_retries_per_step=1, max_replans=0),
        )
        result = await loop.run("do something")
        assert step_one.status == StepStatus.failed
        assert len(result.escalations) == 1

        resumed = await loop.resolve_escalation(
            result.plan, "do something", step_one.node_id, "edit",
            description="Step one (edited)",
        )

        assert step_one.description == "Step one (edited)"
        assert step_one.status == StepStatus.completed
        assert resumed.success is True
        assert resumed.escalations == []

    @pytest.mark.asyncio
    async def test_reject_aborts_without_retrying(self):
        planner = _make_mock_planner()
        step_one = PlanNode(index=1, description="Step one", status=StepStatus.failed)
        plan = Plan(goal="Test task", steps=[step_one])

        loop = ReActLoop(
            llm_backend=_make_mock_backend(),
            tool_registry=_make_mock_tool_registry(),
            context_assembler=_make_mock_context(),
            planner=planner,
            config=ReActConfig(max_steps=4),
        )
        loop._escalations.append(
            EscalationRecord(
                node_id=step_one.node_id, description=step_one.description, message="x"
            )
        )

        result = await loop.resolve_escalation(
            plan, "do something", step_one.node_id, "reject"
        )

        assert result.success is False
        assert step_one.status == StepStatus.failed
        assert loop._escalations == []

    @pytest.mark.asyncio
    async def test_redecompose_calls_planner_with_guidance(self):
        read_call = '<tool_call>{"name": "read", "arguments": {"path": "x.py"}}</tool_call>'
        finish_call = '<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>'
        backend = _make_mock_backend()
        backend.complete = AsyncMock(
            side_effect=[
                CompletionResponse(message=Message(role=Role.assistant, content=read_call)),
                CompletionResponse(message=Message(role=Role.assistant, content=finish_call)),
            ]
        )
        planner = _make_mock_planner()
        step_one = PlanNode(index=1, description="Step one", status=StepStatus.failed)
        plan = Plan(goal="Test task", steps=[step_one])

        registry = _make_mock_tool_registry()
        read_tool = MagicMock()
        read_tool.name = "read"
        read_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        read_tool.run = _make_tool_run("ok", success=True)
        finish_tool = MagicMock()
        finish_tool.name = "finish"
        finish_tool.parameters = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        }
        finish_tool.run = AsyncMock(
            return_value=MagicMock(output="Task complete: done", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {
            "read": read_tool, "finish": finish_tool
        }.get(name)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=_make_mock_context(),
            planner=planner,
            config=ReActConfig(max_steps=4),
        )
        loop._escalations.append(
            EscalationRecord(
                node_id=step_one.node_id, description=step_one.description, message="x"
            )
        )

        resumed = await loop.resolve_escalation(
            plan, "do something", step_one.node_id, "redecompose",
            guidance="try a different approach",
        )

        planner.replan_subtree.assert_awaited_once()
        _, called_node, called_guidance = planner.replan_subtree.await_args.args
        assert called_node is step_one
        assert called_guidance == "try a different approach"
        assert resumed.success is True

    @pytest.mark.asyncio
    async def test_unknown_node_id_raises(self):
        plan = Plan(goal="Test task", steps=[PlanNode(index=1, description="Step one")])
        loop = ReActLoop(
            llm_backend=_make_mock_backend(),
            tool_registry=_make_mock_tool_registry(),
            context_assembler=_make_mock_context(),
            planner=_make_mock_planner(),
            config=ReActConfig(),
        )
        with pytest.raises(ValueError):
            await loop.resolve_escalation(plan, "task", "nonexistent-id", "edit")


class TestPlanFindNode:
    def test_finds_leaf_branch_and_root(self):
        leaf = PlanNode(description="leaf")
        branch = PlanNode(description="branch")
        branch.set_children([leaf])
        plan = Plan(goal="g", steps=[branch])

        assert plan.find_node(leaf.node_id) is leaf
        assert plan.find_node(branch.node_id) is branch
        assert plan.find_node(plan.root.node_id) is plan.root

    def test_missing_id_returns_none(self):
        plan = Plan(goal="g", steps=[PlanNode(description="leaf")])
        assert plan.find_node("does-not-exist") is None


class TestResolveEscalationPreservesState:
    """Regression coverage for issues found in code review of ticket 06:
    a resume via `resolve_escalation` must not reset an unrelated sibling's
    retry/replan budget, and must not re-inject the task message a second
    time into conversation history."""

    @pytest.mark.asyncio
    async def test_sibling_retry_budget_survives_a_resume(self):
        read_call = '<tool_call>{"name": "read", "arguments": {"path": "x.py"}}</tool_call>'
        backend = _make_mock_backend()
        backend.complete.return_value = CompletionResponse(
            message=Message(role=Role.assistant, content=read_call)
        )
        planner = _make_mock_planner()
        step_one = PlanNode(index=1, description="Step one", status=StepStatus.failed)
        step_two = PlanNode(index=2, description="Step two", status=StepStatus.pending)
        plan = Plan(goal="Test task", steps=[step_one, step_two])

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
            context_assembler=_make_mock_context(),
            planner=planner,
            config=ReActConfig(max_steps=1, max_retries_per_step=2, max_replans=0),
        )
        # Pre-seed step_two's retry counter as if run() had already burned
        # one of its two allowed failures earlier in the same run.
        loop._retry_mgr = RetryManager(max_retries_per_step=2, max_replans=0)
        loop._retry_mgr.record_iteration(step_two.node_id, 1, 1, ["earlier failure"])

        await loop.resolve_escalation(
            plan, "do something", step_one.node_id, "edit", description="Step one (edited)"
        )

        # A resume must not have replaced/reset the RetryManager -- step_two's
        # pre-existing failure count is still tracked, one away from its cap.
        assert loop._retry_mgr.consecutive_failures(step_two.node_id) == 1

    @pytest.mark.asyncio
    async def test_resume_does_not_re_add_task_message(self):
        read_call = '<tool_call>{"name": "read", "arguments": {"path": "x.py"}}</tool_call>'
        finish_call = '<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>'
        backend = _make_mock_backend()
        backend.complete = AsyncMock(
            side_effect=[
                CompletionResponse(message=Message(role=Role.assistant, content=read_call)),
                CompletionResponse(message=Message(role=Role.assistant, content=read_call)),
                CompletionResponse(message=Message(role=Role.assistant, content=finish_call)),
            ]
        )
        planner = _make_mock_planner()
        step_one = PlanNode(index=1, description="Step one", status=StepStatus.pending)
        plan = Plan(goal="Test task", steps=[step_one])
        planner.generate_plan.return_value = plan

        registry = _make_mock_tool_registry()
        read_tool = MagicMock()
        read_tool.name = "read"
        read_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        call_count = 0

        async def flaky_run(**kwargs):
            nonlocal call_count
            call_count += 1
            obj = MagicMock()
            obj.success = call_count > 1
            obj.output = "ok" if obj.success else "fail"
            obj.error = None if obj.success else "fail"
            return obj

        read_tool.run = flaky_run
        finish_tool = MagicMock()
        finish_tool.name = "finish"
        finish_tool.parameters = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        }
        finish_tool.run = AsyncMock(
            return_value=MagicMock(output="Task complete: done", success=True, error=None)
        )
        registry.get.side_effect = lambda name: {
            "read": read_tool, "finish": finish_tool
        }.get(name)

        context = _make_mock_context()
        added_messages: list[Message] = []
        context.conversation.add_message = MagicMock(side_effect=added_messages.append)
        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=registry,
            context_assembler=context,
            planner=planner,
            config=ReActConfig(max_steps=4, max_retries_per_step=1, max_replans=0),
        )
        result = await loop.run("do the task")

        def task_message_count() -> int:
            return sum(
                1 for m in added_messages
                if m.role == Role.user and m.content == "do the task"
            )

        assert task_message_count() == 1

        await loop.resolve_escalation(
            result.plan, "do the task", step_one.node_id, "edit",
            description="Step one (edited)",
        )

        assert task_message_count() == 1


def _make_finishing_tool_registry() -> MagicMock:
    """A tool registry whose `finish` tool actually succeeds, for review-pass
    tests that only care about getting past execution, not tool dispatch."""
    registry = _make_mock_tool_registry()
    finish_tool = MagicMock()
    finish_tool.name = "finish"
    finish_tool.parameters = {"type": "object", "properties": {"summary": {"type": "string"}}}
    finish_tool.run = AsyncMock(
        return_value=MagicMock(output="Task complete: done", success=True, error=None)
    )
    registry.get.side_effect = lambda name: {"finish": finish_tool}.get(name)
    return registry


class _ScriptedReviewUI(NoopUI):
    """Test double returning a fixed sequence of `PlanReviewAction`s from
    `on_plan_review`, one per call -- lets a test script a multi-step
    pre-execution review (edit, then approve) without touching stdin."""

    def __init__(self, actions: list[PlanReviewAction]) -> None:
        self._actions = list(actions)
        self.seen_plans: list[Plan] = []

    async def on_plan_review(self, plan: Plan) -> PlanReviewAction:
        self.seen_plans.append(plan)
        return self._actions.pop(0)


class TestPreExecutionPlanReview:
    """Ticket 07: a one-time whole-tree human approval pass after generation,
    before any execution starts -- distinct from ticket 06's retroactive
    mid-execution escalation handling."""

    @pytest.mark.asyncio
    async def test_headless_noop_ui_skips_review_without_blocking(self):
        finish_call = '<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>'
        backend = _make_mock_backend()
        backend.complete = AsyncMock(
            return_value=CompletionResponse(
                message=Message(role=Role.assistant, content=finish_call)
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = Plan(goal="Test task", steps=[])

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=_make_finishing_tool_registry(),
            context_assembler=_make_mock_context(),
            planner=planner,
            config=ReActConfig(max_steps=4),
        )
        result = await loop.run("do something")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_approve_proceeds_to_execution(self):
        finish_call = '<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>'
        backend = _make_mock_backend()
        backend.complete = AsyncMock(
            return_value=CompletionResponse(
                message=Message(role=Role.assistant, content=finish_call)
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = Plan(goal="Test task", steps=[])
        ui = _ScriptedReviewUI([PlanReviewAction(kind="approve")])

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=_make_finishing_tool_registry(),
            context_assembler=_make_mock_context(),
            planner=planner,
            config=ReActConfig(max_steps=4),
            ui=ui,
        )
        result = await loop.run("do something")
        assert result.success is True
        assert len(ui.seen_plans) == 1

    @pytest.mark.asyncio
    async def test_reject_aborts_before_any_execution(self):
        planner = _make_mock_planner()
        step_one = PlanNode(index=1, description="Step one")
        planner.generate_plan.return_value = Plan(goal="Test task", steps=[step_one])
        backend = _make_mock_backend()
        ui = _ScriptedReviewUI([PlanReviewAction(kind="reject")])

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=_make_finishing_tool_registry(),
            context_assembler=_make_mock_context(),
            planner=planner,
            config=ReActConfig(max_steps=4),
            ui=ui,
        )
        result = await loop.run("do something")

        assert result.success is False
        assert step_one.status == StepStatus.pending
        backend.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_edit_changes_description_before_execution(self):
        finish_call = '<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>'
        backend = _make_mock_backend()
        backend.complete = AsyncMock(
            return_value=CompletionResponse(
                message=Message(role=Role.assistant, content=finish_call)
            )
        )
        planner = _make_mock_planner()
        step_one = PlanNode(index=1, description="Original description")
        planner.generate_plan.return_value = Plan(goal="Test task", steps=[step_one])
        ui = _ScriptedReviewUI(
            [
                PlanReviewAction(
                    kind="edit", node_id=step_one.node_id, description="Edited description"
                ),
                PlanReviewAction(kind="approve"),
            ]
        )

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=_make_finishing_tool_registry(),
            context_assembler=_make_mock_context(),
            planner=planner,
            config=ReActConfig(max_steps=4),
            ui=ui,
        )
        result = await loop.run("do something")

        assert result.success is True
        assert step_one.description == "Edited description"
        assert len(ui.seen_plans) == 2

    @pytest.mark.asyncio
    async def test_mark_leaf_discards_children(self):
        finish_call = '<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>'
        backend = _make_mock_backend()
        backend.complete = AsyncMock(
            return_value=CompletionResponse(
                message=Message(role=Role.assistant, content=finish_call)
            )
        )
        planner = _make_mock_planner()
        branch = PlanNode(index=1, description="Branch goal")
        branch.set_children([PlanNode(index=1, description="Child")])
        plan = Plan(goal="Test task", steps=[branch])
        planner.generate_plan.return_value = plan
        ui = _ScriptedReviewUI(
            [
                PlanReviewAction(kind="mark_leaf", node_id=branch.node_id),
                PlanReviewAction(kind="approve"),
            ]
        )

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=_make_finishing_tool_registry(),
            context_assembler=_make_mock_context(),
            planner=planner,
            config=ReActConfig(max_steps=4),
            ui=ui,
        )
        result = await loop.run("do something")

        assert result.success is True
        assert branch.is_leaf

    @pytest.mark.asyncio
    async def test_redecompose_calls_planner_replan_subtree(self):
        finish_call = '<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>'
        backend = _make_mock_backend()
        backend.complete = AsyncMock(
            return_value=CompletionResponse(
                message=Message(role=Role.assistant, content=finish_call)
            )
        )
        planner = _make_mock_planner()
        node = PlanNode(index=1, description="Unclear step")
        plan = Plan(goal="Test task", steps=[node])
        planner.generate_plan.return_value = plan
        ui = _ScriptedReviewUI(
            [
                PlanReviewAction(kind="redecompose", node_id=node.node_id, guidance="be specific"),
                PlanReviewAction(kind="approve"),
            ]
        )

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=_make_finishing_tool_registry(),
            context_assembler=_make_mock_context(),
            planner=planner,
            config=ReActConfig(max_steps=4),
            ui=ui,
        )
        result = await loop.run("do something")

        assert result.success is True
        planner.replan_subtree.assert_awaited_once_with(
            plan, node, "be specific", is_failure=False
        )

    @pytest.mark.asyncio
    async def test_unresolved_uninterpretable_action_does_not_hang(self):
        """A UI double that returns something other than a recognized
        PlanReviewAction kind must not spin the review loop forever --
        bounded by ReActLoop._MAX_PLAN_REVIEW_ITERATIONS, falling back to
        auto-approve."""
        finish_call = '<tool_call>{"name": "finish", "arguments": {"summary": "done"}}</tool_call>'
        backend = _make_mock_backend()
        backend.complete = AsyncMock(
            return_value=CompletionResponse(
                message=Message(role=Role.assistant, content=finish_call)
            )
        )
        planner = _make_mock_planner()
        planner.generate_plan.return_value = Plan(goal="Test task", steps=[])
        ui = MagicMock(spec=NoopUI)

        loop = ReActLoop(
            llm_backend=backend,
            tool_registry=_make_finishing_tool_registry(),
            context_assembler=_make_mock_context(),
            planner=planner,
            config=ReActConfig(max_steps=4),
            ui=ui,
        )
        result = await loop.run("do something")
        assert result.success is True
        assert ui.on_plan_review.await_count == ReActLoop._MAX_PLAN_REVIEW_ITERATIONS
