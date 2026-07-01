from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chef_human.llm.backend import CompletionResponse, Message, Role


@pytest.fixture
def mock_backend():
    backend = MagicMock()
    backend.complete = AsyncMock()
    backend.model_name = "mock-model"
    backend.context_length = 4096
    return backend


class TestCreateAgent:
    def test_wires_all_components(self):
        with (
            patch("chef_human.llm.create_backend"),
            patch("chef_human.agent.create_context_assembler") as mock_ctx_factory,
            patch("chef_human.tools.create_tool_registry"),
            patch("chef_human.agent.planner.Planner"),
            patch("chef_human.agent.react_loop.ReActLoop") as mock_loop_factory,
            patch("chef_human.ui.protocol.NoopUI"),
        ):
            from chef_human.agent import create_agent

            mock_ctx = MagicMock()
            mock_ctx_factory.return_value = mock_ctx

            loop, context = create_agent(max_steps=10)

            mock_ctx_factory.assert_called_once_with(workspace_root=None)
            mock_loop_factory.assert_called_once()
            _, kwargs = mock_loop_factory.call_args
            assert kwargs["config"].max_steps == 10
            assert context is mock_ctx

    def test_workspace_root_forwarded(self):
        with (
            patch("chef_human.llm.create_backend"),
            patch("chef_human.agent.create_context_assembler") as mock_ctx_factory,
            patch("chef_human.tools.create_tool_registry"),
            patch("chef_human.agent.planner.Planner"),
            patch("chef_human.agent.react_loop.ReActLoop"),
            patch("chef_human.ui.protocol.NoopUI"),
        ):
            from chef_human.agent import create_agent

            mock_ctx = MagicMock()
            mock_ctx_factory.return_value = mock_ctx

            create_agent(workspace_root="/custom/path")

            mock_ctx_factory.assert_called_once_with(workspace_root="/custom/path")

    def test_debug_tui_uses_debug_tui_class(self):
        with (
            patch("chef_human.llm.create_backend"),
            patch("chef_human.agent.create_context_assembler"),
            patch("chef_human.tools.create_tool_registry"),
            patch("chef_human.agent.planner.Planner"),
            patch("chef_human.agent.react_loop.ReActLoop"),
            patch("chef_human.ui.debug_tui.DebugTUI") as mock_tui,
            patch("chef_human.ui.protocol.NoopUI") as mock_noop,
        ):
            from chef_human.agent import create_agent

            create_agent(debug_tui=True)

            mock_tui.assert_called_once()
            mock_noop.assert_not_called()

    def test_no_debug_tui_uses_noop_ui(self):
        with (
            patch("chef_human.llm.create_backend"),
            patch("chef_human.agent.create_context_assembler"),
            patch("chef_human.tools.create_tool_registry"),
            patch("chef_human.agent.planner.Planner"),
            patch("chef_human.agent.react_loop.ReActLoop"),
            patch("chef_human.ui.debug_tui.DebugTUI") as mock_tui,
            patch("chef_human.ui.protocol.NoopUI") as mock_noop,
        ):
            from chef_human.agent import create_agent

            create_agent(debug_tui=False)

            mock_noop.assert_called_once()
            mock_tui.assert_not_called()


class TestFullLoopIntegration:
    """End-to-end integration test with all components mocked.

    Validates that a full task flows through plan -> reason -> tool call -> result -> finish.
    """

    TOOL_CALLS = """<tool_call>{"name": "read", "arguments": {"path": "test.txt"}}</tool_call>"""
    ALL_DONE = """The task is complete. <finish>All done</finish>"""

    @pytest.mark.asyncio
    async def test_full_run_cycle(self, mock_backend):
        responses = iter([
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content='["Read the file", "Report the results"]',
                )
            ),
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content=f"I need to read the file first.{self.TOOL_CALLS}",
                )
            ),
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content=self.ALL_DONE,
                )
            ),
        ])
        mock_backend.complete.side_effect = responses

        from chef_human.agent import create_context_assembler
        from chef_human.agent.planner import Planner
        from chef_human.agent.react_loop import ReActConfig, ReActLoop
        from chef_human.tools import create_tool_registry
        from chef_human.ui.protocol import NoopUI

        context = create_context_assembler()
        tool_registry = create_tool_registry(context.workspace)
        planner = Planner(llm_backend=mock_backend)
        config = ReActConfig(max_steps=5)
        ui = NoopUI()

        loop = ReActLoop(
            llm_backend=mock_backend,
            tool_registry=tool_registry,
            context_assembler=context,
            planner=planner,
            config=config,
            ui=ui,
        )

        result = await loop.run("Read test.txt and summarize")

        assert result.success is True
        assert result.steps_taken == 2  # tool call + finish
        assert "All done" in result.message

    @pytest.mark.asyncio
    async def test_handles_tool_error_then_recovers(self, mock_backend):
        responses = iter([
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content='["Read the file"]',
                )
            ),
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content=f"I need to read the file.{self.TOOL_CALLS}",
                )
            ),
            CompletionResponse(
                message=Message(
                    role=Role.assistant,
                    content=self.ALL_DONE,
                )
            ),
        ])
        mock_backend.complete.side_effect = responses

        from chef_human.agent import create_context_assembler
        from chef_human.agent.planner import Planner
        from chef_human.agent.react_loop import ReActConfig, ReActLoop
        from chef_human.tools import create_tool_registry
        from chef_human.ui.protocol import NoopUI

        context = create_context_assembler()
        tool_registry = create_tool_registry(context.workspace)

        # Register a read tool that fails once then succeeds
        read_tool = MagicMock()
        read_tool.name = "read"
        read_tool.description = "Read a file"
        read_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        read_tool.run = AsyncMock(side_effect=[
            MagicMock(output="", success=False, error="File not found"),
            MagicMock(output="file content here", success=True, error=None),
        ])
        tool_registry.register(read_tool)

        planner = Planner(llm_backend=mock_backend)
        config = ReActConfig(max_steps=5, max_retries_per_step=2)
        ui = NoopUI()

        loop = ReActLoop(
            llm_backend=mock_backend,
            tool_registry=tool_registry,
            context_assembler=context,
            planner=planner,
            config=config,
            ui=ui,
        )

        result = await loop.run("Read and report")

        assert result.success is True
        # First attempt fails, retry succeeds, then finish
        assert result.steps_taken >= 1
