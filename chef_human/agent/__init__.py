from __future__ import annotations

from typing import TYPE_CHECKING

from chef_human.agent.context import ContextAssembler, ContextConfig, ContextManager
from chef_human.agent.file_context import FileContextManager
from chef_human.agent.planner import Plan, PlanStep, Planner, StepStatus
from chef_human.agent.repo_map import RepoMap
from chef_human.agent.symbols.extractor import RegexExtractor, create_extractor
from chef_human.agent.workspace import WorkspaceManager
from chef_human.config import settings
from chef_human.llm.tokenizer import create_tokenizer

if TYPE_CHECKING:
    from chef_human.agent.react_loop import ReActLoop


def create_context_assembler(workspace_root: str | None = None) -> ContextAssembler:
    tokenizer = create_tokenizer(settings.ollama_model)
    root = workspace_root or settings.workspace or None
    workspace = WorkspaceManager(root=root)
    config = ContextConfig(
        max_tokens=settings.max_context_tokens,
        max_response_tokens=settings.max_response_tokens,
    )
    conversation = ContextManager(config=config, tokenizer=tokenizer)
    file_ctx = FileContextManager(
        workspace=workspace,
        tokenizer=tokenizer,
    )
    repo_map = RepoMap(workspace=workspace, tokenizer=tokenizer)
    return ContextAssembler(
        conversation=conversation,
        workspace=workspace,
        file_context=file_ctx,
        repo_map=repo_map,
    )


def create_agent(
    debug_tui: bool = False,
    max_steps: int = 25,
    workspace_root: str | None = None,
) -> tuple[ReActLoop, ContextAssembler]:
    from chef_human.agent.planner import Planner
    from chef_human.agent.react_loop import ReActConfig, ReActLoop
    from chef_human.llm import create_backend
    from chef_human.tools import create_tool_registry
    from chef_human.ui.debug_tui import DebugTUI
    from chef_human.ui.protocol import NoopUI

    backend = create_backend()
    context = create_context_assembler(workspace_root=workspace_root)
    tool_registry = create_tool_registry(context.workspace)
    planner = Planner(llm_backend=backend)
    config = ReActConfig(max_steps=max_steps)
    ui = DebugTUI() if debug_tui else NoopUI()

    loop = ReActLoop(
        llm_backend=backend,
        tool_registry=tool_registry,
        context_assembler=context,
        planner=planner,
        config=config,
        ui=ui,
    )
    return loop, context


__all__ = [
    "ContextAssembler",
    "ContextConfig",
    "ContextManager",
    "FileContextManager",
    "Plan",
    "PlanStep",
    "Planner",
    "RepoMap",
    "StepStatus",
    "WorkspaceManager",
    "create_agent",
    "create_context_assembler",
]
