from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from chef_human.agent.context import ContextAssembler, ContextConfig, ContextManager
from chef_human.agent.file_context import FileContextManager
from chef_human.agent.planner import Plan, PlanStep, Planner, StepStatus
from chef_human.agent.repo_map import RepoMap
from chef_human.agent.symbols.extractor import CompositeExtractor
from chef_human.agent.workspace import WorkspaceManager
from chef_human.config import settings
from chef_human.llm.tokenizer import Tokenizer, create_tokenizer

if TYPE_CHECKING:
    from chef_human.agent.react_loop import ReActLoop

logger = logging.getLogger(__name__)


def _build_rag_context_assembler(
    workspace: WorkspaceManager,
    tokenizer: Tokenizer,
    file_ctx: FileContextManager,
    repo_map: RepoMap,
    conversation: ContextManager,
) -> ContextAssembler:
    from chef_human.agent.rag.chunker import CodeChunker
    from chef_human.agent.rag.retriever import RAGRetriever
    from chef_human.agent.rag.store import VectorStore
    from chef_human.agent.symbols.index import SymbolIndex
    from chef_human.llm.embeddings import EmbeddingsBackend

    embedder = EmbeddingsBackend(settings.embed_model)
    chunker = CodeChunker(
        tokenizer=tokenizer,
        target_tokens=settings.rag_chunk_tokens,
        overlap_tokens=settings.rag_chunk_overlap,
    )
    store = VectorStore(
        dimension=embedder.dimension,
        index_dir=workspace.root / settings.rag_index_dir,
    )
    symbol_index = SymbolIndex(workspace=workspace, extractor=CompositeExtractor())
    rag_retriever = RAGRetriever(
        chunker=chunker,
        embedder=embedder,
        store=store,
        workspace=workspace,
        tokenizer=tokenizer,
        symbol_index=symbol_index,
    )

    return ContextAssembler(
        conversation=conversation,
        workspace=workspace,
        file_context=file_ctx,
        repo_map=repo_map,
        rag_retriever=rag_retriever,
    )


def _build_symbol_context_assembler(
    workspace: WorkspaceManager,
    tokenizer: Tokenizer,
    file_ctx: FileContextManager,
    repo_map: RepoMap,
    conversation: ContextManager,
    index_on_init: bool,
) -> ContextAssembler:
    from chef_human.agent.symbols.dependencies import DependencyGraph
    from chef_human.agent.symbols.index import SymbolIndex
    from chef_human.agent.symbols.retriever import SymbolRetriever

    extractor = CompositeExtractor()
    symbol_index = SymbolIndex(workspace=workspace, extractor=extractor)
    dep_graph = DependencyGraph(symbol_index)
    symbol_retriever = SymbolRetriever(index=symbol_index, file_context=file_ctx)

    index_loaded = False

    if index_on_init:
        index_path = workspace.root / ".chef-human" / "index.json"
        deps_path = workspace.root / ".chef-human" / "deps.json"

        if settings.persist_index:
            loaded = SymbolIndex.load(index_path, workspace, extractor)
            if loaded is not None:
                symbol_index = loaded
                symbol_retriever = SymbolRetriever(index=symbol_index, file_context=file_ctx)
                dep_graph = DependencyGraph(symbol_index)
                index_loaded = True
                logger.info(
                    "Loaded symbol index from %s (%d symbols)",
                    index_path,
                    symbol_index.total_symbols(),
                )

        if not index_loaded:
            files = workspace.list_files(max_depth=10)[:settings.max_index_files]
            symbol_index.build(files=files)
            if settings.persist_index:
                symbol_index.save(index_path)

        if symbol_index.total_symbols() > 0:
            if index_loaded and settings.persist_index:
                deps_loaded = DependencyGraph.load(deps_path, workspace.root, symbol_index)
                if deps_loaded is not None:
                    dep_graph = deps_loaded
                    logger.info("Loaded dependency graph from %s", deps_path)
                else:
                    dep_graph.build()
                    dep_graph.save(deps_path, workspace_root=workspace.root)
            else:
                dep_graph.build()
                if settings.persist_index:
                    dep_graph.save(deps_path, workspace_root=workspace.root)

    return ContextAssembler(
        conversation=conversation,
        workspace=workspace,
        file_context=file_ctx,
        repo_map=repo_map,
        symbol_index=symbol_index,
        dep_graph=dep_graph,
        symbol_retriever=symbol_retriever,
    )


def create_context_assembler(
    workspace_root: str | None = None,
    index_on_init: bool = True,
) -> ContextAssembler:
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

    files = workspace.list_files(max_depth=10)
    if len(files) > settings.max_index_files:
        return _build_rag_context_assembler(
            workspace=workspace,
            tokenizer=tokenizer,
            file_ctx=file_ctx,
            repo_map=repo_map,
            conversation=conversation,
        )
    return _build_symbol_context_assembler(
        workspace=workspace,
        tokenizer=tokenizer,
        file_ctx=file_ctx,
        repo_map=repo_map,
        conversation=conversation,
        index_on_init=index_on_init,
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
    tool_registry = create_tool_registry(
        workspace=context.workspace,
        symbol_index=context.symbol_index,
        file_context=context.file_context,
    )
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
