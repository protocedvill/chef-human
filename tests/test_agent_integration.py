from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import patch

from chef_human.agent import create_context_assembler
from chef_human.agent.context import ContextAssembler, ContextConfig, ContextManager, Role
from chef_human.agent.file_context import FileContextManager
from chef_human.agent.repo_map import RepoMap
from chef_human.agent.symbols.extractor import CompositeExtractor, RegexExtractor, create_extractor
from chef_human.agent.workspace import WorkspaceManager
from chef_human.config import settings
from chef_human.llm.tokenizer import ApproxTokenizer, create_tokenizer
from chef_human.tools import create_tool_registry
from chef_human.tools.view_diff import ViewDiffTool


# ---------------------------------------------------------------------------
# Factory sanity checks
# ---------------------------------------------------------------------------

class TestCreateContextAssembler:
    def test_creates_without_error(self):
        ca = create_context_assembler()
        assert isinstance(ca, ContextAssembler)

    def test_assemble_does_not_raise(self):
        ca = create_context_assembler()
        result = ca.assemble(system_prompt="You are a helpful assistant.")
        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0].role == Role.system

    def test_accepts_tool_definitions(self):
        ca = create_context_assembler()
        result = ca.assemble(system_prompt="Be helpful.")
        assert "Be helpful." in result[0].content


# ---------------------------------------------------------------------------
# End-to-end with temporary workspace
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_with_real_files(self, tmp_path: Path):
        (tmp_path / "greet.py").write_text("def hello():\n    print('hi')")
        (tmp_path / "utils.py").write_text("class Helper:\n    pass")
        tokenizer = ApproxTokenizer()
        workspace = WorkspaceManager(root=tmp_path)
        file_ctx = FileContextManager(workspace=workspace, tokenizer=tokenizer)
        file_ctx.get("greet.py")
        file_ctx.get("utils.py")
        repo_map = RepoMap(workspace=workspace, tokenizer=tokenizer)
        config = ContextConfig(max_tokens=1000, max_response_tokens=100, summary_tokens=50)
        conversation = ContextManager(config=config, tokenizer=tokenizer)
        conversation.add_message(type("FakeMessage", (), {"role": "user", "content": "hello"})())
        ca = ContextAssembler(
            conversation=conversation,
            workspace=workspace,
            file_context=file_ctx,
            repo_map=repo_map,
        )
        result = ca.assemble(system_prompt="You are a bot.")
        assert len(result) >= 3
        assert result[0].role == Role.system
        assert "greet.py" in str(result)

    def test_empty_workspace(self, tmp_path: Path):
        tokenizer = ApproxTokenizer()
        workspace = WorkspaceManager(root=tmp_path)
        file_ctx = FileContextManager(workspace=workspace, tokenizer=tokenizer)
        repo_map = RepoMap(workspace=workspace, tokenizer=tokenizer)
        config = ContextConfig(max_tokens=500, max_response_tokens=50, summary_tokens=25)
        conversation = ContextManager(config=config, tokenizer=tokenizer)
        ca = ContextAssembler(
            conversation=conversation,
            workspace=workspace,
            file_context=file_ctx,
            repo_map=repo_map,
        )
        result = ca.assemble(system_prompt="sys")
        system_msgs = [m for m in result if m.role == Role.system]
        repo_msgs = [m for m in result if "Repository Structure" in (m.content or "")]
        file_msgs = [m for m in result if "File Context" in (m.content or "")]
        assert len(repo_msgs) == 0
        assert len(file_msgs) == 0
        assert len(system_msgs) >= 1


# ---------------------------------------------------------------------------
# Factory edge cases
# ---------------------------------------------------------------------------

class TestFactoryEdgeCases:
    def test_create_extractor_returns_composite(self):
        extractor = create_extractor()
        assert isinstance(extractor, CompositeExtractor)

    def test_create_tokenizer_fallback(self):
        tokenizer = create_tokenizer()
        assert isinstance(tokenizer, ApproxTokenizer)


# ---------------------------------------------------------------------------
# RAG context assembler wiring (large-repo path)
# ---------------------------------------------------------------------------

class _FakeEmbeddingsBackend:
    """Stand-in for EmbeddingsBackend that avoids a real sentence-transformers
    model load/download -- these tests only care about the build/update
    wiring, not embedding quality."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        return 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_single(t) for t in texts]

    def embed_single(self, text: str) -> list[float]:
        import hashlib
        h = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        return [((h >> (8 * i)) % 97) / 97.0 + 1e-6 for i in range(4)]


class TestRagContextAssemblerWiring:
    def test_builds_rag_index_on_init(self, tmp_path: Path):
        """Regression test: _build_rag_context_assembler used to construct
        RAGRetriever around an empty store and never call build() or load a
        persisted one -- retrieve() would return [] for the whole session."""
        (tmp_path / "a.py").write_text("def hello_world():\n    pass\n" * 5)
        patched = dataclasses.replace(settings, max_index_files=0)
        with (
            patch("chef_human.agent.settings", patched),
            patch("chef_human.llm.embeddings.EmbeddingsBackend", _FakeEmbeddingsBackend),
        ):
            ca = create_context_assembler(workspace_root=str(tmp_path))
        try:
            assert ca._rag_retriever is not None
            assert ca._rag_retriever.is_built
            assert ca._rag_retriever.total_chunks > 0
        finally:
            if ca.file_watcher is not None:
                ca.file_watcher.stop()

    def test_rag_watcher_updates_index_on_change(self, tmp_path: Path):
        import time

        (tmp_path / "a.py").write_text("def original_name():\n    pass\n" * 5)
        patched = dataclasses.replace(
            settings, max_index_files=0, watch_files=True, watch_interval=0.05,
        )
        with (
            patch("chef_human.agent.settings", patched),
            patch("chef_human.llm.embeddings.EmbeddingsBackend", _FakeEmbeddingsBackend),
        ):
            ca = create_context_assembler(workspace_root=str(tmp_path))
        try:
            assert ca.file_watcher is not None
            before = ca._rag_retriever.total_chunks

            time.sleep(0.1)
            (tmp_path / "b.py").write_text("def a_brand_new_function():\n    pass\n" * 5)
            time.sleep(0.3)

            assert ca._rag_retriever.total_chunks > before
        finally:
            if ca.file_watcher is not None:
                ca.file_watcher.stop()


# ---------------------------------------------------------------------------
# Symbol extractor integration with repo map
# ---------------------------------------------------------------------------

class TestSymbolExtractorIntegration:
    def test_extracts_from_python(self):
        extractor = RegexExtractor()
        symbols = extractor.extract("test.py", "def f():\n    pass\n\nclass C:\n    pass")
        assert len(symbols) == 2
        kinds = [s.kind for s in symbols]
        assert "function" in kinds
        assert "class" in kinds

    def test_extracts_from_rust(self):
        extractor = RegexExtractor()
        symbols = extractor.extract("lib.rs", "pub fn foo() {}\nstruct Bar {}")
        assert len(symbols) == 2
        assert symbols[0].name == "foo"
        assert symbols[1].name == "Bar"


# ---------------------------------------------------------------------------
# FileWatcher wiring (keeps the symbol index from going stale mid-session)
# ---------------------------------------------------------------------------

class TestFileWatcherWiring:
    def test_no_watcher_by_default(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("def f():\n    pass\n")
        ca = create_context_assembler(workspace_root=str(tmp_path))
        assert ca.file_watcher is None

    def test_watcher_started_when_enabled(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("def f():\n    pass\n")
        patched = dataclasses.replace(settings, watch_files=True, watch_interval=0.05)
        with patch("chef_human.agent.settings", patched):
            ca = create_context_assembler(workspace_root=str(tmp_path))
        try:
            assert ca.file_watcher is not None
            assert ca.file_watcher._running
        finally:
            if ca.file_watcher is not None:
                ca.file_watcher.stop()

    def test_watcher_refreshes_symbol_index_on_change(self, tmp_path: Path):
        import time

        (tmp_path / "a.py").write_text("def original():\n    pass\n")
        patched = dataclasses.replace(settings, watch_files=True, watch_interval=0.05)
        with patch("chef_human.agent.settings", patched):
            ca = create_context_assembler(workspace_root=str(tmp_path))
        try:
            assert ca.symbol_index.lookup("original")
            time.sleep(0.1)
            (tmp_path / "a.py").write_text("def renamed():\n    pass\n")
            time.sleep(0.3)
            assert ca.symbol_index.lookup("renamed")
        finally:
            if ca.file_watcher is not None:
                ca.file_watcher.stop()


# ---------------------------------------------------------------------------
# Tool registry integration (Phase 3.3)
# ---------------------------------------------------------------------------

class TestDiffToolRegistry:
    def test_view_diff_tool_registered(self, tmp_path: Path):
        ws = WorkspaceManager(root=tmp_path)
        registry = create_tool_registry(ws)
        tool = registry.get("view_diff")
        assert tool is not None
        assert isinstance(tool, ViewDiffTool)

    def test_edit_tool_has_diff_store(self, tmp_path: Path):
        ws = WorkspaceManager(root=tmp_path)
        registry = create_tool_registry(ws)
        tool = registry.get("edit")
        assert tool is not None
        assert hasattr(tool, "_diff_store")
        assert tool._diff_store is not None

    def test_write_tool_has_diff_store(self, tmp_path: Path):
        ws = WorkspaceManager(root=tmp_path)
        registry = create_tool_registry(ws)
        tool = registry.get("write")
        assert tool is not None
        assert hasattr(tool, "_diff_store")
        assert tool._diff_store is not None

    def test_diff_store_shared_across_tools(self, tmp_path: Path):
        ws = WorkspaceManager(root=tmp_path)
        registry = create_tool_registry(ws)
        edit_tool = registry.get("edit")
        write_tool = registry.get("write")
        view_tool = registry.get("view_diff")
        assert edit_tool._diff_store is write_tool._diff_store
        assert edit_tool._diff_store is view_tool._store
