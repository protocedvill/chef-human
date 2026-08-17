from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chef_human.agent.rag.chunker import CodeChunker
from chef_human.agent.rag.retriever import RAGRetriever
from chef_human.agent.rag.store import VectorStore
from chef_human.agent.workspace import WorkspaceManager
from chef_human.llm.tokenizer import create_tokenizer


class MockEmbedder:
    _dim = 4
    _counter = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_single(t) for t in texts]

    def embed_single(self, text: str) -> list[float]:
        rng = np.random.RandomState(hash(text) % (2**31))
        vec = rng.randn(self._dim).tolist()
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec]

    @property
    def dimension(self) -> int:
        return self._dim


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceManager:
    return WorkspaceManager(root=str(tmp_path))


@pytest.fixture
def tokenizer():
    return create_tokenizer()


@pytest.fixture
def retriever(workspace: WorkspaceManager, tokenizer, tmp_path: Path) -> RAGRetriever:
    chunker = CodeChunker(tokenizer=tokenizer, target_tokens=50, overlap_tokens=10)
    embedder = MockEmbedder()
    store = VectorStore(dimension=embedder.dimension)
    return RAGRetriever(
        chunker=chunker,
        embedder=embedder,
        store=store,
        workspace=workspace,
        tokenizer=tokenizer,
    )


class TestRetrieverBuild:
    def test_build_with_files(self, retriever: RAGRetriever, tmp_path: Path):
        f1 = tmp_path / "a.py"
        f1.write_text("def foo():\n    pass\n" * 50)
        f2 = tmp_path / "b.py"
        f2.write_text("class Bar:\n    pass\n" * 50)
        count = retriever.build([f1, f2])
        assert count > 0
        assert retriever.is_built
        assert retriever.total_chunks > 0

    def test_build_empty_list(self, retriever: RAGRetriever):
        count = retriever.build([])
        assert count == 0
        assert retriever.is_built
        assert retriever.total_chunks == 0

    def test_build_with_nonexistent_file(self, retriever: RAGRetriever, tmp_path: Path):
        count = retriever.build([tmp_path / "nonexistent.py"])
        assert count == 0


class TestRetrieverUpdate:
    def test_update_before_build_falls_back_to_build(self, retriever: RAGRetriever, tmp_path: Path):
        f = tmp_path / "a.py"
        f.write_text("def foo():\n    pass\n" * 20)
        count = retriever.update([f])
        assert count > 0
        assert retriever.is_built

    def test_update_reindexes_changed_file_without_duplicating_or_touching_others(
        self, retriever: RAGRetriever, tmp_path: Path
    ):
        a = tmp_path / "a.py"
        a.write_text("def foo():\n    pass\n" * 20)
        b = tmp_path / "b.py"
        b.write_text("class Bar:\n    pass\n" * 20)
        retriever.build([a, b])
        b_chunks_before = sum(
            1 for r in retriever.retrieve("Bar class", top_k=50) if r.file_path == str(b)
        )

        a.write_text("def foo():\n    return 1\n" * 20)
        retriever.update([a])

        results = retriever.retrieve("foo function bar class", top_k=50)
        a_chunks = [r for r in results if r.file_path == str(a)]
        b_chunks = [r for r in results if r.file_path == str(b)]

        # b.py untouched by the incremental update to a.py.
        assert len(b_chunks) == b_chunks_before
        # a.py's old ("pass") content shouldn't linger as stale duplicates
        # alongside the new ("return 1") content.
        assert all("return 1" in c.content for c in a_chunks)
        assert not any("pass" in c.content for c in a_chunks)

    def test_update_removes_chunks_for_deleted_file(self, retriever: RAGRetriever, tmp_path: Path):
        a = tmp_path / "a.py"
        a.write_text("def only_here():\n    pass\n" * 20)
        retriever.build([a])
        assert retriever.total_chunks > 0

        a.unlink()
        count = retriever.update([a])

        assert count == 0
        assert retriever.total_chunks == 0

    def test_update_does_not_clear_unrelated_files(self, retriever: RAGRetriever, tmp_path: Path):
        """The whole point of update() over build(): untouched files must
        survive, unlike build()'s full store.clear()."""
        a = tmp_path / "a.py"
        a.write_text("def foo():\n    pass\n" * 20)
        b = tmp_path / "b.py"
        b.write_text("def bar():\n    pass\n" * 20)
        retriever.build([a, b])

        c = tmp_path / "c.py"
        c.write_text("def baz():\n    pass\n" * 20)
        retriever.update([c])

        results = retriever.retrieve("foo function", top_k=20)
        assert any(r.file_path == str(a) for r in results)
        assert any(r.file_path == str(b) for r in results)


class TestRetrieverSearch:
    def test_retrieve_returns_chunks(self, retriever: RAGRetriever, tmp_path: Path):
        f = tmp_path / "test.py"
        f.write_text("def foo():\n    pass\n" * 30)
        retriever.build([f])
        results = retriever.retrieve("foo function", top_k=2)
        assert len(results) >= 1
        assert results[0].file_path == str(f)

    def test_retrieve_empty_store(self, retriever: RAGRetriever):
        results = retriever.retrieve("anything")
        assert results == []

    def test_retrieve_k_respected(self, retriever: RAGRetriever, tmp_path: Path):
        f = tmp_path / "test.py"
        f.write_text("def foo():\n    pass\n" * 100)
        retriever.build([f])
        results = retriever.retrieve("test", top_k=3)
        assert len(results) <= 3

    def test_retrieve_not_built(self, retriever: RAGRetriever):
        results = retriever.retrieve("test")
        assert results == []


class TestFormatForPrompt:
    def test_format_respects_budget(self, retriever: RAGRetriever, tmp_path: Path):
        f = tmp_path / "test.py"
        f.write_text("def foo():\n    pass\n" * 10)
        retriever.build([f])
        chunks = retriever.retrieve("foo", top_k=5)
        formatted = retriever.format_for_prompt(chunks, max_tokens=1000)
        assert len(formatted) > 0
        tight = retriever.format_for_prompt(chunks, max_tokens=5)
        assert len(tight) == 0 or retriever._tokenizer.count(tight) <= 5

    def test_format_empty(self, retriever: RAGRetriever):
        formatted = retriever.format_for_prompt([], max_tokens=100)
        assert formatted == ""

    def test_format_contains_content(self, retriever: RAGRetriever, tmp_path: Path):
        f = tmp_path / "test.py"
        f.write_text("def foo():\n    return 42\n")
        retriever.build([f])
        chunks = retriever.retrieve("foo", top_k=1)
        formatted = retriever.format_for_prompt(chunks, max_tokens=500)
        assert "def foo" in formatted
        assert str(f) in formatted
