from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chef_human.agent.rag.chunker import CodeChunker
from chef_human.agent.rag.retriever import RAGRetriever
from chef_human.agent.rag.store import VectorStore
from chef_human.agent.workspace import WorkspaceManager
from chef_human.llm.tokenizer import create_tokenizer


class SimpleEmbedder:
    _dim = 8

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


def test_end_to_end(workspace: WorkspaceManager, tmp_path: Path):
    tokenizer = create_tokenizer()
    chunker = CodeChunker(tokenizer=tokenizer, target_tokens=50, overlap_tokens=10)
    embedder = SimpleEmbedder()
    store = VectorStore(dimension=embedder.dimension)
    retriever = RAGRetriever(
        chunker=chunker,
        embedder=embedder,
        store=store,
        workspace=workspace,
        tokenizer=tokenizer,
    )

    (tmp_path / "math_utils.py").write_text(
        "def add(a, b):\n    return a + b\n\n"
        "def multiply(a, b):\n    return a * b\n"
    )
    (tmp_path / "string_utils.py").write_text(
        "def uppercase(s):\n    return s.upper()\n\n"
        "def lowercase(s):\n    return s.lower()\n"
    )

    files = list(workspace.list_files(max_depth=2))
    count = retriever.build(files)
    assert count >= 2

    results = retriever.retrieve("math addition", top_k=2)
    assert len(results) >= 1


def test_persistence_roundtrip(workspace: WorkspaceManager, tmp_path: Path):
    tokenizer = create_tokenizer()
    chunker = CodeChunker(tokenizer=tokenizer, target_tokens=50, overlap_tokens=10)
    embedder = SimpleEmbedder()
    store = VectorStore(dimension=embedder.dimension, index_dir=tmp_path / ".chef-human")
    retriever = RAGRetriever(
        chunker=chunker,
        embedder=embedder,
        store=store,
        workspace=workspace,
        tokenizer=tokenizer,
    )

    f = tmp_path / "utils.py"
    f.write_text("def foo():\n    return 1\n" * 10)
    retriever.build([f])
    retriever._store.save()

    loaded_store = VectorStore.load(tmp_path / ".chef-human", dimension=embedder.dimension)
    assert loaded_store is not None
    assert len(loaded_store) == len(store)

    loaded_retriever = RAGRetriever(
        chunker=chunker,
        embedder=embedder,
        store=loaded_store,
        workspace=workspace,
        tokenizer=tokenizer,
    )
    results = loaded_retriever.retrieve("foo", top_k=1)
    assert len(results) >= 1


def test_empty_workspace(workspace: WorkspaceManager):
    tokenizer = create_tokenizer()
    chunker = CodeChunker(tokenizer=tokenizer)
    embedder = SimpleEmbedder()
    store = VectorStore(dimension=embedder.dimension)
    retriever = RAGRetriever(
        chunker=chunker,
        embedder=embedder,
        store=store,
        workspace=workspace,
        tokenizer=tokenizer,
    )
    retriever.build([])
    assert retriever.total_chunks == 0
    assert retriever.retrieve("anything") == []
