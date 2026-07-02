from __future__ import annotations

from pathlib import Path

import pytest

from chef_human.agent.file_context import FileContextManager
from chef_human.agent.symbols.extractor import CompositeExtractor
from chef_human.agent.symbols.index import SymbolIndex
from chef_human.agent.symbols.retriever import SymbolRetriever, _NOISE
from chef_human.agent.workspace import WorkspaceManager
from chef_human.llm.tokenizer import create_tokenizer


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceManager:
    return WorkspaceManager(root=str(tmp_path))


@pytest.fixture
def index(workspace: WorkspaceManager, tmp_path: Path) -> SymbolIndex:
    extractor = CompositeExtractor()
    idx = SymbolIndex(workspace=workspace, extractor=extractor)

    lib = tmp_path / "mylib.py"
    lib.write_text(
        "class MyClass:\n"
        "    def method(self):\n"
        "        pass\n"
        "\n"
        "def my_function():\n"
        "    pass\n"
        "\n"
        "MY_CONSTANT = 42\n"
    )

    idx.build(files=[lib])
    return idx


@pytest.fixture
def file_ctx(workspace: WorkspaceManager) -> FileContextManager:
    tokenizer = create_tokenizer()
    return FileContextManager(workspace=workspace, tokenizer=tokenizer)


@pytest.fixture
def retriever(index: SymbolIndex, file_ctx: FileContextManager) -> SymbolRetriever:
    return SymbolRetriever(index=index, file_context=file_ctx)


class TestDetectSymbolReferences:
    def test_finds_known_symbol(self, retriever: SymbolRetriever):
        names = retriever.detect_symbol_references("Use MyClass to do something")
        assert "MyClass" in names

    def test_skips_noise_words(self, retriever: SymbolRetriever):
        names = retriever.detect_symbol_references("This and That are the same")
        assert "This" not in names
        assert "That" not in names

    def test_skips_unknown_symbols(self, retriever: SymbolRetriever):
        names = retriever.detect_symbol_references("UnknownSymbol does not exist")
        assert "UnknownSymbol" not in names

    def test_detects_multiple_symbols(self, retriever: SymbolRetriever):
        names = retriever.detect_symbol_references(
            "MyClass and my_function"
        )
        assert "MyClass" in names

    def test_dotted_name_detection(self, retriever: SymbolRetriever):
        names = retriever.detect_symbol_references("Use MyClass.MyClass from the lib")
        assert "MyClass.MyClass" in names

    def test_skips_recently_fetched(self, retriever: SymbolRetriever):
        retriever._recently_fetched.add("MyClass")
        names = retriever.detect_symbol_references("Use MyClass")
        assert "MyClass" not in names

    def test_no_symbols_in_empty_text(self, retriever: SymbolRetriever):
        names = retriever.detect_symbol_references("hello world")
        assert names == []

    def test_noise_set_coverage(self):
        assert "The" in _NOISE
        assert "This" in _NOISE
        assert "File" in _NOISE


class TestRetrieve:
    def test_retrieve_existing_symbol(self, retriever: SymbolRetriever):
        result = retriever.retrieve("MyClass")
        assert result is not None
        assert "MyClass" in result
        assert "**Class**" in result

    def test_retrieve_unknown_symbol(self, retriever: SymbolRetriever):
        result = retriever.retrieve("NonExistent")
        assert result is None

    def test_retrieve_dotted_name(self, retriever: SymbolRetriever):
        result = retriever.retrieve("MyClass.method")
        assert result is not None
        assert "MyClass" in result

    def test_retrieve_tracks_fetched(self, retriever: SymbolRetriever):
        retriever.retrieve("MyClass")
        assert "MyClass" in retriever._recently_fetched

    def test_retrieve_loads_file_context(
        self,
        retriever: SymbolRetriever,
        file_ctx: FileContextManager,
    ):
        retriever.retrieve("MyClass")
        cached = file_ctx.cached_files()
        assert len(cached) >= 1


class TestResetFetched:
    def test_reset_clears_tracking(self, retriever: SymbolRetriever):
        retriever.retrieve("MyClass")
        assert "MyClass" in retriever._recently_fetched
        retriever.reset_fetched()
        assert "MyClass" not in retriever._recently_fetched

    def test_reset_allows_refetch(self, retriever: SymbolRetriever):
        retriever.retrieve("MyClass")
        retriever.reset_fetched()
        names = retriever.detect_symbol_references("Use MyClass")
        assert "MyClass" in names
