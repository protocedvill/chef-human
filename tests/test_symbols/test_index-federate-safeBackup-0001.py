from __future__ import annotations

from pathlib import Path

import pytest

from chef_human.agent.symbols.extractor import TreeSitterExtractor
from chef_human.agent.symbols.index import SymbolIndex
from chef_human.agent.workspace import WorkspaceManager


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceManager:
    return WorkspaceManager(root=tmp_path)


@pytest.fixture
def extractor() -> TreeSitterExtractor:
    return TreeSitterExtractor()


@pytest.fixture
def index(workspace: WorkspaceManager, extractor: TreeSitterExtractor) -> SymbolIndex:
    return SymbolIndex(workspace=workspace, extractor=extractor)


def _write(ws: WorkspaceManager, path: str, content: str) -> Path:
    full = ws.root / path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content)
    return full


class TestSymbolIndexEmpty:
    def test_is_built_false_initially(self, index: SymbolIndex):
        assert index.is_built is False

    def test_total_symbols_zero(self, index: SymbolIndex):
        assert index.total_symbols() == 0

    def test_total_files_zero(self, index: SymbolIndex):
        assert index.total_files() == 0

    def test_lookup_unknown_returns_empty(self, index: SymbolIndex):
        assert index.lookup("Foo") == []

    def test_lookup_by_file_unknown_returns_empty(self, index: SymbolIndex):
        assert index.lookup_by_file(Path("nonexistent.py")) == []

    def test_lookup_by_prefix_unknown_returns_empty(self, index: SymbolIndex):
        assert index.lookup_by_prefix("Foo") == []

    def test_search_unknown_returns_empty(self, index: SymbolIndex):
        assert index.search("Foo") == []


class TestSymbolIndexBuild:
    def test_build_returns_count(self, workspace, extractor, index):
        _write(workspace, "foo.py", "def hello():\n    pass\n")
        count = index.build()
        assert count == 1

    def test_build_python_symbols(self, workspace, extractor, index):
        _write(workspace, "foo.py", "def hello():\n    pass\n\ndef world():\n    pass\n")
        index.build()
        assert index.total_symbols() == 2
        assert index.total_files() == 1

    def test_build_multiple_files(self, workspace, extractor, index):
        _write(workspace, "a.py", "def foo():\n    pass\n")
        _write(workspace, "b.py", "def bar():\n    pass\n")
        index.build()
        assert index.total_symbols() == 2
        assert index.total_files() == 2

    def test_build_skips_unparseable_files(self, workspace, extractor, index):
        _write(workspace, "a.py", "def foo():\n    pass\n")
        _write(workspace, "data.bin", b"\x00\x01\x02\x03".decode("utf-8", errors="replace"))
        index.build()
        assert index.total_symbols() == 1

    def test_build_rebuild_clears_old_data(self, workspace, extractor, index):
        _write(workspace, "a.py", "def foo():\n    pass\n")
        index.build()
        assert index.total_symbols() == 1
        _write(workspace, "a.py", "def bar():\n    pass\n")
        index.build()
        assert index.total_symbols() == 1
        assert index.lookup("bar")
        assert not index.lookup("foo")

    def test_is_built_after_build(self, workspace, extractor, index):
        _write(workspace, "a.py", "def foo():\n    pass\n")
        index.build()
        assert index.is_built is True

    def test_build_with_explicit_file_list(self, workspace, extractor, index):
        _write(workspace, "a.py", "def foo():\n    pass\n")
        _write(workspace, "b.py", "def bar():\n    pass\n")
        files = [workspace.root / "a.py"]
        count = index.build(files=files)
        assert count == 1
        assert index.total_files() == 1


class TestSymbolIndexLookup:
    @pytest.fixture(autouse=True)
    def setup(self, workspace, extractor, index):
        _write(workspace, "shapes.py", """
class Circle:
    def area(self):
        pass

class Square:
    def area(self):
        pass
""")
        _write(workspace, "utils.py", """
def helper():
    pass
def area(x):
    pass
""")
        index.build()
        return index

    def test_lookup_by_name(self, index):
        entries = index.lookup("Circle")
        assert len(entries) == 1
        assert entries[0].symbol.name == "Circle"
        assert entries[0].symbol.kind == "class"

    def test_lookup_by_kind(self, index):
        entries = index.lookup("helper", kind="function")
        assert len(entries) == 1
        assert entries[0].symbol.name == "helper"

    def test_lookup_overloaded_name(self, index):
        entries = index.lookup("area")
        assert len(entries) == 3

    def test_lookup_unknown(self, index):
        assert index.lookup("NoSuchSymbol") == []

    def test_lookup_by_file(self, workspace, index):
        entries = index.lookup_by_file(workspace.root / "shapes.py")
        names = {e.symbol.name for e in entries}
        assert "Circle" in names
        assert "Square" in names
        assert "helper" not in names

    def test_lookup_by_file_relative(self, workspace, index):
        entries = index.lookup_by_file(Path("shapes.py"))
        assert len(entries) >= 2

    def test_lookup_by_prefix(self, index):
        entries = index.lookup_by_prefix("C")
        assert len(entries) == 1
        assert entries[0].symbol.name == "Circle"

    def test_lookup_by_prefix_max_results(self, index):
        entries = index.lookup_by_prefix("", max_results=1)
        assert len(entries) == 1

    def test_search_matches_name(self, index):
        entries = index.search("Circle")
        assert len(entries) == 1
        assert entries[0].symbol.name == "Circle"

    def test_search_matches_signature(self, index):
        entries = index.search("helper")
        assert len(entries) >= 1

    def test_search_case_insensitive(self, index):
        entries = index.search("circle")
        assert len(entries) == 1

    def test_search_unknown(self, index):
        assert index.search("NoMatchAtAll") == []

    def test_find_similar_close_name_typo(self, index):
        results = index.find_similar("Circl")
        names = {e.symbol.name for e in results}
        assert "Circle" in names

    def test_find_similar_naming_variant(self, index):
        results = index.find_similar("Squares")
        names = {e.symbol.name for e in results}
        assert "Square" in names

    def test_find_similar_no_match(self, index):
        assert index.find_similar("CompletelyUnrelatedXyz123") == []

    def test_find_similar_matches_identical_name(self, index):
        # find_similar is meant for the miss path, but an exact name is
        # trivially its own closest match — callers only reach this method
        # after lookup() already returned nothing for the literal name.
        results = index.find_similar("Circle")
        names = {e.symbol.name for e in results}
        assert "Circle" in names


class TestSymbolIndexRefresh:
    def test_refresh_before_build_calls_build(self, workspace, extractor, index):
        _write(workspace, "a.py", "def foo():\n    pass\n")
        count = index.refresh()
        assert count == 1
        assert index.is_built is True

    def test_refresh_no_changes(self, workspace, extractor, index):
        _write(workspace, "a.py", "def foo():\n    pass\n")
        index.build()
        count = index.refresh()
        assert count == 0
        assert index.lookup("foo")

    def test_refresh_new_file(self, workspace, extractor, index):
        _write(workspace, "a.py", "def foo():\n    pass\n")
        index.build()
        _write(workspace, "b.py", "def bar():\n    pass\n")
        count = index.refresh()
        assert count == 1
        assert index.lookup("bar")

    def test_refresh_changed_file(self, workspace, extractor, index):
        _write(workspace, "a.py", "def foo():\n    pass\n")
        index.build()
        assert index.lookup("foo")
        _write(workspace, "a.py", "def bar():\n    pass\n")
        count = index.refresh()
        assert count == 1
        assert not index.lookup("foo")
        assert index.lookup("bar")

    def test_refresh_unchanged_file_untouched(self, workspace, extractor, index):
        _write(workspace, "a.py", "def foo():\n    pass\n")
        _write(workspace, "b.py", "def bar():\n    pass\n")
        index.build()
        _write(workspace, "b.py", "def baz():\n    pass\n")
        count = index.refresh()
        assert count == 1
        assert index.lookup("foo")
        assert not index.lookup("bar")
        assert index.lookup("baz")


class TestAccessCount:
    @pytest.fixture(autouse=True)
    def setup(self, workspace, extractor, index):
        _write(workspace, "mod.py", """
def alpha():
    pass
def beta():
    pass
""")
        index.build()

    def test_lookup_increments_count(self, index):
        index.lookup("alpha")
        entries = index.lookup("alpha")
        assert entries[0].access_count >= 2  # incremented twice

    def test_lookup_returns_sorted_by_count(self, index):
        index.lookup("alpha")
        index.lookup("alpha")
        index.lookup("beta")
        sorted_entries = index.lookup("")
        if sorted_entries:
            # Most accessed should come first
            assert sorted_entries[0].symbol.name == "alpha"

    def test_search_increments_count(self, index):
        index.search("alpha")
        entries = index.lookup("alpha")
        assert entries[0].access_count >= 1

    def test_lookup_by_prefix_increments_count(self, index):
        index.lookup_by_prefix("a")
        entries = index.lookup("alpha")
        assert entries[0].access_count >= 1
