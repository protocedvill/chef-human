from __future__ import annotations

import json
from pathlib import Path

import pytest

from chef_human.agent.symbols.dependencies import DependencyGraph
from chef_human.agent.symbols.extractor import CompositeExtractor, Symbol
from chef_human.agent.symbols.index import IndexEntry, SymbolIndex
from chef_human.agent.workspace import WorkspaceManager


def create_file(directory: Path, name: str, content: str = "") -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceManager:
    return WorkspaceManager(root=tmp_path)


@pytest.fixture
def extractor() -> CompositeExtractor:
    return CompositeExtractor()


@pytest.fixture
def index(workspace: WorkspaceManager, extractor: CompositeExtractor) -> SymbolIndex:
    idx = SymbolIndex(workspace=workspace, extractor=extractor)
    # Populate with test data
    sym_a = Symbol(name="func_a", kind="function", line=10, signature="def func_a()")
    sym_b = Symbol(name="FuncB", kind="class", line=1, signature="class FuncB")
    idx._entries = {
        "func_a": [
            IndexEntry(symbol=sym_a, file_path=str(workspace.root / "a.py"), content_hash="h1", access_count=3),
        ],
        "FuncB": [
            IndexEntry(symbol=sym_b, file_path=str(workspace.root / "b.py"), content_hash="h2", access_count=1),
        ],
    }
    idx._by_file = {
        workspace.root / "a.py": [IndexEntry(symbol=sym_a, file_path=str(workspace.root / "a.py"), content_hash="h1", access_count=3)],
        workspace.root / "b.py": [IndexEntry(symbol=sym_b, file_path=str(workspace.root / "b.py"), content_hash="h2", access_count=1)],
    }
    idx._content_hashes = {
        workspace.root / "a.py": "h1",
        workspace.root / "b.py": "h2",
    }
    idx._initial_built = True
    return idx


class TestSymbolIndexPersistence:
    def test_save_creates_file(self, workspace: WorkspaceManager, index: SymbolIndex, tmp_path: Path):
        path = tmp_path / "index.json"
        index.save(path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["version"] == 1
        assert "workspace_hash" in data
        assert "func_a" in data["entries"]

    def test_round_trip(self, workspace: WorkspaceManager, index: SymbolIndex, tmp_path: Path):
        path = tmp_path / "index.json"
        index.save(path)
        loaded = SymbolIndex.load(path, workspace, CompositeExtractor())
        assert loaded is not None
        assert loaded.total_symbols() == index.total_symbols()
        assert loaded.total_files() == index.total_files()
        assert loaded.lookup("func_a")
        assert loaded.lookup("FuncB")

    def test_load_missing_file(self, workspace: WorkspaceManager, tmp_path: Path):
        loaded = SymbolIndex.load(tmp_path / "nonexistent.json", workspace, CompositeExtractor())
        assert loaded is None

    def test_load_corrupt_json(self, workspace: WorkspaceManager, tmp_path: Path):
        path = tmp_path / "corrupt.json"
        path.write_text("not json", encoding="utf-8")
        loaded = SymbolIndex.load(path, workspace, CompositeExtractor())
        assert loaded is None

    def test_load_version_mismatch(self, workspace: WorkspaceManager, tmp_path: Path):
        path = tmp_path / "bad_version.json"
        data = {"version": 999, "workspace_hash": "", "entries": {}, "by_file": {}, "content_hashes": {}}
        path.write_text(json.dumps(data), encoding="utf-8")
        loaded = SymbolIndex.load(path, workspace, CompositeExtractor())
        assert loaded is None

    def test_load_preserves_access_count(self, workspace: WorkspaceManager, index: SymbolIndex, tmp_path: Path):
        path = tmp_path / "index.json"
        index.save(path)
        loaded = SymbolIndex.load(path, workspace, CompositeExtractor())
        assert loaded is not None
        entries = loaded.lookup("func_a")
        assert entries[0].access_count >= 3  # incremented by lookup

    def test_empty_index_round_trip(self, workspace: WorkspaceManager, extractor: CompositeExtractor, tmp_path: Path):
        idx = SymbolIndex(workspace=workspace, extractor=extractor)
        idx._initial_built = True
        path = tmp_path / "empty.json"
        idx.save(path)
        loaded = SymbolIndex.load(path, workspace, extractor)
        assert loaded is not None
        assert loaded.total_symbols() == 0

    def test_saved_query_results_match(self, workspace: WorkspaceManager, index: SymbolIndex, tmp_path: Path):
        path = tmp_path / "index.json"
        index.save(path)
        loaded = SymbolIndex.load(path, workspace, CompositeExtractor())
        assert loaded is not None
        assert len(loaded.lookup("func_a")) == 1
        assert loaded.search("func_a")
        assert len(loaded.lookup_by_prefix("func")) == 1


class TestDependencyGraphPersistence:
    def test_save_creates_file(self, workspace: WorkspaceManager, tmp_path: Path):
        idx = SymbolIndex(workspace=workspace, extractor=CompositeExtractor())
        idx._initial_built = True
        dg = DependencyGraph(symbol_index=idx)
        a = workspace.root / "main.py"
        b = workspace.root / "utils.py"
        dg._deps = {a: {b}}
        path = tmp_path / "deps.json"
        dg.save(path, workspace_root=workspace.root)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["version"] == 1
        assert "main.py" in data["graph"]

    def test_round_trip(self, workspace: WorkspaceManager, tmp_path: Path):
        idx = SymbolIndex(workspace=workspace, extractor=CompositeExtractor())
        idx._initial_built = True
        dg = DependencyGraph(symbol_index=idx)
        a = workspace.root / "main.py"
        b = workspace.root / "utils.py"
        dg._deps = {a: {b}}
        dg._reverse_deps = {b: {a}}
        path = tmp_path / "deps.json"
        dg.save(path, workspace_root=workspace.root)
        loaded = DependencyGraph.load(path, workspace.root, idx)
        assert loaded is not None
        assert len(loaded._deps) == 1
        deps = loaded.dependencies(a)
        assert deps == [b]

    def test_load_missing_file(self, workspace: WorkspaceManager):
        idx = SymbolIndex(workspace=workspace, extractor=CompositeExtractor())
        idx._initial_built = True
        loaded = DependencyGraph.load(Path("nonexistent.json"), workspace.root, idx)
        assert loaded is None

    def test_empty_graph(self, workspace: WorkspaceManager, tmp_path: Path):
        idx = SymbolIndex(workspace=workspace, extractor=CompositeExtractor())
        idx._initial_built = True
        dg = DependencyGraph(symbol_index=idx)
        path = tmp_path / "empty_deps.json"
        dg.save(path, workspace_root=workspace.root)
        loaded = DependencyGraph.load(path, workspace.root, idx)
        assert loaded is not None
        assert loaded._deps == {}

    def test_version_mismatch(self, workspace: WorkspaceManager, tmp_path: Path):
        idx = SymbolIndex(workspace=workspace, extractor=CompositeExtractor())
        path = tmp_path / "bad_deps.json"
        data = {"version": 999, "graph": {}}
        path.write_text(json.dumps(data), encoding="utf-8")
        loaded = DependencyGraph.load(path, workspace.root, idx)
        assert loaded is None
