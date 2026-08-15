from __future__ import annotations

from pathlib import Path

import pytest

from chef_human.agent.symbols.extractor import Symbol
from chef_human.agent.symbols.index import IndexEntry, SymbolIndex
from chef_human.agent.workspace import WorkspaceManager
from chef_human.tools.diff import DiffStore
from chef_human.tools.refactor import RefactorTool


def create_file(directory: Path, name: str, content: str = "") -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceManager:
    return WorkspaceManager(root=str(tmp_path))


@pytest.fixture
def symbol_index(workspace: WorkspaceManager, tmp_path: Path) -> SymbolIndex:
    from chef_human.agent.symbols.extractor import CompositeExtractor

    idx = SymbolIndex(workspace=workspace, extractor=CompositeExtractor())
    create_file(tmp_path, "shapes.py", "class Circle:\n    pass\n\nclass Square:\n    pass\n")
    create_file(tmp_path, "main.py", "from shapes import Circle\n\nc = Circle()\ns = Square()\n")
    idx._entries = {
        "Circle": [
            IndexEntry(
                symbol=Symbol(name="Circle", kind="class", line=1, signature="class Circle:"),
                file_path=str(tmp_path / "shapes.py"),
                content_hash="abc",
            ),
        ],
        "Square": [
            IndexEntry(
                symbol=Symbol(name="Square", kind="class", line=4, signature="class Square:"),
                file_path=str(tmp_path / "shapes.py"),
                content_hash="abc",
            ),
        ],
    }
    idx._initial_built = True
    return idx


@pytest.fixture
def diff_store() -> DiffStore:
    return DiffStore()


@pytest.fixture
def tool(workspace: WorkspaceManager, symbol_index: SymbolIndex, diff_store: DiffStore) -> RefactorTool:
    return RefactorTool(workspace=workspace, symbol_index=symbol_index, diff_store=diff_store)


class TestRefactorTool:
    async def test_rename_all_scope(self, tool: RefactorTool, tmp_path: Path):
        result = await tool.run(old_name="Circle", new_name="Ellipse", scope="all")
        assert result.success
        assert "Ellipse" in result.output
        assert (tmp_path / "shapes.py").read_text() == "class Ellipse:\n    pass\n\nclass Square:\n    pass\n"

    async def test_rename_definitions_scope(self, tool: RefactorTool, tmp_path: Path):
        result = await tool.run(old_name="Circle", new_name="Ellipse", scope="definitions")
        assert result.success
        assert "Ellipse" in result.output
        # Only shapes.py (def file) changed, not main.py (reference)
        assert (tmp_path / "main.py").read_text() == "from shapes import Circle\n\nc = Circle()\ns = Square()\n"

    async def test_rename_file_scope(self, tool: RefactorTool, tmp_path: Path):
        result = await tool.run(old_name="Circle", new_name="Ellipse", scope="file", path=str(tmp_path / "main.py"))
        assert result.success
        assert (tmp_path / "main.py").read_text() == "from shapes import Ellipse\n\nc = Ellipse()\ns = Square()\n"
        assert "Ellipse" not in (tmp_path / "shapes.py").read_text()

    async def test_dry_run_does_not_modify(self, tool: RefactorTool, tmp_path: Path):
        original = (tmp_path / "shapes.py").read_text()
        result = await tool.run(old_name="Circle", new_name="Ellipse", dry_run=True)
        assert result.success
        assert "Dry run" in result.output
        assert (tmp_path / "shapes.py").read_text() == original

    async def test_word_boundary_no_partial_match(self, tool: RefactorTool, tmp_path: Path):
        create_file(tmp_path, "circle.py", "circles = []\nCircle = 1\n")
        result = await tool.run(old_name="Circle", new_name="Oval", scope="all")
        assert result.success
        content = (tmp_path / "circle.py").read_text()
        assert "Oval" in content
        assert "circles" in content

    async def test_error_unknown_symbol(self, tool: RefactorTool):
        result = await tool.run(old_name="NonExistent", new_name="Something")
        assert not result.success
        assert "No definitions found" in (result.error or "")

    async def test_error_empty_old_name(self, tool: RefactorTool):
        result = await tool.run(old_name="", new_name="Something")
        assert not result.success

    async def test_error_empty_new_name(self, tool: RefactorTool):
        result = await tool.run(old_name="Circle", new_name="")
        assert not result.success

    async def test_diff_store_receives_entries(self, tool: RefactorTool, diff_store: DiffStore, tmp_path: Path):
        result = await tool.run(old_name="Circle", new_name="Ellipse", scope="all")
        assert result.success
        assert len(diff_store._entries) >= 1

    async def test_rollback_on_write_failure(self, tool: RefactorTool, symbol_index: SymbolIndex, workspace: WorkspaceManager, diff_store: DiffStore, tmp_path: Path):
        original_shapes = (tmp_path / "shapes.py").read_text()
        # Make main.py readonly to trigger write failure
        main_path = tmp_path / "main.py"
        main_path.chmod(0o444)
        result = await tool.run(old_name="Circle", new_name="Ellipse", scope="all")
        assert not result.success
        main_path.chmod(0o644)
        assert (tmp_path / "shapes.py").read_text() == original_shapes

    async def test_rename_multiple_occurrences_in_one_file(self, tool: RefactorTool, tmp_path: Path):
        create_file(tmp_path, "multi.py", "Circle()\nprint(Circle)\nx = Circle\n")
        result = await tool.run(old_name="Circle", new_name="Oval", scope="all")
        assert result.success
        content = (tmp_path / "multi.py").read_text()
        assert content.count("Oval") == 3
        assert "Circle" not in content

    async def test_output_includes_file_list_and_counts(self, tool: RefactorTool):
        result = await tool.run(old_name="Circle", new_name="Ellipse", scope="definitions")
        assert result.success
        assert "shapes.py" in result.output
        assert "updated" in result.output

    async def test_output_includes_diff(self, tool: RefactorTool):
        result = await tool.run(old_name="Circle", new_name="Ellipse", scope="definitions")
        assert result.success
        assert "-Circle" in result.output or "+Ellipse" in result.output or "-class Circle" in result.output

    async def test_no_changes_when_name_matches(self, tool: RefactorTool):
        result = await tool.run(old_name="Circle", new_name="Circle", scope="all")
        assert result.success
        assert "No changes" in result.output

    async def test_file_scope_missing_path(self, tool: RefactorTool):
        result = await tool.run(old_name="Circle", new_name="Ellipse", scope="file", path=None)
        assert not result.success

    async def test_file_scope_nonexistent_path(self, tool: RefactorTool):
        result = await tool.run(old_name="Circle", new_name="Ellipse", scope="file", path="/nonexistent/path.py")
        assert not result.success

    async def test_too_many_files_error(self, tool: RefactorTool, symbol_index: SymbolIndex, tmp_path: Path):
        for i in range(60):
            create_file(tmp_path, f"many{i}.py", f"Circle = {i}\n")
        result = await tool.run(old_name="Circle", new_name="Oval", scope="all")
        assert not result.success
        assert "Too many files" in (result.error or "")

    async def test_rename_same_symbol_different_file_single_scope(self, tool: RefactorTool, tmp_path: Path):
        create_file(tmp_path, "extra.py", "def Circle():\n    return 1\n")
        result = await tool.run(old_name="Circle", new_name="CircleFunc", scope="definitions")
        assert result.success

    async def test_dry_run_counts_occurrences(self, tool: RefactorTool, tmp_path: Path):
        result = await tool.run(old_name="Circle", new_name="Oval", dry_run=True)
        assert result.success
        assert "occurrence" in result.output

    async def test_dep_graph_includes_dependents(
        self, workspace: WorkspaceManager, symbol_index: SymbolIndex, tmp_path: Path
    ):
        from chef_human.agent.symbols.dependencies import DependencyGraph
        create_file(tmp_path, "importer.py", "from shapes import Circle\nc = Circle()\n")
        dg = DependencyGraph(symbol_index)
        dg.build()
        tool = RefactorTool(workspace=workspace, symbol_index=symbol_index, dep_graph=dg)
        result = await tool.run(old_name="Circle", new_name="Oval", scope="all")
        assert result.success
        content = (tmp_path / "importer.py").read_text()
        assert "Oval" in content

    async def test_dep_graph_fallback_when_none(
        self, tool: RefactorTool, tmp_path: Path
    ):
        create_file(tmp_path, "importer.py", "from shapes import Circle\nc = Circle()\n")
        result = await tool.run(old_name="Circle", new_name="Oval", scope="all")
        assert result.success
        # Without dep_graph, importer.py is only found by grep
        content = (tmp_path / "importer.py").read_text()
        assert "Oval" in content

    async def test_dep_graph_empty_no_error(
        self, workspace: WorkspaceManager, symbol_index: SymbolIndex
    ):
        from chef_human.agent.symbols.dependencies import DependencyGraph
        dg = DependencyGraph(symbol_index)
        tool = RefactorTool(workspace=workspace, symbol_index=symbol_index, dep_graph=dg)
        result = await tool.run(old_name="Circle", new_name="Oval", scope="all")
        # Empty dependency graph shouldn't cause error — grep will still try
        assert result.success or "No definitions found" in (result.error or "")
