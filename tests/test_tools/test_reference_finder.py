from __future__ import annotations

from pathlib import Path

import pytest

from chef_human.agent.symbols.extractor import Symbol
from chef_human.agent.symbols.index import IndexEntry, SymbolIndex
from chef_human.agent.workspace import WorkspaceManager
from chef_human.tools.reference_finder import ReferenceFinderTool


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
    create_file(tmp_path, "mod.py", "def foo_func(x: int) -> str:\n    return str(x)\n\nresult = foo_func(42)\n")
    create_file(tmp_path, "utils.py", "from mod import foo_func\n\ndef bar_func():\n    return foo_func('hi')\n")
    create_file(tmp_path, "other.py", "x = 42\n")
    idx._entries = {
        "foo_func": [
            IndexEntry(
                symbol=Symbol(name="foo_func", kind="function", line=1, signature="def foo_func(x: int) -> str"),
                file_path=str(tmp_path / "mod.py"),
                content_hash="abc",
            ),
        ],
        "bar_func": [
            IndexEntry(
                symbol=Symbol(name="bar_func", kind="function", line=3, signature="def bar_func()"),
                file_path=str(tmp_path / "utils.py"),
                content_hash="def",
            ),
        ],
    }
    idx._initial_built = True
    return idx


@pytest.fixture
def tool(symbol_index: SymbolIndex, workspace: WorkspaceManager) -> ReferenceFinderTool:
    return ReferenceFinderTool(symbol_index=symbol_index, workspace=workspace)


class TestReferenceFinderTool:
    async def test_finds_definitions_and_references(self, tool: ReferenceFinderTool):
        result = await tool.run(name="foo_func")
        assert result.success
        assert "foo_func" in result.output
        assert "Definitions" in result.output
        assert "References" in result.output

    async def test_exclude_definitions(self, tool: ReferenceFinderTool):
        result = await tool.run(name="foo_func", include_definitions=False)
        assert result.success
        assert "Definitions" not in result.output

    async def test_max_results_caps_output(self, tool: ReferenceFinderTool, tmp_path: Path):
        for i in range(25):
            create_file(tmp_path, f"ref{i}.py", f"x = foo_func()  # ref {i}\n")
        result = await tool.run(name="foo_func", max_results=5)
        assert result.success
        assert "and " in result.output
        assert "more" in result.output

    async def test_empty_result(self, tool: ReferenceFinderTool):
        result = await tool.run(name="nonexistent_sym")
        assert result.success
        assert "No references found" in result.output

    async def test_grep_fallback_finds_textual_refs(self, tool: ReferenceFinderTool, tmp_path: Path):
        create_file(tmp_path, "consumer.py", "def bar():\n    return foo_func()\n")
        result = await tool.run(name="foo_func")
        assert result.success
        assert "consumer.py" in result.output

    async def test_symbol_without_definitions(self, workspace: WorkspaceManager, tmp_path: Path):
        create_file(tmp_path, "only_refs.py", "my_thing = 1\nprint(my_thing)\n")
        from chef_human.agent.symbols.extractor import CompositeExtractor
        idx = SymbolIndex(workspace=workspace, extractor=CompositeExtractor())
        idx._initial_built = True
        t = ReferenceFinderTool(symbol_index=idx, workspace=workspace)
        result = await t.run(name="my_thing")
        assert result.success
        assert "my_thing" in result.output

    async def test_reference_count_in_output(self, tool: ReferenceFinderTool):
        result = await tool.run(name="bar_func")
        assert result.success
        assert result.output.startswith("Found ")

    async def test_max_results_50_ceiling(self, tool: ReferenceFinderTool):
        result = await tool.run(name="foo_func", max_results=100)
        assert result.success

    async def test_references_respect_file_extensions(self, tool: ReferenceFinderTool, tmp_path: Path):
        create_file(tmp_path, "data.txt", "foo_func is referenced here")
        result = await tool.run(name="foo_func")
        assert result.success
        assert "data.txt" in result.output
