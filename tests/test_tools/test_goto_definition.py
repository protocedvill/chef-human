from __future__ import annotations

from pathlib import Path

import pytest

from chef_human.agent.file_context import FileContextManager
from chef_human.agent.symbols.extractor import Symbol
from chef_human.agent.symbols.index import IndexEntry, SymbolIndex
from chef_human.agent.workspace import WorkspaceManager
from chef_human.llm.tokenizer import ApproxTokenizer
from chef_human.tools.goto_definition import GotoDefinitionTool


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
    create_file(tmp_path, "mod.py", "def foo_func(x: int) -> str:\n    return str(x)\n\nclass FooClass:\n    pass\n")
    create_file(tmp_path, "utils.py", "def bar_func(y: str) -> int:\n    return len(y)\n")
    idx._entries = {
        "foo_func": [
            IndexEntry(
                symbol=Symbol(name="foo_func", kind="function", line=1, signature="def foo_func(x: int) -> str"),
                file_path=str(tmp_path / "mod.py"),
                content_hash="abc",
                access_count=0,
            ),
        ],
        "FooClass": [
            IndexEntry(
                symbol=Symbol(name="FooClass", kind="class", line=4, signature="class FooClass:"),
                file_path=str(tmp_path / "mod.py"),
                content_hash="abc",
                access_count=0,
            ),
        ],
        "bar_func": [
            IndexEntry(
                symbol=Symbol(name="bar_func", kind="function", line=1, signature="def bar_func(y: str) -> int"),
                file_path=str(tmp_path / "utils.py"),
                content_hash="def",
                access_count=0,
            ),
        ],
    }
    idx._initial_built = True
    return idx


@pytest.fixture
def file_context(workspace: WorkspaceManager) -> FileContextManager:
    tokenizer = ApproxTokenizer()
    return FileContextManager(workspace=workspace, tokenizer=tokenizer)


@pytest.fixture
def tool(symbol_index: SymbolIndex, file_context: FileContextManager) -> GotoDefinitionTool:
    return GotoDefinitionTool(symbol_index=symbol_index, file_context=file_context)


class TestGotoDefinitionTool:
    async def test_find_by_name(self, tool: GotoDefinitionTool):
        result = await tool.run(name="foo_func")
        assert result.success
        assert "foo_func" in result.output
        assert "mod.py" in result.output
        assert "line 1" in result.output or "1 " in result.output
        assert "def foo_func" in result.output

    async def test_find_by_kind(self, tool: GotoDefinitionTool):
        result = await tool.run(name="FooClass", kind="class")
        assert result.success
        assert "FooClass" in result.output
        assert "class" in result.output.lower()

    async def test_kind_filter_excludes_wrong_kind(self, tool: GotoDefinitionTool):
        result = await tool.run(name="foo_func", kind="class")
        assert result.success
        assert "No definition found" in result.output

    async def test_unknown_symbol(self, tool: GotoDefinitionTool):
        result = await tool.run(name="nonexistent")
        assert result.success
        assert "No definition found" in result.output

    async def test_multiple_definitions(self, symbol_index: SymbolIndex, file_context: FileContextManager):
        from chef_human.agent.symbols.extractor import Symbol
        from chef_human.agent.symbols.index import IndexEntry
        idx = symbol_index
        idx._entries.setdefault("overloaded_func", []).append(
            IndexEntry(
                symbol=Symbol(name="overloaded_func", kind="function", line=5, signature="def overloaded_func(a: int)"),
                file_path=file_context._workspace.resolve("mod.py"),
                content_hash="abc",
            ),
        )
        idx._entries.setdefault("overloaded_func", []).append(
            IndexEntry(
                symbol=Symbol(name="overloaded_func", kind="function", line=10, signature="def overloaded_func(b: str)"),
                file_path=file_context._workspace.resolve("utils.py"),
                content_hash="def",
            ),
        )
        t = GotoDefinitionTool(symbol_index=idx, file_context=file_context)
        result = await t.run(name="overloaded_func")
        assert result.success
        assert result.output.count("overloaded_func") >= 2

    async def test_loads_file_into_context(self, tool: GotoDefinitionTool, file_context: FileContextManager):
        result = await tool.run(name="foo_func")
        assert result.success
        content = file_context.get("mod.py")
        assert content is not None
        assert "def foo_func" in content

    async def test_multiple_definitions_same_file_dedup(self, symbol_index: SymbolIndex, file_context: FileContextManager):
        idx = symbol_index
        idx._entries["FooClass"] = [
            IndexEntry(
                symbol=Symbol(name="FooClass", kind="class", line=4, signature="class FooClass:"),
                file_path=str(file_context._workspace.root / "mod.py"),
                content_hash="abc",
            ),
            IndexEntry(
                symbol=Symbol(name="FooClass", kind="class", line=10, signature="class FooClass:"),
                file_path=str(file_context._workspace.root / "mod.py"),
                content_hash="abc",
            ),
        ]
        t = GotoDefinitionTool(symbol_index=idx, file_context=file_context)
        result = await t.run(name="FooClass")
        assert result.success
        assert result.output.count("mod.py") == 1

    async def test_empty_name_returns_no_definition(self, tool: GotoDefinitionTool):
        result = await tool.run(name="")
        assert result.success
        assert "No definition found" in result.output

    async def test_output_includes_signature(self, tool: GotoDefinitionTool):
        result = await tool.run(name="bar_func")
        assert result.success
        assert "def bar_func" in result.output
