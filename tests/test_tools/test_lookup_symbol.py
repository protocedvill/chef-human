from __future__ import annotations

from pathlib import Path

import pytest

from chef_human.agent.symbols.extractor import Symbol
from chef_human.agent.symbols.index import IndexEntry, SymbolIndex
from chef_human.agent.workspace import WorkspaceManager
from chef_human.tools.lookup_symbol import LookupSymbolTool


@pytest.fixture
def symbol_index(tmp_path: Path) -> SymbolIndex:
    ws = WorkspaceManager(root=str(tmp_path))
    # Build index with a minimal extractor that returns nothing
    from chef_human.agent.symbols.extractor import CompositeExtractor

    idx = SymbolIndex(workspace=ws, extractor=CompositeExtractor())
    # Manually populate entries for testing
    idx._entries = {
        "foo_func": [
            IndexEntry(
                symbol=Symbol(name="foo_func", kind="function", line=10, signature="def foo_func(x: int) -> str"),
                file_path=str(tmp_path / "mod.py"),
                content_hash="abc",
            ),
        ],
        "FooClass": [
            IndexEntry(
                symbol=Symbol(name="FooClass", kind="class", line=1, signature="class FooClass:"),
                file_path=str(tmp_path / "mod.py"),
                content_hash="abc",
            ),
        ],
        "bar_func": [
            IndexEntry(
                symbol=Symbol(name="bar_func", kind="function", line=20, signature="def bar_func(y: str) -> int"),
                file_path=str(tmp_path / "utils.py"),
                content_hash="def",
            ),
        ],
    }
    idx._initial_built = True
    return idx


@pytest.fixture
def lookup_tool(symbol_index: SymbolIndex, tmp_path: Path) -> LookupSymbolTool:
    ws = WorkspaceManager(root=str(tmp_path))
    return LookupSymbolTool(symbol_index=symbol_index, workspace=ws)


class TestLookupSymbolTool:
    async def test_lookup_by_name(self, lookup_tool: LookupSymbolTool):
        result = await lookup_tool.run(name="foo_func")
        assert result.success
        assert "function" in result.output
        assert "foo_func" in result.output
        assert "mod.py" in result.output

    async def test_lookup_missing_name(self, lookup_tool: LookupSymbolTool):
        result = await lookup_tool.run(name="nonexistent")
        assert result.success
        assert "No symbols found" in result.output

    async def test_lookup_by_prefix(self, lookup_tool: LookupSymbolTool):
        result = await lookup_tool.run(prefix="foo")
        assert result.success
        assert "foo_func" in result.output
        assert "bar_func" not in result.output

    async def test_lookup_by_prefix_no_match(self, lookup_tool: LookupSymbolTool):
        result = await lookup_tool.run(prefix="zzz")
        assert result.success
        assert "No symbols found" in result.output

    async def test_lookup_by_prefix_case_sensitive(self, lookup_tool: LookupSymbolTool):
        result = await lookup_tool.run(prefix="Foo")
        assert result.success
        assert "FooClass" in result.output

    async def test_search(self, lookup_tool: LookupSymbolTool):
        result = await lookup_tool.run(query="bar")
        assert result.success
        assert "bar_func" in result.output

    async def test_search_case_insensitive(self, lookup_tool: LookupSymbolTool):
        result = await lookup_tool.run(query="FOO")
        assert result.success
        assert "foo_func" in result.output

    async def test_search_no_match(self, lookup_tool: LookupSymbolTool):
        result = await lookup_tool.run(query="zzzzz")
        assert result.success
        assert "No symbols found" in result.output

    async def test_no_mode_provided(self, lookup_tool: LookupSymbolTool):
        result = await lookup_tool.run()
        assert not result.success
        assert "Exactly one" in (result.error or "")

    async def test_multiple_modes_provided(self, lookup_tool: LookupSymbolTool):
        result = await lookup_tool.run(name="foo", prefix="bar")
        assert not result.success
        assert "Exactly one" in (result.error or "")
