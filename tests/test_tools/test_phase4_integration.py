from __future__ import annotations

from pathlib import Path

from chef_human.agent.file_context import FileContextManager
from chef_human.agent.symbols.extractor import CompositeExtractor
from chef_human.agent.symbols.index import SymbolIndex
from chef_human.agent.workspace import WorkspaceManager
from chef_human.llm.tokenizer import ApproxTokenizer
from chef_human.tools import create_tool_registry
from chef_human.tools.goto_definition import GotoDefinitionTool
from chef_human.tools.redo import RedoTool
from chef_human.tools.refactor import RefactorTool
from chef_human.tools.reference_finder import ReferenceFinderTool


class TestPhase4ToolRegistry:
    def test_refactor_registered(self, tmp_path: Path):
        ws = WorkspaceManager(root=tmp_path)
        idx = SymbolIndex(workspace=ws, extractor=CompositeExtractor())
        idx._initial_built = True
        registry = create_tool_registry(workspace=ws, symbol_index=idx)
        tool = registry.get("refactor_symbol")
        assert tool is not None
        assert isinstance(tool, RefactorTool)

    def test_goto_definition_registered(self, tmp_path: Path):
        ws = WorkspaceManager(root=tmp_path)
        idx = SymbolIndex(workspace=ws, extractor=CompositeExtractor())
        idx._initial_built = True
        tokenizer = ApproxTokenizer()
        fc = FileContextManager(workspace=ws, tokenizer=tokenizer)
        registry = create_tool_registry(workspace=ws, symbol_index=idx, file_context=fc)
        tool = registry.get("goto_definition")
        assert tool is not None
        assert isinstance(tool, GotoDefinitionTool)

    def test_find_references_registered(self, tmp_path: Path):
        ws = WorkspaceManager(root=tmp_path)
        idx = SymbolIndex(workspace=ws, extractor=CompositeExtractor())
        idx._initial_built = True
        registry = create_tool_registry(workspace=ws, symbol_index=idx)
        tool = registry.get("find_references")
        assert tool is not None
        assert isinstance(tool, ReferenceFinderTool)

    def test_refactor_has_diff_store(self, tmp_path: Path):
        ws = WorkspaceManager(root=tmp_path)
        idx = SymbolIndex(workspace=ws, extractor=CompositeExtractor())
        idx._initial_built = True
        registry = create_tool_registry(workspace=ws, symbol_index=idx)
        tool = registry.get("refactor_symbol")
        assert tool is not None
        assert hasattr(tool, "_diff_store")
        assert tool._diff_store is not None

    def test_goto_definition_requires_file_context(self, tmp_path: Path):
        ws = WorkspaceManager(root=tmp_path)
        idx = SymbolIndex(workspace=ws, extractor=CompositeExtractor())
        idx._initial_built = True
        registry = create_tool_registry(workspace=ws, symbol_index=idx)
        tool = registry.get("goto_definition")
        assert tool is None

    def test_tools_absent_without_symbol_index(self, tmp_path: Path):
        ws = WorkspaceManager(root=tmp_path)
        registry = create_tool_registry(workspace=ws)
        assert registry.get("refactor_symbol") is None
        assert registry.get("find_references") is None
        assert registry.get("goto_definition") is None

    def test_lookup_symbol_still_registered(self, tmp_path: Path):
        ws = WorkspaceManager(root=tmp_path)
        idx = SymbolIndex(workspace=ws, extractor=CompositeExtractor())
        idx._initial_built = True
        registry = create_tool_registry(workspace=ws, symbol_index=idx)
        assert registry.get("lookup_symbol") is not None

    def test_redo_registered(self, tmp_path: Path):
        ws = WorkspaceManager(root=tmp_path)
        registry = create_tool_registry(workspace=ws)
        tool = registry.get("redo")
        assert tool is not None
        assert isinstance(tool, RedoTool)

    def test_undo_registered(self, tmp_path: Path):
        ws = WorkspaceManager(root=tmp_path)
        registry = create_tool_registry(workspace=ws)
        assert registry.get("undo") is not None

    def test_refactor_accepts_dep_graph(self, tmp_path: Path):
        ws = WorkspaceManager(root=tmp_path)
        idx = SymbolIndex(workspace=ws, extractor=CompositeExtractor())
        idx._initial_built = True
        registry = create_tool_registry(workspace=ws, symbol_index=idx)
        tool = registry.get("refactor_symbol")
        assert tool is not None
        assert hasattr(tool, "_dep_graph")
        assert tool._dep_graph is None  # no dep_graph passed = None

    def test_refactor_with_dep_graph(self, tmp_path: Path):
        from chef_human.agent.symbols.dependencies import DependencyGraph
        ws = WorkspaceManager(root=tmp_path)
        idx = SymbolIndex(workspace=ws, extractor=CompositeExtractor())
        idx._initial_built = True
        dg = DependencyGraph(idx)
        registry = create_tool_registry(workspace=ws, symbol_index=idx, dep_graph=dg)
        tool = registry.get("refactor_symbol")
        assert tool is not None
        assert tool._dep_graph is dg

    def test_lint_fix_registered(self, tmp_path: Path):
        ws = WorkspaceManager(root=tmp_path)
        registry = create_tool_registry(workspace=ws)
        tool = registry.get("lint_fix")
        assert tool is not None

    def test_read_write_edit_receive_file_context_without_symbol_index(self, tmp_path: Path):
        """file_context wiring for read/write/edit must not require a
        symbol_index -- large repos on the RAG path have no symbol_index
        but should still keep read/written files visible in context."""
        ws = WorkspaceManager(root=tmp_path)
        tokenizer = ApproxTokenizer()
        fc = FileContextManager(workspace=ws, tokenizer=tokenizer)
        registry = create_tool_registry(workspace=ws, file_context=fc)

        assert registry.get("read")._file_context is fc
        assert registry.get("write")._file_context is fc
        assert registry.get("edit")._file_context is fc

    def test_read_write_edit_tolerate_missing_file_context(self, tmp_path: Path):
        ws = WorkspaceManager(root=tmp_path)
        registry = create_tool_registry(workspace=ws)

        assert registry.get("read")._file_context is None
        assert registry.get("write")._file_context is None
        assert registry.get("edit")._file_context is None
