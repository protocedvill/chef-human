from __future__ import annotations

from typing import TYPE_CHECKING

from chef_human.tools.diff import DiffStore
from chef_human.tools.filesystem import (
    EditTool,
    GlobTool,
    GrepTool,
    LsTool,
    LsTreeTool,
    ReadTool,
    WriteTool,
)
from chef_human.tools.goto_definition import GotoDefinitionTool
from chef_human.tools.lint_fix import LintFixTool
from chef_human.tools.lookup_symbol import LookupSymbolTool
from chef_human.tools.patch_tool import PatchTool
from chef_human.tools.redo import RedoTool
from chef_human.tools.refactor import RefactorTool
from chef_human.tools.reference_finder import ReferenceFinderTool
from chef_human.tools.registry import ToolRegistry, ToolResult
from chef_human.tools.shell import BashTool
from chef_human.tools.undo import UndoTool
from chef_human.tools.user import AskUserTool, FinishTool
from chef_human.tools.view_diff import ViewDiffTool

if TYPE_CHECKING:
    from chef_human.agent.file_context import FileContextManager
    from chef_human.agent.symbols.dependencies import DependencyGraph
    from chef_human.agent.symbols.index import SymbolIndex
    from chef_human.agent.workspace import WorkspaceManager


def create_tool_registry(
    workspace: WorkspaceManager,
    symbol_index: SymbolIndex | None = None,
    file_context: FileContextManager | None = None,
    dep_graph: DependencyGraph | None = None,
) -> ToolRegistry:
    diff_store = DiffStore()

    registry = ToolRegistry()
    registry.register(AskUserTool())
    registry.register(BashTool(workspace))
    registry.register(EditTool(workspace, diff_store=diff_store))
    registry.register(FinishTool())
    registry.register(GlobTool(workspace))
    registry.register(GrepTool(workspace))
    registry.register(LsTool(workspace))
    registry.register(LsTreeTool(workspace))
    registry.register(LintFixTool(workspace, diff_store=diff_store))
    registry.register(ReadTool(workspace))
    registry.register(PatchTool(workspace, diff_store=diff_store))
    registry.register(RedoTool(workspace, diff_store=diff_store))
    registry.register(UndoTool(workspace, diff_store=diff_store))
    registry.register(ViewDiffTool(diff_store=diff_store))
    registry.register(WriteTool(workspace, diff_store=diff_store))
    if symbol_index is not None:
        registry.register(LookupSymbolTool(symbol_index=symbol_index, workspace=workspace))
        registry.register(RefactorTool(
            workspace=workspace,
            symbol_index=symbol_index,
            diff_store=diff_store,
            dep_graph=dep_graph,
        ))
        registry.register(ReferenceFinderTool(
            symbol_index=symbol_index,
            workspace=workspace,
        ))
    if symbol_index is not None and file_context is not None:
        registry.register(GotoDefinitionTool(
            symbol_index=symbol_index,
            file_context=file_context,
        ))
    return registry


__all__ = [
    "ToolRegistry",
    "ToolResult",
    "create_tool_registry",
]
