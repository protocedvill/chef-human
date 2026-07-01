from __future__ import annotations

from typing import TYPE_CHECKING

from chef_human.tools.filesystem import (
    EditTool,
    GlobTool,
    GrepTool,
    LsTool,
    LsTreeTool,
    ReadTool,
    WriteTool,
)
from chef_human.tools.registry import ToolRegistry, ToolResult
from chef_human.tools.shell import BashTool
from chef_human.tools.user import AskUserTool, FinishTool

if TYPE_CHECKING:
    from chef_human.agent.workspace import WorkspaceManager


def create_tool_registry(workspace: WorkspaceManager) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ReadTool(workspace))
    registry.register(WriteTool(workspace))
    registry.register(EditTool(workspace))
    registry.register(GrepTool(workspace))
    registry.register(GlobTool(workspace))
    registry.register(LsTool(workspace))
    registry.register(LsTreeTool(workspace))
    registry.register(BashTool(workspace))
    registry.register(AskUserTool())
    registry.register(FinishTool())
    return registry


__all__ = [
    "ToolRegistry",
    "ToolResult",
    "create_tool_registry",
]
