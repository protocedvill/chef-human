from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, Protocol

from chef_human.llm.backend import ToolDefinition


@dataclass
class ToolResult:
    success: bool = True
    output: str = ""
    error: str | None = None


class MutationScope(str, Enum):
    none = "none"
    path = "path"
    workspace = "workspace"
    history = "history"


class ReadRequirement(str, Enum):
    none = "none"
    explicit_path = "explicit_path"
    tool_snapshot = "tool_snapshot"
    shell_guardrail = "shell_guardrail"
    transaction_history = "transaction_history"


@dataclass(frozen=True)
class ToolPolicy:
    mutation_scope: MutationScope = MutationScope.none
    read_requirement: ReadRequirement = ReadRequirement.none
    path_argument: str | None = None

    @property
    def mutates(self) -> bool:
        return self.mutation_scope != MutationScope.none


# One auditable classification for every built-in tool. Direct file editors
# require model-visible reads; bulk tools snapshot their targets internally;
# undo/redo operate on recorded snapshots; shell commands are only guarded.
TOOL_POLICIES: dict[str, ToolPolicy] = {
    "ask_user": ToolPolicy(),
    "bash": ToolPolicy(MutationScope.workspace, ReadRequirement.shell_guardrail),
    "edit": ToolPolicy(MutationScope.path, ReadRequirement.explicit_path, "path"),
    "finish": ToolPolicy(),
    "glob": ToolPolicy(),
    "grep": ToolPolicy(),
    "ls": ToolPolicy(),
    "ls_tree": ToolPolicy(),
    "lint_fix": ToolPolicy(MutationScope.workspace, ReadRequirement.tool_snapshot, "path"),
    "read": ToolPolicy(),
    "patch": ToolPolicy(MutationScope.path, ReadRequirement.explicit_path, "path"),
    "redo": ToolPolicy(MutationScope.history, ReadRequirement.transaction_history),
    "undo": ToolPolicy(MutationScope.history, ReadRequirement.transaction_history),
    "view_diff": ToolPolicy(),
    "write": ToolPolicy(MutationScope.path, ReadRequirement.explicit_path, "path"),
    "lookup_symbol": ToolPolicy(),
    "refactor_symbol": ToolPolicy(
        MutationScope.workspace, ReadRequirement.tool_snapshot, "path"
    ),
    "find_references": ToolPolicy(),
    "goto_definition": ToolPolicy(),
}


def get_tool_policy(name: str) -> ToolPolicy:
    return TOOL_POLICIES.get(name, ToolPolicy())


class Tool(Protocol):
    name: str
    description: str
    parameters: dict[str, Any]

    @property
    def run(self) -> Callable[..., Awaitable[ToolResult]]: ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return sorted(self._tools)

    def get_definitions(self) -> list[ToolDefinition]:
        definitions: list[ToolDefinition] = []
        for t in self._tools.values():
            definitions.append(
                ToolDefinition(
                    name=t.name,
                    description=t.description,
                    parameters=t.parameters,
                )
            )
        return definitions
