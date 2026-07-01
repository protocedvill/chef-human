from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ToolResult:
    success: bool = True
    output: str = ""
    error: str | None = None


class Tool(Protocol):
    name: str
    description: str
    parameters: dict[str, Any]

    async def run(self, **kwargs: Any) -> ToolResult: ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return sorted(self._tools)

    def get_definitions(self) -> list[dict[str, Any]]:
        definitions: list[dict[str, Any]] = []
        for t in self._tools.values():
            definitions.append({
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            })
        return definitions
