from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chef_human.tools.registry import ToolResult

if TYPE_CHECKING:
    from chef_human.tools.diff import DiffStore


class ViewDiffTool:
    name = "view_diff"
    description = "Show unified diffs of file changes made so far in this task."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Optional path to filter by. Omit to see all changes.",
                "default": None,
            },
        },
    }

    def __init__(self, diff_store: DiffStore) -> None:
        self._store = diff_store

    async def run(self, path: str | None = None) -> ToolResult:
        entries = self._store.get_all(path=path)
        if not entries:
            return ToolResult(output="No changes yet.")

        parts: list[str] = []
        for entry in entries:
            parts.append(f"### {entry.tool_name}: {entry.path}")
            parts.append(entry.diff)
        return ToolResult(output="\n".join(parts))
