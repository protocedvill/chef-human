from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chef_human.tools.registry import ToolResult

if TYPE_CHECKING:
    from chef_human.agent.workspace import WorkspaceManager
    from chef_human.tools.diff import DiffStore


class UndoTool:
    name = "undo"
    description = "Undo the last write or edit, restoring the file to its previous content."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Optional file path. If set, undo the last change to this specific file.",
                "default": None,
            },
        },
    }

    def __init__(self, workspace: WorkspaceManager, diff_store: DiffStore) -> None:
        self._workspace = workspace
        self._store = diff_store

    async def run(self, path: str | None = None) -> ToolResult:
        entry = self._store.pop_last(path=path)
        if entry is None:
            return ToolResult(success=False, error="Nothing to undo.")

        if entry.old_content is None:
            resolved = self._workspace.resolve(entry.path)
            resolved.unlink(missing_ok=True)
            return ToolResult(output=f"Undid {entry.tool_name}: deleted {entry.path} (was new file)")

        resolved = self._workspace.resolve(entry.path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(entry.old_content, encoding="utf-8")

        from chef_human.tools.diff import compute_diff

        reverse_diff = compute_diff(entry.new_content or "", entry.old_content, path=entry.path)

        output_parts = [
            f"Undid {entry.tool_name}: restored {entry.path}",
        ]
        if reverse_diff:
            output_parts.append(reverse_diff)

        return ToolResult(output="\n".join(output_parts))
