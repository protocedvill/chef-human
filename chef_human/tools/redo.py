from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chef_human.tools.registry import ToolResult

if TYPE_CHECKING:
    from chef_human.agent.workspace import WorkspaceManager
    from chef_human.tools.diff import DiffStore


class RedoTool:
    name = "redo"
    description = "Reapply the most recently undone change. Reverses the last undo operation."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
    }

    def __init__(self, workspace: WorkspaceManager, diff_store: DiffStore) -> None:
        self._workspace = workspace
        self._store = diff_store

    async def run(self) -> ToolResult:
        entry = self._store.pop_redo()
        if entry is None:
            return ToolResult(success=False, error="Nothing to redo.")

        resolved = self._workspace.resolve(entry.file_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(entry.old_content, encoding="utf-8")

        from chef_human.tools.diff import compute_diff

        fwd_diff = compute_diff(entry.new_content, entry.old_content, path=entry.file_path)

        output_parts = [
            f"Redid {entry.tool_name}: restored {entry.file_path}",
        ]
        if fwd_diff:
            output_parts.append(fwd_diff)

        return ToolResult(output="\n".join(output_parts))
