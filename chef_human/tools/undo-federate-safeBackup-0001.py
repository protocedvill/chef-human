from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from chef_human.tools.diff import RedoEntry
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

        is_batch = entry.path.startswith("batch:")

        if is_batch:
            old_map: dict[str, str] = json.loads(entry.old_content)
            current_map: dict[str, str] = {}
            for fp, old_content in old_map.items():
                resolved = self._workspace.resolve(fp)
                current = resolved.read_text(encoding="utf-8") if resolved.exists() else ""
                current_map[fp] = current
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_text(old_content, encoding="utf-8")

            self._store.push_redo(RedoEntry(
                file_path=entry.path,
                old_content=json.dumps(current_map),
                new_content=entry.old_content,
                tool_name=entry.tool_name,
            ))

            output_parts = [
                f"Undid {entry.tool_name}: restored {len(old_map)} file{'s' if len(old_map) != 1 else ''}",
            ]
            return ToolResult(output="\n".join(output_parts))

        resolved = self._workspace.resolve(entry.path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        current_content = resolved.read_text(encoding="utf-8") if resolved.exists() else ""
        resolved.write_text(entry.old_content, encoding="utf-8")

        self._store.push_redo(RedoEntry(
            file_path=entry.path,
            old_content=current_content,
            new_content=entry.old_content,
            tool_name=entry.tool_name,
        ))

        from chef_human.tools.diff import compute_diff

        reverse_diff = compute_diff(entry.new_content or "", entry.old_content, path=entry.path)

        output_parts = [
            f"Undid {entry.tool_name}: restored {entry.path}",
        ]
        if reverse_diff:
            output_parts.append(reverse_diff)

        return ToolResult(output="\n".join(output_parts))
