from __future__ import annotations

import json
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

        is_batch = entry.file_path.startswith("batch:")

        if is_batch:
            old_map: dict[str, str] = json.loads(entry.old_content)
            new_map: dict[str, str] = json.loads(entry.new_content)
            current_map: dict[str, str] = {}
            for fp in old_map:
                resolved = self._workspace.resolve(fp)
                current = resolved.read_text(encoding="utf-8") if resolved.exists() else ""
                current_map[fp] = current
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_text(old_map[fp], encoding="utf-8")

            combined_diff_parts: list[str] = []
            for fp in old_map:
                from chef_human.tools.diff import compute_diff
                diff = compute_diff(new_map.get(fp, ""), old_map[fp], path=fp)
                if diff:
                    combined_diff_parts.append(diff)

            from chef_human.tools.diff import DiffEntry
            self._store.push_entry(DiffEntry(
                path=entry.file_path,
                diff="\n\n".join(combined_diff_parts),
                old_content=json.dumps(current_map),
                new_content=json.dumps(old_map),
                timestamp=0,
                tool_name=entry.tool_name,
            ))

            return ToolResult(
                output=f"Redid {entry.tool_name}: restored {len(old_map)} file{'s' if len(old_map) != 1 else ''}"
            )

        resolved = self._workspace.resolve(entry.file_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        current_content = resolved.read_text(encoding="utf-8") if resolved.exists() else ""
        resolved.write_text(entry.old_content, encoding="utf-8")

        from chef_human.tools.diff import compute_diff, DiffEntry

        fwd_diff = compute_diff(entry.new_content, entry.old_content, path=entry.file_path)

        self._store.push_entry(DiffEntry(
            path=entry.file_path,
            diff=fwd_diff or "",
            old_content=current_content,
            new_content=entry.old_content,
            timestamp=0,
            tool_name=entry.tool_name,
        ))

        output_parts = [
            f"Redid {entry.tool_name}: restored {entry.file_path}",
        ]
        if fwd_diff:
            output_parts.append(fwd_diff)

        return ToolResult(output="\n".join(output_parts))
