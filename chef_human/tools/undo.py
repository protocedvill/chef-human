from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from chef_human.tools.diff import FileChange, RedoEntry
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

        changes = entry.changes or (
            FileChange(entry.path, entry.old_content, entry.new_content),
        )
        error = self._apply_atomically(changes, use_new=False)
        if error is not None:
            self._store.push_entry(entry)
            return ToolResult(success=False, error=error)

        self._store.push_redo(RedoEntry(
            file_path=entry.path,
            old_content=entry.old_content,
            new_content=entry.new_content,
            tool_name=entry.tool_name,
            changes=changes,
        ))

        from chef_human.tools.diff import compute_diff

        reverse_diff = (
            compute_diff(entry.new_content or "", entry.old_content or "", path=entry.path)
            if len(changes) == 1
            else ""
        )

        if len(changes) == 1 and changes[0].old_content is None:
            output_parts = [
                f"Undid {entry.tool_name}: deleted {changes[0].path} (was new file)"
            ]
        else:
            target = changes[0].path if len(changes) == 1 else f"{len(changes)} files"
            output_parts = [f"Undid {entry.tool_name}: restored {target}"]
        if reverse_diff:
            output_parts.append(reverse_diff)

        return ToolResult(output="\n".join(output_parts))

    def _apply_atomically(
        self, changes: tuple[FileChange, ...], *, use_new: bool
    ) -> str | None:
        snapshots: dict[Path, str | None] = {}
        try:
            for change in changes:
                resolved = self._workspace.resolve(change.path)
                snapshots[resolved] = (
                    resolved.read_text(encoding="utf-8") if resolved.exists() else None
                )
                content = change.new_content if use_new else change.old_content
                if content is None:
                    resolved.unlink(missing_ok=True)
                else:
                    resolved.parent.mkdir(parents=True, exist_ok=True)
                    resolved.write_text(content, encoding="utf-8")
        except Exception as exc:
            for resolved, content in snapshots.items():
                try:
                    if content is None:
                        resolved.unlink(missing_ok=True)
                    else:
                        resolved.parent.mkdir(parents=True, exist_ok=True)
                        resolved.write_text(content, encoding="utf-8")
                except Exception:
                    pass
            return f"Undo transaction failed and was rolled back: {exc}"
        return None
