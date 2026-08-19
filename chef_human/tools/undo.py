from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chef_human.tools.fsutil import set_file_content
from chef_human.tools.diff import FileChange, RedoEntry
from chef_human.tools.registry import ToolResult

logger = logging.getLogger(__name__)

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
        error = self._apply_atomically(changes)
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

    def _apply_atomically(self, changes: tuple[FileChange, ...]) -> str | None:
        # No use_new toggle -- undo only ever restores old_content (that's
        # its whole job; redo.py's own _apply_atomically covers the
        # opposite direction). A prior version of this method carried a
        # use_new parameter that was always passed False by its one caller.
        snapshots: dict[Path, str | None] = {}
        try:
            for change in changes:
                resolved = self._workspace.resolve(change.path)
                if resolved not in snapshots:
                    # Capture only on first touch -- see redo.py's identical
                    # comment for why: a transaction touching the same path
                    # twice must roll back to the pre-transaction content,
                    # not an intermediate one.
                    snapshots[resolved] = (
                        resolved.read_text(encoding="utf-8") if resolved.exists() else None
                    )
                set_file_content(resolved, change.old_content)
        except Exception as exc:
            rollback_errors: list[str] = []
            for resolved, content in snapshots.items():
                try:
                    set_file_content(resolved, content)
                except Exception as rollback_exc:
                    rollback_errors.append(f"{resolved}: {rollback_exc}")
            if rollback_errors:
                logger.error(
                    "Undo rollback failed for %d file(s) after transaction error (%s): %s",
                    len(rollback_errors), exc, "; ".join(rollback_errors),
                )
                return (
                    f"Undo transaction failed ({exc}) AND rollback also failed for "
                    f"{len(rollback_errors)} file(s) -- workspace may be left in a "
                    f"partially-applied state: {'; '.join(rollback_errors)}"
                )
            return f"Undo transaction failed and was rolled back: {exc}"
        return None
