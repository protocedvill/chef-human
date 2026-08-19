from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chef_human.tools.fsutil import atomic_write_text
from chef_human.tools.diff import DiffEntry, FileChange, compute_diff
from chef_human.tools.registry import ToolResult

logger = logging.getLogger(__name__)

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

        changes = entry.changes or (
            FileChange(entry.file_path, entry.old_content, entry.new_content),
        )
        error = self._apply_atomically(changes)
        if error is not None:
            self._store.push_redo(entry)
            return ToolResult(success=False, error=error)

        fwd_diff = (
            compute_diff(entry.old_content or "", entry.new_content or "", path=entry.file_path)
            if len(changes) == 1
            else "\n".join(
                compute_diff(c.old_content or "", c.new_content or "", path=c.path)
                for c in changes
            )
        )

        self._store.push_entry(DiffEntry(
            path=entry.file_path,
            diff=fwd_diff or "",
            old_content=entry.old_content,
            new_content=entry.new_content,
            timestamp=0,
            tool_name=entry.tool_name,
            changes=changes,
        ))

        target = changes[0].path if len(changes) == 1 else f"{len(changes)} files"
        output_parts = [f"Redid {entry.tool_name}: restored {target}"]
        if fwd_diff:
            output_parts.append(fwd_diff)

        return ToolResult(output="\n".join(output_parts))

    def _apply_atomically(self, changes: tuple[FileChange, ...]) -> str | None:
        snapshots: dict[Path, str | None] = {}
        try:
            for change in changes:
                resolved = self._workspace.resolve(change.path)
                if resolved not in snapshots:
                    # Capture only on first touch -- a transaction that
                    # writes the same path twice must roll back to the
                    # content before *any* of its changes, not the
                    # intermediate content left by an earlier change in
                    # the same transaction.
                    snapshots[resolved] = (
                        resolved.read_text(encoding="utf-8") if resolved.exists() else None
                    )
                if change.new_content is None:
                    resolved.unlink(missing_ok=True)
                else:
                    resolved.parent.mkdir(parents=True, exist_ok=True)
                    atomic_write_text(resolved, change.new_content)
        except Exception as exc:
            rollback_errors: list[str] = []
            for resolved, content in snapshots.items():
                try:
                    if content is None:
                        resolved.unlink(missing_ok=True)
                    else:
                        resolved.parent.mkdir(parents=True, exist_ok=True)
                        atomic_write_text(resolved, content)
                except Exception as rollback_exc:
                    rollback_errors.append(f"{resolved}: {rollback_exc}")
            if rollback_errors:
                logger.error(
                    "Redo rollback failed for %d file(s) after transaction error (%s): %s",
                    len(rollback_errors), exc, "; ".join(rollback_errors),
                )
                return (
                    f"Redo transaction failed ({exc}) AND rollback also failed for "
                    f"{len(rollback_errors)} file(s) -- workspace may be left in a "
                    f"partially-applied state: {'; '.join(rollback_errors)}"
                )
            return f"Redo transaction failed and was rolled back: {exc}"
        return None
