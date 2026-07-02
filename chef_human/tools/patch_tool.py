from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from chef_human.tools.diff import compute_diff
from chef_human.tools.registry import ToolResult

if TYPE_CHECKING:
    from chef_human.agent.workspace import WorkspaceManager
    from chef_human.tools.diff import DiffStore


_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*")


def _strip_prefix_and_newline(line: str) -> str:
    """Remove diff prefix char (space, -, +) and trailing newline."""
    return line[1:].rstrip("\n\r")


def _strip_prefix(line: str) -> str:
    """Remove diff prefix char (space, -, +) keeping trailing newline."""
    return line[1:]


def _apply_patch(file_content: str, patch_text: str, reverse: bool = False) -> str | None:
    lines = file_content.splitlines(keepends=True)
    patch_lines = patch_text.splitlines()
    hunks = _parse_hunks(patch_lines)

    if not hunks:
        return None

    # Apply hunks in reverse order (bottom-up) to preserve line offsets
    for hunk in reversed(hunks):
        old_start, old_count, new_start, new_count, old_lines, new_lines = hunk

        if reverse:
            old_lines, new_lines = new_lines, old_lines
            old_start, old_count, new_start, new_count = (
                new_start,
                new_count,
                old_start,
                old_count,
            )

        if old_count == 0:
            # Insertion: no old lines to match
            insert_pos = old_start - 1  # 0-indexed
            if insert_pos < 0:
                insert_pos = 0
            stripped = [_strip_prefix(line) for line in new_lines]
            lines[insert_pos:insert_pos] = stripped
            continue

        # Check context at the target position
        start_idx = old_start - 1
        end_idx = start_idx + len(old_lines)

        if start_idx < 0 or end_idx > len(lines):
            return None

        old_stripped = [_strip_prefix_and_newline(x) for x in old_lines]
        content_stripped = [
            ln.rstrip("\n\r") for ln in lines[start_idx:end_idx]
        ]

        if old_stripped != content_stripped:
            return None

        replacement = [_strip_prefix(x) for x in new_lines]
        lines[start_idx:end_idx] = replacement

    return "".join(lines)


def _parse_hunks(patch_lines: list[str]) -> list[tuple[int, int, int, int, list[str], list[str]]]:
    hunks: list[tuple[int, int, int, int, list[str], list[str]]] = []
    current_old_lines: list[str] = []
    current_new_lines: list[str] = []
    in_hunk = False
    old_start = 0
    old_count = 0
    new_start = 0
    new_count = 0

    for line in patch_lines:
        # Normalise line ending
        raw = line + "\n" if not line.endswith("\n") else line

        m = _HUNK_HEADER.match(raw)
        if m:
            if in_hunk and (current_old_lines or current_new_lines):
                hunks.append(
                    (
                        old_start,
                        old_count,
                        new_start,
                        new_count,
                        list(current_old_lines),
                        list(current_new_lines),
                    )
                )
                current_old_lines = []
                current_new_lines = []

            old_start = int(m.group(1))
            old_count = int(m.group(2)) if m.group(2) else 1
            new_start = int(m.group(3))
            new_count = int(m.group(4)) if m.group(4) else 1
            in_hunk = True
            continue

        if not in_hunk:
            continue

        if raw.startswith(" ") or raw.startswith("-"):
            current_old_lines.append(raw)
        if raw.startswith(" ") or raw.startswith("+"):
            current_new_lines.append(raw)

    if in_hunk and (current_old_lines or current_new_lines):
        hunks.append(
            (
                old_start,
                old_count,
                new_start,
                new_count,
                list(current_old_lines),
                list(current_new_lines),
            )
        )

    return hunks


class PatchTool:
    name = "patch"
    description = "Apply a unified diff patch to a file. The patch must be in standard unified diff format."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to patch (absolute or relative to workspace)",
            },
            "patch": {
                "type": "string",
                "description": "Unified diff patch content (e.g. from ```diff ... ``` blocks)",
            },
            "reverse": {
                "type": "boolean",
                "description": "Apply the patch in reverse (like patch -R)",
                "default": False,
            },
        },
        "required": ["path", "patch"],
    }

    def __init__(self, workspace: WorkspaceManager, diff_store: DiffStore | None = None) -> None:
        self._workspace = workspace
        self._diff_store = diff_store

    async def run(self, path: str, patch: str, reverse: bool = False) -> ToolResult:
        resolved = self._workspace.resolve(path)

        if not self._workspace.is_within_workspace(resolved):
            return ToolResult(success=False, error=f"Outside workspace: {path}")

        if not resolved.exists():
            return ToolResult(success=False, error=f"File not found: {path}")

        if not patch.strip():
            return ToolResult(success=False, error="Patch is empty")

        # Strip any leading diff header lines that are not hunks
        patch_text = patch.strip("\n").strip()
        # Remove leading/trailing ```diff ... ``` markers if present
        patch_text = re.sub(r"^```(?:diff)?\s*\n?", "", patch_text)
        patch_text = re.sub(r"\n```\s*$", "", patch_text)

        try:
            old_content = resolved.read_text(encoding="utf-8")
        except Exception as exc:
            return ToolResult(success=False, error=f"Cannot read {path}: {exc}")

        new_content = _apply_patch(old_content, patch_text, reverse=reverse)
        if new_content is None:
            return ToolResult(
                success=False,
                error="Patch application failed: hunk context did not match file content. "
                "The patch may be out of date or malformed.",
            )

        if new_content == old_content:
            resolved.write_text(new_content, encoding="utf-8")
            output = f"Applied patch to {path} (no changes)"
            return ToolResult(output=output)

        try:
            resolved.write_text(new_content, encoding="utf-8")
        except Exception as exc:
            return ToolResult(success=False, error=f"Cannot write {path}: {exc}")

        diff = compute_diff(old_content, new_content, path=path)

        if self._diff_store and diff:
            self._diff_store.record(
                path, diff, "patch", old_content=old_content, new_content=new_content
            )

        direction = "reversed " if reverse else ""
        output_parts: list[str] = [f"Applied {direction}patch to {path}"]
        if diff:
            output_parts.append(diff)

        return ToolResult(output="\n".join(output_parts))
