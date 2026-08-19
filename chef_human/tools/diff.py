from __future__ import annotations

import difflib
import time
from dataclasses import dataclass


def compute_diff(old_content: str, new_content: str, path: str = "") -> str:
    """Return a unified-diff string suitable for LLM consumption.

    Uses difflib.unified_diff with 3 lines of context.
    Wraps the output in ```diff ... ``` fences.
    Returns empty string when old and new are identical.
    """
    if old_content == new_content:
        return ""

    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff_lines = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{path}" if path else "a",
            tofile=f"b/{path}" if path else "b",
            n=3,
        )
    )

    if not diff_lines:
        return ""

    joined = "".join(diff_lines)
    return f"```diff\n{joined}```\n"


@dataclass
class MatchResult:
    matched_text: str
    ratio: float
    start_line: int
    end_line: int


def find_closest_match(
    old_string: str,
    content: str,
    min_ratio: float = 0.75,
) -> MatchResult | None:
    """Search content for the closest match to old_string via SequenceMatcher.

    Uses a windowed strategy: for every line in content that shares a common
    substring with old_string's first content line, extract a window of
    old_string's line count + 5 lines and score it with SequenceMatcher.
    Returns the best match above min_ratio, or None.
    """
    if not old_string or not content:
        return None

    old_lines = old_string.splitlines()
    if not old_lines:
        return None

    content_lines = content.splitlines(keepends=False)
    if not content_lines:
        return None

    first_line = old_lines[0].strip()
    if not first_line:
        return None

    needle_len = len(old_lines)

    best: MatchResult | None = None
    candidate_lines: list[int] = []

    # Find candidate anchor lines by SequenceMatcher ratio on single lines
    for i, candidate in enumerate(content_lines):
        line_ratio = difflib.SequenceMatcher(None, first_line, candidate.strip()).ratio()
        if line_ratio > 0.5:
            candidate_lines.append(i)

    for i in candidate_lines:
        start = i
        end = min(len(content_lines), start + needle_len)
        if end - start < needle_len:
            start = max(0, end - needle_len)
        window = "\n".join(content_lines[start:end])

        ratio = difflib.SequenceMatcher(None, old_string, window).ratio()
        if ratio >= min_ratio and (best is None or ratio > best.ratio):
            best = MatchResult(
                matched_text=window,
                ratio=ratio,
                start_line=start + 1,
                end_line=end,
            )

    return best


@dataclass
class FileChange:
    path: str
    old_content: str | None
    new_content: str | None


@dataclass
class DiffEntry:
    path: str
    diff: str
    old_content: str | None
    new_content: str | None
    timestamp: float
    tool_name: str
    changes: tuple[FileChange, ...] = ()


@dataclass
class RedoEntry:
    file_path: str
    old_content: str | None
    new_content: str | None
    tool_name: str
    changes: tuple[FileChange, ...] = ()


class DiffStore:
    """Session-level store of file diffs produced by write/edit tools."""

    def __init__(self) -> None:
        self._entries: list[DiffEntry] = []
        self._redo_stack: list[RedoEntry] = []

    def record(
        self,
        path: str,
        diff: str,
        tool_name: str,
        old_content: str | None = None,
        new_content: str | None = None,
    ) -> None:
        """Record a single-file change. Whether to record at all is decided
        by `diff` being non-empty -- old_content/new_content are optional
        metadata here, not the record/skip signal (unlike record_transaction,
        where comparing old/new content *is* the signal, since it has no
        separately-supplied diff string to trust instead)."""
        if not diff:
            return
        self._append_entry(
            DiffEntry(
                path=path,
                diff=diff,
                old_content=old_content,
                new_content=new_content,
                timestamp=time.time(),
                tool_name=tool_name,
            )
        )

    def record_transaction(
        self,
        changes: list[FileChange],
        tool_name: str,
        *,
        label: str | None = None,
    ) -> None:
        # Copy rather than reuse the caller's FileChange instances -- they're
        # a mutable dataclass, and storing the caller's own objects by
        # reference would let a later mutation on the caller's side silently
        # corrupt already-recorded history.
        effective = [
            FileChange(change.path, change.old_content, change.new_content)
            for change in changes
            if change.old_content != change.new_content
        ]
        if not effective:
            return
        diffs = [
            compute_diff(
                change.old_content or "",
                change.new_content or "",
                path=change.path,
            )
            for change in effective
        ]
        path = label or (
            effective[0].path
            if len(effective) == 1
            else f"transaction:{tool_name}:{len(effective)}-files"
        )
        self._append_entry(
            DiffEntry(
                path=path,
                diff="\n".join(diff for diff in diffs if diff),
                old_content=(effective[0].old_content if len(effective) == 1 else None),
                new_content=(effective[0].new_content if len(effective) == 1 else None),
                timestamp=time.time(),
                tool_name=tool_name,
                changes=tuple(effective),
            )
        )

    def _append_entry(self, entry: DiffEntry) -> None:
        self._entries.append(entry)
        self._redo_stack.clear()

    def _last_index(self, path: str | None) -> int | None:
        """Index of the most recent entry matching `path` (or the very last
        entry if path is None), shared by last()/pop_last() so the two can't
        drift on what "matches" means. get_all() doesn't need an index but
        reuses the same `path is None or e.path == path` predicate below."""
        if path is None:
            return len(self._entries) - 1 if self._entries else None
        for i in range(len(self._entries) - 1, -1, -1):
            if self._entries[i].path == path:
                return i
        return None

    def get_all(self, path: str | None = None) -> list[DiffEntry]:
        return [e for e in self._entries if path is None or e.path == path]

    def last(self, path: str | None = None) -> DiffEntry | None:
        index = self._last_index(path)
        return self._entries[index] if index is not None else None

    def pop_last(self, path: str | None = None) -> DiffEntry | None:
        index = self._last_index(path)
        return self._entries.pop(index) if index is not None else None

    def get_summary(self) -> str:
        if not self._entries:
            return "No changes yet."
        lines: list[str] = []
        seen: set[str] = set()
        for entry in self._entries:
            key = f"{entry.path}:{entry.tool_name}"
            if key not in seen:
                seen.add(key)
                lines.append(f"  {entry.tool_name}: {entry.path}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._entries.clear()
        self._redo_stack.clear()

    def push_redo(self, entry: RedoEntry) -> None:
        self._redo_stack.append(entry)

    def pop_redo(self) -> RedoEntry | None:
        if not self._redo_stack:
            return None
        return self._redo_stack.pop()

    def clear_redo(self) -> None:
        self._redo_stack.clear()

    def push_entry(self, entry: DiffEntry) -> None:
        """Append an entry without clearing the redo stack (used by RedoTool)."""
        self._entries.append(entry)
