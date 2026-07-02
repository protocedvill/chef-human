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
class DiffEntry:
    path: str
    diff: str
    old_content: str | None
    new_content: str | None
    timestamp: float
    tool_name: str


@dataclass
class RedoEntry:
    file_path: str
    old_content: str
    new_content: str
    tool_name: str


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
        if not diff:
            return
        self._entries.append(
            DiffEntry(
                path=path,
                diff=diff,
                old_content=old_content,
                new_content=new_content,
                timestamp=time.time(),
                tool_name=tool_name,
            )
        )
        self._redo_stack.clear()

    def get_all(self, path: str | None = None) -> list[DiffEntry]:
        if path is None:
            return list(self._entries)
        return [e for e in self._entries if e.path == path]

    def last(self, path: str | None = None) -> DiffEntry | None:
        if path is None:
            return self._entries[-1] if self._entries else None
        for entry in reversed(self._entries):
            if entry.path == path:
                return entry
        return None

    def pop_last(self, path: str | None = None) -> DiffEntry | None:
        if not self._entries:
            return None
        if path is None:
            return self._entries.pop()
        for i in range(len(self._entries) - 1, -1, -1):
            if self._entries[i].path == path:
                return self._entries.pop(i)
        return None

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

    def push_redo(self, entry: RedoEntry) -> None:
        self._redo_stack.append(entry)

    def pop_redo(self) -> RedoEntry | None:
        if not self._redo_stack:
            return None
        return self._redo_stack.pop()

    def clear_redo(self) -> None:
        self._redo_stack.clear()
