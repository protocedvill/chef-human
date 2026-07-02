from __future__ import annotations

import re
from dataclasses import dataclass, field

_TAG_PATTERN = re.compile(r"^\[(decision|file|assumption|question)\]\s*(.+)$", re.IGNORECASE)

_CATEGORY_ORDER = ("decision", "file", "assumption", "question", "note")

_CATEGORY_LABELS = {
    "decision": "Decisions",
    "file": "Files touched",
    "assumption": "Assumptions",
    "question": "Open questions",
    "note": "Notes",
}

_EMPTY_HINT = (
    "(empty -- use ## Scratchpad: [decision|file|assumption|question] <note> to add notes)"
)


@dataclass
class Scratchpad:
    """Persistent, append-only working memory for a task.

    Unlike a single free-text buffer that gets overwritten by every model
    update, entries accumulate across turns -- and survive re-planning -- so
    the agent doesn't lose track of decisions, files touched, assumptions,
    or open questions it already worked out.
    """

    entries: dict[str, list[str]] = field(
        default_factory=lambda: {c: [] for c in _CATEGORY_ORDER}
    )

    def add_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        match = _TAG_PATTERN.match(line)
        category, text = (match.group(1).lower(), match.group(2).strip()) if match else ("note", line)
        bucket = self.entries.setdefault(category, [])
        if text not in bucket:
            bucket.append(text)

    def add_lines(self, lines: list[str]) -> None:
        for line in lines:
            self.add_line(line)

    def is_empty(self) -> bool:
        return not any(self.entries.values())

    def render(self) -> str:
        if self.is_empty():
            return _EMPTY_HINT
        parts: list[str] = []
        for category in _CATEGORY_ORDER:
            items = self.entries.get(category, [])
            if not items:
                continue
            parts.append(f"{_CATEGORY_LABELS[category]}:")
            parts.extend(f"  - {item}" for item in items)
        return "\n".join(parts)
