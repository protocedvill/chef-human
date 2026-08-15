from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chef_human.agent.file_context import FileContextManager
    from chef_human.agent.symbols.index import SymbolIndex

logger = logging.getLogger(__name__)

_SYMBOL_REF_PATTERN = re.compile(r"\b([A-Z][a-zA-Z0-9_]+(?:\.[A-Z][a-zA-Z0-9_]*)*)\b")

_NOISE: set[str] = {
    "The", "This", "That", "It", "I", "We", "You", "They",
    "Here", "There", "Then", "Than", "Also", "But", "Not",
    "Step", "Steps", "File", "Task", "Note", "Error", "Result",
    "Let", "Yes", "No", "Ok", "Okay", "First", "Second", "Next",
    "Look", "See", "Check", "Need", "Done", "Good", "What", "How",
    "Why", "When", "Where", "Who", "Which", "All", "Each", "Every",
    "Some", "Many", "Much", "More", "Most", "Few", "Any", "Both",
    "Now", "Just", "Also", "Very", "Too", "Already", "Still", "Even",
    "Only", "Really", "Actually", "Basically", "Essentially",
    "Please", "Sorry", "Thanks", "Thank", "Hi", "Hello", "Hey",
    "Sure", "Right", "Wrong", "True", "False", "None", "Zero",
    "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
    "Nine", "Ten", "Last", "Next", "Previous", "Current",
    "Main", "Side", "Top", "Bottom", "Left", "Right", "Center",
    "Back", "Front", "End", "Start", "Begin", "Finish", "Stop",
}


class SymbolRetriever:
    def __init__(
        self,
        index: SymbolIndex,
        file_context: FileContextManager,
    ) -> None:
        self._index = index
        self._file_context = file_context
        self._recently_fetched: set[str] = set()

    def detect_symbol_references(self, text: str) -> list[str]:
        candidates = set(_SYMBOL_REF_PATTERN.findall(text))
        candidates -= _NOISE

        found: list[str] = []
        for name in candidates:
            if name not in self._recently_fetched:
                simple_name = name.split(".")[0]
                entries = self._index.lookup(simple_name)
                if entries:
                    found.append(name)
        return found

    def retrieve(self, symbol_name: str) -> str | None:
        simple_name = symbol_name.split(".")[0]
        entries = self._index.lookup(simple_name)
        if not entries:
            return None

        entry = entries[0]
        file_path = entry.file_path

        self._file_context.get(file_path)

        lines = [
            f"**{entry.symbol.kind.title()}** `{entry.symbol.name}` — `{entry.file_path}:{entry.symbol.line}`",
            "```",
            entry.symbol.signature,
            "```",
        ]
        self._recently_fetched.add(symbol_name)
        return "\n".join(lines)

    def reset_fetched(self) -> None:
        self._recently_fetched.clear()
