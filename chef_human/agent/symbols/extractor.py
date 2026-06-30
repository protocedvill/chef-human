from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str
    line: int
    signature: str


class SymbolExtractor(Protocol):
    def extract(self, file_path: str, content: str) -> list[Symbol]: ...


_LANG_PATTERNS: dict[str, list[tuple[str, str]]] = {
    ".py": [
        ("function", r"^\s*(?:async\s+)?def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*\S+)?\s*:"),
        ("class", r"^\s*class\s+(\w+)\s*(?:\([^)]*\))?\s*:"),
        ("import", r"^\s*(?:from\s+\S+\s+)?import\s+\S+"),
    ],
    ".js": [
        ("function", r"(?:async\s+)?function\s+(\w+)\s*\([^)]*\)"),
        ("class", r"class\s+(\w+)"),
    ],
    ".ts": [
        ("function", r"(?:async\s+)?function\s+(\w+)\s*\([^)]*\)"),
        ("class", r"class\s+(\w+)"),
        ("interface", r"interface\s+(\w+)"),
    ],
    ".rs": [
        ("function", r"^\s*(?:pub\s+)?(?:unsafe\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\([^)]*\)"),
        ("struct", r"^\s*(?:pub\s+)?struct\s+(\w+)"),
        ("enum", r"^\s*(?:pub\s+)?enum\s+(\w+)"),
        ("trait", r"^\s*(?:pub\s+)?trait\s+(\w+)"),
    ],
    ".go": [
        ("function", r"^\s*func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\([^)]*\)"),
        ("struct", r"^\s*type\s+(\w+)\s+struct"),
        ("interface", r"^\s*type\s+(\w+)\s+interface"),
    ],
    ".java": [
        ("class", r"^\s*(?:public\s+|private\s+|protected\s+)?(?:abstract\s+)?class\s+(\w+)"),
        ("interface", r"^\s*(?:public\s+)?interface\s+(\w+)"),
        ("method", r"^\s*(?:public\s+|private\s+|protected\s+)?(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*(?:throws\s+\S+)?\s*\{"),
    ],
}

_DEFAULT_PATTERNS: list[tuple[str, str]] = [
    ("function", r"^\s*(?:async\s+)?(?:def|function|fn|func)\s+(\w+)\s*\([^)]*\)"),
    ("class", r"^\s*(?:class|struct|trait|interface)\s+(\w+)"),
]


class RegexExtractor:
    def extract(self, file_path: str, content: str) -> list[Symbol]:
        ext = os.path.splitext(file_path)[1].lower()
        patterns = _LANG_PATTERNS.get(ext, _DEFAULT_PATTERNS)
        symbols: list[Symbol] = []
        for line_num, line in enumerate(content.splitlines(), start=1):
            for kind, pattern in patterns:
                m = re.search(pattern, line)
                if m:
                    name = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0)
                    symbols.append(Symbol(
                        name=name,
                        kind=kind,
                        line=line_num,
                        signature=line.strip(),
                    ))
                    break
        return symbols


class TreeSitterExtractor:
    _LANG_MAP: dict[str, str] = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".rb": "ruby",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
    }

    def __init__(self) -> None:
        try:
            import tree_sitter  # noqa: F401
        except ImportError:
            raise ImportError(
                "tree-sitter is required for TreeSitterExtractor. "
                "Install: pip install tree-sitter"
            )
        self._languages: dict[str, object] = {}

    def extract(self, file_path: str, content: str) -> list[Symbol]:
        ext = os.path.splitext(file_path)[1].lower()
        lang_name = self._LANG_MAP.get(ext)
        if lang_name is None:
            return []
        language = self._get_language(lang_name)
        if language is None:
            return []
        return []

    def _get_language(self, name: str) -> object | None:
        if name not in self._languages:
            try:
                import importlib
                mod = importlib.import_module(f"tree_sitter_{name}")
                self._languages[name] = mod.language()
            except ImportError:
                return None
        return self._languages[name]


def create_extractor() -> SymbolExtractor:
    try:
        return TreeSitterExtractor()
    except ImportError:
        logger.info("tree-sitter not available, using regex extractor")
        return RegexExtractor()
