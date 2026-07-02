from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Final, Protocol

from tree_sitter import Language, Parser, Query, QueryCursor

from chef_human.agent.symbols.grammars import GrammarLoader

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


_TS_QUERIES: dict[str, dict[str, str]] = {
    "python": {
        "function": "(function_definition name: (identifier) @name) @def",
        "class": "(class_definition name: (identifier) @name) @def",
        "import": "(import_statement (dotted_name) @name) @def",
        "from_import": (
            "(import_from_statement"
            " module_name: (dotted_name) @module"
            " name: (dotted_name) @name) @def"
        ),
    },
    "javascript": {
        "function": "(function_declaration name: (identifier) @name) @def",
        "class": "(class_declaration name: (identifier) @name) @def",
        "method": "(method_definition name: (property_identifier) @name) @def",
        "import": (
            "(import_statement"
            " (import_clause (named_imports (import_specifier name: (identifier) @name)))"
            " source: (string) @source) @def"
        ),
    },
    "typescript": {
        "function": "(function_declaration name: (identifier) @name) @def",
        "class": "(class_declaration name: (type_identifier) @name) @def",
        "method": "(method_definition name: (property_identifier) @name) @def",
        "interface": "(interface_declaration name: (type_identifier) @name) @def",
        "type_alias": "(type_alias_declaration name: (type_identifier) @name) @def",
        "import": (
            "(import_statement"
            " (import_clause (named_imports (import_specifier name: (identifier) @name)))"
            " source: (string) @source) @def"
        ),
    },
    "rust": {
        "function": "(function_item name: (identifier) @name) @def",
        "struct": "(struct_item name: (type_identifier) @name) @def",
        "enum": "(enum_item name: (type_identifier) @name) @def",
        "trait": "(trait_item name: (type_identifier) @name) @def",
        "impl": "(impl_item type: (type_identifier) @name) @def",
        "use": "(use_declaration (scoped_identifier name: (identifier) @name)) @def",
    },
    "go": {
        "function": "(function_declaration name: (identifier) @name) @def",
        "method": (
            "(method_declaration"
            " receiver: (parameter_list) @receiver"
            " name: (field_identifier) @name) @def"
        ),
        "struct": (
            "(type_declaration"
            " (type_spec name: (type_identifier) @name"
            " type: (struct_type))) @def"
        ),
        "interface": (
            "(type_declaration"
            " (type_spec name: (type_identifier) @name"
            " type: (interface_type))) @def"
        ),
        "import": "(import_declaration (import_spec path: (interpreted_string_literal) @name)) @def",
    },
    "java": {
        "class": "(class_declaration name: (identifier) @name) @def",
        "interface": "(interface_declaration name: (identifier) @name) @def",
        "method": "(method_declaration name: (identifier) @name) @def",
        "import": "(import_declaration (scoped_identifier name: (identifier) @name)) @def",
    },
}


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
        ".tsx": "typescript",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".rb": "ruby",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
    }

    def __init__(self, grammar_loader: GrammarLoader | None = None) -> None:
        try:
            import tree_sitter  # noqa: F401
        except ImportError:
            raise ImportError(
                "tree-sitter is required for TreeSitterExtractor. "
                "Install: pip install tree-sitter"
            )
        self._loader = grammar_loader or GrammarLoader()
        self._compiled: dict[str, dict[str, Query]] = {}

    def extract(self, file_path: str, content: str) -> list[Symbol]:
        ext = os.path.splitext(file_path)[1].lower()
        lang_name = self._LANG_MAP.get(ext)
        if lang_name is None:
            return []

        language = self._loader.load(lang_name)
        if language is None:
            return []

        parser = Parser(language)
        tree = parser.parse(content.encode("utf-8"))
        if tree is None:
            return []
        root = tree.root_node
        source_bytes = content.encode("utf-8")

        symbols: list[Symbol] = []
        queries = self._get_queries(lang_name, language)

        for kind, query in queries.items():
            cursor = QueryCursor(query)
            for _pi, captures in cursor.matches(root):
                name_nodes = captures.get("name")
                def_nodes = captures.get("def")
                if not name_nodes or not def_nodes:
                    continue
                name = source_bytes[name_nodes[0].start_byte:name_nodes[0].end_byte].decode("utf-8")
                def_node = def_nodes[0]
                sig = _header_signature(source_bytes, def_node)
                symbols.append(Symbol(
                    name=name,
                    kind=kind,
                    line=def_node.start_point[0] + 1,
                    signature=sig,
                ))

        return symbols

    def _get_queries(self, lang_name: str, language: Language) -> dict[str, Query]:
        if lang_name not in self._compiled:
            self._compiled[lang_name] = {
                kind: Query(language, qs)
                for kind, qs in _TS_QUERIES.get(lang_name, {}).items()
            }
        return self._compiled[lang_name]


# Regex to detect definition lines that should terminate signature reconstruction.
# Matches lines like `def foo`, `async def foo`, `pub fn foo`, `class Foo`, etc.
_DEF_LINE_RE: Final = re.compile(
    r"^\s*(?:(?:async|pub|unsafe|public|private|protected|static|abstract|virtual|override)"
    r"\s+)*"
    r"(?:def|class|fn|func|function|interface|type|struct|enum|trait|import|use|impl|constructor)"
    r"\b"
)


def _header_signature(source_bytes: bytes, node: Any) -> str:
    start = node.start_byte

    # Walk up to include Python decorators
    ancestor = node.parent
    while ancestor is not None:
        if ancestor.type == "decorated_definition":
            start = ancestor.start_byte
            ancestor = ancestor.parent
        else:
            break

    source = source_bytes[start:node.end_byte].decode("utf-8").strip()
    lines = source.splitlines()
    header_end = 0
    for i, line in enumerate(lines):
        if _DEF_LINE_RE.search(line):
            header_end = i
            break
    header_lines = lines[:header_end + 1]
    return "\n".join(header_lines) if len(header_lines) > 1 else header_lines[0]


class CompositeExtractor:
    def __init__(self) -> None:
        self._regex = RegexExtractor()
        self._ts: TreeSitterExtractor | None = None
        try:
            self._ts = TreeSitterExtractor()
        except ImportError:
            logger.info("tree-sitter not available, CompositeExtractor using regex only")

    def extract(self, file_path: str, content: str) -> list[Symbol]:
        if self._ts is not None:
            symbols = self._ts.extract(file_path, content)
            if symbols:
                return symbols
        return self._regex.extract(file_path, content)


def create_extractor() -> CompositeExtractor:
    return CompositeExtractor()
