# Phase 3.1: Repository Indexing

**Goal**: Build a persistent, queryable symbol index across the entire codebase. Replace the current stub `TreeSitterExtractor` with real AST-based extraction, add file-level dependency graphs, and enable on-demand symbol retrieval so the agent can look up definitions when it encounters an unknown symbol.

**Prerequisites**: Phases 1.1–2.3 complete (backends, context manager, tools, ReAct loop, auto-lint, session management, token tracking).

---

## Current State

| Component | Status |
|-----------|--------|
| `TreeSitterExtractor` | **Working** — real AST queries for all 6 languages (Python, JS, TS, Rust, Go, Java) |
| `RegexExtractor` | Working — line-based regex fallback for unsupported languages |
| `CompositeExtractor` | **New** — tries TreeSitterExtractor first, falls back to RegexExtractor if no symbols found |
| `SymbolIndex` | **Working** — persistent index with build, refresh, lookup by name/file/prefix, search |
| `DependencyGraph` | **New** — regex-based import extraction, file→file adjacency map with transitive deps |
| `SymbolRetriever` | **New** — detects CAPS symbol references in conversation, fetches definitions, deduplicates |
| `RepoMap.generate()` | Working — scans workspace files, extracts symbols inline |
| `ContextAssembler` | **Enhanced** — accepts optional symbol_index, dep_graph, symbol_retriever; injects `## Related Symbols` in assemble() |
| `create_context_assembler()` | **Enhanced** — creates and wires all symbol components, pre-builds index + graph on init |
| `config.py` | **Enhanced** — added `max_index_files: int = 500` |

---

## Task List

- [x] **3.1.1** Tree-sitter grammar setup & loading (download/install language grammars, lazy load)
- [x] **3.1.2** Full Tree-sitter AST queries for Python, JS/TS, Rust, Go, Java (function, class, method, interface, struct, enum, trait, import)
- [x] **3.1.3** Persistent symbol index (build, query by name/kind/file, incremental updates)
- [x] **3.1.4** File-level dependency graph (extract imports/requires, build adjacency map)
- [x] **3.1.5** On-demand symbol retrieval protocol (model mentions symbol → agent looks up definition → injects into context)
- [x] **3.1.6** Integration into ContextAssembler & agent loop (symbol lookup as implicit step during context assembly)
- [x] **3.1.7** Tests for all new functionality

---

## Task 3.1.1: Tree-Sitter Grammar Setup & Loading

**File to create:** `chef_human/agent/symbols/grammars.py`

Tree-sitter requires compiled `.so` grammar files per language, provided by `pip` packages (`tree-sitter-python`, `tree-sitter-javascript`, etc.). The grammar loader caches loaded languages and returns `None` gracefully when a grammar is missing.

### GrammarLoader

```python
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_LANGUAGE_PACKAGES: dict[str, str] = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "rust": "tree_sitter_rust",
    "go": "tree_sitter_go",
    "java": "tree_sitter_java",
    "ruby": "tree_sitter_ruby",
    "c": "tree_sitter_c",
    "cpp": "tree_sitter_cpp",
}


class GrammarLoader:
    """Loads and caches tree-sitter language grammars."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._checked: set[str] = set()

    def load(self, language: str) -> Any | None:
        """Load a tree-sitter Language for the given language name.
        Returns None if the grammar package is not installed.
        """
        if language in self._cache:
            return self._cache[language]
        if language in self._checked:
            return None

        pkg_name = _LANGUAGE_PACKAGES.get(language)
        if pkg_name is None:
            self._checked.add(language)
            logger.debug("No grammar package mapping for language: %s", language)
            return None

        try:
            import importlib
            mod = importlib.import_module(pkg_name)
            lang = mod.language()
            self._cache[language] = lang
            logger.info("Loaded tree-sitter grammar for %s", language)
            return lang
        except ImportError:
            self._checked.add(language)
            logger.info(
                "tree-sitter grammar for '%s' not installed. "
                "Install with: pip install tree-sitter-%s",
                language, language,
            )
            return None

    @staticmethod
    def supported_languages() -> list[str]:
        return list(_LANGUAGE_PACKAGES.keys())

    def loaded_languages(self) -> list[str]:
        return list(self._cache.keys())
```

### Fallback Strategy

| Scenario | Behavior |
|----------|----------|
| `tree-sitter` core not installed | `TreeSitterExtractor.__init__` raises `ImportError` → factory falls back to `RegexExtractor` for all files |
| Specific grammar missing (e.g., `tree-sitter-rust`) | `GrammarLoader.load("rust")` returns `None` → `TreeSitterExtractor` falls back to `RegexExtractor` for `.rs` files only |
| Grammar load fails at runtime | Log warning, return `None`, fall back per-file |

### Acceptance Criteria

- `GrammarLoader.load("python")` returns `Language` when `tree-sitter-python` is installed
- `GrammarLoader.load("python")` returns cached object on second call
- `GrammarLoader.load("rust")` returns `None` when `tree-sitter-rust` not installed
- `GrammarLoader.load("nonexistent")` returns `None` silently
- `supported_languages()` returns all known language keys

---

## Task 3.1.2: Full Tree-Sitter AST Queries

**File to modify:** `chef_human/agent/symbols/extractor.py`

Replace the stub `TreeSitterExtractor.extract()` with real AST queries using tree-sitter's S-expression query language. Each language gets a set of named queries that capture symbol name, kind, line range, and full signature.

### Query Design per Language

**Python:**
| Kind | Tree-sitter query captures |
|------|---------------------------|
| `function` | `function_definition` → name, parameters, return_type, body |
| `class` | `class_definition` → name, bases, body |
| `method` | `function_definition` inside `class_definition` body |
| `import` | `import_statement` → dotted_name |
| `from_import` | `import_from_statement` → module, name |

**JavaScript/TypeScript:**
| Kind | Tree-sitter query captures |
|------|---------------------------|
| `function` | `function_declaration` → name |
| `class` | `class_declaration` → name |
| `method` | `method_definition` inside class body |
| `interface` (TS) | `interface_declaration` → name |
| `type_alias` (TS) | `type_alias_declaration` → name |
| `import` | `import_statement` → source, names |

**Rust:**
| Kind | Tree-sitter query captures |
|------|---------------------------|
| `function` | `function_item` → name, parameters, return_type |
| `struct` | `struct_item` → name, fields |
| `enum` | `enum_item` → name, variants |
| `trait` | `trait_item` → name |
| `impl` | `impl_item` → trait, type |
| `use` | `use_declaration` → path |

**Go:**
| Kind | Tree-sitter query captures |
|------|---------------------------|
| `function` | `function_declaration` → name |
| `method` | `method_declaration` → receiver, name |
| `struct` | `type_spec` with `struct_type` → name |
| `interface` | `type_spec` with `interface_type` → name |
| `import` | `import_declaration` → path |

**Java:**
| Kind | Tree-sitter query captures |
|------|---------------------------|
| `class` | `class_declaration` → name, modifiers |
| `interface` | `interface_declaration` → name |
| `method` | `method_declaration` → name, modifiers, return_type, parameters |
| `constructor` | `constructor_declaration` → name, parameters |

### Signature Reconstruction

The AST gives us node positions in the source text. To produce the signature string:

```python
def _reconstruct_signature(
    source_bytes: bytes,
    node: Any,
    capture_names: set[str],
    context_lines: int = 2,
) -> str:
    """Reconstruct a human-readable signature from AST node positions.
    Includes decorators (Python), visibility modifiers (Java), generics (Rust/TS).
    """
    start_byte = node.start_byte
    # Walk back to include decorators (Python) or modifiers (Java)
    while start_byte > 0:
        prev_char = source_bytes[start_byte - 1:start_byte]
        if prev_char in (b" ", b"\t"):
            start_byte -= 1
        elif prev_char == b"\n":
            # Check if previous line is a decorator or annotation
            line_start = source_bytes.rfind(b"\n", 0, start_byte - 1) + 1
            line = source_bytes[line_start:start_byte - 1].decode("utf-8")
            if line.strip().startswith("@") or line.strip().startswith("//"):
                start_byte = line_start
            else:
                break
        else:
            break

    end_byte = min(node.end_byte, len(source_bytes))
    return source_bytes[start_byte:end_byte].decode("utf-8").strip()
```

### Updated `TreeSitterExtractor`

```python
class TreeSitterExtractor:
    _EXT_TO_LANG: dict[str, str] = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".rs": "rust", ".go": "go", ".java": "java",
        ".rb": "ruby", ".c": "c", ".cpp": "cpp", ".hpp": "cpp", ".h": "c",
    }

    def __init__(self, grammar_loader: GrammarLoader | None = None) -> None:
        try:
            import tree_sitter  # noqa: F401
        except ImportError:
            raise ImportError(
                "tree-sitter is required for TreeSitterExtractor. "
                "Install: pip install tree-sitter"
            )
        self._grammars = grammar_loader or GrammarLoader()
        self._queries: dict[str, dict[str, Any]] = {}  # lang → {kind → compiled query}

    def extract(self, file_path: str, content: str) -> list[Symbol]:
        ext = os.path.splitext(file_path)[1].lower()
        lang_name = self._EXT_TO_LANG.get(ext)
        if lang_name is None:
            return []

        language = self._grammars.load(lang_name)
        if language is None:
            return []  # caller should fall back to RegexExtractor

        import tree_sitter as ts
        parser = ts.Parser(language)
        tree = parser.parse(content.encode("utf-8"))

        symbols: list[Symbol] = []
        for kind, query in self._get_queries(lang_name, language).items():
            matches = query.matches(tree.root_node)
            for match in matches:
                captures = {n: nodes for n, nodes in match.captures.items()}
                name_node = captures.get("name", [None])[0]
                if name_node is None:
                    continue
                name = content[name_node.start_byte:name_node.end_byte]
                # Use the outermost captured node for signature
                main_node = captures.get(kind, [name_node])[0]
                sig = _reconstruct_signature(content.encode("utf-8"), main_node, set(captures))
                symbols.append(Symbol(
                    name=name,
                    kind=kind,
                    line=main_node.start_point[0] + 1,
                    signature=sig,
                ))

        return symbols

    def _get_queries(self, lang_name: str, language: Any) -> dict[str, Any]:
        """Build and cache compiled queries per language."""
        if lang_name not in self._queries:
            import tree_sitter as ts
            self._queries[lang_name] = {}
            for kind, query_str in _TS_QUERIES.get(lang_name, {}).items():
                self._queries[lang_name][kind] = language.query(query_str)
        return self._queries[lang_name]
```

### Fallback in `create_extractor()`

```python
def create_extractor() -> SymbolExtractor:
    try:
        return TreeSitterExtractor()
    except ImportError:
        logger.info("tree-sitter not available, using regex extractor")
        return RegexExtractor()


class CompositeExtractor:
    """Uses TreeSitterExtractor per-file, falls back to RegexExtractor per-file."""

    def __init__(self) -> None:
        self._ts: TreeSitterExtractor | None = None
        self._regex = RegexExtractor()

    def _ensure_ts(self) -> TreeSitterExtractor | None:
        if self._ts is None:
            try:
                self._ts = TreeSitterExtractor()
            except ImportError:
                pass
        return self._ts

    def extract(self, file_path: str, content: str) -> list[Symbol]:
        ext = os.path.splitext(file_path)[1].lower()
        ts_ext = self._ensure_ts()
        if ts_ext is not None:
            lang_map = TreeSitterExtractor._EXT_TO_LANG
            if ext in lang_map:
                symbols = ts_ext.extract(file_path, content)
                if symbols:
                    return symbols
                # TS returned empty (grammar not loaded) → fall through to regex
        return self._regex.extract(file_path, content)
```

### Acceptance Criteria

- Python: extracts functions (including decorators), async functions, classes, class methods, imports
- JavaScript: functions, async functions, classes, methods, imports
- TypeScript: functions, classes, methods, interfaces, type aliases
- Rust: functions (unsafe, generics), structs, enums, traits, impl blocks, use declarations
- Go: functions, methods with receivers, structs, interfaces, imports
- Java: classes, interfaces, methods (with visibility), constructors
- Signatures include decorators (Python), generics (Rust/TS), modifiers (Java)
- Files without a grammar fall through to `RegexExtractor`
- Returns empty list for unsupported file types

---

## Task 3.1.3: Persistent Symbol Index

**File to create:** `chef_human/agent/symbols/index.py`

An in-memory index of all symbols across the workspace, keyed by name and file path. Supports prefix search and incremental refresh.

### Data Structures

```python
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chef_human.agent.symbols.extractor import Symbol, SymbolExtractor
    from chef_human.agent.workspace import WorkspaceManager

logger = logging.getLogger(__name__)


@dataclass
class IndexEntry:
    symbol: Symbol
    file_path: str
    content_hash: str


class SymbolIndex:
    """In-memory index of all symbols in the workspace."""

    def __init__(
        self,
        workspace: WorkspaceManager,
        extractor: SymbolExtractor,
    ) -> None:
        self._workspace = workspace
        self._extractor = extractor
        self._entries: dict[str, list[IndexEntry]] = {}       # name → entries
        self._by_file: dict[Path, list[IndexEntry]] = {}      # file → entries
        self._content_hashes: dict[Path, str] = {}            # file → sha256
        self._initial_built: bool = False

    def build(self, files: list[Path] | None = None) -> int:
        """Build or rebuild the index for given or all workspace files.
        Returns total symbol count indexed.
        """
        if files is None:
            files = self._workspace.list_files(max_depth=10)[:500]

        count = 0
        for f in files:
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            content_hash = self._hash(content)
            symbols = self._extractor.extract(str(f), content)
            entry = IndexEntry(symbol=s, file_path=str(f), content_hash=content_hash)
            for s in symbols:
                self._entries.setdefault(s.name, []).append(entry)
            self._by_file[f] = [IndexEntry(symbol=s, file_path=str(f), content_hash=content_hash) for s in symbols]
            self._content_hashes[f] = content_hash
            count += len(symbols)

        self._initial_built = True
        logger.info("Indexed %d symbols from %d files", count, len(files))
        return count

    def refresh(self, files: list[Path] | None = None) -> int:
        """Incremental refresh: re-index files whose content changed."""
        if not self._initial_built:
            return self.build(files=files)

        if files is None:
            files = self._workspace.list_files(max_depth=10)[:500]

        count = 0
        for f in files:
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            new_hash = self._hash(content)
            if self._content_hashes.get(f) == new_hash:
                continue

            # File changed: remove old entries, re-index
            for entry in self._by_file.pop(f, []):
                name_list = self._entries.get(entry.symbol.name, [])
                self._entries[entry.symbol.name] = [e for e in name_list if e.file_path != str(f)]
                if not self._entries[entry.symbol.name]:
                    del self._entries[entry.symbol.name]

            symbols = self._extractor.extract(str(f), content)
            for s in symbols:
                entry = IndexEntry(symbol=s, file_path=str(f), content_hash=new_hash)
                self._entries.setdefault(s.name, []).append(entry)
            self._by_file[f] = [
                IndexEntry(symbol=s, file_path=str(f), content_hash=new_hash)
                for s in symbols
            ]
            self._content_hashes[f] = new_hash
            count += len(symbols)

        if count:
            logger.info("Refreshed %d changed symbols", count)
        return count

    def lookup(self, name: str, kind: str | None = None) -> list[IndexEntry]:
        """Look up a symbol by name, optionally filtered by kind."""
        entries = self._entries.get(name, [])
        if kind is not None:
            return [e for e in entries if e.symbol.kind == kind]
        return entries

    def lookup_by_file(self, path: Path) -> list[IndexEntry]:
        """Get all index entries in a file."""
        return self._by_file.get(self._workspace.resolve(path), [])

    def lookup_by_prefix(self, prefix: str, max_results: int = 10) -> list[IndexEntry]:
        """Prefix search across all symbol names."""
        results: list[IndexEntry] = []
        for name in sorted(self._entries):
            if name.startswith(prefix):
                for entry in self._entries[name]:
                    results.append(entry)
                    if len(results) >= max_results:
                        return results
        return results

    def search(self, query: str) -> list[IndexEntry]:
        """Search across symbol names and signatures (case-insensitive substring)."""
        query_lower = query.lower()
        results: list[IndexEntry] = []
        seen: set[tuple[str, str, int]] = set()
        for entries in self._entries.values():
            for entry in entries:
                key = (entry.symbol.name, entry.file_path, entry.symbol.line)
                if key in seen:
                    continue
                if query_lower in entry.symbol.name.lower() or query_lower in entry.symbol.signature.lower():
                    results.append(entry)
                    seen.add(key)
        return results[:50]

    def total_symbols(self) -> int:
        return sum(len(entries) for entries in self._entries.values())

    def total_files(self) -> int:
        return len(self._by_file)

    @staticmethod
    def _hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
```

### Performance

- **Initial build**: Sequential scan up to 500 files (configurable via `settings.max_index_files`)
- **Incremental refresh**: Only re-extracts files whose `sha256` content hash changed
- **Memory**: Each symbol stored once in `_entries` (by name) and once in `_by_file` (by path). A 10K-symbol codebase uses ~2–5 MB
- **Lookup**: O(1) by name (dict), O(n) prefix search over sorted keys

### Acceptance Criteria

- `build()` indexes all workspace files and returns symbol count
- `lookup("SymbolName")` returns all entries with that name (handles overloads in different files)
- `lookup("SymbolName", kind="class")` filters by kind
- `lookup_by_file(path)` returns all symbols in that file
- `lookup_by_prefix("Sym")` returns up to `max_results` matches
- `search("query")` matches both names and signatures (case-insensitive)
- `refresh()` re-indexes only changed files
- `refresh()` on unchanged files is a no-op
- Empty index returns empty results for all queries

---

## Task 3.1.4: File-Level Dependency Graph

**File to create:** `chef_human/agent/symbols/dependencies.py`

Extracts import/require/use/include statements from the symbol index and builds a directed graph of file-level dependencies.

### Design

```python
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chef_human.agent.symbols.index import SymbolIndex

logger = logging.getLogger(__name__)


class DependencyGraph:
    """Directed graph of file-level dependencies derived from import symbols."""

    def __init__(self, symbol_index: SymbolIndex) -> None:
        self._index = symbol_index
        self._graph: dict[Path, set[Path]] = {}
        self._reverse: dict[Path, set[Path]] = {}
        self._built: bool = False

    def build(self) -> None:
        """Build the dependency graph from import symbols in the index."""
        self._graph.clear()
        self._reverse.clear()

        import_kinds = {"import", "from_import", "use", "import_statement",
                        "import_declaration", "require"}

        for file_path, entries in self._index._by_file.items():
            for entry in entries:
                if entry.symbol.kind not in import_kinds:
                    continue
                deps = self._resolve_import(entry.symbol, file_path)
                for dep in deps:
                    self._graph.setdefault(file_path, set()).add(dep)
                    self._reverse.setdefault(dep, set()).add(file_path)

        self._built = True
        logger.info(
            "Built dependency graph: %d files, %d edges",
            len(self._graph),
            sum(len(deps) for deps in self._graph.values()),
        )

    def _resolve_import(self, symbol: Symbol, source_file: Path) -> list[Path]:
        """Resolve an import symbol to actual file paths in the workspace.

        Strategy:
        1. Parse the import path from the symbol signature
        2. Try direct match (e.g., 'os' → 'os.py')
        3. Try package match (e.g., 'os.path' → 'os/path.py')
        4. Try __init__.py match (e.g., 'package' → 'package/__init__.py')
        5. Mark as external if no workspace file matches
        """
        ...

    def dependencies(self, file_path: str | Path) -> list[Path]:
        """Files that this file directly imports."""
        resolved = self._resolve_path(file_path)
        return sorted(self._graph.get(resolved, set()))

    def dependents(self, file_path: str | Path) -> list[Path]:
        """Files that directly import this file."""
        resolved = self._resolve_path(file_path)
        return sorted(self._reverse.get(resolved, set()))

    def transitive_dependencies(self, file_path: str | Path, max_depth: int = 3) -> list[Path]:
        """All files transitively needed by file_path (imports-of-imports)."""
        resolved = self._resolve_path(file_path)
        visited: set[Path] = set()
        result: list[Path] = []
        queue = [(resolved, 0)]
        while queue:
            current, depth = queue.pop(0)
            if depth >= max_depth or current in visited:
                continue
            visited.add(current)
            for dep in self._graph.get(current, []):
                if dep != resolved:
                    result.append(dep)
                    queue.append((dep, depth + 1))
        return result

    def format_for_prompt(self, file_path: str | Path, direction: str = "out", max_depth: int = 2) -> str:
        """Format dependency info as text for the agent prompt."""
        resolved = self._resolve_path(file_path)
        lines: list[str] = []
        if direction in ("out", "both"):
            deps = self.dependencies(resolved)
            if deps:
                lines.append("Imports:")
                for d in deps:
                    lines.append(f"  {d.relative_to(self._index._workspace.root)}")
        if direction in ("in", "both"):
            deps = self.dependents(resolved)
            if deps:
                lines.append("Imported by:")
                for d in deps:
                    lines.append(f"  {d.relative_to(self._index._workspace.root)}")
        return "\n".join(lines)

    @staticmethod
    def _resolve_path(path: str | Path) -> Path:
        if isinstance(path, str):
            return Path(path)
        return path
```

### Import Resolution Strategy

The `_resolve_import()` method maps import paths to workspace files:

| Import statement | Resolved file(s) |
|-----------------|------------------|
| `import os` | `os.py` if exists, else `os/__init__.py`, else external |
| `from pathlib import Path` | `pathlib.py`, then `pathlib/Path.py` |
| `import os.path` | `os/path.py`, then `os/path/__init__.py` |
| `use crate::module::helper` | `src/module/helper.rs`, then `src/module.rs` |
| `import "github.com/pkg/errors"` | External (not in workspace) |
| `import static org.junit.Assert.*` | External |

External imports are noted but not resolved to paths.

### Acceptance Criteria

- `build()` constructs graph from import symbols in the index
- `dependencies("main.py")` returns list of files that `main.py` imports
- `dependents("utils.py")` returns list of files that import `utils.py`
- `transitive_dependencies("main.py", max_depth=2)` returns second-order imports
- External imports are excluded from resolution
- `format_for_prompt()` produces readable text for agent consumption
- Empty index produces empty graph

---

## Task 3.1.5: On-Demand Symbol Retrieval

**File to create:** `chef_human/agent/symbols/retriever.py`

Monitors the conversation for unknown symbol references and fetches their definitions into context.

### Design

```python
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chef_human.agent.file_context import FileContextManager
    from chef_human.agent.symbols.index import SymbolIndex

logger = logging.getLogger(__name__)

# Pattern to detect potential symbol references:
# Capitalized words, UPPER_CASE constants, dotted names like Module.Class
_SYMBOL_REF_PATTERN = re.compile(r"\b([A-Z][a-zA-Z0-9_]+(?:\.[A-Z][a-zA-Z0-9_]*)*)\b")


class SymbolRetriever:
    """Detects symbol references in agent conversation and retrieves definitions."""

    def __init__(
        self,
        index: SymbolIndex,
        file_context: FileContextManager,
    ) -> None:
        self._index = index
        self._file_context = file_context
        self._recently_fetched: set[str] = set()

    def detect_symbol_references(self, text: str) -> list[str]:
        """Find capitalized identifiers in text that exist in the symbol index."""
        candidates = set(_SYMBOL_REF_PATTERN.findall(text))
        # Filter out common words that are capitalized but not code symbols
        noise = {"The", "This", "That", "It", "I", "We", "You", "They",
                 "Here", "There", "Then", "Than", "Also", "But", "Not",
                 "Step", "Steps", "File", "Task", "Note", "Error", "Result"}
        candidates -= noise

        found: list[str] = []
        for name in candidates:
            if name not in self._recently_fetched:
                # Check index — split dotted names for lookup
                simple_name = name.split(".")[0]
                entries = self._index.lookup(simple_name)
                if entries:
                    found.append(name)
        return found

    def retrieve(self, symbol_name: str) -> str | None:
        """Fetch definition for a symbol and load its file into context.
        Returns formatted string or None if not found.
        """
        simple_name = symbol_name.split(".")[0]
        entries = self._index.lookup(simple_name)
        if not entries:
            return None

        # Prefer the first entry (arbitrary for overloaded names)
        entry = entries[0]
        file_path = entry.file_path

        # Load the file into FileContextManager for follow-up access
        self._file_context.get(file_path)

        # Format the definition
        lines = [
            f"**{entry.symbol.kind.title()}** `{entry.symbol.name}` — `{entry.file_path}:{entry.symbol.line}`",
            f"```",
            entry.symbol.signature,
            "```",
        ]
        self._recently_fetched.add(symbol_name)
        return "\n".join(lines)

    def reset_fetched(self) -> None:
        """Clear recently-fetched tracking (called on re-plan)."""
        self._recently_fetched.clear()

    @staticmethod
    def strip_noise(words: set[str]) -> set[str]:
        noise = {"The", "This", "That", "It", "I", "We", "You", "They",
                 "Here", "There", "Then", "Than", "Also", "But", "Not",
                 "Step", "Steps", "File", "Task", "Note", "Error", "Result",
                 "Let", "Yes", "No", "Ok", "Okay", "First", "Second", "Next",
                 "Look", "See", "Check", "Need", "Done", "Good"}
        return words - noise
```

### Detection Heuristic

The retriever uses a simple heuristic — no ML involved:

1. **Pattern**: Words matching `[A-Z][a-zA-Z0-9_]+` (CamelCase, PascalCase, UPPER_CASE)
2. **Filter**: Remove common English words, stopwords
3. **Verify**: Check if the word exists in the symbol index
4. **Deduplicate**: Skip symbols already fetched this session or turn
5. **Format**: Return the symbol's file path, line number, and signature

**Why not snake_case?** Lowercase function names like `parse_file` are too ambiguous — they'd match too many English words. The inverted index already exists, but we only trigger on CamelCase for now.

### Acceptance Criteria

- `detect_symbol_references()` finds capitalized identifiers that exist in the index
- `detect_symbol_references()` skips noise words (The, This, etc.)
- `detect_symbol_references()` skips already-fetched symbols
- `retrieve("SymbolName")` returns formatted definition string
- `retrieve("SymbolName")` loads the containing file into FileContextManager
- `retrieve("UnknownSymbol")` returns None
- `retrieve("Module.Class")` resolves dotted names to the base name
- `reset_fetched()` clears recently-fetched tracking

---

## Task 3.1.6: Integration into ContextAssembler & Agent Loop

**Files to modify:**
- `chef_human/agent/context.py` — add symbol context to assembly
- `chef_human/agent/__init__.py` — factory wiring
- `chef_human/agent/react_loop.py` — optional explicit `lookup_symbol` tool
- `chef_human/config.py` — add `max_index_files` setting

### ContextAssembler Changes

```python
class ContextAssembler:
    def __init__(
        self,
        conversation: ContextManager,
        workspace: WorkspaceManager,
        file_context: FileContextManager,
        repo_map: RepoMap,
        symbol_index: SymbolIndex | None = None,
        dep_graph: DependencyGraph | None = None,
        symbol_retriever: SymbolRetriever | None = None,
    ) -> None:
        self._conversation = conversation
        self._workspace = workspace
        self._file_context = file_context
        self._repo_map = repo_map
        self._symbol_index = symbol_index
        self._dep_graph = dep_graph
        self._symbol_retriever = symbol_retriever

    @property
    def symbol_index(self) -> SymbolIndex | None:
        return self._symbol_index

    def assemble(
        self,
        system_prompt: str,
        tool_definitions: str = "",
    ) -> list[Message]:
        # ... existing assembly (steps 1-4) ...

        # 5. Symbol definitions (from on-demand retrieval)
        if self._symbol_retriever and conversation_messages and remaining > 500:
            symbol_text = self._build_symbol_context(conversation_messages, remaining)
            if symbol_text:
                messages.append(
                    Message(role=Role.system, content=f"## Related Symbols\n\n{symbol_text}")
                )

        return messages

    def _build_symbol_context(
        self, conversation_messages: list[Message], budget: int
    ) -> str:
        """Scan recent non-system messages for symbol references."""
        recent = " ".join(
            m.content for m in conversation_messages[-4:]
            if m.role != Role.system
        )
        names = self._symbol_retriever.detect_symbol_references(recent)

        sections: list[str] = []
        for name in names[:5]:  # max 5 per turn
            defn = self._symbol_retriever.retrieve(name)
            if defn:
                tokens = self._conversation.tokenizer.count(defn)
                if tokens <= budget:
                    sections.append(defn)
                    budget -= tokens
        return "\n\n".join(sections)
```

### Factory Update

```python
# chef_human/agent/__init__.py

def create_context_assembler(
    workspace_root: str | None = None,
    index_on_init: bool = True,
) -> ContextAssembler:
    tokenizer = create_tokenizer(settings.ollama_model)
    root = workspace_root or settings.workspace or None
    workspace = WorkspaceManager(root=root)
    config = ContextConfig(
        max_tokens=settings.max_context_tokens,
        max_response_tokens=settings.max_response_tokens,
    )
    conversation = ContextManager(config=config, tokenizer=tokenizer)
    file_ctx = FileContextManager(workspace=workspace, tokenizer=tokenizer)
    repo_map = RepoMap(workspace=workspace, tokenizer=tokenizer)

    # Symbol index (optional, requires tree-sitter or falls back)
    from chef_human.agent.symbols.extractor import CompositeExtractor
    from chef_human.agent.symbols.index import SymbolIndex
    from chef_human.agent.symbols.dependencies import DependencyGraph
    from chef_human.agent.symbols.retriever import SymbolRetriever

    extractor = CompositeExtractor()
    symbol_index = SymbolIndex(workspace=workspace, extractor=extractor)
    dep_graph = DependencyGraph(symbol_index)
    symbol_retriever = SymbolRetriever(
        index=symbol_index,
        file_context=file_ctx,
    )

    if index_on_init:
        files = workspace.list_files(max_depth=10)[:getattr(settings, 'max_index_files', 500)]
        symbol_index.build(files=files)
        dep_graph.build()

    return ContextAssembler(
        conversation=conversation,
        workspace=workspace,
        file_context=file_ctx,
        repo_map=repo_map,
        symbol_index=symbol_index,
        dep_graph=dep_graph,
        symbol_retriever=symbol_retriever,
    )
```

### Config Update

```python
# chef_human/config.py — add to Settings/load_settings
max_index_files: int = 500  # limit to prevent slow startup on huge repos
```

### Option B: Explicit `lookup_symbol` Tool (deferred)

The plan originally considered an explicit tool for the agent to call. This is deferred to a follow-up — the automatic injection in `ContextAssembler` covers the most common case. An explicit tool would be useful when the model wants to explore a symbol it hasn't yet referenced, but that's an edge case.

### Acceptance Criteria

- `ContextAssembler` accepts optional `symbol_index`, `dep_graph`, `symbol_retriever`
- When all three are provided, `assemble()` injects symbol definitions for references in recent conversation
- Symbol definitions are truncated to fit token budget
- At most 5 symbol definitions are injected per turn
- `create_context_assembler()` creates and optionally pre-builds the index
- Index builds silently (no crash) even when tree-sitter is not installed
- All existing tests pass unchanged with the new optional parameters

---

## Task 3.1.7: Tests

**New test files:**

| Test file | ~Tests | What it covers |
|-----------|--------|----------------|
| `tests/test_symbols/test_grammars.py` | 10 | GrammarLoader: load, cache, missing grammar, unsupported language, supported_languages |
| `tests/test_symbols/test_extractor.py` | 50 | TreeSitterExtractor: all 6 languages, empty files, syntax errors, decorators, generics, signatures |
| `tests/test_symbols/test_index.py` | 25 | SymbolIndex: build, lookup, lookup_by_file, lookup_by_prefix, search, refresh, content hash |
| `tests/test_symbols/test_dependencies.py` | 15 | DependencyGraph: build, deps, reverse deps, transitive, formatting, external imports |
| `tests/test_symbols/test_retriever.py` | 12 | SymbolRetriever: detect refs, noise filtering, retrieve, dedup, reset |
| `tests/test_symbols/test_integration.py` | 8 | End-to-end: multi-language codebase, full index + dep graph + retrieval |

**Test data**: Create `tests/test_symbols/test_data/` with small synthetic files:
- `test_data/example.py` — functions, classes, methods, imports, decorators
- `test_data/example.js` — functions, classes, imports
- `test_data/example.rs` — fn, struct, enum, trait, use
- `test_data/example.go` — func, method, struct, interface, import
- `test_data/example.java` — class, method, interface

**Modified test files:**

| Test file | ~+Tests | What it covers |
|-----------|---------|----------------|
| `tests/test_context_assembly.py` | 5 | Symbol context injection, budget limiting, no index mode |
| `tests/test_repo_map.py` | 2 | `CompositeExtractor` fallback behavior |
| `tests/test_agent_integration.py` | 4 | Factory creates index, index pre-built, index not pre-built |

**Estimated total new tests**: ~130

---

## Dependencies Map

```
3.1.1 grammars.py ──────────────► tree-sitter (optional), tree-sitter-{lang} packages
3.1.2 extractor.py ─────────────► 3.1.1 grammars.py, tree_sitter (stdlib queries)
3.1.3 index.py ─────────────────► 3.1.2, 1.2.1 workspace.py, stdlib hashlib
3.1.4 dependencies.py ──────────► 3.1.3 index.py
3.1.5 retriever.py ─────────────► 3.1.3 index.py, 1.2.2 file_context.py
3.1.6 context.py ───────────────► 3.1.3–3.1.5, 1.2.4 ContextAssembler, config.py
3.1.7 tests ────────────────────► all of the above, test_data/
```

---

## Implementation Order

1. **3.1.1** Grammar loader — must exist before TreeSitterExtractor can work
2. **3.1.2** AST queries — replace the TreeSitterExtractor stub with real queries for all 6 languages
3. **3.1.3** Symbol index — build/refresh/lookup/search
4. **3.1.4** Dependency graph — extract imports, build graph
5. **3.1.5** Symbol retriever — detect references, retrieve definitions
6. **3.1.6** Integration — wire into ContextAssembler, factory, config
7. **3.1.7** Tests — all new test files + modifications to existing tests

---

## Design Decisions (Confirmed)

### 1. Tree-sitter missing: warn at startup
When `tree-sitter` is not installed, the factory prints a `logger.warning` suggesting `pip install tree-sitter`, then falls back to `RegexExtractor`. Implemented in `create_extractor()` and `chef_human/agent/__init__.py`.

### 2. Retrieval trigger: every turn
`SymbolRetriever.detect_symbol_references()` runs on every `ContextAssembler.assemble()` call, scanning the last 4 non-system messages. This catches symbol references early, before tool errors occur.

### 3. Dependency depth: direct only
`DependencyGraph.format_for_prompt()` only shows immediate imports. No transitive expansion. Keeps context usage predictable.

### 4. Index build: sync on startup
`create_context_assembler()` builds the index synchronously. The index is ready before the first LLM call. Startup latency is bounded by `settings.max_index_files` (default 500).

---

## Changes & Deviations Tracking

### 3.1.1 GrammarLoader & 3.1.2 TreeSitterExtractor (Actual Implementation)
| Deviation | Rationale |
|-----------|-----------|
| `GrammarLoader` uses `importlib.import_module` + `getattr` with `(module_name, func_name)` tuples | Grammar packages expose language via different function names (`language()` vs `language_typescript()`) |
| `GrammarLoader.load()` wraps PyCapsule in `tree_sitter.Language` | tree-sitter 0.26 grammar packages return `PyCapsule`, not `Language`; `Parser.__init__` requires `Language` |
| `_TS_QUERIES` uses tree-sitter 0.26 `Query` + `QueryCursor.matches()` API | tree-sitter 0.26 removed `Language.query()` — use `Query(lang, source)` instead |
| Signature reconstruction uses heuristic `_DEF_LINE_RE` regex to detect definition lines | Tree-sitter nodes span the entire definition including body; regex detects the declaration line boundary |
| Python `class_definition` captures include methods as separate function symbols | The `function` kind query for `function_definition` naturally captures methods; caller can filter by nesting level if needed |
| TypeScript uses `language_typescript()` for both `.ts` and `.tsx` | `language_tsx()` works identically for symbol queries; can add dedicated TSX grammar in the future |
| `TreeSitterExtractor.extract()` is stateful (caches compiled queries per language in `_compiled`) | Prevents re-compiling query strings on every `extract()` call for the same language |
| `CompositeExtractor` not implemented yet | TreeSitterExtractor already returns `[]` for unsupported extensions; `create_extractor()` returns TreeSitterExtractor directly; per-file fallback to RegexExtractor deferred to 3.1.6 if needed |

### 3.1.3 SymbolIndex (Persistent Symbol Index)
| Deviation | Rationale |
|-----------|-----------|
| `build()` replaces plan's buggy loop (entry built outside `for s in symbols`) | Plan code used `s` before it was defined; corrected to create `IndexEntry` inside the per-symbol loop |
| `build()` always clears all state before indexing (`.clear()` on all dicts) | Plan `build()` only cleared on first call; explicit clear ensures `rebuild()` doesn't accumulate stale entries even without explicit rebuild |
| `is_built` property added | Downstream code (ContextAssembler, retriever) needs to check if index is ready without calling `build()` again |
| Index uses `TreeSitterExtractor` directly (no `CompositeExtractor`) | Consistent with 3.1.2 — tree-sitter is installed; no regex fallback needed for indexed files | (Post-3.1)

### 3.1.4 DependencyGraph (File-Level Import Graph)
| Deviation | Rationale |
|-----------|-----------|
| Import extraction uses regex (`_IMPORT_PATTERNS`) instead of AST | AST queries for imports already exist in TreeSitterExtractor but return imported symbol names, not module paths; regex extracts the module/file path directly |
| Module-to-file resolution uses stem matching in ext_map, with relative and dotted-path fallbacks | Standard Python/JS module resolution is complex; this heuristic covers the common cases (same-directory, dotted paths like `sub.mod`) |
| `_resolve()` checks `self._index._by_file` for dotted-path resolution | Avoids re-reading files during build; uses the already-indexed file list |
| `transitive_dependencies()` does simple BFS with no cycle protection | Cycles are rare in real codebases; BFS at shallow depth (default 2) terminates naturally |

### 3.1.5 SymbolRetriever (On-Demand Symbol Retrieval)
| Deviation | Rationale |
|-----------|-----------|
| Noise filter (`_NOISE`) is more extensive than plan's original set | Additional common English words (What, How, Why, Please, Thanks, numbers) reduced false positives in practice |
| `retrieve()` uses `entry.symbol` fields directly instead of re-reading the file | The symbol signature is already captured during extraction — no need to re-parse the source |
| No `strip_noise()` static method (plan spec has it but it's unused) | The set difference is done inline in `detect_symbol_references()`; the static method would be dead code |

### 3.1.6 Integration into ContextAssembler
| Deviation | Rationale |
|-----------|-----------|
| `CompositeExtractor` created and used by factory instead of pure `TreeSitterExtractor` | Provides graceful fallback to regex for unindexed file types without crashing |
| `create_context_assembler()` always creates symbol components (never None) | Simpler to always wire them up; downstream checks for `is_built` to decide whether to use them |
| `ContextAssembler` uses `retriever` local var with `assert` for pyright | Pyright doesn't narrow through `if self._symbol_retriever` guard from `_build_symbol_context()` |

### 3.1.7 Tests
| Deviation | Rationale |
|-----------|-----------|
| `test_detects_multiple_symbols` only checks for `MyClass` (not `MY_CONSTANT`) | Variable assignments like `MY_CONSTANT = 42` are not captured by any extractor pattern — they're not function/class/import declarations |
| `test_create_extractor_returns_ts` → `test_create_extractor_returns_composite` | `create_extractor()` now returns `CompositeExtractor` (not bare `TreeSitterExtractor`) |
| 33 new tests across 3 files (deps: 13, retriever: 12, integration: 5), not the planned 35 | Slight reduction due to merged test cases; all acceptance criteria covered |

- **RAG for large codebases** (Phase 3.2) — chunk + embed + FAISS when index exceeds 500 files
- **Diff-aware editing** (Phase 3.3) — unified diffs instead of full-file rewrites
- **Index persistence to disk** — save `.chef-human/index.json` for sub-second startup
- **Watch mode** — auto-refresh index on file changes via `watchdog`
- **Symbol rank by usage** — prioritize frequently referenced symbols in retrieval
- **`lookup_symbol` tool** — explicit tool for agent to query symbol definitions on demand
