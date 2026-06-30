# Phase 1.2: Context Manager — File Context, Repo Map & Context Assembly

**Goal**: Build a context assembly system that combines conversation history, on-demand file contents, and a structured project map — all within a token budget. This gives the agent awareness of the codebase it's working on.

**Prerequisites**: Phase 1.1 complete (LLM backends, tokenizer, basic sliding-window context manager).

---

## Task List

- [x] **1.2.1** Workspace manager (path validation, `.gitignore` awareness, root discovery)
- [x] **1.2.2** File context manager (LRU cache, on-demand loading, token-aware eviction)
- [x] **1.2.3** Repository map with symbol extraction (Tree-sitter or regex fallback)
- [x] **1.2.4** Context assembler (compose system prompt + conversation + file context + repo map under token budget)
- [x] **1.2.5** Integration tests & agent `__init__` factory

---

## Task 1.2.1: Workspace Manager

**File:** `chef_human/agent/workspace.py`

Validates file paths against the workspace root, respects `.gitignore`, and auto-discovers the project root.

```python
# chef_human/agent/workspace.py

from __future__ import annotations

import fnmatch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

IGNORE_PATTERNS: set[str] = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".eggs", "*.pyc", "*.pyo", ".DS_Store",
}


class WorkspaceManager:
    """Validates file paths within a workspace and filters ignored files."""

    def __init__(self, root: str | Path | None = None) -> None:
        self._root = Path(root or Path.cwd()).resolve()
        self._gitignore_patterns: list[str] = []
        self._load_gitignore()

    @property
    def root(self) -> Path:
        return self._root

    def _load_gitignore(self) -> None:
        gitignore_path = self._root / ".gitignore"
        if gitignore_path.exists():
            self._gitignore_patterns = [
                line.strip()
                for line in gitignore_path.read_text().splitlines()
                if line.strip() and not line.startswith("#")
            ]

    def resolve(self, path: str | Path) -> Path:
        """Resolve a potentially relative path to an absolute workspace path."""
        p = Path(path)
        if not p.is_absolute():
            p = (self._root / p).resolve()
        else:
            p = p.resolve()
        return p

    def is_within_workspace(self, path: str | Path) -> bool:
        """Check if a resolved path is inside the workspace root."""
        try:
            self.resolve(path).relative_to(self._root)
            return True
        except ValueError:
            return False

    def is_ignored(self, path: str | Path) -> bool:
        """Check if a path matches an ignore pattern (glob or gitignore)."""
        try:
            p = self.resolve(path)
            rel = p.relative_to(self._root)
        except ValueError:
            return False
        parts = rel.parts
        for part in parts:
            part_str = str(part)
            if any(fnmatch.fnmatch(part_str, pat) for pat in IGNORE_PATTERNS):
                return True
            for pattern in self._gitignore_patterns:
                if self._match_gitignore(part_str, pattern):
                    return True
        return False

    @staticmethod
    def _match_gitignore(name: str, pattern: str) -> bool:
        if pattern.startswith("/"):
            pattern = pattern[1:]
        if pattern.endswith("/"):
            return name == pattern.rstrip("/")
        if "*" in pattern:
            return fnmatch.fnmatch(name, pattern)
        return name == pattern

    def list_files(
        self, directory: str | Path | None = None, max_depth: int = 5
    ) -> list[Path]:
        """Recursively list non-ignored files up to max_depth."""
        base = self.resolve(directory) if directory else self._root
        if not base.exists():
            return []
        files: list[Path] = []
        try:
            for entry in base.rglob("*"):
                if entry.is_file():
                    try:
                        rel = entry.relative_to(self._root)
                        if len(rel.parts) > max_depth:
                            continue
                        if not self.is_ignored(entry):
                            files.append(entry)
                    except ValueError:
                        continue
        except PermissionError:
            pass
        return sorted(files)

    @staticmethod
    def discover_root(start: str | Path | None = None) -> Path:
        """Walk up from start to find a project root marker (.git, pyproject.toml)."""
        current = Path(start or Path.cwd()).resolve()
        markers = {".git", "pyproject.toml", "setup.py", "setup.cfg", "Cargo.toml", "package.json"}
        for parent in [current] + list(current.parents):
            for marker in markers:
                if (parent / marker).exists():
                    return parent
        return current
```

**Acceptance criteria:**
- `resolve()` converts relative paths to absolute under workspace root
- `is_within_workspace()` rejects paths outside the root boundary
- `is_ignored()` checks common dirs (`.git`, `__pycache__`) and `.gitignore` patterns
- `discover_root()` walks up to find `.git` or `pyproject.toml`
- `list_files()` returns non-ignored files with depth limit

---

## Task 1.2.2: File Context Manager (LRU Cache)

**File:** `chef_human/agent/file_context.py`

On-demand file reading with LRU eviction, token-aware budget enforcement.

```python
# chef_human/agent/file_context.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chef_human.agent.workspace import WorkspaceManager
    from chef_human.llm.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class FileContextManager:
    """LRU cache of file contents for agent context assembly."""

    def __init__(
        self,
        workspace: WorkspaceManager,
        tokenizer: Tokenizer,
        max_files: int = 50,
        max_tokens: int = 10_000,
    ) -> None:
        self._workspace = workspace
        self._tokenizer = tokenizer
        self._max_files = max_files
        self._max_tokens = max_tokens
        self._files: dict[Path, str] = {}
        self._access_order: list[Path] = []

    def get(self, path: str | Path) -> str | None:
        """Get file content, loading it if not cached."""
        resolved = self._workspace.resolve(path)
        if not resolved.exists() or not resolved.is_file():
            return None
        if not self._workspace.is_within_workspace(resolved):
            logger.warning("File outside workspace: %s", resolved)
            return None

        if resolved in self._files:
            self._touch(resolved)
            return self._files[resolved]

        content = resolved.read_text(encoding="utf-8", errors="replace")
        self._add(resolved, content)
        self._evict_if_needed(resolved)
        return content

    def get_lines(
        self, path: str | Path, start: int = 1, end: int | None = None
    ) -> list[str] | None:
        """Get specific line range from a file."""
        content = self.get(path)
        if content is None:
            return None
        lines = content.splitlines(keepends=True)
        return lines[start - 1 : end]

    def remove(self, path: str | Path) -> None:
        """Remove a file from cache."""
        resolved = self._workspace.resolve(path)
        self._files.pop(resolved, None)
        self._access_order = [p for p in self._access_order if p != resolved]

    def clear(self) -> None:
        self._files.clear()
        self._access_order.clear()

    def contains(self, path: str | Path) -> bool:
        resolved = self._workspace.resolve(path)
        return resolved in self._files

    def total_tokens(self) -> int:
        return sum(self._tokenizer.count(c) for c in self._files.values())

    def cached_files(self) -> list[Path]:
        return list(self._access_order)

    def _touch(self, path: Path) -> None:
        self._access_order.remove(path)
        self._access_order.append(path)

    def _add(self, path: Path, content: str) -> None:
        self._files[path] = content
        self._access_order.append(path)

    def _evict_if_needed(self, new_path: Path) -> None:
        while len(self._files) > self._max_files:
            self._evict_one()
        while self.total_tokens() > self._max_tokens:
            self._evict_one()

    def _evict_one(self) -> None:
        if not self._access_order:
            return
        oldest = self._access_order.pop(0)
        self._files.pop(oldest, None)
        logger.debug("Evicted from file cache: %s", oldest)
```

**Acceptance criteria:**
- Files loaded on first `get()` and cached until evicted
- LRU eviction when `max_files` or `max_tokens` exceeded
- `get_lines()` returns line range (1-indexed, inclusive)
- Files outside workspace are rejected
- `total_tokens()` accurately reflects cached content

---

## Task 1.2.3: Repository Map with Symbol Extraction

**Files:**
- `chef_human/agent/repo_map.py` — RepoMap class, symbol protocols, tree formatting
- `chef_human/agent/symbols/__init__.py` — symbol extraction package
- `chef_human/agent/symbols/extractor.py` — extractor protocol + Tree-sitter / regex implementations
- `chef_human/agent/symbols/languages.py` — extension-to-language mapping, supported languages

Generates a structured overview of the project with both directory tree and key symbols (function signatures, class definitions, imports).

### Architecture

```
RepoMap
├── Directory tree (always available, no deps)
└── Symbol extraction
    ├── TreeSitterExtractor (full AST, needs tree-sitter + language grammars)
    └── RegexExtractor     (fallback, pure Python)
```

### Symbol Protocol & Implementations

```python
# chef_human/agent/symbols/extractor.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str          # "function", "class", "import", "method"
    line: int
    signature: str     # Full signature line (e.g., "def foo(x: int) -> str:")


class SymbolExtractor(Protocol):
    def extract(self, file_path: str, content: str) -> list[Symbol]: ...


class RegexExtractor:
    """Pure-Python regex-based symbol extraction for common languages."""

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
            ("function", r"^\s*(?:pub\s+)?(?:unsafe\s+)?fn\s+(\w+)\s*\([^)]*\)"),
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
            ("method", r"^\s*(?:public\s+|private\s+|protected\s+)?(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*(?:throws\s+\S+)?\s*\{?"),
        ],
    }

    _DEFAULT_PATTERNS: list[tuple[str, str]] = [
        ("function", r"^\s*(?:async\s+)?(?:def|function|fn|func)\s+(\w+)\s*\([^)]*\)"),
        ("class", r"^\s*(?:class|struct|trait|interface)\s+(\w+)"),
    ]

    def extract(self, file_path: str, content: str) -> list[Symbol]:
        import os
        ext = os.path.splitext(file_path)[1].lower()
        patterns = self._LANG_PATTERNS.get(ext, self._DEFAULT_PATTERNS)
        symbols: list[Symbol] = []
        for line_num, line in enumerate(content.splitlines(), start=1):
            for kind, pattern in patterns:
                import re
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
    """AST-based symbol extraction using Tree-sitter."""

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
        import os
        ext = os.path.splitext(file_path)[1].lower()
        lang_name = self._LANG_MAP.get(ext)
        if lang_name is None:
            return []
        language = self._get_language(lang_name)
        if language is None:
            return []
        # Tree-sitter parsing and query logic here
        # (depends on tree-sitter version 0.23+ API)
        ...

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
    """Create best available symbol extractor."""
    try:
        return TreeSitterExtractor()
    except ImportError:
        import logging
        logging.getLogger(__name__).info(
            "tree-sitter not available, using regex extractor"
        )
        return RegexExtractor()
```

### RepoMap

```python
# chef_human/agent/repo_map.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from chef_human.agent.symbols.extractor import (
    Symbol,
    SymbolExtractor,
    create_extractor,
)

if TYPE_CHECKING:
    from chef_human.agent.workspace import WorkspaceManager
    from chef_human.llm.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

MAX_FILES_IN_TREE = 100


class RepoMap:
    """Generates a structured project map with directory tree and symbols."""

    def __init__(
        self,
        workspace: WorkspaceManager,
        tokenizer: Tokenizer,
        extractor: SymbolExtractor | None = None,
    ) -> None:
        self._workspace = workspace
        self._tokenizer = tokenizer
        self._extractor = extractor or create_extractor()
        self._symbol_cache: dict[Path, list[Symbol]] = {}

    def generate_tree(self, directory: str | Path | None = None) -> str:
        """Generate an ASCII directory tree of the workspace."""
        base = self._workspace.resolve(directory) if directory else self._workspace.root
        lines: list[str] = []
        prefix = ""

        files = self._workspace.list_files(directory, max_depth=3)
        files = files[:MAX_FILES_IN_TREE]

        if files:
            lines.append(f"Project tree ({len(files)} files shown):")
            lines.append("")
            current_dir = base
            tree_lines = self._build_tree_lines(base, files)
            lines.extend(tree_lines)

        return "\n".join(lines)

    def _build_tree_lines(self, base: Path, files: list[Path]) -> list[str]:
        """Build ASCII tree lines from a sorted list of files."""
        lines: list[str] = []
        tree: dict[Path, list[Path]] = {}
        for f in files:
            parent = f.parent
            if parent not in tree:
                tree[parent] = []
            tree[parent].append(f)

        sorted_dirs = sorted(tree.keys(), key=lambda p: p.relative_to(base))

        for i, directory in enumerate(sorted_dirs):
            rel = directory.relative_to(base)
            if rel == Path("."):
                continue
            indent = "  " * (len(rel.parts) - 1)
            branch = "└── " if i == len(sorted_dirs) - 1 else "├── "
            lines.append(f"{indent}{branch}{rel.name}/")

            dir_files = sorted(tree[directory], key=lambda p: p.name)
            for j, f in enumerate(dir_files):
                file_indent = "  " * len(rel.parts)
                file_branch = "└── " if j == len(dir_files) - 1 else "├── "
                lines.append(f"{file_indent}{file_branch}{f.name}")

        return lines

    def generate_symbol_map(self, files: list[Path] | None = None) -> str:
        """Generate a symbol map for key files in the workspace."""
        if files is None:
            files = self._workspace.list_files(max_depth=4)[:30]

        sections: list[str] = []
        for f in files:
            content = self._safe_read(f)
            if content is None:
                continue
            symbols = self._extract_symbols(f, content)
            if not symbols:
                continue

            rel_path = f.relative_to(self._workspace.root)
            sections.append(f"### {rel_path}")
            for sym in symbols[:10]:  # max 10 symbols per file
                sections.append(f"  {sym.line}: {sym.signature}")
            sections.append("")

        return "\n".join(sections)

    def generate(self, max_tokens: int = 2000) -> str:
        """Generate a combined repo map (tree + symbols) within token budget."""
        tree = self.generate_tree()
        tree_tokens = self._tokenizer.count(tree)
        remaining = max_tokens - tree_tokens

        map_parts = [tree]

        if remaining > 200:
            symbol_map = self.generate_symbol_map()
            symbol_tokens = self._tokenizer.count(symbol_map)
            if symbol_tokens <= remaining:
                map_parts.append("\n## Symbols\n")
                map_parts.append(symbol_map)
            else:
                truncated = self._truncate_to_tokens(symbol_map, remaining)
                map_parts.append("\n## Symbols\n")
                map_parts.append(truncated)

        return "\n".join(map_parts)

    def _extract_symbols(self, path: Path, content: str) -> list[Symbol]:
        if path in self._symbol_cache:
            return self._symbol_cache[path]
        symbols = self._extractor.extract(str(path), content)
        self._symbol_cache[path] = symbols
        return symbols

    @staticmethod
    def _safe_read(path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

    @staticmethod
    def _truncate_to_tokens(text: str, max_tokens: int) -> str:
        """Crudely truncate text to approximately fit token budget."""
        chars_per_token = 4
        max_chars = max_tokens * chars_per_token
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n... (truncated)"
```

**Acceptance criteria:**
- `generate_tree()` produces an ASCII directory structure
- `generate_symbol_map()` extracts and displays function/class signatures
- `generate()` combines tree + symbols within token budget
- Falls back to `RegexExtractor` when tree-sitter not available
- Tree-sitter path raises `ImportError` gracefully when not installed

---

## Task 1.2.4: Context Assembler

**File:** `chef_human/agent/context.py` (enhanced)

The existing `ContextManager` is extended with file context and repo map integration. A new `ContextAssembler` class composes the final prompt.

```python
# chef_human/agent/context.py (enhanced)

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from chef_human.llm.tokenizer import Tokenizer, create_tokenizer

if TYPE_CHECKING:
    from chef_human.agent.file_context import FileContextManager
    from chef_human.agent.repo_map import RepoMap
    from chef_human.agent.workspace import WorkspaceManager
    from chef_human.llm.backend import Message


@dataclass
class ContextConfig:
    max_tokens: int = 32768
    max_response_tokens: int = 4096
    summary_tokens: int = 512
    # Components that share the remaining token budget
    repo_map_tokens: int = 2000
    file_context_tokens: int = 10000


class ContextManager:
    def __init__(
        self,
        config: ContextConfig | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.config = config or ContextConfig()
        self.tokenizer = tokenizer or create_tokenizer()
        self.messages: list[Message] = []
        self._summary: str = ""

    def add_message(self, msg: Message) -> None:
        self.messages.append(msg)
        self._trim_if_needed()

    def get_messages(self) -> list[Message]:
        return self.messages

    def token_count(self) -> int:
        return sum(self.tokenizer.count(m.content) for m in self.messages)

    def _trim_if_needed(self) -> None:
        budget = self.config.max_tokens - self.config.max_response_tokens - self.config.summary_tokens
        while self.token_count() > budget and len(self.messages) > 1:
            if not self._summary and len(self.messages) > 3:
                old = self.messages[:2]
                self._summary = f"[Previous conversation: {len(old)} messages trimmed]"
                self.messages = self.messages[2:]
            elif len(self.messages) > 2:
                self.messages.pop(0)
            else:
                break


class ContextAssembler:
    """Composes the final context from all sources under a token budget."""

    def __init__(
        self,
        conversation: ContextManager,
        workspace: WorkspaceManager,
        file_context: FileContextManager,
        repo_map: RepoMap,
    ) -> None:
        self._conversation = conversation
        self._workspace = workspace
        self._file_context = file_context
        self._repo_map = repo_map

    def assemble(
        self,
        system_prompt: str,
        tool_definitions: str = "",
    ) -> list[Message]:
        """Assemble the full context as a list of messages.

        Priority order (highest to lowest):
        1. System prompt + tool definitions (always included)
        2. Conversation history (sliding window)
        3. Repository map (truncated to budget)
        4. File context (truncated to budget)
        """
        # 1. System prompt
        system_content = system_prompt
        if tool_definitions:
            system_content += "\n\n" + tool_definitions

        system_tokens = self._conversation.tokenizer.count(system_content)
        remaining = (
            self._conversation.config.max_tokens
            - self._conversation.config.max_response_tokens
            - system_tokens
        )

        # 2. Conversation history
        conversation_messages = self._conversation.get_messages()

        # 3. Repository map
        repo_map_text = ""
        repo_budget = min(
            self._conversation.config.repo_map_tokens,
            int(remaining * 0.15),  # 15% of remaining budget
        )
        if repo_budget > 100:
            repo_map_text = self._repo_map.generate(max_tokens=repo_budget)
            remaining -= self._conversation.tokenizer.count(repo_map_text)

        # 4. File context
        file_text = self._build_file_context()
        file_tokens = self._conversation.tokenizer.count(file_text)
        file_budget = min(
            self._conversation.config.file_context_tokens,
            remaining,
        )
        if file_tokens > file_budget:
            file_text = self._truncate_file_context(file_text, file_budget)

        # Build final message list
        messages: list[Message] = []
        messages.append(Message(role=Role.system, content=system_content))

        if repo_map_text:
            messages.append(
                Message(role=Role.system, content=f"## Repository Structure\n\n{repo_map_text}")
            )

        if file_text:
            messages.append(
                Message(role=Role.system, content=f"## File Context\n\n{file_text}")
            )

        messages.extend(conversation_messages)
        return messages

    def _build_file_context(self) -> str:
        """Build a formatted string of all cached file contents."""
        sections: list[str] = []
        for path in self._file_context.cached_files():
            content = self._file_context.get(path)
            if content is not None:
                rel = self._workspace.resolve(path).relative_to(self._workspace.root)
                lines = content.splitlines()
                sections.append(f"File: {rel} ({len(lines)} lines)")
                sections.append("```")
                sections.append(content)
                sections.append("```")
                sections.append("")
        return "\n".join(sections)

    def _truncate_file_context(self, text: str, max_tokens: int) -> str:
        """Remove file entries from file context until within budget."""
        sections = text.split("\nFile: ")
        kept: list[str] = []
        for sec in sections:
            if not sec.strip():
                continue
            entry = sec if kept else sec
            entry_tokens = self._conversation.tokenizer.count(entry)
            remaining -= entry_tokens  # type: ignore[used-before-def]
            if remaining >= 0:
                kept.append(entry)
            else:
                break
        return "\nFile: ".join(kept)
```

**Note:** The `Role` import needs to be added. The `remaining` variable in `_truncate_file_context` has a bug in the above — it references a closure variable. The actual implementation will pass `max_tokens` as a proper parameter and compute remaining inside the function.

**Acceptance criteria:**
- `assemble()` produces a valid list of `Message` objects
- System prompt always included, never trimmed
- Repo map and file context properly truncated to fit budget
- Conversation history is the final messages (after any sliding window trim)
- Token budget correctly distributed across components

---

## Task 1.2.5: Integration Tests & Agent `__init__` Factory

**Files updated:**
- `chef_human/agent/__init__.py` — factory functions
- `tests/test_workspace.py` — workspace manager tests
- `tests/test_file_context.py` — file context manager tests
- `tests/test_repo_map.py` — repo map & symbol extractor tests
- `tests/test_context_assembly.py` — context assembler tests

**Factory:**

```python
# chef_human/agent/__init__.py

from chef_human.agent.context import ContextAssembler, ContextConfig, ContextManager
from chef_human.agent.file_context import FileContextManager
from chef_human.agent.repo_map import RepoMap
from chef_human.agent.symbols.extractor import RegexExtractor, create_extractor
from chef_human.agent.workspace import WorkspaceManager
from chef_human.config import settings
from chef_human.llm.tokenizer import create_tokenizer


def create_context_assembler() -> ContextAssembler:
    """Create a fully-wired context assembler from config defaults."""
    tokenizer = create_tokenizer(settings.ollama_model)
    workspace = WorkspaceManager(root=settings.workspace or None)
    config = ContextConfig(
        max_tokens=settings.max_context_tokens,
        max_response_tokens=settings.max_response_tokens,
    )
    conversation = ContextManager(config=config, tokenizer=tokenizer)
    file_ctx = FileContextManager(
        workspace=workspace,
        tokenizer=tokenizer,
    )
    repo_map = RepoMap(workspace=workspace, tokenizer=tokenizer)
    return ContextAssembler(
        conversation=conversation,
        workspace=workspace,
        file_context=file_ctx,
        repo_map=repo_map,
    )


__all__ = [
    "ContextAssembler",
    "ContextConfig",
    "ContextManager",
    "FileContextManager",
    "RepoMap",
    "WorkspaceManager",
    "create_context_assembler",
]
```

**Planned tests:**

| Test file | Test count | What it covers |
|-----------|-----------|----------------|
| `test_workspace.py` | ~15 | Path resolution, boundary checks, `.gitignore`, root discovery, file listing |
| `test_file_context.py` | ~31 | CRUD operations, LRU eviction, line ranges, token budget eviction |
| `test_repo_map.py` | ~65 | Tree generation, symbol extraction (regex), combined map, truncation |
| `test_context_assembly.py` | ~22 | Full assembly, budget distribution, system prompt preservation, file context truncation |
| `test_agent_integration.py` | ~9 | Factory `create_context_assembler()`, end-to-end with files, symbol extraction fallback |

**Acceptance criteria:**
- [x] All 248 M2 tests pass (no external deps beyond what Phase 1.1 required)
- [x] `create_context_assembler()` wires all components without error
- [x] Symbol extraction works for Python, JavaScript, TypeScript, Rust, Go, Java
- [x] Falls back gracefully when tree-sitter not available

---

## Dependencies Map

```
1.2.1 workspace.py ───────────► stdlib (pathlib)
1.2.2 file_context.py ────────► 1.2.1 workspace.py, 1.1.6 tokenizer.py
1.2.3 repo_map.py ────────────► 1.2.1 workspace.py, 1.1.6 tokenizer.py
                                optional: tree-sitter, tree-sitter-{lang}
1.2.4 context.py (enhanced) ──► 1.2.1, 1.2.2, 1.2.3, 1.1.6 tokenizer.py
1.2.5 agent/__init__.py ──────► 1.2.1–1.2.4, 1.1.8 config.py
```

---

## Changes & Deviations Tracking

This section will be updated during implementation. Key areas to watch:

1. **Tree-sitter availability**: Python 3.15 beta may lack `tree-sitter` wheels (C extension). `RegexExtractor` serves as fallback. Track whether `tree-sitter` and `tree-sitter-{lang}` packages install successfully.

2. **`pathspec` library**: For full `.gitignore` support, `pathspec` is the standard library. If unavailable, the basic pattern matching in `WorkspaceManager._match_gitignore()` is a simplified alternative.

3. **Settings integration**: `WorkspaceManager` uses `settings.workspace` for auto-root detection. If `settings.workspace` is empty string (current default), it falls back to `Path.cwd()`.

4. **Token budget negotiation**: The `ContextAssembler` uses hardcoded ratios (15% for repo map). May need tuning based on real usage.

### 1.2.1 Implementation Notes

**Bug fix in `is_ignored`**: The plan's code used `part_str in IGNORE_PATTERNS` (literal string comparison), but `IGNORE_PATTERNS` contains glob patterns like `"*.pyc"`. Changed to `any(fnmatch.fnmatch(part_str, pat) for pat in IGNORE_PATTERNS)` so glob patterns match correctly.

**`max_depth` semantics**: The plan's `list_files` checks `len(rel.parts) > max_depth`. Since `rel.parts` includes the filename as the last component, `max_depth` effectively limits total path components (including filename). E.g., `max_depth=5` allows files where the relative path has at most 5 components (e.g., `src/module.py` has 2 components). Files deeper than `max_depth` components are excluded.

**`.gitignore` file listed**: The workspace manager does not treat `.gitignore` itself as ignored. It appears in `list_files` results alongside regular source files. This is intentional — the agent benefits from knowing which files are explicitly ignored.

### 1.2.2 Implementation Notes

No deviations from the plan code. 31 tests pass covering:
- Init, get, get_lines (with line ranges, newline handling, out-of-bounds)
- Cache semantics (caching on first get, returning cached on second get)
- LRU access order tracking and touch behavior
- Remove, clear, contains
- Eviction by max_files and max_tokens (oldest-first, touch preserves)
- total_tokens accuracy

**Test fixes**: Two tests in the plan needed corrected expectations:
- `test_touching_preserves_file`: added 10 extra files (exceeded `max_files=10`), fixed to 9
- `test_eviction_removes_oldest_first`: content was too small to trigger token budget eviction (50 tokens, budget was 100), increased from 200 to 500 chars

### 1.2.3 Implementation Notes

**Tree generation fix — root-level files omitted**: The plan's `_build_tree_lines` skipped files where `rel == Path(".")`, causing root-level files to never appear in the tree. Rewritten to group files by parent directory, register all ancestor directories, and output root files separately before subdirectory branches.

**Rust generics not handled in regex pattern**: The plan's Rust function pattern `r"fn\s+(\w+)\s*\([^)]*\)"` failed on `pub unsafe fn transmute<T>(x: T) -> U` because `<T>` appears between the function name and `(`. Fixed to `fn\s+(\w+)\s*(?:<[^>]*>)?\s*\([^)]*\)`.

**Default patterns require parens**: The `_DEFAULT_PATTERNS` used for unknown file extensions require parentheses `(...)` in the function signature. Languages like Ruby where `def hello` is valid without parens won't match. Noted as a known limitation; could be relaxed with a more permissive optional-paren pattern if needed.

**65 tests pass covering**:
- Symbol dataclass (fields, frozen, hashable)
- RegexExtractor for Python (functions, async, return types, classes, inheritance, imports, comments, multiple symbols)
- RegexExtractor for Rust (fn, pub fn, struct, enum, trait, unsafe fn with generics)
- RegexExtractor for Go (func, method receiver, struct type, interface)
- RegexExtractor for JS/TS (function, async function, class, interface)
- RegexExtractor for Java (class, interface, method, private method)
- RegexExtractor default patterns (unknown extension fallback)
- TreeSitterExtractor (ImportError when not installed, create_extractor fallback)
- RepoMap tree generation (empty, root files, subdirectory, nested dirs, root+subdir mixed, ignored files)
- RepoMap symbol map (empty, no symbols, single/multiple symbols, 10-per-file limit)
- RepoMap combined generate (empty, includes tree, includes symbols, truncation, omission when budget too small)
- Helpers (safe_read, truncate_to_tokens)

### 1.2.4 Implementation Notes

**Bug fix in plan's `_truncate_file_context`**: The plan's code had an undefined `remaining` variable (referenced as a closure variable that didn't exist). Fixed by initializing `remaining = max_tokens` locally within the method, and correctly decrementing it as file entries are kept.

**`Message`/`Role` import**: The plan noted the `Role` import was missing. Changed `context.py` to import `Message` and `Role` directly from `chef_human.llm.backend` (moved out of `TYPE_CHECKING` guard) so tests can construct `Message` objects with `Role.system`.

**`_build_file_context` path handling**: The plan's code was ambiguous about absolute vs. relative paths from `cached_files()`. `FileContextManager` stores resolved (absolute) paths in `_access_order`, so `_build_file_context` can use `path.relative_to(self._workspace.root)` directly.

**ContextConfig extended**: Added `repo_map_tokens: int = 2000` and `file_context_tokens: int = 10000` to `ContextConfig`. Existing tests for `ContextConfig` only check the original three fields and continue to pass.

**22 tests pass covering**:
- ContextConfig extended fields (defaults, custom values)
- ContextAssembler construction (accepts deps, requires all args)
- assemble output (returns Message list, first is system, includes conversation, tool defs appended)
- Includes repo map and file context when sources are populated
- Empty sources produce no repo map / file context messages
- File context truncation via `_truncate_file_context` (under budget, over budget, empty input)
- Build file context formatting (empty cache, formatted files)
- Integration smoke test (no exceptions, all messages have role + content)

### 1.2.5 Implementation Notes

**Plan `__all__` had duplicate `ContextAssembler`**: The plan's `__all__` list included `ContextAssembler` twice. Fixed in implementation.

**9 integration tests pass covering**:
- `create_context_assembler()` factory (creates without error, assemble works, accepts tool_defs)
- End-to-end with real files (file context + repo map + conversation, empty workspace)
- Factory edge cases (extractor fallback to RegexExtractor, tokenizer fallback to ApproxTokenizer)
- Symbol extractor integration (Python function+class, Rust function+struct)

---

## Future Improvements (Post-1.2)

- **Full Tree-sitter grammar support**: Once `tree-sitter` wheels are available, implement full AST queries for all languages.
- **Persistent symbol cache**: Cache extracted symbols to disk between sessions.
- **Incremental repo map updates**: Only re-scan changed files.
- **Semantic file ranking**: Prioritize recently modified or task-relevant files in the repo map.
- **Workspace watch mode**: Use `watchdog` to auto-invalidate file cache on changes.
