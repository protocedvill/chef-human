from __future__ import annotations

import fnmatch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

IGNORE_PATTERNS: set[str] = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".tox",
    ".eggs",
    "*.pyc",
    "*.pyo",
    ".DS_Store",
    # chef-human's own persisted state (symbol index, RAG store, saved
    # sessions -- see DEFAULT_SAVE_DIR in agent/persistence.py and
    # Settings.rag_index_dir) lives under this directory by default. Every
    # exploration/indexing path (ls, ls_tree, grep, glob, repo map,
    # SymbolIndex, RAG file discovery) routes through is_ignored()/
    # list_files() below, so excluding it here keeps the agent from
    # treating its own session cache and index as part of the codebase
    # it's exploring.
    ".chef-human",
}


class WorkspaceManager:
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
        p = Path(path)
        if not p.is_absolute():
            p = (self._root / p).resolve()
        else:
            p = p.resolve()
        return p

    def is_within_workspace(self, path: str | Path) -> bool:
        try:
            self.resolve(path).relative_to(self._root)
            return True
        except ValueError:
            return False

    def relative_display(self, path: str | Path) -> str:
        """Best-effort workspace-relative path for display in tool output,
        falling back to the input as given on any resolution error (e.g. a
        symbol index entry pointing outside the workspace). Several tools
        (reference_finder, lookup_symbol) reimplemented this same
        resolve+relative_to+except pattern independently; centralizing it
        here means a change to how display paths are computed only needs
        to happen once."""
        try:
            return str(self.resolve(path).relative_to(self._root))
        except Exception:
            return str(path)

    def is_ignored(self, path: str | Path) -> bool:
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
        current = Path(start or Path.cwd()).resolve()
        markers = {
            ".git",
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "Cargo.toml",
            "package.json",
        }
        for parent in [current] + list(current.parents):
            for marker in markers:
                if (parent / marker).exists():
                    return parent
        return current
