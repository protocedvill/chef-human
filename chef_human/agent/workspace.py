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
