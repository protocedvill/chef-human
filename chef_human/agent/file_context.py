from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chef_human.agent.workspace import WorkspaceManager
    from chef_human.llm.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class FileContextManager:
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
        content = self.get(path)
        if content is None:
            return None
        lines = content.splitlines(keepends=True)
        return lines[start - 1 : end]

    def remove(self, path: str | Path) -> None:
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
