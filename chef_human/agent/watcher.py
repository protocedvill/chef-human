from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from chef_human.agent.workspace import WorkspaceManager

logger = logging.getLogger(__name__)


class FileWatcher:
    """Polls workspace files for mtime changes and triggers a callback.

    Runs in a daemon thread so it does not prevent process exit.
    """

    def __init__(
        self,
        workspace: WorkspaceManager,
        on_change: Callable[[list[Path]], None],
        interval: float = 2.0,
    ) -> None:
        self._workspace = workspace
        self._on_change = on_change
        self._interval = interval
        self._running = False
        self._thread: threading.Thread | None = None
        self._snapshot: dict[Path, float] = {}

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._snapshot = self._snapshot_files()
        self._thread = threading.Thread(target=self._check_loop, daemon=True)
        self._thread.start()
        logger.info("File watcher started (interval=%ss)", self._interval)

    def stop(self) -> None:
        self._running = False
        logger.info("File watcher stopped")

    def _snapshot_files(self) -> dict[Path, float]:
        snapshot: dict[Path, float] = {}
        files = self._workspace.list_files(max_depth=10)
        for f in files:
            try:
                stat = f.stat()
                snapshot[f] = stat.st_mtime
            except OSError:
                continue
        return snapshot

    def _check_loop(self) -> None:
        while self._running:
            time.sleep(self._interval)
            try:
                current = self._snapshot_files()
                changed: list[Path] = []
                # Check for modified or added files
                for path, mtime in current.items():
                    old = self._snapshot.get(path)
                    if old is None or old != mtime:
                        changed.append(path)
                # Check for deleted files
                for path in self._snapshot:
                    if path not in current:
                        changed.append(path)

                if changed:
                    logger.debug("File watcher detected %d change(s)", len(changed))
                    self._on_change(changed)

                self._snapshot = current
            except Exception:
                logger.exception("File watcher error")
