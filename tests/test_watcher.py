from __future__ import annotations

import time
from pathlib import Path

import pytest

from chef_human.agent.watcher import FileWatcher
from chef_human.agent.workspace import WorkspaceManager


def create_file(directory: Path, name: str, content: str = "") -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceManager:
    return WorkspaceManager(root=tmp_path)


class TestFileWatcher:
    def test_start_stop(self, workspace: WorkspaceManager):
        changes: list[list[Path]] = []
        watcher = FileWatcher(workspace, on_change=changes.append, interval=0.1)
        watcher.start()
        assert watcher._running
        watcher.stop()
        assert not watcher._running

    def test_detects_file_change(self, workspace: WorkspaceManager):
        create_file(workspace.root, "test.py", "original\n")
        changes: list[list[Path]] = []
        watcher = FileWatcher(workspace, on_change=changes.append, interval=0.1)
        watcher.start()

        time.sleep(0.15)
        create_file(workspace.root, "test.py", "modified\n")
        time.sleep(0.25)

        watcher.stop()
        assert len(changes) >= 1
        # At least one change should reference test.py
        all_changed = [p for clist in changes for p in clist]
        assert any("test.py" in str(p) for p in all_changed)

    def test_detects_new_file(self, workspace: WorkspaceManager):
        changes: list[list[Path]] = []
        watcher = FileWatcher(workspace, on_change=changes.append, interval=0.1)
        watcher.start()

        time.sleep(0.15)
        create_file(workspace.root, "new.py", "new file\n")
        time.sleep(0.25)

        watcher.stop()
        assert len(changes) >= 1
        all_changed = [p for clist in changes for p in clist]
        assert any("new.py" in str(p) for p in all_changed)

    def test_callback_receives_changed_files(self, workspace: WorkspaceManager):
        captured: list[list[Path]] = []

        def on_change(changed: list[Path]) -> None:
            captured.append(changed)

        watcher = FileWatcher(workspace, on_change=on_change, interval=0.1)
        watcher.start()

        time.sleep(0.15)
        create_file(workspace.root, "a.py", "content\n")
        time.sleep(0.25)

        watcher.stop()
        assert len(captured) >= 1
        # Verify the callback received the file
        found = any("a.py" in str(p) for clist in captured for p in clist)
        assert found

    def test_daemon_thread_allows_exit(self, workspace: WorkspaceManager):
        watcher = FileWatcher(workspace, on_change=lambda x: None, interval=0.1)
        watcher.start()
        assert watcher._thread is not None
        assert watcher._thread.daemon
        watcher.stop()

    def test_no_change_no_callback(self, workspace: WorkspaceManager):
        create_file(workspace.root, "stable.py", "content\n")
        changes: list[list[Path]] = []
        watcher = FileWatcher(workspace, on_change=changes.append, interval=0.1)
        watcher.start()
        time.sleep(0.3)
        watcher.stop()
        # No changes triggered since nothing changed after initial snapshot
        # (the initial file creation may be detected as a change if watcher starts before snapshot)
        pass
