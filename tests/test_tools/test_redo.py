from __future__ import annotations

from pathlib import Path

import pytest

from chef_human.agent.workspace import WorkspaceManager
from chef_human.tools.diff import DiffStore
from chef_human.tools.redo import RedoTool
from chef_human.tools.undo import UndoTool


def create_file(directory: Path, name: str, content: str = "") -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceManager:
    return WorkspaceManager(root=str(tmp_path))


@pytest.fixture
def store() -> DiffStore:
    return DiffStore()


@pytest.fixture
def undo_tool(workspace: WorkspaceManager, store: DiffStore) -> UndoTool:
    return UndoTool(workspace=workspace, diff_store=store)


@pytest.fixture
def redo_tool(workspace: WorkspaceManager, store: DiffStore) -> RedoTool:
    return RedoTool(workspace=workspace, diff_store=store)


class TestRedoTool:
    async def test_redo_after_undo_restores_content(
        self, workspace: WorkspaceManager, undo_tool: UndoTool, redo_tool: RedoTool, tmp_path: Path
    ):
        f = create_file(tmp_path, "test.py", "original")
        store = undo_tool._store
        store.record("test.py", "diff", "write", old_content="original", new_content="modified")
        f.write_text("modified")

        await undo_tool.run()
        assert f.read_text() == "original"

        result = await redo_tool.run()
        assert result.success
        assert f.read_text() == "modified"

    async def test_nothing_to_redo(self, redo_tool: RedoTool):
        result = await redo_tool.run()
        assert not result.success
        assert "Nothing to redo" in (result.error or "")

    async def test_new_write_clears_redo_stack(
        self, workspace: WorkspaceManager, undo_tool: UndoTool, redo_tool: RedoTool, tmp_path: Path
    ):
        f = create_file(tmp_path, "test.py", "v1")
        store = undo_tool._store
        store.record("test.py", "diff", "write", old_content="v1", new_content="v2")
        f.write_text("v2")

        await undo_tool.run()
        assert f.read_text() == "v1"

        # New modification clears redo stack
        store.record("test.py", "diff2", "write", old_content="v1", new_content="v3")
        f.write_text("v3")

        result = await redo_tool.run()
        assert not result.success
        assert "Nothing to redo" in (result.error or "")

    async def test_multiple_undo_redo_cycles(
        self, workspace: WorkspaceManager, undo_tool: UndoTool, redo_tool: RedoTool, tmp_path: Path
    ):
        f = create_file(tmp_path, "test.py", "start")
        store = undo_tool._store
        store.record("test.py", "d1", "write", old_content="start", new_content="mid")
        f.write_text("mid")
        store.record("test.py", "d2", "write", old_content="mid", new_content="end")
        f.write_text("end")

        await undo_tool.run()
        assert f.read_text() == "mid"
        await undo_tool.run()
        assert f.read_text() == "start"

        result = await redo_tool.run()
        assert result.success
        assert f.read_text() == "mid"

        result = await redo_tool.run()
        assert result.success
        assert f.read_text() == "end"

    async def test_redo_output_contains_file_path(
        self, workspace: WorkspaceManager, undo_tool: UndoTool, redo_tool: RedoTool, tmp_path: Path
    ):
        f = create_file(tmp_path, "test.py", "original")
        store = undo_tool._store
        store.record("test.py", "diff", "edit", old_content="original", new_content="modified")
        f.write_text("modified")
        await undo_tool.run()

        result = await redo_tool.run()
        assert result.success
        assert "test.py" in result.output

    async def test_redo_different_files(
        self, workspace: WorkspaceManager, undo_tool: UndoTool, redo_tool: RedoTool, tmp_path: Path
    ):
        f1 = create_file(tmp_path, "a.py", "a1")
        f2 = create_file(tmp_path, "b.py", "b1")
        store = undo_tool._store
        store.record("a.py", "da", "write", old_content="a1", new_content="a2")
        f1.write_text("a2")
        store.record("b.py", "db", "write", old_content="b1", new_content="b2")
        f2.write_text("b2")

        await undo_tool.run()
        assert f2.read_text() == "b1"
        await undo_tool.run()
        assert f1.read_text() == "a1"

        result = await redo_tool.run()
        assert result.success
        assert f1.read_text() == "a2"

        result = await redo_tool.run()
        assert result.success
        assert f2.read_text() == "b2"

    async def test_redo_creates_parent_dirs(
        self, workspace: WorkspaceManager, undo_tool: UndoTool, redo_tool: RedoTool, tmp_path: Path
    ):
        f = tmp_path / "sub" / "test.py"
        f.parent.mkdir(parents=True)
        f.write_text("original")
        store = undo_tool._store
        store.record(str(f), "diff", "write", old_content="original", new_content="modified")
        f.write_text("modified")
        await undo_tool.run()
        f.unlink()

        result = await redo_tool.run()
        assert result.success
        assert f.read_text() == "modified"
