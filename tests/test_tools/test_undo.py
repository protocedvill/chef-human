from __future__ import annotations

from pathlib import Path

import pytest

from chef_human.agent.workspace import WorkspaceManager
from chef_human.tools.diff import DiffStore, FileChange
from chef_human.tools.redo import RedoTool
from chef_human.tools.registry import ToolResult
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
def diff_store() -> DiffStore:
    return DiffStore()


@pytest.fixture
def undo_tool(workspace: WorkspaceManager, diff_store: DiffStore) -> UndoTool:
    return UndoTool(workspace=workspace, diff_store=diff_store)


class TestUndoTool:
    async def test_nothing_to_undo(self, undo_tool: UndoTool):
        result = await undo_tool.run()
        assert not result.success
        assert "Nothing to undo" in (result.error or "")

    async def test_undo_edit(self, workspace: WorkspaceManager, diff_store: DiffStore, undo_tool: UndoTool):
        path = "f.py"
        full = workspace.root / path
        create_file(workspace.root, path, "old content\n")
        diff_store.record(
            path=path,
            diff="```diff\n-old content\n+new content\n```\n",
            tool_name="edit",
            old_content="old content\n",
            new_content="new content\n",
        )
        # Apply the edit so the file has the new content
        full.write_text("new content\n")

        result = await undo_tool.run()
        assert result.success
        assert "Undid edit" in result.output
        assert full.read_text() == "old content\n"

    async def test_undo_write(self, workspace: WorkspaceManager, diff_store: DiffStore, undo_tool: UndoTool):
        path = "f.py"
        full = workspace.root / path
        create_file(workspace.root, path, "original\n")
        diff_store.record(
            path=path,
            diff="```diff\n-original\n+overwritten\n```\n",
            tool_name="write",
            old_content="original\n",
            new_content="overwritten\n",
        )
        full.write_text("overwritten\n")

        result = await undo_tool.run()
        assert result.success
        assert "Undid write" in result.output
        assert full.read_text() == "original\n"

    async def test_undo_new_file_write(self, workspace: WorkspaceManager, diff_store: DiffStore, undo_tool: UndoTool):
        path = "new.txt"
        full = workspace.root / path
        diff_store.record(
            path=path,
            diff="```diff\n+new file\n```\n",
            tool_name="write",
            old_content=None,
            new_content="new file\n",
        )
        # Create the file (as write would)
        create_file(workspace.root, path, "new file\n")

        result = await undo_tool.run()
        assert result.success
        assert "deleted" in result.output
        assert not full.exists()

    async def test_undo_specific_path(self, workspace: WorkspaceManager, diff_store: DiffStore, undo_tool: UndoTool):
        create_file(workspace.root, "a.py", "a content\n")
        create_file(workspace.root, "b.py", "b content\n")
        diff_store.record("a.py", "diff_a", "edit", old_content="a content\n", new_content="a v2\n")
        diff_store.record("b.py", "diff_b", "edit", old_content="b content\n", new_content="b v2\n")
        workspace.root.joinpath("a.py").write_text("a v2\n")
        workspace.root.joinpath("b.py").write_text("b v2\n")

        # Undo only b.py
        result = await undo_tool.run(path="b.py")
        assert result.success
        assert workspace.root.joinpath("b.py").read_text() == "b content\n"
        # a.py should still be "a v2"
        assert workspace.root.joinpath("a.py").read_text() == "a v2\n"

    async def test_undo_with_reverse_diff(self, workspace: WorkspaceManager, diff_store: DiffStore, undo_tool: UndoTool):
        path = "f.py"
        create_file(workspace.root, path, "line1\nline2\nline3\n")
        diff_store.record(
            path=path,
            diff="```diff\n-line2\n+line2_modified\n```\n",
            tool_name="edit",
            old_content="line1\nline2\nline3\n",
            new_content="line1\nline2_modified\nline3\n",
        )
        workspace.root.joinpath(path).write_text("line1\nline2_modified\nline3\n")

        result = await undo_tool.run()
        assert result.success
        assert "```diff" in result.output

    async def test_multi_file_transaction_undoes_and_redoes_as_one_unit(
        self,
        workspace: WorkspaceManager,
        diff_store: DiffStore,
        undo_tool: UndoTool,
    ):
        a = create_file(workspace.root, "a.py", "new a\n")
        b = create_file(workspace.root, "b.py", "new b\n")
        diff_store.record_transaction(
            [
                FileChange("a.py", "old a\n", "new a\n"),
                FileChange("b.py", "old b\n", "new b\n"),
            ],
            "refactor_symbol",
        )

        undone = await undo_tool.run()
        assert undone.success
        assert a.read_text() == "old a\n"
        assert b.read_text() == "old b\n"
        assert await undo_tool.run() == ToolResult(
            success=False, error="Nothing to undo."
        )

        redone = await RedoTool(workspace, diff_store).run()
        assert redone.success
        assert a.read_text() == "new a\n"
        assert b.read_text() == "new b\n"
