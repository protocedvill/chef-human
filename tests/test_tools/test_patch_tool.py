from __future__ import annotations

from pathlib import Path

import pytest

from chef_human.agent.workspace import WorkspaceManager
from chef_human.tools.diff import DiffStore
from chef_human.tools.patch_tool import PatchTool, _apply_patch


def create_file(directory: Path, name: str, content: str = "") -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


class TestApplyPatch:
    def test_simple_hunk(self):
        old = "line1\nline2\nline3\n"
        patch = "@@ -1,3 +1,3 @@\n line1\n-line2\n+line2_modified\n line3\n"
        result = _apply_patch(old, patch)
        assert result == "line1\nline2_modified\nline3\n"

    def test_identical_no_hunks(self):
        result = _apply_patch("hello\nworld\n", "")
        assert result is None

    def test_context_mismatch(self):
        old = "aaa\nbbb\nccc\n"
        patch = "@@ -1,3 +1,3 @@\n aaa\n-bbb\n+xxx\n ddd\n"  # ddd doesn't match
        result = _apply_patch(old, patch)
        assert result is None

    def test_reverse_patch(self):
        old = "line1\nline2_modified\nline3\n"
        patch = "@@ -1,3 +1,3 @@\n line1\n-line2\n+line2_modified\n line3\n"
        result = _apply_patch(old, patch, reverse=True)
        assert result == "line1\nline2\nline3\n"

    def test_multiple_hunks(self):
        old = "a\nb\nc\nd\ne\nf\ng\n"
        patch = (
            "@@ -1,4 +1,4 @@\n"
            " a\n"
            "-b\n"
            "+B\n"
            " c\n"
            " d\n"
            "@@ -5,3 +5,3 @@\n"
            " e\n"
            "-f\n"
            "+F\n"
            " g\n"
        )
        result = _apply_patch(old, patch)
        assert result == "a\nB\nc\nd\ne\nF\ng\n"

    def test_insertion_hunk(self):
        old = "a\nb\n"
        patch_lines = ["@@ -2,0 +2,1 @@", "+inserted\n"]
        result = _apply_patch(old, "\n".join(patch_lines))
        assert result is not None

    def test_patch_no_changes(self):
        old = "a\nb\nc\n"
        patch = "@@ -1,3 +1,3 @@\n a\n b\n c\n"
        result = _apply_patch(old, patch)
        assert result == old


class TestPatchTool:
    @pytest.fixture
    def workspace(self, tmp_path: Path) -> WorkspaceManager:
        return WorkspaceManager(root=str(tmp_path))

    @pytest.fixture
    def diff_store(self) -> DiffStore:
        return DiffStore()

    @pytest.fixture
    def patch_tool(self, workspace: WorkspaceManager, diff_store: DiffStore) -> PatchTool:
        return PatchTool(workspace=workspace, diff_store=diff_store)

    async def test_apply_simple_patch(self, workspace: WorkspaceManager, patch_tool: PatchTool):
        path = "f.py"
        create_file(workspace.root, path, "a\nb\nc\n")
        patch = "@@ -1,3 +1,3 @@\n a\n-b\n+BB\n c\n"
        result = await patch_tool.run(path=path, patch=patch)
        assert result.success
        assert "Applied patch" in result.output
        assert workspace.root.joinpath(path).read_text() == "a\nBB\nc\n"

    async def test_reverse_patch(self, workspace: WorkspaceManager, patch_tool: PatchTool):
        path = "f.py"
        create_file(workspace.root, path, "a\nBB\nc\n")
        patch = "@@ -1,3 +1,3 @@\n a\n-b\n+BB\n c\n"
        result = await patch_tool.run(path=path, patch=patch, reverse=True)
        assert result.success
        assert workspace.root.joinpath(path).read_text() == "a\nb\nc\n"

    async def test_file_not_found(self, workspace: WorkspaceManager, patch_tool: PatchTool):
        result = await patch_tool.run(path="missing.py", patch="@@ -1,1 +1,1 @@\n")
        assert not result.success
        assert "not found" in (result.error or "")

    async def test_outside_workspace(self, workspace: WorkspaceManager, patch_tool: PatchTool):
        result = await patch_tool.run(path="/etc/passwd", patch="@@ -1,1 +1,1 @@\n")
        assert not result.success
        assert "Outside workspace" in (result.error or "")

    async def test_empty_patch(self, workspace: WorkspaceManager, patch_tool: PatchTool):
        path = "f.py"
        create_file(workspace.root, path, "content\n")
        result = await patch_tool.run(path=path, patch="")
        assert not result.success
        assert "empty" in (result.error or "")

    async def test_bad_hunk_context(self, workspace: WorkspaceManager, patch_tool: PatchTool):
        path = "f.py"
        create_file(workspace.root, path, "aaa\nbbb\nccc\n")
        patch = "@@ -1,3 +1,3 @@\n aaa\n-bbb\n+xxx\n ddd\n"  # ddd doesn't match
        result = await patch_tool.run(path=path, patch=patch)
        assert not result.success
        assert "failed" in (result.error or "").lower()

    async def test_patch_no_changes(self, workspace: WorkspaceManager, patch_tool: PatchTool):
        path = "f.py"
        create_file(workspace.root, path, "a\nb\nc\n")
        patch = "@@ -1,3 +1,3 @@\n a\n b\n c\n"
        result = await patch_tool.run(path=path, patch=patch)
        assert result.success
        assert "no changes" in result.output

    async def test_diff_store_records(self, workspace: WorkspaceManager, diff_store: DiffStore, patch_tool: PatchTool):
        path = "f.py"
        create_file(workspace.root, path, "old line\n")
        patch = "@@ -1,1 +1,1 @@\n-old line\n+new line\n"
        result = await patch_tool.run(path=path, patch=patch)
        assert result.success
        entries = diff_store.get_all(path)
        assert len(entries) == 1
        assert entries[0].tool_name == "patch"
        assert entries[0].old_content == "old line\n"
        assert entries[0].new_content == "new line\n"

    async def test_patch_with_diff_fences(self, workspace: WorkspaceManager, patch_tool: PatchTool):
        path = "f.py"
        create_file(workspace.root, path, "aaa\nbbb\nccc\n")
        patch = "```diff\n@@ -1,3 +1,3 @@\n aaa\n-bbb\n+BBB\n ccc\n```\n"
        result = await patch_tool.run(path=path, patch=patch)
        assert result.success
        assert workspace.root.joinpath(path).read_text() == "aaa\nBBB\nccc\n"
