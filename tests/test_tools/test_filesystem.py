from __future__ import annotations

from pathlib import Path

import pytest

from chef_human.agent.workspace import WorkspaceManager
from chef_human.tools.filesystem import (
    EditTool,
    GlobTool,
    GrepTool,
    LsTool,
    LsTreeTool,
    ReadTool,
    WriteTool,
)


def create_file(directory: Path, name: str, content: str = "") -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceManager:
    return WorkspaceManager(root=tmp_path)


@pytest.fixture
def read_tool(workspace: WorkspaceManager) -> ReadTool:
    return ReadTool(workspace)


@pytest.fixture
def write_tool(workspace: WorkspaceManager) -> WriteTool:
    return WriteTool(workspace)


@pytest.fixture
def edit_tool(workspace: WorkspaceManager) -> EditTool:
    return EditTool(workspace)


@pytest.fixture
def grep_tool(workspace: WorkspaceManager) -> GrepTool:
    return GrepTool(workspace)


@pytest.fixture
def glob_tool(workspace: WorkspaceManager) -> GlobTool:
    return GlobTool(workspace)


@pytest.fixture
def ls_tool(workspace: WorkspaceManager) -> LsTool:
    return LsTool(workspace)


@pytest.fixture
def ls_tree_tool(workspace: WorkspaceManager) -> LsTreeTool:
    return LsTreeTool(workspace)


# ---------------------------------------------------------------------------
# ReadTool
# ---------------------------------------------------------------------------

class TestReadTool:
    async def test_reads_file_content(self, read_tool, tmp_path):
        create_file(tmp_path, "hello.py", "print('hello')")
        result = await read_tool.run(path="hello.py")
        assert result.success
        assert "print('hello')" in result.output

    async def test_missing_file(self, read_tool):
        result = await read_tool.run(path="nonexistent.py")
        assert not result.success
        assert "not found" in result.error

    async def test_outside_workspace(self, read_tool):
        result = await read_tool.run(path="/etc/passwd")
        assert not result.success
        assert "Outside" in result.error

    async def test_offset(self, read_tool, tmp_path):
        create_file(tmp_path, "lines.txt", "a\nb\nc\nd\ne\n")
        result = await read_tool.run(path="lines.txt", offset=3)
        assert result.success
        assert result.output == "c\nd\ne\n"

    async def test_offset_and_limit(self, read_tool, tmp_path):
        create_file(tmp_path, "lines.txt", "a\nb\nc\nd\ne\n")
        result = await read_tool.run(path="lines.txt", offset=2, limit=2)
        assert result.success
        assert result.output == "b\nc\n"

    async def test_negative_offset_clamped(self, read_tool, tmp_path):
        create_file(tmp_path, "f.txt", "line1\nline2\n")
        result = await read_tool.run(path="f.txt", offset=-5)
        assert result.success
        assert "line1" in result.output


# ---------------------------------------------------------------------------
# WriteTool
# ---------------------------------------------------------------------------

class TestWriteTool:
    async def test_writes_file(self, write_tool, tmp_path):
        result = await write_tool.run(path="new.txt", content="hello world")
        assert result.success
        assert (tmp_path / "new.txt").read_text() == "hello world"

    async def test_creates_parent_directory(self, write_tool, tmp_path):
        result = await write_tool.run(path="sub/deep/file.txt", content="nested")
        assert result.success
        assert (tmp_path / "sub/deep/file.txt").read_text() == "nested"

    async def test_overwrites_existing(self, write_tool, tmp_path):
        create_file(tmp_path, "existing.txt", "old")
        result = await write_tool.run(path="existing.txt", content="new")
        assert result.success
        assert (tmp_path / "existing.txt").read_text() == "new"

    async def test_outside_workspace(self, write_tool):
        result = await write_tool.run(path="/tmp/outside.txt", content="bad")
        assert not result.success
        assert "Outside" in result.error

    async def test_reports_line_count(self, write_tool, tmp_path):
        result = await write_tool.run(path="multi.txt", content="a\nb\nc")
        assert "3 lines" in result.output

    async def test_diff_on_overwrite(self, write_tool, tmp_path):
        create_file(tmp_path, "existing.txt", "original content")
        result = await write_tool.run(path="existing.txt", content="new content")
        assert result.success
        assert "```diff" in result.output
        assert "-original content" in result.output
        assert "+new content" in result.output

    async def test_no_diff_on_new_file(self, write_tool, tmp_path):
        result = await write_tool.run(path="brand_new.txt", content="hello")
        assert result.success
        assert "```diff" not in result.output

    async def test_no_diff_on_identical_content(self, write_tool, tmp_path):
        create_file(tmp_path, "same.txt", "content")
        result = await write_tool.run(path="same.txt", content="content")
        assert result.success
        assert "```diff" not in result.output


# ---------------------------------------------------------------------------
# EditTool
# ---------------------------------------------------------------------------

class TestEditTool:
    async def test_single_replace(self, edit_tool, tmp_path):
        create_file(tmp_path, "f.txt", "hello world")
        result = await edit_tool.run(path="f.txt", old_string="world", new_string="there")
        assert result.success
        assert (tmp_path / "f.txt").read_text() == "hello there"

    async def test_replace_all(self, edit_tool, tmp_path):
        create_file(tmp_path, "f.txt", "a a a")
        result = await edit_tool.run(path="f.txt", old_string="a", new_string="b", replace_all=True)
        assert result.success
        assert (tmp_path / "f.txt").read_text() == "b b b"

    async def test_not_found(self, edit_tool, tmp_path):
        create_file(tmp_path, "f.txt", "hello")
        result = await edit_tool.run(path="f.txt", old_string="zzz", new_string="xxx")
        assert not result.success
        assert "not found" in result.error

    async def test_missing_file(self, edit_tool):
        result = await edit_tool.run(path="missing.txt", old_string="a", new_string="b")
        assert not result.success
        assert "not found" in result.error

    async def test_outside_workspace(self, edit_tool):
        result = await edit_tool.run(path="/etc/hosts", old_string="a", new_string="b")
        assert not result.success
        assert "Outside" in result.error

    async def test_reports_count(self, edit_tool, tmp_path):
        create_file(tmp_path, "f.txt", "x x x")
        result = await edit_tool.run(path="f.txt", old_string="x", new_string="y", replace_all=True)
        assert "3 occurrences" in result.output

    async def test_diff_in_output(self, edit_tool, tmp_path):
        create_file(tmp_path, "f.txt", "hello world")
        result = await edit_tool.run(path="f.txt", old_string="world", new_string="there")
        assert result.success
        assert "```diff" in result.output
        assert "-hello world" in result.output
        assert "+hello there" in result.output

    async def test_fuzzy_match_succeeds(self, edit_tool, tmp_path):
        create_file(tmp_path, "f.txt", "def foo():\n    return 42\n")
        result = await edit_tool.run(path="f.txt", old_string="def foo():\n   return 42", new_string="def foo():\n    return 99", fuzzy=True)
        assert result.success
        assert "fuzzy" in result.output
        assert "```diff" in result.output

    async def test_fuzzy_disabled_still_works_on_exact(self, edit_tool, tmp_path):
        create_file(tmp_path, "f.txt", "hello world")
        result = await edit_tool.run(path="f.txt", old_string="world", new_string="there", fuzzy=False)
        assert result.success

    async def test_fuzzy_disabled_fails_on_nonexistent(self, edit_tool, tmp_path):
        create_file(tmp_path, "f.txt", "abc")
        result = await edit_tool.run(path="f.txt", old_string="xyz", new_string="def", fuzzy=False)
        assert not result.success
        assert "not found" in result.error


# ---------------------------------------------------------------------------
# GrepTool
# ---------------------------------------------------------------------------

class TestGrepTool:
    async def test_finds_matches(self, grep_tool, tmp_path):
        create_file(tmp_path, "a.py", "def foo(): pass")
        create_file(tmp_path, "b.py", "def bar(): pass")
        result = await grep_tool.run(pattern="def ")
        assert result.success
        assert "foo" in result.output
        assert "bar" in result.output

    async def test_no_matches(self, grep_tool, tmp_path):
        create_file(tmp_path, "a.py", "xyz")
        result = await grep_tool.run(pattern="abc")
        assert result.success
        assert "No matches" in result.output

    async def test_include_filter(self, grep_tool, tmp_path):
        create_file(tmp_path, "code.py", "def f(): pass")
        create_file(tmp_path, "data.txt", "def f(): pass")
        result = await grep_tool.run(pattern="def", include="*.py")
        assert result.success
        assert "code.py" in result.output
        assert "data.txt" not in result.output

    async def test_invalid_regex(self, grep_tool):
        result = await grep_tool.run(pattern="[invalid")
        assert not result.success
        assert "Invalid regex" in result.error

    async def test_directory_not_found(self, grep_tool):
        result = await grep_tool.run(pattern="foo", path="/nonexistent")
        assert not result.success
        assert "not found" in result.error


# ---------------------------------------------------------------------------
# GlobTool
# ---------------------------------------------------------------------------

class TestGlobTool:
    async def test_matches_pattern(self, glob_tool, tmp_path):
        create_file(tmp_path, "a.py", "")
        create_file(tmp_path, "b.py", "")
        create_file(tmp_path, "data.txt", "")
        result = await glob_tool.run(pattern="**/*.py")
        assert result.success
        assert "a.py" in result.output
        assert "b.py" in result.output
        assert "data.txt" not in result.output

    async def test_no_matches(self, glob_tool, tmp_path):
        result = await glob_tool.run(pattern="**/*.xyz")
        assert result.success
        assert "No files" in result.output

    async def test_in_directory(self, glob_tool, tmp_path):
        create_file(tmp_path, "src/a.py", "")
        create_file(tmp_path, "tests/test_a.py", "")
        result = await glob_tool.run(pattern="**/*.py", path="src")
        assert result.success
        assert "a.py" in result.output
        assert "test_a.py" not in result.output

    async def test_directory_not_found(self, glob_tool):
        result = await glob_tool.run(pattern="*.py", path="/nonexistent")
        assert not result.success
        assert "not found" in result.error


# ---------------------------------------------------------------------------
# LsTool
# ---------------------------------------------------------------------------

class TestLsTool:
    async def test_lists_files(self, ls_tool, tmp_path):
        create_file(tmp_path, "a.py", "")
        create_file(tmp_path, "b.txt", "")
        (tmp_path / "sub").mkdir()
        result = await ls_tool.run()
        assert result.success
        assert "a.py" in result.output
        assert "b.txt" in result.output
        assert "sub/" in result.output

    async def test_empty_directory(self, ls_tool, tmp_path):
        result = await ls_tool.run(path=".")
        assert result.success
        assert "(empty directory)" in result.output

    async def test_ignores_hidden(self, ls_tool, tmp_path):
        create_file(tmp_path, ".git/config", "")
        create_file(tmp_path, "main.py", "")
        result = await ls_tool.run()
        assert ".git" not in result.output
        assert "main.py" in result.output

    async def test_outside_workspace(self, ls_tool):
        result = await ls_tool.run(path="/etc")
        assert not result.success
        assert "Outside" in result.error


# ---------------------------------------------------------------------------
# LsTreeTool
# ---------------------------------------------------------------------------

class TestLsTreeTool:
    async def test_shows_tree(self, ls_tree_tool, tmp_path):
        create_file(tmp_path, "main.py", "x = 1")
        create_file(tmp_path, "src/utils.py", "y = 2")
        result = await ls_tree_tool.run()
        assert result.success
        assert "main.py" in result.output
        assert "src/" in result.output

    async def test_empty_directory(self, ls_tree_tool, tmp_path):
        result = await ls_tree_tool.run()
        assert result.success
        assert "empty" in result.output

    async def test_subdirectory(self, ls_tree_tool, tmp_path):
        create_file(tmp_path, "src/utils.py", "y = 2")
        create_file(tmp_path, "tests/test_main.py", "z = 3")
        result = await ls_tree_tool.run(path="src")
        assert result.success
        assert "utils.py" in result.output
        assert "test_main.py" not in result.output
