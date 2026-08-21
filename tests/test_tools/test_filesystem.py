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

    async def test_large_read_is_capped_to_10kb(self, read_tool, tmp_path):
        content = ("0123456789abcdef\n" * 900)
        create_file(tmp_path, "big.txt", content)

        result = await read_tool.run(path="big.txt")

        assert result.success
        assert len(result.output.encode("utf-8")) <= ReadTool.MAX_OUTPUT_BYTES
        assert "[read output truncated to 10240 bytes]" in result.output
        assert "Original selected size:" in result.output
        assert "Excerpt:" in result.output

    async def test_large_python_read_includes_symbol_map(self, read_tool, tmp_path):
        prefix = '"""module doc"""\n\n'
        functions = []
        for i in range(80):
            functions.append(
                f"def func_{i}():\n"
                f'    """Function {i} doc."""\n'
                f'    return "{("x" * 120)}"\n\n'
            )
        create_file(tmp_path, "big_module.py", prefix + "".join(functions))

        result = await read_tool.run(path="big_module.py")

        assert result.success
        assert len(result.output.encode("utf-8")) <= ReadTool.MAX_OUTPUT_BYTES
        assert "Top-level symbol map:" in result.output
        assert "- def func_0 @ lines" in result.output
        assert "Function 0 doc." in result.output

    async def test_large_python_read_limits_symbol_map_budget(self, read_tool, tmp_path):
        functions = []
        for i in range(200):
            functions.append(
                f"class PublicClass{i}:\n"
                f'    """Public class {i} summary that is intentionally verbose."""\n'
                "    pass\n\n"
            )
        create_file(tmp_path, "huge_symbols.py", "".join(functions))

        result = await read_tool.run(path="huge_symbols.py")

        assert result.success
        symbol_start = result.output.index("Top-level symbol map:\n")
        excerpt_start = result.output.index("\nExcerpt:\n")
        symbol_map = result.output[symbol_start:excerpt_start]
        excerpt = result.output[excerpt_start + len("\nExcerpt:\n") :]
        assert len(symbol_map.encode("utf-8")) <= ReadTool.MAX_SYMBOL_MAP_BYTES
        assert len(excerpt.encode("utf-8")) > len(symbol_map.encode("utf-8"))


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

    async def test_mkdir_failure_returns_error_not_exception(self, write_tool, tmp_path):
        # A file where a parent directory needs to be creates makes mkdir()
        # raise FileExistsError/NotADirectoryError -- must come back as a
        # ToolResult, not propagate out of run().
        create_file(tmp_path, "not_a_dir", "im a file")
        result = await write_tool.run(path="not_a_dir/child.txt", content="x")
        assert not result.success
        assert "Cannot write" in result.error


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

    async def test_missing_file_empty_old_string_creates_it(self, edit_tool, tmp_path):
        result = await edit_tool.run(path="new.txt", old_string="", new_string="hello")
        assert result.success
        assert (tmp_path / "new.txt").read_text() == "hello"

    async def test_missing_file_create_mkdir_failure_returns_error(self, edit_tool, tmp_path):
        create_file(tmp_path, "not_a_dir", "im a file")
        result = await edit_tool.run(
            path="not_a_dir/child.txt", old_string="", new_string="x"
        )
        assert not result.success
        assert "Cannot write" in result.error

    async def test_existing_file_empty_old_string_replaces_whole_content(self, edit_tool, tmp_path):
        # str.replace("", x) inserts x between every character instead of
        # setting the whole content -- old_string="" must not fall into
        # that footgun for a file that already exists.
        f = tmp_path / "existing.txt"
        f.write_text("original content")
        result = await edit_tool.run(
            path="existing.txt", old_string="", new_string="brand new content", replace_all=True
        )
        assert result.success
        assert f.read_text() == "brand new content"

    async def test_existing_file_empty_old_string_message_matches_write_wording(
        self, edit_tool, tmp_path
    ):
        # Regression test: this branch's message used to say "Replaced
        # entire contents of {path}", which a step-verifier LLM read as
        # evidence the file wasn't newly *created* (only that an existing
        # file's contents were replaced) -- causing perpetual
        # partial/not_complete oscillation for a step like "create a new
        # file named X" depending on whether the model happened to call
        # `write` or `edit` that turn. The message must read the same way
        # WriteTool's does, since the two are functionally equivalent here.
        (tmp_path / "existing.txt").write_text("original content")
        result = await edit_tool.run(
            path="existing.txt", old_string="", new_string="line one\nline two"
        )
        assert result.success
        assert "replaced" not in result.output.lower()
        assert result.output.startswith("Wrote 2 lines to existing.txt")

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

    async def test_noop_edit_says_no_changes_made(self, edit_tool, tmp_path):
        create_file(tmp_path, "f.txt", "hello world")
        result = await edit_tool.run(path="f.txt", old_string="world", new_string="world")
        assert result.success
        assert (tmp_path / "f.txt").read_text() == "hello world"
        assert "No changes made" in result.output
        assert "Applied edit" not in result.output
        assert "```diff" not in result.output

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
# FileContextManager integration -- read/write/edit should keep the file
# prominently visible in the "## File Context" prompt section (not just
# buried in conversation history), instead of relying on the model to
# recall it and avoiding redundant re-reads.
# ---------------------------------------------------------------------------

class TestFileContextIntegration:
    async def test_read_populates_file_context(self, workspace, tmp_path):
        from chef_human.agent.file_context import FileContextManager
        from chef_human.llm.tokenizer import ApproxTokenizer

        fcm = FileContextManager(workspace=workspace, tokenizer=ApproxTokenizer())
        tool = ReadTool(workspace, file_context=fcm)
        create_file(tmp_path, "plan.md", "# Plan\nfull content here")

        await tool.run(path="plan.md")

        assert fcm.contains("plan.md")
        assert fcm.get("plan.md") == "# Plan\nfull content here"

    async def test_read_caches_full_content_even_with_offset_limit(self, workspace, tmp_path):
        """A partial read (offset/limit) should still seed the cache with
        the WHOLE file, not just the requested slice -- so a later turn
        working on a different part of the file doesn't need to re-read."""
        from chef_human.agent.file_context import FileContextManager
        from chef_human.llm.tokenizer import ApproxTokenizer

        fcm = FileContextManager(workspace=workspace, tokenizer=ApproxTokenizer())
        tool = ReadTool(workspace, file_context=fcm)
        create_file(tmp_path, "f.txt", "a\nb\nc\nd\ne\n")

        await tool.run(path="f.txt", offset=2, limit=2)

        assert fcm.get("f.txt") == "a\nb\nc\nd\ne\n"

    async def test_read_without_file_context_does_not_error(self, tmp_path):
        tool = ReadTool(WorkspaceManager(root=tmp_path))
        create_file(tmp_path, "f.txt", "hello")
        result = await tool.run(path="f.txt")
        assert result.success

    async def test_write_populates_file_context(self, workspace, tmp_path):
        from chef_human.agent.file_context import FileContextManager
        from chef_human.llm.tokenizer import ApproxTokenizer

        fcm = FileContextManager(workspace=workspace, tokenizer=ApproxTokenizer())
        tool = WriteTool(workspace, file_context=fcm)

        await tool.run(path="new.py", content="x = 1")

        assert fcm.get("new.py") == "x = 1"

    async def test_write_refreshes_stale_file_context(self, workspace, tmp_path):
        from chef_human.agent.file_context import FileContextManager
        from chef_human.llm.tokenizer import ApproxTokenizer

        fcm = FileContextManager(workspace=workspace, tokenizer=ApproxTokenizer())
        read_tool = ReadTool(workspace, file_context=fcm)
        write_tool = WriteTool(workspace, file_context=fcm)
        create_file(tmp_path, "f.txt", "original")

        await read_tool.run(path="f.txt")  # caches "original"
        await write_tool.run(path="f.txt", content="overwritten")

        assert fcm.get("f.txt") == "overwritten"

    async def test_edit_populates_file_context_with_new_content(self, workspace, tmp_path):
        from chef_human.agent.file_context import FileContextManager
        from chef_human.llm.tokenizer import ApproxTokenizer

        fcm = FileContextManager(workspace=workspace, tokenizer=ApproxTokenizer())
        tool = EditTool(workspace, file_context=fcm)
        create_file(tmp_path, "f.txt", "hello world")

        await tool.run(path="f.txt", old_string="world", new_string="there")

        assert fcm.get("f.txt") == "hello there"

    async def test_edit_does_not_populate_file_context_on_failure(self, workspace, tmp_path):
        from chef_human.agent.file_context import FileContextManager
        from chef_human.llm.tokenizer import ApproxTokenizer

        fcm = FileContextManager(workspace=workspace, tokenizer=ApproxTokenizer())
        tool = EditTool(workspace, file_context=fcm)
        create_file(tmp_path, "f.txt", "hello world")

        result = await tool.run(path="f.txt", old_string="zzz", new_string="xxx", fuzzy=False)

        assert not result.success
        assert not fcm.contains("f.txt")


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

    async def test_does_not_follow_symlink_outside_workspace(self, grep_tool, tmp_path, tmp_path_factory):
        outside = tmp_path_factory.mktemp("outside")
        secret = outside / "secret.txt"
        secret.write_text("top secret token")
        (tmp_path / "escape").symlink_to(outside, target_is_directory=True)

        result = await grep_tool.run(pattern="secret")
        assert result.success
        assert "top secret" not in result.output
        assert "secret.txt" not in result.output

    async def test_search_timeout_is_surfaced(self, grep_tool, monkeypatch):
        # Exercises the wait_for/timeout wiring via a mocked slow _search --
        # deliberately not a real catastrophic-backtracking pattern, since a
        # genuinely stuck worker thread cannot be killed and would hang the
        # test process at teardown (confirmed by hand: this is not
        # theoretical -- it hung a real pytest run during development).
        import time

        monkeypatch.setattr(type(grep_tool), "SEARCH_TIMEOUT", 0.05)
        monkeypatch.setattr(
            grep_tool, "_search", lambda base, compiled, include: (time.sleep(1), ([], False))[1]
        )

        result = await grep_tool.run(pattern="anything")

        assert not result.success
        assert "timed out" in result.error

    async def test_oversized_line_is_skipped_not_searched(self, grep_tool, tmp_path):
        # MAX_LINE_LENGTH is the real bound on catastrophic-backtracking
        # cost -- verify it actually skips long lines rather than being
        # decorative.
        grep_tool.MAX_LINE_LENGTH = 20
        create_file(tmp_path, "a.py", "findme " + "x" * 100 + "\nshort findme line\n")

        result = await grep_tool.run(pattern="findme")

        assert result.success
        assert "short findme line" in result.output
        assert "x" * 100 not in result.output

    async def test_permission_denied_surfaced_not_silent(self, grep_tool, tmp_path):
        create_file(tmp_path, "visible.py", "def findme(): pass")
        locked = tmp_path / "locked"
        locked.mkdir()
        (locked / "hidden.py").write_text("def findme(): pass")
        locked.chmod(0o000)
        try:
            result = await grep_tool.run(pattern="findme")
        finally:
            locked.chmod(0o755)  # so pytest can clean up tmp_path afterward

        # Matches found before hitting the inaccessible directory are kept,
        # but the permission gap must not be silently invisible.
        assert "visible.py" in result.output
        assert "permission denied" in result.output.lower()


# ---------------------------------------------------------------------------
# GlobTool
# ---------------------------------------------------------------------------

class TestGlobTool:
    async def test_does_not_follow_symlink_outside_workspace(self, glob_tool, tmp_path, tmp_path_factory):
        outside = tmp_path_factory.mktemp("outside")
        (outside / "secret.py").write_text("")
        (tmp_path / "escape").symlink_to(outside, target_is_directory=True)

        result = await glob_tool.run(pattern="**/*.py")
        assert result.success
        assert "secret.py" not in result.output

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
