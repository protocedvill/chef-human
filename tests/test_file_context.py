from __future__ import annotations

from pathlib import Path

import pytest

from chef_human.agent.file_context import FileContextManager
from chef_human.agent.workspace import WorkspaceManager
from chef_human.llm.tokenizer import ApproxTokenizer


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceManager:
    return WorkspaceManager(root=tmp_path)


@pytest.fixture
def tokenizer() -> ApproxTokenizer:
    return ApproxTokenizer()


@pytest.fixture
def fcm(workspace: WorkspaceManager, tokenizer: ApproxTokenizer) -> FileContextManager:
    return FileContextManager(
        workspace=workspace,
        tokenizer=tokenizer,
        max_files=10,
        max_tokens=100,
    )


def create_file(directory: Path, name: str, content: str = "") -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


class TestFileContextManagerInit:
    def test_accepts_dependencies(self, workspace, tokenizer):
        fcm = FileContextManager(workspace=workspace, tokenizer=tokenizer)
        assert fcm._max_files == 50
        assert fcm._max_tokens == 10_000

    def test_custom_limits(self, workspace, tokenizer):
        fcm = FileContextManager(
            workspace=workspace, tokenizer=tokenizer, max_files=5, max_tokens=500
        )
        assert fcm._max_files == 5
        assert fcm._max_tokens == 500

    def test_starts_empty(self, workspace, tokenizer):
        fcm = FileContextManager(workspace=workspace, tokenizer=tokenizer)
        assert fcm.cached_files() == []
        assert fcm.total_tokens() == 0


class TestFileContextManagerGet:
    def test_returns_file_content(self, tmp_path, fcm):
        create_file(tmp_path, "hello.txt", content="Hello, World!")
        content = fcm.get("hello.txt")
        assert content == "Hello, World!"

    def test_returns_none_for_missing_file(self, fcm):
        assert fcm.get("nonexistent.txt") is None

    def test_returns_none_for_outside_file(self, tmp_path, fcm):
        outside = Path("/tmp/outside.txt")
        outside.write_text("outside")
        assert fcm.get(str(outside)) is None

    def test_caches_content_on_first_get(self, tmp_path, fcm):
        path = create_file(tmp_path, "data.txt", content="data")
        fcm.get("data.txt")
        assert fcm.contains(path)

    def test_returns_cached_content_on_second_get(self, tmp_path, fcm):
        create_file(tmp_path, "data.txt", content="original")
        assert fcm.get("data.txt") == "original"
        # Modify file on disk
        (tmp_path / "data.txt").write_text("modified")
        # Should still return cached version
        assert fcm.get("data.txt") == "original"

    def test_handles_binary_file_gracefully(self, tmp_path, fcm):
        path = tmp_path / "data.bin"
        path.write_bytes(b"\xff\xfe\x00\x01")
        content = fcm.get(str(path))
        assert content is not None
        assert isinstance(content, str)

    def test_updates_access_order_on_second_get(self, tmp_path, fcm):
        a = create_file(tmp_path, "a.txt", content="aaa")
        b = create_file(tmp_path, "b.txt", content="bbb")
        fcm.get("a.txt")
        fcm.get("b.txt")
        fcm.get("a.txt")  # touch a again
        assert fcm.cached_files() == [b, a]


class TestFileContextManagerGetLines:
    def test_returns_line_range(self, tmp_path, fcm):
        create_file(tmp_path, "lines.txt", content="line1\nline2\nline3\n")
        lines = fcm.get_lines("lines.txt", start=2, end=3)
        assert lines == ["line2\n", "line3\n"]

    def test_default_start_is_one(self, tmp_path, fcm):
        create_file(tmp_path, "lines.txt", content="first\nsecond\n")
        lines = fcm.get_lines("lines.txt", end=1)
        assert lines == ["first\n"]

    def test_none_end_returns_to_end(self, tmp_path, fcm):
        create_file(tmp_path, "lines.txt", content="a\nb\nc\n")
        lines = fcm.get_lines("lines.txt", start=2)
        assert lines == ["b\n", "c\n"]

    def test_returns_none_for_missing_file(self, fcm):
        assert fcm.get_lines("missing.txt") is None

    def test_keeps_newlines(self, tmp_path, fcm):
        create_file(tmp_path, "f.txt", content="hello\nworld")
        lines = fcm.get_lines("f.txt")
        assert lines == ["hello\n", "world"]

    def test_start_beyond_file_length(self, tmp_path, fcm):
        create_file(tmp_path, "f.txt", content="only one line")
        lines = fcm.get_lines("f.txt", start=10)
        assert lines == []


class TestFileContextManagerRemove:
    def test_removes_from_cache(self, tmp_path, fcm):
        create_file(tmp_path, "f.txt", content="data")
        fcm.get("f.txt")
        fcm.remove("f.txt")
        assert not fcm.contains("f.txt")
        assert "f.txt" not in [str(p) for p in fcm.cached_files()]

    def test_no_error_if_not_cached(self, fcm):
        fcm.remove("nonexistent.txt")  # should not raise


class TestFileContextManagerClear:
    def test_empties_cache(self, tmp_path, fcm):
        create_file(tmp_path, "a.txt", content="a")
        create_file(tmp_path, "b.txt", content="b")
        fcm.get("a.txt")
        fcm.get("b.txt")
        fcm.clear()
        assert fcm.cached_files() == []
        assert fcm.total_tokens() == 0


class TestFileContextManagerEviction:
    def test_evicts_oldest_when_max_files_exceeded(self, tmp_path, fcm):
        for i in range(12):
            create_file(tmp_path, f"f{i}.txt", content="hello")
            fcm.get(f"f{i}.txt")
        assert len(fcm.cached_files()) <= fcm._max_files

    def test_most_recently_used_survives(self, tmp_path, fcm):
        files = {}
        for i in range(12):
            p = create_file(tmp_path, f"f{i}.txt", content="hello")
            files[i] = p
            fcm.get(f"f{i}.txt")
        # f0 was accessed first, should be evicted
        assert not fcm.contains(files[0])

    def test_touching_preserves_file(self, tmp_path, fcm):
        create_file(tmp_path, "keep.txt", content="hello")
        fcm.get("keep.txt")
        for i in range(9):  # +9 = 10 total, no eviction needed
            create_file(tmp_path, f"f{i}.txt", content="hello")
            fcm.get(f"f{i}.txt")
        assert fcm.contains(tmp_path / "keep.txt")

    def test_evicts_when_token_budget_exceeded(self, tmp_path, fcm):
        # Each "hello" is 5 chars → 1 token (4 chars/token, min 1)
        # max_tokens = 100, so ~100 files fit
        for i in range(150):
            create_file(tmp_path, f"f{i}.txt", content="hello")
            fcm.get(f"f{i}.txt")
        assert fcm.total_tokens() <= fcm._max_tokens

    def test_eviction_removes_oldest_first(self, tmp_path, fcm):
        create_file(tmp_path, "first.txt", content="a" * 8)  # 2 tokens
        fcm.get("first.txt")
        create_file(tmp_path, "second.txt", content="a" * 8)  # 2 tokens
        fcm.get("second.txt")
        # 500 chars = 125 tokens, total = 129 > 100 max → triggers token eviction
        create_file(tmp_path, "third.txt", content="a" * 500)
        fcm.get("third.txt")
        # first.txt should have been evicted (oldest)
        assert not fcm.contains(tmp_path / "first.txt")

    def test_eviction_logs_debug_message(self, tmp_path, fcm, caplog):
        caplog.set_level("DEBUG")
        for i in range(12):
            create_file(tmp_path, f"f{i}.txt", content="hello")
            fcm.get(f"f{i}.txt")
        assert any("Evicted from file cache" in rec.message for rec in caplog.records)


class TestFileContextManagerTotalTokens:
    def test_zero_when_empty(self, fcm):
        assert fcm.total_tokens() == 0

    def test_counts_all_cached_files(self, tmp_path, fcm):
        create_file(tmp_path, "a.txt", content="hello")  # 5 chars → 1 token
        create_file(tmp_path, "b.txt", content="world")  # 5 chars → 1 token
        fcm.get("a.txt")
        fcm.get("b.txt")
        assert fcm.total_tokens() == 2

    def test_updates_after_removal(self, tmp_path, fcm):
        create_file(tmp_path, "a.txt", content="hello")
        fcm.get("a.txt")
        before = fcm.total_tokens()
        fcm.remove("a.txt")
        assert fcm.total_tokens() == before - 1


class TestFileContextManagerContains:
    def test_false_when_not_cached(self, tmp_path, fcm):
        create_file(tmp_path, "f.txt", content="x")
        assert not fcm.contains("f.txt")

    def test_true_after_get(self, tmp_path, fcm):
        create_file(tmp_path, "f.txt", content="x")
        fcm.get("f.txt")
        assert fcm.contains("f.txt")

    def test_false_after_remove(self, tmp_path, fcm):
        create_file(tmp_path, "f.txt", content="x")
        fcm.get("f.txt")
        fcm.remove("f.txt")
        assert not fcm.contains("f.txt")
