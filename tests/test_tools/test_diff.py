from __future__ import annotations

from chef_human.tools.diff import (
    DiffStore,
    MatchResult,
    compute_diff,
    find_closest_match,
)


class TestComputeDiff:
    def test_changed_content(self):
        result = compute_diff("a\nb\nc\n", "a\nx\nc\n", "f.txt")
        assert "```diff" in result
        assert "-b" in result
        assert "+x" in result
        assert "a/f.txt" in result
        assert "b/f.txt" in result

    def test_identical_content(self):
        assert compute_diff("same\n", "same\n") == ""

    def test_empty_strings(self):
        assert compute_diff("", "") == ""

    def test_from_empty(self):
        result = compute_diff("", "hello\n", "new.txt")
        assert "```diff" in result
        assert "+hello" in result

    def test_to_empty(self):
        result = compute_diff("hello\n", "", "gone.txt")
        assert "```diff" in result
        assert "-hello" in result

    def test_trailing_newline_diff(self):
        result = compute_diff("a\nb\nc", "a\nb\nc\n", "f.txt")
        assert result  # should produce a diff (added newline at end)

    def test_no_path(self):
        result = compute_diff("a\n", "b\n")
        assert "```diff" in result
        assert "a/" not in result  # no path prefix when path="" (well, will be "a")

    def test_multiline_diff(self):
        old = "def foo():\n    pass\n\ndef bar():\n    return 1\n"
        new = "def foo():\n    return 42\n\ndef bar():\n    return 1\n"
        result = compute_diff(old, new, "mod.py")
        assert "```diff" in result
        assert "-    pass" in result
        assert "+    return 42" in result


class TestFindClosestMatch:
    def test_exact_match(self):
        content = "def foo():\n    pass\n"
        result = find_closest_match("def foo():\n    pass", content, min_ratio=1.0)
        assert result is not None
        assert result.ratio == 1.0

    def test_fuzzy_whitespace_diff(self):
        content = "def foo():\n    return 42\n"
        result = find_closest_match("return  42", content, min_ratio=0.5)
        assert result is not None
        assert result.ratio > 0.5

    def test_fuzzy_extra_newline(self):
        content = "def foo():\n    pass\n"
        result = find_closest_match("def foo():\n    pass\n\n", content, min_ratio=0.7)
        assert result is not None, "Should match despite extra trailing newline"

    def test_below_threshold(self):
        content = "def foo():\n    pass\n"
        result = find_closest_match(
            "def bar():\n    xyz", content, min_ratio=0.95
        )
        assert result is None

    def test_empty_old_string(self):
        assert find_closest_match("", "content") is None

    def test_empty_content(self):
        assert find_closest_match("foo", "") is None

    def test_no_common_lines(self):
        content = "aaaa\nbbbb\ncccc\n"
        result = find_closest_match("xyz\n123", content, min_ratio=0.5)
        assert result is None

    def test_windows_includes_correct_range(self):
        content = "line1\nline2\ndef target():\n    pass\nline4\nline5\n"
        result = find_closest_match(
            "def target():\n    return 42", content, min_ratio=0.5
        )
        assert result is not None
        # Window should start at or near the target line
        assert "target" in result.matched_text

    def test_custom_min_ratio(self):
        content = "def foo():\n    pass\n"
        result = find_closest_match("def foo():\n    pass", content, min_ratio=0.99)
        assert result is not None and result.ratio >= 0.99

    def test_match_result_dataclass(self):
        r = MatchResult(matched_text="abc", ratio=0.9, start_line=1, end_line=2)
        assert r.matched_text == "abc"
        assert r.ratio == 0.9
        assert r.start_line == 1
        assert r.end_line == 2


class TestDiffStore:
    def test_record_and_get_all(self):
        store = DiffStore()
        store.record("a.py", "diff1", "edit")
        store.record("b.py", "diff2", "write")
        entries = store.get_all()
        assert len(entries) == 2

    def test_filter_by_path(self):
        store = DiffStore()
        store.record("a.py", "diff1", "edit")
        store.record("b.py", "diff2", "write")
        a_entries = store.get_all("a.py")
        assert len(a_entries) == 1
        assert a_entries[0].path == "a.py"

    def test_clear(self):
        store = DiffStore()
        store.record("a.py", "diff", "edit")
        store.clear()
        assert len(store.get_all()) == 0

    def test_skip_empty_diff(self):
        store = DiffStore()
        store.record("a.py", "", "edit")
        assert len(store.get_all()) == 0

    def test_get_summary_empty(self):
        store = DiffStore()
        assert "No changes" in store.get_summary()

    def test_get_summary_with_entries(self):
        store = DiffStore()
        store.record("a.py", "diff1", "edit")
        store.record("b.py", "diff2", "write")
        summary = store.get_summary()
        assert "No changes" not in summary
        assert "edit: a.py" in summary
        assert "write: b.py" in summary

    def test_entries_have_tool_name(self):
        store = DiffStore()
        store.record("a.py", "diff", "edit")
        assert store.get_all()[0].tool_name == "edit"

    def test_record_with_content(self):
        store = DiffStore()
        store.record("a.py", "diff", "edit", old_content="old\n", new_content="new\n")
        entry = store.get_all()[0]
        assert entry.old_content == "old\n"
        assert entry.new_content == "new\n"

    def test_last_returns_most_recent(self):
        store = DiffStore()
        store.record("a.py", "d1", "edit")
        store.record("b.py", "d2", "write")
        last = store.last()
        assert last is not None
        assert last.path == "b.py"

    def test_last_empty_store(self):
        store = DiffStore()
        assert store.last() is None

    def test_last_filter_by_path(self):
        store = DiffStore()
        store.record("a.py", "d1", "edit")
        store.record("b.py", "d2", "write")
        store.record("a.py", "d3", "edit")
        last_a = store.last("a.py")
        assert last_a is not None
        assert last_a.diff == "d3"

    def test_pop_last_removes_and_returns(self):
        store = DiffStore()
        store.record("a.py", "d1", "edit")
        store.record("b.py", "d2", "write")
        popped = store.pop_last()
        assert popped is not None
        assert popped.path == "b.py"
        assert len(store.get_all()) == 1

    def test_pop_last_empty_store(self):
        store = DiffStore()
        assert store.pop_last() is None

    def test_pop_last_filter_by_path(self):
        store = DiffStore()
        store.record("a.py", "d1", "edit")
        store.record("b.py", "d2", "write")
        store.record("a.py", "d3", "edit")
        popped = store.pop_last("a.py")
        assert popped is not None
        assert popped.diff == "d3"
        remaining = store.get_all("a.py")
        assert len(remaining) == 1
        assert remaining[0].diff == "d1"

    def test_pop_last_missing_path(self):
        store = DiffStore()
        store.record("a.py", "d1", "edit")
        assert store.pop_last("missing.py") is None
        assert len(store.get_all()) == 1
