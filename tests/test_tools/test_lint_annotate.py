from __future__ import annotations

from chef_human.agent.linter import annotate_diff_with_lint


class TestAnnotateDiffWithLint:
    def test_empty_diff_returns_empty(self):
        result = annotate_diff_with_lint("", "foo.py:1:1: F401 error")
        assert result == ""

    def test_empty_lint_returns_original_diff(self):
        diff = "@@ -0,0 +1 @@\n+foo\n"
        result = annotate_diff_with_lint(diff, "")
        assert result == diff

    def test_annotates_plus_line(self):
        diff = "@@ -1,3 +1,3 @@\n a\n-b\n+c\n d\n"
        lint = "foo.py:2:1: F841 local variable 'c' is assigned to but never used"
        result = annotate_diff_with_lint(diff, lint)
        assert "ruff: F841" in result
        assert "c" in result

    def test_annotates_plus_line_with_multiple_warnings(self):
        diff = "@@ -1,2 +1,2 @@\n-old\n+new_line\n"
        lint = (
            "foo.py:1:1: F401 `os` imported but unused\n"
            "foo.py:1:5: E302 expected 2 blank lines"
        )
        result = annotate_diff_with_lint(diff, lint)
        assert "ruff: F401" in result
        assert "ruff: E302" in result
        # Second hunk line (current_line = 1) gets both annotations

    def test_no_annotation_on_minus_line(self):
        diff = "@@ -1,2 +1,1 @@\n-old_line\n+new_line\n"
        lint = "foo.py:1:1: F401 error"  # line 1 of old (minus side, should not annotate minus)
        result = annotate_diff_with_lint(diff, lint)
        # old_line is removed, it's a minus line -> no annotation
        # new_line is line 1 on the plus side -> gets annotation
        assert "ruff: F401" in result
        assert "new_line" in result

    def test_no_lint_matches_no_change(self):
        diff = "@@ -1,3 +1,3 @@\n a\n-b\n+c\n d\n"
        lint = "other.py:99:1: X999 some warning"
        result = annotate_diff_with_lint(diff, lint)
        assert result == diff

    def test_context_line_with_warning_not_annotated(self):
        diff = "@@ -0,0 +1 @@\n+new\n"
        lint = "foo.py:2:1: E999 error"
        result = annotate_diff_with_lint(diff, lint)
        assert result == diff

    def test_multiple_hunks_both_annotated(self):
        diff = (
            "@@ -1,2 +1,2 @@\n a\n-b\n+c\n"
            "@@ -10,2 +10,2 @@\n x\n-y\n+z\n"
        )
        # Context lines consume hunk_new_start, so +c is at new line 2 and +z at 11
        lint = (
            "foo.py:2:1: F401 first\n"
            "foo.py:11:1: E302 second"
        )
        result = annotate_diff_with_lint(diff, lint)
        assert "ruff: F401" in result
        assert "ruff: E302" in result

    def test_malformed_lint_line_ignored(self):
        diff = "@@ -0,0 +1 @@\n+line\n"
        lint = "not a valid lint line"
        result = annotate_diff_with_lint(diff, lint)
        assert result == diff

    def test_lint_line_parsing_edge_cases(self):
        diff = "@@ -0,0 +1 @@\n+code\n"
        lint = "foo.py:1:5: F401 `complicated.name.with.dots` is imported but unused"
        result = annotate_diff_with_lint(diff, lint)
        assert "ruff: F401" in result
        assert "complicated.name.with.dots" in result

    def test_custom_linter_name(self):
        diff = "@@ -0,0 +1 @@\n+code\n"
        lint = "foo.py:1:5: F401 unused import"
        result = annotate_diff_with_lint(diff, lint, linter_name="pylint")
        assert "pylint: F401" in result
        assert "# pylint:" in result
