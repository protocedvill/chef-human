from __future__ import annotations

from unittest.mock import patch

from chef_human.agent.linter import (
    _detect_linter,
    _find_ruff,
    format_lint_result,
    run_lint,
)


def test_detect_linter_python():
    assert _detect_linter("foo.py") == "ruff"


def test_detect_linter_non_python():
    assert _detect_linter("foo.js") is None
    assert _detect_linter("foo.md") is None
    assert _detect_linter("") is None


def test_find_ruff_not_found():
    with patch("chef_human.agent.linter.shutil.which", return_value=None):
        assert _find_ruff() is None


def test_run_lint_non_python_returns_empty():
    assert run_lint("foo.js") == ""


def test_run_lint_ruff_not_available():
    with patch("chef_human.agent.linter._find_ruff", return_value=None):
        assert run_lint("foo.py") == ""


def test_format_lint_result_empty():
    assert format_lint_result("") == ""


def test_format_lint_result_single():
    result = format_lint_result("foo.py:1:1: F401 `os` imported but unused")
    assert "Lint results" in result
    assert "1 issue" in result


def test_format_lint_result_multiple():
    result = format_lint_result(
        "foo.py:1:1: F401 error\nfoo.py:2:3: E302 error"
    )
    assert "2 issues" in result
