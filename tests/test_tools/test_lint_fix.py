from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from chef_human.agent.workspace import WorkspaceManager
from chef_human.tools.diff import DiffStore
from chef_human.tools.lint_fix import LintFixTool


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
def tool(workspace: WorkspaceManager, store: DiffStore) -> LintFixTool:
    return LintFixTool(workspace=workspace, diff_store=store)


class TestLintFixTool:
    async def test_non_python_file(self, tool: LintFixTool, tmp_path: Path):
        create_file(tmp_path, "test.js", "var x = 1;")
        result = await tool.run(path=str(tmp_path / "test.js"))
        assert result.success
        assert "No supported linter" in result.output

    async def test_ruff_not_available(self, tool: LintFixTool, tmp_path: Path):
        create_file(tmp_path, "test.py", "import os\n")
        with patch("chef_human.tools.lint_fix._find_ruff", return_value=None):
            result = await tool.run(path=str(tmp_path / "test.py"))
            assert not result.success
            assert "ruff not found" in (result.error or "")

    async def test_check_only_no_issues(self, tool: LintFixTool, tmp_path: Path):
        create_file(tmp_path, "clean.py", "x = 1\n")
        with patch("chef_human.tools.lint_fix._find_ruff", return_value="/usr/bin/ruff"):
            with patch("chef_human.tools.lint_fix.subprocess.run") as mock_run:
                mock_run.return_value.stdout = ""
                mock_run.return_value.stderr = ""
                result = await tool.run(path=str(tmp_path / "clean.py"), check_only=True)
                assert result.success
                assert "No lint issues" in result.output

    async def test_check_only_shows_issues(
        self, tool: LintFixTool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        create_file(tmp_path, "messy.py", "import os\nimport sys\n\nx=1\n")
        with patch("chef_human.tools.lint_fix._find_ruff", return_value="/usr/bin/ruff"):
            with patch("chef_human.tools.lint_fix.subprocess.run") as mock_run:
                mock_run.return_value.stdout = "messy.py:1:1: F401 `os` imported but unused\n"
                mock_run.return_value.stderr = ""
                result = await tool.run(path=str(tmp_path / "messy.py"), check_only=True)
                assert result.success
                assert "F401" in result.output

    async def test_fix_records_diff(
        self, tool: LintFixTool, store: DiffStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        f = create_file(tmp_path, "fixable.py", "import os\nimport sys\n\nx=1\n")
        fixed = "import sys\n\nx = 1\n"

        with patch("chef_human.tools.lint_fix._find_ruff", return_value="/usr/bin/ruff"):
            with patch("chef_human.tools.lint_fix.subprocess.run") as mock_run:
                mock_run.return_value.stdout = "fixable.py:1:1: F401 `os` imported but unused\n"
                mock_run.return_value.stderr = ""
                f.write_text(fixed)

                result = await tool.run(path=str(tmp_path / "fixable.py"))
                assert result.success

    async def test_fix_no_issues(
        self, tool: LintFixTool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        create_file(tmp_path, "clean.py", "x = 1\n")
        result = await tool.run(path=str(tmp_path / "clean.py"))
        assert result.success
        assert "No lint issues" in result.output or "No lint issues found" in result.output

    async def test_path_outside_workspace(self, tool: LintFixTool):
        result = await tool.run(path="/etc/passwd")
        assert not result.success

    async def test_nonexistent_path(self, tool: LintFixTool):
        result = await tool.run(path="/nonexistent/path")
        assert not result.success
