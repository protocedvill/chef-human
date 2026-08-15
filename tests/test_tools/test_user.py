from __future__ import annotations

from unittest.mock import patch

from chef_human.tools.user import AskUserTool, FinishTool


class TestAskUserTool:
    async def test_headless_returns_fallback(self):
        """When not a TTY, AskUserTool returns a deterministic fallback message."""
        with patch("sys.stdin.isatty", return_value=False):
            tool = AskUserTool()
            result = await tool.run("What do you think?")
            assert result.success
            assert "no-tty" in result.output

    async def test_headless_does_not_block(self):
        """In headless mode, the tool returns immediately without reading stdin."""
        with patch("sys.stdin.isatty", return_value=False):
            tool = AskUserTool()
            result = await tool.run("What do you think?")
            assert "Continuing without answer" in result.output

    async def test_interactive_works_with_isatty(self):
        """When stdin is a TTY, the tool attempts to read input."""
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("sys.stdin.readline", return_value="user response"),
        ):
            tool = AskUserTool()
            result = await tool.run("What do you think?")
            assert result.success
            assert result.output == "user response"

    async def test_interactive_skip_returns_fallback(self):
        """When user types 'skip', tool returns a fallback message."""
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("sys.stdin.readline", return_value="skip"),
        ):
            tool = AskUserTool()
            result = await tool.run("Should I continue?")
            assert result.success
            assert "skipped" in result.output

    async def test_interactive_empty_returns_fallback(self):
        """When user types empty string, tool returns a fallback message."""
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("sys.stdin.readline", return_value=""),
        ):
            tool = AskUserTool()
            result = await tool.run("Should I continue?")
            assert result.success
            assert "skipped" in result.output


class TestFinishTool:
    async def test_basic(self):
        tool = FinishTool()
        result = await tool.run()
        assert result.success
        assert result.output == "Task complete"

    async def test_with_summary(self):
        tool = FinishTool()
        result = await tool.run(summary="Fixed the bug")
        assert result.success
        assert "Fixed the bug" in result.output

    async def test_empty_summary(self):
        tool = FinishTool()
        result = await tool.run(summary="")
        assert result.success
        assert result.output == "Task complete"
