from __future__ import annotations

import pytest

from chef_human.tools.user import FinishTool


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
