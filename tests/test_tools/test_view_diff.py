from __future__ import annotations

from chef_human.tools.diff import DiffStore
from chef_human.tools.view_diff import ViewDiffTool


class TestViewDiffTool:
    async def test_empty_store(self):
        store = DiffStore()
        tool = ViewDiffTool(store)
        result = await tool.run()
        assert result.success
        assert "No changes" in result.output

    async def test_single_entry(self):
        store = DiffStore()
        store.record("a.py", "```diff\n-foo\n+bar\n```\n", "edit")
        tool = ViewDiffTool(store)
        result = await tool.run()
        assert result.success
        assert "edit: a.py" in result.output
        assert "```diff" in result.output

    async def test_multiple_files(self):
        store = DiffStore()
        store.record("a.py", "diff1", "edit")
        store.record("b.py", "diff2", "write")
        tool = ViewDiffTool(store)
        result = await tool.run()
        assert result.success
        assert "edit: a.py" in result.output
        assert "write: b.py" in result.output

    async def test_filter_by_path(self):
        store = DiffStore()
        store.record("a.py", "diff_a", "edit")
        store.record("b.py", "diff_b", "write")
        tool = ViewDiffTool(store)
        result = await tool.run(path="a.py")
        assert result.success
        assert "diff_a" in result.output
        assert "diff_b" not in result.output

    async def test_filter_missing_path(self):
        store = DiffStore()
        store.record("a.py", "diff", "edit")
        tool = ViewDiffTool(store)
        result = await tool.run(path="missing.py")
        assert result.success
        assert "No changes" in result.output

    async def test_after_clear(self):
        store = DiffStore()
        store.record("a.py", "diff", "edit")
        store.clear()
        tool = ViewDiffTool(store)
        result = await tool.run()
        assert "No changes" in result.output
