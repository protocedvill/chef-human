from __future__ import annotations

from pathlib import Path

import pytest

from chef_human.agent.workspace import WorkspaceManager
from chef_human.tools.shell import BashTool


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceManager:
    return WorkspaceManager(root=tmp_path)


@pytest.fixture
def bash_tool(workspace: WorkspaceManager) -> BashTool:
    return BashTool(workspace)


class TestBashTool:
    async def test_echo(self, bash_tool):
        result = await bash_tool.run(command="echo hello")
        assert result.success
        assert "hello" in result.output

    async def test_exit_code(self, bash_tool):
        result = await bash_tool.run(command="exit 42")
        assert not result.success
        assert "42" in result.error

    async def test_cwd(self, bash_tool, tmp_path):
        (tmp_path / "marker").write_text("found")
        result = await bash_tool.run(command="cat marker", workdir=".")
        assert result.success
        assert "found" in result.output

    async def test_timeout(self, bash_tool):
        result = await bash_tool.run(command="sleep 10", timeout=1)
        assert not result.success
        assert "timed out" in result.error

    async def test_stderr_captured(self, bash_tool):
        result = await bash_tool.run(command="echo err >&2")
        assert result.success
        assert "err" in result.output

    async def test_blacklist_blocks_rm_rf(self, bash_tool):
        result = await bash_tool.run(command="rm -rf /")
        assert not result.success
        assert "blocked" in result.error

    async def test_blacklist_blocks_dd(self, bash_tool):
        result = await bash_tool.run(command="dd if=/dev/zero of=/dev/sda")
        assert not result.success
        assert "blocked" in result.error

    async def test_blacklist_blocks_reboot(self, bash_tool):
        result = await bash_tool.run(command="reboot")
        assert not result.success
        assert "blocked" in result.error

    async def test_destructive_detected(self, bash_tool):
        assert BashTool._is_destructive("rm file.txt")
        assert BashTool._is_destructive("mv a b")
        assert not BashTool._is_destructive("echo hello")
        assert not BashTool._is_destructive("ls -la")

    async def test_outside_workdir(self, bash_tool):
        result = await bash_tool.run(command="pwd", workdir="/etc")
        assert not result.success
        assert "Outside" in result.error

    async def test_timeout_capped(self, bash_tool):
        assert bash_tool.TIMEOUT_MAX == 300
