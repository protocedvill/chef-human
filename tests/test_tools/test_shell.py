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

    @pytest.mark.parametrize(
        "command",
        [
            "git add foo.py",
            "npm add lodash",
            "yarn add react",
            "echo added",
            "mkdir addons",
            "git commit --amend",
        ],
    )
    async def test_blacklist_does_not_false_positive_on_add(self, bash_tool, command):
        """Regression test: the blacklist's "dd" entry (for the `dd`
        command) must not substring-match the "dd" inside "add"."""
        result = await bash_tool.run(command=command)
        assert result.success or "blocked" not in (result.error or "")

    def test_blacklist_matches_dd_as_a_command_not_a_substring(self):
        assert BashTool._is_blacklisted("dd if=/dev/zero of=/dev/sda")
        assert not BashTool._is_blacklisted("git add foo.py")
        assert not BashTool._is_blacklisted("npm add lodash")

    def test_blacklist_catches_chained_and_wrapped_destructive_commands(self):
        assert BashTool._is_blacklisted("echo hi; dd if=x of=y")
        assert BashTool._is_blacklisted("bash -c 'dd if=x of=y'")
        assert BashTool._is_blacklisted("sudo reboot")

    def test_destructive_detects_chained_commands(self):
        """A leading benign command must not hide a later destructive one."""
        assert BashTool._is_destructive("echo hi && rm important_file.txt")
        assert BashTool._is_destructive("echo hi; rmdir project")

    def test_destructive_detects_wrapped_commands(self):
        assert BashTool._is_destructive('bash -c "rm -rf ./build"')
        assert BashTool._is_destructive("sudo rm -rf /tmp/x")
        assert BashTool._is_destructive("sudo -u root rm file")

    def test_destructive_does_not_flag_ordinary_redirects(self):
        assert not BashTool._is_destructive("pytest > out.log")
        assert not BashTool._is_destructive("echo hi")
