from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from chef_human.tools.registry import Tool, ToolResult

if TYPE_CHECKING:
    from chef_human.agent.workspace import WorkspaceManager

logger = logging.getLogger(__name__)

BLACKLIST: set[str] = {
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd",
    "> /dev/",
    ":(){ :|:& };:",
    "chmod 777 /",
    "chown",
    "halt",
    "poweroff",
    "reboot",
    "shutdown",
}

DESTRUCTIVE_PREFIXES: tuple[str, ...] = (
    "rm",
    "rmdir",
    "mv",
    "dd",
    "format",
    "mkfs",
    ">",
    ">>",
    "|",
)


class BashTool:
    name = "bash"
    description = "Execute a shell command in the workspace"
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "timeout": {"type": "integer", "description": "Timeout in seconds (default: 30)", "default": 30},
            "workdir": {"type": "string", "description": "Working directory (default: workspace root)", "default": None},
        },
        "required": ["command"],
    }

    TIMEOUT_DEFAULT = 30
    TIMEOUT_MAX = 300

    def __init__(self, workspace: WorkspaceManager) -> None:
        self._workspace = workspace

    async def run(self, command: str, timeout: int = TIMEOUT_DEFAULT, workdir: str | None = None) -> ToolResult:
        timeout = min(timeout, self.TIMEOUT_MAX)

        if self._is_blacklisted(command):
            logger.warning("Blocked blacklisted command: %s", command[:80])
            return ToolResult(success=False, error="Command blocked: operation not allowed")

        cwd = self._workspace.resolve(workdir) if workdir else self._workspace.root

        if not self._workspace.is_within_workspace(cwd):
            return ToolResult(success=False, error=f"Outside workspace: {cwd}")

        is_destructive = self._is_destructive(command)
        if is_destructive:
            logger.info("Destructive command detected: %s", command[:80])

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
                env={**os.environ, "HOME": str(Path.home())},
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            return ToolResult(success=False, error=f"Command timed out after {timeout}s")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

        output = ""
        if stdout:
            output += stdout.decode("utf-8", errors="replace")
        if stderr:
            if output:
                output += "\n"
            output += stderr.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            return ToolResult(success=False, output=output, error=f"Exit code {proc.returncode}")

        return ToolResult(output=output)

    def _is_blacklisted(self, command: str) -> bool:
        cmd_lower = command.strip().lower()
        for pattern in BLACKLIST:
            if pattern in cmd_lower:
                return True
        return False

    @staticmethod
    def _is_destructive(command: str) -> bool:
        stripped = command.strip()
        for prefix in DESTRUCTIVE_PREFIXES:
            if stripped.startswith(prefix):
                return True
        return False
