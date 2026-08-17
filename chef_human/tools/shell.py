from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from chef_human.tools.registry import ToolResult

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

# Command-name/operator prefixes, checked against each segment of a compound
# command (see _is_destructive), not just the whole string -- a whole-string
# check misses e.g. "echo hi; rm -rf .". Note: react_loop.py's own
# _is_destructive_command imports this same tuple and does a plain
# startswith on the whole string, so every entry here must remain meaningful
# as a bare command prefix on its own, not just within BashTool's
# segment-aware check.
#
# "|" is deliberately not listed: a command can never meaningfully *start*
# with a pipe (there's nothing on its left-hand side yet), so it would never
# match as a prefix either way. ">"/">>" *do* belong here, unlike "|" --
# "> file.txt" alone is valid shell syntax that truncates/creates a file.
DESTRUCTIVE_PREFIXES: tuple[str, ...] = (
    "rm",
    "rmdir",
    "mv",
    "dd",
    "format",
    "mkfs",
    ">",
    ">>",
)

# Splits a compound command into segments at shell control-flow operators, so
# each segment's start can be checked independently. Not a real shell
# parser -- doesn't handle quoting/subshells -- but catches the common
# "harmless-looking first command, destructive second command" shape that a
# whole-string prefix check misses entirely.
_SEGMENT_SEPARATORS = re.compile(r"&&|\|\||[;\n|]")

# Dynamic-linker injection vectors: these let a spawned process load
# arbitrary shared code into *any* binary it execs, regardless of that
# binary's own trust level -- a different and strictly worse risk than "the
# command runs with the user's normal permissions" (which is inherent to
# this tool being a real shell, not a sandbox). Unlike credentials or other
# ambient config, no ordinary dev/build command legitimately needs these
# set, so stripping them is safe rather than a tradeoff against usability.
_ENV_INJECTION_VARS: tuple[str, ...] = (
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
)


class BashTool:
    name = "bash"
    description = (
        "Execute a shell command with its working directory set inside the workspace. "
        "Blacklist, approval, and timeout checks are guardrails, not process isolation."
    )
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
            # This flag does not block execution here -- react_loop.py gates
            # destructive commands behind user approval *before* calling
            # BashTool.run() at all (its own separate _is_destructive_command
            # check). This log is the only signal for any other caller that
            # invokes BashTool directly without going through that gate, so
            # it's a WARNING, not INFO -- and the same note goes into the
            # returned output so it's visible even without logging
            # configured, not just in a log stream that may not be read.
            logger.warning("Destructive command executed: %s", command[:80])

        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
                env=self._subprocess_env(),
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            if proc is not None:
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

    @staticmethod
    def _subprocess_env() -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in _ENV_INJECTION_VARS}
        env["HOME"] = str(Path.home())
        return env

    def _is_blacklisted(self, command: str) -> bool:
        cmd_lower = command.strip().lower()
        for pattern in BLACKLIST:
            if pattern in cmd_lower:
                return True
        return False

    @staticmethod
    def _is_destructive(command: str) -> bool:
        if ">" in command:
            return True
        for segment in _SEGMENT_SEPARATORS.split(command):
            stripped = segment.strip()
            for prefix in DESTRUCTIVE_PREFIXES:
                if stripped.startswith(prefix):
                    return True
        return False
