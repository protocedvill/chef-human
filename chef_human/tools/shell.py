from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from chef_human.tools.registry import ToolResult

if TYPE_CHECKING:
    from chef_human.agent.workspace import WorkspaceManager

logger = logging.getLogger(__name__)

# Whole-command / multi-word patterns: specific enough that substring
# matching across the raw command text is safe (they won't false-positive on
# unrelated commands the way a bare word like "dd" would).
_BLACKLIST_PATTERNS: tuple[str, ...] = (
    "rm -rf /",
    "rm -rf /*",
    "> /dev/",
    ":(){ :|:& };:",
    "chmod 777 /",
)

# Bare command names: matched against a segment's leading token only (never
# as a raw substring), so "dd" doesn't also match "add" inside `git add`,
# `npm add`, `mkdir addons`, etc.
_BLACKLIST_COMMANDS: frozenset[str] = frozenset({
    "dd", "mkfs", "chown", "halt", "poweroff", "reboot", "shutdown",
})

# Kept for backwards compatibility / introspection; not used for matching
# directly anymore (see `_BLACKLIST_PATTERNS` / `_BLACKLIST_COMMANDS`).
BLACKLIST: set[str] = set(_BLACKLIST_PATTERNS) | _BLACKLIST_COMMANDS

# Command names that require approval (not outright blocked).
_DESTRUCTIVE_COMMANDS: frozenset[str] = frozenset({
    "rm", "rmdir", "mv", "dd", "format", "mkfs",
})
# Leading-segment redirect punctuation that requires approval (e.g. a bare
# `> file` truncate as its own chained sub-command). Deliberately does NOT
# match an ordinary `some_cmd > file.txt` redirect, since ">" only appears as
# its own shlex token there, not as the segment's leading text.
_DESTRUCTIVE_SEGMENT_PREFIXES: tuple[str, ...] = (">", ">>", "|")

# Kept for backwards compatibility / introspection; not used for matching
# directly anymore (see `_DESTRUCTIVE_COMMANDS` / `_DESTRUCTIVE_SEGMENT_PREFIXES`).
DESTRUCTIVE_PREFIXES: tuple[str, ...] = tuple(_DESTRUCTIVE_COMMANDS) + _DESTRUCTIVE_SEGMENT_PREFIXES

# Wrappers that take an inline script as one of their arguments, e.g.
# `bash -c "rm -rf /"` or `python3 -c "..."` -- the inline script is itself
# recursively scanned as a command.
_INLINE_SCRIPT_WRAPPERS: dict[str, str] = {
    "bash": "-c", "sh": "-c", "zsh": "-c", "dash": "-c", "ksh": "-c",
    "python": "-c", "python3": "-c", "perl": "-e",
}
# Wrappers that just prefix the real command without changing it, e.g.
# `sudo rm -rf /`, `nohup rm -rf /`, `env FOO=bar rm -rf /`.
_PASSTHROUGH_WRAPPERS: frozenset[str] = frozenset({"sudo", "nohup", "env", "exec", "xargs"})
# sudo flags that take a separate value token (e.g. `sudo -u root rm file`) --
# without this, the flag-skip loop below would stop at the value ("root")
# instead of reaching the real command ("rm").
_SUDO_VALUE_FLAGS: frozenset[str] = frozenset({"-u", "-g", "-p", "-C", "-h", "-D"})

_SEGMENT_SPLIT_RE = re.compile(r"&&|\|\||[;|\n]")


def _leading_token(segment: str) -> str:
    try:
        tokens = shlex.split(segment)
    except ValueError:
        tokens = segment.split()
    return tokens[0] if tokens else ""


def _command_segments(command: str, _depth: int = 0) -> list[str]:
    """Best-effort expansion of `command` into every sub-command it could
    run: top-level shell segments (split on ``;``, ``&&``, ``||``, ``|`` and
    newlines), plus, recursively, the inline script argument of common
    wrapper invocations (``bash -c "..."``, ``python3 -c "..."``) and the
    remainder after common passthrough prefixes (``sudo``, ``env FOO=bar``,
    ``nice -n 10``, ...).

    This is a heuristic for the destructive-command guard below, not a real
    shell parser -- it will not catch every possible obfuscation (base64
    encoding, unusual quoting, deeply nested indirection), but it closes the
    common chaining and wrapper bypasses without needing a full sandbox.
    """
    if _depth > 5:
        return []

    segments: list[str] = []
    for raw in _SEGMENT_SPLIT_RE.split(command):
        raw = raw.strip()
        if not raw:
            continue
        segments.append(raw)

        try:
            tokens = shlex.split(raw)
        except ValueError:
            continue
        if not tokens:
            continue

        idx = 0
        while idx < len(tokens) and tokens[idx] in _PASSTHROUGH_WRAPPERS:
            wrapper = tokens[idx]
            idx += 1
            while idx < len(tokens) and (
                tokens[idx].startswith("-") or "=" in tokens[idx]
            ):
                flag = tokens[idx]
                idx += 1
                if wrapper == "sudo" and flag in _SUDO_VALUE_FLAGS and idx < len(tokens):
                    idx += 1
        if idx > 0 and idx < len(tokens):
            segments.extend(_command_segments(" ".join(tokens[idx:]), _depth + 1))
            continue

        flag = _INLINE_SCRIPT_WRAPPERS.get(tokens[0])
        if flag and flag in tokens:
            flag_idx = tokens.index(flag)
            if flag_idx + 1 < len(tokens):
                segments.extend(_command_segments(tokens[flag_idx + 1], _depth + 1))

    return segments


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

    @staticmethod
    def _is_blacklisted(command: str) -> bool:
        cmd_lower = command.strip().lower()
        for pattern in _BLACKLIST_PATTERNS:
            if pattern in cmd_lower:
                return True
        for segment in _command_segments(command):
            if _leading_token(segment).lower() in _BLACKLIST_COMMANDS:
                return True
        return False

    @staticmethod
    def _is_destructive(command: str) -> bool:
        for segment in _command_segments(command) or [command]:
            segment = segment.strip()
            if not segment:
                continue
            if any(segment.startswith(p) for p in _DESTRUCTIVE_SEGMENT_PREFIXES):
                return True
            if _leading_token(segment).lower() in _DESTRUCTIVE_COMMANDS:
                return True
        return False
