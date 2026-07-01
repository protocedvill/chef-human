# Phase 1.3: Tool Layer

**Goal**: Build the tool layer that the agent invokes to interact with the codebase — reading/writing files, searching, running commands, and communicating with the user. Each tool has a structured interface, safety guardrails, and a registry for dispatch.

**Prerequisites**: Phases 1.1 (LLM backends) and 1.2 (context manager) complete.

---

## Task List

- [x] **1.3.1** Tool protocol & registry (`Tool` protocol, `ToolResult`, `ToolRegistry`)
- [x] **1.3.2** Filesystem tools (`read`, `write`, `edit`, `grep`, `glob`, `ls`, `ls_tree`)
- [x] **1.3.3** Shell tool (`bash` with sandboxing, blacklist, timeout)
- [x] **1.3.4** User interaction tools (`ask_user`, `finish`)
- [x] **1.3.5** Integration tests & tool definitions for LLM consumption

---

## Architecture

```
ToolRegistry
├── register(tool)           # Add tool by name
├── get(name) → Tool         # Retrieve tool implementation
├── list_tools() → list[str] # All registered tool names
└── get_definitions() → list[ToolDefinition]  # LLM-facing JSON schemas

Tools (each implements Tool protocol)
├── ReadTool       filesystem.py
├── WriteTool      filesystem.py
├── EditTool       filesystem.py
├── GrepTool       filesystem.py
├── GlobTool       filesystem.py
├── LsTool         filesystem.py
├── LsTreeTool     filesystem.py
├── BashTool       shell.py
├── AskUserTool    user.py
└── FinishTool     user.py
```

---

## Task 1.3.1: Tool Protocol & Registry

**Files:**
- `chef_human/tools/__init__.py` — package init, `ToolRegistry`, `ToolResult`
- `chef_human/tools/registry.py` — Tool protocol, ToolResult, ToolRegistry

### Tool Protocol

```python
# chef_human/tools/registry.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolResult:
    success: bool = True
    output: str = ""
    error: str | None = None


class Tool(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def parameters(self) -> dict[str, Any]: ...

    async def run(self, **kwargs: Any) -> ToolResult: ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return sorted(self._tools)

    def get_definitions(self) -> list[dict[str, Any]]:
        from chef_human.llm.backend import ToolDefinition
        return [
            ToolDefinition(
                name=t.name,
                description=t.description,
                parameters=t.parameters,
            )
            for t in self._tools.values()
        ]
```

### Notes

- `ToolResult` carries a string `output` for LLM consumption and an optional `error` field for structured error handling
- `parameters` follows JSON Schema format so it maps directly to the LLM's tool definition format
- `ToolRegistry.get_definitions()` returns `ToolDefinition` objects compatible with the existing `CompletionRequest.tools` field

---

## Task 1.3.2: Filesystem Tools

**File:** `chef_human/tools/filesystem.py`

All filesystem tools share access to a `WorkspaceManager` for path validation and safe file operations.

```python
from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chef_human.tools.registry import Tool, ToolResult

if TYPE_CHECKING:
    from chef_human.agent.workspace import WorkspaceManager


class ReadTool:
    name = "read"
    description = "Read file contents with optional line range"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to file (absolute or relative to workspace)"},
            "offset": {"type": "integer", "description": "Starting line number (1-indexed)", "default": 1},
            "limit": {"type": "integer", "description": "Number of lines to read", "default": None},
        },
        "required": ["path"],
    }

    def __init__(self, workspace: WorkspaceManager) -> None:
        self._workspace = workspace

    async def run(self, path: str, offset: int = 1, limit: int | None = None) -> ToolResult:
        resolved = self._workspace.resolve(path)

        if not resolved.exists():
            return ToolResult(success=False, error=f"File not found: {path}")

        if not resolved.is_file():
            return ToolResult(success=False, error=f"Not a file: {path}")

        if not self._workspace.is_within_workspace(resolved):
            return ToolResult(success=False, error=f"Outside workspace: {path}")

        try:
            text = resolved.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return ToolResult(success=False, error=f"Cannot read {path}: {exc}")

        lines = text.splitlines(keepends=True)
        selected = lines[offset - 1 : limit]

        output = "".join(selected)
        if not output.endswith("\n"):
            output += "\n"

        return ToolResult(output=output)


class WriteTool:
    name = "write"
    description = "Write or overwrite a file"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to write (absolute or relative to workspace)"},
            "content": {"type": "string", "description": "File content to write"},
        },
        "required": ["path", "content"],
    }

    def __init__(self, workspace: WorkspaceManager) -> None:
        self._workspace = workspace

    async def run(self, path: str, content: str) -> ToolResult:
        resolved = self._workspace.resolve(path)

        if not self._workspace.is_within_workspace(resolved):
            return ToolResult(success=False, error=f"Outside workspace: {path}")

        resolved.parent.mkdir(parents=True, exist_ok=True)

        try:
            resolved.write_text(content, encoding="utf-8")
        except Exception as exc:
            return ToolResult(success=False, error=f"Cannot write {path}: {exc}")

        lines = content.count("\n") + 1
        return ToolResult(output=f"Wrote {lines} lines to {path}")


class EditTool:
    name = "edit"
    description = "Find-and-replace text in a file (line-based)"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to file"},
            "old_string": {"type": "string", "description": "Text to replace"},
            "new_string": {"type": "string", "description": "Replacement text"},
            "replace_all": {"type": "boolean", "description": "Replace all occurrences", "default": False},
        },
        "required": ["path", "old_string", "new_string"],
    }

    def __init__(self, workspace: WorkspaceManager) -> None:
        self._workspace = workspace

    async def run(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> ToolResult:
        resolved = self._workspace.resolve(path)

        if not resolved.exists():
            return ToolResult(success=False, error=f"File not found: {path}")

        if not self._workspace.is_within_workspace(resolved):
            return ToolResult(success=False, error=f"Outside workspace: {path}")

        try:
            content = resolved.read_text(encoding="utf-8")
        except Exception as exc:
            return ToolResult(success=False, error=f"Cannot read {path}: {exc}")

        if old_string not in content:
            return ToolResult(success=False, error=f"old_string not found in {path}")

        if replace_all:
            new_content = content.replace(old_string, new_string)
        else:
            new_content = content.replace(old_string, new_string, 1)

        try:
            resolved.write_text(new_content, encoding="utf-8")
        except Exception as exc:
            return ToolResult(success=False, error=f"Cannot write {path}: {exc}")

        count = content.count(old_string)
        return ToolResult(output=f"Applied edit to {path} ({count} occurrence{'s' if count != 1 else ''})")


class GrepTool:
    name = "grep"
    description = "Search file contents with regex"
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search"},
            "include": {"type": "string", "description": "Glob pattern for file types (e.g. *.py)", "default": None},
            "path": {"type": "string", "description": "Directory to search (default: workspace root)", "default": None},
        },
        "required": ["pattern"],
    }

    def __init__(self, workspace: WorkspaceManager) -> None:
        self._workspace = workspace

    async def run(self, pattern: str, include: str | None = None, path: str | None = None) -> ToolResult:
        base = self._workspace.resolve(path) if path else self._workspace.root

        if not base.exists() or not base.is_dir():
            return ToolResult(success=False, error=f"Directory not found: {path or base}")

        if not self._workspace.is_within_workspace(base):
            return ToolResult(success=False, error=f"Outside workspace: {path or base}")

        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            return ToolResult(success=False, error=f"Invalid regex: {exc}")

        matches: list[str] = []
        try:
            for entry in base.rglob("*"):
                if not entry.is_file():
                    continue
                if include and not fnmatch.fnmatch(entry.name, include):
                    continue
                if self._workspace.is_ignored(entry):
                    continue

                try:
                    for line_num, line in enumerate(entry.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                        if compiled.search(line):
                            rel = entry.relative_to(self._workspace.root)
                            matches.append(f"{rel}:{line_num}: {line.rstrip()}")
                except Exception:
                    continue
        except PermissionError:
            pass

        if not matches:
            return ToolResult(output="No matches found")

        output = "\n".join(matches[:100])  # cap at 100 results
        if len(matches) > 100:
            output += f"\n... and {len(matches) - 100} more matches"

        return ToolResult(output=output)


class GlobTool:
    name = "glob"
    description = "Find files by glob pattern"
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern (e.g. **/*.py)"},
            "path": {"type": "string", "description": "Directory to search (default: workspace root)", "default": None},
        },
        "required": ["pattern"],
    }

    def __init__(self, workspace: WorkspaceManager) -> None:
        self._workspace = workspace

    async def run(self, pattern: str, path: str | None = None) -> ToolResult:
        base = self._workspace.resolve(path) if path else self._workspace.root

        if not base.exists() or not base.is_dir():
            return ToolResult(success=False, error=f"Directory not found: {path or base}")

        if not self._workspace.is_within_workspace(base):
            return ToolResult(success=False, error=f"Outside workspace: {path or base}")

        results: list[str] = []
        for entry in sorted(base.rglob(pattern)):
            if entry.is_file() and not self._workspace.is_ignored(entry):
                rel = entry.relative_to(self._workspace.root)
                results.append(str(rel))

        if not results:
            return ToolResult(output="No files matched")

        output = "\n".join(results[:200])
        if len(results) > 200:
            output += f"\n... and {len(results) - 200} more files"

        return ToolResult(output=output)


class LsTool:
    name = "ls"
    description = "List directory contents"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path (default: workspace root)", "default": None},
        },
    }

    def __init__(self, workspace: WorkspaceManager) -> None:
        self._workspace = workspace

    async def run(self, path: str | None = None) -> ToolResult:
        base = self._workspace.resolve(path) if path else self._workspace.root

        if not base.exists():
            return ToolResult(success=False, error=f"Path not found: {path or base}")

        if not self._workspace.is_within_workspace(base):
            return ToolResult(success=False, error=f"Outside workspace: {path or base}")

        try:
            entries = sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return ToolResult(success=False, error=f"Permission denied: {path or base}")

        lines: list[str] = []
        for entry in entries:
            if self._workspace.is_ignored(entry):
                continue
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{entry.name}{suffix}")

        if not lines:
            return ToolResult(output="(empty directory)")

        return ToolResult(output="\n".join(lines))


class LsTreeTool:
    name = "ls_tree"
    description = "Show project directory tree"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path (default: workspace root)", "default": None},
        },
    }

    def __init__(self, workspace: WorkspaceManager) -> None:
        self._workspace = workspace

    async def run(self, path: str | None = None) -> ToolResult:
        from chef_human.agent.repo_map import RepoMap
        from chef_human.llm.tokenizer import create_tokenizer

        tokenizer = create_tokenizer()
        repo_map = RepoMap(workspace=self._workspace, tokenizer=tokenizer)
        tree = repo_map.generate_tree(directory=path)
        return ToolResult(output=tree or "(empty directory)")
```

### Notes

- `ReadTool` uses `splitlines(keepends=True)` to preserve exact line content
- `GrepTool` caps results at 100 matches, `GlobTool` at 200 files — prevents token overflow
- `EditTool` mimics the API of the `edit` tool agents use (find-and-replace)
- `LsTreeTool` wraps `RepoMap.generate_tree()` from Phase 1.2.3
- All tools validate workspace boundaries via `WorkspaceManager.is_within_workspace()`

---

## Task 1.3.3: Shell Tool

**File:** `chef_human/tools/shell.py`

The shell tool is the most security-sensitive tool. It must:
1. Jail execution to the workspace directory
2. Blacklist dangerous commands
3. Enforce timeouts
4. Require explicit user approval for destructive operations

```python
from __future__ import annotations

import asyncio
import logging
import os
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from chef_human.tools.registry import Tool, ToolResult

if TYPE_CHECKING:
    from chef_human.agent.workspace import WorkspaceManager

logger = logging.getLogger(__name__)

# Commands that are NEVER allowed
BLACKLIST: set[str] = {
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd",
    "> /dev/",
    ":(){ :|:& };:",  # fork bomb
    "chmod 777 /",
    "chown",
    "halt",
    "poweroff",
    "reboot",
    "shutdown",
}

# Operations that require user approval
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
            return ToolResult(success=False, error=f"Working directory outside workspace: {cwd}")

        # Warn (not block) for destructive operations — actual approval happens at the agent level
        is_destructive = self._is_destructive(command)
        if is_destructive:
            logger.info("Destructive command detected (will require approval): %s", command[:80])

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
```

### Safety Design

| Concern | Mitigation |
|---------|------------|
| Workspace jail | `cwd` validated with `is_within_workspace()`; `resolve()` prevents `../` escapes |
| Command blacklist | Static set of known-dangerous patterns (rm -rf /, mkfs, dd, etc.) |
| Timeout | Configurable (default 30s, max 300s) enforced via `asyncio.wait_for` |
| Destructive ops | Detected via prefix matching (`rm`, `mv`, `>`, etc.); flagged in logs for agent-level approval gate |
| Output limits | Not capped here — the agent's token budget controls context size |

---

## Task 1.3.4: User Interaction Tools

**File:** `chef_human/tools/user.py`

```python
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from chef_human.tools.registry import Tool, ToolResult

logger = logging.getLogger(__name__)


class AskUserTool:
    name = "ask_user"
    description = "Ask the user a question when you need clarification or approval"
    parameters = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "Question to ask the user"},
        },
        "required": ["question"],
    }

    async def run(self, question: str) -> ToolResult:
        logger.info("User asked: %s", question)
        print(f"\n[Agent asks]: {question}")
        print("[Type your response, or 'skip' to continue without answering]: ", end="", flush=True)

        try:
            import sys
            response = sys.stdin.readline().strip()
        except (EOFError, KeyboardInterrupt):
            response = ""

        if not response or response.lower() == "skip":
            return ToolResult(output="User skipped the question")

        return ToolResult(output=response)


class FinishTool:
    name = "finish"
    description = "Signal that the task is complete"
    parameters = {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "Brief summary of what was accomplished", "default": ""},
        },
    }

    async def run(self, summary: str = "") -> ToolResult:
        msg = "Task complete"
        if summary:
            msg += f": {summary}"
        return ToolResult(output=msg)
```

---

## Task 1.3.5: Integration Tests & Tool Definitions

**Files:**
- `tests/test_tools/__init__.py`
- `tests/test_tools/test_filesystem.py` — tests for read, write, edit, grep, glob, ls, ls_tree
- `tests/test_tools/test_shell.py` — tests for bash sandboxing
- `tests/test_tools/test_registry.py` — tests for ToolRegistry
- `tests/test_tools/test_user.py` — tests for ask_user, finish

### Planned Tests

| Test file | Test count | What it covers |
|-----------|-----------|----------------|
| `test_tools/test_registry.py` | ~8 | Registration, lookup, listing, get_definitions JSON schema output |
| `test_tools/test_filesystem.py` | ~32 | read (exists, missing, outside, offset/limit), write (create, overwrite, outside), edit (single, replace_all, not found), grep (match, no match, include, invalid regex, dir not found), glob (match, no match, subdir, dir not found), ls (dir, empty, ignores hidden, outside), ls_tree (tree, empty, subdir) |
| `test_tools/test_shell.py` | ~11 | Basic command, exit code, cwd, timeout, stderr capture, blacklist (3), destructive detection, outside workdir, timeout cap |
| `test_tools/test_user.py` | ~3 | finish output, finish with summary |

### Factory

The `create_context_assembler()` factory in `chef_human/agent/__init__.py` is extended with a tool registry:

```python
# chef_human/tools/__init__.py

from chef_human.tools.registry import ToolRegistry
from chef_human.tools.filesystem import (
    ReadTool, WriteTool, EditTool, GrepTool, GlobTool, LsTool, LsTreeTool,
)
from chef_human.tools.shell import BashTool
from chef_human.tools.user import AskUserTool, FinishTool


def create_tool_registry(workspace: WorkspaceManager) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ReadTool(workspace))
    registry.register(WriteTool(workspace))
    registry.register(EditTool(workspace))
    registry.register(GrepTool(workspace))
    registry.register(GlobTool(workspace))
    registry.register(LsTool(workspace))
    registry.register(LsTreeTool(workspace))
    registry.register(BashTool(workspace))
    registry.register(AskUserTool())
    registry.register(FinishTool())
    return registry


__all__ = [
    "ToolRegistry",
    "ToolResult",
    "create_tool_registry",
]
```

**Acceptance criteria:**
- [x] All 10 tools can be registered and looked up by name
- [x] `get_definitions()` returns valid JSON Schema for each tool
- [x] Each tool handles its error cases gracefully (bad paths, missing files, timeouts)
- [x] Bash sandboxing blocks blacklisted commands
- [x] Edit tool matches the agent's find-and-replace editing pattern
- [x] All 304 tests pass with no external dependencies (same as Phase 1.1/1.2)

---

## Dependencies Map

```
1.3.1 registry.py ────────────► stdlib (dataclasses, typing)
1.3.2 filesystem.py ──────────► 1.2.1 workspace.py, 1.3.1 registry.py
1.3.3 shell.py ───────────────► 1.2.1 workspace.py, 1.3.1 registry.py
1.3.4 user.py ────────────────► 1.3.1 registry.py
1.3.5 tests ──────────────────► 1.3.1–1.3.4, 1.2.1 workspace.py
```

---

## Changes & Deviations Tracking

*Updated during implementation.*

1. **`Tool` as Protocol**: The plan's `Tool` Protocol used `@property` for `name`, `description`, `parameters`. Changed to plain attribute annotations (`name: str`) since class-level attributes satisfy the structural type check. Also removed `@runtime_checkable` since it doesn't work correctly with `async def` methods. The plain `Protocol` works with static type checkers and is simpler.

2. **Blacklist effectiveness**: Implemented as planned with static pattern matching. No changes needed during implementation. The matching uses `in` substring checks on the lowercased command string, which catches most obvious patterns but could miss encoded variants. Noted as a Phase 2.1 hardening task.

3. **`asyncio` tools**: All tools use `async def run()`. Tests use `pytest-asyncio` (already a dev dependency). The `BashTool` uses `asyncio.create_subprocess_shell` with `asyncio.wait_for` for timeout — works correctly in both sync and async test contexts.

4. **User approval for destructive ops**: The plan's `BashTool._is_destructive()` flags commands starting with `rm`, `mv`, `>`, etc. but does not block them — it only logs. The approval gate belongs in Phase 2.1's ReAct loop. Implementation matches the plan.

5. **`ask_user` stdin reading**: Implemented as `sys.stdin.readline()` — blocks in headless mode as expected. The `FinishTool` is implemented as a simple result returner; the ReAct loop (Phase 2.1) will detect it as a termination signal.

6. **Bug fix: `ReadTool.limit` semantics**: The plan's code used `lines[offset - 1 : limit]` which treated `limit` as an absolute line index, not a count. Fixed to `lines[offset - 1 : offset - 1 + limit]` so `limit` correctly means "number of lines to read" as documented.

7. **`ToolRegistry.get_definitions()` return type**: The plan had `list[dict[str, Any]]` as the return type annotation but returned `list[ToolDefinition]` objects. Fixed return type to `list[dict[str, Any]]` by constructing dicts directly, matching the plan's annotation.

8. **Consistent error messages**: Made workspace boundary error messages consistent across all tools — all use `f"Outside workspace: {path}"` (capital "Outside"). The plan's shell tool used lowercase "outside" — unified during implementation.

### Implementation Notes

**56 tool tests pass covering**:
- Tool protocol and ToolResult dataclass (defaults, custom values)
- ToolRegistry (empty, register, get, list sorted, get_definitions, replace)
- ReadTool (content, missing, outside, offset, offset+limit, negative offset)
- WriteTool (create, nested dirs, overwrite, outside, line count)
- EditTool (single replace, replace all, not found, missing file, outside, count)
- GrepTool (matches, no matches, include filter, invalid regex, dir not found)
- GlobTool (pattern match, no matches, subdirectory, dir not found)
- LsTool (list files/ dirs, empty directory, ignores hidden, outside)
- LsTreeTool (tree output, empty directory, subdirectory)
- BashTool (echo, exit code, cwd, timeout, stderr, blacklist x3, destructive detection, outside workdir, timeout cap)
- FinishTool (basic, with summary, empty summary)

---

## Future Improvements (Post-1.3)

- **Command output streaming**: Stream long-running command output to the user in real-time rather than waiting for completion.
- **`edit` with diff input**: Accept unified diffs in addition to find-and-replace for more precise edits.
- **Batch tool execution**: Group multiple tool calls into a single execution step to reduce round-trips.
- **Tool execution caching**: Cache results of idempotent tools (read, glob, grep) within a turn to avoid redundant work.
- **Plugin-based tools**: Allow external tools to be registered via entry points.

---

## File Structure (after Phase 1.3)

```
chef-human/
├── chef_human/
│   ├── agent/
│   │   ├── __init__.py         # + import create_tool_registry
│   │   ├── context.py
│   │   ├── file_context.py
│   │   ├── repo_map.py
│   │   ├── symbols/
│   │   └── workspace.py
│   ├── tools/                  # NEW
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── filesystem.py
│   │   ├── shell.py
│   │   └── user.py
│   ├── llm/
│   ├── config.py
│   └── __init__.py
└── tests/
    ├── test_tools/             # NEW
    │   ├── __init__.py
    │   ├── test_registry.py
    │   ├── test_filesystem.py
    │   ├── test_shell.py
    │   └── test_user.py
    ├── (all existing test files)
    └── (all existing test files)
```
