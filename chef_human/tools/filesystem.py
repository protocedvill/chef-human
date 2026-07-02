from __future__ import annotations

import fnmatch
import re
from typing import TYPE_CHECKING, Any

from chef_human.tools.diff import compute_diff, find_closest_match
from chef_human.tools.registry import ToolResult

if TYPE_CHECKING:
    from chef_human.agent.file_context import FileContextManager
    from chef_human.agent.workspace import WorkspaceManager
    from chef_human.tools.diff import DiffStore


class ReadTool:
    name = "read"
    description = "Read file contents with optional line range"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to file (absolute or relative to workspace)"},
            "offset": {"type": "integer", "description": "Starting line number (1-indexed)", "default": 1},
            "limit": {"type": "integer", "description": "Number of lines to read", "default": None},
        },
        "required": ["path"],
    }

    def __init__(
        self,
        workspace: WorkspaceManager,
        file_context: FileContextManager | None = None,
    ) -> None:
        self._workspace = workspace
        self._file_context = file_context

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

        if self._file_context is not None:
            # Keep the whole file prominently visible in the "## File
            # Context" section of future prompts (not just buried as one
            # more tool-result message in conversation history), so a small
            # model doesn't reflexively re-read a file it already has.
            self._file_context.remember(path, text)

        lines = text.splitlines(keepends=True)
        if offset < 1:
            offset = 1
        if limit is not None:
            selected = lines[offset - 1 : offset - 1 + limit]
        else:
            selected = lines[offset - 1 :]

        output = "".join(selected)
        if not output.endswith("\n"):
            output += "\n"

        return ToolResult(output=output)


class WriteTool:
    name = "write"
    description = "Write or overwrite a file"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to write (absolute or relative to workspace)"},
            "content": {"type": "string", "description": "File content to write"},
        },
        "required": ["path", "content"],
    }

    def __init__(
        self,
        workspace: WorkspaceManager,
        diff_store: DiffStore | None = None,
        file_context: FileContextManager | None = None,
    ) -> None:
        self._workspace = workspace
        self._diff_store = diff_store
        self._file_context = file_context

    async def run(self, path: str, content: str) -> ToolResult:
        resolved = self._workspace.resolve(path)

        if not self._workspace.is_within_workspace(resolved):
            return ToolResult(success=False, error=f"Outside workspace: {path}")

        old_content: str | None = None
        if resolved.exists():
            try:
                old_content = resolved.read_text(encoding="utf-8")
            except Exception:
                old_content = None

        resolved.parent.mkdir(parents=True, exist_ok=True)

        try:
            resolved.write_text(content, encoding="utf-8")
        except Exception as exc:
            return ToolResult(success=False, error=f"Cannot write {path}: {exc}")

        if self._file_context is not None:
            self._file_context.remember(path, content)

        lines = content.count("\n") + 1
        output_parts: list[str] = [f"Wrote {lines} lines to {path}"]

        if old_content is not None:
            diff = compute_diff(old_content, content, path=path)
            if diff:
                output_parts.append(diff)
                if self._diff_store:
                    self._diff_store.record(path, diff, "write", old_content=old_content, new_content=content)

        return ToolResult(output="\n".join(output_parts))


class EditTool:
    name = "edit"
    description = "Find-and-replace text in a file (supports fuzzy matching)"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to file"},
            "old_string": {"type": "string", "description": "Text to replace"},
            "new_string": {"type": "string", "description": "Replacement text"},
            "replace_all": {"type": "boolean", "description": "Replace all occurrences", "default": False},
            "fuzzy": {"type": "boolean", "description": "Enable fuzzy matching if exact match fails", "default": True},
        },
        "required": ["path", "old_string", "new_string"],
    }

    def __init__(
        self,
        workspace: WorkspaceManager,
        diff_store: DiffStore | None = None,
        file_context: FileContextManager | None = None,
    ) -> None:
        self._workspace = workspace
        self._diff_store = diff_store
        self._file_context = file_context

    async def run(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        fuzzy: bool = True,
    ) -> ToolResult:
        resolved = self._workspace.resolve(path)

        if not resolved.exists():
            return ToolResult(success=False, error=f"File not found: {path}")

        if not self._workspace.is_within_workspace(resolved):
            return ToolResult(success=False, error=f"Outside workspace: {path}")

        try:
            old_content = resolved.read_text(encoding="utf-8")
        except Exception as exc:
            return ToolResult(success=False, error=f"Cannot read {path}: {exc}")

        matched_old = old_string
        fuzzy_note = ""

        if old_string not in old_content:
            if not fuzzy:
                return ToolResult(success=False, error=f"old_string not found in {path}")

            match = find_closest_match(old_string, old_content)
            if match is None:
                return ToolResult(
                    success=False,
                    error=f"old_string not found in {path} (fuzzy: no close match found)",
                )

            matched_old = match.matched_text
            fuzzy_note = (
                f"Note: fuzzy match used (ratio: {match.ratio:.2f}, "
                f"lines {match.start_line}-{match.end_line}).\n"
            )

        count = old_content.count(matched_old)
        if replace_all:
            new_content = old_content.replace(matched_old, new_string)
        else:
            new_content = old_content.replace(matched_old, new_string, 1)

        try:
            resolved.write_text(new_content, encoding="utf-8")
        except Exception as exc:
            return ToolResult(success=False, error=f"Cannot write {path}: {exc}")

        if self._file_context is not None:
            self._file_context.remember(path, new_content)

        output_parts: list[str] = []
        output_parts.append(f"Applied edit to {path} ({count} occurrence{'s' if count != 1 else ''})")
        if fuzzy_note:
            output_parts.append(fuzzy_note.rstrip())

        diff = compute_diff(old_content, new_content, path=path)
        if diff:
            output_parts.append(diff)
            if self._diff_store:
                self._diff_store.record(path, diff, "edit", old_content=old_content, new_content=new_content)

        return ToolResult(output="\n".join(output_parts))


class GrepTool:
    name = "grep"
    description = "Search file contents with regex"
    parameters: dict[str, Any] = {
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
            return ToolResult(success=False, error=f"Directory not found: {path or str(base)}")

        if not self._workspace.is_within_workspace(base):
            return ToolResult(success=False, error=f"Outside workspace: {path or str(base)}")

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

        output = "\n".join(matches[:100])
        if len(matches) > 100:
            output += f"\n... and {len(matches) - 100} more matches"

        return ToolResult(output=output)


class GlobTool:
    name = "glob"
    description = "Find files by glob pattern"
    parameters: dict[str, Any] = {
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
            return ToolResult(success=False, error=f"Directory not found: {path or str(base)}")

        if not self._workspace.is_within_workspace(base):
            return ToolResult(success=False, error=f"Outside workspace: {path or str(base)}")

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
    parameters: dict[str, Any] = {
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
            return ToolResult(success=False, error=f"Path not found: {path or str(base)}")

        if not self._workspace.is_within_workspace(base):
            return ToolResult(success=False, error=f"Outside workspace: {path or str(base)}")

        try:
            entries = sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return ToolResult(success=False, error=f"Permission denied: {path or str(base)}")

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
    parameters: dict[str, Any] = {
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
