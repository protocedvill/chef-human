from __future__ import annotations

import ast
import asyncio
import fnmatch
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chef_human.tools.fsutil import atomic_write_text, count_lines
from chef_human.tools.diff import FileChange, compute_diff, find_closest_match
from chef_human.tools.registry import ToolResult

if TYPE_CHECKING:
    from chef_human.agent.file_context import FileContextManager
    from chef_human.agent.workspace import WorkspaceManager
    from chef_human.tools.diff import DiffStore


class ReadTool:
    name = "read"
    description = "Read file contents with optional line range"
    MAX_OUTPUT_BYTES = 10 * 1024
    MAX_SYMBOL_MAP_BYTES = 3 * 1024
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

    @staticmethod
    def _truncate_to_bytes(text: str, limit_bytes: int) -> str:
        encoded = text.encode("utf-8")
        if len(encoded) <= limit_bytes:
            return text
        truncated = encoded[:limit_bytes]
        return truncated.decode("utf-8", errors="ignore")

    @staticmethod
    def _selected_line_span(offset: int, line_count: int) -> tuple[int, int]:
        if line_count <= 0:
            return offset, offset
        start = max(offset, 1)
        end = start + line_count - 1
        return start, end

    @staticmethod
    def _node_kind(node: ast.AST) -> str:
        if isinstance(node, ast.AsyncFunctionDef):
            return "async def"
        if isinstance(node, ast.FunctionDef):
            return "def"
        return "class"

    @staticmethod
    def _symbol_rank(node: ast.AST, lineno: int, name: str) -> tuple[int, int, int, str]:
        kind_rank = 0 if isinstance(node, ast.ClassDef) else 1
        visibility_rank = 1 if name.startswith("_") else 0
        return (kind_rank, visibility_rank, lineno, name)

    def _python_symbol_map(self, text: str, start_line: int, end_line: int) -> str:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return "Symbol map: unavailable (file could not be parsed as Python).\n"

        symbols: list[tuple[tuple[int, int, int, str], str]] = []
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            lineno = getattr(node, "lineno", None)
            end_lineno = getattr(node, "end_lineno", lineno)
            if lineno is None or end_lineno is None:
                continue
            if end_lineno < start_line or lineno > end_line:
                continue
            kind = self._node_kind(node)
            docstring = ast.get_docstring(node)
            line = f"- {kind} {node.name} @ lines {lineno}-{end_lineno}"
            if docstring:
                summary = docstring.splitlines()[0].strip()
                if summary:
                    line += f": {summary}"
            symbols.append((self._symbol_rank(node, lineno, node.name), line))

        if not symbols:
            return "Symbol map: no Python functions/classes found in the selected range.\n"

        entries = ["Top-level symbol map:"]
        used = len("Top-level symbol map:\n".encode("utf-8"))
        for _, line in sorted(symbols):
            candidate = line + "\n"
            candidate_bytes = len(candidate.encode("utf-8"))
            if used + candidate_bytes > self.MAX_SYMBOL_MAP_BYTES:
                break
            entries.append(line)
            used += candidate_bytes
        return "\n".join(entries) + "\n"

    def _build_truncated_output(
        self,
        path: str,
        output: str,
        offset: int,
        line_count: int,
    ) -> str:
        start_line, end_line = self._selected_line_span(offset, line_count)
        header = (
            f"[read output truncated to {self.MAX_OUTPUT_BYTES} bytes]\n"
            f"Path: {path}\n"
            f"Selected lines: {start_line}-{end_line}\n"
            f"Original selected size: {len(output.encode('utf-8'))} bytes\n"
        )
        symbol_map = self._python_symbol_map(output, start_line, end_line)
        prelude = header + symbol_map + "\nExcerpt:\n"
        remaining = self.MAX_OUTPUT_BYTES - len(prelude.encode("utf-8")) - len("\n".encode("utf-8"))
        if remaining <= 0:
            return self._truncate_to_bytes(prelude, self.MAX_OUTPUT_BYTES)
        excerpt = self._truncate_to_bytes(output, remaining)
        if not excerpt.endswith("\n"):
            excerpt += "\n"
        return prelude + excerpt

    async def run(self, path: str, offset: int = 1, limit: int | None = None) -> ToolResult:
        resolved = self._workspace.resolve(path)

        # Check workspace membership before touching the filesystem at all --
        # exists()/is_file() are themselves stat calls that would otherwise
        # leak a bit of information about paths outside the workspace (e.g.
        # "File not found" vs "Not a file" distinguishes existence) before
        # the boundary check gets a chance to reject the path outright.
        if not self._workspace.is_within_workspace(resolved):
            return ToolResult(success=False, error=f"Outside workspace: {path}")

        if not resolved.exists():
            return ToolResult(success=False, error=f"File not found: {path}")

        if not resolved.is_file():
            return ToolResult(success=False, error=f"Not a file: {path}")

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
        if len(output.encode("utf-8")) > self.MAX_OUTPUT_BYTES:
            output = self._build_truncated_output(path, output, offset, len(selected))

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

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(resolved, content)
        except Exception as exc:
            return ToolResult(success=False, error=f"Cannot write {path}: {exc}")

        if self._file_context is not None:
            self._file_context.remember(path, content)

        lines = count_lines(content)
        output_parts: list[str] = [f"Wrote {lines} lines to {path}"]

        diff = compute_diff(old_content or "", content, path=path)
        if diff and old_content is not None:
            output_parts.append(diff)
        if self._diff_store and diff:
            self._diff_store.record_transaction(
                [FileChange(path, old_content, content)], "write"
            )

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

    def _write_and_remember(self, resolved: Path, path: str, content: str) -> str | None:
        """Write `content` to disk and refresh the file-context cache.

        Shared by every EditTool branch that ends in "write the new content" --
        each branch still records its own diff/transaction afterward (that part
        genuinely differs: create uses a transaction with no old_content, the
        other branches use `record()` with a real diff), but the write +
        mkdir + file_context.remember mechanics were identical three times over.
        Returns an error string on failure, None on success.
        """
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(resolved, content)
        except Exception as exc:
            return f"Cannot write {path}: {exc}"
        if self._file_context is not None:
            self._file_context.remember(path, content)
        return None

    def _record_diff(self, path: str, diff: str, old_content: str, new_content: str) -> None:
        """Record a non-empty diff to the shared DiffStore. Both the
        old_string="" branch and the main replace branch computed and
        conditionally recorded a diff identically; this is the common part
        (each branch still builds its own diff/output text beforehand,
        since that wording genuinely differs)."""
        if diff and self._diff_store:
            self._diff_store.record(path, diff, "edit", old_content=old_content, new_content=new_content)

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
            if old_string != "":
                return ToolResult(success=False, error=f"File not found: {path}")
            # old_string="" against a missing file is an unambiguous
            # "create it" -- treat it the same way WriteTool would rather
            # than erroring and forcing the model (or, worse, a needless
            # ask_user round-trip) to notice and retry with `write` instead.
            if not self._workspace.is_within_workspace(resolved):
                return ToolResult(success=False, error=f"Outside workspace: {path}")
            error = self._write_and_remember(resolved, path, new_string)
            if error is not None:
                return ToolResult(success=False, error=error)
            if self._diff_store:
                self._diff_store.record_transaction(
                    [FileChange(path, None, new_string)], "edit"
                )
            lines = count_lines(new_string)
            return ToolResult(output=f"Created {path} ({lines} lines)")

        if not self._workspace.is_within_workspace(resolved):
            return ToolResult(success=False, error=f"Outside workspace: {path}")

        try:
            old_content = resolved.read_text(encoding="utf-8")
        except Exception as exc:
            return ToolResult(success=False, error=f"Cannot read {path}: {exc}")

        if old_string == "":
            # str.replace("", new_string) doesn't mean "set the whole
            # content" -- Python inserts new_string between *every*
            # character (and at both ends), silently mangling the file
            # instead of erroring. Since old_string="" already means
            # "no anchor, just set the content" for a missing file (see
            # above), treat it the same way here for consistency, rather
            # than falling into that interleaving footgun.
            new_content = new_string
            error = self._write_and_remember(resolved, path, new_content)
            if error is not None:
                return ToolResult(success=False, error=error)
            # Deliberately phrased like WriteTool's own message ("Wrote N
            # lines to {path}"), not "replaced the contents of an existing
            # file" -- that wording was observed causing the step-verifier
            # LLM to read this as evidence *against* a step like "create a
            # new file named X" (it read "replaced" as "the file already
            # existed, so this isn't creation"), even though the file's
            # resulting content was already correct. The model would then
            # alternate between `write` (read as creation, verdict:
            # partial) and this branch (read as NOT creation, verdict:
            # not_complete) every turn, forever, since the two tools'
            # wording described the identical action inconsistently.
            lines = count_lines(new_content)
            output_parts = [f"Wrote {lines} lines to {path}"]
            diff = compute_diff(old_content, new_content, path=path)
            if diff:
                output_parts.append(diff)
                self._record_diff(path, diff, old_content, new_content)
            return ToolResult(output="\n".join(output_parts))

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

        if replace_all:
            count = old_content.count(matched_old)
            new_content = old_content.replace(matched_old, new_string)
        else:
            # Only one occurrence is actually replaced in this branch --
            # report 1, not the total occurrence count in the file (that
            # previously misreported "replaced N occurrences" when only
            # the first one was touched).
            count = 1
            new_content = old_content.replace(matched_old, new_string, 1)

        error = self._write_and_remember(resolved, path, new_content)
        if error is not None:
            return ToolResult(success=False, error=error)

        diff = compute_diff(old_content, new_content, path=path)

        output_parts: list[str] = []
        if diff:
            output_parts.append(
                f"Applied edit to {path} ({count} occurrence{'s' if count != 1 else ''})"
            )
        else:
            # old_string and new_string were identical, so nothing changed --
            # say so explicitly rather than the generic "Applied edit"
            # phrasing, which reads as ambiguous-but-successful to the step
            # verifier. Without this, a step verifier judging a later,
            # redundant "fix X" step (e.g. after a replan re-describes work
            # already done and already verified under a different step's
            # evidence key) sees a no-op edit with no visible diff and
            # concludes the fix was never applied -- even though the
            # "Current file contents" evidence shown alongside it already
            # has the correct code from an earlier turn.
            output_parts.append(
                f"No changes made to {path}: the requested content was already present "
                "(old_string and new_string are identical, or the file already matches "
                "new_string). This is not an error -- it means the file is already in the "
                "desired state."
            )
        if fuzzy_note:
            output_parts.append(fuzzy_note.rstrip())

        if diff:
            output_parts.append(diff)
            self._record_diff(path, diff, old_content, new_content)

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

    # Wall-clock cap on the whole search. re.search on a pathological pattern
    # (e.g. "(a+)+b") can backtrack catastrophically; asyncio.wait_for can't
    # preempt that mid-search since it's synchronous CPU work, so the search
    # runs in a worker thread instead. That thread can't actually be killed
    # if it's genuinely stuck (Python has no safe way to do that) -- this
    # timeout only stops the *event loop* from being blocked, so other
    # concurrently dispatched tool calls can still proceed. MAX_LINE_LENGTH
    # is what actually bounds the worst case to something fast: catastrophic
    # backtracking's cost is roughly exponential in input length, so capping
    # the length capped bounds it to genuinely fast rather than merely
    # finite-in-principle. This is defense-in-depth, not a hardened sandbox
    # (matches BashTool's stated "guardrails, not process isolation").
    SEARCH_TIMEOUT = 10.0
    MAX_LINE_LENGTH = 2000

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

        try:
            matches, permission_denied = await asyncio.wait_for(
                asyncio.to_thread(self._search, base, compiled, include),
                timeout=self.SEARCH_TIMEOUT,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error=f"Search timed out after {self.SEARCH_TIMEOUT}s -- pattern may be "
                "pathologically slow (catastrophic backtracking); try a simpler regex",
            )

        if not matches:
            if permission_denied:
                return ToolResult(
                    success=False, error=f"Permission denied while searching {path or str(base)}"
                )
            return ToolResult(output="No matches found")

        output = "\n".join(matches[:100])
        if len(matches) > 100:
            output += f"\n... and {len(matches) - 100} more matches"
        if permission_denied:
            output += "\n(warning: search stopped early -- permission denied on a subdirectory)"

        return ToolResult(output=output)

    def _search(
        self, base: Path, compiled: re.Pattern[str], include: str | None
    ) -> tuple[list[str], bool]:
        matches: list[str] = []
        permission_denied = False

        def on_error(_exc: OSError) -> None:
            # rglob() silently swallows PermissionError from an inaccessible
            # subdirectory during traversal (verified: it never reaches an
            # except clause around rglob() at all) -- Path.walk()'s on_error
            # callback is the one hook that actually surfaces this, so the
            # gap can be reported instead of the search just going quiet.
            nonlocal permission_denied
            permission_denied = True

        # follow_symlinks=False (the default) means a symlinked directory
        # inside the workspace pointing outside it is never descended into
        # at all, unlike rglob(). A symlinked *file* still appears in
        # filenames though, so is_within_workspace is still checked below.
        for dirpath, _dirnames, filenames in base.walk(on_error=on_error):
            for name in filenames:
                entry = dirpath / name
                if not self._workspace.is_within_workspace(entry):
                    continue
                if include and not fnmatch.fnmatch(name, include):
                    continue
                if self._workspace.is_ignored(entry):
                    continue

                try:
                    for line_num, line in enumerate(entry.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                        if len(line) > self.MAX_LINE_LENGTH:
                            continue
                        if compiled.search(line):
                            rel = entry.relative_to(self._workspace.root)
                            matches.append(f"{rel}:{line_num}: {line.rstrip()}")
                except Exception:
                    continue
        return matches, permission_denied


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

        try:
            # Path.rglob("") raises ValueError as of Python 3.13 (this
            # project supports up to 3.13, per pyproject.toml) -- doesn't
            # reproduce on 3.12, but a bad/empty pattern should be a clean
            # tool error either way, not an unhandled crash on whichever
            # interpreter happens to enforce it.
            entries = sorted(base.rglob(pattern))
        except ValueError as exc:
            return ToolResult(success=False, error=f"Invalid glob pattern {pattern!r}: {exc}")

        results: list[str] = []
        for entry in entries:
            if not entry.is_file() or self._workspace.is_ignored(entry):
                continue
            # rglob follows symlinks; a symlink inside the workspace can
            # point outside it, so re-check every matched entry rather than
            # trusting the one check on `base`.
            if not self._workspace.is_within_workspace(entry):
                continue
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

        if not base.is_dir():
            return ToolResult(success=False, error=f"Not a directory: {path or str(base)}")

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
