from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chef_human.tools.diff import compute_diff
from chef_human.tools.registry import ToolResult

if TYPE_CHECKING:
    from chef_human.agent.symbols.dependencies import DependencyGraph
    from chef_human.agent.symbols.index import SymbolIndex
    from chef_human.agent.workspace import WorkspaceManager
    from chef_human.tools.diff import DiffStore

logger = logging.getLogger(__name__)

_MAX_RENAME_FILES = 50


class RefactorTool:
    name = "refactor_symbol"
    description = "Rename a symbol across all files that define or reference it."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "old_name": {
                "type": "string",
                "description": "Current symbol name to rename",
            },
            "new_name": {
                "type": "string",
                "description": "New symbol name",
            },
            "scope": {
                "type": "string",
                "description": "Scope of rename: 'definitions' (only defining files), 'all' (all references), 'file' (single file only)",
                "enum": ["definitions", "all", "file"],
                "default": "all",
            },
            "path": {
                "type": "string",
                "description": "Single file to rename in (only used when scope='file')",
                "default": None,
            },
            "dry_run": {
                "type": "boolean",
                "description": "Preview changes without applying them",
                "default": False,
            },
        },
        "required": ["old_name", "new_name"],
    }

    def __init__(
        self,
        workspace: WorkspaceManager,
        symbol_index: SymbolIndex,
        diff_store: DiffStore | None = None,
        dep_graph: DependencyGraph | None = None,
    ) -> None:
        self._workspace = workspace
        self._index = symbol_index
        self._diff_store = diff_store
        self._dep_graph = dep_graph

    async def run(
        self,
        old_name: str,
        new_name: str,
        scope: str = "all",
        path: str | None = None,
        dry_run: bool = False,
    ) -> ToolResult:
        old_name = old_name.strip()
        new_name = new_name.strip()

        if not old_name or not new_name:
            return ToolResult(success=False, error="Both old_name and new_name are required")

        if old_name == new_name:
            return ToolResult(output="No changes made — old name equals new name.")

        # Check if new_name already exists in index
        existing = self._index.lookup(new_name)
        if existing:
            existing_names = set(e.symbol.name for e in existing)
            if new_name in existing_names:
                logger.warning(
                    "Symbol '%s' already exists in index; rename may cause conflicts",
                    new_name,
                )

        # Phase 1: Discover files to change
        files_to_rename: list[Path] = []

        if scope == "file":
            if not path:
                return ToolResult(success=False, error="path is required when scope='file'")
            resolved = self._workspace.resolve(path)
            if not resolved.exists():
                return ToolResult(success=False, error=f"File not found: {path}")
            if not self._workspace.is_within_workspace(resolved):
                return ToolResult(success=False, error=f"Outside workspace: {path}")
            files_to_rename = [resolved]
        else:
            entries = self._index.lookup(old_name)
            if not entries:
                return ToolResult(
                    success=False,
                    error=f"No definitions found for '{old_name}'",
                )

            seen: set[str] = set()
            for entry in entries:
                fp = str(entry.file_path)
                if fp not in seen:
                    seen.add(fp)
                    files_to_rename.append(self._workspace.resolve(fp))

            if scope == "all":
                # Add dependent files via dependency graph
                if self._dep_graph is not None:
                    for entry in entries:
                        try:
                            deps = self._dep_graph.dependents(Path(entry.file_path))
                            for d in deps:
                                ds = str(d)
                                if ds not in seen:
                                    seen.add(ds)
                                    files_to_rename.append(d)
                        except Exception:
                            continue

                # Add textual references via grep
                grep_files = self._find_textual_refs(old_name)
                for f in grep_files:
                    if str(f) not in seen:
                        seen.add(str(f))
                        files_to_rename.append(f)

        if len(files_to_rename) > _MAX_RENAME_FILES:
            return ToolResult(
                success=False,
                error=f"Too many files ({len(files_to_rename)}) — "
                f"use scope='definitions' or scope='file'",
            )

        # Phase 2: Apply rename per file
        pattern = re.compile(r"\b" + re.escape(old_name) + r"\b")
        results: list[dict[str, Any]] = []

        for file_path in files_to_rename:
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception as exc:
                return ToolResult(
                    success=False,
                    error=f"Cannot read {file_path}: {exc}",
                )

            new_content, count = pattern.subn(new_name, content)
            if count == 0:
                continue

            if dry_run:
                diff = compute_diff(content, new_content, path=str(file_path))
                results.append({
                    "path": str(file_path),
                    "count": count,
                    "diff": diff,
                })
            else:
                try:
                    file_path.write_text(new_content, encoding="utf-8")
                except Exception as exc:
                    # Rollback all previous changes
                    self._rollback(results)
                    return ToolResult(
                        success=False,
                        error=f"Cannot write {file_path}: {exc}",
                    )

                diff = compute_diff(content, new_content, path=str(file_path))
                if self._diff_store and diff:
                    self._diff_store.record(
                        str(file_path),
                        diff,
                        "refactor_symbol",
                        old_content=content,
                        new_content=new_content,
                    )
                results.append({
                    "path": str(file_path),
                    "count": count,
                    "diff": diff,
                    "old_content": content,
                    "new_content": new_content,
                })

        if not results:
            return ToolResult(
                output=f"No occurrences of '{old_name}' found in the selected files."
            )

        # Phase 3: Build output
        if dry_run:
            return self._format_dry_run(old_name, new_name, results)
        return self._format_applied(old_name, new_name, results)

    def _find_textual_refs(self, name: str) -> list[Path]:
        pattern = re.compile(r"\b" + re.escape(name) + r"\b")
        matches: list[Path] = []
        for f in self._workspace.list_files(max_depth=10):
            if self._workspace.is_ignored(f):
                continue
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                if pattern.search(content):
                    matches.append(f)
            except Exception:
                continue
        return matches

    def _rollback(self, results: list[dict[str, Any]]) -> None:
        for r in reversed(results):
            if "old_content" in r:
                try:
                    Path(r["path"]).write_text(r["old_content"], encoding="utf-8")
                except Exception:
                    pass

    def _format_dry_run(
        self, old_name: str, new_name: str, results: list[dict[str, Any]]
    ) -> ToolResult:
        parts = [f"**Dry run — {len(results)} file{'s' if len(results) != 1 else ''} would change:**"]
        for r in results:
            try:
                rel = self._workspace.resolve(r["path"])
                rp = str(rel.relative_to(self._workspace.root))
            except Exception:
                rp = r["path"]
            parts.append(f"  {rp}:{r['count']} occurrence{'s' if r['count'] != 1 else ''}")
            if r.get("diff"):
                parts.append(f"    {r['diff'].strip()}")
        return ToolResult(output="\n".join(parts))

    def _format_applied(
        self, old_name: str, new_name: str, results: list[dict[str, Any]]
    ) -> ToolResult:
        parts = [
            f"Renamed '{old_name}' → '{new_name}' across {len(results)} file{'s' if len(results) != 1 else ''}:"
        ]
        for r in results:
            try:
                rel = self._workspace.resolve(r["path"])
                rp = str(rel.relative_to(self._workspace.root))
            except Exception:
                rp = r["path"]
            parts.append(f"  • {rp} — updated {r['count']} occurrence{'s' if r['count'] != 1 else ''}")

        diffs = [r.get("diff", "") for r in results if r.get("diff")]
        if diffs:
            parts.append("")
            parts.extend(diffs)

        return ToolResult(output="\n".join(parts))
