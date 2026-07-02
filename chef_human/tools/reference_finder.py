from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from chef_human.tools.registry import ToolResult

if TYPE_CHECKING:
    from chef_human.agent.symbols.index import SymbolIndex
    from chef_human.agent.workspace import WorkspaceManager


class ReferenceFinderTool:
    name = "find_references"
    description = "Find all usages of a symbol across the workspace (definitions + references)."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Symbol name to find references for",
            },
            "include_definitions": {
                "type": "boolean",
                "description": "Include definition sites in results",
                "default": True,
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum files to report (default 20, max 50)",
                "default": 20,
            },
        },
        "required": ["name"],
    }

    def __init__(
        self,
        symbol_index: SymbolIndex,
        workspace: WorkspaceManager,
    ) -> None:
        self._index = symbol_index
        self._workspace = workspace

    async def run(
        self,
        name: str,
        include_definitions: bool = True,
        max_results: int = 20,
    ) -> ToolResult:
        max_results = min(max_results, 50)

        # Tier 1: Index-based definitions
        definition_files: set[str] = set()
        if include_definitions:
            entries = self._index.lookup(name)
            for e in entries:
                definition_files.add(e.file_path)

        # Tier 2: Grep-based textual references
        grep_results: list[tuple[str, int]] = self._grep_references(name)

        # Combine results
        all_defs = sorted(definition_files)
        all_refs: list[tuple[str, int]] = []
        seen_refs: set[str] = set(definition_files)

        for file_path_str, line_num in grep_results:
            if file_path_str not in seen_refs:
                all_refs.append((file_path_str, line_num))
                seen_refs.add(file_path_str)

        # Cap results
        total = len(all_defs) + len(all_refs)
        if total == 0:
            return ToolResult(output=f"No references found for '{name}'.")

        output_parts: list[str] = [
            f"Found {total} reference{'s' if total != 1 else ''} to '{name}':"
        ]

        if include_definitions and all_defs:
            output_parts.append(f"\n  **Definitions ({len(all_defs)}):**")
            for f in all_defs:
                try:
                    rel = self._workspace.resolve(f)
                    r = str(rel.relative_to(self._workspace.root))
                except Exception:
                    r = f
                output_parts.append(f"    {r}")

        if all_refs:
            refs_shown = all_refs[: max_results - len(all_defs)]
            output_parts.append(
                f"\n  **References ({len(all_refs)}):**"
            )
            for file_path_str, line_num in refs_shown:
                try:
                    rel = self._workspace.resolve(file_path_str)
                    r = str(rel.relative_to(self._workspace.root))
                except Exception:
                    r = file_path_str
                output_parts.append(f"    {r}:{line_num}")

            remaining = len(all_refs) - len(refs_shown)
            if remaining > 0:
                output_parts.append(f"    ... and {remaining} more")

        return ToolResult(output="\n".join(output_parts))

    def _grep_references(self, name: str) -> list[tuple[str, int]]:
        pattern = re.compile(r"\b" + re.escape(name) + r"\b")
        matches: list[tuple[str, int]] = []
        for f in self._workspace.list_files(max_depth=10):
            if self._workspace.is_ignored(f):
                continue
            try:
                for i, line in enumerate(
                    f.read_text(encoding="utf-8", errors="replace").splitlines(), 1
                ):
                    if pattern.search(line):
                        matches.append((str(f), i))
                        if len(matches) >= 50:
                            return matches
            except Exception:
                continue
        return matches
