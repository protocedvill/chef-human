from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chef_human.tools.registry import ToolResult

if TYPE_CHECKING:
    from chef_human.agent.symbols.index import IndexEntry, SymbolIndex
    from chef_human.agent.workspace import WorkspaceManager


class LookupSymbolTool:
    name = "lookup_symbol"
    description = "Look up code symbols (functions, classes, methods) by name, prefix, or search query"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Exact symbol name to look up",
            },
            "prefix": {
                "type": "string",
                "description": "Prefix to match symbol names (returns up to 10 matches)",
            },
            "query": {
                "type": "string",
                "description": "Case-insensitive substring search across symbol names and signatures",
            },
        },
    }

    def __init__(
        self,
        symbol_index: SymbolIndex,
        workspace: WorkspaceManager,
    ) -> None:
        self._symbol_index = symbol_index
        self._workspace = workspace

    async def run(
        self,
        name: str | None = None,
        prefix: str | None = None,
        query: str | None = None,
    ) -> ToolResult:
        modes = sum(1 for x in (name, prefix, query) if x is not None)
        if modes != 1:
            return ToolResult(
                success=False,
                error="Exactly one of 'name', 'prefix', or 'query' must be provided",
            )

        if name is not None:
            entries = self._symbol_index.lookup(name)
            if not entries:
                similar = self._symbol_index.find_similar(name)
                if similar:
                    header = (
                        f"No exact match for '{name}'. Found similarly-named symbols "
                        "that may already do what you need — check these before writing "
                        "new code:"
                    )
                    return ToolResult(output=header + "\n" + self._format(similar))
                return ToolResult(
                    output=(
                        f"No symbol named '{name}' or anything similar exists in this "
                        "codebase. There is nothing to reuse — implement it from scratch."
                    )
                )
        elif prefix is not None:
            entries = self._symbol_index.lookup_by_prefix(prefix)
        else:
            entries = self._symbol_index.search(query or "")

        if not entries:
            return ToolResult(output="No symbols found.")

        return ToolResult(output=self._format(entries))

    def _format(self, entries: list[IndexEntry]) -> str:
        lines: list[str] = []
        for entry in entries[:50]:
            rel = entry.file_path
            try:
                p = self._workspace.resolve(entry.file_path)
                rel = str(p.relative_to(self._workspace.root))
            except Exception:
                pass
            lines.append(
                f"{entry.symbol.kind:12s} {entry.symbol.name:30s} "
                f"{rel}:{entry.symbol.line}  {entry.symbol.signature}"
            )

        output = "\n".join(lines)
        if len(entries) > 50:
            output += f"\n... and {len(entries) - 50} more results"
        return output
