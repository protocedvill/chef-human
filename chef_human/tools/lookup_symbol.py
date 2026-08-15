from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chef_human.tools.registry import ToolResult

if TYPE_CHECKING:
    from chef_human.agent.symbols.index import SymbolIndex
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
        elif prefix is not None:
            entries = self._symbol_index.lookup_by_prefix(prefix)
        else:
            entries = self._symbol_index.search(query or "")

        if not entries:
            return ToolResult(output="No symbols found.")

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
        return ToolResult(output=output)
