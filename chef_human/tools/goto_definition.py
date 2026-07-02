from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from chef_human.tools.registry import ToolResult

if TYPE_CHECKING:
    from chef_human.agent.file_context import FileContextManager
    from chef_human.agent.symbols.index import SymbolIndex


_CONTEXT_LINES = 3


class GotoDefinitionTool:
    name = "goto_definition"
    description = "Find where a symbol is defined and load the surrounding source into context."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Symbol name to find the definition of",
            },
            "kind": {
                "type": "string",
                "description": "Optional filter by kind (function, class, etc.)",
                "default": None,
            },
        },
        "required": ["name"],
    }

    def __init__(
        self,
        symbol_index: SymbolIndex,
        file_context: FileContextManager,
    ) -> None:
        self._index = symbol_index
        self._file_context = file_context

    async def run(self, name: str, kind: str | None = None) -> ToolResult:
        entries = self._index.lookup(name, kind=kind)
        if not entries:
            return ToolResult(output=f"No definition found for '{name}'.")

        output_parts: list[str] = []
        seen_files: set[str] = set()

        for entry in entries:
            if entry.file_path in seen_files:
                continue
            seen_files.add(entry.file_path)

            self._file_context.get(entry.file_path)

            file_path = Path(entry.file_path)
            try:
                lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                output_parts.append(
                    f"  {entry.file_path}:{entry.symbol.line} — {entry.symbol.signature}"
                )
                continue

            start = max(0, entry.symbol.line - 1 - _CONTEXT_LINES)
            end = min(len(lines), entry.symbol.line + _CONTEXT_LINES)
            context = "\n".join(
                f"{i + 1:4d} {lines[i]}"
                for i in range(start, end)
            )

            output_parts.append(
                f"  {entry.file_path}:{entry.symbol.line} — {entry.symbol.signature}\n"
                f"```\n{context}\n```"
            )

        title = f"**Definitions of '{name}'"
        if kind:
            title += f" ({kind})"
        title += ":**"
        output_parts.insert(0, title)

        return ToolResult(output="\n\n".join(output_parts))
