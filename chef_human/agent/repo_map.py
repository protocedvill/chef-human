from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from chef_human.agent.symbols.extractor import (
    Symbol,
    SymbolExtractor,
    create_extractor,
)

if TYPE_CHECKING:
    from chef_human.agent.workspace import WorkspaceManager
    from chef_human.llm.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

MAX_FILES_IN_TREE = 100


class RepoMap:
    def __init__(
        self,
        workspace: WorkspaceManager,
        tokenizer: Tokenizer,
        extractor: SymbolExtractor | None = None,
    ) -> None:
        self._workspace = workspace
        self._tokenizer = tokenizer
        self._extractor = extractor or create_extractor()
        self._symbol_cache: dict[Path, list[Symbol]] = {}

    def generate_tree(self, directory: str | Path | None = None) -> str:
        base = self._workspace.resolve(directory) if directory else self._workspace.root
        files = self._workspace.list_files(directory, max_depth=3)
        files = files[:MAX_FILES_IN_TREE]

        if not files:
            return ""

        lines: list[str] = []
        lines.append(f"Project tree ({len(files)} files shown):")
        lines.append("")
        tree_lines = self._build_tree_lines(base, files)
        lines.extend(tree_lines)

        return "\n".join(lines)

    def _build_tree_lines(self, base: Path, files: list[Path]) -> list[str]:
        lines: list[str] = []
        dir_files: dict[Path, list[str]] = {}
        for f in files:
            try:
                rel = f.relative_to(base)
            except ValueError:
                continue
            parent = rel.parent
            if parent not in dir_files:
                dir_files[parent] = []
            dir_files[parent].append(rel.name)

        all_dirs: set[Path] = set()
        for p in dir_files:
            parent = p
            while parent != Path(".") and parent not in all_dirs:
                all_dirs.add(parent)
                parent = parent.parent

        for fname in sorted(dir_files.pop(Path("."), [])):
            lines.append(f"  {fname}")

        if lines and all_dirs:
            lines.append("")

        for dir_path in sorted(all_dirs, key=lambda p: p.parts):
            depth = len(dir_path.parts) - 1
            indent = "  " * depth
            lines.append(f"{indent}{dir_path.name}/")
            if dir_path in dir_files:
                file_indent = "  " * len(dir_path.parts)
                for fname in sorted(dir_files[dir_path]):
                    lines.append(f"{file_indent}{fname}")

        return lines

    def generate_symbol_map(self, files: list[Path] | None = None) -> str:
        if files is None:
            files = self._workspace.list_files(max_depth=4)[:30]

        sections: list[str] = []
        for f in files:
            content = self._safe_read(f)
            if content is None:
                continue
            symbols = self._extract_symbols(f, content)
            if not symbols:
                continue

            rel_path = f.relative_to(self._workspace.root)
            sections.append(f"### {rel_path}")
            for sym in symbols[:10]:
                sections.append(f"  {sym.line}: {sym.signature}")
            sections.append("")

        return "\n".join(sections)

    def generate(self, max_tokens: int = 2000) -> str:
        tree = self.generate_tree()
        tree_tokens = self._tokenizer.count(tree)
        remaining = max_tokens - tree_tokens

        map_parts = [tree]

        if remaining > 200:
            symbol_map = self.generate_symbol_map()
            symbol_tokens = self._tokenizer.count(symbol_map)
            if symbol_tokens <= remaining:
                if symbol_map.strip():
                    map_parts.append("\n## Symbols\n")
                    map_parts.append(symbol_map)
            else:
                truncated = self._truncate_to_tokens(symbol_map, remaining)
                if truncated.strip():
                    map_parts.append("\n## Symbols\n")
                    map_parts.append(truncated)

        return "\n".join(map_parts)

    def _extract_symbols(self, path: Path, content: str) -> list[Symbol]:
        if path in self._symbol_cache:
            return self._symbol_cache[path]
        symbols = self._extractor.extract(str(path), content)
        self._symbol_cache[path] = symbols
        return symbols

    @staticmethod
    def _safe_read(path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

    @staticmethod
    def _truncate_to_tokens(text: str, max_tokens: int) -> str:
        chars_per_token = 4
        max_chars = max_tokens * chars_per_token
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n... (truncated)"
