from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chef_human.agent.symbols.index import SymbolIndex

logger = logging.getLogger(__name__)

_IMPORT_PATTERNS: dict[str, re.Pattern] = {
    ".py": re.compile(r"(?:from\s+([.\w]+)\s+import|import\s+([.\w]+(?:,\s*[.\w]+)*))"),
    ".js": re.compile(r"from\s+['\"]([^'\"]+)['\"]"),
    ".ts": re.compile(r"from\s+['\"]([^'\"]+)['\"]"),
    ".tsx": re.compile(r"from\s+['\"]([^'\"]+)['\"]"),
    ".rs": re.compile(r"use\s+([\w:]+)"),
    ".go": re.compile(r"import\s+['\"]([^'\"]+)['\"]"),
    ".java": re.compile(r"import\s+([.\w]+);"),
}


class DependencyGraph:
    def __init__(self, symbol_index: SymbolIndex) -> None:
        self._index = symbol_index
        self._deps: dict[Path, set[Path]] = {}
        self._reverse_deps: dict[Path, set[Path]] = {}

    def build(self) -> None:
        self._deps.clear()
        self._reverse_deps.clear()

        indexed_files = list(self._index._by_file.keys())
        ext_map = self._build_ext_map(indexed_files)

        for file_path in indexed_files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            imports = self._extract_imports(file_path, content)
            resolved: set[Path] = set()
            for module_name in imports:
                target = self._resolve(module_name, file_path, ext_map)
                if target is not None:
                    resolved.add(target)
            self._deps[file_path] = resolved
            for target in resolved:
                self._reverse_deps.setdefault(target, set()).add(file_path)

        logger.info(
            "Built dependency graph: %d files, %d edges",
            len(self._deps),
            sum(len(v) for v in self._deps.values()),
        )

    SAVE_VERSION = 1

    def save(
        self,
        path: str | Path = ".chef-human/deps.json",
        workspace_root: Path | None = None,
    ) -> None:
        path_obj = Path(path) if isinstance(path, str) else path
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        root = workspace_root or getattr(self._index._workspace, "root", None)

        deps_data: dict[str, list[str]] = {}
        for src, targets in self._deps.items():
            src_rel = self._rel(src, root)
            deps_data[src_rel] = [self._rel(t, root) for t in sorted(targets)]

        data: dict[str, object] = {
            "version": self.SAVE_VERSION,
            "graph": deps_data,
        }
        path_obj.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info(
            "Saved dependency graph to %s (%d files)", path_obj, len(self._deps)
        )

    def _rel(self, path: Path, root: Path | None = None) -> str:
        r = root or getattr(self._index._workspace, "root", None)
        if r is not None:
            try:
                return str(path.relative_to(r))
            except (ValueError, AttributeError):
                pass
        return str(path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        workspace_root: Path,
        symbol_index: SymbolIndex,
    ) -> DependencyGraph | None:
        path_obj = Path(path) if isinstance(path, str) else path
        if not path_obj.exists():
            return None

        try:
            raw = path_obj.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load deps from %s: %s", path_obj, exc)
            return None

        version = data.get("version", 0)
        if version != cls.SAVE_VERSION:
            logger.warning(
                "Deps version %d != expected %d, ignoring", version, cls.SAVE_VERSION
            )
            return None

        dg = cls(symbol_index=symbol_index)
        graph_data = data.get("graph", {})
        for src_rel, target_rels in graph_data.items():
            src = workspace_root / src_rel
            dg._deps[src] = set()
            for trel in target_rels:
                target = workspace_root / trel
                dg._deps[src].add(target)
                dg._reverse_deps.setdefault(target, set()).add(src)

        logger.info(
            "Loaded dependency graph from %s (%d files)", path_obj, len(dg._deps)
        )
        return dg

    def dependencies(self, file: str | Path) -> list[Path]:
        p = self._resolve_path(file)
        if p is None:
            return []
        return sorted(self._deps.get(p, set()))

    def dependents(self, file: str | Path) -> list[Path]:
        p = self._resolve_path(file)
        if p is None:
            return []
        return sorted(self._reverse_deps.get(p, set()))

    def transitive_dependencies(
        self, file: str | Path, max_depth: int = 2
    ) -> set[Path]:
        p = self._resolve_path(file)
        if p is None:
            return set()

        visited: set[Path] = set()
        current: set[Path] = {p}
        for _depth in range(max_depth):
            children: set[Path] = set()
            for f in current:
                for dep in self._deps.get(f, set()):
                    if dep not in visited:
                        children.add(dep)
            visited.update(children)
            if not children:
                break
            current = children
        visited.discard(p)
        return visited

    def format_for_prompt(self, max_files: int = 20) -> str:
        if not self._deps:
            return ""

        lines: list[str] = ["# Dependency Graph"]
        count = 0
        for file_path in sorted(self._deps.keys()):
            if count >= max_files:
                lines.append(f"... and {len(self._deps) - count} more files")
                break
            deps = self._deps.get(file_path, set())
            rev = self._reverse_deps.get(file_path, set())
            if not deps and not rev:
                continue
            rel = self._rel(file_path)
            lines.append(f"\n## {rel}")
            for d in sorted(deps):
                lines.append(f"  → {self._rel(d)}")
            for r in sorted(rev):
                lines.append(f"  ← {self._rel(r)}")
            count += 1

        return "\n".join(lines)

    def _resolve_path(self, file: str | Path) -> Path | None:
        p = Path(file)
        if not p.is_absolute():
            try:
                root = self._index._workspace.root
                p = root / p
            except AttributeError:
                return p.resolve() if p.exists() else None
        p = p.resolve()
        return p if p in self._deps else None

    @staticmethod
    def _build_ext_map(
        indexed_files: list[Path],
    ) -> dict[str, set[Path]]:
        ext_map: dict[str, set[Path]] = {}
        for f in indexed_files:
            stem = f.stem
            ext_map.setdefault(stem, set()).add(f)
        return ext_map

    @staticmethod
    def _extract_imports(file_path: Path, content: str) -> list[str]:
        ext = file_path.suffix.lower()
        pattern = _IMPORT_PATTERNS.get(ext)
        if pattern is None:
            return []
        imports: list[str] = []
        for m in pattern.finditer(content):
            groups = [g for g in m.groups() if g is not None]
            for g in groups:
                parts = [p.strip() for p in g.split(",") if p.strip()]
                for p in parts:
                    imports.append(p)
        return imports

    def _resolve(
        self, module_name: str, source_file: Path, ext_map: dict[str, set[Path]]
    ) -> Path | None:
        raw = module_name.strip().split("#")[0].strip().strip("'\"")
        if not raw:
            return None

        # Handle file-path-style imports: ./config.js, ../utils.py
        if raw.startswith("."):
            candidate = (source_file.parent / raw).resolve()
            for f in ext_map.get(candidate.stem, set()):
                if f == candidate or f.parent == candidate.parent:
                    return f
            return None

        # Handle dotted module paths: sub.mod → sub/mod.py
        if "." in raw and not raw.endswith((".py", ".js", ".rs", ".go", ".java")):
            path_candidate = raw.replace(".", "/")
            for f in self._index._by_file:
                try:
                    root = self._index._workspace.root
                    rel = f.relative_to(root)
                    if rel.with_suffix("").as_posix() == path_candidate:
                        return f
                except (ValueError, AttributeError):
                    continue

        stem = raw.split(".")[0] if "." in raw else raw
        candidates = ext_map.get(stem, set())
        if not candidates:
            return None

        same_dir = [c for c in candidates if c.parent == source_file.parent]
        if same_dir:
            return same_dir[0]

        return next(iter(candidates))
