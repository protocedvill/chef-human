from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chef_human.agent.symbols.extractor import Symbol, SymbolExtractor
    from chef_human.agent.workspace import WorkspaceManager

logger = logging.getLogger(__name__)


@dataclass
class IndexEntry:
    symbol: Symbol
    file_path: str
    content_hash: str
    access_count: int = 0


def _symbol_to_dict(s: Symbol) -> dict[str, object]:
    return {"name": s.name, "kind": s.kind, "line": s.line, "signature": s.signature}


def _symbol_from_dict(d: dict[str, object]) -> Symbol:
    from chef_human.agent.symbols.extractor import Symbol

    return Symbol(
        name=str(d["name"]),
        kind=str(d["kind"]),
        line=int(str(d["line"])),
        signature=str(d["signature"]),
    )


class SymbolIndex:
    SAVE_VERSION = 1
    def __init__(
        self,
        workspace: WorkspaceManager,
        extractor: SymbolExtractor,
    ) -> None:
        self._workspace = workspace
        self._extractor = extractor
        self._entries: dict[str, list[IndexEntry]] = {}
        self._by_file: dict[Path, list[IndexEntry]] = {}
        self._content_hashes: dict[Path, str] = {}
        self._initial_built: bool = False

    def build(self, files: list[Path] | None = None) -> int:
        if files is None:
            files = self._workspace.list_files(max_depth=10)[:500]

        self._entries.clear()
        self._by_file.clear()
        self._content_hashes.clear()

        count = 0
        for f in files:
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            content_hash = self._hash(content)
            symbols = self._extractor.extract(str(f), content)
            for s in symbols:
                entry = IndexEntry(
                    symbol=s, file_path=str(f), content_hash=content_hash
                )
                self._entries.setdefault(s.name, []).append(entry)
            self._by_file[f] = [
                IndexEntry(symbol=s, file_path=str(f), content_hash=content_hash)
                for s in symbols
            ]
            self._content_hashes[f] = content_hash
            count += len(symbols)

        self._initial_built = True
        logger.info("Indexed %d symbols from %d files", count, len(files))
        return count

    def refresh(self, files: list[Path] | None = None) -> int:
        if not self._initial_built:
            return self.build(files=files)

        if files is None:
            files = self._workspace.list_files(max_depth=10)[:500]

        count = 0
        for f in files:
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            new_hash = self._hash(content)
            if self._content_hashes.get(f) == new_hash:
                continue

            for entry in self._by_file.pop(f, []):
                name_list = self._entries.get(entry.symbol.name, [])
                self._entries[entry.symbol.name] = [
                    e for e in name_list if e.file_path != str(f)
                ]
                if not self._entries[entry.symbol.name]:
                    del self._entries[entry.symbol.name]

            symbols = self._extractor.extract(str(f), content)
            for s in symbols:
                entry = IndexEntry(
                    symbol=s, file_path=str(f), content_hash=new_hash
                )
                self._entries.setdefault(s.name, []).append(entry)
            self._by_file[f] = [
                IndexEntry(symbol=s, file_path=str(f), content_hash=new_hash)
                for s in symbols
            ]
            self._content_hashes[f] = new_hash
            count += len(symbols)

        if count:
            logger.info("Refreshed %d changed symbols", count)
        return count

    def lookup(self, name: str, kind: str | None = None) -> list[IndexEntry]:
        entries = self._entries.get(name, [])
        for e in entries:
            e.access_count += 1
        if kind is not None:
            return sorted(
                [e for e in entries if e.symbol.kind == kind],
                key=lambda e: e.access_count,
                reverse=True,
            )
        return sorted(entries, key=lambda e: e.access_count, reverse=True)

    def lookup_by_file(self, path: Path) -> list[IndexEntry]:
        return self._by_file.get(self._workspace.resolve(path), [])

    def lookup_by_prefix(
        self, prefix: str, max_results: int = 10
    ) -> list[IndexEntry]:
        results: list[IndexEntry] = []
        for name in sorted(self._entries):
            if name.startswith(prefix):
                for entry in self._entries[name]:
                    entry.access_count += 1
                    results.append(entry)
                    if len(results) >= max_results:
                        return results
        return sorted(results, key=lambda e: e.access_count, reverse=True)

    def search(self, query: str) -> list[IndexEntry]:
        query_lower = query.lower()
        results: list[IndexEntry] = []
        seen: set[tuple[str, str, int]] = set()
        for entries in self._entries.values():
            for entry in entries:
                key = (entry.symbol.name, entry.file_path, entry.symbol.line)
                if key in seen:
                    continue
                if (
                    query_lower in entry.symbol.name.lower()
                    or query_lower in entry.symbol.signature.lower()
                ):
                    entry.access_count += 1
                    results.append(entry)
                    seen.add(key)
        return sorted(results[:50], key=lambda e: e.access_count, reverse=True)

    def save(self, path: str | Path = ".chef-human/index.json") -> None:
        path_obj = self._workspace.root / path if isinstance(path, str) else path
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        files = list(self._by_file.keys())
        workspace_hash = self._compute_workspace_hash(files)

        by_file_data: dict[str, list[str]] = {}
        for f, entries in self._by_file.items():
            rel = self._rel(f)
            by_file_data[rel] = [e.symbol.name for e in entries]

        content_hashes_data: dict[str, str] = {}
        for f, h in self._content_hashes.items():
            content_hashes_data[self._rel(f)] = h

        entries_data: dict[str, list[dict[str, object]]] = {}
        for name, entry_list in self._entries.items():
            entries_data[name] = [
                {
                    "symbol": _symbol_to_dict(e.symbol),
                    "file_path": self._rel(Path(e.file_path)),
                    "content_hash": e.content_hash,
                    "access_count": e.access_count,
                }
                for e in entry_list
            ]

        data: dict[str, object] = {
            "version": self.SAVE_VERSION,
            "workspace_hash": workspace_hash,
            "entries": entries_data,
            "by_file": by_file_data,
            "content_hashes": content_hashes_data,
        }
        path_obj.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Saved symbol index to %s (%d symbols)", path_obj, self.total_symbols())

    @classmethod
    def load(
        cls,
        path: str | Path,
        workspace: WorkspaceManager,
        extractor: SymbolExtractor,
    ) -> SymbolIndex | None:
        path_obj = Path(path) if isinstance(path, str) else path
        if not path_obj.exists():
            return None

        try:
            raw = path_obj.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load index from %s: %s", path_obj, exc)
            return None

        version = data.get("version", 0)
        if version != cls.SAVE_VERSION:
            logger.warning(
                "Index version %d != expected %d, ignoring", version, cls.SAVE_VERSION
            )
            return None

        idx = cls(workspace=workspace, extractor=extractor)
        idx._entries = {}
        idx._by_file = {}
        idx._content_hashes = {}
        idx._initial_built = True

        # Reconstruct _entries
        for name, entry_list in data.get("entries", {}).items():
            for ed in entry_list:
                entry = IndexEntry(
                    symbol=_symbol_from_dict(ed["symbol"]),
                    file_path=str(ed["file_path"]),
                    content_hash=str(ed["content_hash"]),
                    access_count=int(ed.get("access_count", 0)),
                )
                idx._entries.setdefault(name, []).append(entry)

        # Reconstruct _by_file and _content_hashes
        for rel, symbol_names in data.get("by_file", {}).items():
            abs_path = workspace.resolve(rel)
            idx._by_file[abs_path] = []
            for sym_name in symbol_names:
                for entry in idx._entries.get(sym_name, []):
                    if entry.file_path == str(abs_path) or cls._paths_equal(
                        entry.file_path, str(abs_path)
                    ):
                        idx._by_file[abs_path].append(entry)
                        break

        for rel, h in data.get("content_hashes", {}).items():
            abs_path = workspace.resolve(rel)
            idx._content_hashes[abs_path] = str(h)

        # Check workspace hash
        files = list(idx._by_file.keys())
        stored_hash = data.get("workspace_hash", "")
        current_hash = cls._compute_workspace_hash(files)
        if stored_hash != current_hash:
            logger.warning(
                "Workspace hash mismatch (stored=%s, current=%s), index may be stale",
                stored_hash,
                current_hash,
            )

        logger.info(
            "Loaded symbol index from %s (%d symbols)", path_obj, idx.total_symbols()
        )
        return idx

    @staticmethod
    def _compute_workspace_hash(files: list[Path]) -> str:
        sorted_names = sorted(str(f) for f in files)
        return hashlib.sha256(
            "\n".join(sorted_names).encode()
        ).hexdigest()[:16]

    def _rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self._workspace.root))
        except (ValueError, AttributeError):
            return str(path)

    @staticmethod
    def _paths_equal(a: str, b: str) -> bool:
        return Path(a).resolve() == Path(b).resolve()

    @property
    def is_built(self) -> bool:
        return self._initial_built

    def total_symbols(self) -> int:
        return sum(len(entries) for entries in self._entries.values())

    def total_files(self) -> int:
        return len(self._by_file)

    @staticmethod
    def _hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
