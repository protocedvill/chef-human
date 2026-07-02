# Phase 4.2: Persistence & Performance

**Goal**: Speed up startup by persisting the `SymbolIndex` and `DependencyGraph` to disk. Add a file watcher mode that auto-refreshes the index when files change. Track symbol usage frequency to prioritise important symbols during retrieval.

**Prerequisites**: Phase 4.1 complete (or at least `SymbolIndex` accessible — 4.1 is not strictly required; this phase can run independently as long as Phase 3.1 is done).

---

## Current State

| Component | Status |
|-----------|--------|
| `SymbolIndex` | **In-memory only** — built from scratch on every `create_context_assembler()` call. No `save()`/`load()`. For a 500-file codebase this takes 2–10 seconds |
| `DependencyGraph` | **In-memory only** — rebuilt from `SymbolIndex` every time. Adds another 0.5–2 seconds |
| `RAG VectorStore` | **Already persisted** — `save()`/`load()` writes `rag.index` + `rag.meta.json` to `.chef-human/` |
| Persistence dir | **`.chef-human/`** exists — used by RAG index and project config lookup. No index persistence |
| File watcher | **Does not exist** — no `watchdog` dependency, no auto-refresh code |
| Symbol usage rank | **Does not exist** — all symbols are equal; `SymbolRetriever` returns first match arbitrarily |
| `LookupSymbolTool` | May exist (4.1.2) — will benefit from persisted index |

---

## Task List

- [ ] **4.2.1** `SymbolIndex.serialize()` / `SymbolIndex.deserialize()` — save/load to `.chef-human/index.json`
- [ ] **4.2.2** `DependencyGraph.serialize()` / `DependencyGraph.deserialize()` — save/load alongside index
- [ ] **4.2.3** Startup integration — load persisted state when available, skip build
- [ ] **4.2.4** File watcher mode — `watchdog`-based auto-refresh of index on file changes
- [ ] **4.2.5** Symbol usage rank — track and persist access counts, prioritise frequent symbols
- [ ] **4.2.6** Config — `persist_index`, `watch_files`, `watch_interval` settings
- [ ] **4.2.7** Tests — persistence round-trip, startup load, file watcher, usage rank

---

## Task 4.2.1: SymbolIndex Serialization

**File to modify:** `chef_human/agent/symbols/index.py`

Add `save()` and `load()` classmethods to `SymbolIndex`.

### Data Format

Use JSON for readability and debuggability. The serialization format mirrors the in-memory structure:

```json
{
  "version": 1,
  "workspace_hash": "a1b2c3d4",
  "entries": {
    "SymbolName": [
      {
        "symbol": {"name": "SymbolName", "kind": "class", "line": 42, "signature": "class SymbolName:"},
        "file_path": "src/module.py",
        "content_hash": "abcd1234"
      }
    ]
  },
  "by_file": {
    "src/module.py": ["SymbolName", ...]
  },
  "content_hashes": {
    "src/module.py": "abcd1234"
  }
}
```

### API

```python
class SymbolIndex:
    SAVE_VERSION = 1

    def save(self, path: Path | str = ".chef-human/index.json") -> None:
        """Serialize the index to JSON."""

    @classmethod
    def load(
        cls,
        path: Path | str,
        workspace: WorkspaceManager,
        extractor: SymbolExtractor,
    ) -> SymbolIndex | None:
        """Load from JSON. Returns None if file doesn't exist or is corrupt."""
```

**`workspace_hash`**: SHA256 of the sorted list of indexed files. Used to detect if the workspace has changed significantly (new files added, files removed). If the hash doesn't match, the persisted index is stale and should be rebuilt.

```python
@staticmethod
def _compute_workspace_hash(files: list[Path]) -> str:
    sorted_names = sorted(str(f) for f in files)
    return hashlib.sha256("\n".join(sorted_names).encode()).hexdigest()[:16]
```

**Load behaviour:**
1. Check if the file exists → return `None` if missing
2. Parse JSON → validate `version` field
3. Reconstruct `_entries`, `_by_file`, `_content_hashes` from the JSON data
4. Recompute workspace hash → if mismatch, log warning "Workspace changed, index may be stale"
5. Return the `SymbolIndex` instance (without `build()`)

### Edge Cases

| Scenario | Behaviour |
|----------|-----------|
| No saved index | `load()` returns `None` silently |
| Corrupt JSON | Log warning, return `None` |
| Version mismatch | Log warning, return `None` (trigger rebuild) |
| Workspace changed (hash mismatch) | Load but log warning; caller decides rebuild |
| Index saved with 500 files, workspace now has 600 | Hash mismatch triggers incremental refresh |

### Acceptance Criteria

- `save()` writes a valid JSON file to the specified path
- `load()` returns a fully functional `SymbolIndex` with same query results as pre-save
- `load()` returns `None` when path doesn't exist
- `load()` returns `None` when JSON is corrupt
- Saved file has `version` field matching `SAVE_VERSION`
- Saved file has `workspace_hash` field
- Workspace hash mismatch produces a warning log

---

## Task 4.2.2: DependencyGraph Serialization

**File to modify:** `chef_human/agent/symbols/dependencies.py`

Add `save()` and `load()` mirroring the pattern from 4.2.1.

### Data Format

```json
{
  "version": 1,
  "graph": {
    "src/main.py": ["src/utils.py", "src/config.py"],
    "src/utils.py": ["src/helpers.py"]
  }
}
```

File paths stored as relative to workspace root (for portability). The `_reverse` map is reconstructed on load.

### API

```python
class DependencyGraph:
    SAVE_VERSION = 1

    def save(self, path: Path | str = ".chef-human/deps.json", workspace_root: Path | None = None) -> None:
        """Serialize the dependency graph to JSON with relative paths."""

    @classmethod
    def load(
        cls,
        path: Path | str,
        workspace_root: Path,
        symbol_index: SymbolIndex,
    ) -> DependencyGraph | None:
        """Load from JSON. Returns None if file doesn't exist."""
```

### Acceptance Criteria

- `save()` writes valid JSON with relative paths
- `load()` returns a functional `DependencyGraph`
- `load()` returns `None` when path doesn't exist
- Round-trip (save → load) preserves all edges
- Empty graph serializes correctly

---

## Task 4.2.3: Startup Integration

**File to modify:** `chef_human/agent/__init__.py` (the factory)

Modify `create_context_assembler()` to attempt loading persisted state before building from scratch.

### Logic

```python
def create_context_assembler(workspace_root=None, index_on_init=True):
    # ... existing setup ...

    extractor = CompositeExtractor()
    symbol_index = SymbolIndex(workspace=workspace, extractor=extractor)

    index_loaded = False
    if index_on_init:
        index_path = workspace.root / ".chef-human" / "index.json"
        deps_path = workspace.root / ".chef-human" / "deps.json"

        loaded = SymbolIndex.load(index_path, workspace, extractor)
        if loaded is not None:
            symbol_index = loaded
            index_loaded = True
            logger.info("Loaded symbol index from %s (%d symbols)", index_path, symbol_index.total_symbols())
        else:
            files = workspace.list_files(max_depth=10)[:settings.max_index_files]
            symbol_index.build(files=files)

    if settings.use_rag and len(workspace.list_files(max_depth=10)) > settings.max_index_files:
        # RAG path (unchanged from 3.2)
        ...
    else:
        dep_graph = DependencyGraph(symbol_index)
        if index_loaded:
            deps_loaded = DependencyGraph.load(deps_path, workspace.root, symbol_index)
            if deps_loaded is not None:
                dep_graph = deps_loaded
            elif index_loaded:
                dep_graph.build()

        # ... build dep graph if needed ...
        # On shutdown or after refresh, save:
        # symbol_index.save(index_path)
        # dep_graph.save(deps_path)
```

### Save-on-shutdown

The index is saved:
- After initial `build()` if no persisted index existed
- After each `refresh()` if files changed
- Via an explicit `atexit` handler or `__del__` on `ReActLoop`

**Simpler approach**: Save after build/refresh operations. Don't add an exit handler — the save is cheap (sequential JSON dump of <10 MB).

```python
# In create_context_assembler, after build:
if not index_loaded:
    symbol_index.save(index_path)
    dep_graph.save(deps_path)
```

### Acceptance Criteria

- Startup with a persisted index loads instead of building
- Startup without persisted index builds from scratch (current behaviour)
- Startup with stale index loads and logs a warning
- Index is saved after initial build
- RAG path is unaffected (RAG already persists)

---

## Task 4.2.4: File Watcher Mode

**File to create:** `chef_human/agent/watcher.py`

A background file watcher that triggers incremental index refresh when tracked files change.

### Design

```python
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from chef_human.agent.workspace import WorkspaceManager

logger = logging.getLogger(__name__)


class FileWatcher:
    """Watches workspace files for changes and triggers refresh callbacks."""

    def __init__(
        self,
        workspace: WorkspaceManager,
        on_change: Callable[[list[Path]], None],
        interval: float = 2.0,
    ) -> None:
        self._workspace = workspace
        self._on_change = on_change
        self._interval = interval
        self._running = False
        self._thread: threading.Thread | None = None
        self._snapshot: dict[Path, float] = {}  # path -> last mtime

    def start(self) -> None:
        """Start the watcher in a background thread."""

    def stop(self) -> None:
        """Stop the watcher."""

    def _snapshot_files(self) -> dict[Path, float]:
        """Build a snapshot of file mtimes for tracked files."""

    def _check_loop(self) -> None:
        """Polling loop: compare mtimes, call on_change for changed files."""
```

### Polling Strategy

`watchdog` is **optional** — the initial implementation uses a polling thread that checks `os.path.getmtime()` every `interval` seconds. This is simpler, avoids a new dependency, and works on all platforms.

If `watchdog` is available, the watcher can optionally use it for instant notification:

```python
def _use_watchdog(self) -> bool:
    try:
        import watchdog
        return True
    except ImportError:
        return False
```

For this phase, polling is sufficient. `watchdog` support is deferred to Future Work.

### Refresh Trigger

When `on_change` is called with a list of changed files, the caller (`ReActLoop` or factory) calls `symbol_index.refresh(changed_files)` and `dep_graph.build()`.

### Integration

The `FileWatcher` is created and started in `create_agent()` when `config.watch_files` is `True`.

```python
# In ReActLoop or agent factory:
if self._config.watch_files:
    self._watcher = FileWatcher(
        workspace=workspace,
        on_change=self._on_file_change,
        interval=settings.watch_interval,
    )
    self._watcher.start()

def _on_file_change(self, changed_files: list[Path]) -> None:
    """Refresh index for changed files."""
    self._context.symbol_index.refresh(changed_files)
    self._context.dep_graph.build()
```

### Acceptance Criteria

- `FileWatcher.start()` begins polling in a daemon thread
- `FileWatcher.stop()` stops polling cleanly
- Changing a file triggers `on_change` within `interval * 2` seconds
- `on_change` receives a list of changed file paths
- Watcher skips ignored files (`.git/`, `__pycache__/`, etc.)
- Watcher does not prevent process exit (daemon thread)
- Thread-safe: `on_change` called from watcher thread, index operations are single-threaded

---

## Task 4.2.5: Symbol Usage Rank

**File to modify:** `chef_human/agent/symbols/index.py` and optionally `retriever.py`

Track how often each symbol is looked up. Use this to prioritise frequently accessed symbols in retrieval results.

### Design

Add an `access_count` field to `IndexEntry`:

```python
@dataclass
class IndexEntry:
    symbol: Symbol
    file_path: str
    content_hash: str
    access_count: int = 0      # NEW
```

Update lookup methods to increment access counts:

```python
def lookup(self, name: str, kind: str | None = None) -> list[IndexEntry]:
    entries = self._entries.get(name, [])
    for e in entries:
        e.access_count += 1
    if kind is not None:
        return [e for e in entries if e.symbol.kind == kind]
    return sorted(entries, key=lambda e: e.access_count, reverse=True)
```

Similar increment in `lookup_by_prefix()` and `search()`.

### Persistence

Access counts are saved as part of `serialize()` (already included in the `entries` JSON as an `access_count` field). On `load()`, restored counts are used immediately for ranking.

### Usage in Retrieval

The `SymbolRetriever.retrieve()` method already returns the first entry for a symbol. With ranking, it should return the most-frequently-accessed file first:

```python
def retrieve(self, symbol_name: str) -> str | None:
    simple_name = symbol_name.split(".")[0]
    entries = self._index.lookup(simple_name)
    if not entries:
        return None
    entry = entries[0]  # now sorted by access_count desc
```

### Acceptance Criteria

- `lookup("Foo")` increments access_count for all matching entries
- `lookup("Foo")` returns entries sorted by access_count descending
- `search()` also increments and sorts by access_count
- Access count is persisted in `save()` and restored in `load()`
- Symbols looked up more frequently appear first in retrieval results

---

## Task 4.2.6: Config

**File to modify:** `chef_human/config.py`

Add to `Settings`:

```python
persist_index: bool = True              # save/load SymbolIndex from .chef-human/
watch_files: bool = False               # enable file watcher
watch_interval: float = 2.0             # polling interval in seconds
```

Defaults keep backward compatibility: persistence is on (fast startup), watching is off (no background thread).

### Acceptance Criteria

- New fields have correct defaults
- `persist_index=True` by default
- `watch_files=False` by default
- `watch_interval=2.0` by default

---

## Task 4.2.7: Tests

**New test files:**

| Test file | ~Tests | What it covers |
|-----------|--------|----------------|
| `tests/test_symbols/test_persistence.py` | 20 | Index save/load, deps save/load, round-trip, corrupt file, version mismatch, empty index, workspace hash change |
| `tests/test_watcher.py` | 10 | FileWatcher start/stop, detects changes, ignores ignored files, calls on_change, daemon thread |

**Modified test files:**

| Test file | ~+Tests | What it covers |
|-----------|---------|----------------|
| `tests/test_symbols/test_index.py` | 5 | Access count increments, sort order, persistence of counts |
| `tests/test_context_assembly.py` | 2 | Startup loads from persisted index when available |
| `tests/test_config.py` | 1 | New config fields |

**Estimated total new tests**: ~38

**Test approach:**
- Persistence: Save index, modify in-memory, load from disk, assert same query results
- File watcher: Use `tmp_path`, create watcher, touch a file, wait for polling cycle, assert `on_change` was called
- Use `time.sleep(interval * 1.5)` in watcher tests (keep interval very short, like 0.1s)

---

## Dependencies Map

```
4.2.1 index.py ──────────────► json, hashlib, .chef-human/ path
4.2.2 dependencies.py ───────► json, .chef-human/ path, 4.2.1 (same pattern)
4.2.3 agent/__init__.py ─────► 4.2.1, 4.2.2 (load/save on startup)
4.2.4 watcher.py ────────────► threading, os.path.getmtime, WorkspaceManager
4.2.5 index.py ──────────────► (standalone, adds access_count)
4.2.6 config.py ─────────────► (standalone)
4.2.7 tests ─────────────────► all of the above
```

---

## Implementation Order

1. **4.2.1** SymbolIndex serialization — core persistence
2. **4.2.5** Symbol usage rank — trivial change, can be done alongside 4.2.1
3. **4.2.2** DependencyGraph serialization — follows same pattern
4. **4.2.3** Startup integration — load persisted state
5. **4.2.4** File watcher — independent, can be done first or last
6. **4.2.6** Config — settings for persistence and watching
7. **4.2.7** Tests — all new tests

---

## Design Decisions

### 1. JSON over pickle or msgpack

JSON is human-readable, debuggable, safe (no arbitrary code execution), and standard-library. For a 500-file codebase with ~10K symbols, JSON serialization is ~1–2 MB and takes <100 ms. If size becomes an issue, gzip compression or msgpack can be added later.

### 2. Polling over watchdog for initial implementation

Polling avoids a new dependency (`watchdog`) and works identically on all platforms. The polling interval (default 2s) is fast enough for a coding assistant — sub-second file change notification isn't critical. `watchdog` can be added as an optional performance upgrade.

### 3. `access_count` persisted in the same JSON file

Access counts are small (one integer per entry) and belong with the entry data. This avoids a separate tracking file and simplifies the save/load flow.

### 4. Workspace hash for staleness detection

A simple hash of sorted file paths detects new/deleted files. Content-level staleness is already handled by per-file `content_hash` in the index itself, which is checked during `refresh()`.

---

## Changes & Deviations Tracking

### 4.2.1 SymbolIndex Serialization
| Deviation | Rationale |
|-----------|-----------|

### 4.2.2 DependencyGraph Serialization
| Deviation | Rationale |
|-----------|-----------|

### 4.2.3 Startup Integration
| Deviation | Rationale |
|-----------|-----------|

### 4.2.4 File Watcher
| Deviation | Rationale |
|-----------|-----------|

### 4.2.5 Symbol Usage Rank
| Deviation | Rationale |
|-----------|-----------|

### 4.2.6 Config
| Deviation | Rationale |
|-----------|-----------|

### 4.2.7 Tests
| Deviation | Rationale |
|-----------|-----------|

---

## Future Work (Post-4.2)

- **Watchdog native backend** — use `watchdog.observers.Observer` for instant file notifications
- **Partial index save** — persist only changed entries instead of full rewrite
- **Index compression** — gzip the JSON or switch to msgpack for faster load on very large repos
- **Cross-session usage rank** — access counts persist across sessions, building long-term relevance
- **Git-aware refresh** — only re-index files changed since last commit, not since last poll
