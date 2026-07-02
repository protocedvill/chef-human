# Phase 4.1: Agent Utility Tools

**Goal**: Give the agent explicit tools for symbol lookup, undo, and patch-based editing. These tools make the agent more self-sufficient — it can query symbol definitions on demand, revert its own mistakes, and apply precise patches instead of full-file rewrites.

**Prerequisites**: Phases 3.1–3.3 complete (symbol index, RAG, diff-aware editing, 805 tests passing).

---

## Current State

| Component | Status |
|-----------|--------|
| `EditTool` | **Diff-aware** — shows diffs, supports fuzzy matching. Still uses old_string/new_string pattern |
| `WriteTool` | **Diff-aware** — shows diff on overwrite. Writes whole files |
| `DiffStore` (diff.py) | **Stores only diff strings** — no old_content/new_content for undo. API: `record()`, `get_all()`, `get_summary()`, `clear()` |
| `ViewDiffTool` | **Exists** — shows session changes to the agent |
| `SymbolIndex` | **In-memory** — `lookup()`, `lookup_by_file()`, `lookup_by_prefix()`, `search()` available |
| `SymbolRetriever` | **Passive** — only fires during `ContextAssembler.assemble()`, not callable by the agent |
| Tool registry | **11 tools** — read, write, edit, grep, glob, ls, ls_tree, bash, ask_user, finish, view_diff |
| `lookup_symbol` | **Does not exist** — no explicit tool for the agent to query symbols |
| `UndoTool` | **Does not exist** — no way to revert edits |
| `PatchTool` | **Does not exist** — no way to apply unified diffs |

---

## Task List

- [ ] **4.1.1** DiffStore enhancement — store `old_content` and `new_content` alongside diff for undo
- [ ] **4.1.2** `LookupSymbolTool` — new agent tool to query the symbol index by name/prefix
- [ ] **4.1.3** `UndoTool` — revert the last write/edit using stored content
- [ ] **4.1.4** `PatchTool` — apply a unified diff patch to a file
- [ ] **4.1.5** Tool registry wiring — register all new tools, expose SymbolIndex to tool constructors
- [ ] **4.1.6** Tests — all new tools, DiffStore enhancement, integration

---

## Task 4.1.1: DiffStore Enhancement for Undo

**File to modify:** `chef_human/tools/diff.py`

The current `DiffStore.record()` stores `(path, diff, tool_name)` — the diff string is sufficient for display but not for undo. To revert an edit, we need the original file content before the edit.

### Changes

Extend `DiffEntry` to carry content snapshots:

```python
@dataclass
class DiffEntry:
    path: str
    diff: str
    old_content: str | None   # file content before the tool ran
    new_content: str | None   # file content after the tool ran
    timestamp: float
    tool_name: str            # "edit" or "write"
```

Update `DiffStore`:

```python
class DiffStore:
    def record(
        self,
        path: str,
        diff: str,
        tool_name: str,
        old_content: str | None = None,
        new_content: str | None = None,
    ) -> None: ...

    def pop_last(self, path: str | None = None) -> DiffEntry | None:
        """Remove and return the most recent entry, optionally filtered by path."""

    def last(self, path: str | None = None) -> DiffEntry | None:
        """Return the most recent entry without removing it."""
```

The `pop_last()` method enables the UndoTool to consume the most recent diff entry. When `path` is given, it finds the last entry for that specific file.

### Caller Updates

- `EditTool.run()` — pass `old_content` and `new_content` to `diff_store.record()`
- `WriteTool.run()` — pass `old_content` and `new_content` to `diff_store.record()`

### Acceptance Criteria

- `DiffEntry` has `old_content` and `new_content` fields (both `str | None`)
- `DiffStore.record()` accepts the new keyword arguments
- `DiffStore.pop_last()` returns and removes the most recent entry
- `DiffStore.pop_last(path="f.py")` returns the last entry for that path
- `DiffStore.pop_last()` on empty store returns `None`
- `DiffStore.last()` returns without removing
- Existing `ViewDiffTool` still works unchanged (doesn't use content fields)

---

## Task 4.1.2: LookupSymbolTool

**File to create:** `chef_human/tools/lookup_symbol.py`

A new tool that lets the agent query the `SymbolIndex` directly by symbol name, prefix, or substring search.

### Design

```python
class LookupSymbolTool:
    name = "lookup_symbol"
    description = "Look up symbol definitions in the codebase index by name, prefix, or search query."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Exact symbol name to look up (e.g. 'ContextAssembler')",
                "default": None,
            },
            "prefix": {
                "type": "string",
                "description": "Prefix to search for (e.g. 'Context')",
                "default": None,
            },
            "query": {
                "type": "string",
                "description": "Full-text search across names and signatures (case-insensitive substring)",
                "default": None,
            },
            "kind": {
                "type": "string",
                "description": "Filter by symbol kind: function, class, method, struct, enum, etc.",
                "default": None,
            },
            "file": {
                "type": "string",
                "description": "Filter by file path (e.g. 'tools/filesystem.py')",
                "default": None,
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return (default 10, max 50)",
                "default": 10,
            },
        },
    }

    def __init__(self, symbol_index: SymbolIndex) -> None:
        self._index = symbol_index

    async def run(
        self,
        name: str | None = None,
        prefix: str | None = None,
        query: str | None = None,
        kind: str | None = None,
        file: str | None = None,
        max_results: int = 10,
    ) -> ToolResult:
        ...
```

**Query logic (priority order):**

1. If `name` is given → `index.lookup(name, kind)` — exact symbol name match
2. If `prefix` is given → `index.lookup_by_prefix(prefix, max_results)` — prefix scan
3. If `query` is given → `index.search(query)` — full-text substring search
4. If none given → error: "Provide one of: name, prefix, query"

**Result formatting:**

```
Found {n} match(es):
  • `MyClass` (class) — src/models.py:42
    ```python
    class MyClass(BaseModel):
        field: str
    ```
  • `MyClass` (class) — src/utils.py:15
    ```python
    class MyClass(BaseModel):
        other: int
    ```
```

Each result shows:
- Symbol name, kind, file path, line number
- Signature (from `IndexEntry.symbol.signature`) fenced in the detected language
- Results sorted by file path, then line number

**Filtering:**
- `kind` parameter is passed through to `lookup()` (for exact name), applied post-filter for prefix/search
- `file` parameter is applied post-filter as a substring match on `entry.file_path`

**If index is not built:** Return `"Symbol index is not available. The index may be empty or still building."`

### Acceptance Criteria

- `run(name="SymbolName")` returns exact-match definitions
- `run(name="SymbolName", kind="class")` filters by kind
- `run(prefix="Context")` returns prefix-matched symbols
- `run(query="database")` returns substring-matched symbols
- `run(file="models.py")` filters by file path
- `run()` with no query argument returns error
- Returns `"No matches found."` for empty results
- Results are truncated to `max_results` (default 10, max 50)
- Long results list includes `"... and N more"` footer

---

## Task 4.1.3: UndoTool

**File to create:** `chef_human/tools/undo.py`

Reverts the most recent write or edit by restoring the old content from `DiffStore`.

### Design

```python
class UndoTool:
    name = "undo"
    description = "Undo the last write or edit, restoring the file to its previous content."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Optional file path. If set, undo the last change to this specific file.",
                "default": None,
            },
        },
    }

    def __init__(self, workspace: WorkspaceManager, diff_store: DiffStore) -> None:
        self._workspace = workspace
        self._store = diff_store

    async def run(self, path: str | None = None) -> ToolResult:
        entry = self._store.pop_last(path=path)
        if entry is None:
            return ToolResult(success=False, error="Nothing to undo.")

        if entry.old_content is None:
            # File didn't exist before (new file created via write) → delete it
            resolved = self._workspace.resolve(entry.path)
            resolved.unlink(missing_ok=True)
            return ToolResult(output=f"Undid {entry.tool_name}: deleted {entry.path} (was new file)")

        # Restore old content
        resolved = self._workspace.resolve(entry.path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(entry.old_content, encoding="utf-8")

        # Compute reverse diff for confirmation
        from chef_human.tools.diff import compute_diff
        reverse_diff = compute_diff(entry.new_content or "", entry.old_content, path=entry.path)

        output_parts = [
            f"Undid {entry.tool_name}: restored {entry.path}",
        ]
        if reverse_diff:
            output_parts.append(reverse_diff)

        return ToolResult(output="\n".join(output_parts))
```

### Edge Cases

| Scenario | Behaviour |
|----------|-----------|
| Undo on empty store | `"Nothing to undo."` |
| Undo a new-file write | Delete the file |
| Undo an overwrite | Restore original content |
| Undo an edit | Restore content before the edit |
| Undo a deleted file | Restore the file with old_content |
| Multiple undos in sequence | Each pops the most recent entry, working backwards |
| Undo after restart | Store is empty (in-memory) — nothing to undo |

### Acceptance Criteria

- `run()` undoes the most recent edit/write across all files
- `run(path="f.py")` undoes the most recent edit/write to `f.py`
- Success output includes `"Undid {tool_name}: restored {path}"` and a reverse diff
- Undoing a new-file write deletes the file
- "Nothing to undo" when store is empty
- Reverse diff shows what was restored

---

## Task 4.1.4: PatchTool

**File to create:** `chef_human/tools/patch_tool.py`

Applies a unified diff patch to a file using Python's `difflib` patch application logic. No external `patch` command required.

### Design

```python
class PatchTool:
    name = "patch"
    description = "Apply a unified diff patch to a file. The patch must be in standard unified diff format."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to patch (absolute or relative to workspace)",
            },
            "patch": {
                "type": "string",
                "description": "Unified diff patch content (e.g. from ```diff ... ``` blocks)",
            },
            "reverse": {
                "type": "boolean",
                "description": "Apply the patch in reverse (like patch -R)",
                "default": False,
            },
        },
        "required": ["path", "patch"],
    }

    def __init__(self, workspace: WorkspaceManager, diff_store: DiffStore | None = None) -> None:
        self._workspace = workspace
        self._diff_store = diff_store

    async def run(self, path: str, patch: str, reverse: bool = False) -> ToolResult:
        ...
```

### Patch Application Strategy

Python's stdlib doesn't have a unified diff *applier* — only a generator (`difflib.unified_diff`). We need to implement or adapt patch application.

**Approach: parse-and-apply**

1. Parse the unified diff header to find `@@ -start,count +start,count @@` hunks
2. For each hunk:
   a. Extract the old lines (prefixed with ` ` or `-`) and new lines (prefixed with ` ` or `+`)
   b. Find the `old` lines in the current file content at the given offset
   c. Replace them with the `new` lines
3. If `reverse=True`, swap old/new in each hunk

**Simpler implementation**: For the initial version, parse each hunk as a mini `EditTool` invocation:

```python
def _apply_patch(file_content: str, patch_text: str, reverse: bool = False) -> str | None:
    """Apply a unified diff to file_content. Returns new content or None on failure."""
```

Parse `@@ -line,count +line,count @@` markers, then for each hunk, extract the old block (lines starting with `-` or ` `) and new block (lines starting with `+` or ` `). Use `str.replace` to swap them at the given line offset. Return `None` if any hunk fails to match.

This is a best-effort approach — it handles the common case of clean diffs. Malformed or misaligned patches return a clear error (`"Hunk N failed: could not match context lines"`).

**Edge-case handling:**

| Scenario | Behaviour |
|----------|-----------|
| Clean patch, no fuzz | Applies all hunks successfully |
| Patch with surrounding context | Context lines validate the match position |
| Hunk context doesn't match | Return error at first failed hunk |
| Empty patch text | Error: "Patch is empty" |
| Patch applies without changes | Succeed but note "Patch applied, no changes" |
| Reverse patch | Swap old/new in each hunk |

**DiffStore integration:**

After successfully applying the patch, record the change in the diff store (same as `EditTool`):

```python
diff = compute_diff(old_content, new_content, path=path)
if self._diff_store and diff:
    self._diff_store.record(path, diff, "patch",
                            old_content=old_content, new_content=new_content)
```

### Acceptance Criteria

- `run(path="f.py", patch=diff_text)` applies the diff correctly
- `run(..., reverse=True)` reverses the patch
- Output includes summary and forward/reverse diff
- Patch with bad context returns failure with hunk number
- Empty patch text returns error
- Patch to non-existent file returns file-not-found error
- Patch to outside-workspace path returns error
- DiffStore receives the recorded diff

---

## Task 4.1.5: Tool Registry Wiring

**File to modify:** `chef_human/tools/__init__.py` and `chef_human/agent/__init__.py`

### Changes to `create_tool_registry()`

```python
def create_tool_registry(
    workspace: WorkspaceManager,
    symbol_index: SymbolIndex | None = None,
) -> ToolRegistry:
    diff_store = DiffStore()

    registry = ToolRegistry()
    registry.register(ReadTool(workspace))
    registry.register(WriteTool(workspace, diff_store=diff_store))
    registry.register(EditTool(workspace, diff_store=diff_store))
    # ... existing tools ...
    registry.register(LookupSymbolTool(symbol_index=symbol_index))
    registry.register(UndoTool(workspace, diff_store=diff_store))
    registry.register(PatchTool(workspace, diff_store=diff_store))
    return registry
```

### Changes to `create_context_assembler()` / `create_agent()`

The `SymbolIndex` needs to be accessible to `create_tool_registry()`. Currently, `create_tool_registry()` only receives a `WorkspaceManager`. We need to pass the `SymbolIndex` through.

**Option A**: Pass `SymbolIndex` directly to `create_tool_registry()` (preferred — minimal coupling).

**Option B**: Create `LookupSymbolTool` in the agent factory after both index and registry exist.

Option A is cleaner. The agent creation flow becomes:

```python
# chef_human/agent/__init__.py
def create_agent(...) -> ReActLoop:
    workspace = WorkspaceManager(root=workspace_root)
    context_assembler = create_context_assembler(...)
    symbol_index = context_assembler.symbol_index  # already created inside

    tool_registry = create_tool_registry(
        workspace=workspace,
        symbol_index=symbol_index,
    )
    # ...
```

### Acceptance Criteria

- `create_tool_registry()` accepts optional `symbol_index` parameter
- `LookupSymbolTool` is registered and accessible via `registry.get("lookup_symbol")`
- `UndoTool` is registered and accessible
- `PatchTool` is registered and accessible
- All three tools share the same `DiffStore` instance
- All existing integration tests pass

---

## Task 4.1.6: Tests

**New test files:**

| Test file | ~Tests | What it covers |
|-----------|--------|----------------|
| `tests/test_tools/test_lookup_symbol.py` | 15 | Lookup by name, prefix, query, kind filter, file filter, max_results, empty results, not-built index |
| `tests/test_tools/test_undo.py` | 15 | Undo last edit, undo by path, undo new file, multiple undos, empty store, reverse diff |
| `tests/test_tools/test_patch.py` | 15 | Apply clean patch, reverse patch, hunk failure, empty patch, file not found |

**Modified test files:**

| Test file | ~+Tests | What it covers |
|-----------|---------|----------------|
| `tests/test_tools/test_diff.py` | 5 | Enhanced DiffEntry fields, pop_last, last |
| `tests/test_tools/test_filesystem.py` | 2 | EditTool/WriteTool pass old/new content to DiffStore |
| `tests/test_agent_integration.py` | 3 | All new tools registered, share DiffStore |

**Estimated total new tests**: ~55

**Test approach:**
- `LookupSymbolTool`: Build a small `SymbolIndex` with known symbols, run the tool, assert formatted output
- `UndoTool`: Write/edit a file via the normal tools, then call undo, verify file content restored
- `PatchTool`: Generate diffs via `compute_diff`, feed them to `PatchTool.run()`, verify exact byte-level result

---

## Dependencies Map

```
4.1.1 diff.py ───────────────► (enhances DiffEntry/DiffStore)
4.1.2 lookup_symbol.py ──────► SymbolIndex (3.1.3)
4.1.3 undo.py ───────────────► 4.1.1 (enhanced DiffStore), workspace.py
4.1.4 patch_tool.py ─────────► compute_diff (3.3.1), workspace.py
4.1.5 __init__.py ───────────► 4.1.2–4.1.4, agent/__init__.py
4.1.6 tests ─────────────────► all of the above
```

---

## Implementation Order

1. **4.1.1** DiffStore enhancement — must be first so tools can store old/new content
2. **4.1.2** LookupSymbolTool — independent, easy to verify
3. **4.1.3** UndoTool — depends on 4.1.1, uses enhanced DiffStore
4. **4.1.4** PatchTool — independent, uses compute_diff from 3.3.1
5. **4.1.5** Registry wiring — wire everything together
6. **4.1.6** Tests — all new tests + modifications

---

## Design Decisions

### 1. LookupSymbolTool: three query modes, not one

Three separate parameters (`name`, `prefix`, `query`) rather than a single `text` parameter. This lets the LLM choose the right lookup strategy without ambiguity. The parameter descriptions make it clear which to use when.

### 2. UndoTool: destructive, no approval gate

Undo is inherently safe (it restores previous content). No destructive-operation approval is needed. If the model undoes something by mistake, it can redo via write/edit.

### 3. PatchTool: pure Python, no `patch` CLI

Using Python's `difflib` avoids a system dependency and works cross-platform. The hunk-by-hunk approach is simpler than a full patch implementation and covers 95% of real use cases (single-hunk patches with clean context).

### 4. PatchTool: best-effort, not strict patch

Does not implement the full POSIX patch semantics (fuzz factor, timestamps, multi-file patches). The agent is expected to generate one patch per file. Multi-file changes are handled by multiple `patch` tool calls.

### 5. `SymbolIndex` passed to `create_tool_registry()`, not created inside

The index already exists in the `ContextAssembler`. Passing it avoids duplicating the (potentially expensive) build step. The parameter is optional — `LookupSymbolTool` degrades gracefully when index is `None`.

---

## Changes & Deviations Tracking

### 4.1.1 DiffStore Enhancement
| Deviation | Rationale |
|-----------|-----------|

### 4.1.2 LookupSymbolTool
| Deviation | Rationale |
|-----------|-----------|

### 4.1.3 UndoTool
| Deviation | Rationale |
|-----------|-----------|

### 4.1.4 PatchTool
| Deviation | Rationale |
|-----------|-----------|

### 4.1.5 Registry Wiring
| Deviation | Rationale |
|-----------|-----------|

### 4.1.6 Tests
| Deviation | Rationale |
|-----------|-----------|

---

## Future Work (Post-4.1)

- **Batch undo** — undo N steps at once
- **RedoTool** — re-apply the last undone change
- **Multi-file patch** — `PatchTool` accepts patches for multiple files in one call
- **`refactor_symbol` tool** — rename a symbol across all files (Phase 4.3)
- **`explain_symbol` tool** — use LLM to generate a plain-English explanation of a symbol based on its signature and usages
