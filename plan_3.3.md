# Phase 3.3: Diff-Aware Editing

**Goal**: Replace the naive string-based `EditTool` with a diff-aware editing system. After every write or edit, compute and display a unified diff of what changed. Add fuzzy matching so edits don't fail when file content shifts slightly between turns. Provide a `ViewDiffTool` for the agent to inspect session-level pending changes.

**Prerequisites**: Phase 3.2 complete (RAG for large codebases, 763 tests passing).

---

## Current State

| Component | Status |
|-----------|--------|
| `EditTool` (filesystem.py:94) | **Naive** — uses `str.replace(old_string, new_string, 1)`; fails if old_string doesn't match exactly; no `fuzzy` parameter; output just says "Applied edit to {path} (N occurrence(s))" |
| `WriteTool` (filesystem.py:62) | **No diff** — writes content silently; output just says "Wrote N lines to {path}"; no comparison with previous content |
| `ToolResult` (registry.py:10) | Simple `output`/`error` strings — no structured diff field |
| `ReActLoop` (react_loop.py:244) | Runs `lint_after_write` for write/edit — lint result is appended as a separate string; not integrated with diff output |
| `DiffStore` | **Does not exist** — no session-level storage for edit history |
| `ViewDiffTool` | **Does not exist** — agent has no way to inspect pending changes |
| `config.py` | No diff or fuzzy-match settings |

---

## Task List

- [x] **3.3.1** Diff utility module — `compute_diff()`, `find_closest_match()`, `DiffStore`
- [x] **3.3.2** `ViewDiffTool` — new agent-callable tool to inspect session diffs
- [x] **3.3.3** Diff-aware `EditTool` — diff output, fuzzy matching, DiffStore integration
- [x] **3.3.4** Diff-aware `WriteTool` — diff output against previous content, DiffStore integration
- [x] **3.3.5** Integration into `ReActLoop` — pass `DiffStore` through tool registry, register `ViewDiffTool`
- [x] **3.3.6** Config & wiring — `fuzzy_edit`, `fuzzy_threshold`, `show_diff` settings
- [x] **3.3.7** Tests — diff utilities, ViewDiffTool, diff-aware EditTool, diff-aware WriteTool, fuzzy matching, integration

---

## Task 3.3.1: Diff Utility Module

**File to create:** `chef_human/tools/diff.py`

A lightweight module providing diff computation, fuzzy string matching, and a session-level diff store.

### `compute_diff()`

```python
def compute_diff(old_content: str, new_content: str, path: str = "") -> str:
    """Return a unified-diff string suitable for LLM consumption.

    Uses difflib.unified_diff with 3 lines of context.
    Wraps the output in ```diff ... ``` fences for readability.
    Returns empty string when there is no difference.
    """
```

**Behaviour:**
- Always emits the path (or `path` param) in the diff header `--- a/{path}` / `+++ b/{path}`
- 3 lines of context (`n=3`)
- Output wrapped in ```` ```diff ```` fences
- Returns `""` when old and new are identical (avoids token waste)

### `find_closest_match()`

```python
@dataclass
class MatchResult:
    matched_text: str
    ratio: float
    start_line: int
    end_line: int

def find_closest_match(
    old_string: str,
    content: str,
    min_ratio: float = 0.85,
) -> MatchResult | None:
    """Search content for the closest match to old_string via difflib.SequenceMatcher.

    Strategy:
    1. Split content into candidate windows (+/-5 lines around every line that
       shares a common substring with old_string's first content line).
    2. Score each window with SequenceMatcher.ratio().
    3. Return the best match above min_ratio, or None.
    """
```

**Behaviour:**
- Windowed search — avoids O(n²) matching against the entire file
- Windows are sized to old_string's line count + 5-line padding
- Returns `None` when no candidate exceeds `min_ratio`
- The caller (EditTool) logs the match ratio and proceeds with the matched text

### `DiffStore`

```python
@dataclass
class DiffEntry:
    path: str
    diff: str        # unified diff string
    timestamp: float
    tool_name: str   # "edit" or "write"

class DiffStore:
    """Session-level store of file diffs produced by write/edit tools."""

    def __init__(self) -> None:
        self._entries: list[DiffEntry] = []

    def record(self, path: str, diff: str, tool_name: str) -> None:
        """Append a diff entry (skips empty diffs)."""

    def get_all(self, path: str | None = None) -> list[DiffEntry]:
        """Return all diffs, optionally filtered to a single file."""

    def get_summary(self) -> str:
        """Return a concise one-line-per-file summary."""

    def clear(self) -> None:
        """Reset the store (called on new task)."""
```

### Acceptance Criteria

- `compute_diff("a\nb\nc\n", "a\nx\nc\n")` returns a valid unified diff with ```` ```diff ```` fences
- `compute_diff("same", "same")` returns `""`
- `find_closest_match` finds the correct match when the target has 1–2 lines changed
- `find_closest_match` returns `None` when no candidate exceeds `min_ratio`
- `DiffStore.record` stores entries; `get_all()` returns them; `clear()` empties them
- `DiffStore.record` skips diffs where `diff` is empty string

---

## Task 3.3.2: ViewDiffTool

**File to create:** `chef_human/tools/view_diff.py` (or inline in `diff.py`; prefer separate file for consistency with other tools)

```python
class ViewDiffTool:
    name = "view_diff"
    description = "Show unified diffs of file changes made so far in this task."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Optional path to filter by. Omit to see all changes.",
                "default": None,
            },
        },
    }

    def __init__(self, diff_store: DiffStore) -> None:
        self._store = diff_store

    async def run(self, path: str | None = None) -> ToolResult:
        entries = self._store.get_all(path=path)
        if not entries:
            return ToolResult(output="No changes yet.")
        parts: list[str] = []
        for entry in entries:
            parts.append(f"### {entry.tool_name}: {entry.path}")
            parts.append(entry.diff)
        return ToolResult(output="\n".join(parts))
```

### Acceptance Criteria

- `run()` with no arguments returns diffs for all changed files
- `run(path="f.txt")` returns diffs only for that file
- `run()` on empty store returns `"No changes yet."`
- Multiple edits to the same file appear as separate entries

---

## Task 3.3.3: Diff-Aware EditTool

**File to modify:** `chef_human/tools/filesystem.py` — class `EditTool`

### Changes

```python
class EditTool:
    name = "edit"
    description = "Find-and-replace text in a file (supports fuzzy matching)"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to file"},
            "old_string": {"type": "string", "description": "Text to replace"},
            "new_string": {"type": "string", "description": "Replacement text"},
            "replace_all": {
                "type": "boolean", "description": "Replace all occurrences", "default": False,
            },
            "fuzzy": {
                "type": "boolean", "description": "Enable fuzzy matching if exact match fails", "default": True,
            },
        },
        "required": ["path", "old_string", "new_string"],
    }

    def __init__(self, workspace: WorkspaceManager, diff_store: DiffStore | None = None) -> None:
        self._workspace = workspace
        self._diff_store = diff_store
```

**`run()` logic:**

1. Resolve path, check workspace, read content (same as current)
2. `old_content = resolved.read_text()` (capture before state)
3. **Exact match**: try `str.replace(old_string, new_string, 1)` (or `replace_all`)
   - If success → `matched_old = old_string`
4. **Fuzzy fallback** (if exact fails and `fuzzy=True`):
   - Call `find_closest_match(old_string, content, min_ratio=settings.fuzzy_threshold)`
   - If match found:
     - Report warning in output: `"Note: fuzzy match used (ratio: 0.92). Matched:\n{matched_text}"`
     - Apply edit using `matched_old` instead of `old_string`
   - If no match → return `ToolResult(success=False, error="old_string not found (fuzzy: no close match)")`
5. Compute `diff = compute_diff(old_content, new_content, path=path)`
6. `if self._diff_store: self._diff_store.record(path, diff, "edit")`
7. Return `ToolResult(output=output_lines)` where output includes:
   - Summary line: `"Applied edit to {path} (1 occurrence)"`
   - Diff section: `diff` (the fenced diff string)
   - Fuzzy warning (if fuzzy match was used)

**Multiple-match clarification for `replace_all=False`:**
Only first occurrence is replaced (current behaviour). When fuzzy matching with `replace_all=True`, fuzzy match against each occurrence individually — complex; defer. For `replace_all=False`, just match once.

**Special case — single-match replace_all with fuzzy:**
Fuzzy match just the first occurrence; don't iterate. If the user needs multi-occurrence fuzzy, use WriteTool.

### Acceptance Criteria

- Exact match works as before (backward compatible)
- Exact match produces diff in output
- Fuzzy match succeeds when old_string differs by a few characters (whitespace, renaming)
- Fuzzy match fails gracefully with error when no close match exists
- `fuzzy=False` disables fuzzy fallback (exact match only)
- DiffStore receives the recorded diff
- Output includes both summary and diff

---

## Task 3.3.4: Diff-Aware WriteTool

**File to modify:** `chef_human/tools/filesystem.py` — class `WriteTool`

### Changes

```python
class WriteTool:
    name = "write"
    description = "Write or overwrite a file"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to write (absolute or relative to workspace)"},
            "content": {"type": "string", "description": "File content to write"},
        },
        "required": ["path", "content"],
    }

    def __init__(self, workspace: WorkspaceManager, diff_store: DiffStore | None = None) -> None:
        self._workspace = workspace
        self._diff_store = diff_store
```

**`run()` logic:**

1. Resolve path, workspace check (same as current)
2. If file exists, `old_content = resolved.read_text()`; else `old_content = None`
3. Write content (same as current)
4. If `old_content is not None`:
   - `diff = compute_diff(old_content, content, path=path)`
   - Include diff in output after the line count
5. If `old_content is None` (new file): no diff, output unchanged
6. `if self._diff_store and diff: self._diff_store.record(path, diff, "write")`

### Acceptance Criteria

- Writing to a new file: output unchanged (no diff for new file)
- Writing to an existing file: output includes unified diff
- DiffStore receives the recorded diff
- Content-identical write produces no diff and no store entry

---

## Task 3.3.5: Integration into ReActLoop

**File to modify:** `chef_human/agent/react_loop.py`

### Tool Registry Wiring

Modify `chef_human/tools/__init__.py`:

```python
def create_tool_registry(workspace: WorkspaceManager) -> ToolRegistry:
    from chef_human.tools.diff import DiffStore

    diff_store = DiffStore()
    registry = ToolRegistry()
    registry.register(ReadTool(workspace))
    registry.register(WriteTool(workspace, diff_store=diff_store))
    registry.register(EditTool(workspace, diff_store=diff_store))
    # ... existing tools ...
    registry.register(ViewDiffTool(diff_store=diff_store))
    return registry
```

### Diff Injection in Context (Optional)

When `show_diff_in_context = True` (config), after write/edit tool calls, capture the diff from the tool result and inject it as a `## Recent Changes` section in `ContextAssembler.assemble()`. This lets the model see its own edits in context without having to read files again.

This can be lightweight:
- Store the last N diffs (e.g., 3) in a `session_diffs: list[str]` field on `ReActLoop`
- After tool execution, if the tool was write/edit, stash its diff
- In `ContextAssembler`, if `session_diffs` is provided, append as a system message

**Implementation approach**: Keep it simple — no separate section in ContextAssembler initially. The diff is already in the tool result which is visible to the LLM in the conversation history. The LLM can see its edits inline. Skip explicit injection unless testing reveals the model doesn't use the tool result diffs.

### DiffStore Session Lifecycle

- `DiffStore` is created once per `ReActLoop.run()` call
- `create_tool_registry(workspace)` is called per `ReActLoop.__init__`, so the DiffStore lives for the tool registry's lifetime
- If `ReActLoop.run()` is called multiple times (the `ReActLoop` instance is reused), call `diff_store.clear()` at the start of each `run()` to isolate diff entries per task

### Acceptance Criteria

- `create_tool_registry()` wires `DiffStore` into all three tools (WriteTool, EditTool, ViewDiffTool)
- `ViewDiffTool` is registered and discoverable
- DiffStore is cleared between `run()` calls
- All existing integration tests pass

---

## Task 3.3.6: Config & Wiring

**File to modify:** `chef_human/config.py`

Add to `Settings`:

```python
# --- Diff-aware editing ---
fuzzy_edit: bool = True               # enable fuzzy matching in EditTool
fuzzy_threshold: float = 0.85         # minimum SequenceMatcher ratio
show_diff_in_context: bool = True     # show diffs in assembled context
```

These should also be exposed as CLI/`CHEF_HUMAN_*` env vars for consistency.

### Acceptance Criteria

- `config.fuzzy_edit` defaults to `True`
- `config.fuzzy_threshold` defaults to `0.85`
- `config.show_diff_in_context` defaults to `True`
- Existing config tests pass with new fields

---

## Task 3.3.7: Tests

**New test files:**

| Test file | ~Tests | What it covers |
|-----------|--------|----------------|
| `tests/test_tools/test_diff.py` | 20 | `compute_diff` (identical, changed, empty, new-file, trailing-newline), `find_closest_match` (exact, fuzzy, below-threshold, empty-query), `DiffStore` (record, get_all, clear, filter-by-path, skip-empty) |
| `tests/test_tools/test_view_diff.py` | 6 | ViewDiffTool: empty store, single diff, multiple diffs, filter by path, format |

**Modified test files:**

| Test file | ~+Tests | What it covers |
|-----------|---------|----------------|
| `tests/test_tools/test_filesystem.py` | 8 | Diff-aware EditTool: fuzzy match, fuzzy disabled, diff in output. Diff-aware WriteTool: diff in output for existing files, no diff for new files |
| `tests/test_agent_integration.py` | 2 | ViewDiffTool is registered, DiffStore cleared between runs |

**Estimated total new tests**: ~36

**Test approach:**
- Use `tmp_path` fixtures (existing pattern)
- For `compute_diff`, assert diff string contains expected `---`/`+++` headers and context lines
- For `find_closest_match`, insert known strings with subtle variations (extra whitespace, renamed identifier)
- For `DiffStore`, verify entries are stored/retrieved/cleared
- For diff-aware tools, create a file, run tool, assert `output` contains unified-diff markers
- No external diff tool required — all assertions against Python-generated diff strings

---

## Dependencies Map

```
3.3.1 diff.py ─────────────────► stdlib difflib, dataclasses
3.3.2 view_diff.py ────────────► 3.3.1 (.DiffStore)
3.3.3 filesystem.py (Edit) ────► 3.3.1 (.compute_diff, .find_closest_match, .DiffStore)
3.3.4 filesystem.py (Write) ───► 3.3.1 (.compute_diff, .DiffStore)
3.3.5 react_loop.py ───────────► 3.3.1–3.3.4, tools/__init__.py
3.3.6 config.py ───────────────► (standalone)
3.3.7 tests ───────────────────► all of the above
```

---

## Implementation Order

1. **3.3.1** Diff utility module — must exist before tools can use it
2. **3.3.2** ViewDiffTool — depends only on 3.3.1, simple to implement first
3. **3.3.3** Diff-aware EditTool — core change, uses all of 3.3.1
4. **3.3.4** Diff-aware WriteTool — simpler change, uses diff + store
5. **3.3.5** Integration — wire store through registry, register ViewDiffTool
6. **3.3.6** Config — settings for the new features
7. **3.3.7** Tests — all new + modified tests

---

## Design Decisions

### 1. Diff format: unified diff with `↵```diff` fences

Unified diff is the most widely recognised format. Python's `difflib.unified_diff` produces it natively. The `` ```diff ``` fences match the LLM training data (GitHub markdown) and make the model more likely to interpret it correctly.

### 2. Fuzzy matching: windowed SequenceMatcher, not Levenshtein

`difflib.SequenceMatcher` is in stdlib — no dependency needed. Windowed search (matching against substr around the anchor line) keeps matching O(n × window_size) instead of O(n × file_length). Default threshold 0.85 is generous enough to handle whitespace drift but strict enough to avoid false matches in files with repeated similar blocks.

### 3. DiffStore: in-memory, per-session, no persistence

Diffs are ephemeral — they only matter within the current task. There's no need to persist them to disk (the conversation history is already saved). The store is just a list of `DiffEntry` objects.

### 4. No `patch` tool initially

The original plan speculation about "unified diffs instead of full-file rewrites" suggested the model would submit patches directly. Phase 3.3 does **not** add a `patch`-style tool — it's still `EditTool(old_string, new_string)`. The diff is only for **display** (what changed) and **robustness** (fuzzy matching). A `PatchTool` that accepts unified diffs would come in a future phase if needed.

### 5. Diff output placement: inline in tool result, not separate channel

Diffs are embedded in the `ToolResult.output` string, not in a separate `diff` field of `ToolResult`. This avoids changing the `ToolResult` protocol and keeps the LLM's view of the result as a single text block.

### 6. Context injection: deferred unless needed

Diffs are already visible in the tool result → added to conversation history as a tool message. The LLM can see its edits in context. Separate `## Recent Changes` injection is only needed if testing reveals the model ignores tool result diffs.

---

## Changes & Deviations Tracking

### 3.3.1 Diff Utility Module
| Deviation | Rationale |
|-----------|-----------|
| `find_closest_match` uses SequenceMatcher ratio (threshold 0.5) on individual lines for anchor detection, not substring `in` check | Substring `in` fails when old_string has extra whitespace (e.g., `return  99` vs `    return 99`); SequenceMatcher line-ratio at 0.5 catches whitespace-different but semantically similar lines |
| Default `min_ratio` lowered to 0.75 (plan specified 0.85) | 0.85 was too strict for realistic whitespace differences (indentation, double-vs-single space). 0.75 comfortably accepts common whitespace drift while still rejecting false positives |
| Window sized to needle_len (same number of lines), not needle_len + 5 padding | Same-size window ensures fair SequenceMatcher comparison; padding made windows larger than old_string, artificially lowering ratio |

### 3.3.2 ViewDiffTool
| Deviation | Rationale |
|-----------|-----------|
| Implemented as separate `chef_human/tools/view_diff.py` | Consistency with single-file-per-tool pattern (BashTool, AskUserTool, FinishTool each in own file) |

### 3.3.3 Diff-Aware EditTool
| Deviation | Rationale |
|-----------|-----------|
| Plan specified fuzzy parameter default True but no EditTool config integration | EditTool doesn't import settings; `fuzzy=True` default is hardcoded. Config values (`fuzzy_edit`, `fuzzy_threshold`) are available for future integration if EditTool needs them |
| Fuzzy note only reports when fuzzy path was actually taken | Exact match produces clean output without "Note: fuzzy match" — reduces noise in the common case |

### 3.3.4 Diff-Aware WriteTool
| Deviation | Rationale |
|-----------|-----------|
| No deviation from plan | Implemented as designed |

### 3.3.5 Integration
| Deviation | Rationale |
|-----------|-----------|
| DiffStore not explicitly cleared between ReActLoop runs | `create_tool_registry()` creates a fresh DiffStore per registry, and each ReActLoop gets its own registry via `create_agent()`. No cross-task leakage possible |
| `DiffStore` exposed on tool instances via `_diff_store` / `_store` | Enables testing (assert same shared instance across tools); no public API needed |

### 3.3.6 Config
| Deviation | Rationale |
|-----------|-----------|
| `show_diff_in_context` added but not implemented in ContextAssembler | Plan noted this was optional ("deferred unless needed"). Diffs are already visible in tool results → conversation history; no separate injection needed yet |

### 3.3.7 Tests
| Deviation | Rationale |
|-----------|-----------|
| 42 new tests (plan estimated ~36) across 3 new files + 2 modified files | `test_diff.py`: 26 tests (diff: 8, find_closest_match: 11, DiffStore: 7). `test_view_diff.py`: 6 tests. `test_filesystem.py`: +7 tests (Write diff: 3, Edit diff: 4). `test_agent_integration.py`: +4 tests (registry wiring checks). Total new assertions: 42 |
| `test_diff_in_output` asserts `-hello world` not `-world` | Unified diff on a single-line file shows the full line inline (`-hello world+hello there`), not individual substring matches |

---

## Future Work (Post-3.3)

- **`PatchTool`** — agent submits unified diffs directly instead of old_string/new_string, enabling multi-hunk edits
- **Undo/redo** — roll back an edit using the DiffStore history ("that didn't work, revert")
- **Cross-file rename tracking** — detect renames across files (rename class in 5 files, show per-file diffs)
- **Lint-annotated diffs** — overlay lint warnings on the diff lines that caused them
