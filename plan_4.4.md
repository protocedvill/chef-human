# Phase 4.4: Agent Autonomy & Self-Healing

**Goal**: Make the agent more autonomous and robust by adding self-correction tools, completing the undo/redo system, fixing critical integration bugs, and cleaning up technical debt from earlier phases.

**Prerequisites**: Phases 4.1–4.3 complete (all utility tools, persistence, advanced code intelligence tools).

---

## Current State

| Component | Status |
|-----------|--------|
| `UndoTool` | Works — undo last change. No redo capability |
| `RefactorTool` | Word-boundary rename, dry-run, rollback. **Missing `DependencyGraph` integration** — only uses grep fallback for cross-file references |
| CLI (`main.py`) | `_execute_task()` calls `create_tool_registry(context.workspace)` — **missing `symbol_index` and `file_context`**. All Phase 4.1/4.3 symbol-aware tools unavailable from CLI |
| Agent creation | `main.py`'s `_execute_task()` duplicates `agent/__init__.py`'s `create_agent()` — two separate code paths that diverge |
| `ContextAssembler.assemble()` | Accepts `tool_definitions` parameter but never uses it |
| Lint | `annotate_diff_with_lint()` works for ruff. No in-diff auto-fix capability |
| Backup files | `*federate-safeBackup*` files checked in — clutter |
| `DependencyGraph` | Built and persisted. Saved in `.chef-human/`. Only used by startup loading, never by any tool |

---

## Task List

- [x] **4.4.1** Fix CLI integration — pass `symbol_index` and `file_context` from `main.py`
- [x] **4.4.2** Consolidate agent creation — deduplicate `main.py` vs `agent/__init__.py`
- [x] **4.4.3** `RedoTool` — complement to `UndoTool`
- [x] **4.4.4** `InlineLintFixTool` — auto-fix lint warnings via `ruff --fix`
- [x] **4.4.5** Wire `DependencyGraph` into `RefactorTool` — smarter dependent-file discovery
- [x] **4.4.6** Cleanup — remove backup files, unused parameter
- [x] **4.4.7** Tests & verification

---

## Task 4.4.1: Fix CLI Integration

**Files to modify:** `chef_human/main.py`

### Problem

```python
# main.py:185 — current broken line
tool_registry = create_tool_registry(context.workspace)
```

This means all tools that depend on `symbol_index` or `file_context` are never registered when running via the `chef-human run` CLI command:
- `LookupSymbolTool`
- `RefactorTool`
- `GotoDefinitionTool`
- `ReferenceFinderTool`

`context.workspace` is available via the `create_context_assembler()` call, `context.symbol_index` and `context.file_context` are also available.

### Fix

```python
tool_registry = create_tool_registry(
    workspace=context.workspace,
    symbol_index=context.symbol_index,
    file_context=context.file_context,
)
```

### Acceptance Criteria

- `chef-human run` makes all Phase 4.1–4.3 tools available
- `registry.get("refactor_symbol")` returns a `RefactorTool` instance (not `None`)
- `registry.get("goto_definition")` returns a `GotoDefinitionTool` instance
- All existing tests still pass

---

## Task 4.4.2: Consolidate Agent Creation

**Files to modify:** `chef_human/main.py`, `chef_human/agent/__init__.py`

### Problem

There are two separate code paths that create the agent:

1. `chef_human/agent/__init__.py:create_agent()` (lines 155–196)
2. `chef_human/main.py:_execute_task()` (lines 162–203)

Both instantiate `ReActLoop`, `Planner`, `ReActConfig`, the UI, the tool registry, and the context assembler. These have drifted apart — `create_agent()` passes `file_context` to `create_tool_registry()` (correct), while `_execute_task()` does not (bug). `_execute_task()` has session resume logic that `create_agent()` lacks.

### Fix

Refactor `_execute_task()` to call `create_agent()` internally:

```python
async def _execute_task(
    task: str,
    debug_tui: bool = True,
    max_steps: int = 25,
    workspace: str | None = None,
    stream: bool = True,
    headless: bool = False,
    resume: str | None = None,
    save_dir: str | None = None,
) -> AgentResult:
    logging.basicConfig(level=logging.WARNING)

    context = create_context_assembler(workspace_root=workspace)

    if resume:
        session_data = load_session_data(resume, save_dir=save_dir or ".")
        if session_data is not None:
            conv_data = session_data.get("conversation")
            if conv_data:
                loaded = ContextManager.from_dict(conv_data)
                context.conversation.messages = loaded.messages

    loop, _ = create_agent(
        workspace_root=workspace,
        max_steps=max_steps,
        debug_tui=debug_tui,
    )
    loop._config = ReActConfig(
        max_steps=max_steps,
        stream=stream,
        save_dir=save_dir,
    )
    # Use the pre-loaded context (with resumed conversation)
    loop._context_assembler = context

    return await loop.run(task)
```

Alternatively, add session resume parameters to `create_agent()`.

### Design Decision

The simplest, least-risk approach: add `resume_session_id` and `save_dir` parameters to `create_agent()`. Then `_execute_task()` becomes a thin wrapper:

```python
async def _execute_task(...) -> AgentResult:
    logging.basicConfig(level=logging.WARNING)
    loop, _ = create_agent(
        workspace_root=workspace,
        max_steps=max_steps,
        debug_tui=debug_tui,
        stream=stream,
        headless=headless,
        resume_session_id=resume,
        save_dir=save_dir,
    )
    return await loop.run(task)
```

### Acceptance Criteria

- `create_agent()` accepts optional `resume_session_id` and `save_dir` parameters
- Session resume works identically before and after the refactor
- `_execute_task()` is reduced to ~10 lines
- All existing tests pass

---

## Task 4.4.3: RedoTool

**File to create:** `chef_human/tools/redo.py`

Complements `UndoTool` — replays changes that were previously undone. Uses `DiffStore`'s existing `pop_last()`/`last()` and the `old_content`/`new_content` fields to reverse an undo.

### Design

```python
class RedoTool:
    name = "redo"
    description = "Reapply the most recently undone change. Reverses the last undo operation."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, workspace: WorkspaceManager, diff_store: DiffStore) -> None:
        ...

    async def run(self) -> ToolResult: ...
```

### How It Works

`RedoTool` relies on a **redo stack** in `DiffStore`. When `UndoTool` pops an entry and reverses it, it pushes the original diff + content to a redo stack:

```
DiffStore._redo_stack: list[DoneDiffEntry]
```

Where `DoneDiffEntry` is a new dataclass (or we reuse the existing diff tracking):

```python
@dataclass
class RedoEntry:
    file_path: str
    old_content: str   # content before undo was applied
    new_content: str   # content after undo (i.e., current state)
```

**Undo flow with redo support:**

1. `UndoTool.run()` pops the last diff from `_entries`
2. Reverses the diff (swaps old ↔ new content)
3. Applies the reverse diff to the file
4. Pushes a `RedoEntry` to `_redo_stack`

**Redo flow:**

1. `RedoTool.run()` pops from `_redo_stack`
2. Applies the forward diff (`new_content` → `old_content` effectively)
3. Pushes the original back to `_entries` (optional, so the change can be undone again)

### Implementation

Enhance `DiffStore`:

```python
@dataclass
class RedoEntry:
    file_path: str
    old_content: str
    new_content: str

class DiffStore:
    def __init__(self) -> None:
        self._entries: list[DoneDiffEntry] = []
        self._redo_stack: list[RedoEntry] = []

    def push_redo(self, entry: RedoEntry) -> None: ...
    def pop_redo(self) -> RedoEntry | None: ...
    def clear_redo(self) -> None: ...
```

Clear `_redo_stack` whenever a new write/edit/refactor happens (redo is only valid until the next modification).

### Output

```
Reapplied change to src/utils.py (1 occurrence).
```

If nothing to redo:
```
Nothing to redo.
```

### Acceptance Criteria

- `redo()` after `undo()` restores the file to its pre-undo state
- `redo()` with no prior undo returns "Nothing to redo"
- New write after undo clears the redo stack (can't redo a change that was superseded)
- Multiple undo → redo cycles work correctly
- `RedoTool` is registered in `create_tool_registry()`

---

## Task 4.4.4: InlineLintFixTool

**File to create:** `chef_human/tools/lint_fix.py`

Gives the agent the ability to auto-fix lint warnings without manually editing each line. Runs `ruff check --fix` (or equivalent for other linters) on specified files.

### Design

```python
class LintFixTool:
    name = "lint_fix"
    description = "Auto-fix lint warnings in the specified file(s) using ruff --fix (or other supported linter)."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Single file or directory to fix. If not provided, fixes all workspace files.",
                "default": None,
            },
            "check_only": {
                "type": "boolean",
                "description": "Only check for lint issues, don't apply fixes. Returns the issue list.",
                "default": False,
            },
            "select": {
                "type": "string",
                "description": "Comma-separated list of rule codes to fix (e.g., 'F401,F841'). Fixes all by default.",
                "default": None,
            },
        },
        "required": [],
    }

    def __init__(
        self,
        workspace: WorkspaceManager,
        diff_store: DiffStore | None = None,
    ) -> None:
        ...
```

### Behaviour

1. Resolve `path` relative to workspace root (default: whole workspace)
2. Detect which linter applies (only `ruff` for Python for now)
3. For `check_only=True`: run `ruff check path` → return formatted issue list
4. For `check_only=False`:
   a. Read file(s) before fix (to compute diff later)
   b. Run `ruff check --fix path` (with `--select` filter if given)
   c. Read file(s) after fix
   d. Compute unified diff for each changed file
   e. Record diffs in `DiffStore` for undo support
   f. Return summary of changes

### Linter Detection

Reuse `_detect_linter()` from `linter.py`. For now, only `ruff` is supported:
- Python → `ruff check --fix {path}`
- Non-Python → skip with a message ("No supported linter for {extension}")

### Safety

- Only fix files within the workspace boundary
- Record diffs in `DiffStore` so `UndoTool` can reverse auto-fixes
- Skip ignored files (gitignore)
- Clear redo stack (since this is a new modification)

### Output (check_only)

```
Lint issues in src/main.py:
  src/main.py:10:5: F401 `os` imported but unused
  src/main.py:23:1: E302 expected 2 blank lines, found 1
```

### Output (with fix)

```
Fixed 2 issues in src/main.py:
  • F401: removed unused import `os`
  • E302: added blank line
```

### Acceptance Criteria

- `lint_fix("src/main.py")` fixes auto-fixable lint issues and records diffs
- `lint_fix("src/main.py", check_only=True)` shows issues without modifying
- `lint_fix(path="src/")` recursively fixes all Python files in a directory
- `lint_fix(select="F401")` only fixes unused imports
- Lint fix on clean file returns "No issues found"
- Non-Python file returns "No supported linter"
- Changes are undoable via `UndoTool`

---

## Task 4.4.5: Wire DependencyGraph into RefactorTool

**Files to modify:** `chef_human/tools/refactor.py`

### Problem

`RefactorTool` currently only uses grep-based `_find_textual_refs()` for the `scope="all"` mode. It misses the planned dependency-graph-based discovery of importing files. A rename of `Circle` in `shapes.py` should also find `from shapes import Circle` in files whose dependency graph points to `shapes.py`.

### Fix

1. Accept optional `dep_graph: DependencyGraph | None` in `RefactorTool.__init__`
2. In `scope == "all"`, use `dep_graph.dependents()` for each definition file to find importing files
3. Only fall back to grep for files not already found by the dependency graph

```python
class RefactorTool:
    def __init__(
        self,
        workspace: WorkspaceManager,
        symbol_index: SymbolIndex,
        diff_store: DiffStore | None = None,
        dep_graph: DependencyGraph | None = None,
    ) -> None:
        self._dep_graph = dep_graph
        ...
```

### Discovery order (scope="all")

1. Definition files from `index.lookup()` — always included
2. Dependent files from `dep_graph.dependents()` — included if dep_graph is available
3. Grep-based textual refs — fallback for files not yet found

### Acceptance Criteria

- `RefactorTool` accepts optional `dep_graph` parameter
- When `dep_graph` is available and `scope="all"`, dependent files are included in the rename
- When `dep_graph` is not available, falls back to grep-only behavior (current)
- Wire `dep_graph` in `create_tool_registry()` (pass from context if available)

### Integration

In `create_tool_registry()`:

```python
if symbol_index is not None:
    registry.register(RefactorTool(
        workspace=workspace,
        symbol_index=symbol_index,
        diff_store=diff_store,
        dep_graph=getattr(symbol_index, '_dep_graph', None),  # or separate parameter
    ))
```

Better: pass `dep_graph` explicitly as a parameter to `create_tool_registry()`.

---

## Task 4.4.6: Cleanup

**Various files.**

### Remove backup files

Delete `chef_human/tools/diff-federate-safeBackup-0001.py` and `chef_human/tools/filesystem-federate-safeBackup-0001.py` — these are accidental backups from a file-sync tool and should not be in the repository.

### Remove unused `tool_definitions` parameter

`ContextAssembler.assemble()` signature:

```python
def assemble(
    self,
    system_prompt: str,
    tool_definitions: str | None = None,  # <-- remove this
) -> list[Message]:
```

The parameter is accepted but never used — tool definitions are already embedded in the system prompt. Remove it to simplify the interface, and update all callers.

### Sort registered tools alphabetically

In `create_tool_registry()`, sort tool registrations by tool name for easier maintenance.

### Fix `annotate_diff_with_lint()` linter-format hardcoding

The current regex `_LINT_LINE_RE` is ruff-specific (`file:line:col: code message`). Add a `linter_name` parameter so the regex can adapt to other linter output formats if needed in the future:

```python
def annotate_diff_with_lint(
    diff: str,
    lint_output: str,
    linter_name: str = "ruff",
) -> str:
```

This is a minimal change that makes the function extensible without rewriting the parsing logic.

### Acceptance Criteria

- Backup files removed from filesystem
- `ContextAssembler.assemble()` no longer has `tool_definitions` parameter
- All callers of `assemble()` updated
- Tool registrations in `create_tool_registry()` are in a consistent order
- `annotate_diff_with_lint()` has `linter_name` parameter (backward compatible)

---

## Task 4.4.7: Tests & Verification

**New and modified tests:**

| Test file | ~Tests | What it covers |
|-----------|--------|----------------|
| `tests/test_tools/test_redo.py` | 10 | Basic redo, undo→redo cycle, redo cleared on new write, multiple rounds |
| `tests/test_tools/test_lint_fix.py` | 8 | check_only, fix single file, directory, select filter, clean file, non-python, diff store recording |
| `tests/test_tools/test_refactor.py` (modified) | +3 | DependencyGraph integration — dependents included, fallback when dep_graph absent |
| `tests/test_main.py` | +3 | Tool registry completeness after fix (symbol-aware tools present) |
| `tests/test_context_assembly.py` (modified) | +1 | `assemble()` without `tool_definitions` still works |
| `tests/test_linter.py` (modified) | +1 | `annotate_diff_with_lint()` with `linter_name` parameter |

**Estimated total new tests**: ~26

---

## Dependencies Map

```
4.4.1 main.py ────────────► create_tool_registry(symbol_index, file_context)
4.4.2 __init__.py + main ──► create_agent() with resume support
4.4.3 redo.py ────────────► DiffStore (new redo stack), UndoTool
4.4.4 lint_fix.py ────────► linter.py (_detect_linter, _find_ruff), DiffStore, WorkspaceManager
4.4.5 refactor.py ────────► DependencyGraph, SymbolIndex
4.4.6 various ────────────► linter.py, context.py, __init__.py
4.4.7 tests ──────────────► all of the above
```

---

## Implementation Order

1. **4.4.6** Cleanup — easy wins (remove backups, sort tools, remove unused param)
2. **4.4.5** Wire DependencyGraph into RefactorTool — small change, big impact
3. **4.4.1** Fix CLI integration — critical bug, unblocks all Phase 4.x tools for CLI users
4. **4.4.2** Consolidate agent creation — reduces maintenance burden
5. **4.4.3** RedoTool — completes the undo/redo story
6. **4.4.4** LintFixTool — most complex, benefits from DiffStore redo integration
7. **4.4.7** Tests

---

## Design Decisions

### 1. Redo stack in DiffStore, not separate store

The redo stack is conceptually coupled to the undo stack. Keeping `RedoEntry` and the redo stack in `DiffStore` means one import, one interface, and clear lifecycle (clear on new write). A separate `RedoStore` would add unnecessary complexity.

### 2. LintFixTool uses subprocess, not library

`ruff check --fix` is a subprocess call, not a Python API call. While `ruff` has a Python API, using the CLI ensures the same behavior as manual runs. The tool captures stdout/stderr and parses the output to build the diff.

### 3. LintFixTool records diffs for undo

This is important — auto-fixes should be reversible via `UndoTool`. By computing diffs before and after the fix and recording them in `DiffStore`, the agent can undo any auto-fix that had unintended consequences.

### 4. RefactorTool gets dep_graph via constructor injection

Rather than pulling `DependencyGraph` from the symbol index (which would create an unnecessary coupling), pass it explicitly in the constructor. This keeps the dependency inversion clean — `RefactorTool` accepts what it needs, and the registry wires everything together.

### 5. `tool_definitions` removal is backward compatible

Removing a parameter from a function is a breaking change in theory, but since this is an internal function called only from the ReActLoop and tests, and both callers pass `tool_definitions=None` (the implicit default), removing it is safe.

---

## Changes & Deviations Tracking

### 4.4.1 Fix CLI Integration
| Deviation | Rationale |
|-----------|-----------|
| Bug fixed via delegation to `create_agent()` (which passes all args correctly) rather than fixing the `create_tool_registry()` call in `_execute_task()` | `_execute_task()` now delegates entirely to `create_agent()` (see 4.4.2), which already passes `symbol_index`, `file_context`, and `dep_graph` to `create_tool_registry()`. All symbol-aware tools are available via CLI. |

### 4.4.2 Consolidate Agent Creation
| Deviation | Rationale |
|-----------|-----------|
| `_execute_task()` now calls `create_agent()` and no longer duplicates agent creation. But `create_agent()` does not accept `resume_session_id`/`save_dir` — those are handled externally in `_execute_task()` after the call. | The core duplication bug is fixed. The plan's alternative approach (adding params to `create_agent()`) was not taken; instead `_execute_task()` handles resume/save_dir on the result. |

### 4.4.3 RedoTool
| Deviation | Rationale |
|-----------|-----------|

### 4.4.4 InlineLintFixTool
| Deviation | Rationale |
|-----------|-----------|

### 4.4.5 DependencyGraph in RefactorTool
| Deviation | Rationale |
|-----------|-----------|

### 4.4.6 Cleanup
| Deviation | Rationale |
|-----------|-----------|
| Backup files removed | ✓ |
| `tool_definitions` removed from `assemble()` | ✓ |
| Tools in `create_tool_registry()` are NOT sorted alphabetically | Not done — no apparent reason; likely an oversight |
| `annotate_diff_with_lint()` has `linter_name` parameter | ✓ |

### 4.4.7 Tests
| Deviation | Rationale |
|-----------|-----------|

---

## Future Work (Post-4.4)

- **ExplainCodeTool** — extract and inject documentation/analysis for a symbol
- **Call Hierarchy Tool** — tree of "who calls what" using dep graph + symbol index
- **AST-based rename** — tree-sitter-aware symbol replacement (preserves comments/strings)
- **Cross-workspace references** — find symbol usages across monorepo projects
- **Diff-aware commit suggestions** — group diffs into commit-ready messages
- **Git-aware index refresh** — only re-index files changed since last git commit
- **Watchdog native backend** — faster file change detection (inotify/FSEvents)
