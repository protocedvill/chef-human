# Phase 4.3: Advanced Code Intelligence

**Goal**: Give the agent cross-file code manipulation capabilities — rename symbols across all files, find definition sites and usages, and annotate diffs with lint warnings. These are higher-level operations that build on the symbol index, dependency graph, and diff infrastructure from earlier phases.

**Prerequisites**: Phases 3.1–3.3 complete (symbol index, RAG, diff-aware editing). Phase 4.1 recommended but not required (undo and patch tools are independent). Phase 4.2 required for `access_count` tracking used in ranking.

---

## Current State

| Component | Status |
|-----------|--------|
| `SymbolIndex` | `lookup()` — finds entries by name. `search()` — substring search. `lookup_by_file()` — per-file listing |
| `DependencyGraph` | `dependencies()` / `dependents()` — file-level import graph. No symbol-level reference tracking |
| `EditTool` | **Single-file** — edits one file at a time. No cross-file rename capability |
| `LookupSymbolTool` | May exist (4.1.2) — can be extended for reference finding |
| `RefactorTool` | **Does not exist** — no cross-file rename |
| `GotoDefinitionTool` | **Does not exist** — no way to jump to a symbol's definition |
| `ReferenceFinder` | **Does not exist** — no way to find all usages of a symbol |
| `linter.py` | Has `run_lint(file_path)` and `format_lint_result(lint_output)` — lint only runs after write/edit |
| Lint-annotated diffs | **Does not exist** — lint output is appended separately, not overlaid on diff |

---

## Task List

- [x] **4.3.1** `RefactorTool` — rename a symbol across all files that define it
- [x] **4.3.2** `GotoDefinitionTool` — locate the definition site of a symbol and load it into context
- [x] **4.3.3** `ReferenceFinderTool` — find all usages of a symbol across the workspace
- [x] **4.3.4** Lint-annotated diffs — overlay lint warnings on the diff output after edits
- [x] **4.3.5** Integration — wire new tools into registry, integrate lint-annotated diffs into ReActLoop
- [x] **4.3.6** Tests

---

## Task 4.3.1: RefactorTool (Cross-File Symbol Rename)

**File to create:** `chef_human/tools/refactor.py`

Renames a symbol (function, class, variable) across all files in the workspace that define or reference it. This is the most complex tool in Phase 4.3.

### Design

```python
class RefactorTool:
    name = "refactor_symbol"
    description = "Rename a symbol across all files that define or reference it."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "old_name": {
                "type": "string",
                "description": "Current symbol name to rename",
            },
            "new_name": {
                "type": "string",
                "description": "New symbol name",
            },
            "scope": {
                "type": "string",
                "description": "Scope of rename: 'definitions' (only defining files), 'all' (all references), 'file' (single file only)",
                "enum": ["definitions", "all", "file"],
                "default": "all",
            },
            "path": {
                "type": "string",
                "description": "Single file to rename in (only used when scope='file')",
                "default": None,
            },
            "dry_run": {
                "type": "boolean",
                "description": "Preview changes without applying them",
                "default": False,
            },
        },
        "required": ["old_name", "new_name"],
    }

    def __init__(
        self,
        workspace: WorkspaceManager,
        symbol_index: SymbolIndex,
        diff_store: DiffStore | None = None,
    ) -> None:
        ...

    async def run(
        self,
        old_name: str,
        new_name: str,
        scope: str = "all",
        path: str | None = None,
        dry_run: bool = False,
    ) -> ToolResult:
        ...
```

### Rename Strategy

**Phase 1 — Discover files to change:**

```
if scope == "definitions":
    entries = index.lookup(old_name)
    files = {entry.file_path for entry in entries}
elif scope == "all":
    # Find all files that reference this symbol
    # (definitions + files that import/include it)
    entries = index.lookup(old_name)
    files = {entry.file_path for entry in entries}
    # Add dependents (files that import files defining the symbol)
    for entry in entries:
        deps = dep_graph.dependents(Path(entry.file_path))
        files.update(str(d) for d in deps)
    # Add files where the symbol name appears textually
    # (use grep-style search as fallback for non-imported usages)
    grep_files = _find_textual_refs(old_name, index, workspace)
    files.update(grep_files)
elif scope == "file":
    resolved = workspace.resolve(path)
    files = {str(resolved)} if resolved.exists() else set()
```

**Phase 2 — Apply rename per file:**

For each file, read content, replace `old_name` with `new_name` using word-boundary-aware replacement:

```python
def _rename_in_file(content: str, old: str, new: str) -> str:
    """Replace whole-word occurrences of `old` with `new` in content."""
    pattern = re.compile(r'\b' + re.escape(old) + r'\b')
    return pattern.sub(new, content)
```

**Phase 3 — Dry run vs. apply:**

- `dry_run=True`: return a summary of files and changes without writing
- `dry_run=False`: apply changes, record diffs in `diff_store`

### Output Format (dry run)

```
**Dry run — 3 files would change:**
  src/models.py:2 occurrences
    @@ -15,3 +15,3 @@
    -class OldName:
    +class NewName:
  src/utils.py:1 occurrence
    @@ -42,1 +42,1 @@
    -    return OldName()
    +    return NewName()
```

### Output Format (applied)

```
Renamed 'OldName' → 'NewName' across 3 files:
  • src/models.py — updated 2 occurrences
  • src/utils.py — updated 1 occurrence
  • tests/test_models.py — updated 1 occurrence

Diffs:
```diff
--- a/src/models.py
+++ b/src/models.py
...
```
```

### Safety & Precautions

| Concern | Mitigation |
|---------|------------|
| Accidental rename of common words | `\b` word boundary prevents `rename` from matching `renamed` |
| Rename across too many files | Limit to 50 files; beyond that, error: "Too many files ({n}) — use scope='definitions' or scope='file'" |
| Symbol not found | Error: "No definitions found for '{old_name}'" |
| New name conflicts with existing symbol | Check index for `new_name` before applying; warn if it exists |
| Partial failure (some files fail) | Roll back all changes; report first failure |
| Dry run is always safe | No files are modified |

### Acceptance Criteria

- `run("OldName", "NewName")` renames in all definition + reference files
- `run("OldName", "NewName", scope="definitions")` renames only in defining files
- `run("OldName", "NewName", scope="file", path="f.py")` renames in single file
- `run(..., dry_run=True)` returns summary without modifying files
- Word-boundary enforcement prevents partial matches
- Error when symbol not found
- Error when too many files (>50)
- DiffStore receives per-file diffs
- Reverse rename restores original state

---

## Task 4.3.2: GotoDefinitionTool

**File to create:** `chef_human/tools/goto_definition.py`

Finds the definition site(s) of a symbol and loads the file into context at the relevant line.

### Design

```python
class GotoDefinitionTool:
    name = "goto_definition"
    description = "Find where a symbol is defined and load it into the file context."
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
        ...
```

### Behaviour

1. Query `index.lookup(name, kind)` for matching entries
2. For each unique file, call `file_context.get(file_path)` to load it into context
3. Read the relevant lines around the definition from the file
4. Return formatted output:

```
**Definitions of 'ContextAssembler' (class):**
  1. src/agent/context.py:42
     ```python
     class ContextAssembler:
         """Assembles context from conversation, files, and symbols."""
         def __init__(self, ...):
     ```
```

If no definition found:
```
No definition found for 'UnknownSymbol'.
```

### Interaction with `LookupSymbolTool`

`GotoDefinitionTool` is a higher-level version of `LookupSymbolTool` that additionally:
- Loads the file into `FileContextManager` (making it available for the next read/edit calls)
- Shows surrounding source lines (not just the signature line)
- Sorts by `access_count` descending (if 4.2.5 is implemented)

### Acceptance Criteria

- `run("ContextAssembler")` returns definitions and loads files into context
- `run("ContextAssembler", kind="class")` filters by kind
- Output includes file path, line number, and surrounding source lines
- Multiple definition sites (overloads in different files) are listed
- "No definition found" for unknown symbols

---

## Task 4.3.3: ReferenceFinderTool

**File to create:** `chef_human/tools/reference_finder.py`

Finds all usages of a symbol across the workspace — both definition sites and references (imports, calls, instantiations).

### Design

```python
class ReferenceFinderTool:
    name = "find_references"
    description = "Find all usages of a symbol across the workspace (definitions + references)."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Symbol name to find references for",
            },
            "include_definitions": {
                "type": "boolean",
                "description": "Include definition sites in results",
                "default": True,
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum files to report (default 20, max 50)",
                "default": 20,
            },
        },
        "required": ["name"],
    }

    def __init__(
        self,
        symbol_index: SymbolIndex,
        workspace: WorkspaceManager,
    ) -> None:
        ...
```

### Reference Discovery Strategy

Two-tier approach:

**Tier 1 — Index-based (fast, precise):**
- Definition sites via `index.lookup(name)`
- Import sites via `dep_graph.dependents()` on files that define the symbol

**Tier 2 — Grep-based (slow, comprehensive):**
- Use `rg` or `grep` to find `\bSymbolName\b` across workspace files
- Only used when Tier 1 returns few results
- Filter to files that aren't already in the result set

```python
def _grep_references(self, name: str) -> list[tuple[str, int]]:
    """Fallback grep for textual references."""
    matches: list[tuple[str, int]] = []
    for f in self._workspace.list_files(max_depth=10):
        if self._workspace.is_ignored(f):
            continue
        try:
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if re.search(r'\b' + re.escape(name) + r'\b', line):
                    matches.append((str(f), i))
        except Exception:
            continue
    return matches[:50]
```

### Output Format

```
Found 12 references to 'SymbolName':
  Definitions (3):
    src/models.py:42
    src/legacy.py:15

  References (9):
    src/main.py:10  —  import SymbolName
    src/main.py:23  —  SymbolName()
    src/utils.py:8   —  from models import SymbolName
    ... and 6 more
```

### Acceptance Criteria

- `find_references("ContextAssembler")` returns definition and usage sites
- `include_definitions=False` omits definition sites
- Results capped at `max_results` with `... and N more` footer
- Works with symbols that appear in text (non-indexed files)
- Empty result returns "No references found"

---

## Task 4.3.4: Lint-Annotated Diffs

**File to modify:** `chef_human/agent/linter.py` and `chef_human/agent/react_loop.py`

Currently, after a write/edit, lint runs and its output is appended as a separate tool message:

```python
# react_loop.py:244-253
if self._config.lint_after_write and tool_result.success and tc.name in ("write", "edit"):
    lint_output = run_lint(file_path)
    if lint_output:
        lint_result = format_lint_result(lint_output)
        tool_results.append(lint_result)
```

This gives the agent two separate messages — the diff and the lint output. For better UX, merge the lint warnings into the diff output so the agent can see which lines have issues.

### Design

New function in `linter.py`:

```python
def annotate_diff_with_lint(diff: str, lint_output: str) -> str:
    """Overlay lint warnings on a unified diff.

    For each lint warning that references a line in the diff's +++ side,
    append an annotation to the relevant hunk.

    Returns the annotated diff string, or the original diff if no annotations.
    """
```

**Annotation format:**

```
@@ -10,7 +10,7 @@
  class Foo:
      def bar(self):
+         x = 1          # ruff: F841 local variable 'x' is unused
          y = 2
```

**How it works:**

1. Parse the lint output for `{file}:{line}:{col}: {code} {message}` patterns
2. Parse the diff for `@@` hunk headers that give `+start,count`
3. For each lint warning, check if the line number falls within a `+` hunk range
4. If yes, annotate the `+` line with a trailing comment `# {linter}: {code} {message}`
5. Lines in the `-` (removed) section are not annotated (the line doesn't exist anymore)

If `annotate_diff_with_lint` returns a non-empty annotated diff, use it instead of the original diff in the EditTool/WriteTool output. The lint result message is still appended as separate feedback for actionable issue lists.

### Integration Point

Modify `react_loop.py`'s lint-after-write section:

```python
if self._config.lint_after_write and tool_result.success and tc.name in ("write", "edit"):
    lint_output = run_lint(file_path)
    if lint_output:
        # Try to annotate the last tool result's diff
        last_idx = len(tool_results) - 1
        if last_idx >= 0 and "```diff" in tool_results[last_idx]:
            annotated = annotate_diff_with_lint(tool_results[last_idx], lint_output)
            if annotated:
                tool_results[last_idx] = annotated
        # Still append raw lint output for complete issue list
        lint_result = format_lint_result(lint_output)
        tool_results.append(lint_result)
```

### Acceptance Criteria

- Lint warning on a `+` line annotates that line in the diff
- Lint warning on a context line (` `) annotates the line
- Lint warning on a `-` line is not annotated
- No lint warnings → diff is unchanged
- No diff in tool result → lint output appended separately (existing behaviour)
- Annotation format is `# {linter}: {code} {message}`

---

## Task 4.3.5: Integration

**Files to modify:** `chef_human/tools/__init__.py` and `chef_human/agent/react_loop.py`

### Tool Registry

```python
def create_tool_registry(
    workspace: WorkspaceManager,
    symbol_index: SymbolIndex | None = None,
    file_context: FileContextManager | None = None,
) -> ToolRegistry:
    diff_store = DiffStore()
    registry = ToolRegistry()
    # ... existing tools ...
    registry.register(RefactorTool(workspace=workspace, symbol_index=symbol_index, diff_store=diff_store))
    registry.register(GotoDefinitionTool(symbol_index=symbol_index, file_context=file_context))
    registry.register(ReferenceFinderTool(symbol_index=symbol_index, workspace=workspace))
    return registry
```

### Lint-Annotated Diff Wiring

Already covered in 4.3.4 — the integration is in `react_loop.py`'s lint-after-write section.

### Acceptance Criteria

- All three new tools are registered and discoverable
- `RefactorTool` has access to `SymbolIndex` and `DiffStore`
- `GotoDefinitionTool` has access to `SymbolIndex` and `FileContextManager`
- `ReferenceFinderTool` has access to `SymbolIndex` and `WorkspaceManager`
- Lint-annotated diffs don't break when there's no diff or no lint

---

## Task 4.3.6: Tests

**New test files:**

| Test file | ~Tests | What it covers |
|-----------|--------|----------------|
| `tests/test_tools/test_refactor.py` | 20 | Single-file rename, cross-file rename, dry run, word boundary, scope filters, unknown symbol, too-many-files limit |
| `tests/test_tools/test_goto_definition.py` | 12 | Exact match, kind filter, multiple definitions, unknown symbol, file context loading |
| `tests/test_tools/test_reference_finder.py` | 10 | Definitions + references, include_definitions flag, max_results, empty results, grep fallback |

**Modified test files:**

| Test file | ~+Tests | What it covers |
|-----------|---------|----------------|
| `tests/test_tools/test_filesystem.py` | 2 | Lint-annotated diff integration (mock lint, assert annotation format) |
| `tests/test_agent_integration.py` | 3 | All new tools registered, wired correctly |
| `tests/test_linter.py` (if exists) | 5 | `annotate_diff_with_lint()` unit tests |

**Estimated total new tests**: ~52

**Test approach:**
- Refactor: Create multi-file workspace with known symbol, run rename, assert all occurrences changed
- Goto: Create a SymbolIndex with test data, run tool, assert output contains file/line/source
- Reference: Similar to goto but also checks for grep-based cross-file refs
- Lint-annotate: Craft known diff + known lint output, assert annotation format

---

## Dependencies Map

```
4.3.1 refactor.py ───────────► SymbolIndex, DependencyGraph, DiffStore, workspace
4.3.2 goto_definition.py ────► SymbolIndex, FileContextManager, workspace
4.3.3 reference_finder.py ───► SymbolIndex, workspace (grep fallback)
4.3.4 linter.py + react_loop ─► diff strings, lint output format
4.3.5 __init__.py ───────────► 4.3.1–4.3.3
4.3.6 tests ─────────────────► all of the above
```

---

## Implementation Order

1. **4.3.4** Lint-annotated diffs — simplest, modifies existing code
2. **4.3.2** GotoDefinitionTool — straightforward, good warm-up
3. **4.3.3** ReferenceFinderTool — builds on index + grep
4. **4.3.1** RefactorTool — most complex, should be done last
5. **4.3.5** Integration — wire everything
6. **4.3.6** Tests — all new tests

---

## Design Decisions

### 1. RefactorTool: word-boundary regex, not AST replacement

`\b` word boundaries plus `re.escape` is sufficient for the common case (renaming `MyClass` to `MyRenamedClass`). AST-based replacement would be more precise (avoiding renames in comments/strings) but requires per-language parsing for every file. The regex approach covers 95% of use cases and is language-agnostic.

### 2. GotoDefinitionTool loads files into `FileContextManager`

This makes the definition immediately available for follow-up reads/edits without an extra `read` tool call. It matches the pattern from `SymbolRetriever.retrieve()` which also calls `file_context.get()`.

### 3. ReferenceFinderTool: two-tier (index + grep)

The index provides fast, precise results for known symbols. The grep fallback catches references that the index missed (symbols referenced in comments, string literals, or in files that the compositor hasn't indexed). The combination gives both speed and completeness.

### 4. Lint annotations: in-diff comments, not separate overlay

Adding `# linter: code message` comments directly in the diff lines is the most explicit way to flag issues. The agent sees exactly which line has a problem. A separate overlay (e.g., a table of warnings) would require the agent to cross-reference line numbers.

### 5. RefactorTool: rollback on partial failure

If renaming fails for any file (permissions, encoding error), the tool rolls back all changes. This prevents a partially-applied rename that would leave the codebase in an inconsistent state. Rollback uses the `DiffStore` reverse diffs.

---

## Changes & Deviations Tracking

### 4.3.1 RefactorTool
| Deviation | Rationale |
|-----------|-----------|

### 4.3.2 GotoDefinitionTool
| Deviation | Rationale |
|-----------|-----------|

### 4.3.3 ReferenceFinderTool
| Deviation | Rationale |
|-----------|-----------|

### 4.3.4 Lint-Annotated Diffs
| Deviation | Rationale |
|-----------|-----------|

### 4.3.5 Integration
| Deviation | Rationale |
|-----------|-----------|

### 4.3.6 Tests
| Deviation | Rationale |
|-----------|-----------|

---

## Future Work (Post-4.3)

- **Call hierarchy** — tree of "who calls what" using the dependency graph + symbol index
- **AST-based rename** — use tree-sitter for language-aware symbol replacement (preserves comments and strings)
- **Inline lint fix** — a tool that auto-fixes lint warnings using `ruff --fix` or similar
- **Diff-aware commit suggestions** — group related diffs into commit-ready messages
- **Cross-workspace references** — find symbol usages across multiple projects in a monorepo
- **Symbol documentation extraction** — pull docstrings/comments for symbols and inject into context
