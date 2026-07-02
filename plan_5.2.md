# Phase 5.2: TUI (Textual split-pane interface)

**Goal**: Add a full split-pane terminal UI — chat/log, file tree, diff/preview — as an
additional, opt-in interactive mode. Per PLAN.md: "Terminal UI with split panes (chat, file
tree, diff view). Built with Textual or Bubble Tea (Go)."

**Prerequisites**: Phase 5.1 (CLI/REPL) complete. Reuses the same agent-construction path
(`create_context_assembler` / `create_backend` / `create_tool_registry`) that `repl` uses.

**Scope decision**: additive, not a replacement. `DebugTUI` (Rich `Live`, single-pane
plan/reasoning/tool/log) stays as-is and remains the default for `run --debug-tui`. The new
Textual UI is a separate command, `chef-human tui`, so existing behavior is untouched and this
phase carries no regression risk to already-working code.

**New dependency**: `textual>=0.60`, added as an optional extra (`pip install -e ".[tui]"`),
consistent with how `llamacpp`/`embeddings`/`indexing`/`rag` are optional.

---

## Task List

- [x] **5.2.1** `chef_human/ui/textual_tui.py` — `ChefHumanTUI` Textual `App` with 3 panes:
  left = `DirectoryTree` over the workspace root; top-right = chat/log (`RichLog`, shows plan,
  reasoning, tool calls/results); bottom-right = preview/diff pane (`RichLog`, shows the
  selected file's content, or the latest write/edit/patch diff once one occurs). Footer =
  `Input` widget for typing the next task.
- [x] **5.2.2** `TuiUI` class implementing the `ReActUI` protocol (same protocol `ReplUI`/
  `StreamingUI`/`DebugTUI` implement), pushing formatted lines into the app's widgets instead
  of printing to a `Console`.
- [x] **5.2.3** Wire `chef-human tui` command in `main.py`, modeled on `repl`/`_run_repl`:
  build context/backend/tool_registry/planner once, then for each submitted task construct a
  fresh `ReActLoop` with `TuiUI` and run it as a Textual worker (`self.run_worker(...)`) so the
  UI stays responsive while the agent runs.
- [x] **5.2.4** Tests for `TuiUI` protocol compliance and diff/log formatting, and for the App's
  widget wiring, using Textual's `App.run_test()` pilot harness.

---

## Design notes / deviations

| Decision | Rationale |
|---|---|
| Diff pane is reused as a file preview pane when no diff is active yet | Avoids a 4th pane; "file tree, diff view" from PLAN.md is satisfied by one pane that shows file content until a write/edit/patch happens, then shows that diff — mirrors how a human glances at "what changed" only after something changes. |
| Diff extraction re-parses the same `` ```diff\n...\n``` `` fenced block convention `format_lint_result`/tool results already use | No new wire format; `react_loop.py` already checks `"```diff" in tool_results[...]` for lint annotation, so tool output is guaranteed to use this convention already. |
| Each submitted task gets a fresh `ReActLoop` sharing one `ContextAssembler`/`ToolRegistry` | Same pattern as `_run_repl` in 5.1 — conversation state lives in the shared context, not the loop. |
| `on_approval_request` uses a Textual `ModalScreen` with Yes/No buttons instead of Rich `Confirm.ask` | Rich's blocking `Confirm.ask` would freeze Textual's own event loop; Textual apps need modal screens for blocking-style prompts. |
| No file-tree-driven editing (click-to-edit) | Out of scope for this pass — PLAN.md only asks for a file tree pane, not an editor; the pane is read-only preview. |
| `--tui` flag on `run`/`repl` NOT added | Keeping the new mode as its own subcommand (`chef-human tui`) avoids touching the already-large `run`/`repl` option surface from 5.1. |
