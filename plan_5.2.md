# Phase 5.2: TUI (Textual split-pane interface)

**Goal**: Add a full split-pane terminal UI — chat/log, file tree, diff/preview — as an
additional, opt-in interactive mode. Per PLAN.md: "Terminal UI with split panes (chat, file
tree, diff view). Built with Textual or Bubble Tea (Go)."

**Prerequisites**: Phase 5.1 (CLI/REPL) complete. Reuses the same agent-construction path
(`create_context_assembler` / `create_backend` / `create_tool_registry`) that `repl` uses.

**Scope decision (superseded, see 5.2.5)**: originally additive, not a replacement — `DebugTUI`
stayed as the default for `run --debug-tui` and the new Textual UI was a separate `chef-human tui`
command. This was revised: the split-pane TUI is now the mandatory default for interactive runs
(see **5.2.5** below).

**New dependency**: `textual>=0.60`, originally an optional extra (`pip install -e ".[tui]"`);
promoted to a core dependency in 5.2.5.

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
- [x] **5.2.5** (revision) Made the split-pane TUI mandatory: `textual` moved from the `[tui]`
  optional extra into core `dependencies` (and into `scripts/setup.sh`'s fallback no-deps install
  list, plus a hard import check at the end of setup). `chef-human run` / `chef-human "task"` now
  launch inside `ChefHumanTUI` by default (auto-submitting the CLI-provided task, then exiting
  once it completes) instead of the old `DebugTUI`. `DebugTUI` (`chef_human/ui/debug_tui.py`) and
  its dedicated tests (`tests/test_agent/test_tui.py`) were deleted outright rather than kept
  alongside as dead code. `create_agent()` lost its `debug_tui` param and now always builds
  `NoopUI` — the split-pane TUI, `StreamingUI`, `ReplUI`, etc. are all wired by callers in
  `main.py`, not by `create_agent()` itself. `--headless` and `--no-debug-tui` are unchanged:
  they still skip the TUI entirely (StreamingUI or plain JSON output), since CI/scripting usage
  must not require a real terminal.
- [x] **5.2.6** (revision) Quit rebound from `ctrl+c` to `ctrl+q`: the original `BINDINGS =
  [("ctrl+c", "quit", "Quit")]` shadowed Textual's built-in `ctrl+c`/`super+c` -> copy-selected-
  text-to-clipboard binding (`Screen.action_copy_text`). Removing the override was necessary but
  **not sufficient** -- see 5.2.8, this alone did not actually make selection work. `ctrl+q` was
  already inherited from `App` with `priority=True` so quitting was never actually broken, just
  re-bound for footer visibility.
- [x] **5.2.7** (revision) Added a 4th pane, bottom-left: `#stats-panel` (`RichLog`), under the
  file tree (`#tree-pane`/`#stats-panel` split 60/40 within a new `#left-pane` container). Backed
  by a new `SessionStats` dataclass owned by `TuiUI` (persists across every task submitted in the
  session, not reset per task): tasks run, tool calls/errors, replan count, cumulative prompt/
  completion tokens, current status, current plan step + progress, and the last 5 warnings.
  "Warnings" specifically includes the `plan-check`/`repeat-guard` synthetic corrective messages
  react_loop.py injects (step-not-done feedback, repeated-call nudges, blocked premature
  `finish`/`ask_user`/`edit`) — previously these were only visible mixed into the scrolling chat
  log; now they're also surfaced as a persistent, bounded list.
- [x] **5.2.8** (bugfix, corrects 5.2.6) `ALLOW_SELECT = True` being the default on `RichLog`
  does NOT mean drag-selection actually works there: `Widget.get_selection()` (the base
  implementation `RichLog` never overrides) reads from `self._render()` / `self.render()`, but
  `RichLog` manages its content through an internal scrolled line buffer (`self.lines: list[Strip]`)
  via `render_line()` instead, so `render()` never reflects the log's real content and selection
  silently extracted `""` every time — confirmed empirically via `Pilot.mouse_down`/`hover`/
  `mouse_up` + `Screen.get_selected_text()` in a headless test before writing the fix. Added
  `SelectableRichLog(RichLog)` overriding `get_selection()` to join `strip.text for strip in
  self.lines` (the currently-visible lines) and feed that into `Selection.extract()`; all three
  log panes (`#chat-log`, `#preview-log`, `#stats-panel`) now use it instead of plain `RichLog`.
- [x] **5.2.9** (bugfix) Root cause of two reports — "token counter doesn't work" and "panel
  freezes after the `goto_definition` step" — traced to the same bug: every `RichLog`/`Label` in
  `textual_tui.py` defaults `markup=True`, so any dynamic string written into one (tool-call args,
  tool results, reasoning text, error messages, raw file previews) is parsed as Rich markup.
  Content with brackets that don't form well-formed tags — type hints (`Dict[str, int]`), slices
  (`arr[i:j]`), or a genuinely unmatched closing tag like `foo[/bar]` (exactly the shape of
  `goto_definition` output referencing a symbol path) — either gets silently dropped or raises
  `rich.errors.MarkupError`. An uncaught exception inside a Textual worker (`self.run_worker(...)`,
  used to run the `ReActLoop` task) kills that worker silently with no visible crash: the task's
  `display_result()` (which sums that task's tokens into `SessionStats`) never runs, so the token
  counter appears stuck, and the UI stops updating for that task — "freezing". Fix: import
  `rich.markup.escape` and wrap every dynamic/untrusted string interpolated into a markup-enabled
  widget across `TuiUI` (`on_start`, `on_plan`, `on_reasoning`, `on_tool_call`, `on_tool_result`,
  `_show_diff`, `on_error`, `display_result`, `render_stats`) and `ApprovalModal.compose`. Also
  found and fixed a more severe instance of the same bug in
  `ChefHumanTUI.on_directory_tree_file_selected`, which wrote raw file content into the preview
  `RichLog` with zero escaping — clicking almost any real source file with type hints, regexes, or
  slices in the tree pane would corrupt or crash the preview. Verified via two new regression
  tests in `tests/test_ui/test_textual_tui.py`
  (`test_file_selected_escapes_rich_markup_in_content`,
  `test_on_tool_result_does_not_crash_on_malformed_markup`) asserting bracket-heavy content
  survives intact with no exception.
- [x] **5.2.10** (bugfix, corrects 5.2.8) Reported "selecting still isn't functional" after 5.2.8
  landed. Root cause: `SelectableRichLog.get_selection()` joined the *entire* `self.lines`
  scrollback buffer and treated row 0 of that buffer as row 0 of the selection, but Textual
  computes the selection's row/col in **viewport-relative** coordinates (row 0 == first visible
  row — traced to `_compositor.get_widget_and_offset_at`, which subtracts the widget's screen
  region offset but never adds `scroll_offset`). `RichLog` auto-scrolls to the bottom as content
  streams in, so in any real session the buffer is scrolled and row 0 of the full buffer is not
  row 0 of the viewport — dragging over visible text silently selected stale lines from the very
  start of the session (or nothing, once the offset ran past the buffer length). Fix: slice
  `self.lines[scroll_offset.y : scroll_offset.y + scrollable_content_region.height]` before
  joining, so row 0 of the extracted text lines up with the first line actually on screen.
  Verified with a live `Pilot` drag test — writing 100 lines (forcing scroll), dragging near the
  top of the viewport, and confirming the selected text matched the visible line at
  `scroll_offset.y` rather than `line-000` — added as
  `test_drag_selects_correct_text_when_scrolled` in `tests/test_ui/test_textual_tui.py`.
- [x] **5.2.11** (bugfix) Reported "the program froze when exiting/quitting". Root cause:
  `OllamaBackend.complete()` (`chef_human/llm/ollama_backend.py`) was declared `async def` but
  called `self._client.chat(...)` on the **synchronous** `ollama.Client` — a real blocking HTTP
  request executed directly on the Textual app's asyncio event loop. For however long the local
  model takes to respond, the entire UI thread is stuck: no repaint, no input handling, and
  critically no `ctrl+q` handling either, since quitting just posts an `ExitApp` message that the
  blocked loop can't process until the call returns. `complete_stream()` already used
  `ollama.AsyncClient` correctly, but `complete()` didn't — and `complete()` is what `Planner`
  uses for plan-building and step-by-step verification (`planner.py` — every task's planning step
  and every step's post-hoc verification), so this fires on essentially every task, not just an
  edge case. `embed()` had the same bug (sync `self._client.embeddings(...)`), affecting the RAG
  path. Fix: added `self._async_client = ollama.AsyncClient(...)` in `__init__` alongside the
  existing sync client (kept only for the one-time startup connectivity check in `__init__`,
  which runs before the event loop is driving the TUI so blocking there is fine), and switched
  `complete()`, `embed()`, and `complete_stream()` (which previously constructed a throwaway
  `AsyncClient` per call) to all reuse it. Verified via the live-Ollama integration tests in
  `tests/test_ollama_backend.py` (`test_ollama_basic_chat`, `test_ollama_tool_call`, both pass
  against a real local server) plus the full non-integration suite (1228 passed; the 5 remaining
  failures are a pre-existing, unrelated `sentence_transformers` not being installed in this dev
  environment).
- [x] **5.2.12** (revision, replaces 5.2.6/5.2.8/5.2.10) Reported "selection still doesn't work"
  after two rounds of bugfixes on the drag-select approach. Reverted `SelectableRichLog` entirely
  (all three log panes are plain `RichLog` again) rather than continuing to patch its
  `get_selection()` coordinate math: it required exactly reconstructing Textual's internal
  viewport/scroll-offset/compositor mapping, which had already produced two distinct real bugs
  (5.2.8's `render()`-vs-`self.lines` mismatch, 5.2.10's viewport-vs-full-buffer row-indexing
  mismatch) and no confidence a third edge case wasn't lurking. Replaced with a simpler mechanism
  that can't have the same class of bug because it never touches scroll/viewport coordinates:
  clicking a log pane focuses it (`RichLog.can_focus` is `True` by default), and `ctrl+c` copies
  that pane's *entire* content (`"\n".join(strip.text for strip in log.lines)`, unconditionally,
  not just what's visible) to the clipboard via `App.copy_to_clipboard()`. The binding
  (`ChefHumanTUI.action_copy_focused_pane`) is declared `priority=True` because Screen's built-in
  `ctrl+c -> screen.copy_text` binding (for drag-selection, which no widget here implements
  anymore) would otherwise shadow it, since screen-level bindings resolve before app-level ones.
  Trade-off: this copies the whole pane rather than an arbitrary user-chosen span — accepted as a
  reasonable simplification given drag-select's fragility here; a future revision could add
  "copy last N lines" or similar if whole-pane copy proves too coarse. `tests/test_ui/
  test_textual_tui.py`'s `TestTextSelectionAndCopy` class (drag-simulation tests exercising the
  now-deleted subclass) was replaced with `TestCopyPane`, covering: the binding resolves to the
  new action (not shadowed by Screen's), plain `RichLog` is used, `ctrl+c` copies a focused pane's
  content, copying is unaffected by scroll position (unlike the old approach), and `ctrl+c` is a
  no-op when the task-input (not a log pane) has focus or the focused pane is empty.
- [x] **5.2.13** (bugfix) Reported the token counter still reads 0↑/0↓ while a task is actively
  running (visible mid-task: "Tasks run: 1 ... Tokens: 0↑ / 0↓ ... Status: Running"). Root cause:
  `ReActLoop` accumulates `_total_prompt_tokens`/`_total_completion_tokens` internally per LLM
  call (react_loop.py, previously lines ~185-187) but only ever exposed the running total once,
  at the very end, via `AgentResult` -- consumed by `TuiUI.display_result()`, which is only called
  after the whole task (which can run for minutes across many LLM calls and tool calls) finishes.
  So the counter wasn't broken, it was just correctly showing 0 for the entire duration of every
  task by design, which reads as "not updating." Fix: added `on_token_usage(prompt_tokens,
  completion_tokens)` to the `ReActUI` protocol (`chef_human/ui/protocol.py`, plus no-op
  implementations in `NoopUI`, `StreamingUI`, `ReplUI` -- required since `ReActLoop` now calls it
  unconditionally after every LLM response with usage data, and those UIs don't inherit from a
  common base class). `react_loop.py` now calls `self._ui.on_token_usage(prompt_delta,
  completion_delta)` with that single call's usage immediately after accumulating it into the
  loop's own running total. `TuiUI.on_token_usage()` adds the delta into `SessionStats` and
  re-renders the stats panel immediately, so the counter now ticks up live as the task progresses.
  Removed the corresponding `self.stats.total_prompt_tokens += result.total_prompt_tokens` (and
  completion-tokens) lines from `display_result()` -- since totals are now accumulated live during
  the run, adding the end-of-task cumulative total again there would double-count every token.
  Note: `Planner`'s own LLM calls (plan-building, step verification) go through a separate call
  path that was never wired into `ReActLoop`'s token accumulation at all (a pre-existing gap, not
  introduced by this fix) -- so the displayed total still slightly undercounts vs. true Ollama
  usage; out of scope for this fix, which only addresses the "counter doesn't move" complaint.
  Verified live via the `Pilot` harness (`ui.on_token_usage(123, 45)` while the app is mounted
  immediately shows `Tokens: 123↑ / 45↓` in the rendered stats panel, with no task having
  completed) plus two new tests in `tests/test_ui/test_textual_tui.py`
  (`test_on_token_usage_accumulates_live`, `test_display_result_does_not_double_count_tokens`).
- [x] **5.2.14** (revision) Closes the gap noted in 5.2.13: `Planner`'s own LLM calls
  (`generate_plan`, `verify_step`, `update_plan` -- plan-building and step verification) run on a
  call path entirely separate from `ReActLoop`'s main reasoning loop and were not wired into token
  accounting at all, so the displayed total undercounted true usage by however much planning/
  verification cost (often non-trivial, since these fire once per task and once per step
  respectively). Fix: `Planner` gained a public `on_usage: Callable[[int, int], None] | None`
  attribute (default `None`) and a private `_complete()` helper that all three LLM call sites now
  go through; `_complete()` invokes `on_usage(prompt_tokens, completion_tokens)` after every call
  that returns usage data. `ReActLoop.__init__` sets `self._planner.on_usage = self._record_usage`
  (a new method factoring out the accumulate-and-notify-UI logic that was previously inlined at
  the one call site in `run()`, now shared by both the main loop and the planner callback) so
  planner calls feed the exact same running total and the exact same live UI updates
  (`on_token_usage`) as the main loop's calls, with no separate code path to keep in sync.
  Verified via new tests: `tests/test_agent/test_planner.py`'s `TestUsageCallback` (5 tests --
  each of the three methods reports usage through the callback with the right values, no crash
  when `on_usage` is unset, and no callback invoked when a response carries no usage data) plus
  `tests/test_agent/test_react_loop.py`'s `test_planner_usage_is_included_in_totals` (a real
  `Planner` wired into a `ReActLoop` run: planner's 40/8 tokens + main loop's 50/10 tokens sum to
  90/18 in the final result) and `test_planner_usage_reported_to_ui_live` (confirms
  `ui.on_token_usage(40, 8)` is called for the planner's usage specifically, not just folded
  silently into the end total). Also manually verified live via the `Pilot` harness: a real
  `Planner.generate_plan()` call with `on_usage` wired to `TuiUI.on_token_usage` immediately shows
  the usage in the rendered stats panel.
- [x] **5.2.15** (feature + revision) Reported: a task ("Implement the plan described in plan.md")
  called `finish` after only reading the file (rejected correctly by the require-plan-complete
  guard), the agent doesn't reliably explore the actual codebase before implementing from a plan
  document, another freeze was observed after a rejected finish, and asked for a file-based
  logging/debug mode to diagnose this kind of thing going forward.
  - **Logging**: `chef_human/agent/react_loop.py` gained `logger.debug`/`.info`/`.warning` calls
    at every previously-invisible decision point: LLM call start/finish (with wall-clock duration
    and usage), each tool call dispatched (name + args) and the batch's total dispatch time, each
    of the four pre-dispatch guard rejections (premature finish, vague ask_user, destructive-
    command approval, read-before-edit) with the specific reason, step auto-completion via the
    investigative-keyword heuristic, step verification verdicts, replans, escalation, and max-
    steps-exceeded. Added `chef_human/main.py`'s `_configure_logging(log_file)`: with no
    `--log-file`, behavior is unchanged (WARNING to stderr); with `--log-file PATH`, everything
    above goes to that file at DEBUG level instead (`%(asctime)s %(levelname)-8s %(name)s:
    %(message)s`). Added `--log-file` to `run`, `repl`, and `tui` CLI commands, threaded down to
    `_execute_task`/`_run_repl`/`_run_tui` (writing to a file rather than stdout/stderr matters
    specifically for the TUI, which owns the terminal). Verified end-to-end with a scripted replay
    of the reported scenario (mocked LLM: read plan.md, then two rejected `finish` attempts, then
    max-steps) confirming the log file captures the full turn-by-turn sequence including exactly
    which guard fired and why -- this is now available for diagnosing a future freeze by checking
    the last line written before the hang, rather than guessing from the TUI transcript.
  - **Root cause of "why does it try to finish immediately"**: two contributing factors, both now
    visible in the log output above. (1) The model itself (a small local model) decided to call
    `finish` after only completing the first plan step -- this is a base-model capability
    limitation that can be nudged but not eliminated by prompting; the existing
    `require_plan_complete_to_finish` guard already caught and rejected it correctly, which is
    exactly the intended behavior (not a bug). (2) `_looks_investigative()`'s keyword heuristic
    (`read`, `identify`, `check`, `list`, etc. -- react_loop.py `_INVESTIGATIVE_KEYWORDS`)
    auto-completes a plan step the instant *any* tool call succeeds while it's current, skipping
    the LLM verifier entirely for steps whose description merely contains one of those words. In
    the reported plan, steps 1-3 ("Read...", "Identify...", "Create a task list...") all match, so
    progress can look faster than real work performed -- flagged for awareness, not changed here
    (this exact heuristic was already independently flagged by the Angle Altitude finder in the
    stopped code-review; changing it is a larger behavioral tradeoff, better done as its own
    reviewed change than folded into a bugfix).
  - **Codebase exploration**: `PLANNER_SYSTEM_PROMPT` and `AGENT_SYSTEM_PROMPT`
    (`chef_human/agent/prompts.py`) both gained an explicit rule that implementation tasks must
    explore the actual source files (ls/glob/grep/read) before writing code, and must not plan or
    implement straight from a task description or plan/design document alone -- what to build
    depends on what already exists in the repo, not just on what the document says. Also
    strengthened the finish-eligibility language to state directly (not just in the rejection
    error) that having *read* a step's description or a document describing it is not evidence of
    completion. These are prompt-level nudges, not guarantees -- a small local model can still
    ignore them, same caveat as (1) above; `tests/test_agent/test_prompts.py` (20 tests) still
    pass unchanged, confirming the additions didn't break existing prompt-format assertions.
- [x] **5.2.16** (bugfix) The new debug log (5.2.15) immediately paid off: it showed the "freeze"
  was not slow inference (prior LLM calls in the same log took up to 125s and still completed
  normally) but a genuine deadlock, with the log's last line being `chef_human.tools.user: User
  asked: ...` and nothing after it. Root cause: `AskUserTool.run()`
  (`chef_human/tools/user.py`) calls `print()` and `sys.stdin.readline()` directly, bypassing the
  `ReActUI` abstraction entirely. Under the Textual TUI, Textual owns the terminal in raw/
  alternate-screen mode, so that blocking synchronous stdin read both hangs the whole asyncio
  event loop (no rendering, no input processing -- the actual "freeze") and can never receive a
  real answer (no visible prompt, and normal keystrokes are captured by Textual's own input
  driver, not delivered to that file descriptor) -- a true deadlock, not a slow response.
  - Added `on_ask_user(question: str) -> str` to the `ReActUI` protocol
    (`chef_human/ui/protocol.py`), plus a shared `ask_via_stdin()` helper factored out of the old
    `AskUserTool.run()` body for terminal-based UIs where a blocking stdin read is actually safe
    (`NoopUI`, `StreamingUI`, `ReplUI` all delegate to it -- none of them put the terminal in raw
    mode the way Textual does).
  - `TuiUI.on_ask_user` (`chef_human/ui/textual_tui.py`) instead pushes a new `AskUserModal`
    (`ModalScreen[str]`, sibling to the existing `ApprovalModal` used for destructive-command
    approval) via `push_screen_wait` -- a text `Input` + Submit/Skip buttons, collected through
    Textual's own event loop exactly like the approval flow already does, instead of blocking
    the loop itself.
  - `react_loop.py` now intercepts `ask_user` entirely before tool dispatch (same pattern as the
    existing `finish` and destructive-`bash` interceptions): the vague-question guard runs first,
    then `self._ui.on_ask_user(question)` is awaited directly and its answer used as the tool
    result -- `AskUserTool.run()` is never called from the main loop anymore (it's left in place,
    still tested directly in `tests/test_tools/test_user.py`, as a standalone implementation for
    any caller outside `ReActLoop`). This also fixes a secondary latent issue: previously
    `ask_user` was dispatched via `asyncio.wait_for(tool.run(...), timeout=tool_timeout)` (default
    60s) alongside ordinary tool calls, which would have raced against how long a human takes to
    type an answer; the interception path has no such timeout.
  - Updated the three existing `TestAskUserVagueQuestionGuard` tests in
    `tests/test_agent/test_react_loop.py` that asserted `ask_tool.run.assert_awaited_once()` --
    now asserting `ask_tool.run.assert_not_awaited()` plus `ui.on_ask_user.assert_awaited_once_with(...)`.
    Added `test_on_ask_user_submits_typed_answer` and `test_on_ask_user_skip_returns_default_message`
    to `tests/test_ui/test_textual_tui.py` (real `Pilot` click/type/submit through the modal).
    Manually verified end-to-end via a scripted `ReActLoop` + `TuiUI` run through the `Pilot`
    harness: the modal appears, typing "Postgres" and clicking Submit correctly delivers that
    answer back into the loop with no hang -- confirming the deadlock is gone.

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
