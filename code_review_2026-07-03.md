# Code review — 2026-07-03

Scope: `git diff HEAD` against the working tree at time of review (~5900 lines: `chef_human/agent/`,
`chef_human/tools/`, `chef_human/ui/`, `chef_human/llm/`, `chef_human/main.py`, plus tests).

Process: 8 finder angles (line-by-line diff scan, removed-behavior audit, cross-file tracer, reuse,
simplification, efficiency, altitude, conventions) surfaced 27 deduped candidates; each was
independently verified (CONFIRMED / PLAUSIBLE / REFUTED) against the actual code before inclusion
here. Ranked most-severe first; correctness bugs outrank cleanup/efficiency/architecture findings.

## Correctness bugs

1. **`--model`/`--temperature`/`--config` are silently ignored everywhere, not just in the new TUI
   path.** `chef_human/agent/__init__.py:12` and `chef_human/llm/__init__.py:1` both do
   `from chef_human.config import settings` at module-import time. `main.py`'s monkeypatch pattern
   (`config_module.settings = _resolve_settings(...)`) reassigns the *module attribute*
   `chef_human.config.settings`, which does nothing to those already-bound names — `from X import Y`
   binds a snapshot, not a live reference. Initially flagged (by two independent finders) as a
   scoping bug specific to the new `_run_task_in_tui`; verification disproved that framing
   empirically — `_execute_task`'s "correct" pattern is equally broken. This is a pre-existing,
   repo-wide bug that CLAUDE.md documents as working but isn't; the new code just adds another
   instance of it.

2. **`_verify_and_mark_step` can crash the entire task run.** `react_loop.py`: the step-verification
   LLM call (`self._planner.verify_step(...)`) has no try/except, and `run()`'s while-loop only has
   a bare `finally`. A transient failure in that one call aborts the whole task instead of being
   retried like every other LLM/tool failure in the loop.

3. **Read-then-edit in the same turn wrongly gets rejected.** `react_loop.py`: `files_read` is only
   populated after the whole tool-call batch dispatches via `asyncio.gather()`, but the
   read-before-edit precondition check runs earlier, per-call, in the same loop. A model emitting
   `read(x.py)` + `edit(x.py)` in one response gets the edit incorrectly blocked with "you haven't
   read this file yet."

4. **`patch`, `refactor_symbol`, and `lint_fix` bypass the read-before-edit guard entirely.** The
   guard and `files_read` population both hardcode `tc.name in ("write", "edit")`, so these three
   other mutating tools can edit a file the model never read, with no safeguard.

5. **Step verification can misclassify a complete step as partial.** `planner.py`'s `_parse_verdict`
   checks `"PARTIAL" in upper` before `"COMPLETE" in upper`. A verifier response like `"VERDICT:
   COMPLETE\nREASON: previously partial, now done"` gets matched as PARTIAL.

6. **Verification failures never trigger recovery.** Unparseable verifier output silently defaults
   to `not_complete`, and this path runs entirely outside `RetryManager`'s failure counting
   (`record_iteration` already ran before verification happens). A step can bounce
   pending→verify→pending all the way to `max_steps` with no replan/escalate ever firing.

7. **`finish` can succeed while a step is stuck `failed`/`skipped`.** `Plan.current_step()` only
   checks for `StepStatus.pending`, so the finish-blocking guard doesn't see failed or skipped steps
   as unresolved.

8. **TUI can hang silently on an unhandled exception.** `ChefHumanTUI._run_initial_task` awaits
   `self._on_submit(...)` with no try/except; an exception kills the Textual worker silently and
   `self.exit()` never runs, even with `auto_exit_after_initial_task=True`.

9. **`OllamaBackend`'s shared `AsyncClient` can break across separate event loops.** A single
   `ollama.AsyncClient` is now built once in `__init__` instead of per-call. If a backend instance
   is reused across two separate `asyncio.run()` invocations (REPL-style reuse), it can raise
   "attached to a different loop."

10. **Investigative-step auto-complete never fires for reasoning-only turns.** The shortcut that
    skips the LLM verifier for steps like "review X" is gated on `has_tool_evidence=True`, but the
    plain-reasoning call site always passes the default `False` — so exactly the redo-loop risk the
    shortcut exists to prevent still happens on that path.

## Refuted during verification (not bugs, but worth recording so they aren't re-flagged)

- **Ctrl+Q does correctly interrupt a hung task.** Textual's `App.exit()` cancels all workers
  (`workers.cancel_all()`) and `OllamaBackend`'s calls are real `await`s (not blocking), so
  `CancelledError` propagates correctly through the awaited chain.
- **No `config.toml` default-model change exists in the diff.** A finder claimed the default model
  was changed from `qwen2.5-coder:7b` to `deepseek-coder:33b`; the diff contains no hunk touching
  `config.toml`, and the file still has `qwen2.5-coder:7b`. Finder hallucination.
- **`planner.on_usage` reference cycle is not a real leak.** `ReActLoop.__init__` does create a
  `loop ↔ planner` cycle, but each new task's `ReActLoop.__init__` reassigns
  `planner.on_usage` to the new loop's method, breaking the old reference synchronously (not
  deferred to GC) well within normal session lifetime.

## Confirmed but lower-priority (cleanup / efficiency / architecture)

Didn't make the severity cutoff, but confirmed real and worth addressing eventually:

- `main.py` has three near-identical copies of agent-wiring construction logic
  (`create_agent()`, `_run_task_in_tui`, `_run_tui`).
- The "reject this tool call" 5-step pattern (build error, `on_tool_result`, append, `failed_calls
  += 1`, `continue`) is hand-copied at 6 separate sites in `react_loop.py`'s tool-call loop — any
  future guard added the same way risks omitting `failed_calls += 1`.
- `OllamaBackend.embed()` awaits texts one at a time in a loop instead of dispatching concurrently
  — a large-repo RAG build's embedding phase pays N sequential round-trips.
- `TuiUI.render_stats()` does a full clear-and-rewrite of the stats panel on every event, including
  the new `on_token_usage` (fires after every LLM call) — far more frequent full rebuilds than
  before.
- Several new `logger.debug`/`.info` calls build list comprehensions eagerly, evaluated even when
  logging is at the default WARNING level (no `--log-file`).
- `_INVESTIGATIVE_KEYWORDS` and `_VAGUE_ASK_USER_PATTERNS` are naive substring-matching tuples,
  trivially defeated by rephrasing and prone to false positives/negatives.
- `Scratchpad` entries accumulate with no cap/eviction and are deliberately preserved across
  replans — unbounded growth will eventually crowd out prompt budget on long tasks.
- Three pre-dispatch guards (finish-gating, ask_user-gating, read-before-edit) are each
  hand-implemented inline rather than a shared policy-check abstraction.
- `_verify_and_mark_step` sets `step.status = in_progress` then immediately overwrites it — dead
  intermediate state, never observed by any other code.
- `file_context.py`'s `remember()` reimplements bookkeeping that `_add()` already encapsulates.
- A star-rating format string is duplicated verbatim twice in `main.py`'s `recommend_model_cmd`.
- `files_read` (react_loop.py) duplicates file-visibility tracking `FileContextManager` already
  provides, with different path-resolution and no eviction — latent desync risk (PLAUSIBLE, not
  confirmed as currently manifesting).
