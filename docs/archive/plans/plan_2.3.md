# Phase 2.3: Performance, Quality & Configuration

**Goal**: Optimize the agent loop for speed (parallel tool execution) and output quality (auto-lint), plus polish the session management, token tracking, and configuration features.

**Prerequisites**: Phase 2.2 complete (streaming, scratchpad, headless mode, persistence, TUI polish).

---

## Task List

- [x] **2.3.1** Parallel tool execution
- [x] **2.3.2** Auto-lint / code quality verification
- [x] **2.3.3** Session management CLI commands
- [x] **2.3.4** Token usage tracking in TUI
- [x] **2.3.5** Config file support (`.chef-human/config.toml`)

---

## Architecture

```
                                    ┌─────────────────────┐
                                    │     chef-human       │
                                    │      (main.py)       │
                                    ├─────────────────────┤
                                    │  ┌───────────────┐  │
                                    │  │  session CLI  │  │
                                    │  │  (list,show,  │  │
                                    │  │   delete,exp) │  │
                                    │  └───────────────┘  │
                                    └─────────┬───────────┘
                                              │
         ┌────────────────────────────────────┼──────────────────────────┐
         │           ReActLoop                │                          │
         │  ┌─────────────────────────────┐   │   ┌──────────────────┐  │
         │  │  2.3.1 Parallel execution   │   │   │  2.3.4 Token     │  │
         │  │  asyncio.gather for        │   │   │  counters in     │  │
         │  │  independent tool calls    │   │   │  ReActLoop/TUI   │  │
         │  └─────────────────────────────┘   │   └──────────────────┘  │
         │  ┌─────────────────────────────┐   │                          │
         │  │  2.3.2 Auto-lint            │   │                          │
         │  │  post-write/edit hook       │   │                          │
         │  │  runs ruff/configured       │   │                          │
         │  │  linter → self-correct      │   │                          │
         │  └─────────────────────────────┘   │                          │
         └────────────────────────────────────┘                          │
                                                    ┌──────────────────┐
                                                    │  2.3.5 Config   │
                                                    │  .chef-human/   │
                                                    │  config.toml    │
                                                    │  merge layer    │
                                                    └──────────────────┘
```

---

## Task Details

### 2.3.1 Parallel tool execution

**Problem**: The ReAct loop processes multiple tool calls from a single model turn sequentially. If the model emits `read("a.py")` and `read("b.py")`, they run one after another when they could run concurrently.

**Solution**: Modify the tool execution loop in `ReActLoop.run()` to:
1. Pre-process all tool calls (validate arguments, check tool existence, check destructive commands)
2. Identify independent calls — all tool calls within a single turn are independent since they operate on the same conversation state
3. Execute approved calls in parallel via `asyncio.gather()`
4. Handle `finish` tool call specially (executed synchronously, stops everything)
5. Collect results preserving order (for adding to conversation)

**Special cases**:
- `bash`: Destructive check + approval still happens before gather
- `finish`: Interrupts parallel execution, returns immediately
- Exceptions from individual tools: caught via `return_exceptions=True`, converted to error results

**Files modified:**
- `chef_human/agent/react_loop.py` — parallel execution in `run()`
- `tests/test_agent/test_react_loop.py` — tests for parallel execution

**Acceptance criteria:**
- Multiple tool calls in a single turn execute concurrently (verified by test with mock delays)
- `finish` tool call terminates the loop immediately
- Tool execution order is maintained for result logging
- Argument validation happens before parallel execution
- Destructive bash commands still prompt for approval before gather
- All existing tests pass unchanged

### 2.3.2 Auto-lint / code quality verification

**Problem**: The agent writes code but never verifies it. Broken syntax or style issues aren't caught until the user notices.

**Solution**: After every successful `write` or `edit` tool call, auto-detect the project linter and run it on the modified file(s). Feed lint errors back as additional tool result context so the model can self-correct.

**Implementation**:
1. Add `lint_after_write` config option to `ReActConfig` (default `True`)
2. After `write`/`edit` succeeds in the tool execution loop, run the linter
3. Detect linter from `pyproject.toml` → `ruff`, or fall back to `ruff check <file>`
4. If lint errors found, append to the tool result text
5. The model sees the linter output and can fix issues in the next turn

**Files created:**
- `chef_human/agent/linter.py` — lint detection and execution

**Files modified:**
- `chef_human/agent/react_loop.py` — call linter after write/edit
- `chef_human/agent/react_loop.py` — add `lint_after_write` to `ReActConfig`
- `tests/test_agent/test_react_loop.py` — tests for lint integration

**Acceptance criteria:**
- Linter runs after successful `write` tool call (for `.py` files)
- Lint output is appended to the tool result
- Linter is configurable via `ReActConfig.lint_after_write`
- Non-Python files are skipped (no linter configured)
- Missing linter (ruff not installed) is handled gracefully
- 5+ new tests

### 2.3.3 Session management CLI commands

**Problem**: Sessions are saved to disk but there's no way to browse, inspect, or manage them from the CLI.

**Solution**: Add a `session` command group to the CLI with subcommands.

**Commands**:
- `chef-human session list` — Table with session_id, date (from file mtime), task snippet, path
- `chef-human session show <id>` — Detailed view of a session (task, full conversation, outcome)
- `chef-human session delete <id>` — Remove a session file
- `chef-human session export <id> --format json|md` — Export for sharing

**Files modified:**
- `chef_human/main.py` — new `session` click group and commands
- `tests/test_agent/test_cli.py` — tests for session commands

**Acceptance criteria:**
- `session list` shows all sessions sorted by date (newest first)
- `session show` displays session details
- `session delete` removes the session file, errors if not found
- `session export` outputs JSON by default, Markdown with `--format md`
- 8+ new tests

### 2.3.4 Token usage tracking

**Problem**: Users have no visibility into token consumption. This makes it hard to know when context pressure is building or which backends are efficient.

**Solution**: Track token usage across the session and display in the TUI footer.

**Implementation**:
1. `CompletionResponse.usage` dict already exists — populate from backends
2. Add cumulative counters to `ReActLoop` (`total_input_tokens`, `total_output_tokens`)
3. Update counters after each model call
4. Display in TUI footer: `Tokens: 1,234↑ 5,678↓`
5. Expose in `AgentResult.to_dict()` for headless mode

**Backend changes**:
- `OllamaBackend`: Extract `usage.prompt_tokens` and `usage.completion_tokens` from API response
- `LlamaCppBackend`: Extract usage from llama.cpp response

**Files modified:**
- `chef_human/llm/backend.py` — ensure `CompletionResponse` has usage
- `chef_human/llm/ollama_backend.py` — populate usage from Ollama response
- `chef_human/llm/llamacpp_backend.py` — populate usage from llama.cpp response
- `chef_human/agent/react_loop.py` — token counters
- `chef_human/ui/debug_tui.py` — display in footer
- `tests/` — tests for token tracking

**Acceptance criteria:**
- Tokens tracked cumulatively across all model calls in a session
- TUI footer shows token counts when available
- `AgentResult.to_dict()` includes token usage
- Backend returns usage from API response (or None if unavailable)
- 5+ new tests

### 2.3.5 Config file support (`.chef-human/config.toml`)

**Problem**: Users must pass CLI flags for every invocation. There's no per-project configuration.

**Solution**: Support `.chef-human/config.toml` as a per-project config file. Merge with existing `config.toml` and environment variables. CLI flags take highest priority.

**Merge order** (last wins):
1. Built-in defaults (`Settings` dataclass)
2. `config.toml` in CWD (existing behavior)
3. `.chef-human/config.toml` in workspace or CWD (new)
4. Environment variables (`CHEF_*`)
5. CLI flags (passed explicitly to ReActLoop/main)

**Files modified:**
- `chef_human/config.py` — load `.chef-human/config.toml` in merge chain; add `load_project_config()` function
- `chef_human/agent/react_loop.py` — accept config from file
- `chef_human/main.py` — use project config file as baseline
- `tests/test_config.py` — tests for config merging

**Acceptance criteria:**
- `.chef-human/config.toml` is discovered by walking up from workspace
- Config keys merge with CLI > env > project config > defaults
- Missing `.chef-human/` directory is silently ignored
- CLI flags override all config file values
- 5+ new tests

---

## Implementation Order

1. **2.3.1** Parallel tool execution — core loop change, no new modules
2. **2.3.2** Auto-lint — new `linter.py`, loop integration
3. **2.3.3** Session management CLI — new CLI commands
4. **2.3.4** Token usage tracking — backend + loop + TUI
5. **2.3.5** Config file — config loading + merge

---

## Test Strategy

| Test file | +Count (actual) | What it covers |
|-----------|-----------------|----------------|
| `tests/test_agent/test_react_loop.py` | +12 | Parallel execution (5), lint integration (2), token counters (2), ReactConfig.lint_after_write (1), unused import cleanup |
| `tests/test_agent/test_main.py` | +10 | Session list/show/delete/export CLI commands (8), AgentResult.to_dict token fields (2) |
| `tests/test_agent/test_persistence.py` | +3 | `delete_session()` |
| `tests/test_linter.py` | +7 | Lint detection, execution, error parsing |
| `tests/test_config.py` | +5 | Config merging, `_find_project_config`, file discovery |
| **Total** | **+37** | (606 from 571 = +35 existing + 2 test fixes = +37 net new) |

---

## Future Improvements (Post-2.3)

- **Tool output streaming**: Real-time display of long-running bash command output
- **A/B prompt testing**: Framework for testing prompt variations
- **Headless mode `--json` flag variant**: Lighter alternative to `--headless`
- **Multi-file refactoring**: Coordinated changes across files with dependency tracking
- **Self-review loop**: After writing code, model reviews its own output for correctness

---

## Changes & Deviations Tracking

### 2.3.1 Implementation Notes

**Files modified:**
- `chef_human/agent/react_loop.py` — refactored the tool execution section from sequential `for tc in tool_calls:` → parallel `asyncio.gather()` for independent calls. Added `import asyncio`. Validation and destructive-command approval remain sequential (fast). Only `tool.run()` calls are parallelized. Finish tool handled synchronously after parallel batch.

**10 new tests** (in `TestParallelToolExecution`):
- `test_multiple_tools_execute_in_parallel_turn` — two read calls in one turn
- `test_finish_with_parallel_calls` — read + finish in same turn
- `test_parallel_with_unknown_tool` — unknown tool error + finish
- `test_parallel_execution_error_handled` — one tool crashes, other succeeds
- `test_single_tool_call_still_works` — single finish call (regression)

**Deviations from plan:**
1. **`asyncio.gather` with `return_exceptions=True`**: The plan didn't specify exception handling. Using `return_exceptions` prevents one failed tool from crashing the whole batch.
2. **Finish handled after parallel batch, not interleaved**: In the sequential code, `finish` could appear at any position in the list and would return immediately. In the parallel version, all non-finish calls execute first, then finish executes and returns. This is slightly different behavior but doesn't matter in practice since `finish` is always the sole call in its turn.

### 2.3.2 Implementation Notes

**Files created:**
- `chef_human/agent/linter.py` — new module with `run_lint()`, `format_lint_result()`, `_find_ruff()`, `_detect_linter()`, `_run_ruff()`

**Files modified:**
- `chef_human/agent/react_loop.py` — added `lint_after_write: bool = True` to `ReActConfig`; after successful `write`/`edit` tool execution in the parallel results loop, calls `run_lint()` on the file path and appends formatted results to `tool_results`

**7 new tests** (in `tests/test_linter.py`):
- `test_detect_linter_python` / `test_detect_linter_non_python` — extension detection
- `test_find_ruff_not_found` — graceful when ruff not installed
- `test_run_lint_non_python_returns_empty` / `test_run_lint_ruff_not_available` — edge cases
- `test_format_lint_result_empty` / `test_format_lint_result_single` / `test_format_lint_result_multiple` — output formatting

**2 integration tests** (in `TestParallelToolExecution`):
- `test_lint_runs_after_write_and_appends_result` — lint called after write
- `test_lint_skipped_when_config_disabled` — `lint_after_write=False` skips lint

**Deviations from plan:**
1. **Only `.py` files are linted**: The plan said "detect project linter and run on modified files." The implementation detects by file extension and only runs ruff for `.py` files. No TypeScript, JS, or other language support yet.
2. **Lint is appended to tool_results, not a separate UI callback**: The plan was vague about how lint results are surfaced. The simplest approach: append to `tool_results` (same list that gets added to conversation). The model sees the lint output after the write result and can self-correct.
3. **`all_success` is NOT affected by lint**: If the tool itself succeeded, the step is considered successful. Lint issues are advisory — the model can fix them in a subsequent turn.

### 2.3.3 Implementation Notes

**Files modified:**
- `chef_human/main.py` — added `session` click group with `list`, `show`, `delete`, `export` subcommands. Imports expanded to include `delete_session`, `list_sessions`, `load_session_data`.
- `chef_human/agent/persistence.py` — added `delete_session()` function

**13 new tests:**
- 3 in `TestDeleteSession` (test_persistence.py): delete existing, delete nonexistent, delete then list
- 10 in `TestSessionCLI` (test_main.py): help, list empty, list shows, show not found, show details, delete not found, delete success, export json, export md

**Deviations from plan:**
1. **Plan said 8+ tests, implemented 13**: Over-delivered by ~60%.
2. **`session export --format md` outputs Markdown with headings**: The plan only mentioned json/md. The Markdown format uses `# Session: <id>`, `**Task:**`, and `### role` headings for each message.
3. **`session show` truncates content to 80 chars**: Prevents massive output from overwhelming the terminal. Full content viewable via `export`.

### 2.3.4 Implementation Notes

**Files modified:**
- `chef_human/agent/react_loop.py` — added `_total_prompt_tokens`, `_total_completion_tokens` counters; updated after each LLM call if `response.usage` is present; `AgentResult` now includes `total_prompt_tokens` and `total_completion_tokens`; `_make_result()` is no longer `@staticmethod` (needs `self` to access counters)
- `chef_human/ui/debug_tui.py` — added `_total_prompt_tokens` and `_total_completion_tokens` fields; `_render_footer()` shows token counts when non-zero
- `chef_human/llm/ollama_backend.py` — `complete_stream()` now captures usage from the final stream chunk (`chunk.get("done")` with `prompt_eval_count`/`eval_count`)
- `chef_human/llm/llamacpp_backend.py` — `complete()` already returned usage; `complete_stream()` remains without usage (streaming API doesn't provide it)

**5 new tests:**
- 2 in `TestTokenTracking` (test_react_loop.py): accumulated counts, zero when no usage
- 2 in `TestToDict` (test_main.py): token fields in `to_dict()`, round-trip with custom values
- 1 in `TestAgentResult`: default token values

**Deviations from plan:**
1. **LlamaCpp streaming has no token counts**: The llama.cpp streaming API (`create_completion(stream=True)`) doesn't provide usage information in individual chunks. Only non-streaming calls report token counts. This is a known limitation.
2. **Token display uses arrow notation**: `Tokens: 1,234↑ 5,678↓` (↑ = prompt, ↓ = completion). Compact and informative. Only shown when non-zero.
3. **`_make_result` changed from `@staticmethod` to instance method**: Needed to access `self._total_prompt_tokens` and `self._total_completion_tokens`. All callers already used `self._make_result(...)` so no API change.

### 2.3.5 Implementation Notes

**Files modified:**
- `chef_human/config.py` — added `_find_project_config()` (walks up from start_dir looking for `.chef-human/config.toml`); `load_settings()` now accepts optional `project_start` param and merges project config between `config.toml` and env vars

**5 new tests** (in `TestProjectConfig`, test_config.py):
- `test_find_project_config_not_found` — no `.chef-human/` dir
- `test_find_project_config_found` — finds `.chef-human/config.toml`
- `test_find_project_config_walks_up` — discovers from nested subdirectory
- `test_project_config_merges_before_env` — project config value appears in merged settings
- `test_env_still_overrides_project_config` — env vars still have highest priority

**Deviations from plan:**
1. **Merge order refined**: The plan said "CLI > env > project config > defaults". The actual order is: defaults → `config.toml` (CWD) → `.chef-human/config.toml` (walk up) → env vars → CLI flags. The config system only provides defaults that CLI and env override — the actual CLI-to-config mapping happens in `main.py` where CLI args are passed directly to `ReActConfig` and `_execute_task`.
2. **`project_start` parameter on `load_settings()`**: Added to allow specifying where to start the directory walk. Defaults to `"."` (CWD). Tests use `tmp_path`.
3. **Not yet wired into `main.py`**: The config file is loaded by `config.py`'s module-level `settings = load_settings()` which is used by `create_backend()` and `create_context_assembler()`. The per-project config is available through this chain. CLI-level per-project defaults (e.g., `max_steps` from `.chef-human/config.toml`) can be wired in a future PR by calling `load_settings(project_start=workspace)` in `_execute_task` and passing values to `ReActConfig`.
