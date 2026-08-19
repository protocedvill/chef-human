# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

chef-human is a local AI software development tool: a ReAct-style coding agent that runs entirely
against local LLMs (Ollama or llama.cpp), with its own file tools, symbol index, and RAG retrieval.
There is no cloud LLM dependency — everything is designed to run on consumer hardware.

## Agent skills

### Issue tracker

Local markdown under `.scratch/<feature>/` (no GitHub remote configured). See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

Note: `docs/INSTALL.md` and `docs/USAGE.md` are stale (they describe a pre-agent-loop state where
"the CLI entry point is not yet wired" and tools/agent are "future"). The CLI, agent loop, and tools
are all implemented — trust the code in `chef_human/` over those docs.

## Commands

```bash
# Install (dev mode, editable)
pip install -e ".[dev]"

# Optional extras
pip install -e ".[llamacpp]"    # llama.cpp backend
pip install -e ".[embeddings]"  # sentence-transformers, for RAG
pip install -e ".[indexing]"    # tree-sitter grammars, for symbol indexing/refactoring
pip install -e ".[rag]"         # faiss-cpu + numpy, for the vector store

# Run the CLI (installed as a console script)
chef-human "some task"          # `run` is the default subcommand
chef-human run "some task" --headless --json
chef-human repl                 # interactive REPL
chef-human session list / show / delete / export
chef-human show-config

# Run tests
pytest tests/ -v
pytest tests/test_agent/test_react_loop.py -v          # single file
pytest tests/test_agent/test_react_loop.py::TestRecordIteration::test_retry_after_first_failure -v  # single test
pytest tests/ -v -m integration   # integration tests (require a running Ollama server + pulled model)

# Lint / typecheck
ruff check .
pyright
```

Ollama must be running locally (`ollama serve`) with the configured model pulled
(default `qwen2.5-coder:7b`) for anything that actually calls the LLM.

### Environment on this machine

The working Python venv is `/home/louis/chef-human/.env` (note: `.env`, not `.venv`) — already has
the project installed editable. It may be missing the `indexing`/`rag`/`embeddings` extras depending
on when it was last touched; if symbol/RAG tests fail with `ModuleNotFoundError` or grammar-loader
`None` results, run `.env/bin/pip install -e ".[indexing,rag,embeddings]"` before assuming a real
regression. Ollama runs as a systemd service already listening on `http://localhost:11434` — it does
not need to be started manually. See `AGENTS.md` for the same info in agent-facing form.

## Configuration

`Settings` (`chef_human/config.py`) is a frozen dataclass loaded once at import time into the module-level
`settings` singleton. Load order (later overrides earlier): `config.toml` in cwd → nearest ancestor
`.chef-human/config.toml` → environment variables prefixed `CHEF_` (e.g. `CHEF_OLLAMA_MODEL`). CLI flags
like `--model`/`--temperature`/`--config` are applied by monkeypatching `chef_human.config.settings` for
the duration of agent construction (see `_execute_task`/`_run_repl` in `main.py`) — don't assume
`settings` is stable across the process lifetime.

`ollama_think` (`bool`, default `False`) controls Ollama's native `think` chat parameter — needed for
reasoning/thinking-style models like Qwen3.x. Set via `config.toml` or `CHEF_OLLAMA_THINK=true`; there
is no CLI flag for it yet. Qwen3.6 (`qwen3.6:27b`, `qwen3.6:35b-a3b` locally) both work well as
`ollama_model` values as of this writing — validated against the full benchmark suite (see below).

## Architecture

### Entry point and orchestration

`chef_human/main.py` is a Click CLI (`cli()` group: `run`, `repl`, `session ...`, `show-config`). `run` and
`repl` both go through `chef_human.agent.create_agent()` / `create_context_assembler()`
(`chef_human/agent/__init__.py`), which wires together the workspace, tokenizer, context assembler, tool
registry, LLM backend, and planner into a `ReActLoop`.

`create_context_assembler()` branches on repo size (`len(files) > settings.max_index_files`):
- **Symbol path** (`_build_symbol_context_assembler`): builds/loads a `SymbolIndex` + `DependencyGraph`
  synchronously (tree-sitter based), used for exact-match code intelligence tools.
- **RAG path** (`_build_rag_context_assembler`): for large repos, builds a `RAGRetriever` (embeddings +
  vector store) instead. This path does **not** populate `ContextAssembler.symbol_index`/`dep_graph`, so
  `create_tool_registry()` silently skips registering the symbol-aware tools (`lookup_symbol`,
  `refactor_symbol`, `reference_finder`, `goto_definition`) on large repos — worth knowing before assuming
  those tools are always available.

### The agent loop (`chef_human/agent/react_loop.py`)

`ReActLoop.run()` is the core loop: assemble context → call the LLM → parse tool calls
(`chef_human/agent/parser.py`) → dispatch non-`finish` tool calls one at a time, in model order (each
wrapped in `asyncio.wait_for(..., timeout=config.tool_timeout)`) → run `finish` serially/terminally →
feed results back into context → let `RetryManager` (`chef_human/agent/retry.py`) decide whether to
retry the step, replan (`Planner`, `chef_human/agent/planner.py`), or escalate/complete. Dispatch within
one response batch is deliberately serialized (not `asyncio.gather`) specifically so two mutating calls
in the same turn can't race on the same path — see the comment above the dispatch loop. The tool classes
themselves still have no per-file locking of their own, though, so that safety only holds as long as
every caller goes through this loop; a caller that invoked tools directly and concurrently (nothing in
`agent/*.py` does today) would not be protected.

Post-write, if `lint_after_write` is set, `run_lint` (`chef_human/agent/linter.py`) lints the touched file;
lint errors trigger a rollback of that write (`_rollback_file`) using content captured before dispatch
(`_capture_file_content`).

**Step verification (`_verify_and_mark_step`) is where most of the interesting bugs live.** It decides,
per turn, whether the plan's current step is done — via deterministic guards first (file-creation/
mutation/execution-step checks against real disk state and tool exit codes) and an LLM verifier
(`Planner.verify_step`) as the fallback for anything not objectively checkable. `_verify_and_mark_step`
now has `logger.debug("VERIFIER_DEBUG ...")` lines logging the exact evidence/history/finish_summary
sent to the verifier on every call — turn on debug logging (`--log-file` + appropriate level) before
trying to diagnose a false escalation rather than re-instrumenting from scratch.

Two related false-escalation bugs were found and fixed by running the larger benchmark cases (marathon,
expert) against Qwen3.6 repeatedly — both looked like "the agent failed" but were actually the harness
rejecting already-correct work until `RetryManager`'s 5-consecutive-failure cap escalated:
- A step description matching both `_looks_investigative` (e.g. "identify") and `_looks_like_execution_step`
  (e.g. "run"/"test") — a very natural phrasing like "run the tests to identify which one is failing" —
  used to route through the investigative branch first, whose no-named-files evidence fallback only
  accepted `read`/`ls`/`grep`/`glob`, not `bash`. A step phrased that way rejected every turn that
  actually ran the tests, no matter how many times they passed. Fixed by widening that fallback to also
  accept a successful `bash` command when the step's wording independently reads as an execution step.
- `EditTool` reported a no-op edit (`old_string == new_string`, or content already matched) identically
  to a real edit: `"Applied edit to X (N occurrences)"` with no diff. When a replan generates a new step
  demanding a fix that was already applied and verified under an *earlier* step's evidence (evidence is
  keyed by exact step-description text via `_step_evidence_key`, so a replan's reworded step starts with
  an empty bucket even though the underlying work is done), the model's redundant re-edit came back
  as this ambiguous "success" message, and the LLM verifier read "no visible diff" as "not fixed" even
  though the file's actual current content (also in its prompt) was already correct. Fixed by making
  `EditTool`'s no-op message explicit ("No changes made: ... already in the desired state") and adding a
  sentence to `STEP_VERIFY_PROMPT` establishing current file contents as ground truth over whether *this*
  turn's tool call produced a visible change.

**Known, not-yet-fixed follow-up**: the root cause underlying both of the above is broader than either
individual fix — `_step_evidence_key` keys accumulated per-step evidence (files written, successful
commands) by the exact step-description string, so *any* replan that rewords a step orphans all
evidence accumulated under the old wording, even when the step's real-world goal was already achieved.
This can still surface as a false escalation on long tasks with multiple replans; if it recurs, the fix
likely needs evidence to survive across a replan for steps whose underlying goal is unchanged, not
another narrow evidence-acceptance patch.

### Tools (`chef_human/tools/`)

Each tool is a plain class with `name`, `description`, `parameters` (JSON schema) and an async `run()`,
registered into a `ToolRegistry` (`registry.py`) by `create_tool_registry()` (`tools/__init__.py`).
File-mutating tools (`write`, `edit`, `patch`, `refactor`, `lint_fix`) share a single `DiffStore`
(`diff.py`) instance so `undo`/`redo` can reverse the most recent recorded diff. `EditTool` does pass
`old_content`/`new_content` into `diff_store.record()`, same as `WriteTool` — this was previously broken
(fixed in `c858b14`) but the two paths still duplicate similar entry-building logic rather than sharing
one; `DiffStore.record()` now delegates to `record_transaction()` internally for that reason, though the
two public methods keep distinct "should I record at all" conditions (`record` trusts its caller-supplied
`diff` string; `record_transaction` compares `old_content`/`new_content` itself, since it has no
separately-supplied diff to trust instead).

`refactor_symbol` renames across multiple files but records them as a single `DiffStore` transaction (one
`record_transaction()` call with every changed file's `FileChange`), so a single `undo` call reverts the
whole rename atomically, not just the last file touched.

`EditTool`'s wording for its result matters more than it looks like it should: the step verifier LLM
reads tool output text as evidence, so ambiguous phrasing directly causes false step-completion
judgments. A no-op edit (`old_string == new_string`) says `"No changes made: ... already in the desired
state"`, not a generic `"Applied edit"` with no diff (see the step-verification note above) — follow this
pattern (explicit, unambiguous success/no-op wording) for any new tool output text, not just edits.

`BashTool`'s destructive-command guard (`shell.py` `BLACKLIST`/`DESTRUCTIVE_PREFIXES`) and the duplicated
approval gate in `react_loop.py` (`_is_destructive_command`) both match on literal command prefixes/
substrings — indirect invocations (`python3 -c "..."`, `bash -c "rm -rf /"`) bypass both.

### Symbol indexing (`chef_human/agent/symbols/`)

`SymbolIndex` builds/persists (`.chef-human/index.json`) a per-file symbol table via `CompositeExtractor`,
which tries `TreeSitterExtractor` first (tree-sitter query patterns per language in `extractor.py`'s
`_TS_QUERIES`, language→module mapping in `grammars.py`'s `GrammarLoader`) and falls back to
`RegexExtractor` (`_LANG_PATTERNS`). Adding a language requires updating multiple tables in lockstep:
`grammars.py`'s package map, `extractor.py`'s `_LANG_MAP`/`_TS_QUERIES`/`_LANG_PATTERNS`, and
`dependencies.py`'s `_IMPORT_PATTERNS` for `DependencyGraph`. Missing one degrades that subsystem silently
(e.g. currently ruby/c/cpp are "supported" per the language map but have no `_TS_QUERIES` entries, so
extraction returns nothing for them with no error).

`SymbolIndex.refresh()` exists for incremental re-indexing and `chef_human/agent/watcher.py`'s
`FileWatcher` exists to drive it from filesystem events, but neither is currently wired up anywhere in
`agent/__init__.py` or `main.py` — the index is built once at startup and does not update as the agent
edits files during a session.

### RAG (`chef_human/agent/rag/`)

`CodeChunker` splits files into token-bounded chunks (via the active `Tokenizer`), `VectorStore` wraps
FAISS, `RAGRetriever` ties chunking + embedding (`chef_human/llm/embeddings.py`'s `EmbeddingsBackend`,
sentence-transformers) + store together. `RAGRetriever.build()` always clears and rebuilds the whole
store — there's no incremental update path analogous to `SymbolIndex.refresh()`.

### LLM backends (`chef_human/llm/`)

`create_backend()` (`llm/__init__.py`) picks `OllamaBackend` or `LlamaCppBackend` based on
`settings.llm_backend`. Both implement a shared `LLMBackend` protocol (`backend.py`): `complete()`,
`complete_stream()`, `embed()`, `count_tokens()`.

`OllamaBackend` supports two tool-call paths: it first checks `response.message.tool_calls` (Ollama's
native structured tool-calling field, populated for models whose template supports it, e.g. Qwen3.x's
`qwen3.5` renderer/parser) via `parser.parse_native_tool_calls`, and falls back to scraping
ChatML-style `<tool_call>{...}</tool_call>` tags out of raw content (`parser.parse_tool_calls`) only
when the native field is empty — for models/templates with no native tool-calling support at all.
`LlamaCppBackend` has no native path and always expects `<tool_call>` tags. `react_loop.py` calls
whichever path fires via `response.message.tool_calls` truthiness (`react_loop.py` ~line 717) — a
`Message` mock in a test must set `tool_calls=None` explicitly, since a bare `MagicMock(content=...)`
auto-fabricates a truthy `.tool_calls` attribute and silently takes the wrong path (bit us once, see
`tests/test_agent/test_persistence.py`).

### Context assembly (`chef_human/agent/context.py`, `file_context.py`, `repo_map.py`)

`ContextAssembler.assemble()` combines the system prompt, conversation history (`ContextManager`,
token-budget aware), a `RepoMap` (directory-tree-like summary), relevant open-file contents
(`FileContextManager`), and — depending on which path `create_context_assembler` took — symbol lookups or
RAG-retrieved chunks, all trimmed to `settings.max_context_tokens`.

### Persistence and UI

Sessions (conversation history + task) are saved/loaded as JSON via `chef_human/agent/persistence.py`
(default dir `DEFAULT_SAVE_DIR`), surfaced through `chef-human session list/show/delete/export` and
`--resume`/`--continue`. UI is pluggable via the `ReActUI` protocol (`chef_human/ui/protocol.py`):
`DebugTUI`, `StreamingUI`, `ReplUI`, or `NoopUI` for headless runs.

## Benchmark suite (`chef_human/benchmark.py`)

`python -m chef_human.benchmark` runs the agent (as a real subprocess, `chef-human run --headless`)
against a fixed set of `BenchmarkCase`s in disposable workspaces, verifying each with an independent
external command (never the agent's own say-so) plus a protected-files integrity check. `--list` shows
cases, `--case <id>` runs one, `--through <level>` runs every case up to that difficulty, `--model` and
`CHEF_OLLAMA_THINK=true` (env, no CLI flag) override the model/backend config for the run. Seven cases
across seven levels, increasing in scope/difficulty: `smoke` (hello_world) → `core` (slugify_contract,
TDD against a supplied spec+tests) → `stretch` (inventory_refactor, multi-file repair+extend) →
`expert` ×2 (lru_cache_repair, scroll_grid_navigation_repair — diagnose a subtle bug in an otherwise-
plausible implementation; the latter is a real niri bug, PR #686) → `frontier` (task_scheduler, build
a topological-sort scheduler from a spec alone, no starter code) → `adversarial`
(rate_limiter_config_trap, spec includes real traps: a protected shared config to read not hardcode, a
banned `time.sleep` call) → `marathon` (library_system, ~24-test multi-module system built from a large
spec, no starter code — the largest-scope case, and the one that originally surfaced the false-
escalation bugs described above since it needs the most turns).

Every case was validated against a hand-written reference solution (and, for bug-hunt cases, the exact
buggy seed) before being wired in — confirm any new case fails/passes exactly as intended in isolation
(`python -c "from chef_human.benchmark import CASES; ..."` + `subprocess.run` the verification command
directly against seed files) before trusting a live model run's pass/fail as ground truth for the case
itself. As of this writing, `qwen3.6:27b` and `qwen3.6:35b-a3b` both pass all 7 cases cleanly (100%);
`qwen3.6:35b-a3b` is consistently ~2x the completion-token throughput of `qwen3.6:27b` (MoE, ~3B active
params) and noticeably faster wall-clock on the harder cases despite being the larger model on disk.

## Testing conventions

`pytest-asyncio` runs in `asyncio_mode = "auto"` (see `pyproject.toml`) — async test functions don't need
an explicit marker. Tests mirror the package layout under `tests/` (`test_agent/`, `test_tools/`,
`test_symbols/`, `test_rag/`, `test_ui/`). Tests marked `@pytest.mark.integration` require a live Ollama
server with a pulled model and are excluded by default; run them explicitly with `-m integration`.
