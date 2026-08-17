# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

chef-human is a local AI software development tool: a ReAct-style coding agent that runs entirely
against local LLMs (Ollama or llama.cpp), with its own file tools, symbol index, and RAG retrieval.
There is no cloud LLM dependency — everything is designed to run on consumer hardware.

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
```

All four extras are genuinely optional — the base `pip install -e ".[dev]"` install is enough to run the
CLI. `tree-sitter` in particular is only ever imported lazily (inside `TYPE_CHECKING` guards / try-excepts
in `symbols/extractor.py` and `symbols/grammars.py`); `CompositeExtractor` falls back to `RegexExtractor`
when it isn't installed. Keep any future `tree_sitter`/`faiss`/`sentence_transformers` import in those
files lazy — a top-level `from tree_sitter import ...` there previously made `import chef_human.main`
(and therefore the whole CLI) crash for anyone who only ran the documented base install.

```bash
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

## Configuration

`Settings` (`chef_human/config.py`) is a frozen dataclass loaded once at import time into the module-level
`settings` singleton. Load order (later overrides earlier): `config.toml` in cwd → nearest ancestor
`.chef-human/config.toml` → environment variables prefixed `CHEF_` (e.g. `CHEF_OLLAMA_MODEL`). CLI flags
like `--model`/`--temperature`/`--config` are applied by monkeypatching `chef_human.config.settings` for
the duration of agent construction (see `_execute_task`/`_run_repl` in `main.py`) — don't assume
`settings` is stable across the process lifetime.

`load_settings()` drops (with a `logger.warning`) any merged key that isn't an actual `Settings` field
before constructing it — an unrelated `CHEF_`-prefixed env var or a typo'd `config.toml` key used to raise
`TypeError` straight out of the module-level `settings = load_settings()` and crash the whole process at
import time. A malformed `config.toml` raises a `ValueError` with the file path and parse error rather
than a raw `tomllib.TOMLDecodeError` traceback.

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
  vector store) instead, plus a `SymbolIndex` with no `DependencyGraph`. Because `dep_graph` is always
  `None` on this path, `create_tool_registry()` silently skips registering the symbol-aware tools
  (`lookup_symbol`, `refactor_symbol`, `reference_finder`, `goto_definition`) on large repos — worth
  knowing before assuming those tools are always available.

### The agent loop (`chef_human/agent/react_loop.py`)

`ReActLoop.run()` is the core loop: assemble context → call the LLM → parse tool calls
(`chef_human/agent/parser.py`) → dispatch non-`finish` tool calls concurrently via `asyncio.gather` (each
wrapped in `asyncio.wait_for(..., timeout=config.tool_timeout)`) → run `finish` serially/terminally →
feed results back into context → let `RetryManager` (`chef_human/agent/retry.py`) decide whether to
retry the step, replan (`Planner`, `chef_human/agent/planner.py`), or escalate/complete. Concurrent tool
calls dispatched together share no per-file locking; `run()` guards against the one case this actually
bites (two `write`/`edit`/`patch` calls targeting the same path in one turn — see `claimed_write_paths`)
by rejecting the later one with a tool error instead of racing them, since both would otherwise capture
the same pre-write snapshot for the lint-rollback path below.

Post-write, if `lint_after_write` is set, `run_lint` (`chef_human/agent/linter.py`) lints the touched file;
lint errors trigger a rollback of that write (`_rollback_file`) using content captured before dispatch
(`_capture_file_content`).

A replan (`RetryAction.REPLAN`) marks the step that triggered it `StepStatus.failed` before calling
`Planner.update_plan()` — `update_plan()` only carries `completed` steps forward into the revised plan, so
this is mostly for accurate state at the moment of replanning rather than something that survives into the
new plan. `StepStatus.skipped` has no producer anywhere and is currently unused.

The no-tool-calls turn (plain reasoning, or a malformed tool call) and the has-tool-calls turn both route
their `RetryAction` through the shared `_handle_replan_or_escalate()` for `REPLAN`/`ESCALATE`. This matters
because a small local model can get stuck *narrating* an action ("Let's call the `ls` tool...") without
ever emitting a real `<tool_call>`, turn after turn, verbatim — seen in production with qwen2.5-coder:7b.
The no-tool-calls branch now runs the step verifier on that reasoning *before* deciding the `RetryAction`
(a rejection is fed to `RetryManager.record_iteration` as a real failure, `(1, 1, [...])`, not the
unconditional `(0, 0, [])` it used to pass regardless of the verdict) and detects verbatim-repeated
reasoning the same way the tool-call path's `is_repeat_call` detects a repeated tool call. Before this,
`record_iteration(0, 0, [])` always returned `STEP_COMPLETED` and reset `consecutive_failures` to 0 no
matter what the verifier concluded, and the no-tool-calls branch never acted on `REPLAN`/`ESCALATE` at
all — so this exact failure mode was invisible to every safety net here and just silently burned all of
`max_steps` before reporting a generic "Max steps exceeded" failure.

### Tools (`chef_human/tools/`)

Each tool is a plain class with `name`, `description`, `parameters` (JSON schema) and an async `run()`,
registered into a `ToolRegistry` (`registry.py`) by `create_tool_registry()` (`tools/__init__.py`).
File-mutating tools (`write`, `edit`, `patch`, `refactor`, `lint_fix`) share a single `DiffStore`
(`diff.py`) instance so `undo`/`redo` can reverse the most recent recorded diff. Both `WriteTool` and
`EditTool` pass `old_content`/`new_content` into `diff_store.record()`, and `RefactorTool` records
multi-file renames as a single batch entry (`path="batch:refactor_symbol:{old}→{new}"`, old/new content
maps as JSON) that `UndoTool`'s `is_batch` path restores in one call — a single `undo()` after either an
`edit()` or a `refactor_symbol()` correctly restores everything (verified directly; earlier notes here
claiming otherwise were stale).

`BashTool`'s destructive-command guard (`shell.py`: `_is_blacklisted`/`_is_destructive`) matches against
each shell segment's leading token (split on `;`/`&&`/`||`/`|`), recursing into common wrapper invocations
(`bash -c "..."`, `sudo ...`, `env ...`, etc. — see `_command_segments`) rather than just the whole
command's literal prefix/substring, so chained (`echo hi && rm -rf ./x`) and wrapped commands are now
caught too. `react_loop.py`'s `_is_destructive_command` just delegates to `BashTool._is_destructive` (no
longer a separate re-implementation that could drift). This is still a lexical heuristic, not a sandbox —
things with no shell-level trace of the dangerous operation (e.g. `python3 -c "shutil.rmtree(...)"`,
Python API calls rather than shell commands) are inherently outside what it can catch.

### Symbol indexing (`chef_human/agent/symbols/`)

`SymbolIndex` builds/persists (`.chef-human/index.json`) a per-file symbol table via `CompositeExtractor`,
which tries `TreeSitterExtractor` first (tree-sitter query patterns per language in `extractor.py`'s
`_TS_QUERIES`, language→module mapping in `grammars.py`'s `GrammarLoader`) and falls back to
`RegexExtractor` (`_LANG_PATTERNS`). Adding a language requires updating multiple tables in lockstep:
`grammars.py`'s package map, `extractor.py`'s `_LANG_MAP`/`_TS_QUERIES`/`_LANG_PATTERNS`, and
`dependencies.py`'s `_IMPORT_PATTERNS` for `DependencyGraph`. Missing one degrades that subsystem silently
(e.g. currently ruby/c/cpp are "supported" per the language map but have no `_TS_QUERIES` entries, so
extraction returns nothing for them with no error).

When `settings.watch_files` is true, both `_build_symbol_context_assembler` and
`_build_rag_context_assembler` (`agent/__init__.py`) start a `chef_human/agent/watcher.py` `FileWatcher`
(a daemon thread polling mtimes every `settings.watch_interval` seconds) whose `on_change` calls
`SymbolIndex.refresh()` (and, on the RAG path, `RAGRetriever.update()` too) so the index doesn't go stale
as files change mid-session — either externally or via the agent's own writes. The watcher instance is
exposed as `ContextAssembler.file_watcher` (`None` if `watch_files` is off). `refresh()` now also purges
entries for files that no longer exist (`SymbolIndex._remove_file`) instead of silently leaving them
stale. `DependencyGraph` is *not* refreshed incrementally here — it only supports a full `build()` — so it
can still drift from the symbol index between full rebuilds; that's a known gap, not a bug to "fix" by
rebuilding it on every change (too expensive to do per-file-change).

### RAG (`chef_human/agent/rag/`)

`CodeChunker` splits files into token-bounded chunks (via the active `Tokenizer`), `VectorStore` wraps
FAISS (`faiss.IndexIDMap(faiss.IndexFlatIP(dim))`, so individual vectors can be removed by id — plain
`IndexFlatIP` can't), `RAGRetriever` ties chunking + embedding (`chef_human/llm/embeddings.py`'s
`EmbeddingsBackend`, sentence-transformers) + store together. `RAGRetriever.build()` always clears and
rebuilds the whole store (used once at startup by `_build_rag_context_assembler`, which also tries
`VectorStore.load()` of a persisted index first when `settings.persist_index` is set); `update(files)` is
the incremental path used by the `FileWatcher` wiring described above — it removes and re-embeds just the
given files' chunks via `VectorStore.remove_by_file()` instead of a full rebuild. `_build_rag_context_assembler`
also builds the store/`SymbolIndex` on init the same way the symbol path does — it used to construct
`RAGRetriever` around a permanently-empty store with no build or load step at all, so `retrieve()` silently
returned `[]` for the whole session on every large-repo run; that's now fixed, so don't reintroduce it by
skipping the `rag_retriever.build(files)` call in `agent/__init__.py`.

### LLM backends (`chef_human/llm/`)

`create_backend()` (`llm/__init__.py`) picks `OllamaBackend` or `LlamaCppBackend` based on
`settings.llm_backend`. Both implement a shared `LLMBackend` protocol (`backend.py`): `complete()`,
`complete_stream()`, `embed()`, `count_tokens()`.

`ReActLoop` passes `tools=self._tools.get_definitions()` on every `CompletionRequest` — `OllamaBackend`
forwards that as Ollama's native `tools=` chat parameter and reads `response["message"]["tool_calls"]`
first, only falling back to regex-scraping `<tool_call>{...}</tool_call>` tags out of the raw content
(`_parse_tool_calls_from_content`) if the native field is empty (e.g. a model/template with no
tool-calling support at all). For a model like `qwen2.5-coder` whose Ollama template *does* support
`.Tools`, Ollama itself injects the tool list and the `<tool_call>` format instruction into the system
message from that `tools=` param — so `AGENT_SYSTEM_PROMPT` (`agent/prompts.py`) deliberately does **not**
also embed a tool list or format instruction; it used to, and that duplication (two overlapping,
differently-worded "how to call a tool" blocks in the same system message) is suspected to have made
tool-call adherence *worse*, not better, for small local models. `AGENT_SYSTEM_PROMPT` only tells the
model to actually invoke the (natively-provided) tool rather than describing the action in prose — the
observed failure mode with `qwen2.5-coder` was repeatedly narrating an intended tool call
("Let's call the `ls` tool...") without ever emitting one, see the `react_loop.py` retry/escalate notes
above. `LlamaCppBackend.format_chatml` has its own separate, simpler tool-list injection (a bare JSON
dump, no explicit format instruction) since llama.cpp has no equivalent native/template-driven path;
`parse_tool_calls`/`strip_tool_calls` there still expect `<tool_call>` tags in the raw text, unconditionally
(no native alternative to fall back from).

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

## Testing conventions

`pytest-asyncio` runs in `asyncio_mode = "auto"` (see `pyproject.toml`) — async test functions don't need
an explicit marker. Tests mirror the package layout under `tests/` (`test_agent/`, `test_tools/`,
`test_symbols/`, `test_rag/`, `test_ui/`). Tests marked `@pytest.mark.integration` require a live Ollama
server with a pulled model and are excluded by default; run them explicitly with `-m integration`.
