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
(`chef_human/agent/parser.py`) → dispatch non-`finish` tool calls concurrently via `asyncio.gather` (each
wrapped in `asyncio.wait_for(..., timeout=config.tool_timeout)`) → run `finish` serially/terminally →
feed results back into context → let `RetryManager` (`chef_human/agent/retry.py`) decide whether to
retry the step, replan (`Planner`, `chef_human/agent/planner.py`), or escalate/complete. Concurrent tool
calls dispatched together share no per-file locking, so parallel writes to the same file can race.

Post-write, if `lint_after_write` is set, `run_lint` (`chef_human/agent/linter.py`) lints the touched file;
lint errors trigger a rollback of that write (`_rollback_file`) using content captured before dispatch
(`_capture_file_content`).

### Tools (`chef_human/tools/`)

Each tool is a plain class with `name`, `description`, `parameters` (JSON schema) and an async `run()`,
registered into a `ToolRegistry` (`registry.py`) by `create_tool_registry()` (`tools/__init__.py`).
File-mutating tools (`write`, `edit`, `patch`, `refactor`, `lint_fix`) share a single `DiffStore`
(`diff.py`) instance so `undo`/`redo` can reverse the most recent recorded diff. Note: `EditTool` does not
currently pass `old_content`/`new_content` into `diff_store.record()` the way `WriteTool` does, so `undo`
after an `edit()` treats the file as newly-created rather than restoring prior content — check this before
relying on undo/edit interaction.

`refactor_symbol` renames across multiple files but records one `DiffStore` entry per file, so a single
`undo` call only reverts the last file touched, not the whole rename.

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
`complete_stream()`, `embed()`, `count_tokens()`. Tool calls are communicated to the model via
ChatML-style `<tool_call>{...}</tool_call>` tags (`chatml.py` formats the system prompt/tool
definitions; each backend's `parse_tool_calls` extracts them from raw completion text) rather than a
native function-calling API — this is deliberate, for compatibility with small local models that don't
reliably support structured tool-calling.

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
