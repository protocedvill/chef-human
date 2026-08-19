# Development

This is the environment/workflow doc for contributing to chef-human itself. For running the agent
as a user, see [INSTALL.md](INSTALL.md) and [USAGE.md](USAGE.md). For the full test taxonomy and
live-backend markers, see [TESTING.md](TESTING.md) — this doc only summarizes the everyday loop.

## Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Add extras as you touch the subsystems they gate:

```bash
python -m pip install -e ".[indexing]"    # tree-sitter symbol indexing/refactoring
python -m pip install -e ".[rag]"         # faiss-cpu + numpy + sentence-transformers
python -m pip install -e ".[llamacpp]"    # llama.cpp backend
```

Supported Python versions are 3.12 and 3.13 (see `requires-python` in `pyproject.toml`). Newer
Python versions are intentionally excluded until the project's binary optional dependencies
(tree-sitter, faiss, llama-cpp-python) publish compatible wheels — don't chase a failure on a newer
interpreter as a real regression before checking that first.

Ollama is only needed for live/integration testing and for actually running the agent, not for the
default unit suite:

```bash
ollama serve                       # if not already running as a service
ollama pull qwen2.5-coder:7b       # default model
```

Run `chef-human doctor` any time you want a quick read on whether your local environment (Python
version, config, backend reachability, optional extras, workspace) is in a state the agent can
actually run in.

## Everyday commands

```bash
# Full default test suite (deterministic, no live model calls)
pytest

# One file / one test
pytest tests/test_agent/test_react_loop.py -v
pytest tests/test_agent/test_react_loop.py::TestRecordIteration::test_retry_after_first_failure -v

# Live-backend tests (opt-in, excluded by default — see TESTING.md)
pytest -m integration_ollama

# Lint and typecheck
ruff check .
pyright

# Package build sanity check
python -m build
```

See [TESTING.md](TESTING.md) for the full marker taxonomy (`unit`, `ui`, `rag`, `indexing`,
`integration_ollama`, `integration_llamacpp`) and what each requires.

## Before opening a PR

1. `pytest` passes (default marker set — no live backend needed).
2. `ruff check .` and `pyright` are clean, or any new warning is justified in the PR description.
3. If you touched a documented command or config option, the relevant doc (`USAGE.md`,
   `INSTALL.md`, `SAFETY.md`, `ARCHITECTURE.md`) reflects the change in the same PR — this project
   has previously shipped with docs describing a "future" agent loop that had, in fact, already
   shipped; don't repeat that.
4. If you touched benchmark cases, validate the case against its reference/seed in isolation before
   trusting a live model run's pass/fail (see [BENCHMARKS.md](BENCHMARKS.md)).

## Where things live

See [ARCHITECTURE.md](ARCHITECTURE.md) for component boundaries and extension points. The short
version, for orientation:

- `chef_human/main.py` — Click CLI, wires flags into `create_agent()`/`create_context_assembler()`.
- `chef_human/agent/` — the ReAct loop, planner, retry manager, context assembly, symbol index, RAG.
- `chef_human/tools/` — file/shell/patch/diff/undo/refactor/navigation tools + the registry.
- `chef_human/llm/` — backend protocol plus Ollama/llama.cpp implementations.
- `chef_human/ui/` — pluggable UI implementations (`DebugTUI`, `StreamingUI`, `ReplUI`, `NoopUI`,
  the Textual TUI).
- `tests/` mirrors this layout (`test_agent/`, `test_tools/`, `test_symbols/`, `test_rag/`,
  `test_ui/`).

## Concurrency and shared orchestration files

`ReActLoop` deliberately dispatches tool calls within one model turn serially, not concurrently
(see the comment above the dispatch loop in `react_loop.py`), specifically so two mutating calls in
the same turn can't race on the same file. If you're changing dispatch, keep that property — the
individual tool classes have no locking of their own.

For the same reason, avoid parallel changes to shared orchestration files (`main.py`, agent
construction in `agent/__init__.py`, `react_loop.py`, the UI protocol) in flight at the same time;
merge conflicts there tend to be semantic, not textual.

## A note on the repository's history

Earlier in this project's life the working tree accumulated conflicting backup copies of key files
and untracked implementations that diverged from what was committed (see
`docs/finish-line/00-repository-recovery.md` in git history / `docs/archive/` for the full account
if you're curious). That's resolved — the tree is canonical and CI-verified now — but it's why this
doc is explicit about keeping docs, code, and tests in sync per PR rather than trusting a stale
snapshot.
