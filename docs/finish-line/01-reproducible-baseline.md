# Phase 1: Reproducible Baseline

## Goal

Produce a clean install and trustworthy automated signal before changing the user experience.

## Supported baseline

- Python: 3.12 and 3.13.
- Primary backend: Ollama.
- Default tests: no running model server, GPU, sentence-transformers, FAISS, or tree-sitter grammar is
  required.
- Integration tests: explicitly selected and skipped with a useful reason when prerequisites are
  absent.
- Target platform: Linux first unless the owner chooses a wider matrix.

## Work package 1A — Packaging and environments

Files: `pyproject.toml`, `scripts/setup.sh`, new CI workflow, optional lock/constraints files if chosen.

Tasks:

1. Make extras reflect runtime boundaries: `dev`, `textual` only if the TUI remains optional,
   `indexing`, `rag`, and `llamacpp`. Anything imported by the default command path must be a core
   dependency or lazily imported behind a clear diagnostic.
2. Replace the setup script's silent/partial dependency fallback with an explicit failure and recovery
   instructions. A setup command must never claim success after required dependencies failed.
3. Add `python -m chef_human.main` support through `chef_human/__main__.py` if absent.
4. Validate build metadata with a wheel/sdist build and install the wheel into a fresh environment.

Acceptance criteria:

- `pip install -e '.[dev]'` succeeds in a clean Python 3.12 and 3.13 environment.
- The built wheel installs and `chef-human --help` runs outside the repository.
- Base install imports do not require optional RAG/indexing/llama.cpp packages.

## Work package 1B — Test taxonomy and green suite

Tasks:

1. Reconcile source and tests from Phase 0, then categorize tests as unit, integration-Ollama,
   integration-llama.cpp, optional-RAG, and UI.
2. Mark live Ollama tests correctly. Tests currently named as ordinary tests must not contact
   `localhost:11434` during the default suite.
3. Make optional dependency tests use `pytest.importorskip` or an equivalent marker before importing
   binary packages. This prevents collection crashes such as the observed Python 3.15/NumPy failure.
4. Fix failures by compatibility cluster, starting with imports/protocols, then agent loop, undo/redo,
   workspace discovery, UI, and optional subsystems.
5. Add a small command-surface smoke test for every advertised CLI command.

Acceptance criteria:

- `pytest -m 'not integration'` passes from a base dev install.
- Integration tests are opt-in and report prerequisites clearly.
- No test relies on `/tmp` lacking a parent project marker; workspace tests isolate discovery inputs.
- The test summary in README/CI is generated or conservatively phrased, never hand-maintained as an
  exact count.

## Work package 1C — Static checks and CI

Tasks:

1. Make Ruff pass on authored source and tests; exclude archives rather than suppressing arbitrary
   errors.
2. Configure Pyright for optional imports and the supported Python version. Fix real type errors in
   protocols and async result handling; document narrow ignores.
3. Add CI jobs for lint/type check, Python 3.12 unit tests, Python 3.13 unit tests, and package build.
4. Add dependency caching and reasonable timeouts. Do not download models in ordinary CI.
5. Add a status badge only after the workflow is stable on the public default branch.

Acceptance criteria:

- Ruff, Pyright, unit tests, and package build pass locally and in CI.
- CI starts from the checked-in repository with no generated files or hidden local configuration.
- A failure in any required job blocks the release checklist.

## Work package 1D — Highest-risk correctness fixes

Use `code_review_2026-07-03.md` as input, then re-verify every finding against the recovered code.

Priority candidates:

1. Configuration overrides must be injected, not implemented by rebinding a module singleton already
   imported elsewhere.
2. Planner verification failures must not crash or loop forever.
3. Read-before-write policy must cover every mutating tool and define same-turn behavior.
4. Plan completion must account for failed/skipped steps.
5. Undo/redo must represent an edit or multi-file refactor as a coherent transaction.
6. Concurrent tool calls must not race when they can mutate the same path.
7. Shell checks must be described as guardrails; do not imply they provide process isolation.

Acceptance criteria:

- Each fix begins with a failing regression test.
- Safety policy tests cover all registered mutating tools through one shared classification.
- No correctness fix is accepted solely because an LLM-backed manual run happened to succeed.

