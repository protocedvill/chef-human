# Testing

The default suite is deterministic and does not contact a model server:

```bash
python -m pip install -e '.[dev]'
pytest
```

`pyproject.toml` excludes the `integration` marker by default. The same rule is stated explicitly in
CI as `pytest -m 'not integration'`, so a newly added live-backend test cannot silently contact a
developer's local Ollama instance during an ordinary run.

## Test taxonomy

| Marker | Purpose | Default suite |
|---|---|---|
| `unit` | Deterministic tests with mocked or in-process collaborators | Included |
| `ui` | Streaming, REPL, Rich, and Textual UI behavior | Included |
| `rag` | Experimental retrieval behavior; binary-package tests skip without the `rag` extra | Included when collectable |
| `indexing` | Experimental tree-sitter indexing behavior | Included when collectable |
| `integration_ollama` | Live Ollama requests using the configured model | Excluded |
| `integration_llamacpp` | Live llama.cpp requests using a local GGUF model | Excluded |

UI, RAG, and indexing are cross-cutting categories. A test in one of those categories is also a unit
unless it carries `integration`. Every integration test must declare exactly one live-backend marker;
collection fails if that rule is violated.

## Running optional and live tests

Install optional subsystems before selecting their full test groups:

```bash
python -m pip install -e '.[dev,indexing,rag]'
pytest -m 'rag or indexing'
```

Ollama integration tests are opt-in and check both the server and configured model before running:

```bash
ollama pull qwen3.8:27b
pytest -m integration_ollama
```

If a prerequisite is unavailable, pytest reports an actionable skip reason. The llama.cpp integration
contract uses `CHEF_TEST_LLAMACPP_MODEL` for a readable GGUF path:

```bash
python -m pip install -e '.[dev,llamacpp]'
CHEF_TEST_LLAMACPP_MODEL=/path/to/model.gguf pytest -m integration_llamacpp
```

There are currently no live llama.cpp cases; the marker and prerequisite contract reserve that test
lane without claiming unsupported coverage.

## Real-agent capability benchmark

Unit and integration tests validate components and backend protocols. The separate opt-in benchmark
drives the complete application through plan, tool use, file changes, validation, and finish against
disposable fixture workspaces:

```bash
python scripts/run_benchmark.py --through smoke
python scripts/run_benchmark.py --through all --keep-workspaces
```

It does not run in ordinary CI or download models. See [BENCHMARKS.md](BENCHMARKS.md) for cases,
scoring, machine-readable reports, and the safety boundary.

## Static and package checks

Run the same required checks as CI from a Python 3.12 development environment:

```bash
python -m ruff check .
python -m pyright --pythonpath "$(command -v python)"
python -m build
```

Pyright checks the authored `chef_human` package against Python 3.12 semantics. Tests remain guarded
by Ruff and pytest rather than being included in the static type contract: their extensive dynamic
mocks intentionally do not model every concrete protocol. Optional imports carry narrow
`reportMissingImports` ignores at the lazy import boundary; other Pyright diagnostics remain enabled.

CI exposes a final `Required CI gate` job which fails unless Ruff, Pyright, both unit-test matrix
entries, and the package build all pass. Once the workflow is proven on the public default branch,
that job should be selected as a required branch-protection check. A status badge is deliberately
deferred until then.
