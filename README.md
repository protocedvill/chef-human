# chef-human

**A ReAct coding agent that runs entirely on your own hardware — no API keys, no cloud LLM, no
data leaving your machine.**

chef-human plans, writes, edits, tests, and self-corrects against local models served by Ollama
or llama.cpp. It has its own file tools, a tree-sitter symbol index, RAG retrieval for large repos,
and a planner/retry loop that catches its own false completions before calling a task done.

> Linux-first learning prototype, not a production autonomous agent. It reduces common accidents
> (workspace confinement, a shell command blacklist, approval prompts, timeouts) but does not
> sandbox anything. Run it only in a disposable checkout or a sandbox you control — see
> [SAFETY.md](docs/SAFETY.md).

## Why local

Every other coding agent worth using assumes a frontier API in the loop. chef-human doesn't:
point it at `ollama serve` and a pulled model, and it plans and executes tasks with nothing
leaving the box. That constraint shapes the whole design — the context assembler, symbol index,
and step-verification logic all exist because local models need more scaffolding than a frontier
model to stay honest about what they've actually finished.

The core bet is **atomic decomposition**: a 27B local model can't hold a whole feature in its head
the way a frontier model can, but it doesn't have to. chef-human's planner breaks a difficult task
down into steps small and concrete enough that a local model can actually finish each one, verifies
each step against real disk state before moving on, and replans the moment a step turns out to be
too big or too vague — so the model is never asked to do more in one shot than it's capable of.

## What makes it different

- **Atomic task decomposition sized for local models.** The planner turns a difficult, high-level
  task into a sequence of byte-sized, independently-verifiable steps instead of one large prompt —
  the same strategy that makes hard problems tractable for a smaller model, not just a bigger one.
- **A ReAct loop that verifies its own steps**, not just its own say-so. Each plan step is checked
  against real disk state and tool exit codes first (deterministic guards), and falls back to an
  LLM verifier only when a step can't be checked objectively — closing the "the model claims done
  but nothing changed" failure mode that plagues local-model agents.
- **Retry → replan → escalate**, bounded. `RetryManager` decides per step whether to retry, ask the
  planner to revise the plan, or stop and hand control back to you — instead of looping forever or
  silently giving up.
- **Reversible by default.** Every mutating tool (`write`, `edit`, `patch`, `refactor`, `lint_fix`)
  records through a shared diff store, so `undo`/`redo` works from the CLI, REPL, or TUI — including
  atomic multi-file symbol renames.
- **Real code intelligence, not just embeddings.** Small repos get a tree-sitter symbol index and
  dependency graph for exact-match lookups (`goto_definition`, `reference_finder`,
  `refactor_symbol`); large repos fall back to FAISS-backed RAG retrieval automatically.
- **Benchmarked, not just demoed.** A 12-tier capability suite runs the agent as a real subprocess
  against fixed tasks in disposable workspaces, verified by an independent external command — never
  the agent's own say-so.

## Quick start

```bash
bash scripts/setup.sh
chef-human doctor           # confirms Python/config/backend/model/workspace are ready
chef-human "add input validation to the signup form"
```

Requires [Ollama](https://ollama.com) running locally (`ollama serve`) with a pulled model
(default `qwen3.6:27b`). llama.cpp is also supported as a backend.

```bash
chef-human repl                          # interactive REPL
chef-human run "some task" --headless --json
chef-human session list / show / delete / export
```

## Proven capability, tier by tier

The [capability benchmark](docs/BENCHMARKS.md) runs the real agent, as a subprocess, through
14 cases across 12 tiers of increasing difficulty — from "create and run one file" through
"diagnose a subtle stateful bug in an unfamiliar module" to "vague feature request against a real,
unfamiliar external codebase." Every case is verified externally (a real test suite or command
exit code), and several cases include protected files specifically to catch an agent that edits
its way to a false pass instead of solving the task.

```bash
python scripts/run_benchmark.py --through 4    # sanity through subtle-bug-repair tiers
python scripts/run_benchmark.py --through all  # the whole ladder
```

`qwen3.6:27b` and `qwen3.6:35b-a3b` currently pass the full agent-tier suite cleanly.

## How it works

```
                         ┌─────────────────────────┐
 CLI / REPL / TUI ─────► │      create_agent()      │
                         │  create_context_assembler │
                         └────────────┬─────────────┘
                                      │ wires together
        ┌─────────────┬──────────────┼──────────────┬─────────────┐
        ▼             ▼              ▼               ▼             ▼
   WorkspaceManager  Tokenizer  ContextAssembler  ToolRegistry  LLMBackend
                                      │                              │
                                      │      symbol index / RAG      │
                                      ▼                              ▼
                                 ReActLoop  ◄───────────────────  Planner
                                      │
                    parse tool calls → dispatch → verify step →
                       retry / replan / escalate
                                      │
                                      ▼
                                 ReActUI  (Debug / Streaming / REPL / Textual TUI / headless)
```

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full component-by-component breakdown.

## Documentation

- [Installation](docs/INSTALL.md) — setup, dependencies, troubleshooting
- [Usage](docs/USAGE.md) — CLI commands, API examples, configuration, testing
- [Architecture](docs/ARCHITECTURE.md) — component boundaries, request sequence, design decisions
- [Safety](docs/SAFETY.md) — threat model, guardrails, and what they don't cover
- [Development](docs/DEVELOPMENT.md) — environment setup and everyday dev workflow
- [Contributing](CONTRIBUTING.md) — scope, PR checklist, what not to do
- [Testing](docs/TESTING.md) — test taxonomy, optional extras, live-backend checks
- [Capability benchmark](docs/BENCHMARKS.md) — tiered real-agent tasks with external verification
- [Changelog](CHANGELOG.md) — release history
