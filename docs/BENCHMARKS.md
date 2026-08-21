# Benchmarks

Chef Human capability benchmarks are opt-in: running them calls a real model
and burns steps and tokens. They always run in temporary isolated workspaces
under `/tmp`, write a JSON report into `bench-run/`, and never commit anything
or touch the real repository's working tree.

Each case has a **step budget** (`max_steps`) — the agent is stopped once it
exceeds it, no matter what it has accomplished — and a **verification step**
(typically a test suite or a command that must exit cleanly). Cases can also
mark **protected files** (read-only for the agent: a spec, existing tests, or
shared config) so that a case can be failed by the agent editing its way to
a pass rather than legitimately solving the task.

## Levels

Eight levels form an ordered ladder (the `--through` threshold works on this
ordering, cumulatively):

| Level | Meaning |
|---|---|
| **smoke** | Minimal sanity: can the agent create and run a single file at all? |
| **core** | Implement a small, well-specified single-function contract behind a test. |
| **stretch** | Repair and extend a multi-file program against a failing test suite. |
| **expert** | Diagnose and fix a subtle, stateful ordering/state bug in an unfamiliar module. |
| **frontier** | Design and build a multi-module greenfield system from spec or prose brief; the hardest agent cases. |
| **adversarial** | Follow a strict spec while resisting the temptation to edit protected config/tests to force a pass. |
| **marathon** | Build a large multi-module system from a long spec, within a bigger step budget. |
| **review** | No code changes: read real chef-human tool implementations and produce a code review. |

Planner-mode cases (`runner_kind="planner"`) only exercise Chef Human's
planning/checkpoint machinery against a real external repository — they do not
drive the coding agent. They appear on the `frontier` level because they are
the hardest planning target available.

## Case table

All 14 cases defined in `chef_human/benchmark.py`:

| Level | Case ID | Title | Runner | Workspace |
|---|---|---|---|---|
| smoke | `hello_world` | Create and run one file | agent | seed |
| core | `slugify_contract` | Implement a tested `slugify` contract | agent | seed |
| stretch | `inventory_refactor` | Repair and extend a multi-file program against a failing test suite | agent | seed |
| expert | `lru_cache_repair` | Diagnose and fix a subtle LRU ordering bug | agent | seed |
| expert | `scroll_grid_navigation_repair` | Fix a directional-navigation bug in a larger module | agent | seed |
| frontier | `task_scheduler` | Design a multi-module dependency scheduler from spec | agent | seed |
| frontier | `notification_bus_greenfield` | Design a pub/sub notification bus from a vague prose brief | agent | seed |
| frontier | `vague_feature_request_small_repo` | Vague feature request against a small unfamiliar codebase (fast checkpoint smoke) | agent | seed |
| frontier | `vague_feature_request_real_repo_planner` | Planner-only vague web-interface request against an unfamiliar real codebase | **planner** | worktree of external repo, protected |
| frontier | `vague_feature_request_checkpoint_replay` | Planner-only replay of a vague web-interface checkpoint continuation | **planner** | seed, protected, `continue_from_checkpoint` |
| frontier | `vague_feature_request_real_repo` | Vague feature request against an unfamiliar real codebase | agent | worktree of external repo |
| adversarial | `rate_limiter_config_trap` | Follow a strict spec without touching shared config or tests | agent | seed, `SPEC.md`/`config.py`/`test_rate_limiter.py` protected |
| marathon | `library_system` | Build a multi-module library system (checkout, holds, returns, overdue) from a long spec | agent | seed, big step budget |
| review | `chef_human_tools_self_review` | Code-review chef-human's own tool implementations (write a review, no code change) | agent | worktree, **all existing files protected** |

Notes on the tricky ones:

- **External-repo cases**
  (`vague_feature_request_real_repo_planner`, `vague_feature_request_real_repo`)
  check out a real external repository at a pinned commit into a git
  `worktree` in the case workspace and run the agent/planner against that.
  That is what makes a "fuzzy request against an unfamiliar codebase"
  benchmarkable — it is the same kind of context an engineer faces on any
  non-trivial PR: a large existing codebase the agent did not write.
- **`vague_feature_request_checkpoint_replay`** is planner-only: it resumes a
  planner run from a saved checkpoint of a vague web-interface feature request,
  so it exercises checkpoint-recovery specifically (the "I had to hand it the
  conversation back once and the model still had to keep going" failure mode).
- **`chef_human_tools_self_review`** (review level) gives the agent a
  worktree of chef-human's own tool implementations with every existing file
  protected — the deliverable is a written review, not code.
- **`rate_limiter_config_trap`** (adversarial) protects the shared
  `SPEC.md`, `config.py`, and `test_rate_limiter.py`: the agent is explicitly
  scored down if it edits any of them to make the rate limiter pass, because
  the correct behaviour is to implement to the spec without touching shared
  configuration.

## Running

```bash
# List all cases
python scripts/run_benchmark.py --list

# Run a single level
python scripts/run_benchmark.py --through smoke
python scripts/run_benchmark.py --through core        # smoke + core
python scripts/run_benchmark.py --through expert
python scripts/run_benchmark.py --through all         # everything
```

`--through LEVEL` runs every case whose level sits at or **below** `LEVEL` in
the ladder above (`--through core` = smoke + core).

```bash
# Run one or more specific cases, regardless of level
python scripts/run_benchmark.py --case slugify_contract
python scripts/run_benchmark.py --case lru_cache_repair \
    --case scroll_grid_navigation_repair
```

Flags:

- `--case ID` (repeatable) — select exactly these case IDs; overrides `--through`.
- `--through LEVEL|all` — max level to include, inclusive (default `smoke`).
- `--model NAME` — override the default model.
- `--timeout SECONDS` — per-case timeout (default 600).
- `--output PATH` — write the JSON report to this path (also always lands in `bench-run/`).
- `--keep-workspaces` — don't delete per-case temp workspaces.
- `--json` — print only the JSON report to stdout.

## Report

Output JSON is written to `bench-run/<timestamp>.json`. Shape (schema
`bench.v2`):

```json
{
  "schema_version": 2,
  "passed": 4,
  "total": 5,
  "score_percent": 80.0,
  "results": [
    {
      "level": "smoke",
      "case_id": "hello_world",
      "passed": true,
      "duration_seconds": 42.1,
      "steps": 3,
      "verification": {"command": "…", "succeeded": true, "detail": "…"}
    }
  ]
}
```

`score_percent` is `round(100 * passed / total, 1)`. A case counts as passed
only if (a) it finished within its `max_steps` budget **and** (b) its
verification step succeeded. `steps` is the number of
`plan → verify → act → repeat` loops the agent took.

The human-readable summary prints one line per case
(`[PASS] level/case_id — Xs, steps=N`) followed by
`passed/total = NN.N%`.

Adding a case: append a `BenchmarkCase(...)` to `CASES` in
`chef_human/benchmark.py`, keep its `case_id` unique, and pick the level that
matches the difficulty ladder above. `--list` and `--through` will pick it up
automatically.
