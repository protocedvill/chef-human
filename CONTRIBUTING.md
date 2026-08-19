# Contributing

Chef Human is a personal/portfolio-scale prototype, not a project with a formal governance
process. Contributions are welcome, but keep scope small — see below.

## Before you start

1. Read [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) to set up the environment and run the test
   suite.
2. Read [CLAUDE.md](CLAUDE.md) for the architectural context an agent (human or otherwise) needs
   before touching the ReAct loop, tools, or context assembly — it documents non-obvious invariants
   (serialized tool dispatch, step-verification evidence keying, native vs. scraped tool-call
   parsing) that are easy to break silently.
3. For anything beyond a small fix, open an issue describing the change before writing code. This
   project intentionally favors "a reliable narrow prototype" over scope creep (see
   `docs/finish-line/README.md` for the stated product principles).

## Scope

Prefer one focused change per PR:

- One user-visible behavior or one bug fix, not a whole subsystem rewrite.
- Don't mix a refactor with a behavior change — make them separate PRs if both are warranted.
- Avoid touching shared orchestration files (`chef_human/main.py`, `chef_human/agent/__init__.py`,
  `chef_human/agent/react_loop.py`, `chef_human/ui/protocol.py`, `pyproject.toml`) alongside
  unrelated work; conflicts there tend to be semantic, not just textual.

## Checklist for a PR

- [ ] `pytest` passes (default marker set — see [docs/TESTING.md](docs/TESTING.md)).
- [ ] `ruff check .` and `pyright` are clean.
- [ ] New behavior has a test. Bug fixes should have a test that fails before the fix when
      practical.
- [ ] Any documented command, flag, or config option you touched is updated in the same PR
      (`docs/USAGE.md`, `docs/INSTALL.md`, `docs/SAFETY.md`, `docs/ARCHITECTURE.md`, `README.md`).
      Don't leave a doc claiming something is "not yet implemented" once it is.
- [ ] If you touched a benchmark case (`chef_human/benchmark.py`), validate it against its
      reference/seed in isolation, not just a live model run's pass/fail (see
      [docs/BENCHMARKS.md](docs/BENCHMARKS.md)).
- [ ] Commit message explains *why*, not just what changed.

## What not to do

- Don't add a feature flag or backwards-compatibility shim for something that can just be changed
  directly — there's no external user base depending on API stability yet.
- Don't add error handling for scenarios that can't happen; only validate at real boundaries (user
  input, external processes/APIs).
- Don't bypass the destructive-command guard or approval gate (`chef_human/tools/shell.py`,
  `_is_destructive_command` in `react_loop.py`) to make a task "just work" — if a legitimate command
  is being incorrectly flagged, fix the guard's logic and add a test, don't route around it.
- Don't weaken workspace confinement (`WorkspaceManager.is_within_workspace`) to make a tool more
  convenient. See [docs/SAFETY.md](docs/SAFETY.md) for why this boundary matters even though the
  tool isn't a full sandbox.

## Reporting bugs / requesting features

Open a GitHub issue. Include: what you ran, what you expected, what happened, and — for agent
behavior issues — the model you used and, if possible, a `--log-file` capture (see
[docs/USAGE.md](docs/USAGE.md)).
