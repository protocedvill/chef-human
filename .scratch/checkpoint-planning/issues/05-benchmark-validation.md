# 05: Benchmark validation

**What to build:** Ground-truth confirmation that the checkpoint mechanism actually fixes the
explore-only-collapse failure on `frontier/vague_feature_request_real_repo` — a green unit-test suite
alone is not sufficient evidence for this failure class (see `.scratch/vague-task-planning/map.md`'s own
history of a fix passing unit-level checks while still failing live runs).

**Blocked by:** 02, 03, 04

**Status:** ready-for-agent

- [ ] Re-run `python -m chef_human.benchmark --case vague_feature_request_real_repo --keep-workspaces`
      (with `CHEF_OLLAMA_THINK=true` and DEBUG logging, per this repo's standing working conventions).
- [ ] The run produces a nonempty `git status --porcelain` diff via a genuine explore-then-implement
      checkpoint flow (checkpoint declared, real exploration children, continuation call fires, spliced
      implementation steps executed) — not a collapse into pure exploration.
- [ ] The kept workspace and `agent.log` are inspected to confirm the checkpoint mechanism was actually
      exercised (not merely that the run happened to pass some other way).
- [ ] Findings (pass/fail, and if fail, the specific failure mode) are recorded in
      `.scratch/vague-task-planning/map.md`'s Decisions-so-far, cross-referencing this feature, since that
      map's own "not yet specified" blocker is directly relevant to whatever this run shows.
