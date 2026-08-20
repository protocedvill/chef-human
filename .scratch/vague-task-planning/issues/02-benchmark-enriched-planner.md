Type: task
Status: closed
Blocked by: 01

## Question

Run `python -m chef_human.benchmark --case vague_feature_request_real_repo --keep-workspaces` (with
DEBUG logging) against the enriched planner from Repo-context enrichment for the planning prompt, and
record the outcome as fact: did the agent produce a nonempty `git status --porcelain -- .
':(exclude).chef-human'` diff (i.e. write something), or did it still collapse into pure exploration?

If it still fails, capture *why* from the kept workspace/logs — specifically whether the plan itself
still reads as explore-only (grounding problem, enrichment didn't help) or whether the plan now includes
real implementation steps but something else derails execution (a different, not-yet-understood
failure) — since that distinction is exactly what Deciding whether mid-run replan is still needed needs
to route on.

## Answer

Ran `python -m chef_human.benchmark --case vague_feature_request_real_repo --keep-workspaces` twice.

The first run (`benchmark-runs/20260820-015725/`) reported `git status --porcelain` empty but was
invalid ground truth: it surfaced a pre-existing harness bug, not a planner result. `run_case()`
(`chef_human/benchmark.py`) passed a **relative** `workspace` path into `_create_worktree`, which runs
`git worktree add` with `cwd=source_repo`. For `source_repo="~/ubertooth"` that resolved the relative
path against ubertooth's directory, creating the actual worktree at
`~/ubertooth/benchmark-runs/.../vague_feature_request_real_repo` — while `--workspace
str(workspace.resolve())`, later in the same function, resolved the *same relative Path* against the
benchmark process's own cwd instead, pointing the agent at a different, plain (non-git) directory that
had only been `mkdir`'d, never checked out. The agent ran there against an empty folder with no
ubertooth content at all, asked a clarifying question, and wrote generic FastAPI boilerplate — not a
real test of the enriched planner against an unfamiliar codebase. (This bug only stayed hidden this long
because every other worktree-kind case defaults `source_repo` to chef-human itself, whose repo root
happens to coincide with the benchmark process's typical invocation cwd, masking the divergence.)

Fixed by resolving `workspace = workspace.resolve()` at the top of `run_case()` before it's used by
either `_create_worktree` or the `--workspace` arg, so both paths agree
(`chef_human/benchmark.py::run_case`).

Second run (`benchmark-runs/20260820-015956/`), against the real ubertooth worktree this time: still
**FAIL** — `git status --porcelain` is empty (0/1), agent hit max_steps (40) exceeded.

Diagnosis, from the kept workspace's `.chef-human/agent.log`:

- **Grounding did improve.** The enriched `repo_context` (ticket 01) produced a plan whose top-level
  steps are genuinely scoped to the real project — "Implement a minimal HTTP server in **C** that
  communicates with the Ubertooth device", "Add build system integration (Makefile/**CMakeLists.txt**)",
  "Integrate the Ubertooth device communication API into the HTTP server" — not a generic web-app
  boilerplate plan. This is not the explore-only-plan failure the map's Notes anticipated as the likely
  outcome; enrichment did its job.
- **Execution derailed instead.** The planner's atomicity check recursively explodes the first
  (exploration) top-level step into several fine-grained leaves ("Read docs/source/software.rst...",
  etc.). The agent then gets stuck: it repeatedly calls `finish` reporting the same leaf complete
  ("Step 1 complete: Read docs/source/software.rst...") — that finish is rejected, triggering
  "Replanning after repeated failures" (visible in the log at steps 7, 12, 17, 22, 27, 32) — and the
  *same* leaf goal recurs after each replan. This loop consumes the entire 40-step budget purely
  re-exploring/re-verifying one exploration sub-branch; the plan's implementation branches (HTTP server,
  build integration, device API) are never reached, so `git status --porcelain` stays empty even though
  the plan itself called for real writes.

This is the "real-plan-but-execution-derailed" branch the map's ticket flagged, not the "plan still reads
as explore-only" branch — routes ticket 03 toward scoping mid-run replan / exploration-budget work rather
than further grounding/enrichment work.
