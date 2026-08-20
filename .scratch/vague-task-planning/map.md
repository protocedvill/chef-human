# Fixing explore-only planning on vague real-repo tasks

Status: open

## Destination

Make the `vague_feature_request_real_repo` benchmark case pass: given "Let's add a web interface to
this project" against a real, unfamiliar repo (ubertooth) with no scope hints, the agent must produce a
nonempty `git status --porcelain` diff (i.e. actually write something) instead of collapsing the whole
run into pure exploration (ls/glob/grep/read-only) and finishing with nothing changed.

This map's Notes override the usual plan-don't-do default: it carries execution through to a green
benchmark run, not just a design spec. Tickets do real work, not just decide.

## Notes

- Domain: chef-human's agent planning subsystem (`chef_human/agent/planner.py`, `react_loop.py`). See
  `CLAUDE.md`'s "Step verification" section and `docs/adr/0001-evidence-carry-forward-across-whole-plan-replan.md`
  for prior false-escalation/replan-evidence history in this area.
- Ground truth check: `python -m chef_human.benchmark --case vague_feature_request_real_repo --keep-workspaces`
  (always with `--keep-workspaces` and DEBUG logging per this repo's working conventions).
- Grounding facts already gathered (don't re-derive):
  - `ReActLoop._get_repo_context()` (`react_loop.py` ~1816) currently returns only
    `self._context._repo_map.generate_tree()[:1000]` — a bare truncated tree, no language breakdown, no
    doc excerpts, no symbol info — fed straight into `Planner.generate_plan`'s `repo_context` param,
    which already threads it into the planning system message (`planner.py` ~624-632).
  - ubertooth (the real repo behind this benchmark case) has 407 files, under `settings.max_index_files`
    (500) — it takes the **symbol path**, not RAG (`create_tool_registry` builds a real `SymbolIndex` +
    `DependencyGraph` for it). Enrichment can draw on both, not just `RepoMap`.
  - `planning_facts` (`_collect_planning_facts`) is a separate, narrowly-scoped mechanism: file-existence
    yes/no checks for task-referenced paths only. Settled: it stays exactly that scope; all enrichment
    content is prose and belongs in `repo_context`, not `planning_facts`.
  - `max_context_tokens` is 32768; `repo_context` shares that budget with conversation history and file
    context, so enrichment must be capped (low hundreds of tokens), not open-ended.
  - Two composable ideas were on the table going in: (1) deterministic repo-context enrichment
    (`RepoMap`/`SymbolIndex`/`FileContextManager` → richer `repo_context`), and (2) an explicit mid-run
    "exploration budget exhausted → real replan" phase transition in `ReActLoop`. Settled ordering:
    enrichment first, benchmark it, and only design/build (2) if the benchmark still fails — cheaper
    diagnostic ordering that produces evidence instead of guessing which theory (grounding problem vs.
    control-flow problem) is right.
  - If mid-run replan does turn out to be needed, it must be built as a **caller** of the subtree-replan
    machinery already decided in `docs/adr/0001-evidence-carry-forward-across-whole-plan-replan.md` and
    `.scratch/planning-tree/issues/04-evidence-propagation-and-subtree-replan.md` (status
    `ready-for-agent`, not yet implemented) — not a parallel replan mechanism. If ticket 04 lands first,
    call into it; do not duplicate its evidence-preserving replan logic.
  - Related but orthogonal effort: `.scratch/planning-tree-adr/` and `.scratch/planning-tree/` (recursive
    `PlanNode` tree, mostly done) — this map composes with that machinery, doesn't restate it.

## Decisions so far

- Repo-context enrichment (ticket 01) is implemented, tested, and confirmed necessary: without it the
  planner's top-level plan for `vague_feature_request_real_repo` was generic; with it, the plan's steps
  are genuinely repo-specific (C HTTP server, CMakeLists integration, Ubertooth device API) and include
  real implementation work, not pure exploration.
- Enrichment alone is **not sufficient**: the benchmark case still fails (ticket 02). Root cause is
  execution-derailment, not a grounding gap — the run gets stuck replanning one exploration sub-branch
  repeatedly and exhausts its step budget before ever reaching the plan's implementation steps.
  Idea 1 (mid-run "exploration budget exhausted → replan") is confirmed necessary (ticket 03), not
  ruled out.
- Mid-run replan implementation is blocked on `.scratch/planning-tree/issues/04-evidence-propagation-and-subtree-replan.md`
  landing (itself blocked on its own "03 Rollup verification" ticket) — see ticket 03's Answer for the
  operational shape it should take once unblocked. Not ticketed standalone here yet since it would sit
  with nothing to do until 04 lands.
- Found and fixed, as a side effect of getting real ground truth for ticket 02: a benchmark-harness bug
  in `chef_human/benchmark.py::run_case` where a relative `workspace` path was resolved inconsistently
  between `_create_worktree` (against `source_repo`'s cwd) and the `--workspace` CLI arg (against the
  benchmark process's own cwd), silently running worktree-kind cases against external repos (like
  ubertooth) in an empty, non-git directory instead of the real checkout. Fixed by resolving `workspace`
  once, up front. This affects every `workspace_kind="worktree"` case with a non-default `source_repo`,
  not just this one.
- `.scratch/planning-tree/issues/04-evidence-propagation-and-subtree-replan.md` landed (found already
  fully implemented and tested when checked, 2026-08-20 -- no code changes needed, just closed as `done`).
  That unblocked graduating the mid-run-replan idea into a real ticket:
  `.scratch/planning-tree/issues/10-subtree-replan-streak-escalation.md`, built and unit-tested per the
  operational shape ticket 03 recorded (a per-branch streak of progress-free subtree replans, widening
  the replan target to the whole branch once the streak crosses a threshold).
- Ticket 10 does **not** turn this benchmark case green. A fresh benchmark run (2026-08-20,
  `qwen3.6:35b-a3b`, think-mode off) shows the case failing for a *different* reason than ticket 02/03
  diagnosed: the plan generated this time was a 7-leaf, entirely explore-only plan that finished in 8
  steps with zero replans (every step individually verified complete first try) -- never a stuck-replan
  loop at all, so ticket 10's mechanism never had anything to fire on. This is the *original*
  explore-only-plan failure mode ticket 01 (repo-context enrichment) was supposed to have already fixed.
  Either enrichment's fix is not reliable/deterministic across runs, or something about a fresh worktree
  checkout differs from the run ticket 01/02's evidence was captured against -- not yet isolated further.
  See ticket 10's Answer for the run's specifics (kept workspace: `benchmark-runs/20260820-024854/`).

## Not yet specified

- Why repo-context enrichment (ticket 01, `done`) didn't produce a repo-specific, implementation-bearing
  plan on the 2026-08-20 run the way ticket 02's evidence showed it did previously -- is this run-to-run
  planner variance (despite temperature=0.0, the multi-call iterative decomposition has many LLM calls
  that could each individually drift) or a real regression/environment difference? This is the actual
  remaining blocker on this map's destination now, not mid-run replan (ticket 10 is done and validated
  in isolation, just not sufficient alone). Not yet ticketed -- needs at least one more benchmark run
  (maybe two, to check for run-to-run variance) with the *generated plan itself* logged before it's worth
  a design ticket; currently `agent.log` only logs step-level verification, not the plan generation calls
  or the rendered tree, which made this run harder to diagnose than it should have been. Also worth
  ticketing: log `Planner.format_full_tree`'s output right after `generate_plan` returns, at DEBUG, so a
  future kept workspace shows the actual plan without having to reconstruct it from step verification log
  lines the way this investigation had to.

## Out of scope

