# 10: Subtree replan-streak escalation ("exploration budget exhausted")

**What to build:** A per-subtree counter of consecutive subtree replans that make no progress, so a
branch whose leaves keep getting individually replanned into equivalent-but-differently-worded leaves
(each replan minting a fresh `node_id`, so `RetryManager`'s per-`node_id` cap never accumulates against
it) eventually widens the replan scope to the whole containing branch instead of looping forever on one
leaf at a time. Graduated from `.scratch/vague-task-planning/issues/03-decide-mid-run-replan-necessity.md`
now that its blocker (`.scratch/planning-tree/issues/04-evidence-propagation-and-subtree-replan.md`,
subtree-scoped replan + evidence propagation) has landed.

**Root cause this addresses:** `vague_feature_request_real_repo` burns its entire step budget stuck
replanning one exploration sub-branch repeatedly, never reaching the plan's implementation steps (see
ticket 03's Answer). `_replan_failing_node` already scopes each individual replan correctly (ticket 04),
but nothing tracks that the *same branch* keeps needing a replan turn after turn — each replanned leaf is
a structurally new node, so `RetryManager`'s existing 1-replan/5-consecutive-failure cap resets clean
every time and the branch never runs out of budget on its own.

**Blocked by:** 04 (done)

**Status:** done (implemented + unit-tested; did **not** turn the benchmark case green -- see Answer)

- [x] `ReActLoop` tracks a streak, keyed by the nearest branch ancestor's `node_id`, of consecutive
      subtree replans scoped to a descendant of that branch (`_widen_stuck_replan_target`,
      `self._subtree_replan_streak` in `react_loop.py`)
- [x] The streak resets to zero when any leaf or branch under that ancestor is marked complete (real
      progress), not just when a replan happens to succeed (`_reset_subtree_replan_streak`, called from
      `_process_rollups` for both the just-completed leaf and any branch its rollup verification passes)
- [x] Once the streak crosses `_SUBTREE_REPLAN_STREAK_LIMIT` (3), the next replan widens its target from
      the individual failing leaf/branch to the ancestor branch itself -- discarding and regenerating the
      whole branch's children, not just the one leaf that most recently failed
- [x] The widened replan still goes through the existing subtree-replan machinery from ticket 04
      (`_discard_subtree_evidence`/`Planner.replan_subtree`/`_rebuild_ancestor_evidence`), scoped to the
      ancestor node -- no parallel replan path (`_replan_failing_node` calls `_widen_stuck_replan_target`
      before doing anything else, then proceeds exactly as before with whatever target it returns)
- [x] A leaf directly under the plan root (no real branch ancestor to widen to) is left to the existing
      per-node retry/replan/escalate behavior unchanged (`_widen_stuck_replan_target` returns `target`
      unchanged whenever `target.parent is None or target.parent is plan.root`)
- [ ] Benchmark: `python -m chef_human.benchmark --case vague_feature_request_real_repo --keep-workspaces`
      still fails -- **for a different reason than this ticket addresses**, see Answer below. Leaving this
      box unchecked rather than marking it done under a technicality.

## Answer (post-implementation benchmark run)

Ran the benchmark (`qwen3.6:35b-a3b`, think-mode **off** per
`project_thinkmode_quality_drop` memory -- think-mode is confirmed to cause false step-escalation on
this suite, so a think-mode run's pass/fail isn't trustworthy evidence either way). Result: still FAIL,
but the failure mode is not the one this ticket targets.

This run generated a 7-leaf, entirely explore-only plan (`ls host/`, `ls firmware/`, `ls tools/`, two
`grep`s, two `cat`s of doc files) and finished in 8 steps total with **zero replans** -- every step
individually verified `complete` on the first try (see `.chef-human/agent.log` in the kept workspace,
`benchmark-runs/20260820-024854/`: `Step 'ls host/' verification verdict: complete`, etc., then `Task
finished via finish tool after 8 step(s)`). `git status --porcelain` in the workspace shows only the
untracked `.chef-human/` dir -- no real file ever changed.

This is the *original* explore-only-plan failure mode that
`.scratch/vague-task-planning/issues/01-repo-context-enrichment.md` (status `done`) was supposed to have
fixed, and that the map's "Decisions so far" claims was confirmed fixed (repo-context enrichment made the
plan "genuinely repo-specific ... include real implementation work"). It did not reproduce that way here
-- the plan this run generated never got past exploration, and never even reached a point where the
subtree-replan-streak mechanism in this ticket could matter (there were no failures/replans to
streak-count in the first place). So this ticket's fix is real, tested, and correctly implements what it
set out to build, but it is not sufficient by itself to turn this benchmark case green -- the currently
blocking failure mode has reverted to (or was always intermittently) the explore-only-plan problem, not
the stuck-replan-loop problem ticket 03/04 diagnosed last time.

Not chasing this further within this ticket's scope -- it's a different root cause (plan generation
producing a too-short, purely-investigative plan for this particular run) that belongs with
`.scratch/vague-task-planning/issues/01-repo-context-enrichment.md`'s territory (repo-context
enrichment reliability/determinism), not this one (mid-execution replan escalation). Recorded in the
map's Decisions so the next agent picking this up doesn't have to re-derive it from a fresh benchmark
run.
