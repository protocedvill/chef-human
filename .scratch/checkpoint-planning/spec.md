Status: ready-for-agent

## Problem Statement

chef-human's planner (`chef_human/agent/planner.py`) generates the entire plan tree up front, before any
execution happens — including implementation steps for parts of the task it has no grounding to plan
confidently. For a vague or unfamiliar-codebase task ("let's add a web interface to this project"
against a repo the planner has never seen), the model has to guess at implementation shape (what
framework, what directory layout, what existing code to wrap) at the exact moment it knows the least.

This produces two related, already-observed failure modes on the `frontier/vague_feature_request_real_repo`
benchmark case (see `.scratch/vague-task-planning/map.md`, an open, related wayfinder effort attacking the
same failure from a different angle — repo-context enrichment and mid-run replan-streak escalation,
neither of which proved sufficient alone):

1. **Explore-only collapse**: the whole plan degenerates into exploration steps (`ls`/`glob`/`read`) with
   no implementation steps at all, or so few that the loop auto-concludes "all complete" once the trivial
   exploration plan is exhausted, having built nothing.
2. **Drift on replan**: when a branch's rollup verification correctly rejects an under-specified
   implementation (see today's `_rollup_evidence` fix, `docs/adr/`-adjacent), `replan_subtree`'s fresh
   decomposition has no mechanism to preserve prior design decisions (e.g. a chosen directory name) from
   the branch it's replacing, so the agent redoes exploration and setup work under a different, arbitrary
   design each time it retries.

Both stem from the same root cause: the planner is forced to decide implementation shape before it has
the knowledge to do so, and has no first-class way to say "I need to go find out, then come back and plan
the rest for real."

## Solution

Add a **checkpoint**: a new declared `PlanNode` kind the planner can emit anywhere in the tree, meaning
"do this real work first (exploration, a partial implementation, whatever it takes to reduce
uncertainty), and only once that's genuinely done, use what was learned to plan what comes next — instead
of guessing now."

Mechanically, a checkpoint is a branch (reusing the existing `PlanNode`/`_expand_node`/atomicity-check/
rollup-verification machinery from `.scratch/planning-tree-adr/`) whose children are populated lazily, at
first-execution-time rather than generation-time. Nothing is speculatively planned past a checkpoint when
it's first declared. Once its own rollup verification passes — a real, verified unit of progress, not a
guess — a dedicated success-framed planning call generates the next steps using what the checkpoint's
children actually discovered (files read, each child's own reasoning, not just files written), and those
steps are spliced in as fresh siblings immediately following the checkpoint in its own parent's children
list, continuing the plan's flow rather than nesting deeper. Those spliced steps go through the same
classification pipeline as any other generated step, so a checkpoint can itself lead to another checkpoint
further down the line, recursively, anywhere a fresh round of uncertainty-reduction turns out to be
needed again (e.g. mid-implementation, not just after initial exploration).

The planner's system prompt is updated to actively teach this pattern: when it doesn't know how to
implement something, plan the steps that would gain the missing knowledge or context first, then end that
phase with a checkpoint — not place a checkpoint reflexively, and not chain checkpoints into
similar/redundant states with no real work done between them (backed by a structural harness guard, not
prompt wording alone, given this repo's documented history of prompt-only guidance being insufficient for
the small local models it targets).

## User Stories

1. As a user giving chef-human a vague, exploration-requiring task against an unfamiliar codebase, I want
   the planner to explore first and plan implementation second, so that implementation steps are grounded
   in what was actually found rather than guessed from the task text alone.
2. As a user, I want the planner to be able to declare a checkpoint anywhere in the plan tree — not only
   immediately after initial exploration — so that a step deep inside an implementation branch that turns
   out to need more investigation can also defer its own continuation.
3. As a user, I want a checkpoint's own sub-goal (the exploration or partial-implementation work) to be
   marked complete as soon as it is genuinely done, independent of whether the work it later spawns
   succeeds, so that a checkpoint is a real, individually-verifiable unit of progress and not an
   open-ended placeholder that never resolves.
4. As a user, I want the follow-up planning call that fires when a checkpoint completes to see what its
   children actually read and reasoned about — not just files they wrote — so that exploration-heavy
   checkpoints (which mostly read, not write) produce a genuinely informed continuation instead of one
   starved of the evidence the checkpoint existed to gather.
5. As a user, I want the checkpoint's continuation steps to appear as new siblings immediately after the
   checkpoint in its own parent's step sequence, not nested as further children underneath it, so that the
   plan's linear flow at that level continues naturally instead of the checkpoint's own sub-goal expanding
   to swallow the rest of the plan.
6. As a user, I want the checkpoint node itself to remain in the tree (marked complete) after it fires,
   rather than being removed or replaced, so that the plan retains a visible record of what was explored
   and when, for later steps' context and for a human reviewing the run.
7. As a user, I want checkpoints to be declared only by explicit planner output (a new step `"type"`
   value), not inferred by the harness from step wording, so that checkpoint declaration is as reliable as
   the existing leaf/branch/uncertain classification the planner already does, and not exposed to the
   same brittle-text-matching failure class today's `continues_node_id` fix exists to move away from.
8. As a user, I want a checkpoint's own decomposition to be required to include at least one non-checkpoint
   step, so that a degenerate pass-through chain (checkpoint whose only child is immediately another
   checkpoint, forever) can't happen even if the model tries to emit one.
9. As a user, I want chained/nested checkpoints to be protected against runaway recursion independently
   from the existing generation-time `_ConvergenceTracker`, so that a pathological chain triggered across
   multiple execution turns (a different call shape than `_ConvergenceTracker` was built for) still gets
   bounded.
10. As a user, I want the planner's system prompt to explicitly teach "gain the missing knowledge first,
    then checkpoint" rather than leaving checkpoint usage undiscovered or used reflexively, so that this
    mechanism actually gets reached for on the exact vague-task failure mode that motivated it, instead of
    remaining an unused capability.
11. As a user, I want continuation steps spliced in by a checkpoint's follow-up planning call to always be
    treated as brand-new nodes (no identity-continuation tagging via `continues_node_id`), so that the
    mechanism stays simple and doesn't need to solve a node-collision problem that can't structurally occur
    given nothing is speculatively planned past a checkpoint in the first place.
12. As a user, I want this to ship as an explicitly planner-declared mechanism only for now (no reactive
    harness-injected checkpoints on stuck branches), so that the feature's scope stays bounded to what's
    already been designed, with reactive injection left as a clearly separate future decision rather than
    bundled in and risking the whole feature stalling on its hardest edge case.
13. As a developer maintaining this codebase, I want checkpoint orchestration (when to fire, how to splice,
    how to count chains, how to enforce the pass-through guard) tested at the `ReActLoop` rollup-completion
    seam with a mocked `Planner`, and checkpoint classification/continuation-prompt construction tested at
    the `Planner` seam with a mocked LLM backend, so that the two genuinely separate responsibilities
    (orchestration vs. LLM-call shaping) stay independently testable the way `update_plan` and its callers
    already are.
14. As a developer, I want the `vague_feature_request_real_repo` benchmark case re-run as ground truth once
    both seams are unit-tested, so that a green unit-test suite is verified against the actual failure mode
    that motivated this feature, not trusted on its own.

## Implementation Decisions

- **New declared node kind**: the planner's structured JSON step output gains a `"type": "checkpoint"`
  value alongside the existing `"leaf"`/`"branch"` values (and the existing `"uncertain"` flag).
  `PlanNode` gains a field recording this declared kind.
- **Lazy children, reusing existing decomposition machinery**: a checkpoint's children are not populated
  at generation time. The first time a checkpoint becomes the node execution is about to work on, it is
  run through the existing `_expand_node`/atomicity-check decomposition pipeline (same as an ordinary
  branch), just triggered at first-execution-time instead of generation-time. Nothing is speculatively
  generated past a checkpoint when it is first declared.
- **Completion semantics**: a checkpoint's own rollup verification (existing `_process_rollups` machinery,
  including today's `_rollup_evidence` fix) governs when its own sub-goal is considered done. On success,
  the checkpoint branch is marked `completed` immediately — its completion is independent of whether the
  continuation steps it spawns later succeed.
- **New planner method for the continuation call**: `Planner` gains a dedicated method (distinct from
  `update_plan`, which is framed around failure recovery) for the success-framed "what comes next, given
  what was learned" call — working name `continue_from_checkpoint`. Its prompt does not reuse
  `update_plan`'s "the previous plan had a failure" framing.
- **Evidence for the continuation call**: broader than today's `_rollup_evidence` fix (which only surfaces
  written-file contents). The continuation call's context includes files *read* by the checkpoint's
  children (via the existing `_read_file_names` mechanism) and each child's own recorded
  reasoning/evidence — not only files written — since checkpoint children are typically exploration-heavy
  (reads), and write-only evidence would starve the continuation call of exactly the information the
  checkpoint exists to gather.
- **Splice mechanics**: the continuation call's output steps are inserted as new siblings immediately
  following the checkpoint node in the checkpoint's own parent's children list. The checkpoint node itself
  is not removed or replaced. This is orchestrated from `ReActLoop`'s rollup-completion path
  (`_process_rollups`, extended), not from within `Planner`.
- **Node identity**: every spliced continuation step is a brand-new `PlanNode` (fresh `node_id`). The
  `continues_node_id` mechanism (from `docs/adr/0001-evidence-carry-forward-across-whole-plan-replan.md`)
  is deliberately not used here — nothing pre-exists past a checkpoint for a spliced step to continue the
  identity of, by construction.
- **Rollup scope after splice**: spliced siblings become real children of the checkpoint's parent branch
  (if any), so that branch's own rollup naturally waits on them too, via existing `ready_rollup_branches`
  semantics — no new rollup-scoping mechanism needed.
- **Pipeline reuse for spliced steps**: spliced continuation steps go through the same atomicity-check and
  `_classify_children` pipeline as any other generated step, so a spliced step can itself be classified as
  a further checkpoint (enabling arbitrarily deep/chained checkpoints, anywhere in the tree, not only a
  root-level explore-then-implement phase).
- **Chain-progress counter**: a dedicated counter (separate from `_ConvergenceTracker`, which is scoped to
  one synchronous generation-time recursion and not the right shape for a mechanism triggered across
  multiple execution turns via `_process_rollups`) tracks checkpoint-chain progress, similar in spirit to
  the existing `_subtree_replan_streak` mechanism elsewhere in `react_loop.py`.
- **Structural pass-through guard**: the harness rejects (or forces reclassification of) a checkpoint whose
  own decomposition consists of exactly one child that is itself another checkpoint with no other work —
  enforced structurally, not left to prompt wording alone, following this codebase's established pattern
  (e.g. the existing single-child-breakdown guard that forces `requested_branch = False` on a degenerate
  atomicity-reclassification).
- **System prompt update**: `PLANNER_SYSTEM_PROMPT` (and/or the relevant per-call prompt in `planner.py`)
  gains explicit guidance: when the planner doesn't know how to implement something, it should plan the
  steps that would gain the missing knowledge/context first, then end that phase with a checkpoint —
  and should not place checkpoints reflexively or chain them into similar/redundant states with no
  concrete work done between them.
- **Scope of declaration**: checkpoints are declared only by the planner's own structured output, at
  generation time (including recursively, via the classification pipeline for spliced steps). No reactive,
  harness-injected checkpoint creation on a stuck/repeatedly-replanned branch is in scope for this spec.

## Testing Decisions

- Tests must exercise externally observable behavior (plan shape after a checkpoint fires, node
  statuses, what gets spliced where) rather than internal call sequencing, consistent with this codebase's
  existing planner/react_loop test style.
- **`ReActLoop` seam** (primary): extend `TestRollupVerification`-style tests in `tests/test_agent/test_react_loop.py`
  (a class already exists there, extended earlier today for the `_rollup_evidence` fix) to cover: a
  checkpoint branch's rollup succeeding triggers a call into the mocked `Planner`'s continuation method;
  the returned steps are spliced as siblings in the correct position; the checkpoint node itself ends up
  `completed`; the parent branch's own rollup correctly waits on the newly spliced siblings; the chain
  counter advances across repeated checkpoint completions; the structural pass-through guard rejects a
  single-checkpoint-child decomposition. Use `_make_mock_planner()` / `_make_mock_backend()`, the existing
  helpers in that file.
- **`Planner` seam** (necessary second seam): extend `tests/test_agent/test_planner.py` (mirroring the
  existing `TestAtomicityCheck`, `TestUpdatePlan`, and today's new `TestUpdatePlan` `continues_node_id`
  test for style) to cover: `_parse_steps`/`_normalize_steps` correctly carrying a `"type": "checkpoint"`
  value through; `_expand_node`'s lazy-children trigger path; the new continuation method's prompt
  including files-read and children's-reasoning evidence (not just written-file contents); the prompt
  being success-framed rather than reusing `update_plan`'s failure-framed text.
- **Validation, not a unit-test seam**: once both seams are green, re-run
  `python -m chef_human.benchmark --case vague_feature_request_real_repo --keep-workspaces` (with
  `CHEF_OLLAMA_THINK=true` and DEBUG logging, per this repo's standing working conventions) as ground
  truth, per this repo's documented practice of never trusting a unit test alone over a live benchmark
  run for this exact failure class (see `.scratch/vague-task-planning/map.md`'s own history of enrichment
  passing unit-level checks while still failing live runs).
- Prior art: `tests/test_agent/test_react_loop.py::TestRollupVerification` (rollup orchestration seam,
  just extended today) and `tests/test_agent/test_planner.py::TestUpdatePlan`/`TestAtomicityCheck` (planner
  LLM-call-shaping seam) are the direct structural precedents for both seams in this spec.

## Out of Scope

- Reactive, harness-injected checkpoint creation on a branch that keeps failing/replanning without ever
  having been declared a checkpoint by the planner (a separate, harder mechanism — noted as a candidate
  future decision, not bundled here).
- Using `continues_node_id`-style identity continuation for spliced checkpoint-continuation steps.
- Any change to `update_plan`'s existing failure-recovery framing or behavior — this spec adds a new,
  separate method rather than modifying it.
- Resolving `.scratch/vague-task-planning/map.md`'s still-open "not yet specified" question about
  run-to-run planning variance in repo-context enrichment — that map should be revisited once this ships,
  not folded into this spec.
- A dedicated new benchmark case specifically engineered to require checkpointing; validation reuses the
  existing `vague_feature_request_real_repo` case.

## Further Notes

- This composes directly with `.scratch/planning-tree-adr/` and `.scratch/planning-tree/`'s
  `PlanNode`/rollup-verification/evidence-propagation machinery (tickets 01-10, mostly `done`) rather than
  building anything in parallel to it — a checkpoint is a `PlanNode` branch, full stop, with new behavior
  attached at specific points (lazy children, post-rollup splice), not a new tree structure.
- `.scratch/vague-task-planning/map.md` is an open, related wayfinder effort attacking the same underlying
  benchmark failure from a different angle (context enrichment + mid-run replan escalation). That map's
  own notes conclude those mechanisms are necessary but not sufficient — this spec is a structurally
  different, composable fix for the same failure class, not a replacement for that map's completed
  tickets.
- Today's two live diagnosed-and-fixed bugs are directly relevant prior art for *why* several decisions
  above lean structural over prompt-only: the `continues_node_id` fix
  (`docs/adr/0001-evidence-carry-forward-across-whole-plan-replan.md`) and the `_rollup_evidence`
  files-written fallback (`chef_human/agent/react_loop.py`, undocumented in an ADR — diagnosed via
  `tests/test_agent/test_react_loop.py::TestRollupVerification::test_rollup_evidence_falls_back_to_files_children_actually_wrote`)
  were both cases of the local model needing a harness-enforced mechanism, not just better prompt wording,
  to behave reliably.
