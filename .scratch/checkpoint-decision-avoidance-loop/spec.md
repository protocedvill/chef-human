Status: done

## Problem Statement

The checkpoint mechanism (`docs/adr/` from the `checkpoint-planning` effort) lets the planner defer
implementation decisions until real exploration has reduced uncertainty. In practice, on the
`frontier/vague_feature_request_real_repo` benchmark case, it instead got stuck: a top-level checkpoint's
rollup verification kept coming back `not_complete`, and each retry produced *another* checkpoint instead
of an actual decision — three nested checkpoints deep, each re-deriving the same codebase structure from
scratch, until the run hit its 600s timeout having written zero lines of implementation.

Diagnosed via `/diagnosing-bugs` and grilled into a design via `/grill-with-docs`; the decision and its
reasoning are recorded in `docs/adr/0002-checkpoint-decision-avoidance-loop.md`. Root causes:

1. The existing anti-spiral safety net (`_bound_checkpoint_chain`, caps 3 consecutive checkpoint-only
   splices) only fires on the rollup-*success* continuation path. The incident's actual path — repeated
   rollup *failure* triggering `replan_subtree` on the same checkpoint node — never touches it.
2. A rollup-triggered replan discards the checkpoint's accumulated evidence (files read, findings)
   entirely; the replan prompt only sees the rejection *reason text*, not what was actually found, so
   each retry re-explores from zero.
3. The existing stuck-replan escalation (`_widen_stuck_replan_target`) silently no-ops for any node whose
   parent is the plan root — exactly the shape of a top-level checkpoint, so it was structurally unable to
   engage for the entire run.
4. Planning calls (`_expand_node`, `continue_from_checkpoint`, `replan_subtree`) only ever see a bare
   ancestor-description chain — no visibility into sibling or cousin plan structure, so a new checkpoint
   has no way to notice a nearby node already covered the same ground.

Forcing every checkpoint decomposition to include an implementation/decision leaf was considered and
rejected: it would defeat the reason checkpoints exist — genuine, possibly multi-round narrowing on
large or unfamiliar codebases before a decision is safe to make. This spec targets the actual defects
(a safety net with a coverage gap, discarded evidence, dead escalation code, blind planning calls), not
the checkpoint mechanism's legitimate deferral itself.

## Solution

Four changes, all decided together in `docs/adr/0002-checkpoint-decision-avoidance-loop.md`:

1. Unify the checkpoint-chain streak counter so it's reachable from both the rollup-success and the
   rollup-failure (replan) paths, keyed by the checkpoint node's own identity rather than its parent's.
2. Preserve a snapshot of a checkpoint's accumulated findings into the replan prompt before its
   descendant evidence is discarded, so a rollup-triggered replan builds on what's known instead of
   rediscovering it.
3. Add a dedicated, separately-tracked counter for "this exact node's rollup has been rejected N times,"
   applying to any branch (not just checkpoints), which escalates prompt pressure to converge on a
   concrete decision without mandating any particular step shape. Fix the stuck-replan escalation's
   silent no-op for root-parented nodes as part of this (bookkeeping fix only — it still defers to
   RetryManager's existing retry/replan/escalate caps for what happens next).
4. Give every planning call the same nearby-plan-tree visibility the atomicity check already has (a
   40-node BFS over the plan tree, rendered as an indented tree), extended for these new call sites with
   each node's status inline, so a checkpoint's decomposition can notice a sibling already covered the
   same ground.

## User Stories

1. As the agent operating on a large or unfamiliar codebase, I want a checkpoint's rollup-failure retries
   to be counted toward the same chain-progress limit as its continuation-success splices, so that a
   checkpoint stuck oscillating between "rollup rejected" and "replan into another checkpoint" is bounded
   the same way a checkpoint stuck chaining through successful-but-empty continuations already is.
2. As the agent replanning a checkpoint whose rollup was rejected, I want to see what my own prior
   exploration under that checkpoint actually found, so that I don't re-read the same files and re-derive
   the same facts on every retry.
3. As the agent stuck on a checkpoint whose parent is the plan root, I want the stuck-replan escalation
   bookkeeping to actually track my streak (instead of silently no-op'ing), so that repeated failures on a
   top-level sub-goal are visible and eventually escalate through the normal retry/replan/escalate path
   like any other stuck node would.
4. As the agent replanning any branch (checkpoint or not) whose rollup keeps coming back not-complete, I
   want escalating pressure in my replan prompt telling me explicitly how many times this exact sub-goal
   has been rejected, so that I'm nudged toward committing to a concrete decision rather than repeating
   the same shape of attempt indefinitely.
5. As the agent decomposing a checkpoint (or any branch) into its next steps, I want to see the status of
   nearby sibling and cousin plan nodes, not just my ancestor chain, so that I can notice when a nearby
   node already completed the same kind of work and avoid proposing a near-duplicate.
6. As the agent generating a checkpoint's continuation steps after its rollup passes, I want the same
   nearby-node visibility as ordinary decomposition/replan calls, so that the continuation's next steps
   are chosen with awareness of what else already exists in the plan.
7. As a developer maintaining this codebase, I want the three streak/counter dicts on `ReActLoop`
   (`_subtree_replan_streak`, the unified `_checkpoint_chain_streak`, and the new rollup-reject counter)
   to each track one clearly distinct axis with its own reset condition, so that progress on one axis
   never silently resets a counter tracking a different one.
8. As a developer benchmarking this fix, I want `frontier/vague_feature_request_real_repo` to no longer
   hang in a nested-checkpoint spiral, so that the case either completes with real implementation changes
   or fails for a genuinely different, more informative reason.

## Implementation Decisions

- **`chef_human/agent/react_loop.py`, `_checkpoint_chain_streak`**: re-key by the checkpoint node's own
  `node_id` instead of its parent's. Increment/consult it from both `_fire_checkpoint_continuation` (the
  existing call site) and from `_replan_failing_node`/the point where `replan_subtree` regenerates a
  checkpoint node's children in place. Trigger condition (checkpoint-only output) and limit (3) unchanged.
- **Evidence snapshot on rollup-triggered replan**: before `_discard_subtree_evidence` wipes a node's
  descendant evidence in `_replan_failing_node`, build a snapshot in the same shape
  `_checkpoint_continuation_evidence` already produces (files read + each descendant's
  `last_verdict_reason`) and fold it into `failure_context` passed to `replan_subtree`. The structured
  evidence buckets themselves are still wiped afterward — only the prompt text changes.
- **New counter `_rollup_reject_streak: dict[str, int]`** on `ReActLoop`, keyed by a branch's own
  `node_id`. Incremented each time `_process_rollups` gets a `not_complete` (or equivalent non-passing)
  verdict for that node; reset to 0 only when that same node's rollup subsequently passes. At threshold 3,
  append an explicit line to `failure_context` naming the rejection count and asking the model to converge
  on something concrete, without specifying step shape. Applies to any branch, not just checkpoints.
- **`_widen_stuck_replan_target`**: remove the silent early return for `ancestor is plan.root` so the
  streak bookkeeping increments correctly for root-parented nodes; the widening behavior itself (return
  target unchanged when there's no bigger branch to widen to) is otherwise unaffected — no new escalation
  path is introduced here, RetryManager's existing caps remain what eventually surfaces a stuck top-level
  node.
- **Nearby-node context wiring**: reuse `Planner._collect_nearby_nodes`/`_render_nearby_tree` (currently
  only called from `_classify_children`'s atomicity check) inside `_build_expand_messages` (covers
  `_expand_node` and `expand_checkpoint`) and `_build_replan_messages` (covers `replan_subtree`), and add
  equivalent context to `continue_from_checkpoint`'s message construction. Limit stays 40, consistent with
  the atomicity check. For these new call sites only, extend the tree render to show each node's status
  inline (e.g. `completed`/`pending`) — the atomicity check's own render is left unchanged.
- No change to the checkpoint declaration/lazy-expansion mechanism itself, the splice mechanics, or the
  planner prompt's step-type guidance — all from the original `checkpoint-planning` effort and out of
  scope here.

## Testing Decisions

- Tests target external behavior of the modules above (streak/counter state transitions, prompt content
  produced for a given plan shape, message assembly), not internal call sequencing.
- `chef_human/agent/planner.py` changes (nearby-node wiring in `_build_expand_messages`/
  `_build_replan_messages`/`continue_from_checkpoint`): unit tests asserting on assembled message content
  for a constructed plan tree, following `TestCheckpointPromptGuidance`'s pattern in `test_planner.py`
  (assert specific substrings/node descriptions/status markers appear in the system or user message).
- `chef_human/agent/react_loop.py` changes (unified streak, evidence snapshot, new rejection counter,
  `_widen_stuck_replan_target` fix): unit tests against `ReActLoop` with a mocked LLM backend and a
  hand-built `Plan`/`PlanNode` tree, following the existing pattern in `test_react_loop.py`'s approval-gate
  and step-verification test classes (e.g. `TestInvestigativeStepBypassesVerification`) — construct the
  specific tree/history shape that should trigger each counter or evidence-snapshot behavior, assert on
  the resulting streak values, prompt/`failure_context` content, and demotion decisions.
- Prior art: `TestReadyRollupBranches` and `TestUnresolvedSteps` in `test_planner.py` for pure
  `Plan`/`PlanNode` state tests; `test_self_cleanup_rm_auto_approved_without_prompt` and
  `test_named_file_with_directory_path_read_step_runs_through_verifier` in `test_react_loop.py` for the
  mocked-backend `ReActLoop` pattern most of this work will follow.
- Final validation: run `python -m chef_human.benchmark --case vague_feature_request_real_repo
  --keep-workspaces --json` (with `CHEF_OLLAMA_THINK=true`) after implementation. This is confirmatory,
  not a substitute for the unit tests above — a real-model run is not deterministic enough to be the sole
  signal, but it is the ground truth that the fix addresses the actual observed failure, per this
  project's existing benchmark-validation convention (see ticket 05 of the original `checkpoint-planning`
  effort).

## Out of Scope

- Any change to when/whether the planner emits a checkpoint in the first place (the prompt guidance from
  `checkpoint-planning`'s ticket 04 stands as-is).
- Forcing a checkpoint's decomposition to include an implementation/decision-shaped leaf — explicitly
  rejected in `docs/adr/0002-checkpoint-decision-avoidance-loop.md`.
- Tuning the nearby-node BFS limit (kept at 40) — revisit only if a real run shows measured prompt-size
  problems from the new call sites.
- Any change to `unresolved_steps()`, the directory-qualified named-file comparison, or the headless
  destructive-command approval gate — those were separate bugs, already fixed and committed
  (`0add1bc`, "Fix three false-completion bugs in the react loop and planner").
- Collapsing `_subtree_replan_streak`, the unified `_checkpoint_chain_streak`, and the new
  `_rollup_reject_streak` into a single counter — explicitly rejected; each tracks a distinct axis.

## Further Notes

This is a direct follow-on to the `checkpoint-planning` effort (`.scratch/checkpoint-planning/`), whose
own ticket 03 already anticipated chained/nested checkpoints as a risk and built the continuation-path
half of the safety net; this effort closes the replan-failure-path half that ticket didn't cover, plus
the three adjacent gaps (evidence discard, dead escalation code, blind planning calls) found while
diagnosing why the existing safety net didn't catch the observed incident.
