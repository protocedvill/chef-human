---
status: accepted
---

# Fixing the checkpoint decision-avoidance loop

## Context

`vague_feature_request_real_repo` (a benchmark case: a vague, real-world-phrased task against
an unfamiliar large repo) ran to its 600s timeout stuck in a chain of nested `checkpoint` plan
nodes, each decomposing into more exploration, none ever committing to an implementation
decision. Diagnosed via `/diagnosing-bugs`; see `CLAUDE.md`'s "Step verification" section for
the related false-completion bugs this surfaced along the way
(`Plan.unresolved_steps()` ignoring branch/checkpoint rollup status, and the directory-qualified
named-file comparison bug).

Root causes, once the false-completion bugs were fixed and the loop was actually visible:

1. **Existing safety net doesn't cover the path that triggers.** `_bound_checkpoint_chain`
   (caps 3 consecutive checkpoint-only splices) only fires from `_fire_checkpoint_continuation`
   — the rollup-*success* path. The incident's actual path was repeated rollup *failure* →
   `_replan_failing_node` → `replan_subtree` regenerating the same checkpoint's children in
   place. That path never touched the streak counter.
2. **Evidence is discarded, not just re-summarized, on a rollup-triggered replan.**
   `_discard_subtree_evidence` wipes a checkpoint's descendant evidence before `replan_subtree`
   regenerates its children. The replan prompt's `failure_context` only carries the rollup
   verifier's rejection *reasoning text* ("no architecture summary evidenced"), never the
   actual findings — so each retry re-derives the codebase structure from scratch. This is why
   the transcript showed near-identical "list structure / read configs / read entry points"
   three separate times under three separate checkpoints.
3. **The existing stuck-replan escalation is dead code for top-level nodes.**
   `_widen_stuck_replan_target` widens scope to `target.parent` and explicitly no-ops when that
   parent is `plan.root`. The incident's checkpoint was a direct child of root, so this
   mechanism was silently inert for the entire run — not merely uninvolved, structurally unable
   to engage.
4. **Planning calls have no visibility into sibling/cousin plan structure.** Every planning
   call (`_expand_node`, `continue_from_checkpoint`, `replan_subtree`) only receives
   `_ancestor_descriptions()` — a bare straight-line chain of ancestor description strings. A
   new checkpoint spliced in next to a completed one has no way to see what that sibling
   already covered, structurally or evidentially.

We explicitly rejected forcing a checkpoint's decomposition to always include an
implementation/decision leaf: that would defeat the reason checkpoints exist — genuine,
possibly multi-round narrowing on large/unfamiliar codebases before a decision is safe to make.
The fix targets the actual defects (a safety net with a gap, discarded evidence, dead escalation
code, blind planning calls), not the checkpoint mechanism's legitimate deferral itself.

## Decision

Four changes, decided together:

1. **Unify the checkpoint-chain streak.** Key `_checkpoint_chain_streak` by the checkpoint
   node's own `node_id` (not its parent's), and increment/check it from both
   `_fire_checkpoint_continuation` (success path) and `_replan_failing_node`/`replan_subtree`
   (failure path). Same trigger logic and limit (3) as before, just reachable from where the
   incident actually occurred.
2. **Preserve evidence across a rollup-triggered replan.** Before `_discard_subtree_evidence`
   wipes a checkpoint's descendant buckets, snapshot them in the same shape
   `_checkpoint_continuation_evidence` already builds (files read + each descendant's
   `last_verdict_reason`) and fold that into `failure_context`. The structured evidence buckets
   still get wiped afterward (their `node_id`s are genuinely gone) — only the *prompt text* the
   replan call sees changes, so it replans from what's known instead of rediscovering it.
3. **Add a rollup-reject pressure counter, separate from the two existing streaks.** New dict
   `_rollup_reject_streak`, keyed by a branch's own `node_id`, counts consecutive `not_complete`
   rollup verdicts on that exact node (any branch, not just checkpoints — an ordinary branch can
   loop on rollup rejection for the same underlying reason). At threshold 3, append an explicit
   "this sub-goal has failed rollup verification N times; converge on something concrete now,
   even if partial" line to `failure_context`. Resets only when that specific node's rollup
   passes, not by unrelated progress elsewhere in the tree. Kept separate from
   `_subtree_replan_streak` (ancestor-widening pressure) and the unified `_checkpoint_chain_streak`
   (children-shape signal) because it tracks a third, independent axis — conflating any two of
   these would mean progress on one axis silently resets a counter tracking a different one.
   Also fix `_widen_stuck_replan_target`'s root-parent no-op: stop the silent early return so the
   counter increments correctly for root-parented nodes too, but don't invent a second
   escalation trigger — leave RetryManager's own retry/replan/escalate caps as the thing that
   eventually surfaces a stuck root-level node to a human.
4. **Wire nearby-node context into all planning calls.** Reuse `_collect_nearby_nodes`/
   `_render_nearby_tree` (currently only used by the atomicity check, BFS over the plan tree as
   undirected, limit 40) in `_build_expand_messages` and `_build_replan_messages` (covering
   `_expand_node`, `expand_checkpoint`, and `replan_subtree`) and in `continue_from_checkpoint`.
   For these new call sites (not the existing atomicity-check render, left unchanged), add each
   node's status inline in the tree render — the missing signal that lets a checkpoint's
   decomposition notice "a sibling already covered this" instead of re-deriving it blind. Limit
   stays 40 uniformly; not tuned down for the higher call frequency absent a measured prompt-size
   problem.

## Consequences

Three separate streak/counter dicts now coexist on `ReActLoop`
(`_subtree_replan_streak`, `_checkpoint_chain_streak`, `_rollup_reject_streak`), each tracking a
distinct axis (ancestor-widening pressure, children-shape, same-node rejection count). This is
deliberate, not incidental complexity — collapsing any two would reintroduce the kind of blind
spot this decision exists to close.
