# 03: Rollup-reject pressure counter + stuck-replan escalation bookkeeping fix

**What to build:** Add a new counter, `_rollup_reject_streak: dict[str, int]`, keyed by a branch's own
`node_id`, tracking consecutive `not_complete` (non-passing) rollup verdicts for that exact node —
applies to any branch, checkpoint or not. Increment it in `_process_rollups` on each rejection; reset it
to 0 only when that same node's rollup subsequently passes (not by unrelated progress elsewhere in the
tree). At a threshold of 3, append an explicit line to `failure_context` naming the rejection count and
asking the model to converge on something concrete now, without specifying what shape that step must
take (no forced implementation/decision leaf — see spec's Out of Scope).

Keep this counter's dict and reset condition entirely separate from `_subtree_replan_streak` (ancestor-
widening pressure) and the unified `_checkpoint_chain_streak` from ticket 01 (children-shape signal) —
each tracks a genuinely distinct axis, and conflating any two would let progress on one silently reset a
counter tracking a different one.

As part of the same ticket, fix `_widen_stuck_replan_target`'s silent early return for
`ancestor is plan.root`: remove the no-op so the (separate) `_subtree_replan_streak` bookkeeping actually
increments for root-parented nodes, matching the same reasoning above. Do not introduce any new
escalation behavior when the streak crosses its threshold for a root-parented node — widening still
returns the node unchanged (there's no bigger branch to widen to); RetryManager's existing retry/replan/
escalate caps remain what eventually surfaces a stuck top-level node to a human.

**Blocked by:** None (can start immediately)

**Status:** done

- [x] `_rollup_reject_streak` increments on each `not_complete` rollup verdict for a node, keyed by that
      node's own `node_id`, and resets to 0 only when that node's own rollup subsequently passes.
- [x] At 3 consecutive rejections for the same node, the next replan's `failure_context` contains an
      explicit line stating the rejection count and asking for a concrete decision — verified via a unit
      test constructing that exact rejection sequence.
- [x] This applies to an ordinary (non-checkpoint) branch too, not just checkpoints — covered by a
      distinct unit test case.
- [x] `_rollup_reject_streak` is a separate dict from `_subtree_replan_streak` and the unified
      `_checkpoint_chain_streak`; a test confirms that resetting one does not affect the others' state
      for the same node.
- [x] `_widen_stuck_replan_target` no longer silently no-ops for `ancestor is plan.root` — its streak
      bookkeeping increments for a root-parented node's repeated stuck replans, confirmed by a unit test.
      Its returned target/widening decision for that case is otherwise unchanged from today.
