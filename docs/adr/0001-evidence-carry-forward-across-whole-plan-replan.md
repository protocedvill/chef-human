---
status: partially implemented
---

# Evidence carry-forward across a whole-plan replan

`Planner.update_plan()` is the whole-plan replan fallback, reached from `ReActLoop._replan_failing_node`
only when no specific failing node can be identified (as opposed to `replan_subtree`, the normal path,
which scopes a replan to one failing node's subtree and is out of scope for this decision — its evidence
reset there is intentional: the node's approach is being redone, so its own prior-attempt evidence should
go, while its `node_id`, siblings, and ancestors are untouched).

`update_plan()` currently carries forward only nodes already `StepStatus.completed` (the same `PlanNode`
object, so `node_id` and its evidence bucket in `ReActLoop._step_evidence` survive). Every non-completed
node — `pending` or `in_progress` — is discarded and replaced by a brand-new `PlanNode` from the LLM's
fresh output, deduped only by exact description-string match. A replan that merely rewords an in-flight
step ("Create notify.py..." → "Write notify.py...") therefore silently orphans that node's accumulated
evidence and forces redundant re-decomposition and re-implementation of already-completed work, even
though the underlying goal didn't change. This was observed in a real benchmark run
(`frontier/notification_bus_greenfield`) and contributed to a case timeout.

Two changes, decided together:

1. **Shrink how often `update_plan()` is reached at all.** `_replan_failing_node` falls back to it when
   both `self._last_failed_node is None` and `plan.current_leaf() is None` — but `current_leaf()` only
   matches `pending`, so an `in_progress` node (real work mid-flight) falls through both checks. Give
   `_replan_failing_node` its own broader target lookup, checked before `current_leaf()`, that also
   accepts an `in_progress` node — routing it through the identity-preserving `replan_subtree` instead.
   `current_leaf()` itself stays narrow: other callers (finish-gating via `unresolved_steps()`) depend on
   its pending-only meaning.

2. **For `update_plan()`'s remaining legitimate whole-plan cases** (a real structural failure with no
   specific node in play — still a real scenario after (1), not dead code): the replan prompt shows the
   current non-completed nodes with their `node_id`s, and each new step in the LLM's JSON response may
   carry an optional `"continues_node_id": "<id>" | null`. A tag matching a real, still-non-completed
   prior node causes `update_plan()` to reuse that node's `node_id` (its evidence bucket survives) with
   the *new* description text, resetting `status` to `pending`. Any malformed tag, a tag naming a
   nonexistent or already-completed node, or two new steps claiming the same old node — all fall back
   silently to treating that step as a brand-new node (debug-logged), never guessed or picked between
   claimants.

## Considered options

- **Position/order matching** ("the Nth remaining step in the new output is the Nth old step"): rejected
  as too fragile — replans routinely reorder, split, or drop steps, so position carries no reliable
  identity signal.
- **Verbatim description re-emission** (the model repeats the old step's exact wording to signal
  continuity): rejected — this is the same brittle exact-string-matching behavior this decision exists to
  get away from; it also can't survive a *deliberate* rewording of an unchanged step.
- **Explicit `continues_node_id` tagging** (chosen): the model already sees the old tree with `node_id`s
  rendered (the same style used elsewhere, e.g. pre-execution plan review); asking it to reference
  identity directly is more reliable than inferring identity after the fact, and fails safe (unmatched →
  fresh node) rather than fails open (wrongly claiming continuity).

See `CLAUDE.md`'s "Step verification" section for the historical false-escalation bugs that first
surfaced the evidence-keying problem this decision closes the remaining gap on.

## Implementation status

Item 2 (`continues_node_id` tagging in `update_plan()`) is implemented: `PlanNode.continues_node_id`
carries the tag through `_parse_steps`/`_normalize_steps`, and `update_plan()` reuses a matching
unclaimed non-completed node's `node_id` (resetting `status` to `pending`), falling back to a fresh
node on any malformed/unmatched/double-claimed tag (debug-logged). Diagnosed and fixed via a real
`frontier/vague_feature_request_real_repo` benchmark run that got stuck alternating between "Read
docs/source/software.rst" and "Use the read tool on docs/source/software.rst" for 25 replans straight,
losing the prior read's evidence each time; regression test:
`tests/test_agent/test_planner.py::TestUpdatePlan::test_reworded_pending_step_keeps_node_id_via_continues_node_id`.

Item 1 (widening `_replan_failing_node`'s target lookup to catch an `in_progress` node before falling
back to `update_plan()`) remains unimplemented — it wasn't the mechanism behind the observed bug (the
stuck node was `pending`, so `plan.current_leaf()` already found it and routed through `replan_subtree`
correctly); it's a separate, still-theoretical gap.
