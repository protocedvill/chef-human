# 04: Evidence propagation + subtree-scoped replan

**What to build:** Make a failure — whether a leaf failing its own verification or a branch getting
rejected by rollup verification — trigger a replan of just the failing node's own subtree, not the whole
plan, with evidence correctly scoped and cleaned up around that replan.

Evidence propagates upward at write time: recording evidence for a node also appends it into every
ancestor's bucket immediately, so a branch's bucket always reflects everything that happened under it.
Replan is structurally confined to the failing node's own subtree — the replan call only ever
sees/produces that node's descendants; siblings and ancestors elsewhere in the tree are mechanically
untouched (same ids, same evidence, no LLM involved in deciding what counts as "the same goal reworded").
The failing node itself keeps its own `node_id` through the replan (its goal is unchanged, only how to
achieve it is being reconsidered) and its evidence bucket resets. Its old descendants (if it was a
branch) are discarded entirely. Because evidence propagates upward, the replan also scrubs every ancestor
above the failing node of the now-stale propagated entries that traced back to the discarded descendants.

**Blocked by:** 03 (Rollup verification)

**Status:** done

- [x] Recording evidence for a node appends it into every ancestor's evidence bucket at write time
      (`ReActLoop._verify_and_mark_step`, `react_loop.py` ~2107-2116: `step_evidence.merge_turn(...)`
      then walks `step.parent` upward calling `_step_evidence_for(ancestor).merge_turn(...)`).
- [x] A leaf failing its own verification triggers a replan scoped to just that leaf (promotable to a
      branch if the replan decomposes it further) (`_replan_failing_node` called from the retry-exhaustion
      paths at `react_loop.py` ~1128, ~1181, ~1764; targets `self._last_failed_node`, which
      `_verify_and_mark_step` sets to the current leaf up front).
- [x] A branch rejected by rollup verification triggers a replan scoped to just that branch, regenerating
      fresh children under the same `node_id` (`_process_rollups` sets `self._last_failed_node = branch`
      on a non-complete rollup verdict; the same `_replan_failing_node` path picks it up and calls
      `Planner.replan_subtree(plan, target, ...)`, which keeps `node.node_id` and only replaces
      `node.children`).
- [x] A subtree replan never touches sibling or ancestor nodes' ids or evidence outside the failing node's
      own subtree (`Planner.replan_subtree` only ever calls `node.set_children(...)` on the target node;
      `_discard_subtree_evidence`/`_rebuild_ancestor_evidence` only touch the target's descendants and
      ancestors, never siblings).
- [x] After a subtree replan, every ancestor above the failing node has the stale propagated entries from
      the discarded descendants removed from its own evidence bucket (`_rebuild_ancestor_evidence`
      recomputes each ancestor's bucket from scratch as the union of its *current* descendants' evidence,
      called right after `Planner.replan_subtree` in `_replan_failing_node`).
- [x] A later rollup verification of an ancestor above a replanned subtree never sees evidence from the
      discarded attempt (follows from the above: `_rollup_evidence`/`verify_rollup` only ever read the
      rebuilt bucket, which by construction excludes anything traceable to discarded descendants).

Verified already implemented and covered by tests (`tests/test_agent/test_react_loop.py` —
`_replan_failing_node`, `_discard_subtree_evidence`, `_rebuild_ancestor_evidence` tests around lines
5024-5130, 5432-5433, 5767-5797; `tests/test_agent/test_planner.py` `replan_subtree` tests around
1197-1251) and by a full `pytest tests/` run (1587 passed; the 6 failures present — `test_embeddings.py`
ImportError-mock cases, `test_integration.py` full-loop cases, `test_lint_fix.py::test_fix_no_issues` —
are pre-existing and unrelated to this ticket, not caused by any change here since none was needed).
No code changes were required for this ticket; closing it as done rather than leaving it
`ready-for-agent`.
