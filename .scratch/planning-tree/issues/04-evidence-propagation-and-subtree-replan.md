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

**Status:** ready-for-agent

- [ ] Recording evidence for a node appends it into every ancestor's evidence bucket at write time
- [ ] A leaf failing its own verification triggers a replan scoped to just that leaf (promotable to a branch if the replan decomposes it further)
- [ ] A branch rejected by rollup verification triggers a replan scoped to just that branch, regenerating fresh children under the same `node_id`
- [ ] A subtree replan never touches sibling or ancestor nodes' ids or evidence outside the failing node's own subtree
- [ ] After a subtree replan, every ancestor above the failing node has the stale propagated entries from the discarded descendants removed from its own evidence bucket
- [ ] A later rollup verification of an ancestor above a replanned subtree never sees evidence from the discarded attempt
