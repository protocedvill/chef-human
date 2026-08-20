# 03: Chain-progress counter + structural pass-through guard

**What to build:** Chained/nested checkpoints (a checkpoint's continuation spawning another checkpoint,
recursively, anywhere in the tree) are bounded by a dedicated progress counter — separate from
`_ConvergenceTracker`, which is scoped to one synchronous generation-time recursion and is the wrong
shape for a mechanism triggered across multiple execution turns via `_process_rollups` — similar in
spirit to the existing `_subtree_replan_streak` mechanism elsewhere in `react_loop.py`. Additionally, the
harness structurally forbids a checkpoint's own decomposition from consisting of exactly one child that
is itself another checkpoint with no other work — enforced the same way the existing single-child
breakdown guard forces `requested_branch = False` on a degenerate atomicity-reclassification, not left to
prompt wording alone.

**Blocked by:** 02

**Status:** done

- [x] A dedicated counter tracks checkpoint-chain progress across execution turns, independent of
      `_ConvergenceTracker`.
- [x] A pathological chain of checkpoints producing no real (non-checkpoint) work between them is bounded
      by this counter, with behavior analogous to existing stuck-replan escalation elsewhere in
      `react_loop.py`.
- [x] A legitimate chain — real exploration/implementation work happening between each checkpoint — is
      unaffected by the counter.
- [x] A checkpoint whose own decomposition is a single child that is itself another checkpoint (a
      degenerate pass-through with no other work) is structurally rejected or forced to reclassify,
      independent of prompt wording.
- [x] Test coverage at the `ReActLoop` seam (`tests/test_agent/test_react_loop.py`): both the chain
      counter's bounding behavior and the structural pass-through guard, using synthetic checkpoint chains.
