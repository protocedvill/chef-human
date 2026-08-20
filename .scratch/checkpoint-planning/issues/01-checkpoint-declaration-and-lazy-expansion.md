# 01: Checkpoint declaration + lazy expansion

**What to build:** A plan step can be declared as a checkpoint by the planner (a new `"type":
"checkpoint"` value in the structured JSON step output, alongside the existing `"leaf"`/`"branch"`
values and the `"uncertain"` flag), and its decomposition into real children is deferred from
generation-time to the moment execution first reaches it — reusing the existing `_expand_node`/
atomicity-check decomposition pipeline, just triggered later. Nothing is speculatively generated past a
checkpoint when it is first declared.

**Blocked by:** None (can start immediately)

**Status:** done

- [x] `PlanNode` gains a field recording its declared kind (leaf/branch/checkpoint), parsed and preserved
      through `Planner._parse_steps`/`_normalize_steps` the same way `requested_branch`/`flagged` already
      are.
- [x] A plan containing a checkpoint step has zero children for that node immediately after
      `Planner.generate_plan()` returns.
- [x] The checkpoint node is decomposed into real children (via the existing `_expand_node`/atomicity-check
      pipeline) exactly when it first becomes the node execution is about to work on — not before.
- [x] Test coverage at the `Planner` seam (`tests/test_agent/test_planner.py`, mirroring
      `TestAtomicityCheck`/`TestUpdatePlan` style): parsing/classification of `"type": "checkpoint"`.
- [x] Test coverage at the `ReActLoop` seam (`tests/test_agent/test_react_loop.py`, mirroring
      `TestRollupVerification` style): the lazy-expansion trigger fires at first-execution-time, not
      generation-time.
