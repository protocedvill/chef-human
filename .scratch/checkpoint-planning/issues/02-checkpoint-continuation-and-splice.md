# 02: Checkpoint completion → continuation call + sibling splice

**What to build:** When a checkpoint's own rollup verification passes, it is marked `completed`
immediately — independent of what happens to any steps it later spawns — and a new, dedicated
`Planner.continue_from_checkpoint` call fires: success-framed (not a repurposing of `update_plan`'s
failure-recovery framing), fed the checkpoint's children's files-read (via the existing
`_read_file_names` mechanism) and each child's own recorded reasoning/evidence, not just files written.
The call's output steps are spliced in as fresh sibling `PlanNode`s immediately following the checkpoint
in its own parent's children list, continuing the plan's flow at that level rather than nesting deeper.

**Blocked by:** 01

**Status:** done

- [x] `Planner` gains `continue_from_checkpoint` (or equivalent), with its own success-framed prompt,
      distinct from `update_plan`.
- [x] The continuation call's assembled evidence includes files read by the checkpoint's children and
      each child's own reasoning/evidence — not only written-file contents.
- [x] On a checkpoint branch's rollup success, `ReActLoop`'s rollup-completion path (`_process_rollups`,
      extended) marks the checkpoint `completed` immediately, then invokes the continuation call.
- [x] The continuation call's output steps are inserted as new siblings immediately after the checkpoint
      node in its parent's children list; the checkpoint node itself is not removed or replaced.
- [x] Every spliced step is a brand-new `PlanNode` (fresh `node_id`) — no `continues_node_id` involvement.
- [x] Spliced siblings become real children of the checkpoint's parent branch (if any), so that branch's
      own rollup naturally waits on them via existing `ready_rollup_branches` semantics — no new
      rollup-scoping mechanism.
- [x] Spliced steps go through the same atomicity-check/`_classify_children` pipeline as any other
      generated step, so a spliced step can itself become a further checkpoint.
- [x] Test coverage at the `ReActLoop` seam (`tests/test_agent/test_react_loop.py`,
      `TestRollupVerification`-style, mocked `Planner`): splice position, checkpoint completion semantics
      independent of spawned work, parent-branch rollup waiting on spliced siblings.
- [x] Test coverage at the `Planner` seam (`tests/test_agent/test_planner.py`): `continue_from_checkpoint`
      prompt content and evidence assembly (files-read + reasoning), distinct from `update_plan`'s framing.
