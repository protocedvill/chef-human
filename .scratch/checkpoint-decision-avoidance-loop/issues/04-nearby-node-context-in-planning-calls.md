# 04: Nearby-plan-tree context in all planning calls

**What to build:** `Planner._collect_nearby_nodes`/`_render_nearby_tree` (a 40-node BFS over the plan
tree treated as undirected — parent and children both count as neighbors — rendered as an indented tree)
is currently wired only into the atomicity check (`_classify_children`). Every other planning call
(`_expand_node`/`expand_checkpoint` via `_build_expand_messages`, `replan_subtree` via
`_build_replan_messages`, and `continue_from_checkpoint`) only sees a bare straight-line ancestor-
description chain — no visibility into sibling or cousin plan structure.

Wire the same nearby-node context into `_build_expand_messages`, `_build_replan_messages`, and
`continue_from_checkpoint`'s message construction. Keep the BFS limit at 40, consistent with the
atomicity check. For these three new call sites only, extend the tree render to show each collected
node's status inline (e.g. `completed`/`pending`) — the missing signal that lets a decomposition or
replan notice "a nearby node already covered this." Leave the atomicity check's existing render
unchanged (no status needed there).

**Blocked by:** None (can start immediately)

**Status:** done

- [x] `_build_expand_messages` includes a nearby-node tree render (status-annotated) alongside the
      existing ancestor chain, for non-root calls.
- [x] `_build_replan_messages` includes the same, for both `is_failure=True` and `is_failure=False`
      replan framings.
- [x] `continue_from_checkpoint`'s message construction includes the same.
- [x] The new renders show node status inline; the atomicity check's own render (`_classify_children`'s
      call to `_render_nearby_tree`) is unchanged — no status added there.
- [x] BFS limit is 40 at every call site, matching the atomicity check.
- [x] Unit tests (following `TestCheckpointPromptGuidance`'s pattern) construct a small plan tree with a
      completed sibling near the node being expanded/replanned, and assert the assembled message content
      includes that sibling's description and status.
