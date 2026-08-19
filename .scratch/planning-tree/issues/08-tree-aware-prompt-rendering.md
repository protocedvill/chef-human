# 08: Tree-aware adaptive prompt rendering

**What to build:** Replace `Planner.format_plan_for_prompt`'s hardcoded linear "Step N" numbering with a
tree-aware render fed into the main-loop reasoning prompt every turn, bounded so a large tree doesn't
consume the context budget on irrelevant, distant parts of the plan.

The active path — the current leaf, its full ancestor chain up to the root, and each ancestor's other
immediate children (siblings at every level of the active path) — is shown expanded (full description +
status). Every other subtree, whether already completed or not yet reached, collapses to one line
(description + status marker only, no descendants shown). One uniform rule decides expanded-vs-collapsed
(is this node an ancestor, the current node, or a sibling of one of those), not different treatment for
completed-vs-pending. The render is recomputed fresh every turn from current tree state, same pattern as
today's `format_plan_for_prompt` being called fresh every turn.

**Blocked by:** 02 (Multi-level decomposition)

**Status:** done

- [x] The main-loop prompt shows the active path (current leaf + ancestors + their siblings) in full detail
- [x] Every subtree not on the active path collapses to one line, regardless of completed-vs-pending status
- [x] The render is recomputed fresh each turn, reflecting the tree's current mutated state (completions, replans, rollup results)
- [x] A deep tree (e.g. produced by the `marathon` benchmark case) produces a bounded-size render rather than growing unbounded with tree size
