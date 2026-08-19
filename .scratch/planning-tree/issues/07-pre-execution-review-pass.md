# 07: Pre-execution whole-tree human review pass

**What to build:** A one-time whole-tree approval pass after generation completes and before execution
starts (interactive mode only — headless/benchmark runs skip this entirely). The full tree is shown with
flagged nodes (ones the planner was uncertain how to decompose) marked inline, not shown in isolation, so
the human has full surrounding context to judge them.

Action set: **approve** (accept as-is), **edit** a node's description, **mark-as-leaf** (override further
decomposition, discarding any children it already had), **redecompose-with-guidance** (ask the planner to
retry just that branch, steered by the human's hint), and **reject** (abandon the task). This is
distinct from ticket 06's mid-execution escalation — this happens once, before any execution, on a
tree that hasn't run yet.

A throwaway CLI mockup validating this view/action set exists on branch `prototype/flagged-node-review-ux`
(commit `6a84c82`) — reference it as the primary source for the interaction shape rather than re-deriving
it.

**Blocked by:** 02 (Multi-level decomposition)

**Status:** done

- [x] After generation, interactive mode shows the human the full tree once, before any execution begins
- [x] Flagged nodes are marked inline within the full tree view, not shown as an isolated list
- [x] The human can approve the tree as-is
- [x] The human can edit any node's description before execution starts
- [x] The human can mark a flagged (or any) node as a leaf, discarding its children
- [x] The human can trigger a guided redecomposition of any branch before execution starts
- [x] The human can reject, abandoning the task before any execution happens
- [x] Headless/benchmark runs skip this review pass entirely — no blocking on a human that isn't there
