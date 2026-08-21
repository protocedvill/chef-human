# Context

## Glossary

**Plan** — the outer container for a unit of agent work: a goal plus a tree of nodes describing how to reach it.

**PlanNode** — one element of a plan tree. Carries a stable identity that survives replans, a description, a status, and zero or more child nodes.

**Leaf** — a plan node with no children. A leaf must correspond to exactly one tool call.

**Branch** — a plan node with children. Represents a goal too large to execute directly; broken down rather than executed.

**Decompose / decomposition** — the act of turning a branch into its children. Continues until every node in a subtree is a leaf.

**Leaf verification** — the check that a leaf's single tool call actually accomplished what the leaf described.

**Rollup verification** — the check that a branch's goal was actually accomplished by the sum of its children, beyond just "all children completed."

**Checkpoint** — a plan node whose own implementation shape isn't known yet: what to go find out is known, what to build once found out is not. Left undecomposed at generation time; decomposed lazily the first time execution reaches it. Distinct from an ordinary branch, which *is* fully decomposable up front. A checkpoint that passes rollup verification triggers a success-framed continuation call, which plans the concrete next steps (implementation included, where the checkpoint's own findings support it) as fresh siblings after it — not as further children nested under it. See `docs/adr/0002-checkpoint-decision-avoidance-loop.md` for the failure mode where a checkpoint's continuation kept producing another checkpoint instead of committing, and the fix.

**Flagged (node)** — a node the planner was uncertain how to decompose. Surfaced to a human operator during the one-time whole-tree review pass, when one is present. In headless operation, a flagged node is instead decomposed further rather than left as a guessed leaf.

**Current leaf** — the leaf execution should work on next: the first *pending* leaf found in tree order. Deliberately narrower than "not yet completed" — an `in_progress` leaf (work already underway on it) is not a current leaf, which matters for replan-target selection (see `docs/adr/0001-evidence-carry-forward-across-whole-plan-replan.md`).
