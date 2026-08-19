# Context

## Glossary

**Plan** — the outer container for a unit of agent work: a goal plus a tree of nodes describing how to reach it.

**PlanNode** — one element of a plan tree. Carries a stable identity that survives replans, a description, a status, and zero or more child nodes.

**Leaf** — a plan node with no children. A leaf must correspond to exactly one tool call.

**Branch** — a plan node with children. Represents a goal too large to execute directly; broken down rather than executed.

**Decompose / decomposition** — the act of turning a branch into its children. Continues until every node in a subtree is a leaf.

**Leaf verification** — the check that a leaf's single tool call actually accomplished what the leaf described.

**Rollup verification** — the check that a branch's goal was actually accomplished by the sum of its children, beyond just "all children completed."

**Flagged (node)** — a node the planner was uncertain how to decompose. Surfaced to a human operator during the one-time whole-tree review pass, when one is present. In headless operation, a flagged node is instead decomposed further rather than left as a guessed leaf.

**Current leaf** — the leaf execution should work on next: the first non-completed leaf found in tree order.
