# Flagged-node human review UX

Type: prototype
Status: resolved

## Question

Human review is settled as a one-time whole-tree approval pass after generation, with flagged nodes
(where the planner was uncertain how to decompose) surfaced there. What isn't settled is what the human
can actually *do* at that review step, concretely enough to design the interaction.

Build a rough prototype (CLI/TUI mockup or interaction script is fine — this doesn't need real
generation behind it) to react to: can the human only approve-or-reject the tree wholesale, or also edit
a flagged node's description in place, manually mark a flagged node as a leaf (overriding further
decomposition), or trigger a re-decomposition of just that one branch with guidance? Does review show
the *whole* tree or only the flagged nodes with enough surrounding context to judge them? This directly
shapes what `ReActUI.on_plan` (see `chef_human/ui/protocol.py`) needs to support for the new tree shape.

## Answer

Review shows the **whole tree**, not just flagged nodes — flags (⚑) marked inline against each node, so
the human sees full context around what's uncertain rather than isolated fragments.

The action set at review time is all five prototyped: **approve** (accept as-is), **edit** a node's
description in place, **mark-as-leaf** (override further decomposition, discarding any children the
node already had), **redecompose-with-guidance** (ask the planner to retry just that branch, steered),
and **reject** (abandon the task). This gives the human both direct fixes (edit, mark-as-leaf) and an
escape hatch back to the planner (redecompose-with-guidance) without forcing a full plan restart.

Prototype (throwaway CLI mockup, both view variants + a simulated review session) captured on
`prototype/flagged-node-review-ux` (commit `6a84c82`), primary source for this decision — not merged to
main. This shapes `ReActUI.on_plan` (`chef_human/ui/protocol.py`): it will need to accept the full tree
plus support this action set as a review-time callback, not just a display hook, for the new tree shape.
