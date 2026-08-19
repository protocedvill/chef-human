# Tree-aware rendering for the main-loop LLM prompt

Type: grilling
Status: resolved

## Question

`Planner.format_plan_for_prompt` currently renders the flat plan as a linear numbered list
(`Step N: description`) fed into the main ReAct loop's reasoning prompt every turn, via
`chef_human/agent/prompts.py`. A tree can't be shown the same way without a redesign, and
`chef_human/agent/context.py`'s context assembler is already token-budget constrained — showing every
node of a large tree every turn is a real cost, not just a formatting question.

What should the main-loop prompt actually show: the full tree every turn (costly, but full context),
just the current leaf's ancestor path plus its immediate siblings (cheaper, matches what a human skimming
a task list would want — "where am I, what's around me"), or something adaptive (e.g. collapse completed
subtrees to one line, expand only the active branch)? This is distinct from
[flagged-node-review-ux](03-flagged-node-review-ux.md)'s tree view — that one is a one-time human-facing review; this one is what the
executing LLM sees on every turn, budget-constrained and running unattended.

## Answer

Adaptive collapse, replacing `Planner.format_plan_for_prompt`'s linear numbering: the active path — the
current leaf, its full ancestor chain up to the root, and each ancestor's other immediate children
(siblings at every level of the active path) — is shown expanded (full description + status). Every
other subtree, whether already completed or not yet reached, collapses to one line (description +
status marker only, no descendants shown).

The collapse rule is uniform, not split by completed-vs-pending: one predicate (is this node an
ancestor, the current node, or a sibling of one of those) decides expanded-or-not. Most of the budget
savings come from collapsing large not-yet-reached subtrees rather than small completed ones, but a
single rule keeps the rendering logic simple.

The render is recomputed fresh every turn from current tree state, matching today's pattern of calling
`format_plan_for_prompt` fresh each turn in `build_agent_prompt` — no incremental-maintenance/
invalidation logic, since the tree is cheap to re-walk relative to the LLM call itself and mutates
between turns anyway (nodes complete, subtrees get replanned).
