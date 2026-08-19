# Rollup verification design

Type: grilling
Status: resolved

## Question

Non-leaf `PlanNode` completion is settled as "child aggregation + a rollup verification check" that
produces a `StepVerdict`, reusing the existing verdict type from leaf verification
(`Planner.verify_step`). What isn't settled: what evidence does the rollup check actually see, and how
does it decide COMPLETE / PARTIAL / NOT_COMPLETE for a branch whose children are all individually
verified-complete but whose combined effect might still miss the branch's real goal?

Concretely: does the rollup verifier see each child's own verification evidence/reason (aggregated
up), the branch's description plus current repo state (independent of child evidence, similar to how
leaf verification already treats current file contents as ground truth per the `EditTool` no-op fix
documented in CLAUDE.md), or both? And does a rollup check ever need to *reject* a branch whose children
are all complete (e.g. decomposition itself was flawed — the children collectively don't cover the
branch's goal), and if so what happens next — does that trigger a subtree replan (see
[per-node-evidence-and-replan-scope](02-per-node-evidence-and-replan-scope.md)) or something else?

## Answer

Rollup verification uses ground-truth repo/file state relevant to the branch's own goal as the
primary evidence, mirroring leaf verification's `STEP_VERIFY_PROMPT` framing (state outranks a turn's
own narrated evidence). Each child's verification verdict/reason is appended as supporting context, not
the deciding signal — this is what catches a coverage gap (every child individually correct, but the
sum still misses the branch's real goal), which pure child-aggregation would miss.

A rollup check *can* reject a branch whose children are all individually complete — the tree shape is
not itself proof of completion. A rejection is treated as the branch failing verification, symmetric
with a leaf failing: it goes through the same subtree-replan path (see
[per-node-evidence-and-replan-scope](02-per-node-evidence-and-replan-scope.md)), scoped at the branch's own `node_id` — the branch is freshly
decomposed rather than any specific child being patched.
