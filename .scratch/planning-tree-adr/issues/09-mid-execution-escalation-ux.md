# Mid-execution escalation UX for a node exhausting its retry/replan budget

Type: grilling
Status: resolved

## Question

[retrymanager-per-node-failure-counting](07-retrymanager-per-node-failure-counting.md) settled that a node exhausting its own retry/replan budget
escalates just that node/subtree rather than aborting the run, surfacing "to the human" in interactive
mode. That's a new touchpoint, distinct from [flagged-node-review-ux](03-flagged-node-review-ux.md)'s one-time pre-execution review
pass — this happens mid-execution, potentially deep into an otherwise-unattended run.

What isn't settled: is this a blocking prompt (execution pauses until the human responds, similar to how
the review pass blocks before execution starts) or an async notification the human can address whenever
convenient while the rest of the tree keeps executing? And what can the human actually do at that
point — the same action set as the review pass (edit/mark-as-leaf/redecompose-with-guidance/reject), or
something narrower given this is a node that's already been tried and failed repeatedly, not a fresh
uncertain decomposition?

## Answer

Async, not a blocking prompt — but grounded in a fact that shapes what "async" can mean here:
chef-human's executor is already sequential (`ReplUI`'s synchronous terminal loop, `ReActLoop` dispatch
deliberately serialized per CLAUDE.md, one leaf at a time, no concurrent tool dispatch). There's no real
concurrency for a notification to let something "keep running in the background" past, so async here
means: the harness applies the **same default fallback as headless mode** automatically (mark the
escalating node failed, continue execution at the next node in tree order) and surfaces a notification —
the human's involvement is optional and retroactive, not a forward gate. Interactive mode's default
mid-execution-escalation behavior is thus identical to headless's; a human present just gets the option
to intervene after the fact, where headless has no one to notify at all.

Given that, the action set is what the human can do *after* the default fallback has already applied:
**edit** (revise the node's description) + **redecompose-with-guidance** (regenerate this subtree with a
hint about what went wrong) to override the auto-applied failure and retry, or **reject** to abort the
whole run. **Mark-failed-and-continue** isn't a separate human action — it's the automatic default
already in effect unless the human overrides it. **Approve** and **mark-as-leaf** from the pre-execution
review's action set don't apply here (nothing to approve on an already-executed node; granularity isn't
the problem for a node that failed at whatever granularity it was already at).

This was the last open ticket on the planning-tree-adr map — the map is now fully clear.
