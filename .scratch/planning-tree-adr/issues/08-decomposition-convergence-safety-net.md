# Soft safety net for slow-converging decomposition

Type: grilling
Status: resolved

## Question

There's no hard depth cap on decomposition (settled earlier) — convergence relies on the LLM eventually
emitting `leaf: true` for every child (see
[iterative-expansion-protocol](04-iterative-expansion-protocol.md)). In headless mode especially, uncertainty biases toward *more*
decomposition, which could in principle spiral on a genuinely ambiguous task.

Is a soft safety net warranted — e.g. the harness tracking expansion depth or node count per subtree and
logging a warning (or escalating via `RetryManager`, see
[retrymanager-per-node-failure-counting](07-retrymanager-per-node-failure-counting.md)) past some threshold, without hard-stopping generation — or is
this an acceptable risk to leave unaddressed given the design otherwise trusts the LLM's own leaf/branch
judgment throughout?

## Answer

Yes, a safety net is warranted, but purely observational — a telemetry/warning, not a behavior change.
Generation is never stopped or forced to commit to a leaf early; a subtree's expansion depth or node
count is tracked and logged past some threshold, giving a human reviewing the whole-tree approval pass
(or reading logs on a headless run) a signal that something's off, in the same spirit as the
`VERIFIER_DEBUG` logging CLAUDE.md already documents for diagnosing false escalations. This doesn't
contradict the earlier "no hard depth cap" decision — it adds visibility, not a ceiling.

This is a separate, independent counter from `RetryManager`'s per-node escalation
([retrymanager-per-node-failure-counting](07-retrymanager-per-node-failure-counting.md)), not routed through it. `RetryManager` was just resolved to
specifically track *execution* failures (a leaf's tool calls/verification failing) — a distinct, later
phase from decomposition/expansion, which has no notion of "failure" at all (a branch that's decomposed
40 levels deep hasn't failed anything, it just hasn't converged yet). Keeping this as a standalone
depth/node-count counter checked during expansion preserves that distinction rather than overloading
`RetryManager` with a second, unrelated kind of "too much happened here."
