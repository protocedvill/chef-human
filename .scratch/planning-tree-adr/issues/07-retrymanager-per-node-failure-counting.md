# RetryManager failure counting under subtree-scoped replans

Type: grilling
Status: resolved

## Question

`RetryManager` (`chef_human/agent/retry.py`) currently tracks a single `consecutive_failures` counter
that escalates after 5 failures, decoupled from any notion of which step is failing. Now that replanning
is subtree-scoped and every node has its own evidence bucket (see
[per-node-evidence-and-replan-scope](02-per-node-evidence-and-replan-scope.md)), does the failure counter stay global across the whole run (a
failure anywhere counts toward the same threshold), or does each node accumulate its own failure count —
so a node that's failed 3 times but is now working under a fresh subtree replan doesn't inherit failure
pressure from an unrelated branch elsewhere in the tree, and vice versa (a node that's failed 4 times
shouldn't get a clean slate just because a sibling branch replanned)?

## Answer

`RetryManager`'s baseline today (confirmed by reading `retry.py` and `react_loop.py` line 624) is a
single instance for the entire `ReActLoop.run()` call — `consecutive_failures` and `replan_count` are
genuinely global across every step, with `on_replan()` resetting both instance-wide whenever *any*
replan happens anywhere. This changes: both become per-node, keyed by `node_id`, consistent with the
per-node evidence model already resolved ([per-node-evidence-and-replan-scope](02-per-node-evidence-and-replan-scope.md)) — a node's failure
pressure no longer leaks into unrelated branches, and a sibling's replan no longer wipes another node's
accumulated failure count.

The replan budget (`max_replans`) also becomes per-node rather than a shared budget across the whole
tree — otherwise one persistently-troublesome branch could exhaust the entire run's replan allowance and
starve every other branch of its own independent retry budget, defeating the point of subtree isolation.

`ESCALATE`'s meaning changes accordingly: today it aborts the whole run (there's only one flat plan, so
there's nothing else to fall back to). With a tree, a node exhausting its own retry/replan budget
escalates just that node/subtree — interactively, it surfaces to the human (a similar surface to a
flagged node at review time, but now mid-execution); in headless mode, the node is marked failed and
execution continues on the rest of the tree. The root's own rollup verification then naturally reflects
the incompleteness rather than the run terminating outright — a tree exists specifically so one hard
sub-problem doesn't block unrelated progress elsewhere.
