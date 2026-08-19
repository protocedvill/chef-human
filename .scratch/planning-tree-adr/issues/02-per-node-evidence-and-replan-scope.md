# Per-node evidence model and subtree replan mechanics

Type: grilling
Status: resolved

## Question

Node identity is settled (stable UUID, survives replans) and replan scope is settled ("just the failing
subtree, not the whole tree"). What isn't settled is the mechanics: when a subtree is replanned, which
nodes keep their existing `node_id` (and thus their accumulated evidence) versus get fresh ids?

Specifically: does the replanner get shown the old subtree's ids and asked to reuse the ones whose
underlying goal is unchanged (mirroring today's `update_plan`'s dedup-by-description logic, upgraded to
ids) — and if a replan produces a node that's semantically "the same goal, different wording" as the
node it replaced, is that reuse decision made by the replanning LLM call itself, or by a deterministic
post-process step (e.g. matching against the failing node's own id, since only *that* one is definitely
being replaced, with siblings/ancestors definitely kept)?

Also: does a branch's evidence (for its own rollup verification, see
[rollup-verification-design](01-rollup-verification-design.md)) exist as its own accumulated bucket, or is it always derived by reading
its children's evidence at verification time — this affects what a subtree replan needs to invalidate
versus preserve.

## Answer

Replan is structurally confined to the failing node's own subtree — the replan call only ever
sees/produces that node's descendants; siblings and ancestors elsewhere in the tree are mechanically
untouched (same ids, same evidence, no LLM involvement). This eliminates the "is this the same goal
reworded" fuzzy-matching problem update_plan has today (description-based dedup) by construction: there
is nothing outside the subtree for the replanner to accidentally rename or duplicate.

The failing node itself (leaf or branch — including a branch rejected by its own rollup verification,
see [rollup-verification-design](01-rollup-verification-design.md)) keeps its own `node_id` through the replan: its goal is unchanged,
only how to achieve it is being reconsidered. Its old descendants (if it was a branch) are discarded
entirely — their ids simply stop existing in the tree.

Every node, leaf or branch, has its own accumulated evidence bucket (replacing today's
description-keyed `_step_evidence` dict in `react_loop.py`, keyed by `node_id` instead). Evidence
propagates upward at write time: recording evidence for a node also appends it into every ancestor's
bucket immediately, so a branch's bucket always reflects everything that happened under it without a
separate read-time aggregation pass at rollup-verification time.

On a subtree replan: the failing node's own bucket is reset (a fresh attempt starts clean), every
discarded descendant's bucket is dropped (nothing references those ids anymore), and — because evidence
propagated upward — every ancestor above the failing node also has its stale propagated entries (the
ones tracing back to the now-discarded descendants) explicitly scrubbed, so no ancestor's rollup
verification ever sees evidence from a discarded attempt.
