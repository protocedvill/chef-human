# 02: Preserve checkpoint evidence into rollup-triggered replan prompts

**What to build:** When a checkpoint's rollup verification is rejected, `_replan_failing_node` discards
its descendant evidence (`_discard_subtree_evidence`) before `replan_subtree` regenerates its children.
The replan prompt's `failure_context` currently only carries the rollup verifier's rejection *reasoning
text* ("no architecture summary evidenced"), never the actual findings — so every retry re-explores the
codebase from scratch instead of building on what it already found.

Before evidence is discarded, build a snapshot in the same shape `_checkpoint_continuation_evidence`
already produces (files read + each descendant's `last_verdict_reason`) and fold it into the
`failure_context` passed to `replan_subtree`. The structured evidence buckets themselves still get wiped
afterward (their `node_id`s are genuinely gone once the subtree is replaced) — only the prompt text the
replan call sees changes.

**Blocked by:** None (can start immediately)

**Status:** done

- [x] A rollup-triggered replan's prompt includes a snapshot of the discarded subtree's accumulated
      findings (files read + descendant verdict reasoning), not just the rejection reason.
- [x] The structured `StepEvidence` buckets for the discarded descendants are still cleared after the
      snapshot is taken — no change to that bookkeeping's end state.
- [x] Unit test: construct a checkpoint with descendant evidence (files read, a `last_verdict_reason` on
      at least one child), trigger a rollup-rejection replan, and assert the resulting `failure_context`
      (or the message actually sent to the planner) contains the pre-discard findings.
- [x] A replan of a node with no prior evidence (nothing to snapshot) is unaffected — no spurious content
      or errors when the snapshot is empty.
