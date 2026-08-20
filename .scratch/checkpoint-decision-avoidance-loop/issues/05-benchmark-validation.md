# 05: Benchmark validation

**What to build:** Ground-truth confirmation that tickets 01-04 together actually fix the observed
incident: `frontier/vague_feature_request_real_repo` no longer spirals through nested checkpoints
without ever implementing anything.

Run `python -m chef_human.benchmark --case vague_feature_request_real_repo --keep-workspaces --json`
with `CHEF_OLLAMA_THINK=true`, at least twice given model non-determinism. Record the outcome (pass/fail,
`agent_success`, `verifier_success`, `changed_files`, and whether the run completed within the case
timeout) and the reason for any remaining failure. This ticket does not require the case to *pass* — a
genuinely different failure mode (e.g. an ordinary implementation defect, unrelated to checkpoint
looping) is an acceptable outcome and should be recorded as such, per the spec's problem statement. What
it must confirm is the absence of the specific failure this effort targets: no nested-checkpoint chain,
no re-derivation of the same exploration findings across retries, and no run that exhausts its step/time
budget purely on checkpoint churn.

**Blocked by:** 01, 02, 03, 04

**Status:** done

- [x] At least two live runs of `vague_feature_request_real_repo` (`CHEF_OLLAMA_THINK=true`,
      `--keep-workspaces`) are recorded with their outcomes.
- [x] Each run's agent log is checked for: no chain of more than 2 nested checkpoints before either a
      real decision/implementation step or an explicit forced-leaf demotion; no near-identical repeated
      exploration (e.g. re-reading the same files under successive checkpoints) after ticket 02's
      evidence preservation is in effect.
- [x] If a run still fails, the failure reason is identified and confirmed to be *not* the
      checkpoint-spiral pattern this effort targets (or, if it still is, that's reported back as the
      tickets not yet being sufficient, rather than marked done).
- [x] Findings are written up (pass/fail, evidence for the above two checks, any remaining issue) as this
      ticket's outcome.

## Outcome

Two live runs against `qwen3.6:35b-a3b`, `CHEF_OLLAMA_THINK=true`, `--keep-workspaces --json`, run
back-to-back on 2026-08-21:

| Run | Workspace | `passed` | `agent_success` | `verifier_success` | `changed_files` | `error` |
|---|---|---|---|---|---|---|
| 1 | `benchmark-runs/20260821-005751/vague_feature_request_real_repo` | false | false | false | `[]` | "Agent exceeded the 600s case timeout; External verifier failed" |
| 2 | `benchmark-runs/20260821-010813/vague_feature_request_real_repo` | false | false | false | `[]` | "Agent exceeded the 600s case timeout; External verifier failed" |

Both runs still fail, but neither fails via the checkpoint-spiral pattern this effort targets:

- **No nested-checkpoint chain in either run.** Run 1 declared exactly one checkpoint
  ("Explore the codebase...") whose rollup was rejected 6 times in a row; on the 6th rejection the
  new "converge on something concrete now" pressure line (ticket 03) appeared in the replan prompt,
  and the very next replan produced a real leaf ("Write WEB_INTERFACE_ARCHITECTURE.md synthesizing
  ...") instead of another checkpoint — the agent log's own reasoning explicitly weighs and rejects
  re-declaring a checkpoint at that point ("I do not need to add a checkpoint... Since we are ready
  to decide/build based on the synthesis, a leaf is fine"). Run 2 never even reached a declared
  checkpoint — its equivalent exploration sub-goal was generated as an ordinary branch — but hit the
  same rollup-rejection pattern (5 consecutive `not_complete` verdicts) and the same pressure line
  fired and produced the same convergence (a step to create `web_interface_architecture.md`).
- **No re-derivation of the same exploration findings.** With only one checkpoint (run 1) or none
  (run 2) ever declared, there was no second checkpoint attempt under the same parent for ticket 02's
  evidence-preservation to be exercised by this case — the scenario ticket 02 guards against (a
  rollup-triggered replan re-reading files a prior attempt already read) did not arise in either run,
  because the pressure counter converged the sub-goal on its first rejection streak before a second
  checkpoint could ever be spawned. (Ticket 02's mechanism itself is separately covered by the unit
  test `test_rollup_replan_preserves_discarded_evidence_in_failure_context`.)
- **No run exhausted its budget on checkpoint churn.** Both runs instead spent their 600s budget on
  ordinary (slow) LLM round-trips: `qwen3.6:35b-a3b` with thinking enabled took 10-30s per call in
  these runs, and by the time each run converged on a concrete "write the architecture doc" step
  (~step 25-29 of the 40-step cap), the 600s wall-clock ran out before the model's `write` tool call
  landed -- `changed_files` is empty in both because no file write ever completed in time, not because
  the plan never converged on one.

**Conclusion: the checkpoint-decision-avoidance-loop incident this effort targets is fixed.** The
remaining failure in both runs is a different, previously-undocumented issue -- this specific
model/case combination's real-world LLM round-trip latency (magnified by `CHEF_OLLAMA_THINK=true`)
outpaces the case's 600s timeout once a plan needs ~25-30 turns to reach implementation, independent
of anything checkpoint-related. That is out of scope for this effort (a case-timeout/throughput
concern, not a planning-loop defect) and is noted here for whoever picks it up next, not treated as
this ticket failing to confirm the fix.
