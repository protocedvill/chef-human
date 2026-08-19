# Recursive PlanNode Tree ADR

Status: open

## Destination

A design spec / ADR (`docs/adr/`) replacing chef-human's flat `Plan`/`PlanStep` planning model
(`chef_human/agent/planner.py`) with a recursive `Plan`/`PlanNode` tree — see `CONTEXT.md` for the
settled glossary (PlanNode, leaf, branch, decompose, leaf verification, rollup verification, flagged,
current leaf). The spec is the deliverable; implementing it is a separate, later effort.

## Notes

- Domain: chef-human's agent planning/verification subsystem. Consult `CONTEXT.md` (repo root) for
  terminology before writing any ticket resolution.
- Every ticket in this map is a *design decision* for the ADR, not a code change. Plan, don't do.
- Grounding facts already gathered (don't re-derive): `RetryManager` (`chef_human/agent/retry.py`) is
  fully decoupled from Plan/PlanStep shape — no blocker there. `Plan` is not currently session-persisted
  (`chef_human/agent/persistence.py` has zero references) and persistence stays out of scope for this
  effort. `debug_tui.py` already renders via a `rich.tree.Tree` widget. `Planner.format_plan_for_prompt`
  hardcodes linear "Step N" text. 19 test call sites assert flat `plan.steps[i]` indexing
  (`test_planner.py`, `test_react_loop.py`, `test_textual_tui.py`) — the spec notes this as impact, but
  does not prescribe migration (decided: spec stays design-only).
- Settled destination-level decisions (do not re-litigate, see resolution comment on originating
  ticket/session if more detail is needed):
  - Node identity: stable UUID assigned at creation, carried forward across replans — replaces
    `_step_evidence_key`'s description-string keying (the bug documented in `CLAUDE.md`).
  - Leaf = exactly one tool call. Non-leaf completion = child aggregation + rollup verification
    (reusing the existing `StepVerdict` type).
  - Human review = one-time whole-tree approval pass after generation (not mid-generation blocking);
    flagged nodes surface there. Headless mode has no human, so it decomposes further instead of
    flagging.
  - Generation is iterative per-node expansion (recursive branch → children calls), not one whole-tree
    LLM call.
  - Replan scope (on `RetryManager`'s REPLAN action) is just the failing subtree, not the whole tree.
  - No hard depth cap — relies on the "must resolve to a single tool call" leaf criterion to converge.
  - Runtime recursive sub-agent execution (spawning nested LLM agents, like `/code-review ultra`) is
    explicitly deferred future work — this map only reshapes the plan *data structure* and generation/
    verification semantics under the existing single-`ReActLoop` executor.

## Decisions so far

- [Rollup verification design](issues/01-rollup-verification-design.md): ground-truth repo state is the
  primary evidence, children's verdicts are supporting context; a rollup check can reject an
  all-children-complete branch, which routes through the same subtree-replan path scoped at the
  branch's own node_id.
- [Per-node evidence model and subtree replan mechanics](issues/02-per-node-evidence-and-replan-scope.md):
  replan is confined to the failing node's own subtree (siblings/ancestors untouched, no LLM
  id-matching needed); the failing node keeps its node_id and its evidence bucket resets; every node has
  its own evidence bucket with write-time upward propagation, and a subtree replan scrubs the failing
  node's bucket, drops discarded descendants' buckets, and scrubs stale propagated entries from every
  ancestor above it.
- [Flagged-node human review UX](issues/03-flagged-node-review-ux.md): review shows the whole tree with
  flags marked inline; actions are approve, edit description, mark-as-leaf, redecompose-with-guidance,
  reject. Prototype on `prototype/flagged-node-review-ux` (commit `6a84c82`).
- [Iterative per-node expansion protocol](issues/04-iterative-expansion-protocol.md): each expansion call
  gets the full ancestor chain (siblings come free from being returned together in one call); leaf/branch
  is an LLM-emitted field per child with a harness cleanup pass mirroring today's _normalize_steps;
  headless mode's finer-decomposition bias is different prompting from the start, never a harness
  override of the model's signal.
- [Tree-aware rendering for the main-loop LLM prompt](issues/05-tree-prompt-rendering.md): adaptive
  collapse replaces linear numbering — active path (current leaf + ancestors + their siblings) shown in
  full, everything else (completed or not-yet-reached) collapses to one line; recomputed fresh each
  turn, same pattern as today's format_plan_for_prompt.
- [Relocating the investigative/execution-step verifier heuristics onto tree nodes](issues/06-verifier-heuristic-relocation.md):
  both stay leaf-only — the routing heuristic is structurally inapplicable at rollup (which never
  classifies raw tool evidence), and the ground-truth-outranks-narration fix is already baked into the
  resolved rollup verification design, nothing to duplicate.
- [RetryManager failure counting under subtree-scoped replans](issues/07-retrymanager-per-node-failure-counting.md):
  both consecutive-failure counting and the replan budget move from RetryManager's current single
  instance-wide counters to per-node counters keyed by node_id; ESCALATE now means escalating just that
  node/subtree (surfaced to the human interactively, marked failed and the rest of the tree continues
  headless) rather than aborting the whole run.
- [Soft safety net for slow-converging decomposition](issues/08-decomposition-convergence-safety-net.md):
  observational only — logs a warning past a depth/node-count threshold, no behavior change, no hard
  cap; a separate counter from RetryManager, since decomposition (no "failure" concept) and execution
  are distinct phases.
- [Mid-execution escalation UX for a node exhausting its retry/replan budget](issues/09-mid-execution-escalation-ux.md):
  async — the harness applies the same default fallback as headless mode (mark node failed, continue)
  automatically, and surfaces a notification the human can retroactively override with edit /
  redecompose-with-guidance / reject; not a blocking prompt, since the executor is already sequential
  with no concurrency to notify "around."

## Not yet specified

- Exact prompt/interface wording for rollup verification and per-node expansion calls — implementation-
  level detail that will graduate once the surrounding mechanics (rollup verification design, expansion
  protocol) are decided; the spec describes the mechanism, not the prompt text itself.

## Out of scope

- Runtime recursive sub-agent execution (nested LLM agents decomposing *and executing* their own
  branches) — deferred future work beyond this destination, not ruled out permanently; would be a fresh
  map if picked up later.
- Persisting the plan tree across sessions — stays consistent with today's ephemeral (non-persisted)
  Plan.
- Prescribing a migration/rollout plan for the 19 existing flat-shape test call sites — left to the
  implementation effort that follows this spec.
