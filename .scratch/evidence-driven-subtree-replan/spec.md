Status: ready-for-agent

## Problem Statement

chef-human can already replan a failing subtree after repeated execution or verification failures, but it
has no first-class way for the acting ReAct agent to say "the current subtree is based on an assumption
that source evidence has now disproved; continuing this plan would be wrong." In the current design, that
kind of contradiction is forced through retry and verifier pressure even when the problem is not that the
agent failed to execute the step, but that the step itself has become invalid in light of newly-read
evidence.

This creates a specific failure mode in real runs: the agent may correctly discover that the repository's
source of truth contradicts the current step or subtree decomposition, but the loop has no dedicated
control path for "discard this subtree and replan from the new evidence." Instead, the run keeps trying
to satisfy the old step wording, or forces the verifier to reject a technically correct implementation
until the subtree is mutated toward the wrong abstraction. The benchmark analysis of the transport-layer
failure in `vague_feature_request_real_repo` exposed this gap clearly: the agent found the real protocol,
but the plan/verifier contract had no explicit invalidation mechanism, so the run devolved into a
verification death march.

The missing capability is not generic uncertainty handling. It is a precise, evidence-triggered subtree
invalidation path with clear semantics: the acting node's current subtree is discarded, the evidence that
proved it wrong is preserved, and the planner rebuilds that subtree in place without repeating the
disproven assumptions.

## Solution

Add a real model-visible control tool, `request_replan`, that the ReAct agent can call when fresh,
objective read evidence contradicts the current step or subtree strongly enough that continuing the
existing decomposition would be wasteful or incorrect.

This tool does not let the model mutate the plan directly. Instead, it returns a structured
"replan requested" acknowledgment that `ReActLoop` interprets. When accepted, the loop immediately ends
the acting turn, marks the current node `invalidated`, archives the discarded live subtree off-tree for
history/debugging, preserves the contradiction evidence, and calls a new planner entrypoint to rebuild the
same node's subtree in place. The target scope is always the current acting node or its current subtree;
the model never chooses an arbitrary node id.

This path is intentionally distinct from retry-driven replans. Retry-driven replans mean "this subtree
kept failing to complete." Evidence-driven replans mean "the subtree's assumptions are no longer valid."
The new path therefore gets its own planner framing, per-node budget, UI event, cooldown, and status
language. It still reuses the existing subtree-replacement mechanics and evidence-snapshot seams where
possible, so the feature fits the current architecture instead of bypassing it.

## User Stories

1. As a ReAct agent working on a leaf, I want to explicitly request a subtree replan when fresh source
   evidence disproves the current step's assumptions, so that I do not keep executing against a known-bad
   plan.
2. As a user running chef-human on an unfamiliar codebase, I want the agent to recover from newly-found
   contradictory repository evidence without having to fail the same step repeatedly first, so that real
   progress is faster and less destructive.
3. As a user, I want evidence-driven replans to affect only the current subtree, so that unrelated
   siblings and ancestors are not discarded when one branch's assumptions collapse.
4. As a user, I want the subtree's live children to be discarded and replaced in place, so that the new
   work actually starts from a corrected decomposition instead of being awkwardly appended after the bad
   plan.
5. As a user, I want the planner to receive the contradiction summary and the objective evidence that
   caused the invalidation, so that the rebuilt subtree does not simply reproduce the disproven
   assumptions.
6. As a user, I want the invalidated subtree's history preserved somewhere for debugging, so that I can
   understand what the agent used to believe and why that plan was thrown away.
7. As a user, I want the acting turn to stop immediately after a replan request succeeds, so that no
   further tool calls reasoned under the discarded subtree are allowed to execute.
8. As a user, I want the model to be unable to pick arbitrary replan targets, so that this tool cannot be
   abused to jump around the tree or destroy unrelated work.
9. As a user, I want the model to be allowed to request a replan only when it has fresh evidence from
   read-only tools in the current turn, so that `request_replan` cannot become a generic "I changed my
   mind" escape hatch.
10. As a user, I want the contradiction request to cite the specific assumption, child step, or subtree
    idea that was invalidated, so that the planner knows exactly what must not be repeated.
11. As a user, I want evidence-driven replans not to consume the normal retry/replan budget, so that a
    subtree invalidated by better evidence is not treated as if it had merely flailed.
12. As a user, I want evidence-driven replans to have their own per-node budget, so that the model cannot
    invalidate the same subtree forever.
13. As a user, I want a node that was just evidence-replanned to avoid immediately triggering a second
    retry-based replan for the same condition, so that the loop does not churn between two control paths.
14. As a user, I want the UI to distinguish "replanned after repeated failures" from "replanned because
    source evidence invalidated the subtree," so that live runs and postmortems remain legible.
15. As a user, I want invalidated nodes to have an explicit status distinct from `failed`, so that the
    plan communicates the difference between "execution failed" and "the plan was superseded by better
    evidence."
16. As a user, I want completed descendants from a discarded subtree to be preserved only when the new
    planner output explicitly continues them, so that valid exploration can survive while stale structure
    does not.
17. As a maintainer, I want the new feature to use the existing `Planner` and `ReActLoop` seams rather
    than a tool mutating the plan directly, so that orchestration logic stays in one place.
18. As a maintainer, I want the off-tree archive to be persisted in a backward-compatible way, so that
    benchmark replay and plan loading do not break when the new metadata appears.
19. As a maintainer, I want `request_replan` to be a sole-call control action, so that ordering and
    rollback semantics are simple and auditable.
20. As a maintainer, I want the planner's new entrypoint to be framed around invalidated assumptions
    rather than execution failure, so that prompt semantics match the actual reason for the subtree
    rebuild.
21. As a maintainer, I want unit and integration-style tests for gating, archival, subtree replacement,
    cooldowns, and planner call shape, so that this control-path feature is proven at the orchestration
    seam where the real bugs happen.

## Implementation Decisions

- **New acting-model tool**: expose a real model-visible tool named `request_replan` with the argument
  contract `reason` plus `evidence_summary`. The model never supplies a target node id.
- **Structured control result**: the tool returns a machine-distinct acknowledgment rather than plain
  prose, and `ReActLoop` branches on that result.
- **Loop-owned mutation**: the tool itself does not modify the plan. `ReActLoop` remains the sole owner
  of execution-time tree mutation and is responsible for archiving the old subtree, updating statuses,
  enforcing gating, and invoking the planner.
- **Scope of invalidation**: an accepted request invalidates the current acting node's subtree and
  rebuilds it in place. This is replacement semantics, not checkpoint-style continuation and not a
  whole-plan replan.
- **Immediate turn termination**: when `request_replan` is accepted, the current acting turn ends
  immediately. The same assistant response cannot mix ordinary tool calls with this control action.
- **Sole-call rule**: `request_replan` must be the only tool call in its assistant message. This keeps
  control-plane actions separate from work-plane actions.
- **Evidence gate**: the loop only honors the request if the current turn includes fresh read-only,
  evidence-producing tool output and the contradiction cites specific files/lines or equivalent current
  evidence. Generic uncertainty is not sufficient.
- **Planner API separation**: add a dedicated planner entrypoint for evidence-driven subtree rebuild, with
  framing equivalent to "the previous subtree has been invalidated by new evidence; rebuild this same node
  without repeating the disproven assumptions." This path is separate from ordinary retry/failure replans
  and separate from checkpoint continuation.
- **Evidence payload**: the planner receives both the model-authored contradiction summary and deterministic
  evidence assembled by the loop, reusing the subtree-evidence snapshot shape already used to preserve
  read findings across subtree replans and checkpoint continuation.
- **Preserved history, clean live tree**: discarded live descendants are removed from the executable plan
  tree, but archived off-tree in `Plan` metadata keyed by the stable `node_id` of the invalidated node.
  The archive stores enough structure and reason text to support debugging, replay, and UI inspection.
- **New explicit status**: add `StepStatus.invalidated` to distinguish evidence-driven invalidation from
  `failed`, `skipped`, and `pending`.
- **Continuation of prior valid work**: completed descendants from the discarded subtree are not preserved
  automatically. They may survive only if the rebuilt subtree explicitly continues them via the existing
  identity-carry-forward mechanism.
- **Separate budgets**: evidence-driven replans get their own per-node counter and config budget, separate
  from `RetryManager`'s retry-driven `replan_count`.
- **Default evidence-replan limit**: the default per-node evidence-driven replan budget is `1`, tighter
  than ordinary retry-driven replans because this tool is a high-authority escape hatch.
- **Cooldown between control paths**: after a node is evidence-replanned, the loop temporarily suppresses
  an immediate retry-driven replan on that same node so the new subtree has a chance to execute before a
  second control path fires.
- **UI distinction**: add a distinct UI event for evidence-driven replans instead of reusing the existing
  generic replan signal.
- **Persistence migration**: new metadata and statuses are added in backward-compatible optional fields
  first so existing benchmark loaders and plan serializers continue to function while the feature lands.
- **Prompt guidance for the acting model**: the main agent prompt explicitly teaches that
  `request_replan` is only for cases where fresh source evidence contradicts the current step or subtree
  strongly enough that continuing would be wrong or wasteful.
- **No arbitrary target selection**: the target node is always the current acting node or subtree known by
  the loop. The acting model cannot invalidate unrelated branches.

## Testing Decisions

- Good tests here verify externally visible orchestration behavior rather than internal implementation
  detail. The core question is not "how was the branch archive stored internally" but "did the old
  subtree get invalidated, archived, and replaced under the right evidence and control rules."
- **Primary seam: `ReActLoop` orchestration tests.** Extend the existing `tests/test_agent/test_react_loop.py`
  coverage to verify:
  - a model-issued `request_replan` is accepted only when the current turn has fresh read evidence and a
    contradiction grounded in current facts
  - `request_replan` must be the sole call in a response
  - the acting turn ends immediately after acceptance
  - the current subtree is archived and replaced in place
  - the current node status becomes `invalidated` before the new subtree is installed
  - the new planner entrypoint is called with the contradiction reason plus deterministic subtree evidence
  - the UI emits the evidence-replan event
  - retry-driven replans are cooled down immediately afterward for that same node
- **Planner seam tests.** Extend `tests/test_agent/test_planner.py` to verify:
  - the new `invalidated` status round-trips safely through plan data structures
  - the new evidence-driven subtree rebuild entrypoint is prompt-shaped differently from ordinary failure
    replans and from checkpoint continuation
  - archived invalidation metadata can coexist with the existing serialized plan format without breaking
    loaders
  - rebuilt subtrees can explicitly continue prior completed descendants through the existing identity
    mechanism, but do not preserve them automatically
- **Tool/registry tests.** Extend registry/tool tests to verify:
  - `request_replan` is registered as a real tool
  - its result contract is structured rather than plain prose
  - it is treated as a control-plane action and not as an ordinary mutating repo tool
- **Integration-style loop test.** Add at least one test that drives `ReActLoop` through an actual
  model-emitted `request_replan` flow end-to-end with mocked planner/backend/tool registry behavior. This
  is required because the failure mode being fixed lives in the interaction between parser, loop, planner,
  and evidence bookkeeping, not in any single isolated method.
- **Prior art.** The closest existing prior art is the current subtree replan path, checkpoint
  continuation, retry accounting, and evidence-carry-forward behavior already covered in
  `tests/test_agent/test_react_loop.py` and `tests/test_agent/test_planner.py`. The new tests should be
  added alongside those seams rather than inventing a new test harness.

## Out of Scope

- Reactive harness-injected checkpoint creation.
- Whole-plan invalidation triggered by the acting model.
- Letting the model choose arbitrary target node ids for replans.
- Mixing `request_replan` with ordinary tool calls in the same assistant response.
- A generic "uncertain, please replan" escape hatch with no evidence requirement.
- Changing checkpoint semantics or ordinary retry-driven replan semantics beyond the minimum cooldown
  needed to prevent double-replan churn.
- Persistent cross-session resume semantics for archived invalidation history beyond backward-compatible
  plan serialization support.
- Ticket breakdown or implementation of the feature itself; this spec only defines the behavior and
  architecture.

## Further Notes

- This feature is deliberately aimed at the exact class of bug where primary-source repository evidence
  outranks a mistaken descendant step or verifier expectation, but the loop currently has no way to say
  so explicitly.
- The design should be implemented so that repository source-of-truth continues to outrank derived plan
  wording when the two conflict.
- The archived invalidation history is for observability and debugging, not for live execution. The live
  plan tree should remain clean, executable, and focused on the current best-known decomposition.
