Status: ready-for-agent

## Problem Statement

chef-human's planner (`chef_human/agent/planner.py`) produces a single flat, linear list of `PlanStep`s
for every task, from a trivial one-file fix to a large greenfield build. This forces two things that
don't belong together: the harness has no way to represent "this step is itself a large sub-problem
that needs its own breakdown" — a large task either gets crammed into unnaturally coarse steps, or the
planner produces many shallow steps with no structure connecting related ones. Verification suffers the
same flattening: every step is checked the same way regardless of whether it's a single concrete action
or something that should really be judged by whether its own sub-parts succeeded.

This also underlies a documented, unfixed bug: `_step_evidence_key` (`chef_human/agent/react_loop.py`)
keys accumulated evidence by the exact step-description string, so any replan that rewords a step orphans
all evidence accumulated under the old wording — even when the step's real-world goal didn't change.

The user wants an agent that can take a large, greenfield request and break it down significantly — with
detail handled at the leaves and overarching implementation handled at higher levels — the way the
`/code-review ultra` tree of reviewers already does for review, rather than forcing everything into one
linear list.

## Solution

Replace the flat `Plan`/`PlanStep` model with a recursive `Plan`/`PlanNode` tree. A node is either a
**leaf** (resolves to exactly one tool call) or a **branch** (decomposed into child nodes). A simple task
that doesn't need decomposition just produces a depth-1 tree — one code path handles every task size, not
a special case for "small" vs. "large."

Each node carries a stable identity (a UUID assigned at creation, surviving replans) instead of being
keyed by its description text, which is what actually fixes the evidence-orphaning bug: evidence is keyed
by `node_id`, not wording. Leaves are verified individually, as today; branches are verified by an
additional rollup check that can reject an all-children-complete branch if their combined effect still
misses the branch's real goal — the tree shape alone is never proof of completion.

Generation is iterative: one call decomposes one branch into its immediate children, recursively, with
each call seeing the branch's ancestor chain. A human operator gets one whole-tree review pass before
execution starts, with nodes the planner was uncertain about flagged inline; headless/benchmark runs
never involve a human, and use finer decomposition instead of flagging when uncertain. `RetryManager`
moves from a single run-wide failure counter to a per-node counter, so a struggling branch's retries
don't bleed into an unrelated one and a sibling's replan doesn't wipe out a different node's failure
history.

This is a design/architecture change only — implementing it is a separate, later effort. Runtime
recursive sub-agent execution (spawning nested LLM agents to actually *execute* their own decomposed
branches, the way `/code-review ultra` spawns reviewer agents) is explicitly deferred: this spec only
reshapes the plan data structure and its generation/verification semantics under the existing single-
`ReActLoop` executor, which still runs one leaf at a time.

## User Stories

1. As a developer giving chef-human a large, underspecified greenfield task, I want the agent to break
   the task down into a tree of sub-goals rather than a flat list, so that the plan's structure reflects
   the task's actual shape instead of forcing everything into steps of the same granularity.
2. As a developer giving chef-human a small, well-scoped task, I want the plan to still be simple (a
   shallow tree, effectively a flat list), so that small tasks aren't burdened with unnecessary tree
   ceremony.
3. As a developer running chef-human interactively, I want to review the whole proposed plan tree once
   before execution starts, with the parts the planner was unsure about clearly flagged, so that I can
   catch a bad decomposition before the agent spends turns executing it.
4. As a developer reviewing a flagged node, I want to edit its description, mark it as a leaf myself
   (overriding further decomposition), ask the planner to redecompose it with my guidance, or reject the
   whole plan, so that I have real editorial control rather than an approve/reject binary.
5. As a developer running chef-human headless (benchmarks, CI), I want the planner to resolve its own
   uncertainty by decomposing further rather than blocking on a human that isn't there, so that
   unattended runs never stall waiting for input that will never come.
6. As a developer whose task needed a replan partway through, I want only the failing part of the plan to
   be regenerated — not the whole tree — so that unrelated, already-correct branches aren't discarded or
   re-attempted for no reason.
7. As a developer whose task needed a replan, I want the node that failed to keep the same identity
   before and after the replan, so that a rewording of "the same underlying goal" doesn't wipe out the
   evidence and history already accumulated for it.
8. As a developer whose plan includes a branch (a sub-goal broken into children), I want that branch's
   completion to be checked against its own real goal, not just "did all its children individually
   finish," so that a flawed decomposition that leaves a coverage gap gets caught instead of silently
   passing.
9. As a developer whose branch fails its own rollup check even though all children completed, I want the
   system to treat that the same way as a leaf failing — replanning just that node, not the whole tree —
   so that decomposition mistakes are recoverable the same way execution mistakes are.
10. As a developer reading the agent's reasoning turn-by-turn, I want to see where I am in the plan tree
    (my current position's ancestors and siblings) without having the entire tree dumped into context
    every turn, so that the model's prompt budget isn't consumed by irrelevant, distant parts of a large
    tree.
11. As a developer whose task has failed and retried several times in one branch, I want that branch's
    retry/replan pressure to be tracked separately from unrelated branches elsewhere in the tree, so that
    one hard sub-problem doesn't exhaust the whole run's retry budget or falsely reset another branch's
    count.
12. As a developer whose branch has exhausted its retry/replan budget, I want execution to continue on
    the rest of the tree rather than the whole run aborting, so that one intractable sub-problem doesn't
    block unrelated progress elsewhere.
13. As a developer running interactively when a branch exhausts its retry/replan budget, I want a
    notification I can act on when convenient — editing the node, asking for a guided redecomposition, or
    aborting the whole run — rather than the session freezing mid-run waiting on me, so that an otherwise
    unattended run doesn't stall on one escalation.
14. As a developer, I want the investigative/execution-step verifier heuristics that fixed prior
    false-escalation bugs to keep working for leaf verification exactly as they do today, so that this
    refactor doesn't reintroduce bugs that were already hard-won fixes.
15. As a developer watching a long headless run, I want some visibility (a log line) if a branch is
    decomposing unusually deep without converging to leaves, so that a spiraling decomposition isn't
    silently invisible, even though there's no hard depth cap forcing it to stop.
16. As a maintainer of chef-human's test suite, I want the existing flat-plan-shaped tests to have a clear
    target shape to migrate toward, so that the 19 call sites asserting `plan.steps[i]` have a documented
    replacement pattern to follow (even though this spec doesn't sequence that migration itself).
17. As a maintainer, I want `RetryManager`'s public interface to still cleanly separate "should we retry,
    replan, or escalate" from plan/step bookkeeping, so that `ReActLoop` remains the only place that maps
    retry decisions onto tree mutations, consistent with `RetryManager`'s current fully-decoupled design.
18. As a maintainer, I want the plan tree to remain ephemeral (not session-persisted), consistent with
    today's behavior, so that this refactor doesn't silently expand scope into session-resume semantics
    that were never part of the ask.

## Implementation Decisions

**Data model.** `PlanStep` is replaced in place by a recursive `PlanNode`: a stable `node_id` (UUID,
assigned at creation and carried forward across replans), a description, a status, and zero or more
child `PlanNode`s. `Plan` is kept as the outer container (`goal` + a root `PlanNode`) — this is the
smallest change to the public shape callers (`on_plan`, `format_plan_for_prompt`-equivalent) already key
off. A node with no children is a **leaf**; a leaf must correspond to exactly one tool call. A node with
children is a **branch**. A depth-1 tree (a root with only leaf children) is the natural shape for a
small task — there is no separate "flat mode."

**Generation.** Iterative, per-node expansion: one LLM call decomposes one branch into its immediate
children. Each call receives the branch's full ancestor chain (root goal down through every ancestor
description to this branch), so a deeply nested node doesn't drift from the overall task. Sibling
visibility needs no separate passing — one call already returns all of a node's immediate children
together, so they're visible to each other within that call; cross-branch duplication between unrelated
branches elsewhere in the tree is an accepted gap, not solved by this design. Leaf-vs-branch is a field
the LLM emits per child (it's a semantic judgment the harness can't reliably check syntactically),
followed by a lightweight harness-side cleanup pass over every expansion call's output — the same role
`_normalize_steps`'s noise filters (`_ENV_SETUP_RE`, `_EDITOR_MECHANICS_RE`, dedup) play today, just
applied per-call instead of once over a flat list. There is no hard depth cap; convergence relies on the
LLM eventually emitting a leaf for every child.

**Human review.** A one-time whole-tree approval pass after generation completes, before execution
starts — not a mid-generation blocking prompt. The full tree is shown with flags marked inline (not just
the flagged nodes in isolation), so the human has full surrounding context. The action set: **approve**,
**edit** a node's description, **mark-as-leaf** (override further decomposition, discarding any children
it already had), **redecompose-with-guidance** (ask the planner to retry just that branch, steered), and
**reject** (abandon the task). A node the planner was uncertain about is flagged for this review. In
headless/benchmark runs there is no human at all — uncertainty biases toward decomposing further via
different prompting for headless mode, rather than the harness overriding the model's own leaf/branch
signal after the fact.

**Verification.** Leaf verification is unchanged in mechanism from today's `Planner.verify_step` /
`STEP_VERIFY_PROMPT`. Branch (rollup) verification is new: it uses ground-truth repo/file state relevant
to the branch's own goal as primary evidence — mirroring the existing "current file contents are ground
truth" framing — with each child's own verification verdict/reason appended as supporting context, not
the deciding signal. A rollup check can reject a branch whose children are all individually complete;
that rejection is treated exactly like a node failing verification (see Replan, below).

The investigative/execution-step routing heuristic (`_looks_investigative`/`_looks_like_execution_step`)
stays leaf-only, unchanged in mechanism (still matching a node's own description text, now keyed by
`node_id`) — rollup verification never classifies raw per-turn tool evidence the way leaf verification
does, so there's no equivalent ambiguity for it to resolve. The `EditTool` no-op /
ground-truth-outranks-narration fix needs no separate duplication into rollup verification's prompt — the
resolved rollup design (ground-truth primary, child verdicts supporting) already is that fix, applied at
the branch level.

**Evidence.** Every node — leaf or branch — has its own evidence bucket, keyed by `node_id` (replacing
`_step_evidence_key`'s description-string keying, the current bug). Evidence propagates upward at write
time: recording evidence for a node also appends it into every ancestor's bucket immediately, so a
branch's bucket always reflects everything that happened under it without a separate read-time
aggregation step at rollup-verification time.

**Replan.** Scoped to just the failing node's own subtree — structurally confined, not just by
convention: the replan call only ever sees/produces that node's descendants, so siblings and ancestors
elsewhere in the tree are mechanically untouched (same ids, same evidence, no LLM involved in deciding
what counts as "the same goal reworded"). The failing node itself (leaf or branch, including a branch
rejected by its own rollup check) keeps its own `node_id` through the replan — its goal is unchanged,
only how to achieve it is being reconsidered — and its evidence bucket is reset. Its old descendants (if
it was a branch) are discarded entirely. Because evidence propagates upward, a subtree replan also
scrubs every ancestor above the failing node of the now-stale propagated entries that traced back to the
discarded descendants, so no ancestor's later rollup verification sees evidence from a discarded attempt.

**Retry/escalation.** `RetryManager`'s `consecutive_failures` counter and replan budget (`max_replans`)
both move from a single run-wide instance (today's actual behavior — one `RetryManager` per
`ReActLoop.run()` call, not per-step) to per-node counters keyed by `node_id`. `ESCALATE`'s meaning
changes accordingly: today it aborts the whole run; with a tree, a node exhausting its own budget
escalates just that node/subtree. In headless mode the node is marked failed and execution continues on
the rest of the tree — the root's own eventual rollup verification then reflects the incompleteness
rather than the run terminating. In interactive mode, this is async, not a blocking prompt: since
`ReActLoop` dispatch is already sequential (one leaf at a time, no concurrent tool dispatch — see the
comment above the dispatch loop in `react_loop.py`), there's no real concurrency for a notification to
let something "keep running in the background" past. The harness applies the same default fallback as
headless (mark failed, continue to the next node in tree order) and surfaces a notification; the human's
involvement is optional and retroactive — **edit** or **redecompose-with-guidance** to override the
already-applied default and retry, or **reject** to abort the whole run. (**Mark-failed-and-continue**
isn't a separate human action here — it's the automatic default already in effect unless overridden.
**Approve** and **mark-as-leaf** from the pre-execution review's action set don't apply — there's nothing
to approve on an already-executed node, and granularity isn't the problem for a node that failed at
whatever granularity it was already at.)

**Slow-convergence safety net.** Purely observational: the harness tracks expansion depth/node-count per
subtree during generation and logs a warning past some threshold. This is a separate, independent counter
from `RetryManager` — decomposition (which has no notion of "failure," a branch that's gone 40 levels
deep hasn't failed anything, it just hasn't converged) and execution are distinct phases, and
`RetryManager` stays scoped to the latter. No behavior change: generation is never stopped or forced to
commit to a leaf early.

**Main-loop prompt rendering.** `Planner.format_plan_for_prompt`'s hardcoded linear "Step N" numbering is
replaced with adaptive collapse, recomputed fresh every turn (same pattern as today — the tree mutates
between turns via completion, replan, and rollup verification, so there's no benefit to incremental
maintenance). The active path — the current leaf, its full ancestor chain, and each ancestor's other
immediate children — is shown expanded (full description + status). Every other subtree, whether already
completed or not yet reached, collapses to one line (description + status marker only). One uniform rule
(is this node an ancestor, the current node, or a sibling of one of those) rather than different
treatment for completed-vs-pending, even though most of the budget savings come from collapsing large
not-yet-reached subtrees rather than small completed ones.

**UX prototype.** The flagged-node review pass's action set and whole-tree-with-inline-flags view were
validated against a throwaway CLI mockup on branch `prototype/flagged-node-review-ux` (commit `6a84c82`)
— referenced here as the primary source for that decision rather than re-derived in this spec.

## Testing Decisions

Good tests here exercise external behavior (does the tree end up in the right shape, does evidence/replan
scoping actually isolate what it should) rather than internals (exactly how a node walks its children
in memory). Three existing seams cover this refactor; no new seams are needed — the tree reshapes what
flows through these surfaces, not where testing happens:

- **`Planner`'s public interface** (`generate_plan`-equivalent, per-node expansion, `verify_step`, rollup
  verification) with a mocked `LLMBackend` — the existing pattern in `test_planner.py`, exercised against
  tree shapes (a depth-1 tree for the small-task case, deeper trees for decomposition, replan scoping)
  instead of a flat list.
- **`RetryManager`** — pure unit tests, no LLM involved, the existing pattern in `test_retry.py`. New
  coverage: per-`node_id` isolation (one node's failures don't affect another's counter; a sibling's
  replan doesn't reset an unrelated node's count), and per-node `ESCALATE` behavior.
- **`ReActLoop` end-to-end** with a stub backend/UI — the existing pattern in `test_react_loop.py`. Covers
  node-id-keyed evidence propagation and subtree-replan scrubbing, mid-execution escalation's
  default-fallback-then-notification behavior, and adaptive tree-prompt rendering, all as integration
  behavior observable through the loop's public `run()` rather than by inspecting internals directly.

## Out of Scope

- **Implementation.** This spec describes the target architecture; building it is a separate, later
  effort.
- **Runtime recursive sub-agent execution** — spawning nested LLM agents to decompose *and execute* their
  own branches (the way `/code-review ultra` spawns reviewer agents). This spec only reshapes the plan
  data structure and generation/verification semantics under the existing single-`ReActLoop` executor,
  which still executes one leaf at a time. Deferred future work, not ruled out permanently.
- **Persisting the plan tree across sessions.** Stays consistent with today's ephemeral (non-persisted)
  `Plan` — `chef_human/agent/persistence.py` has zero references to `Plan`/`PlanStep` today, and a
  resumed session already replans from scratch.
- **Migration/rollout sequencing** for the 19 existing test call sites that assert flat `plan.steps[i]`
  indexing (`test_planner.py`, `test_react_loop.py`, `test_textual_tui.py`), and for
  `Planner.format_plan_for_prompt`'s current hardcoded linear numbering. This spec describes the target
  shape and notes the impact; sequencing how to get there is left to the implementation effort.
- **Exact prompt/interface wording** for rollup verification and per-node expansion calls. This spec
  describes the mechanism (what evidence is seen, what a call returns); the literal prompt text is an
  implementation-time concern.

## Further Notes

This spec is the collapsed output of a `/wayfinder` map ("Recursive PlanNode Tree ADR",
`.scratch/planning-tree-adr/map.md`) that resolved nine decision tickets
(`.scratch/planning-tree-adr/issues/01` through `09`) one at a time across several sessions. Each
Implementation Decision above traces to one of those tickets; consult the linked issue file for the full
reasoning and rejected alternatives behind a given decision if more detail is needed than this spec
carries.

`CONTEXT.md` at the repo root holds the settled glossary this spec uses throughout: `PlanNode`, leaf,
branch, decompose/decomposition, leaf verification, rollup verification, flagged, current leaf.
