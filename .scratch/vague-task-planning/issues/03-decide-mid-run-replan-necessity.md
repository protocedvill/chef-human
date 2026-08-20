Type: grilling
Status: closed
Blocked by: 02

## Question

Given Running the benchmark against the enriched planner's recorded outcome: does the
`vague_feature_request_real_repo` benchmark case now pass?

- If it passes: this map's destination is reached. Close this ticket recording that, add a
  Decisions-so-far line, and close the map — the mid-run "exploration budget exhausted → replan" idea
  (Idea 1) turned out unnecessary and moves to Out of scope for this effort (not ruled out forever, just
  not needed to reach this destination).
- If it still fails: use ticket 02's captured evidence (explore-only plan vs. real-plan-but-execution-
  derailed) to scope what mid-run replan actually needs to do here, then graduate that into fresh
  ticket(s): what "exploration budget" operationally means (step count? distinct read/glob/grep calls
  before any write/edit? a planner-visible signal in the prompt?), where the phase-transition trigger
  lives in `ReActLoop`, and — per the map's Notes — how it calls into the subtree-replan machinery from
  `docs/adr/0001-evidence-carry-forward-across-whole-plan-replan.md` /
  `.scratch/planning-tree/issues/04-evidence-propagation-and-subtree-replan.md` rather than duplicating
  it (check ticket 04's status first: if still `ready-for-agent`, the new ticket(s) here are blocked on
  it landing first).

## Answer

`vague_feature_request_real_repo` still fails after the enriched planner (Benchmark the enriched
planner). Per ticket 02's captured evidence, this is the "real-plan-but-execution-derailed" case, not
"plan still reads as explore-only": the plan's top-level steps are genuinely repo-specific and include
real implementation work, but the run burns its entire 40-step budget stuck replanning one exploration
sub-branch and never reaches the implementation steps. So: **yes, mid-run intervention is still needed**
to reach this map's destination — Idea 1 (exploration-budget-exhausted phase transition) is not moving to
out-of-scope; it graduates to a real ticket. Idea 2 (deterministic repo-context enrichment) is done and
validated as necessary-but-not-sufficient — keep it, don't revert it.

Checked `.scratch/planning-tree/issues/04-evidence-propagation-and-subtree-replan.md` per the map's
Notes, as required before ticketing anything here: still `status: ready-for-agent`, and itself blocked
by its own "03 (Rollup verification)" (a `planning-tree` ticket, unrelated numbering — not this map's
ticket 03), which hasn't landed either. So the subtree-replan machinery this map's mid-run-replan idea is
supposed to call into does not exist yet.

Consequence for scoping: the new ticket this map needs (mid-run "exploration budget exhausted → replan"
in `ReActLoop`) is **blocked on `planning-tree`'s ticket 04 landing first** — per the map's Notes, it
must call into that subtree-replan machinery, not duplicate it, and 04 is what makes a scoped,
evidence-preserving replan of just the stuck branch possible instead of a blunt whole-plan replan (which
`ReActLoop` already has, and which is exactly what's looping unproductively here today — see the "Replanning
after repeated failures" log lines in ticket 02's evidence, all of which regenerate the *same* stuck leaf).

Not ticketing the mid-run-replan implementation itself yet, in this map, for that reason — it would sit
blocked with nothing to do until 04 lands elsewhere. Recording the operational shape it'll need instead,
so whoever picks it up after 04 lands doesn't have to re-derive this from the benchmark evidence:

- **Exploration budget, operationally:** a per-subtree (not global) counter of consecutive leaf
  verification-rejections under the same branch node — the failure mode observed is one branch's leaf
  repeatedly failing verification and re-replanning to an equivalent leaf, not slow-but-varied progress
  across many distinct leaves. A count of `RetryManager`-triggered replans against the *same* `node_id`
  (or its replaced successor under a subtree replan) is the natural signal, already computed for the
  existing 5-consecutive-failure escalation cap — this needs a *lower*, subtree-scoped version of that
  same counter, not a new mechanism from scratch.
- **Trigger location:** `ReActLoop`'s existing "Replanning after repeated failures" call site
  (`react_loop.py`, the code around the `INFO Replanning after repeated failures (step N)` log line) is
  where this counter should be read — once it crosses the threshold for a given subtree, escalate to a
  *scoped* replan of that subtree via ticket 04's subtree-replan call (once it exists) instead of the
  current whole-plan replan.
- **Do not build a parallel replan path.** `ReActLoop` already has a whole-plan replan
  (`update_plan()`) that this exact failure loop is already exercising every 5 steps; the fix is routing
  its trigger through 04's subtree-scoped replan once available, not adding a second, independent replan
  mechanism.

This map's destination (a green `vague_feature_request_real_repo` benchmark run) stays open, now blocked
transitively on `planning-tree`'s ticket 04 (and its own blocker, `planning-tree`'s "03 Rollup
verification"). Leaving this map open rather than closing it, since the destination — a real fix, not
just a diagnosis — hasn't been reached; the diagnosis is what this ticket's "grilling" type asked for,
and it is now settled as fact, not left as an open question.
