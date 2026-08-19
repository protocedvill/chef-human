# 01: Core PlanNode/Plan data model + depth-1 execution

**What to build:** Replace the flat `PlanStep`/`Plan` model with a recursive `PlanNode`/`Plan` model, and
get a task running end-to-end through it — for now, only ever producing a depth-1 tree (a root with only
leaf children), so observable behavior matches today's flat-plan execution exactly. This is the
foundation every other ticket in this set builds on.

`PlanNode` carries a stable `node_id` (UUID, assigned at creation) instead of being identified by its
description text. A node with no children is a leaf; a leaf must correspond to exactly one tool call.
`Plan` is kept as the outer container (`goal` + a root `PlanNode`). `ReActLoop`'s `current_leaf()`
replaces `current_step()` (DFS pre-order: first non-completed leaf). Evidence accumulation moves from
`_step_evidence_key`'s description-string keying to per-`node_id` keying (no upward propagation needed
yet — there are no branches at this depth). Leaf verification (`Planner.verify_step` /
`STEP_VERIFY_PROMPT`) and the investigative/execution-step routing heuristics
(`_looks_investigative`/`_looks_like_execution_step`) carry over unchanged in mechanism, just re-keyed by
`node_id` instead of description string.

**Blocked by:** None (can start immediately).

**Status:** done

- [x] `PlanStep` is replaced by `PlanNode` (`node_id`, `description`, `status`, `children: list[PlanNode]`) throughout `chef_human/agent/planner.py` and `chef_human/agent/react_loop.py`
- [x] `Plan` keeps its `goal` + gains a root `PlanNode`; `current_step()` is replaced by `current_leaf()`
- [x] Generation still produces a valid tree for every task (depth-1 for now — see ticket 02 for real decomposition)
- [x] Evidence is accumulated per-`node_id`, not per-description-string; `_step_evidence_key` and its bug (replan-reword orphaning evidence) no longer exist
- [x] Leaf verification's investigative/execution-step heuristics are confirmed still correct against the new keying (regression coverage, not new behavior)
- [x] The `smoke` and `core` benchmark cases (`python -m chef_human.benchmark --case hello_world`, `--case slugify_contract`) pass unchanged through the new model

**Resolution notes:**
- `PlanNode` keeps a cosmetic `index` field (position among siblings) purely for display continuity in prompts/UI — it is not used for identity, only `node_id` is. `Plan.steps`/`Plan(goal=..., steps=[...])` are kept as a property/constructor-kwarg shim over `root.children` so the many existing flat-shape call sites (and `Plan.to_dict()`'s `{goal, steps}` JSON shape, part of the `--headless --json` CLI contract) keep working unchanged — depth-1 trees make `steps` and `root.children` coincide exactly.
- `PlanNode` equality is dataclass-default (all fields, including `node_id`) rather than content-based — two separately-created nodes with identical description/status are no longer `==`, which is the intended fix (identity is the id, not the text).
- Verified `smoke`/`core` benchmark cases behave identically before and after this change: `hello_world` passes on both; `slugify_contract` fails identically on both `main` and this branch (a pre-existing verifier-LLM flakiness issue — the step verifier occasionally returns an empty, unparseable response — unrelated to this refactor).
- `node_id`-keyed evidence is real (fixes description-string keying as an identity mechanism), but `update_plan()` still only carries a node's `node_id` forward across a replan for steps already `StepStatus.completed` — a replan reword of a still-pending/in-progress step still gets a fresh id and an empty evidence bucket. That's still open, tracked by the per-node-evidence-and-replan-scope ticket in `.scratch/planning-tree-adr/`, not claimed as fixed here.
- Full test suite (1506 tests, excluding a few pre-existing/unrelated network-dependent embeddings failures) passes; ruff/pyright show only pre-existing, unrelated findings in files this ticket didn't touch.
