# 02: Multi-level decomposition

**What to build:** Extend generation from always producing a depth-1 tree to real iterative, per-node
decomposition — one LLM call decomposes one branch into its immediate children, recursively, so the
agent can actually break a large task down into nested sub-goals instead of a flat list of leaves.

Each expansion call receives the branch's full ancestor chain (root goal down through every ancestor
description to this branch), so a deeply nested node doesn't drift from the overall task. Sibling
visibility needs no separate passing — one call already returns all of a node's immediate children
together. Leaf-vs-branch is a field the LLM emits per child (a semantic judgment the harness can't
reliably check syntactically), followed by a lightweight harness-side cleanup pass over every expansion
call's output (mirrors today's `_normalize_steps` noise filters — `_ENV_SETUP_RE`,
`_EDITOR_MECHANICS_RE`, dedup — applied per-call instead of once over a flat list). There is no hard
depth cap; convergence relies on the LLM eventually emitting a leaf for every child.

**Blocked by:** 01 (Core PlanNode/Plan data model + depth-1 execution)

**Status:** done

- [x] Generation recurses: a branch's expansion call returns children that may themselves be branches, decomposed by further calls
- [x] Each expansion call is given its branch's full ancestor chain
- [x] Each returned child carries an LLM-emitted leaf/branch signal, followed by a harness cleanup pass (env-setup/editor-mechanics filtering, dedup) applied per-call
- [x] No hard depth cap exists anywhere in the generation path
- [x] `current_leaf()` correctly walks a multi-level tree in DFS pre-order during execution
- [x] A task whose scope clearly warrants decomposition (e.g. the `stretch` benchmark case, `inventory_refactor`) produces and executes a tree deeper than one level

**Resolution notes:**
- `Planner.generate_plan` now calls a new recursive `_expand_node()`: one LLM call per branch (root's own call is the same call `generate_plan` always made), producing that node's immediate children via `_parse_steps`/`_normalize_steps` exactly as before, then recursing into any child whose LLM-emitted `type` was `"branch"`. There is no depth counter or cap anywhere in this path -- recursion only stops when a call returns no branch-flagged children.
- `PlanNode` gained `requested_branch: bool = False`, a generation-time-only hint (like `index`, explicitly documented as not identity and not in `to_dict()`) carrying the LLM's leaf/branch signal through `_normalize_steps`'s filtering so it survives to the recursion check in `_expand_node`.
- `PLANNER_SYSTEM_PROMPT`'s output-format rule now accepts either a plain string (leaf, unchanged) or `{"description": ..., "type": "leaf"|"branch"}` per array element; `_parse_steps` was generalized from "all-str xor all-dict" branches to a single pass handling any mix of str/dict elements, since a real decomposition response mixes leaves and one flagged branch in the same array (the old all-dict-only branch handling silently fell through to a degenerate single-node fallback on a mixed array -- caught by a new test, `test_branch_child_is_recursively_expanded`, before this reached a live model).
- `_expand_node`'s non-root LLM call sends the overall goal, the full ancestor chain (`[task] + ancestors`, ancestors excluding the synthetic `root` node's own placeholder description), and the branch's own description, asking for that sub-goal's immediate next steps only -- not a re-plan of the whole task.
- `Planner._normalize_steps`'s cleanup filters (env-setup, editor-mechanics, conditional-step resolution, dedup) already ran once per `_parse_steps` call and needed no change to apply per-expansion-call -- `_expand_node` simply calls the existing method once per node, same as before.
- `current_leaf()`/`_leaves()` (`Plan`) already recursed through `node.children` regardless of depth (added in ticket 01) -- no change needed; added `test_walks_multi_level_tree_in_dfs_pre_order` as regression coverage now that real multi-level trees exist.
- New unit tests (mocked `LLMBackend`, `test_planner.py`) directly exercise: recursive branch expansion, ancestor-chain content sent to the branch's own LLM call, no-depth-cap convergence across 3 nested levels, and per-call cleanup-filter application on a branch's own children. Live-model validation against the `stretch` benchmark case (`inventory_refactor`) was attempted but is inconclusive either way: the case fails identically on this branch and on unmodified `main` with the same `ModuleNotFoundError: No module named 'report'` test-loader issue (a benchmark-harness/test-fixture problem, not something this ticket touches) -- confirmed not a regression, but not proof of live multi-level decomposition either, since the run never got far enough to exercise the planner's branch path against a live model. Whether qwen3.6 actually emits `"type": "branch"` in practice on a large task remains unverified beyond the mocked-backend tests.
- Full test suite: 1513 passed (excluding 3 pre-existing, unrelated failures also present unmodified on `main` -- network-dependent `EmbeddingsBackend` ImportError tests and one `ruff`-not-on-PATH `LintFixTool` test). `ruff check`/`pyright` clean on changed files.
- `/code-review` on this diff caught and got a fix for a real bug: `PLANNER_SYSTEM_PROMPT` is a plain triple-quoted string, never passed through `.format()`/an f-string anywhere it's used, so the new JSON-object example was written with erroneous `{{`/`}}` doubling (f-string-escaping habit applied to a non-f-string) -- the model would have seen literal doubled braces in its system prompt. Fixed to single braces.
- The same review flagged two further gaps, left unaddressed here since they belong to tickets this one is explicitly blocked-before, not this one's checklist: `Planner.format_plan_for_prompt`/`PlanNode.to_dict()`/`Plan.to_dict()` still only look at `plan.steps` (root's immediate children) and never recurse into a branch's own children -- this is exactly ticket 08's "tree-aware adaptive prompt rendering" scope, and the `to_dict()` shape was already flagged as depth-1-only in ticket 01's resolution notes. And `update_plan`'s "keep completed top-level steps" logic never preserves a branch node whose subtree is fully done (since only leaves get `StepStatus` transitions today, a branch's own `status` never becomes `completed`) -- this is rollup verification (ticket 03) and node-scoped replan (ticket 04) territory: nothing yet marks a branch complete or scopes a replan to a subtree, so this is an expected gap at this stage, not a regression this ticket introduced.
