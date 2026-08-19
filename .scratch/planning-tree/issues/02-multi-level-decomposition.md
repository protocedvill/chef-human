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

**Status:** ready-for-agent

- [ ] Generation recurses: a branch's expansion call returns children that may themselves be branches, decomposed by further calls
- [ ] Each expansion call is given its branch's full ancestor chain
- [ ] Each returned child carries an LLM-emitted leaf/branch signal, followed by a harness cleanup pass (env-setup/editor-mechanics filtering, dedup) applied per-call
- [ ] No hard depth cap exists anywhere in the generation path
- [ ] `current_leaf()` correctly walks a multi-level tree in DFS pre-order during execution
- [ ] A task whose scope clearly warrants decomposition (e.g. the `stretch` benchmark case, `inventory_refactor`) produces and executes a tree deeper than one level
