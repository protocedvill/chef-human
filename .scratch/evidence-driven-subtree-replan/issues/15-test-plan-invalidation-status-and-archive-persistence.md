# 15: Test plan invalidation status and archive persistence

**What to build:** Prove the new `invalidated` status, off-tree archive model, and backward-compatible
serialization behavior independently of loop orchestration.

**Blocked by:** 04 (Add `invalidated` status to plan nodes), 05 (Add archived subtree history to the plan model), 06 (Persist invalidation metadata compatibly).

**Status:** done

- [ ] Tests prove the new invalidation status behaves coherently in the plan model
- [ ] Tests prove discarded subtrees are archived off-tree rather than left in the live executable tree
- [ ] Tests prove serialization/loading remains backward-compatible while preserving new metadata when present

