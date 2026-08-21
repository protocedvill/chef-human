# 09: Support explicit continuation of valid completed descendants

**What to build:** Allow rebuilt subtrees to preserve prior completed descendant work only when the new
planner output explicitly continues that work, rather than preserving completed descendants automatically.

**Blocked by:** 07 (Add a dedicated planner API for evidence-driven subtree rebuild).

**Status:** ready-for-agent

- [ ] A rebuilt subtree can explicitly continue prior valid completed descendants
- [ ] Completed descendants are not preserved automatically when a subtree is invalidated
- [ ] The continuation path reuses the existing identity-carry-forward mechanism rather than introducing a second one

