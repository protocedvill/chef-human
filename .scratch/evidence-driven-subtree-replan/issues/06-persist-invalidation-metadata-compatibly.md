# 06: Persist invalidation metadata compatibly

**What to build:** Persist the new invalidation status and archive metadata through backward-compatible
optional fields so existing plan serialization and loading continue to work while exposing the new
behavior.

**Blocked by:** 04 (Add `invalidated` status to plan nodes), 05 (Add archived subtree history to the plan model).

**Status:** ready-for-agent

- [ ] The new status and archive metadata can be serialized without breaking the existing top-level plan contract
- [ ] Older plan consumers that do not know about the new metadata continue to load plans safely
- [ ] Benchmark or replay loaders can round-trip the new metadata when present

