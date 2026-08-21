# 07: Add a dedicated planner API for evidence-driven subtree rebuild

**What to build:** Give the planner a dedicated entrypoint for rebuilding a subtree that has been
invalidated by new evidence, framed around disproven assumptions rather than ordinary execution failure.

**Blocked by:** 04 (Add `invalidated` status to plan nodes), 05 (Add archived subtree history to the plan model).

**Status:** ready-for-agent

- [ ] The planner exposes a distinct entrypoint for evidence-driven subtree rebuild
- [ ] Its prompt framing differs from ordinary retry/failure replans
- [ ] The API is shaped for rebuilding the same node’s subtree in place rather than doing checkpoint continuation or whole-plan replacement

