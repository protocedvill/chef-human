# 17: Test end-to-end evidence-driven subtree replan orchestration

**What to build:** Prove the full control path for a model-emitted `request_replan`: accepted gate,
immediate turn stop, subtree archive/replacement, planner rebuild call, cooldown, budgeting, and UI
signaling.

**Blocked by:** 10 (Execute accepted evidence-driven replans in `ReActLoop`), 11 (Add cooldown between evidence-driven and retry-driven replans), 12 (Add separate per-node budgeting for evidence-driven replans), 13 (Add distinct UI signaling for evidence-driven replans), 14 (Test tool registration, gating, and sole-call enforcement), 15 (Test plan invalidation status and archive persistence), 16 (Test planner-side evidence-driven subtree rebuild behavior).

**Status:** ready-for-agent

- [ ] An integration-style test proves a model-emitted `request_replan` flows through the real orchestration path correctly
- [ ] The accepted request ends the acting turn and replaces only the current subtree
- [ ] The archive, cooldown, budgeting, and UI behaviors all appear in the same end-to-end flow
