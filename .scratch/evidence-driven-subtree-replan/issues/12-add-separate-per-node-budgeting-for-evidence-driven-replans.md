# 12: Add separate per-node budgeting for evidence-driven replans

**What to build:** Give evidence-driven replans their own per-node budget and accounting, independent of
ordinary retry/replan pressure.

**Blocked by:** 10 (Execute accepted evidence-driven replans in `ReActLoop`).

**Status:** ready-for-agent

- [ ] Evidence-driven replans are counted separately from ordinary retry-driven replans
- [ ] The budget is tracked per node rather than globally across the run
- [ ] The default evidence-driven replan cap is bounded and configurable

