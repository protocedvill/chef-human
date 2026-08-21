# 11: Add cooldown between evidence-driven and retry-driven replans

**What to build:** Prevent a node that was just evidence-replanned from immediately triggering a second,
retry-based subtree replan for the same condition.

**Blocked by:** 10 (Execute accepted evidence-driven replans in `ReActLoop`).

**Status:** ready-for-agent

- [ ] A node that was just evidence-replanned does not immediately trigger an ordinary retry-driven replan
- [ ] The cooldown is scoped per node rather than globally across the plan
- [ ] The cooldown still allows later ordinary retry behavior once the rebuilt subtree has had a real chance to execute

