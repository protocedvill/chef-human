# 10: Execute accepted evidence-driven replans in `ReActLoop`

**What to build:** When an accepted `request_replan` arrives, end the current turn immediately, mark the
current node invalidated, archive and discard its live subtree, and install a rebuilt subtree in place.

**Blocked by:** 03 (Add evidence gating for `request_replan`), 04 (Add `invalidated` status to plan nodes), 05 (Add archived subtree history to the plan model), 07 (Add a dedicated planner API for evidence-driven subtree rebuild), 08 (Carry deterministic subtree evidence into evidence-driven replans).

**Status:** ready-for-agent

- [ ] An accepted `request_replan` ends the current acting turn immediately
- [ ] The current node is marked invalidated before the replacement subtree is installed
- [ ] The old live subtree is archived and discarded
- [ ] The planner is called to rebuild the same node’s subtree in place
- [ ] The live plan tree after replacement is executable and scoped to the current node only

