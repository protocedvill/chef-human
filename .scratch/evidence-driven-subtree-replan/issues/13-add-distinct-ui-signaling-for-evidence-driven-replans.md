# 13: Add distinct UI signaling for evidence-driven replans

**What to build:** Surface evidence-driven replans through a dedicated UI event so live runs and logs
distinguish them from ordinary replans after repeated failures.

**Blocked by:** 10 (Execute accepted evidence-driven replans in `ReActLoop`).

**Status:** ready-for-agent

- [ ] The UI has a dedicated evidence-replan event distinct from the generic replan event
- [ ] Live run output can distinguish evidence-driven replans from retry-driven replans
- [ ] Existing UI implementations remain coherent when the new event is introduced

