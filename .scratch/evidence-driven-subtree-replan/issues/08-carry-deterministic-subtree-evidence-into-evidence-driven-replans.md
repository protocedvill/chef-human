# 08: Carry deterministic subtree evidence into evidence-driven replans

**What to build:** Ensure the planner receives both the model’s contradiction summary and a deterministic
subtree evidence bundle so the rebuilt subtree does not rediscover or repeat disproven assumptions.

**Blocked by:** 07 (Add a dedicated planner API for evidence-driven subtree rebuild).

**Status:** ready-for-agent

- [ ] The planner receives a contradiction summary in human terms
- [ ] The planner also receives deterministic subtree evidence assembled by the loop
- [ ] Rebuild prompts explicitly carry enough evidence to avoid simply recreating the invalidated subtree

