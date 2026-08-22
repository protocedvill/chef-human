# 04: Add `invalidated` status to plan nodes

**What to build:** Extend the plan status model so evidence-driven invalidation is represented distinctly
from ordinary failure, skipping, and pending work.

**Blocked by:** None (can start immediately).

**Status:** done

- [ ] Plan nodes can represent an explicit `invalidated` status
- [ ] The new status is treated distinctly from `failed` throughout the in-memory plan model
- [ ] Existing plan operations that inspect node status remain coherent with the new status present

