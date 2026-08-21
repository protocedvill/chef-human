# 03: Add evidence gating for `request_replan`

**What to build:** Only accept `request_replan` when the current turn has fresh read-only evidence and the
contradiction is explicitly grounded in current facts about the repo or files under investigation.

**Blocked by:** 01 (Register the `request_replan` tool), 02 (Enforce `request_replan` as a sole-call control action).

**Status:** ready-for-agent

- [ ] The loop requires fresh read-only, evidence-producing tool output from the current turn before accepting `request_replan`
- [ ] The contradiction request must cite the specific assumption, child step, or subtree idea being invalidated
- [ ] Ungrounded or generic uncertainty does not qualify for an accepted evidence-driven replan

