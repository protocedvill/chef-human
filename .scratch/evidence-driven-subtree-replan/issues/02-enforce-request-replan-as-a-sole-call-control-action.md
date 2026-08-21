# 02: Enforce `request_replan` as a sole-call control action

**What to build:** Treat `request_replan` as a control-plane action that cannot be combined with ordinary
work tools in the same assistant response.

**Blocked by:** 01 (Register the `request_replan` tool).

**Status:** ready-for-agent

- [ ] A response containing `request_replan` plus any other tool call is rejected or treated as invalid
- [ ] `request_replan` is handled as a control action rather than a normal work-step tool
- [ ] The loop’s behavior is deterministic when `request_replan` appears in a response

