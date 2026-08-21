# 14: Test tool registration, gating, and sole-call enforcement

**What to build:** Prove the `request_replan` tool’s registry behavior, structured result contract,
evidence gate, and sole-call rule independently of the full subtree-replacement flow.

**Blocked by:** 01 (Register the `request_replan` tool), 02 (Enforce `request_replan` as a sole-call control action), 03 (Add evidence gating for `request_replan`).

**Status:** ready-for-agent

- [ ] Tests prove `request_replan` is a real registered tool with the intended argument/result contract
- [ ] Tests prove mixed responses containing `request_replan` plus ordinary tools are rejected or handled as invalid
- [ ] Tests prove ungated or weakly grounded requests are not accepted as evidence-driven replans

