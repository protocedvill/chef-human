# 01: Register the `request_replan` tool

**What to build:** Expose a real model-visible `request_replan(reason, evidence_summary)` tool that the
acting model can call as a structured control action, with a machine-distinct acknowledgment result, but
without yet changing plan behavior.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] The acting model can emit a real `request_replan` tool call with `reason` and `evidence_summary`
- [ ] The tool returns a structured acknowledgment result rather than plain prose
- [ ] The tool is wired into the same registry/parsing path as other model-visible tools

