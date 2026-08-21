# 16: Test planner-side evidence-driven subtree rebuild behavior

**What to build:** Prove the planner’s new evidence-driven subtree rebuild entrypoint, evidence feed, and
explicit continuation behavior independently of the live loop.

**Blocked by:** 07 (Add a dedicated planner API for evidence-driven subtree rebuild), 08 (Carry deterministic subtree evidence into evidence-driven replans), 09 (Support explicit continuation of valid completed descendants).

**Status:** ready-for-agent

- [ ] Tests prove the planner’s evidence-driven subtree rebuild path is framed differently from ordinary failure replans
- [ ] Tests prove deterministic subtree evidence is included in the planner-side call
- [ ] Tests prove valid completed descendants are only continued when explicitly requested by the rebuilt subtree output

