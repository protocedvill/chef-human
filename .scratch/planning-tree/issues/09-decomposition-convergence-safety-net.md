# 09: Slow-convergence safety net

**What to build:** Purely observational instrumentation for decomposition that isn't converging to
leaves quickly — the harness tracks expansion depth or node count per subtree during generation and logs
a warning past some threshold. No behavior change: generation is never stopped or forced to commit to a
leaf early, and this stays a separate, independent counter from `RetryManager` (decomposition has no
notion of "failure," it's a distinct phase from execution).

**Blocked by:** 02 (Multi-level decomposition)

**Status:** done

- [x] Generation tracks expansion depth and/or node count per subtree as it decomposes
- [x] A warning is logged when a subtree's decomposition crosses a defined threshold without converging to leaves
- [x] No generation behavior changes as a result of crossing the threshold — no forced leaf, no hard stop
- [x] This tracking is independent of `RetryManager` — it does not consume or affect any per-node retry/replan counters
