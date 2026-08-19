# 05: RetryManager per-node counters + headless escalation

**What to build:** Move `RetryManager`'s failure/replan tracking from a single run-wide instance (today's
actual behavior — one `RetryManager` per `ReActLoop.run()` call, not per-step) to per-`node_id` counters,
so one struggling branch's retry pressure doesn't leak into an unrelated branch, and a sibling's replan
doesn't reset a different node's accumulated failure count.

Both `consecutive_failures` and the replan budget (`max_replans`) become per-node. `ESCALATE`'s meaning
changes accordingly: a node exhausting its own budget escalates just that node/subtree, not the whole
run. In headless mode, the node is marked failed and execution continues on the rest of the tree — the
root's own eventual rollup verification then reflects the incompleteness rather than the run terminating.

**Blocked by:** 04 (Evidence propagation + subtree-scoped replan)

**Status:** done

- [x] `RetryManager`'s failure counter and replan budget are tracked per-`node_id`, not as a single instance-wide counter
- [x] A node's failures/replans don't affect an unrelated node's counters elsewhere in the tree
- [x] A sibling branch's replan doesn't reset a different node's accumulated failure count
- [x] In headless mode, a node exhausting its own retry/replan budget is marked failed and execution continues on the rest of the tree (verified: the run doesn't abort, and unrelated branches still complete)
- [x] `RetryManager`'s public interface stays fully decoupled from `Plan`/`PlanNode` shape — it still only knows about counters and node ids, not tree structure; `ReActLoop` remains the only place mapping retry decisions onto tree mutations
