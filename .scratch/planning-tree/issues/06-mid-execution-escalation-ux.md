# 06: Mid-execution escalation UX (interactive)

**What to build:** When a node exhausts its retry/replan budget during an interactive run, don't freeze
the session waiting on the human — apply the same default fallback as headless mode automatically (mark
the node failed, continue execution at the next node in tree order), and surface a notification the human
can act on retroactively.

Since `ReActLoop` dispatch is already sequential (one leaf at a time, no concurrent tool dispatch), there
is no real concurrency for a notification to let something "keep running in the background" past — the
default fallback already applies before the human sees the notification. The human's involvement is
optional and comes after the fact: **edit** the node's description, **redecompose-with-guidance** (ask
the planner to retry the subtree with a hint), or **reject** to abort the whole run. There's no
"mark-failed-and-continue" as a separate action — that's the automatic default already in effect unless
overridden.

**Blocked by:** 05 (RetryManager per-node counters + headless escalation)

**Status:** ready-for-agent

- [x] In interactive mode, a node exhausting its budget gets the same automatic fallback as headless (marked failed, execution continues) — the run never blocks waiting on the human
- [x] A notification of the escalation is surfaced to the human, referencing the specific node
- [x] The human can retroactively edit the node's description and have it retried
- [x] The human can retroactively trigger a guided redecomposition of the node's subtree
- [x] The human can reject, aborting the whole run
- [x] Overriding an already-applied default correctly reverses it (e.g. an edit-and-retry un-marks the node as failed and re-attempts it)
