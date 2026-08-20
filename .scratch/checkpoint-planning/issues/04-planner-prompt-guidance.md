# 04: Planner system-prompt guidance for checkpoint usage

**What to build:** `PLANNER_SYSTEM_PROMPT` (and/or the relevant per-call prompt in `planner.py`) teaches
the planner, explicitly: when it doesn't know how to implement something, plan the steps that would gain
the missing knowledge or context first, then end that phase with a checkpoint — rather than placing
checkpoints reflexively, or chaining them into similar/redundant states with no real work done between
them. This is a nudge on top of (not a substitute for) ticket 03's structural guard, per this repo's
documented history of prompt-only guidance being insufficient on its own for the small local models it
targets.

**Blocked by:** 01

**Status:** done

- [x] The planner's system prompt includes explicit guidance on when and how to use a checkpoint,
      including the "gain the knowledge first, then checkpoint" framing and a caution against reflexive or
      redundant chaining.
- [x] Test coverage at the `Planner` seam (`tests/test_agent/test_planner.py`): the guidance text is
      present in the system message sent for plan generation; can run independent of tickets 02/03 since it
      only depends on the checkpoint type existing (ticket 01).
