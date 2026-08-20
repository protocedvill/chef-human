# 01: Unify checkpoint-chain streak across success and failure paths

**What to build:** `_checkpoint_chain_streak` currently only tracks consecutive checkpoint-only
continuation splices (the rollup-*success* path via `_fire_checkpoint_continuation`), keyed by the
checkpoint's parent node. The path that actually produced the observed incident — repeated rollup
*failure* on a checkpoint, triggering `_replan_failing_node`/`replan_subtree` to regenerate that same
checkpoint's children in place — never touches this counter, so a checkpoint stuck oscillating between
"rollup rejected" and "replan produces another checkpoint" is never bounded.

Re-key the streak by the checkpoint node's own identity (not its parent's), and increment/consult it
from both the existing continuation-success call site and the rollup-failure replan path. Trigger
condition (the fresh output is checkpoint-only, no real leaf/branch work) and the limit (3) stay the
same — once crossed, force any checkpoint in the fresh output to become an ordinary leaf, exactly as the
success path already does. A splice/replan that includes at least one non-checkpoint step still resets
the streak to 0.

**Blocked by:** None (can start immediately)

**Status:** done

- [x] `_checkpoint_chain_streak` is keyed by the checkpoint node's own `node_id`.
- [x] A checkpoint whose rollup is rejected repeatedly, and whose replan keeps regenerating it as another
      checkpoint with no intervening real work, is force-demoted to a leaf after 3 consecutive
      checkpoint-only outcomes — reproduced via a unit test constructing that exact sequence against a
      mocked `ReActLoop`/`Planner`, not just the pre-existing success-path case.
- [x] The existing continuation-success streak behavior (unchanged trigger/limit/reset semantics) still
      passes its existing tests under the new keying.
- [x] A splice/replan that includes any non-checkpoint step resets the streak to 0, on both paths.
