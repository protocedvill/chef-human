# 05: Add archived subtree history to the plan model

**What to build:** Preserve discarded live subtrees off-tree, keyed by the stable node identity, so a
subtree invalidated by better evidence is still inspectable later without polluting the live executable
tree.

**Blocked by:** 04 (Add `invalidated` status to plan nodes).

**Status:** done

- [ ] The plan can archive discarded subtrees keyed by the invalidated node’s stable identity
- [ ] Archived history is stored off-tree rather than as live children of the node
- [ ] The archive retains enough structure and reason context to support debugging and replay

