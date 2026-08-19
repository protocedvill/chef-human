# 03: Rollup verification

**What to build:** A branch's completion check beyond "did all its children individually finish" — a
rollup verification that can catch a flawed decomposition where the children collectively don't cover
the branch's real goal, even though each one passed its own leaf verification.

The rollup check uses ground-truth repo/file state relevant to the branch's own goal as primary evidence,
mirroring leaf verification's existing "current file contents are ground truth" framing — with each
child's own verification verdict/reason appended as supporting context, not the deciding signal. It
produces the same `StepVerdict` type leaf verification already uses. A rollup check can reject a branch
whose children are all individually complete; the tree shape alone is never proof of completion.

**Blocked by:** 02 (Multi-level decomposition)

**Status:** done

- [x] Every branch node, once all its children report complete, goes through a rollup verification call before the branch itself is marked complete
- [x] The rollup check's primary evidence is ground-truth state for the branch's own goal; children's verdicts/reasons are supporting context only
- [x] A branch whose children are all complete can still be rejected by the rollup check (verified with a case constructed to have a genuine coverage gap)
- [x] A branch that's genuinely complete passes rollup verification without false rejection
