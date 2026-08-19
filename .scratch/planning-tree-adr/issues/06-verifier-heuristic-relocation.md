# Relocating the investigative/execution-step verifier heuristics onto tree nodes

Type: grilling
Status: resolved
Blocked by: 01

## Question

CLAUDE.md documents two hard-won false-escalation fixes in today's flat step verifier: the
`_looks_investigative` / `_looks_like_execution_step` routing heuristic (widened to accept a successful
`bash` command as evidence when a step's wording reads as both investigative and execution-shaped), and
`EditTool`'s explicit no-op wording so the verifier doesn't misread "no visible diff" as "not fixed" when
the file's current content is already correct.

Once rollup verification is designed (see [rollup-verification-design](01-rollup-verification-design.md)), decide: do these heuristics
apply only at leaf verification (since leaves are now guaranteed single-tool-call, arguably making the
investigative/execution-step ambiguity *less* likely to arise than it was for today's coarser flat
steps), or does an analogous ambiguity reappear at the rollup level (a branch description that reads as
both "investigate" and "execute" across its children)? Should the fix (widen evidence acceptance,
ground-truth-over-diff-text framing) be duplicated into the rollup verifier's prompt, or does the tree
structure make the original failure mode structurally impossible at the branch level?

## Answer

Both fixes stay leaf-only, for different reasons — neither needs a rollup-level equivalent.

The `_looks_investigative`/`_looks_like_execution_step` routing heuristic exists solely to classify raw
per-turn tool evidence (does a successful `bash` call count as evidence?) before the leaf verifier LLM
call runs. It carries over unchanged in mechanism — still keyword-matching a node's own description
text — just keyed by `node_id` instead of the old description string. It doesn't need an analogous
rollup version: the resolved rollup verification design ([rollup-verification-design](01-rollup-verification-design.md)) never classifies
raw per-turn tool evidence at all — it reads ground-truth repo state and children's already-computed
verdicts, so there's nothing for an equivalent heuristic to route between at the branch level. The
failure mode is structurally inapplicable there, not just less likely.

The `EditTool` no-op / "ground-truth state outranks a turn's narrated evidence" fix needs no separate
duplication into the rollup verifier's prompt — it's already present by construction. Rollup
verification's resolved design (ground-truth state as primary evidence, children's verdicts as
supporting context) *is* this exact fix, already applied at the branch level when that ticket was
resolved.
