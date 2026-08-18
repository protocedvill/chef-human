# review_tree: efficacy/efficiency report and BashTool truncation diagnosis

Run analyzed: `tree-review5`, 2026-08-18 01:12:24–02:32:33 (1h20m9s), model
`qwen3.6:35b-a3b`, `think="low"`, `--max-completion-tokens 30000` (the single-shot
default, no retry). Target: the 4-file `tools-subset` scratch fixture used throughout
this session (`diff.py`, `filesystem.py`, `refactor.py`, `shell.py` from
`chef_human/tools/`, copied to a scratch dir). This is the first run against the
function/class-boundary-leaf redesign with cross-file connected context
(commit `0f1ed7b`).

## TL;DR

- **Truncation dropped to 3/120 calls (2.5%)**, down from 21/36 (58%) and 27/29 (93%
  needing retries) in earlier bin-packed-group designs.
- **The truncation that remains is not a sizing problem.** The 3 truncated leaves had
  the *smallest* prompts in the entire run (966–1716 tokens); the largest prompt in the
  run (3062 tokens, `RefactorTool`) mostly completed in 5–8k tokens across four lenses,
  but nearly truncated on a fifth (29,723/30,000) with byte-identical input. This is the
  model getting stuck reasoning at length on specific calls, not exceeding a budget it
  needs — see full diagnosis below.
- **Spot-checked 3 findings against the real code by actually running it: 2 confirmed,
  1 empirically false** despite a specific, confident "failure scenario." The tool
  surfaces real, previously-unknown bugs, but nothing here should be trusted without
  verification.
- **Every single synthesis call (8/8) needed structured-findings recovery** —
  `_merge_child_findings` backfilled findings the model's own JSON array dropped, on
  100% of synthesis nodes in this run, not just some. This is now a reliable, expected
  behavior of this model at synthesis time, not an occasional glitch.
- Total cost: 120 LLM calls, 810,609 tokens (183,528 prompt / 627,081 completion),
  ~40s average per call.
- **Dual-model routing (quick model for leaves, powerful model for synthesis)** was
  implemented and validated live tonight: cuts wall-clock time dramatically (a 17-call
  run finished in 2.5 minutes vs. minutes-per-call with a single strong model), but a
  live run also immediately surfaced a real bug (Ollama hard-errors when `think` is
  sent to a non-thinking model) and shows a real quality tradeoff (a false positive
  slipped through the leaf model that flagged the target function's own documented
  behavior as a bug). See the dedicated section below.

## BashTool truncation: diagnosis

`root/error_handling/shell.BashTool` hit the full 30,000-token completion cap with a
prompt of only 1472 tokens (`shell.py:42-130`, no connected context — the whole
`BashTool` class is fairly self-contained). The other two truncated nodes
(`correctness/diff.DiffStore`, 1716 tokens; `correctness/filesystem.GlobTool`, 966
tokens) show the same pattern: **small inputs, full-budget completions, zero JSON
output recovered.**

Ruled out input size as the cause by comparing against `RefactorTool`, the largest unit
in this run (3045–3062 prompt tokens, roughly 2–3x the truncated nodes):

| Lens | RefactorTool prompt tokens | RefactorTool completion tokens |
|---|---|---|
| modularity | 3062 | 6785 |
| simplification | 3053 | 7771 |
| security | 3036 | 6133 |
| error_handling | 3035 | 5013 |
| state_and_side_effects | 3045 | **29,723** (nearly truncated) |

Same class, same body, same connected context, same lens-independent structure — five
calls with near-identical prompts produced completion lengths spanning 5,013 to 29,723
tokens. That variance can only come from the model's own reasoning trajectory, not the
input. **Truncation here is best modeled as a rare reasoning-loop / non-convergence
failure mode of this specific model under `think="low"`, triggered unpredictably by
certain unit+lens combinations (shell/subprocess code, diff/history state-machine code,
and glob/traversal code all being un-then-re-considered at length), not as a budget
that's too small.**

Practical implication: raising `--max-completion-tokens` further would not reliably
fix this — it would just let the model spin longer before probably still not
converging, burning more time/tokens for the same empty result. The harness's job here
(mark it `truncated`, don't let it silently read as "no findings", recover what
`_merge_child_findings` can from siblings) is already working as designed. If this
needs a further fix, the right lever is model-side: try `think=False` outright for
leaf calls (removing reasoning entirely rather than "low" effort), or add a
"reasoning got too long, bail and retry once with think off" escape hatch specifically
for leaves — not a bigger number.

## Efficacy: spot-check results

Checked 3 of the 93 final findings against the real source, including actually
executing the claimed failure scenario where checkable.

| Finding | Verdict |
|---|---|
| `filesystem.py:269` — `EditTool` reports the *total* occurrence count (`old_content.count(matched_old)`) in its output message even when `replace_all=False` only replaced one occurrence via `.replace(..., 1)` | **Confirmed.** Read the code directly: `count` is computed once from `.count()` before the `replace_all` branch, so a single-occurrence replace still reports "N occurrences." Real bug, not previously found by any prior run. |
| `filesystem.py:63` — `ReadTool` doesn't validate `limit`; `limit=0` silently returns a single blank-newline "success" result instead of erroring or being treated as unlimited | **Confirmed.** Traced the logic: `limit=0` makes `selected = lines[offset-1:offset-1]` (empty), `output = ""`, then the trailing-newline-enforcement step appends `"\n"` — exactly the claimed behavior. Real edge-case bug: a caller passing `limit=0` gets a deceptively successful empty-ish response. |
| `diff.py:18` — claims `splitlines(keepends=True)` plus `difflib.unified_diff`'s default `lineterm='\n'` produces **double newlines** between every diff line | **False.** Actually ran `compute_diff('a\nb', 'a\nc')`: output is `'--- a\n+++ b\n@@ -1,2 +1,2 @@\n a\n-b+c'` — no double newlines anywhere. `lineterm` only affects the `---`/`+++` header lines, not content lines. The *real* subtlety here (not what the model claimed) is that `'b'` has no trailing newline in the input, so `-b` and the following `+c` get concatenated with no separator (`-b+c`) — a genuinely different, more subtle bug than the confidently-stated one. |

2 of 3 real, previously-undocumented bugs; 1 of 3 a fabricated failure scenario stated
with the same confident, specific tone as the real ones. The anti-nitpick prompt design
(require a concrete failure scenario) does not prevent confident hallucination of a
scenario that doesn't reproduce — verification against actually running the code
remains necessary before trusting any individual finding.

## Duplication

7 of 93 findings (7.5%) share a `(file, line)` location with another finding — down
from a much higher rate in the pre-function-boundary-leaf design, where the same
location routinely appeared 2-3x under slightly different freeform category strings.
Tighter, per-class leaves plus the `(file, line, category)` dedup key in
`_merge_child_findings` appear to be working as intended; residual duplication is
mostly legitimate (two different lenses independently flagging the same line for
different reasons).

## Synthesis findings-array reliability: 8/8 needed recovery

Every synthesis node in this run (7 method-level + 1 root) had its own `findings` JSON
array backfilled by `_merge_child_findings` — not just "sometimes," every single one,
totaling 186 recovery events across levels (note: this count double-counts a finding
recovered at method-level and then again at root-level from that same method node, so
it's not a unique-finding count — root's own final deduplicated list is 93). This
confirms last session's diagnosis was structural, not a fluke: at this model/size,
**synthesis calls reliably write a coherent narrative summary and unreliably populate
the parallel structured array**, regardless of truncation status. The merge-recovery
fix (added last session) is doing real, constant work here, not covering an edge case —
treat it as load-bearing, not optional.

## Cost summary

| Metric | Value |
|---|---|
| Total LLM calls | 120 (112 leaves + 8 synthesis) |
| Wall clock | 1h 20m |
| Total tokens | 810,609 (183,528 prompt / 627,081 completion) |
| Avg tokens/call | 6,755 |
| Truncated calls | 3 / 120 (2.5%) |
| Connected-context signatures included | 140 (0 omitted — 30k-token leaf budget was never actually the constraint on connected context in this run) |
| Final findings | 93 |

Units per lens: exactly 16 in all 7 lenses (7 classes in `filesystem.py`, 7 units in
`diff.py`, 1 class each in `refactor.py`/`shell.py`) — decomposition is now driven
purely by code structure, identical across lenses, as designed.

## Recommendations

1. **Don't chase the truncation with a bigger number.** It's not a sizing issue (see
   diagnosis above). If it needs fixing, try `think=False` for leaves, or a
   detect-and-bail-once heuristic, not a higher `--max-completion-tokens`.
2. **Never present a single finding as ground truth.** Even with function-level
   scoping and real cross-file context, 1/3 spot-checked findings this run was a
   confidently-fabricated failure scenario. A verification pass (re-check each
   finding's failure scenario against the actual file, ideally by executing it like
   the checks above) is the natural next capability to build, not more decomposition
   refinement.
3. **Dual-model routing (`synthesis_backend`/`--synthesis-model`, commit `1018fd6`)
   validated live** against `diff.py` (7 units, 2 lenses, 17 calls): leaves on
   `qwen2.5-coder:7b`, synthesis on `qwen3.6:35b-a3b`. Results below.

## Dual-model routing: live validation

**A real bug surfaced immediately on the first attempt**: pairing a thinking-capable
synthesis model with a non-thinking-capable leaf model crashed the whole run —
`ollama._types.ResponseError: "qwen2.5-coder:7b" does not support thinking (status
code: 400)`. This is exactly the failure mode dual-model routing invites (fast/small
models frequently aren't thinking-capable), so it's not an edge case, it's close to
the common case. Fixed in `OllamaBackend`: catch that specific error, downgrade to
`think=False`, retry once, and remember the downgrade for the rest of that backend
instance's life (commit `1d74753`). Re-ran clean afterward — the warning logged
exactly once, then every subsequent leaf call went straight to `think=False`.

**Speed**: dramatic. 17 calls (14 leaves + 3 synthesis) completed in **2.5 minutes**
total — individual leaf calls took 1–5 seconds each (avg completion: 197 tokens),
versus 20–70+ seconds per leaf and ~5,000+ average completion tokens when qwen3.6
handled leaves itself. Zero truncation across all 14 leaves.

**Quality**: mixed. All 3 synthesis nodes still needed `_merge_child_findings`
recovery (same 100% pattern as the single-model run — this looks like a qwen3.6
synthesis-task characteristic, independent of which model produced the leaves it's
synthesizing). But spot-checking the leaves' own output: the *first* finding
(`compute_diff` "returns an empty string when old_content and new_content are
identical") is a **false positive** — that's the function's own documented, intended
behavior per its docstring ("Returns empty string when old and new are identical"),
not a bug. `qwen2.5-coder:7b` also didn't consistently follow the `category` field
convention (`"Bug"`, `"Correctness"` instead of the lens's own id like
`"correctness"`), unlike qwen3.6's leaves in the full run.

**Verdict**: dual-model routing is a legitimate lever for cutting wall-clock time
dramatically (this is the single biggest efficiency change available, far more than
any budget/retry tuning), but the quick leaf model traded real precision for that
speed — expect more false positives needing verification, not fewer. Worth using when
iteration speed matters more than precision (e.g. a first pass to find candidates
worth a slower, more careful second look), not as a drop-in replacement for a
single-strong-model run.
