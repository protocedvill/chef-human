# Iterative per-node expansion protocol and leaf-termination criteria

Type: grilling
Status: resolved

## Question

Generation is settled as iterative per-node expansion: recursive calls, one branch at a time, each
returning that branch's immediate children. What isn't settled is the call contract.

Specifically: what context does each expansion call receive — just the branch's own description, or
also its ancestor chain (so a deeply nested node knows the overall goal it serves) and/or its siblings
(so decomposition doesn't duplicate or gap coverage across a branch's children)? How does an expansion
call signal "this child is a leaf, don't recurse into it further" versus "this child is itself a branch,
recurse" — is that a field the LLM emits per child, or a deterministic check the harness applies (e.g.
does this description still resolve to more than one tool call)? And in headless mode, where uncertainty
biases toward finer decomposition rather than flagging (see `map.md` Notes) — is that bias applied by
the harness overriding a leaf/branch signal the model gave, or by prompting the model differently in
headless mode from the start?

## Answer

Each expansion call receives the branch's full ancestor chain — root goal down through every ancestor
description to this branch — so a deeply nested node doesn't drift from the overall task the deeper the
tree gets. Sibling visibility needs no separate passing: one expansion call already returns *all* of a
node's immediate children together in a single LLM response, so they're inherently visible to each other
within that call. Cross-branch duplication between unrelated branches elsewhere in the tree isn't fully
solved by this (only the shared ancestor context helps, indirectly) — an accepted gap rather than
requiring full-tree visibility on every call.

Leaf-vs-branch is a field the LLM emits directly per child (e.g. `"leaf": true/false`), since "does this
resolve to one tool call" is a semantic judgment the harness can't reliably check syntactically. The
harness still runs a lightweight deterministic cleanup pass over every expansion call's output —
mirroring today's `_normalize_steps` noise filters (`_ENV_SETUP_RE`, `_EDITOR_MECHANICS_RE`, dedup) —
just applied per-call instead of once over a flat list, rather than a separate leaf/branch classifier.

Headless mode's bias toward finer decomposition is applied via different prompting from the start: the
headless expansion prompt explicitly instructs the model to decompose further when uncertain rather than
commit to an ambiguous leaf. The model's emitted leaf/branch signal is always trusted directly and never
overridden after the fact — headless vs. interactive changes what's asked of the model, not who has final
say over the signal.
