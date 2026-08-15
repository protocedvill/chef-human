# Chef Human: Finish-Line Brief

Status: proposed roadmap based on a repository survey on 2026-08-16.

## Recommended product position

Chef Human is an experimental, local-first coding agent for learning how agent loops, tool use,
code retrieval, and human approval fit together. It is a working prototype rather than a production
replacement for commercial coding assistants.

Suggested one-line description:

> A local-first coding-agent laboratory that makes planning, tool calls, diffs, and human approval
> visible while running open-weight models on your own machine.

This framing is honest about the project's maturity while still highlighting substantial engineering:
multiple inference backends, a ReAct loop, code-aware retrieval, reversible editing, session
persistence, and terminal interfaces.

## Survey snapshot

### What is already valuable

- A Python package with Ollama and llama.cpp backends and a shared backend protocol.
- A plan/act/observe loop with retries, replanning, tool timeouts, and approval hooks.
- Workspace-scoped file, search, shell, patch, diff, undo, refactor, and navigation tools.
- Tree-sitter symbol indexing for smaller repositories and embedding/FAISS retrieval for larger
  ones.
- Session persistence, streaming/REPL UI modules, and a substantial Textual TUI implementation.
- A large test corpus: the non-backup Python source and tests total roughly 20,000 lines.
- Thoughtful design notes and a prior code review that identify real trade-offs rather than hiding
  them.

### What currently blocks a public presentation

- The working tree is not a coherent version. Git reports 32 modified and 113 untracked paths.
- There are 58 `federate-safeBackup` files. In important cases the backup is a much newer version
  than the canonical filename (for example, the main CLI and agent loop).
- Only 138 paths are tracked while 206 files are visible through `rg`; significant implemented
  features therefore exist only as untracked work.
- The current CLI exposes only `run` and `session`, while the newer main-file backup describes
  `repl`, `tui`, `show-config`, and `recommend-model`. Other current modules and tests expect that
  newer API.
- Test collection is broken by cross-version imports (`FILE_MUTATING_TOOLS` and `ask_via_stdin` are
  representative examples). A partial run reached 899 passes, 47 failures, 89 errors, and 4 skips;
  that number is diagnostic, not a release baseline.
- The complete suite segfaults while importing NumPy under the local Python 3.15 environment.
  Python 3.12 or 3.13 should be the supported verification environment until dependencies certify
  newer versions.
- Ruff reports nine straightforward issues when backup files are excluded. Type checking also
  needs a clean source selection and optional-dependency policy before its output is meaningful.
- The public README is only a few lines, and `docs/USAGE.md` explicitly says the CLI and agent loop
  do not exist. That contradicts the implementation.
- There is no CI workflow, contributor guide, architecture document aimed at readers, screenshot or
  recording, release checklist, or clearly documented limitations/security model.

## Product principles for the remaining work

1. Recover before redesigning. Establish one source of truth and a green baseline before features.
2. Make the happy path obvious. A new reader should understand, install, verify, and run a safe demo
   without reading internal plans.
3. Show the engineering. The visible plan, tool timeline, approval requests, diff preview, and local
   model constraints are the portfolio story.
4. Prefer a reliable narrow prototype to a broad unreliable assistant. RAG, multi-file refactors,
   and alternate backends may be marked experimental.
5. Be explicit about safety. Workspace confinement and approvals are guardrails, not a hardened
   sandbox.
6. Every public claim must have a test, recorded demo, or clearly marked limitation behind it.

## Phases

| Phase | Outcome | Exit gate |
| --- | --- | --- |
| 0 | Recover a canonical repository | No conflict-copy source files; intended features preserved in Git |
| 1 | Establish a reproducible engineering baseline | Clean install, lint, and unit tests pass on Python 3.12/3.13 in CI |
| 2 | Make one end-to-end workflow dependable | Preflight, first-run guidance, run/REPL/TUI behavior, and failure messages are coherent |
| 3 | Package the portfolio story | README, architecture, demo media, examples, limitations, and release metadata agree with reality |
| 4 | Add only evidence-driven enhancements | Optional features are isolated, tested, and do not weaken the core demo |

Detailed specifications:

- [Phase 0 — Repository recovery](00-repository-recovery.md)
- [Phase 1 — Reproducible baseline](01-reproducible-baseline.md)
- [Phase 2 — Product and UX spine](02-product-ux-spine.md)
- [Phase 3/4 — Portfolio release and optional enhancements](03-portfolio-release.md)

## Owner decisions

The owner selected the following scope on 2026-08-16:

1. Recover the newest coherent backup generation as the canonical implementation.
2. Use the Textual TUI as the visual showcase and the streaming CLI as the reliability fallback.
3. Support Linux first.
4. Label llama.cpp, RAG, and semantic refactoring experimental while retaining explicit stretch plans.
5. Move verbose historical plans out of the repository root and treat them as archive/reference
   material rather than the current product roadmap.

## Definition of “recruiter-ready prototype”

A fresh reader can, in under five minutes, understand the project and see an authentic demo. On a
supported machine they can follow one install path, run `chef-human doctor`, start a small example
task, approve or reject a change, inspect its diff, and undo it. CI is green, the repository is clean,
all screenshots match the current UI, and limitations are easy to find. No claim suggests production
sandboxing, guaranteed autonomous correctness, or support for every model and platform.
