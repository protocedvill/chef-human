# Phase 3 and 4: Portfolio Release

## Phase 3 goal

Make the repository tell a coherent engineering story before a recruiter runs anything.

## Work package 3A — README and positioning

The README should contain, in this order:

1. Name, one-line position, and a current screenshot or short terminal recording.
2. A three-sentence explanation of the problem, approach, and prototype status.
3. Four to six concrete capabilities, separating supported from experimental.
4. A 60-second tour using the deterministic example, followed by the real Ollama quick start.
5. A compact architecture diagram.
6. “What I learned / engineering highlights” covering local-model constraints, structured tool calls,
   context budgets, transactional editing, and async terminal UX.
7. Supported environment, limitations, safety note, test commands, and links to deeper docs.

Avoid claims such as “fully sandboxed,” “production-ready,” “autonomous,” or “works on any codebase.”
Avoid exact test counts unless generated automatically.

## Work package 3B — Reader-focused documentation

Create or rewrite:

- `docs/INSTALL.md`: one supported path first, optional backends afterward.
- `docs/USAGE.md`: real current commands and three task examples.
- `docs/ARCHITECTURE.md`: component boundaries, a request sequence, and extension points.
- `docs/SAFETY.md`: threat model, workspace checks, approvals, subprocess limitations, and recovery.
- `docs/DEVELOPMENT.md`: environment, test markers, lint/type checks, and optional dependencies.
- `CONTRIBUTING.md`: small contribution path and expected checks.

Acceptance criteria:

- Every documented command is executed in a docs smoke check or CI test.
- No document describes implemented modules as “future.”
- Public docs do not rely on `CLAUDE.md` or historical internal plans for essential knowledge.

## Work package 3C — Visual and release polish

Tasks:

1. Capture a short, legible recording of the canonical journey and one still image for README social
   previews. Use the deterministic harness for repeatability and disclose that in alt text/caption.
2. Add consistent terminal copy: sentence case, actionable errors, stable status icons, and no raw
   tracebacks unless `--debug` is enabled.
3. Add project URLs, classifiers, license metadata, and a version source to package metadata.
4. Add `CHANGELOG.md` with a clearly labeled `0.1.0` prototype release and a release checklist.
5. Create a GitHub release only after installing the built artifact and replaying the demo from a
   clean directory.

Acceptance criteria:

- README media renders on GitHub and matches the release UI.
- `pip install` of the built artifact, not just editable install, passes the quick start.
- The default branch is clean and CI-green at the tagged commit.

## Work package 3D — Recruiter-facing engineering narrative

Add a concise design retrospective, preferably inside `docs/ARCHITECTURE.md`:

- why tagged tool calls were chosen for small local models;
- why context selection differs for small and large repositories;
- how retries and replanning are bounded;
- how UI callbacks keep the core independent of presentation;
- what failed during development and what changed (for example blocking I/O in an async TUI,
  transaction semantics for undo, and sync-conflict recovery);
- what would be required for production: real sandboxing, stronger evals, concurrency isolation,
  secrets handling, telemetry/privacy decisions, and cross-platform validation.

This is often more persuasive than another feature because it demonstrates judgment.

## Phase 4: Optional enhancements after release

Choose enhancements only after the canonical journey has usage evidence.

### Candidate 4A — Evaluation suite

Create five to ten tiny, deterministic coding tasks scored on file outcome, tests, number of steps,
invalid tool calls, and completion status. Run them manually against a small set of local models and
publish methodology plus caveats. This is the strongest “learning tool” extension.

### Candidate 4B — Capability-aware startup

Use hardware detection and model recommendations to explain trade-offs, then let the user select a
model explicitly. Never auto-download a multi-gigabyte model without confirmation.

### Candidate 4C — Incremental code intelligence

Wire the file watcher/change events to symbol and dependency index refresh. Keep RAG experimental until
incremental updates, optional dependency behavior, and stale-index visibility are reliable.

### Candidate 4D — Policy profiles

Offer documented `observe`, `approve-writes`, and `autonomous-workspace` policies. Profiles configure a
single enforcement layer; they do not duplicate checks in UI and agent-loop code.

## Task format for weaker implementation agents

Every delegated task should be copied into an issue using this template:

```text
Objective:
One observable outcome, expressed from the user's perspective.

Allowed files:
An explicit, narrow list. Ask before expanding it.

Read first:
The exact source, tests, and design section that define current behavior.

Requirements:
Numbered behavior statements, including error and optional-dependency behavior.

Non-goals:
Adjacent refactors/features the agent must not attempt.

Verification:
Exact targeted commands followed by the required repository-level checks.

Acceptance criteria:
Binary observable checks. “Code looks clean” is not an acceptance criterion.

Handoff:
Changed files, tests run/results, assumptions, and unresolved risks.
```

Delegation rules:

- Give an agent one compatibility cluster or one user-visible behavior, not an entire phase.
- Require tests to fail before a behavior fix when practical.
- Do not let parallel agents edit shared orchestration files (`main.py`, agent construction, ReAct
  loop, UI protocol, or package metadata) at the same time.
- Assign documentation only after the command/API it documents is merged.
- Have a stronger review agent integrate each package and run the phase exit gate.

