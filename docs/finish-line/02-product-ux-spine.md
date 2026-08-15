# Phase 2: Product and UX Spine

## Goal

Make one small-repository workflow understandable and dependable from installation through review and
undo. This is the feature-completion phase; broad new autonomy is out of scope.

## Canonical journey

1. The user installs Chef Human and Ollama.
2. `chef-human doctor` checks Python, configuration, backend reachability, selected model, workspace,
   and optional capability availability.
3. The user opens the example repository or their own small repository.
4. `chef-human run "add validation and tests"` shows a plan and progress.
5. Reads/searches proceed without noise; mutations and risky shell commands follow a documented
   approval policy.
6. The user sees concise tool results and diffs, then a final summary with files changed, validation
   performed, token usage, and any unresolved limitations.
7. The user can inspect and undo the last logical change.

The streaming CLI is the reliability baseline. The Textual TUI may be the visual showcase, but it must
use the same application service and policies rather than construct a separate agent stack.

## Work package 2A — One application factory

Files: `chef_human/main.py`, `chef_human/agent/__init__.py`, configuration/backend/tool construction.

Tasks:

1. Introduce one typed application/agent factory that receives resolved settings, workspace, UI, and
   mode options. `run`, REPL, TUI, tests, and programmatic usage all call it.
2. Remove module-level settings snapshots from behavior that must honor CLI or project overrides.
3. Pass symbol index, dependency graph, and file context consistently to the tool registry. Make tool
   availability visible instead of silently varying with repository size.
4. Return structured startup errors that every UI can render appropriately.

Acceptance criteria:

- A test changes model, temperature, config path, workspace, and max steps through each supported CLI
  mode and observes the same resolved settings.
- Tool capabilities reported at startup match the actual registry.
- Agent construction has one implementation path.

## Work package 2B — Doctor and first-run guidance

Add `chef-human doctor` with human-readable output and `--json`.

Checks:

- supported Python version and installed Chef Human version;
- parsed configuration and source of important values;
- writable/discovered workspace and Git presence;
- Ollama executable/server reachability and selected model availability;
- optional TUI, indexing, embeddings/RAG, and llama.cpp capability status;
- a safe, short next command.

Exit codes: 0 ready, 1 required check failed, 2 invalid configuration. Optional missing features are
warnings, not failures.

Acceptance criteria:

- All checks are unit-testable through injected probes; tests do not require the real machine state.
- Errors explain one concrete remedy and never dump a raw stack trace by default.
- `--json` has a versioned schema and writes no decorative output to stdout.

## Work package 2C — Coherent interaction modes

Tasks:

1. Define `chef-human run TASK` as the obvious non-interactive command, `chef-human repl` as ongoing
   conversation, and `chef-human tui` as the full-screen interface. Avoid a hidden/default subcommand
   trick unless help and error behavior remain conventional.
2. Make cancellation responsive during planning, inference, tool execution, and user prompts.
3. Use one UI protocol with async approval and question methods. No UI may read `stdin` synchronously
   while another framework owns the terminal.
4. Give all modes the same final result fields: status, summary, changed files, validations, steps,
   usage, session ID, and warnings.
5. Keep reasoning presentation modest. Prefer plan/progress/action summaries over exposing verbose raw
   model chain-of-thought.

Acceptance criteria:

- Headless Textual tests cover submit, approval, ask-user, cancel, error, and clean exit.
- CLI/REPL tests cover EOF, Ctrl+C, failed backend, invalid workspace, and resume.
- JSON mode is stable and contains no Rich/streaming contamination.

## Work package 2D — Trustworthy changes

Tasks:

1. Define a single `ChangeSet` transaction for one model-requested logical change, including multiple
   files. Diff preview, approval, undo, redo, and session summary consume it.
2. Serialize writes to overlapping paths. Independent read-only tools may remain concurrent.
3. Refresh or invalidate repository map, file context, symbol index, dependencies, and RAG state after
   successful changes. If a subsystem cannot refresh incrementally, mark it stale visibly.
4. Run configurable validation after a change and distinguish “agent says done” from “tests/lint
   verified.”
5. Add `--dry-run` only if it can reliably execute through diff creation without touching the
   workspace; otherwise omit it rather than simulate safety.

Acceptance criteria:

- A multi-file rename is previewed and undone atomically.
- The agent cannot edit a path using stale pre-change content without receiving a conflict.
- Final output names exactly which checks ran and their result.

## Work package 2E — Example and deterministic harness

Tasks:

1. Add a tiny fixture repository with a deliberately simple bug and tests.
2. Add a scripted/mock backend that replays protocol-level responses for UI tests and documentation
   capture. Label it a demo/test backend; never imply that replay demonstrates model intelligence.
3. Add one real Ollama smoke script for maintainers, outside default CI, covering plan → read → edit →
   test → finish.
4. Record expected resource requirements and typical runtime as ranges measured on named hardware.

Acceptance criteria:

- UI screenshots and recordings can be reproduced without nondeterministic model output.
- The real smoke test is short, opt-in, and produces a diagnostic log on failure.

