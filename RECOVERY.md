# Repository Recovery — 2026-08-16

## Purpose

The working directory contained multiple generations of Chef Human after a file-sync conflict. Newer
modules and tests were saved under `federate-safeBackup` names while older counterparts retained the
canonical import paths. This report records how the source of truth was recovered.

## Preservation

- Original branch: `main` at `dfa72a1`, plus a large modified/untracked working tree.
- Complete raw snapshot: commit `db2dff7` on branch `recovery/raw-20260816`.
- Recovery branch: `finish-line/recovery`.
- The raw snapshot contains every authored file, conflict copy, generated artifact, and planning file
  visible before cleanup.

## Inventory

- 58 paths contained the `federate-safeBackup` suffix.
- 32 paths were modified and 113 were untracked before preservation.
- Every backup had a canonical counterpart, and every pair differed.
- Important backup files formed a mutually compatible later generation: the CLI, ReAct loop, planner,
  UI protocol, Textual UI, tool transactions, and their tests referred to the same expanded APIs.
- Two paths had two backup generations. `filesystem-federate-safeBackup-0002.py` and
  `diff-federate-safeBackup-0002.py` were selected because they were the later revisions and supplied
  behavior required by the recovered undo/redo and tool tests.

## Recovery decision

The latest backup in each compatibility cluster became the canonical file:

| Cluster | Representative recovered behavior |
| --- | --- |
| CLI, config, packaging | `run`, `repl`, `tui`, `show-config`, `recommend-model`, session commands |
| Agent orchestration | richer planning/verification, safety gates, logging, usage reporting |
| UI protocol and interfaces | async questions/approvals, streaming, REPL, Textual TUI |
| File and refactoring tools | diff history, undo/redo, patching, navigation, lint support |
| Code intelligence | symbol/dependency indexing, watcher, optional RAG path |
| Tests | the test generation matching the recovered public and internal APIs |

This selection was not based on timestamps alone. After canonicalization, the recovered non-RAG,
non-live-integration suite passed 1,279 tests with 7 deselected. Before canonicalization, the comparable
tree had dozens of failures and import errors. The passing suite is the compatibility evidence for the
choice.

## Cleanup

- Removed all conflict-suffixed product and test files from the recovery branch. They remain available
  in the raw snapshot.
- Removed tracked bytecode, egg-info metadata, local editor settings, and captured command output.
- Extended `.gitignore` for those local/generated artifacts and crash dumps.
- Moved verbose historical roadmaps and prompt notes to `docs/archive/`.
- Kept the current finish-line documents under `docs/finish-line/`.

## Validation and known limitations

- `python -m chef_human.main --help`: passes and exposes the recovered command set.
- `pytest --ignore=tests/test_rag -q -m 'not integration'`: 1,279 passed, 7 deselected on the available
  Python 3.15 environment.
- Full collection on that interpreter crashes inside the installed NumPy binary while importing RAG
  tests. This environment is Python 3.15.0rc1 and is outside the planned Linux/Python 3.12–3.13 release
  baseline.
- A separate Python 3.14 installation is present but lacks project dependencies, so it is not a valid
  clean-environment result.
- Ruff issues found immediately after recovery were mechanical except for a missing runtime `Settings`
  import in `main.py`; both are corrected on the recovery branch.
- Pyright is not yet a release gate. Its current environment cannot resolve several installed and
  optional packages, and it also reports genuine typing work in the agent loop. That belongs to Phase
  1 rather than source-generation recovery.

## Product decisions recorded during recovery

- The Textual TUI is the visual showcase; the streaming CLI remains the reliability fallback.
- The initial supported platform is Linux with Python 3.12 and 3.13.
- Ollama is the supported backend path.
- llama.cpp, RAG, and semantic refactoring are experimental and retain stretch plans in the finish-line
  roadmap.
- Historical plans are reference material, not the active public roadmap.

