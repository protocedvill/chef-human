# Safety and threat model

Chef Human is a prototype. It reduces common accidents; it does not sandbox anything. Read this
before pointing it at a workspace you care about.

## What "safe" means here

Chef Human runs an LLM-driven agent loop that reads, writes, and executes shell commands inside a
workspace directory. Guardrails exist to stop *accidental* damage from a model doing something
unintended (editing outside the project, deleting the wrong file, running an obviously destructive
command). They are not a security boundary against a malicious or adversarial model, task, or
codebase.

**Run it only in a disposable checkout, VM, or container you're willing to lose.** Do not point it
at a workspace with secrets, production data, or anything irreplaceable.

## Guardrails that exist

- **Workspace confinement.** `WorkspaceManager` (`chef_human/agent/workspace.py`) resolves every
  path the agent's file tools touch and requires it to be inside the workspace root
  (`is_within_workspace`). Tools built on it refuse to read or write outside that root. It also
  applies `.gitignore` plus a built-in ignore list (`.git`, `node_modules`, `.venv`, `.chef-human`,
  etc.) so exploration/indexing tools don't wander into noise or the agent's own session state.
- **Shell command guard.** `BashTool` (`chef_human/tools/shell.py`) rejects an outright blacklist
  (`rm -rf /`, `mkfs`, fork bombs, `shutdown`/`reboot`, ...) and flags a `DESTRUCTIVE_PREFIXES` set
  (`rm`, `mv`, `dd`, redirects, ...) checked per-segment of a compound command, not just the whole
  string, so `echo hi; rm -rf .` is still caught. It also strips dynamic-linker injection
  environment variables (`LD_PRELOAD` and friends) from spawned processes.
- **Human approval on destructive commands.** `react_loop.py` has its own
  `_is_destructive_command` check (deliberately duplicating the same prefix logic) that gates a
  y/N terminal prompt — or, in the Textual TUI, a modal Approve/Reject dialog — before a flagged
  shell command actually runs.
- **Tool timeouts.** Every non-`finish` tool call is wrapped in `asyncio.wait_for(...,
  timeout=config.tool_timeout)` (default 60s), so a hung subprocess or runaway loop doesn't stall
  the agent indefinitely.
- **Reversible edits.** `write`, `edit`, `patch`, `refactor`, and `lint_fix` all record their diffs
  through a shared `DiffStore` (`chef_human/tools/diff.py`), so `undo`/`redo` (CLI, REPL, and TUI)
  can reverse the most recent change — including a multi-file `refactor_symbol` rename, recorded as
  one atomic transaction.
- **Lint-triggered rollback.** If `lint_after_write` is enabled, a write that introduces a lint
  error is rolled back automatically using content captured before the tool ran.
- **Serialized dispatch.** Tool calls within a single model turn are dispatched one at a time, in
  order, specifically so two mutating calls in the same turn can't race on the same file. This
  protection only holds for callers that go through `ReActLoop` — nothing in the tool classes
  themselves enforces per-file locking.

## What these guardrails do *not* do

- **No process sandboxing.** `BashTool` runs real shell commands with the permissions of the user
  running chef-human. A command can reach the network, read environment variables and ambient
  credentials, write anywhere the OS user can write, or spawn child processes — the workspace-root
  check applies to chef-human's own file tools, not to what a shell command you approve can do.
- **String-matching guards are evadable.** Both the `BashTool` blacklist and
  `_is_destructive_command` match literal command prefixes and substrings. An indirect invocation —
  `python3 -c "import shutil; shutil.rmtree('/')"`, `bash -c "rm -rf /"`, a script that does the
  damage a level removed from the literal string — bypasses both. Treat the guard as tripwire for
  obvious cases, not a policy engine.
- **No sandboxed filesystem.** Workspace confinement stops chef-human's *own* file tools from
  writing outside the root. It does not stop a shell command you approve from writing anywhere
  else, and it does not undo damage from a command that ran before you noticed it was wrong.
- **No secrets handling.** Chef Human does not scrub prompts, tool output, or logs for credentials.
  If your workspace or its `.env` files contain secrets, assume they can end up in a prompt sent to
  your local model, in `--log-file` output, or in a saved session JSON.
- **No cross-platform hardening.** Linux is the only validated platform. Guardrail behavior on
  macOS/Windows is untested.
- **No guarantee of correct or non-harmful *edits*.** An approved shell command not being
  destructive says nothing about whether the code changes the agent makes are correct. Review
  diffs (`view_diff`/TUI diff pane) before trusting a "success" result, and keep `undo` in mind.

## Recommended posture

1. Run inside a disposable git checkout, ideally inside a VM or container, especially for anything
   beyond reading/editing files.
2. Keep the terminal/TUI approval prompt on (the default) rather than pre-approving destructive
   commands; read what's being asked before approving.
3. Run `chef-human doctor` before a session to confirm the environment is what you expect (Python
   version, config, backend reachability, workspace).
4. Treat `--headless` runs (which cannot stop for approval — see below) as higher risk; use them
   only against workspaces/tasks you already trust, e.g. CI on your own disposable checkout.
5. Review diffs before relying on a change; use `undo` liberally.

## Headless mode and approval

`chef-human run --headless` promises a non-interactive, machine-readable result and therefore
disables `ask_user` and any interactive approval prompt (`disable_ask_user`). This means destructive
commands are not gated by a human approval step in headless mode — the blacklist/prefix guard is
the only thing standing between the agent and running them. Do not run `--headless` against a
workspace or task you wouldn't also let run fully unattended.

## Recovery

- `undo` (CLI command inside REPL/TUI, or the `undo` tool) reverts the most recent recorded file
  change; `redo` reapplies it.
- Because the workspace is expected to be a git checkout, `git status`/`git diff`/`git checkout --`
  remain your ground-truth recovery path if `undo` doesn't cover what changed (e.g. a shell command
  that touched files directly rather than going through chef-human's own tools).
- Saved sessions (`chef-human session ...`) let you inspect conversation history after the fact,
  which is often the fastest way to see exactly what the agent did and why.

## Reporting a concern

This is a personal/portfolio prototype, not a maintained security-sensitive project with a formal
disclosure process. If you find a guardrail bypass or a way the tool does something more dangerous
than documented here, please open an issue describing it.
