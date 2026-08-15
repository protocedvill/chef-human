# Phase 5.1: CLI Polish & Interactive Mode

**Goal**: Transform the CLI from a single-shot task runner into a polished developer experience with interactive REPL mode, streaming output outside the debug TUI, colorized diffs, and robust non-interactive mode for CI.

**Prerequisites**: Phases 1–4 complete (all core engine, agent loop, code understanding, advanced features). Phase 4.4 (agent autonomy) — specifically the CLI integration fix (4.4.1) and agent creation consolidation (4.4.2) — must be merged first.

**PLAN.md requirements**:
- Interactive REPL-style interface
- Non-interactive mode for CI (`chef-human "fix this bug"`)
- Streaming output of model reasoning
- Colorized diff output

---

## Current State

| Component | Status |
|-----------|--------|
| `main.py` | Click CLI with `run`, `repl`, `show-config`, and `session` group (`list`/`show`/`delete`/`export`). `run` accepts `--debug-tui`, `--max-steps`, `--workspace`, `--no-stream`, `--headless`, `--json`, `--quiet`, `--model`, `--temperature`, `--config`, `--resume`, `--continue`, `--save-dir`. Inline task `chef-human "task"` via `main()` wrapper. Stdin pipe. `show-config` command displays effective config as Rich table. |
| `ReActUI` protocol | `on_start`, `on_planning_start`, `on_plan`, `on_reasoning_start`, `on_stream`, `on_reasoning`, `on_tool_call`, `on_tool_result`, `on_replan`, `on_error`, `on_approval_request` |
| `NoopUI` | Silent stub — all methods no-op |
| `DebugTUI` | Full-screen Rich Live layout with plan/reasoning/tool/log panels. Only works in full-screen mode |
| Streaming | Wired through `ReActLoop` — works inside `DebugTUI` and `StreamingUI`. Outside non-TUI mode, `StreamingUI` renders all protocol events with Rich markup. `--quiet` suppresses streaming, shows only result |
| Non-TUI output | Rich-formatted with color-coded success/failure, syntax-highlighted code blocks in message, token usage when available |
| Headless mode | JSON output to stdout — works but minimal |
| Session save dir | Default `.chef-human/sessions/` project-relative, configurable via `--save-dir` |
| `create_agent()` | Has diverged from `_execute_task()` (to be fixed by 4.4.2) |
| Tests | 1099+ tests total across the project; 119 new Phase 5.1 tests in `test_repl.py` (34), `test_streaming.py` (27), `test_repl_ui.py` (30), `test_main.py` (+28) |

### What's Missing vs PLAN.md

| PLAN.md Requirement | Current State | Gap |
|---------------------|---------------|-----|
| Interactive REPL | `chef-human repl` command with multi-turn conversation, `/commands`, auto-save, `--resume` | Gaps closed |
| Non-interactive CI mode | `--headless` for JSON, `--json` for rich+JSON, `--quiet` to suppress streaming, `chef-human "task"` inline, stdin pipe | All gaps closed |
| Streaming output (non-TUI) | `StreamingUI` renders all protocol events in real-time with Rich markup; `--quiet` suppresses | Gaps closed |
| Colorized diff output | Rich-formatted with syntax-highlighted code blocks | Gaps closed — colorized result output implemented |

---

## Task List

- [x] **5.1.1** Interactive REPL mode — `chef-human repl`
- [x] **5.1.2** Streaming output UI — `StreamingUI` for non-TUI mode
- [x] **5.1.3** Colorized result output — Rich-formatted final results
- [x] **5.1.4** Non-interactive CI mode polish — `--json`, `--quiet`, stdin
- [x] **5.1.5** Configuration overrides from CLI — `--model`, `--temperature`, etc.
- [x] **5.1.6** Session management improvements — default save dir, auto-save, `--continue`
- [x] **5.1.7** Tests & verification

---

## Task 5.1.1: Interactive REPL Mode

**File to create:** `chef_human/ui/repl.py`  
**File to modify:** `chef_human/main.py`

### Goal

A `chef-human repl` command that starts an interactive REPL session. The user types a message, the agent processes it (ReAct loop with tool execution), returns the result, and prompts for the next message. This enables iterative coding conversations: "Create a module" → "Add error handling" → "Write tests for it".

### Design

```
$ chef-human repl
╭──────────────────────────────────────╮
│  chef-human interactive mode         │
│  Type /help for commands             │
╰──────────────────────────────────────╯

You: add a fibonacci function to utils.py

[plan] 1. Read utils.py  2. Add function  3. Verify

[reasoning] The user wants a fibonacci implementation...

[tool] read path="utils.py" → File not found (new file)
[tool] write path="utils.py" → Done

Result: ✓ Added fibonacci function to utils.py (8 lines)

You: add memoization to it

[plan] 1. Read current implementation  2. Add lru_cache  3. Verify

...

You: /exit
Goodbye!
```

### Implementation

#### `chef_human/ui/repl.py` — `ReplUI` class

```python
class ReplUI:
    """Interactive REPL UI for ongoing agent conversations."""

    def __init__(self) -> None:
        self._console = Console()
        self._current_task: str = ""
        self._history: list[str] = []

    def on_start(self, task: str) -> None:
        self._current_task = task
        self._console.print(f"\n[bold cyan]You:[/] {task}\n")

    def on_planning_start(self) -> None:
        self._console.print("[dim]Planning...[/]")

    def on_plan(self, plan: Plan) -> None:
        self._console.print(f"[bold yellow]Plan:[/] {plan.goal}")
        for step in plan.steps:
            status_icon = self._status_icon(step.status)
            self._console.print(f"  {status_icon} {step.description}")

    def on_reasoning_start(self) -> None:
        self._console.print("[dim]Thinking...[/]")

    def on_stream(self, chunk: str) -> None:
        # Write chunk to stdout without newline for live feel
        sys.stdout.write(chunk)
        sys.stdout.flush()

    def on_reasoning(self, content: str) -> None:
        if content:
            self._console.print(f"\n[dim]{content}[/]")

    def on_tool_call(self, tool_call: ParsedToolCall) -> None:
        args_summary = ", ".join(
            f"{k}={v}" for k, v in tool_call.arguments.items()
        )[:100]
        self._console.print(f"  [bold green]▶[/] [cyan]{tool_call.name}[/]({args_summary})")

    def on_tool_result(self, name: str, result: str) -> None:
        icon = "✗" if result.startswith("Error:") else "✓"
        summary = result[:80].replace("\n", " ")
        self._console.print(f"  {icon} [bold]{name}[/] → {summary}")

    def on_replan(self) -> None:
        self._console.print("[bold yellow]↻ Re-planning...[/]")

    def on_error(self, message: str) -> None:
        self._console.print(f"[bold red]Error:[/] {message}")

    async def on_approval_request(self, tool_call: ParsedToolCall) -> bool:
        cmd = tool_call.arguments.get("command", "")
        self._console.print(f"\n[bold red]⚠ Destructive command:[/] {cmd}")
        result = Confirm.ask("Approve?", default=False)
        return result

    async def read_input(self) -> str | None:
        """Read user input, handling /commands. Returns None on /exit."""
        try:
            text = Prompt.ask("\n[bold cyan]You[/]", default="")
        except (EOFError, KeyboardInterrupt):
            return None
        if not text:
            return ""
        if text.startswith("/"):
            cmd = text[1:].lower().strip()
            if cmd in ("exit", "quit", "q"):
                return None
            elif cmd == "help":
                self._print_help()
                return ""
            elif cmd.startswith("save"):
                # Trigger session save
                ...
            else:
                self._console.print(f"[yellow]Unknown command: {text}[/]")
            return ""
        return text

    def display_result(self, result: AgentResult) -> None:
        status = "[bold green]✓ Success[/]" if result.success else "[bold red]✗ Failed[/]"
        self._console.print(f"\nResult: {status}")
        self._console.print(f"  Steps: {result.steps_taken}")
        self._console.print(f"  Message: {result.message}")
```

#### `main.py` — New `repl` command

```python
@cli.command()
@click.option("--max-steps", type=int, default=25, help="Max agent steps per turn")
@click.option("--workspace", type=click.Path(exists=True), help="Workspace directory")
@click.option("--resume", default=None, type=str, help="Session ID to resume")
@click.option("--save-dir", default=None, type=click.Path(), help="Save directory")
def repl(
    max_steps: int,
    workspace: str | None,
    resume: str | None,
    save_dir: str | None,
) -> None:
    """Start an interactive REPL session."""
    asyncio.run(_run_repl(
        max_steps=max_steps,
        workspace=workspace,
        resume=resume,
        save_dir=save_dir,
    ))
```

```python
async def _run_repl(
    max_steps: int,
    workspace: str | None,
    resume: str | None,
    save_dir: str | None,
) -> None:
    from chef_human.ui.repl import ReplUI

    logging.basicConfig(level=logging.WARNING)

    # Create context assembler (shared across turns)
    context = create_context_assembler(workspace_root=workspace)

    if resume:
        session_data = load_session_data(resume, save_dir=save_dir or ".")
        if session_data is not None:
            conv_data = session_data.get("conversation")
            if conv_data:
                loaded = ContextManager.from_dict(conv_data)
                context.conversation.messages = loaded.messages

    ui = ReplUI()
    banner = "[bold cyan]chef-human interactive mode[/]\nType /help for commands. Ctrl+C or /exit to quit."
    ui._console.print(banner)

    while True:
        task = await ui.read_input()
        if task is None:  # /exit or Ctrl+D
            ui._console.print("\n[bold cyan]Goodbye![/]")
            return
        if not task:
            continue

        # Create a fresh ReActLoop for this turn (shares context)
        loop = _build_loop(
            context=context,
            max_steps=max_steps,
            workspace=workspace,
            ui=ui,
        )

        result = await loop.run(task)

        ui.display_result(result)
```

Where `_build_loop()` extracts the common ReActLoop construction logic (this naturally follows from 4.4.2's consolidation):

```python
def _build_loop(
    context: ContextAssembler,
    max_steps: int,
    workspace: str | None,
    ui: ReActUI,
) -> ReActLoop:
    from chef_human.agent.planner import Planner
    from chef_human.agent.react_loop import ReActConfig, ReActLoop
    from chef_human.llm import create_backend
    from chef_human.tools import create_tool_registry

    backend = create_backend()
    tool_registry = create_tool_registry(
        workspace=context.workspace,
        symbol_index=context.symbol_index,
        file_context=context.file_context,
        dep_graph=context.dep_graph,
    )
    planner = Planner(llm_backend=backend)
    config = ReActConfig(
        max_steps=max_steps,
        tool_timeout=settings.tool_timeout,
        stream=True,
    )
    return ReActLoop(
        llm_backend=backend,
        tool_registry=tool_registry,
        context_assembler=context,
        planner=planner,
        config=config,
        ui=ui,
    )
```

**Key design decision**: The `ContextAssembler` is shared across all turns, so the conversation accumulates naturally. A fresh `ReActLoop` (and `Planner`, `tool_registry`) is created each turn to avoid stale state. The LLM backend connection is cheap to recreate.

### REPL `/commands`

| Command | Action |
|---------|--------|
| `/exit`, `/quit`, `/q` | Exit REPL |
| `/help` | Show available commands |
| `/save [name]` | Save current session |
| `/clear` | Clear conversation history (starts fresh) |
| `/undo` | Undo last change (delegates to `UndoTool`) |
| `/redo` | Redo last undone change |
| `/tokens` | Show token usage for this session |
| `/history` | Show recent messages |

Commands are handled in `ReplUI.read_input()` before any text is sent to the agent.

### Acceptance Criteria

- `chef-human repl` starts an interactive session with a banner and prompt
- User can type a message, agent processes it, result is displayed
- User can type another follow-up message (conversation context preserved)
- `/exit`, `/quit`, `/q` exit the REPL
- `/help` lists available commands
- Ctrl+C and Ctrl+D gracefully exit
- Agent's ReAct loop (planning, reasoning, tool calls, results) is displayed in real-time with Rich formatting
- Session is saved on exit
- `--resume` loads a prior conversation into the REPL
- The REPL works with `--workspace` to set the project root

---

## Task 5.1.2: Streaming Output UI

**File to create:** `chef_human/ui/streaming.py`  
**File to modify:** `chef_human/main.py`, `chef_human/ui/__init__.py`

### Goal

When running `chef-human run` without `--debug-tui`, show the agent's reasoning and tool calls in real-time, not just the final result. This gives the user visibility into what the agent is doing.

### Design

A `StreamingUI` class that implements `ReActUI` and renders output to the terminal with Rich markup:

```
$ chef-human run "add type hints to utils.py" --no-debug-tui

Planning...
  ☐ Read utils.py and understand current code
  ☐ Add type hints to all functions
  ☐ Verify with mypy

Thinking... ████████░░

The utils.py file has 3 functions without type hints...

▶ read(path="utils.py") → def greet(name): ...  (2KB)
▶ grep(pattern="def .* ->") → No existing type hints found
▶ edit(path="utils.py", old="def greet(name):", new="def greet(name: str) -> str:") → Applied

Thinking...

✓ write(path="utils.py") → Added type hints to all 3 functions

════════════════════════════════════════
Result: SUCCESS
Steps: 4
```

### Implementation

```python
# chef_human/ui/streaming.py

class StreamingUI:
    """Real-time streaming UI for non-TUI mode."""

    def __init__(self, quiet: bool = False) -> None:
        self._console = Console()
        self._quiet = quiet
        self._spinner = self._create_spinner()
        self._spinner_active = False

    def on_start(self, task: str) -> None:
        if not self._quiet:
            self._console.print(f"\n[bold]Task:[/] {task}\n")

    def on_planning_start(self) -> None:
        if not self._quiet:
            self._console.print("[dim]Planning...[/]")

    def on_plan(self, plan: Plan) -> None:
        if self._quiet:
            return
        self._console.print("[bold yellow]Plan:[/]")
        for step in plan.steps:
            self._console.print(f"  {self._status_icon(step.status)} {step.description}")
        self._console.print()

    def on_reasoning_start(self) -> None:
        if not self._quiet:
            self._start_spinner()

    def on_stream(self, chunk: str) -> None:
        if self._quiet:
            return
        self._stop_spinner()
        sys.stdout.write(chunk)
        sys.stdout.flush()

    def on_reasoning(self, content: str) -> None:
        if self._quiet or not content:
            return
        self._stop_spinner()
        self._console.print(f"\n[dim]{content}[/]\n")

    def on_tool_call(self, tool_call: ParsedToolCall) -> None:
        if self._quiet:
            return
        args = ", ".join(
            f"{k}={v}" for k, v in tool_call.arguments.items()
        )[:120]
        self._console.print(f"  [bold cyan]▶[/] [green]{tool_call.name}[/]({args})")

    def on_tool_result(self, name: str, result: str) -> None:
        if self._quiet:
            return
        icon = "✗" if result.startswith("Error:") else "✓"
        summary = result[:100].replace("\n", " ").strip()
        self._console.print(f"  {icon} [bold]{name}[/] → {summary}")

    def on_replan(self) -> None:
        if not self._quiet:
            self._console.print("[bold yellow]↻ Re-planning...[/]")

    def on_error(self, message: str) -> None:
        self._console.print(f"[bold red]Error:[/] {message}")

    async def on_approval_request(self, tool_call: ParsedToolCall) -> bool | None:
        return None  # Falls back to stdin prompt

    def _start_spinner(self) -> None:
        """Start a simple spinner animation."""
        ...

    def _stop_spinner(self) -> None:
        """Stop the spinner and clear the line."""
        ...
```

### Integration in `main.py`

When `--no-debug-tui` is set (and not `--headless`), use `StreamingUI` instead of `NoopUI`:

```python
# In _execute_task or the consolidated create_agent:
if headless:
    ui = NoopUI()
elif debug_tui:
    ui = DebugTUI()
else:
    ui = StreamingUI(quiet=False)
```

Or via a new `--verbose` / `--quiet` flag:

```python
@click.option("--quiet", is_flag=True, help="Suppress all output except final result")
```

### Acceptance Criteria

- `chef-human run "task" --no-debug-tui` shows planning, reasoning, tool calls, and results in real-time
- `chef-human run "task" --quiet` shows only the final result line
- `chef-human run "task" --headless` still works (NoopUI + JSON output)
- `chef-human run "task" --debug-tui` still uses full-screen DebugTUI
- Streaming works even without DebugTUI
- Reasoning text is displayed with Rich dim styling
- Tool calls are color-coded (green for success, red for errors)
- No flickering or broken terminal output

---

## Task 5.1.3: Colorized Result Output

**File to modify:** `chef_human/main.py` (output section)

### Goal

The final result output (non-TUI, non-REPL modes) should be Rich-formatted with:
- Syntax-highlighted diffs
- Color-coded success/failure
- Better visual structure

### Current output

```
========================================
Result: SUCCESS
Steps: 4
Message: Added type hints to utils.py
```

### Target output

```
╭──────────────────────────────────────╮
│ ✅ Result: SUCCESS                   │
│    Steps: 4                           │
╰──────────────────────────────────────╯

Message:
  Added type hints to utils.py

Token usage: 1,234 prompt / 567 completion
```

### Implementation

Replace the final output block in `main.py.run()`:

```python
# Before: plain text output
if headless:
    click.echo(json.dumps(result.to_dict(), indent=2))
else:
    click.echo(f"\n{'=' * 40}")
    click.echo(f"Result: {'SUCCESS' if result.success else 'FAILURE'}")
    click.echo(f"Steps: {result.steps_taken}")
    click.echo(f"Message: {result.message}")
```

```python
# After: Rich-formatted output
if headless:
    click.echo(json.dumps(result.to_dict(), indent=2))
else:
    from rich.console import Console
    console = Console()
    status = "[bold green]✓ SUCCESS[/]" if result.success else "[bold red]✗ FAILURE[/]"
    console.print(f"\n[bold]Result:[/] {status}")
    console.print(f"  [dim]Steps:[/] {result.steps_taken}")
    if result.message:
        console.print(f"\n[bold]Message:[/]")
        console.print(f"  {result.message}")
    if result.total_prompt_tokens or result.total_completion_tokens:
        console.print(f"\n[dim]Tokens:[/] {result.total_prompt_tokens:,} prompt / {result.total_completion_tokens:,} completion")
```

For diff highlighting in messages: use Rich's `Syntax` to highlight any code blocks or diffs found in `result.message`.

### Acceptance Criteria

- `chef-human run "task"` shows Rich-formatted output with color
- Success is green with ✓, failure is red with ✗
- Token usage shown when available
- Headless mode unchanged (still JSON)
- No Rich errors or unsupported-format crashes on any message content

---

## Task 5.1.4: Non-Interactive CI Mode Polish

**File to modify:** `chef_human/main.py`

### Goal

Support the PLAN.md vision of `chef-human "fix this bug"` — a shorthand for running a single task without subcommands. Also add `--json` and `--quiet` flags, and support piping task from stdin.

### 5.1.4.1: Inline task argument on `cli` group

Add a `task` argument to the `cli` group so `chef-human "fix this bug"` works directly:

```python
@click.group(invoke_without_command=True)
@click.argument("task", required=False, default=None)
@click.pass_context
def cli(ctx: click.Context, task: str | None) -> None:
    if task is not None:
        ctx.invoke(run, task=task)
```

This means `chef-human "add a test"` is equivalent to `chef-human run "add a test"`.

**Edge case**: If `task` is provided AND a subcommand is used (e.g., `chef-human "task" session list`), Click will error on the extra argument. This is acceptable — the user should use `chef-human run` for explicit subcommand usage.

**Alternative**: Use Click's `default_map` or a separate entry point. The simplest approach: make the `cli` group accept an optional task argument and forward to `run`.

### 5.1.4.2: `--json` flag

Add a `--json` flag to `run` (distinct from `--headless`):

| Flag | Effect |
|------|--------|
| `--headless` | NoopUI + JSON output (existing) |
| `--json` | StreamingUI + JSON output (new — shows streaming AND JSON result) |

```python
@click.option("--json/--no-json", default=False, help="Output final result as JSON")
```

When both `--headless` and `--json` are false → `StreamingUI` + plain text result.
When `--json` is true → `StreamingUI` + JSON result at end.
When `--headless` is true → `NoopUI` + JSON result (existing).

### 5.1.4.3: Stdin pipe

When no `task` argument is provided and stdin is not a TTY, read the task from stdin:

```python
if not task:
    if not sys.stdin.isatty():
        task = sys.stdin.read().strip()
```

This enables: `echo "fix the bug" | chef-human` or `cat prompt.txt | chef-human`.

### 5.1.4.4: `--quiet` flag

Suppresses all streaming output; only shows the final result line:

```python
@click.option("--quiet", is_flag=True, help="Suppress all output except the final result")
```

When `--quiet` is set, use `NoopUI` instead of `StreamingUI`, but still show a one-line summary at the end.

### Acceptance Criteria

- `chef-human "add a test"` works without the `run` subcommand
- `chef-human run "add a test"` still works (backward compatible)
- `echo "add a test" | chef-human` reads task from stdin
- `chef-human` with no args and TTY stdin prompts for task (existing behavior)
- `--json` flag outputs JSON result after streaming
- `--quiet` flag suppresses all streaming, shows only the result
- `--headless` mode unchanged
- All existing tests pass

---

## Task 5.1.5: Configuration Overrides from CLI

**File to modify:** `chef_human/main.py`

### Goal

Allow overriding `config.toml` settings from the CLI without editing the config file. Also add a `--show-config` flag to display the effective configuration.

### New options on `run` (and `repl`)

```python
@click.option("--model", default=None, help="LLM model (overrides config)")
@click.option("--temperature", type=float, default=None, help="Model temperature")
@click.option("--config", "config_path", type=click.Path(exists=True), help="Path to config.toml")
@click.option("--show-config", is_flag=True, help="Display effective configuration and exit")
```

### Implementation

Pass these through to `create_agent()` or the `ReActConfig`:

```python
def _resolve_settings(
    model: str | None,
    temperature: float | None,
    config_path: str | None,
) -> Settings:
    from chef_human.config import load_settings
    if config_path:
        return load_settings(config_path=config_path)

    # Override individual settings
    overrides = {}
    if model is not None:
        overrides["ollama_model"] = model
    if temperature is not None:
        overrides["temperature"] = temperature

    if overrides:
        base = Settings()
        return Settings(**{**base.__dict__, **overrides})
    return settings  # global default
```

### `--show-config`

```python
@cli.command()
@click.option("--config", "config_path", type=click.Path(exists=True), help="Path to config.toml")
def show_config(config_path: str | None) -> None:
    """Display the effective configuration."""
    from rich.console import Console
    from rich.table import Table
    cfg = _resolve_settings(model=None, temperature=None, config_path=config_path)
    table = Table(title="chef-human Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    for field_name in cfg.__dataclass_fields__:
        value = getattr(cfg, field_name)
        table.add_row(field_name, str(value))
    Console().print(table)
```

### Acceptance Criteria

- `chef-human run "task" --model llama3` overrides the model
- `chef-human run "task" --temperature 0.5` overrides temperature
- `chef-human run "task" --config /path/to/config.toml` loads a custom config
- `chef-human show-config` displays the effective config as a Rich table
- `chef-human show-config --config /path/to/config.toml` shows that config
- Without overrides, the existing config.toml is used (backward compatible)

---

## Task 5.1.6: Session Management Improvements

**Files to modify:** `chef_human/main.py`, `chef_human/agent/persistence.py`

### 5.1.6.1: Default save directory

Change the default `save_dir` from `"."` to `".chef-human/sessions/"`:

```python
# In persistence.py
DEFAULT_SAVE_DIR = ".chef-human/sessions"

def _resolve_save_dir(save_dir: str | None) -> Path:
    if save_dir is not None:
        return Path(save_dir)
    return Path(DEFAULT_SAVE_DIR)
```

### 5.1.6.2: Auto-save on completion

The existing `_save_conversation()` in `ReActLoop` already saves on completion when `save_sessions=True`. Ensure it uses the correct default path.

### 5.1.6.3: Session list improvements

Update `session list` to show timestamps and a longer task preview:

```
Session ID     Date                 Task
abc123f4       2026-07-02 10:30    Add type hints to utils.py
def45678       2026-07-02 11:15    Fix bug in parser module
```

```python
# In persistence.py list_sessions() — return more metadata
def list_sessions(save_dir: str | None = None) -> list[dict[str, str]]:
    """List saved sessions with metadata."""
    save_path = _resolve_save_dir(save_dir)
    sessions = []
    for f in sorted(save_path.glob("*.json"), key=os.path.getmtime, reverse=True):
        data = json.loads(f.read_text())
        sessions.append({
            "session_id": data.get("session_id", f.stem),
            "task": data.get("task", "")[:80],
            "created": data.get("created", ""),
            "file_path": str(f),
        })
    return sessions
```

### 5.1.6.4: `--continue` alias for `--resume`

```python
@click.option("--continue", "resume", default=None, type=str,
              help="Continue a previous session (alias for --resume)")
```

### Acceptance Criteria

- Sessions are saved to `.chef-human/sessions/` by default
- `session list` shows timestamps and longer task descriptions
- `session list` sorts by most recent first
- `--continue` works identically to `--resume`
- Sessions are saved on completion in all modes (run, repl, headless)
- Existing sessions in the current directory are still loadable (backward compat)

---

## Task 5.1.7: Tests & Verification

### New test files

| Test file | ~Tests | Coverage |
|-----------|--------|----------|
| `tests/test_agent/test_repl.py` | 34 | REPL banner, input loop, process task, /exit, /help, follow-up preserves context, --resume, Ctrl+C |
| `tests/test_ui/test_streaming.py` | 27 | StreamingUI all protocol methods, quiet mode, output formatting, colors, status icons |
| `tests/test_ui/test_repl_ui.py` | 30 | ReplUI protocol methods, /command parsing, input reading, display_result, status icons, events |

### Modified test files

| Test file | +Tests | Coverage |
|-----------|--------|----------|
| `tests/test_agent/test_main.py` | +15 | Inline task on group, `--json`, `--quiet`, stdin pipe, `--model`, `--temperature`, `--show-config`, `--continue`, new `repl` command help |

### Actual test counts (built so far)

| 5.1.x | Test file | Tests | Status |
|-------|-----------|-------|--------|
| 5.1.1 | `tests/test_agent/test_repl.py` | 34 | Complete |
| 5.1.2 | `tests/test_ui/test_streaming.py` | 27 | Complete |
| 5.1.4 | `tests/test_agent/test_main.py` | +7 | Complete |
| 5.1.5 | `tests/test_agent/test_main.py` | +14 | Complete |
| 5.1.6 | `tests/test_agent/test_main.py` | +7 | Complete |
| 5.1.7 | `tests/test_ui/test_repl_ui.py` | 30 | Complete |

### Integration tests

- Non-TUI streaming output correctness (capture stdout, verify ordering of plan → tool → result markers)
- REPL multi-turn: send message → verify output → send another → verify conversation context

### Actual total: 119 new tests (91 above plan estimate)

---

## Dependencies Map

```
5.1.1 repl.py ────────► ReActUI protocol, ReActLoop, ContextAssembler
5.1.1 main.py ────────► repl command, _build_loop()
5.1.2 streaming.py ───► ReActUI protocol, main.py UI selection
5.1.3 main.py ────────► Rich Console in output section
5.1.4 main.py ────────► cli group invoke_without_command, --json, stdin
5.1.5 main.py ────────► config.py Settings, show-config command
5.1.6 persistence.py ─► DEFAULT_SAVE_DIR, list_sessions metadata
5.1.7 tests ──────────► all of the above
```

---

## Implementation Order (completed)

1. **5.1.2** StreamingUI — foundation for non-TUI streaming (needed by repl) ✅
2. **5.1.3** Colorized result output — small, independent change to main.py output ✅
3. **5.1.1** Interactive REPL — major feature, depends on StreamingUI patterns ✅
4. **5.1.4** CI mode polish — inline task arg, stdin, `--json`, `--quiet` ✅
5. **5.1.5** Configuration overrides — `--model`, `--temperature`, `--show-config` ✅
6. **5.1.6** Session management improvements — default dir, better list, `--continue` ✅
7. **5.1.7** Tests — 30 ReplUI tests covering protocol, input, events, display, icons ✅

---

## Design Decisions

### 1. ReplUI as a new class, not an extension of DebugTUI

DebugTUI is full-screen (Live layout), which doesn't fit an interactive conversation where the user types between turns. ReplUI is a prompt-based interface that uses `Console.print()` for output and `Prompt.ask()` for input. The implementations share nothing except the ReActUI protocol interface.

### 2. Shared ContextAssembler, fresh ReActLoop per turn

The REPL shares the conversation history across turns by keeping the `ContextAssembler` alive. A fresh `ReActLoop` is created each turn to avoid issues with stale tool registries or planner state. The LLM backend creation is lightweight (Ollama HTTP client).

### 3. StreamingUI as default for non-TUI non-headless mode

The PLAN.md requires streaming output. Making `StreamingUI` the default for `--no-debug-tui` satisfies this without breaking `--headless` (NoopUI) or `--debug-tui` (DebugTUI). Users who want silent operation can use `--quiet` (which falls back to NoopUI).

### 4. `--json` and `--headless` are distinct

`--headless` = NoopUI + JSON (for scripts that don't want any stdout pollution).  
`--json` = StreamingUI + JSON result at end (for users who want to see progress AND get structured output).

### 5. Inline task on cli group (not top-level argument)

Using `invoke_without_command=True` on the cli group allows `chef-human "task"` to work without a subcommand while preserving `chef-human run`, `chef-human repl`, `chef-human session list` etc. This is the standard Click pattern for this use case.

### 6. `_build_loop()` extraction

The REPL needs to create multiple ReActLoops during its lifetime. Extracting loop construction into `_build_loop()` avoids code duplication and is a natural follow-on to the 4.4.2 consolidation. If 4.4.2 is already done, `_build_loop()` may simply call `create_agent()`.

---

## Changes & Deviations Tracking

### 5.1.1 Interactive REPL
| Deviation | Rationale |
|-----------|-----------|
| REPL uses `_build_loop()` not `create_agent()` directly | More explicit control over shared context and per-turn fresh state |

### 5.1.2 Streaming Output UI
| Deviation | Rationale |
|-----------|-----------|
| Uses Rich Console with markup, not raw print | Consistent styling, color support |
| `_reasoning_started` flag tracks whether we're in streaming reasoning state | Avoids printing "Thinking..." twice; cleared when `on_stream` or `on_reasoning` fires |
| No spinner implemented | The `_start_spinner`/`_stop_spinner` from design doc omitted — `on_reasoning_start` prints "Thinking..." once, then `on_stream`/`on_reasoning` overwrites it. Spinner added complexity with no clear benefit for non-TUI |
| Wired in `_execute_task()`, not `create_agent()` | UI assignment happens after `create_agent()` returns, before `loop.run()`. Modifying `create_agent()` would require plumbing a `quiet` flag through to a function that currently only accepts `debug_tui` |
| `--quiet` suppresses all per-step output, shows only final result | Falls back to `NoopUI` behavior for protocol events while still showing the result summary. User sees total outcome without the play-by-play |
| No spinner animation | Design doc included spinner; omitted as it adds complexity without clear value for non-TUI mode while streaming tokens |

### 5.1.3 Colorized Result Output
| Deviation | Rationale |
|-----------|-----------|
| No deviation from PLAN.md | Pure additive change |
| Message rendering uses custom regex-based code block detection instead of Rich `Markdown` renderer | Allows indented "Message:" header with code blocks inlined; `Markdown` would take over the full message rendering and lose the "  " prefix |
| `_print_message_with_code_highlighting()` helper created in `main.py` | Isolates the highlighting logic; could be extracted to `chef_human/ui/` if reused elsewhere |
| Regex pattern `r"```(\w+)?\n(.*?)\n```"` requires newline before closing fences | Matches common markdown formatting; inline `` `code` `` not handled (no plan requirement) |

### 5.1.4 CI Mode Polish
| Deviation | Rationale |
|-----------|-----------|
| `--json` added as separate from `--headless` | `--headless` suppresses all output which is too aggressive for interactive CI debugging |
| Inline task routing via `main()` wrapper rather than Click `invoke_without_command=True` | Click's `invoke_without_command=True` with a positional argument greedily consumes subcommand names as the task string, breaking `chef-human run --help`. The `main()` wrapper injects `run` into `sys.argv` before Click parses |
| `sys.stdin.isatty()` check used instead of Click's built-in stdin detection | Click doesn't expose a cross-platform TTY check in its public API; `sys.stdin.isatty()` is the standard Python approach |
| Stdin read happens after the interactive prompt check | `click.prompt()` would consume stdin intended for pipe if called first; checking `isatty()` first ensures correct behavior for both TTY and pipe modes |
| `main()` replaces `cli` as the entry point in `pyproject.toml` | Necessary to intercept argv before Click routing; `cli()` remains the Click group for testing |
| Existing test `test_interactive_mode_prompts_for_task` updated to mock `sys.stdin.isatty` | Without the mock, CliRunner's non-TTY stdin triggers the pipe path instead of the interactive prompt path |

### 5.1.5 Configuration Overrides
| Deviation | Rationale |
|-----------|-----------|
| `--show-config` not mentioned in PLAN.md | Users need visibility into effective config; a common CLI pattern |
| Overrides applied by temporarily replacing `chef_human.config.settings` before `create_agent()` | `create_agent()` and `create_backend()` read from module-level `settings`; no clean override path without changing their signatures. Settings restored via try/finally after agent creation |
| `_resolve_settings()` uses `dataclasses.replace()` for immutable `Settings` | `Settings` is a frozen dataclass; `replace()` creates a new instance with overridden fields |
| `--model` and `--temperature` only available on `run` and `repl` commands | These are the only commands that actually use the LLM; `session` commands don't need them |
| `--config` must point to an existing file (Click `Path(exists=True)`) | Standard Click validation prevents silent misconfiguration; `show-config --config <path>` can be used to preview a config before running with it |

### 5.1.6 Session Management
| Deviation | Rationale |
|-----------|-----------|
| Default save dir is `.chef-human/sessions/` | Consistent with existing `.chef-human/` convention |
| `--continue` added as alias | Friendlier name for the common use case |
| `list_sessions()` returns both `"path"` (backward compat) and `"file_path"` keys | Existing tests check `"path"`; added `"file_path"` for clarity |
| `session list` uses Rich Table with columns | Better visual presentation than raw text, includes timestamp column |
| `save_conversation()` now stores `"created"` timestamp | Enables chronological display in `session list` without reading file mtime |
| `DEFAULT_SAVE_DIR` changed from `~/.cache/chef-human/sessions` to `.chef-human/sessions/` | Project-relative saves are discoverable and git-ignorable; user can still pass `--save-dir` for custom location |

### 5.1.7 Tests
| Deviation | Rationale |
|-----------|-----------|
| Split into `test_repl.py`, `test_streaming.py`, `test_repl_ui.py` | Separate concerns for maintainability |
| 119 new tests created vs 48 estimated | Each UI class got thorough coverage (30-34 tests each) across protocol compliance, edge cases, and output formatting. `test_main.py` gained +28 new tests across 5.1.4, 5.1.5, 5.1.6 features |
| E2E/integration tests deferred | The plan mentioned integration tests for streaming output ordering and REPL multi-turn. These require a running Ollama server and are covered by the existing e2e test pattern. Manual verification of streaming output in terminal confirmed correct ordering |

---

## Future Work (Post-5.1)

- **5.2 TUI** — Full Textual-based terminal UI with split panes
- **5.3 IDE Extension** — VS Code extension using LSP
- **Tab completion** — Click's shell completion for bash/zsh/fish
- **Configuration wizard** — `chef-human init` to generate config.toml
- **Progress bars** — For long operations (indexing, embedding)
- **Notification on completion** — Desktop notification when long-running task finishes
- **Pipe output to file** — `--output result.md` to write the result to a file
