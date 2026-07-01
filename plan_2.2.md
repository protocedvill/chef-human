# Phase 2.2: Agent Loop Polish & Production Readiness

**Goal**: Take the working ReAct loop from Phase 2.1 and make it production-quality. Add streaming model output to the TUI, agent working memory (scratchpad), headless mode for CI/scripting, conversation persistence, and polish the debug TUI.

**Prerequisites**: Phase 2.1 complete (ReAct loop, parser, planner, retry, approval gate, TUI, CLI, prompts).

---

## Task List

- [ ] **2.2.1** Streaming model output to TUI
- [ ] **2.2.2** Agent scratchpad (working memory)
- [ ] **2.2.3** Headless mode & structured output
- [ ] **2.2.4** Conversation persistence (save/load)
- [ ] **2.2.5** TUI polish & reliability

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│  2.2.5 TUI Polish (Rich)                                 │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐           │
│  │ Plan Panel  │ │ Reasoning  │ │  Log        │           │
│  │ (color-coded)│ │ (streaming)│ │ (searchable)│          │
│  └────────────┘ └────────────┘ └────────────┘           │
├──────────────────────────────────────────────────────────┤
│  2.2.3 Headless Mode ───→ JSON result to stdout          │
│  2.2.4 Conversation ───→ JSON save/load to disk          │
├──────────────────────────────────────────────────────────┤
│  ReActLoop (2.1)                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ 2.2.2 Scratchpad: {scratchpad} injected in prompt  │  │
│  │ 2.2.1 Streaming: on_stream(chunk) callback          │  │
│  └────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────┤
│  LLM Backend                                              │
│  ┌────────────────────────────────────────────────────┐  │
│  │ 2.2.1 Streaming: async generator from complete()    │  │
│  │ Ollama: ollama.AsyncClient().generate() stream=True │  │
│  │ LlamaCpp: __call__() stream=True                    │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## Task 2.2.1: Streaming Model Output to TUI

**Files:**
- `chef_human/llm/backend.py` — add `complete_stream()` to ABC
- `chef_human/llm/ollama_backend.py` — implement streaming
- `chef_human/llm/llamacpp_backend.py` — implement streaming
- `chef_human/llm/__init__.py` — factory may need update if signature changes
- `chef_human/ui/protocol.py` — add `on_stream(chunk)` to `ReActUI` protocol
- `chef_human/ui/debug_tui.py` — render streaming text in Reasoning panel
- `chef_human/agent/react_loop.py` — use streaming when `config.stream=True`
- `tests/test_agent/test_react_loop.py` — update streaming tests

### Why now
The current TUI shows nothing during model reasoning. The user stares at a blank panel for 10-30 seconds while the model generates a response. Streaming shows tokens as they arrive, making the tool feel responsive and letting the user see the model's chain of thought develop.

### Design

#### LLMBackend changes

Add an async generator method to `LLMBackend`:

```python
class LLMBackend(ABC):
    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        ...

    async def complete_stream(
        self, request: CompletionRequest
    ) -> AsyncGenerator[tuple[str, CompletionResponse | None], None]:
        """Yields (token_chunk, optional_final_response) tuples.
        
        Each yield is a (token, None) for intermediate tokens.
        The final yield is ("", CompletionResponse) with the full message.
        Falls back to non-streaming by default.
        """
        resp = await self.complete(request)
        yield resp.message.content, resp
```

Default implementation: fall back to non-streaming (yield entire content as single chunk). Backends override for actual streaming.

#### OllamaBackend streaming

```python
class OllamaBackend(LLMBackend):
    async def complete_stream(
        self, request: CompletionRequest
    ) -> AsyncGenerator[tuple[str, CompletionResponse | None], None]:
        client = ollama.AsyncClient(host=self._host)
        messages = [format_chatml(msg) for msg in request.messages]
        options = {
            "temperature": request.temperature,
            "num_predict": request.max_tokens,
        }
        if request.stop:
            options["stop"] = request.stop

        stream = await client.chat(
            model=self._model,
            messages=messages,
            tools=[tool_to_dict(t) for t in (request.tools or [])],
            options=options,
            stream=True,
        )

        full_content = ""
        async for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                token = chunk["message"]["content"]
                full_content += token
                yield token, None

        msg = Message(role=Role.assistant, content=full_content)
        yield "", CompletionResponse(message=msg)
```

#### LlamaCppBackend streaming

```python
class LlamaCppBackend(LLMBackend):
    async def complete_stream(
        self, request: CompletionRequest
    ) -> AsyncGenerator[tuple[str, CompletionResponse | None], None]:
        prompt = format_chatml(request.messages)
        full_content = ""

        # run_in_executor because llama-cpp-python is synchronous
        def _stream():
            nonlocal full_content
            for token in self._model(
                prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop=request.stop or [],
                stream=True,
            ):
                text = token.get("choices", [{}])[0].get("text", "")
                full_content += text
                return text

        loop = asyncio.get_event_loop()
        while True:
            text = await loop.run_in_executor(None, _stream)
            if not text:
                break
            yield text, None

        msg = Message(role=Role.assistant, content=full_content)
        yield "", CompletionResponse(message=msg)
```

#### ReActUI protocol changes

```python
class ReActUI(Protocol):
    def on_stream(self, chunk: str) -> None: ...
    # existing methods unchanged
```

#### DebugTUI changes

Add `on_stream` to `DebugTUI`:
- Accumulate chunks in a buffer
- Update the reasoning panel text with accumulated content
- Use `self._live.update()` for Rich live refresh

```python
class DebugTUI:
    def __init__(self, task: str):
        self._task = task
        self._reasoning_text = ""
        self._reasoning_panel = Panel("", title="Reasoning", border_style="blue")
        # ... rest of constructor

    def on_stream(self, chunk: str) -> None:
        self._reasoning_text += chunk
        self._reasoning_panel = Panel(
            self._reasoning_text[-2000:],  # truncate for performance
            title="Reasoning",
            border_style="blue",
        )
        self._refresh()
```

#### ReActLoop changes

In `run()`, when `self._config.stream` is True:

```python
if self._config.stream:
    full_content = ""
    async for token, final_response in self._llm.complete_stream(
        CompletionRequest(...)
    ):
        if final_response:
            response = final_response
        else:
            full_content += token
            self._ui.on_stream(token)
    response.message.content = full_content
else:
    response = await self._llm.complete(CompletionRequest(...))

self._ui.on_reasoning(response.message.content)  # still called after completion
```

### Acceptance Criteria

- [ ] `complete_stream()` default implementation falls back to non-streaming
- [ ] `complete_stream()` yields `(token, None)` per chunk
- [ ] `complete_stream()` final yield is `("", CompletionResponse)`
- [ ] OllamaBackend overrides `complete_stream()` with real async streaming
- [ ] LlamaCppBackend overrides `complete_stream()` using `run_in_executor`
- [ ] `on_stream(chunk)` callback added to `ReActUI` protocol
- [ ] `NoopUI` implements `on_stream(chunk)` as no-op
- [ ] `DebugTUI.on_stream(chunk)` updates reasoning panel in real-time
- [ ] `ReActLoop.run()` uses streaming when `config.stream=True`
- [ ] All streaming tests use `AsyncMock` to mock the async generator
- [ ] 20+ new tests across backend streaming, UI streaming, and loop streaming

---

## Task 2.2.2: Agent Scratchpad (Working Memory)

**Files:**
- `chef_human/agent/prompts.py` — add `{scratchpad}` placeholder to `AGENT_SYSTEM_PROMPT`
- `chef_human/agent/parser.py` — add scratchpad extraction from model output
- `chef_human/agent/react_loop.py` — manage scratchpad state, inject into prompt
- `chef_human/agent/retry.py` — reset scratchpad on re-plan
- `tests/test_agent/test_prompts.py` — update tests
- `tests/test_agent/test_react_loop.py` — update tests

### Why now
The model currently has no way to persist notes across turns. Each turn starts fresh. A scratchpad lets the model maintain a todo list, track assumptions, or store intermediate state. This is a cheap but effective improvement to model coherence.

### Design

Add to `AGENT_SYSTEM_PROMPT`:

```python
## Notes / Scratchpad
{scratchpad}

Use the scratchpad to keep notes across turns — track assumptions, 
list sub-tasks, or remember file paths. Update it when your 
understanding changes. To update, start a line with "## Scratchpad:" 
followed by the new content. Only one scratchpad exists; each update 
replaces the previous content.
```

#### Parser changes

Add `extract_scratchpad(content: str) -> str | None`:

```python
_SCRATCH_PATTERN = re.compile(
    r"^## Scratchpad:\s*(.+?)$",
    re.MULTILINE | re.DOTALL,
)

def extract_scratchpad(content: str) -> str | None:
    """Extract scratchpad update from model output.
    
    Returns the new scratchpad content, or None if no update.
    Only the last ## Scratchpad: block is used.
    """
    matches = list(_SCRATCH_PATTERN.finditer(content))
    if not matches:
        return None
    return matches[-1].group(1).strip()
```

#### ReActLoop changes

In `run()`:
- Initialize `scratchpad = ""` 
- After each model response:
  - Call `extract_scratchpad(content)` to check for updates
  - If update found, replace `scratchpad`
- Pass `scratchpad=scratchpad` to `build_agent_prompt()`

```python
class ReActLoop:
    async def run(self, task: str) -> AgentResult:
        ...
        scratchpad = ""
        
        while steps_taken < self._config.max_steps:
            system_prompt = build_agent_prompt(
                plan=plan,
                tool_defs=self._tools.get_definitions(),
                scratchpad=scratchpad,
            )
            ...
            # After getting response:
            new_scratchpad = extract_scratchpad(response.message.content)
            if new_scratchpad is not None:
                scratchpad = new_scratchpad
                logger.debug("Scratchpad updated: %s", scratchpad[:100])
```

#### Prompts changes

```python
def build_agent_prompt(
    plan: Plan,
    tool_defs: list[ToolDefinition],
    repo_map: str = "",
    scratchpad: str = "",
) -> str:
    ...
    return AGENT_SYSTEM_PROMPT.format(
        repo_map=repo_map or "(no project context loaded)",
        plan_text=plan_text,
        tool_definitions=tool_text,
        scratchpad=scratchpad or "(empty — start tracking notes here if needed)",
    )
```

### Acceptance Criteria

- [ ] `AGENT_SYSTEM_PROMPT` has `{scratchpad}` placeholder with instructions
- [ ] `extract_scratchpad(content)` returns `None` when no scratchpad block
- [ ] `extract_scratchpad(content)` returns content after `## Scratchpad:` header
- [ ] Only the last scratchpad block is used (multiple updates in one turn)
- [ ] `build_agent_prompt()` accepts and formats `scratchpad` parameter
- [ ] `ReActLoop` initializes scratchpad as empty string
- [ ] `ReActLoop` updates scratchpad from model output each turn
- [ ] Scratchpad resets on re-plan (via `update_plan()`)
- [ ] 10+ tests covering parser extraction, prompt format, and loop behavior

---

## Task 2.2.3: Headless Mode & Structured Output

**Files:**
- `chef_human/main.py` — add `--headless` flag, output JSON
- `chef_human/agent/react_loop.py` — add `AgentResult.to_dict()` for serialization
- `chef_human/agent/planner.py` — add `Plan.to_dict()` and `PlanStep.to_dict()`
- `chef_human/agent/parser.py` — add `ParsedToolCall.to_dict()`
- `tests/test_agent/test_main.py` — update for headless mode

### Why now
The tool currently always launches the TUI. For CI/CD pipelines, automated testing, or batch processing, a headless mode that outputs structured JSON results is essential.

### Design

#### CLI changes

```python
@cli.command()
@click.argument("task", required=False)
@click.option("--debug-tui/--no-debug-tui", default=True)
@click.option("--max-steps", default=25, type=int)
@click.option("--workspace", default=None, type=click.Path(exists=True, file_okay=False))
@click.option("--headless", is_flag=True, default=False, help="Run without TUI, output JSON result")
def run(task, debug_tui, max_steps, workspace, headless):
    """Execute a task with the chef-human agent."""
    if headless:
        debug_tui = False
    asyncio.run(_execute_task(task, debug_tui=debug_tui, max_steps=max_steps, workspace=workspace, headless=headless))

async def _execute_task(task, debug_tui, max_steps, workspace, headless):
    ...
    result = await loop.run(task)
    
    if headless:
        import json
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"Result: {'SUCCESS' if result.success else 'FAILURE'}")
        print(f"Steps: {result.steps_taken}")
        print(f"Message: {result.message}")
```

#### Serialization helpers

```python
@dataclass
class AgentResult:
    plan: Plan
    steps_taken: int
    message: str
    success: bool = True

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "steps_taken": self.steps_taken,
            "message": self.message,
            "plan": self.plan.to_dict(),
        }

@dataclass
class Plan:
    goal: str
    steps: list[PlanStep]

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
        }

@dataclass
class PlanStep:
    index: int
    description: str
    status: StepStatus = StepStatus.pending

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "description": self.description,
            "status": self.status.value,
        }
```

### Acceptance Criteria

- [ ] `--headless` flag added to `run` command
- [ ] `--headless` implies `--no-debug-tui`
- [ ] In headless mode, `NoopUI` is used instead of `DebugTUI`/`NoopUI`
- [ ] `AgentResult.to_dict()` returns JSON-serializable dict
- [ ] `Plan.to_dict()` and `PlanStep.to_dict()` implemented
- [ ] JSON output printed to stdout in headless mode
- [ ] Non-zero exit code on failure in headless mode
- [ ] 8+ tests: flag parsing, JSON output shape, exit codes, NoopUI used in headless

---

## Task 2.2.4: Conversation Persistence (Save/Load)

**Files:**
- `chef_human/agent/persistence.py` — new module, conversation save/load logic
- `chef_human/agent/context.py` — add `ContextManager.to_dict()` and `from_dict()`
- `chef_human/agent/react_loop.py` — add save on interrupt, optional resume
- `chef_human/main.py` — add `--resume` flag, configure save path
- `tests/test_agent/test_persistence.py` — new test file

### Why now
Long agent sessions can be interrupted (network issues, OOM, user Ctrl+C). Saving conversation state allows resumption without losing context. Also enables post-mortem analysis of agent behavior.

### Design

#### ContextManager serialization

```python
class ContextManager:
    def to_dict(self) -> dict:
        return {
            "max_tokens": self._max_tokens,
            "messages": [
                {
                    "role": m.role.value,
                    "content": m.content,
                    "tool_calls": m.tool_calls,
                    "tool_call_id": m.tool_call_id,
                }
                for m in self._messages
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> ContextManager:
        msgs = [
            Message(
                role=Role(msg["role"]),
                content=msg["content"],
                tool_calls=msg.get("tool_calls"),
                tool_call_id=msg.get("tool_call_id"),
            )
            for msg in data["messages"]
        ]
        cm = cls(max_tokens=data.get("max_tokens", 32000))
        cm._messages = msgs
        return cm
```

#### Persistence module

```python
# chef_human/agent/persistence.py

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SAVE_DIR = Path.home() / ".cache" / "chef-human" / "sessions"


def save_conversation(
    conversation: dict,
    task: str,
    save_dir: str | Path = DEFAULT_SAVE_DIR,
    session_id: str | None = None,
) -> Path:
    """Save conversation state to a JSON file."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if session_id is None:
        import hashlib
        import time
        session_id = hashlib.sha256(
            f"{task}-{time.time()}".encode()
        ).hexdigest()[:12]
    
    path = save_dir / f"session_{session_id}.json"
    data = {
        "session_id": session_id,
        "task": task,
        "conversation": conversation,
    }
    path.write_text(json.dumps(data, indent=2))
    logger.info("Conversation saved to %s", path)
    return path


def load_conversation(session_id: str, save_dir: str | Path = DEFAULT_SAVE_DIR) -> dict | None:
    """Load conversation state from a JSON file."""
    path = Path(save_dir) / f"session_{session_id}.json"
    if not path.exists():
        logger.warning("Session file not found: %s", path)
        return None
    data = json.loads(path.read_text())
    return data.get("conversation")


def list_sessions(save_dir: str | Path = DEFAULT_SAVE_DIR) -> list[dict]:
    """List saved sessions with metadata."""
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return []
    sessions = []
    for f in sorted(save_dir.glob("session_*.json"), reverse=True):
        data = json.loads(f.read_text())
        sessions.append({
            "session_id": data.get("session_id"),
            "task": data.get("task"),
            "path": str(f),
        })
    return sessions
```

#### ReActLoop changes

```python
class ReActLoop:
    def __init__(self, ..., save_sessions: bool = True, save_dir: str | None = None):
        ...
        self._save_sessions = save_sessions
        self._save_dir = save_dir

    async def run(self, task: str) -> AgentResult:
        ...
        try:
            # ... main loop ...
        finally:
            if self._save_sessions:
                self._save_conversation(task)
    
    def _save_conversation(self, task: str) -> None:
        from chef_human.agent.persistence import save_conversation
        conv = self._context.conversation.to_dict()
        save_conversation(conv, task=task, save_dir=self._save_dir or DEFAULT_SAVE_DIR)
```

#### CLI changes

```python
@click.option("--resume", default=None, type=str,
              help="Session ID to resume")
@click.option("--save-dir", default=None, type=click.Path(),
              help="Directory for saving sessions")
```

If `--resume` is provided, load conversation and inject it into the `ContextManager` before starting the loop.

### Acceptance Criteria

- [ ] `ContextManager.to_dict()` produces JSON-serializable dict
- [ ] `ContextManager.from_dict()` reconstructs from dict
- [ ] `save_conversation()` writes JSON file to configurable directory
- [ ] `load_conversation()` reads JSON file by session ID
- [ ] `list_sessions()` returns sorted list of session metadata
- [ ] `ReActLoop` saves conversation on normal completion (in `finally` block)
- [ ] `ReActLoop` saves conversation on Ctrl+C (SIGINT handler)
- [ ] `--save-dir` CLI option configures save location
- [ ] `--resume` CLI flag loads conversation from save file
- [ ] 15+ tests for serialization, persistence I/O, loop save-on-exit, resume

---

## Task 2.2.5: TUI Polish & Reliability

**Files:**
- `chef_human/ui/debug_tui.py` — all improvements in this file
- `tests/test_agent/test_tui.py` — update for new features

### Why now
The debug TUI is functional but bare. Making it production-quality improves developer experience during Phase 3+ development.

### Design

#### 1. Color-coded plan steps

Each plan step gets a color based on status:
- `pending`: white
- `in_progress`: yellow (with spinner)
- `completed`: green with [✓]
- `failed`: red with [✗]
- `skipped`: dim/gray with [-]

```python
_STATUS_STYLES = {
    StepStatus.pending: "white",
    StepStatus.in_progress: "bold yellow",
    StepStatus.completed: "green",
    StepStatus.failed: "bold red",
    StepStatus.skipped: "dim white",
}

def _render_plan(self, plan: Plan) -> Tree:
    tree = Tree("📋 Plan", guide_style="dim")
    for step in plan.steps:
        style = _STATUS_STYLES.get(step.status, "white")
        label = f"[{style}]{step.index}. {step.description}[/{style}]"
        tree.add(label)
    return tree
```

#### 2. Expandable/collapsible reasoning panel

Use Rich `Syntax` with language="markdown" to render the reasoning text. Add a `max_lines` parameter that shows first N lines with "[... expanded ...]" toggle.

```python
class DebugTUI:
    def __init__(self, task: str, max_reasoning_lines: int = 50):
        self._reasoning_collapsed = False
        self._max_reasoning_lines = max_reasoning_lines
```

Press `r` to toggle reasoning panel between collapsed (last 5 lines) and expanded (full content).

#### 3. Searchable log panel

Use Rich `Text` with highlight. Add `/` key to enter search mode, type query, highlight matching lines.

```python
def _render_log(self) -> Panel:
    text = Text()
    search = getattr(self, "_log_search", None)
    for entry in self._log[-self._max_log_entries:]:
        if search and search in entry:
            text.append(entry + "\n", style="reverse")
        else:
            text.append(entry + "\n")
    return Panel(text, title="Log", border_style="dim")
```

Press `/` to focus search input at bottom, type query, results highlight in real-time.

#### 4. SIGINT handler

Gracefully stop the TUI on Ctrl+C:

```python
import signal

class DebugTUI:
    def _setup_signal_handler(self):
        signal.signal(signal.SIGINT, self._handle_sigint)
    
    def _handle_sigint(self, sig, frame):
        self._stop_live()
        print("\nTask interrupted by user.")
        sys.exit(130)
```

#### 5. Footer with status bar

Add a footer panel showing:
- Current step index / total steps
- Token count (from tracking, if available)
- Elapsed time
- Key bindings: `r` toggle reasoning, `/` search, `q` quit

```python
def _render_footer(self) -> Panel:
    elapsed = time.time() - self._start_time
    text = (
        f"Steps: {self._step_count}/{self._total_steps or '?'} | "
        f"Elapsed: {elapsed:.0f}s | "
        f"[dim]r[/dim] toggle reasoning  [dim]/[/dim] search  [dim]Ctrl+C[/dim] quit"
    )
    return Panel(text, border_style="dim")
```

#### 6. Extract `_render_layout()` as a method

Refactor `__init__` to use a `_render_layout()` method for cleaner structure. Keep `_refresh()` as the update trigger.

### Acceptance Criteria

- [ ] Plan steps color-coded by status (5 colors, one per status)
- [ ] Toggle key `r` expands/collapses reasoning panel
- [ ] `/` key enters search mode in log panel
- [ ] Search highlights matching log entries
- [ ] SIGINT handler gracefully stops TUI and exits
- [ ] Footer shows step count, elapsed time, key bindings
- [ ] `max_reasoning_lines` configurable (default 50)
- [ ] All existing TUI tests still pass
- [ ] 8+ new tests for color coding, toggle, search, footer

---

## Dependencies Map

```
2.2.1 streaming ───────────► 1.1 backend.py, 1.1.5 chatml.py,
                              2.1.3 react_loop.py, 2.1.6 debug_tui.py,
                              2.1.9 protocol.py
2.2.2 scratchpad ──────────► 2.1.9 prompts.py, 2.1.2 parser.py,
                              2.1.3 react_loop.py
2.2.3 headless ────────────► 2.1.7 main.py, 2.1.8 factories, 2.1.3 react_loop.py
2.2.4 persistence ─────────► 2.1.3 react_loop.py, 1.2 context.py,
                              2.1.7 main.py
2.2.5 tui polish ──────────► 2.1.6 debug_tui.py
```

## Implementation Order

1. **2.2.1** Streaming (backend → protocol → TUI → loop) — highest user impact
2. **2.2.2** Scratchpad (parser → prompts → loop) — improves model quality
3. **2.2.3** Headless mode (serialization → CLI → loop) — enables CI/scripting
4. **2.2.4** Persistence (context serialization → persistence module → loop → CLI) — saves work
5. **2.2.5** TUI polish (all in debug_tui.py) — production-quality UX

## Test Files (to be updated during implementation)

| Test file | Est. count | What it covers |
|-----------|-----------|----------------|
| `tests/test_agent/test_react_loop.py` | +10 | Streaming loop behavior, scratchpad updates, headless result format, save-on-exit |
| `tests/test_agent/test_prompts.py` | +2 | Scratchpad placeholder in prompt |
| `tests/test_agent/test_parser.py` | +5 | `extract_scratchpad()` with various formats |
| `tests/test_agent/test_persistence.py` | 15 | Serialization, save/load, list sessions, error handling |
| `tests/test_agent/test_tui.py` | +8 | Streaming callback, color-coded plan, toggle, search, footer |
| `tests/test_agent/test_main.py` | +6 | Headless flag, resume flag, save-dir flag |
| `tests/test_ollama_backend.py` | +3 | Streaming via mocked client (unit) |
| `tests/test_llamacpp_backend.py` | +3 | Streaming via mocked model (unit) |
| `tests/test_chatml.py` | +2 | Streaming message format |

**Estimated new tests**: ~54  
**Estimated total after 2.2**: ~540+ (up from ~486)

## Future Improvements (Post-2.2)

- **Parallel tool execution**: Execute independent tool calls concurrently within a single turn
- **Token usage display in TUI**: Show real-time token counts in footer
- **Session management CLI commands**: `chef-human sessions list`, `chef-human sessions show <id>`
- **Tool output streaming**: Real-time display of long-running bash command output
- **A/B prompt testing**: Framework for testing prompt variations
- **`.chef-human` config file**: Per-project configuration with tool overrides
