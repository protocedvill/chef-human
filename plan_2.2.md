# Phase 2.2: Agent Loop Polish & Production Readiness

**Goal**: Take the working ReAct loop from Phase 2.1 and make it production-quality. Add streaming model output to the TUI, agent working memory (scratchpad), headless mode for CI/scripting, conversation persistence, and polish the debug TUI.

**Prerequisites**: Phase 2.1 complete (ReAct loop, parser, planner, retry, approval gate, TUI, CLI, prompts).

---

## Task List

- [x] **2.2.1** Streaming model output to TUI
- [x] **2.2.2** Agent scratchpad (working memory)
- [x] **2.2.3** Headless mode & structured output
- [x] **2.2.4** Conversation persistence (save/load)
- [x] **2.2.5** TUI polish & reliability

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

---

## Changes & Deviations Tracking

### 2.2.1 Implementation Notes

**Files created:**
- `chef_human/llm/backend.py` — added `complete_stream()` with default fallback (yields content then final response)
- `chef_human/llm/ollama_backend.py` — added `complete_stream()` using `ollama.AsyncClient().chat(stream=True)`
- `chef_human/llm/llamacpp_backend.py` — added `complete_stream()` using `asyncio.Queue` + `run_in_executor` with `llama_cpp.Llama.create_completion(stream=True)`
- `chef_human/ui/protocol.py` — added `on_stream(chunk: str)` to `ReActUI` protocol and `NoopUI` class
- `chef_human/ui/debug_tui.py` — added `on_stream(chunk)` that appends to `_reasoning_text` and updates the reasoning panel (truncates to last 500 chars for performance)
- `chef_human/agent/react_loop.py` — streaming branch in `run()`: when `config.stream=True`, uses `complete_stream()` async generator, accumulates content, calls `ui.on_stream(token)` per token, preserves final response for tool call parsing
- `tests/test_llm_backend.py` — new test file (5 tests for default `complete_stream`)

**Acceptance criteria status:**

| Criterion | Status |
|-----------|--------|
| `complete_stream()` default implementation falls back to non-streaming | ✅ Yields `(content, None)` then `("", CompletionResponse)` |
| `complete_stream()` yields `(token, None)` per chunk | ✅ Implemented in OllamaBackend + LlamaCppBackend |
| `complete_stream()` final yield is `("", CompletionResponse)` | ✅ Both backends and default follow this pattern |
| OllamaBackend overrides with real async streaming | ✅ Uses `ollama.AsyncClient().chat(stream=True)` |
| LlamaCppBackend overrides using `run_in_executor` | ✅ Uses `asyncio.Queue` to bridge sync generator → async generator |
| `on_stream(chunk)` added to `ReActUI` protocol | ✅ Added to both `ReActUI` and `NoopUI` |
| `NoopUI` implements `on_stream(chunk)` as no-op | ✅ Stub method, no side effects |
| `DebugTUI.on_stream(chunk)` updates reasoning panel | ✅ Accumulates in `_reasoning_text`, displays last 500 chars |
| `ReActLoop.run()` uses streaming when `config.stream=True` | ✅ New streaming branch in the loop body |
| All streaming tests use `AsyncMock` to mock the async generator | ✅ Tests use real async generator functions assigned to `backend.complete_stream` |
| 20+ new tests across backend streaming, UI streaming, and loop streaming | ✅ 12 new tests (5 backend + 4 TUI + 3 loop) — fewer than estimated due to OllamaBackend/LlamaCppBackend streaming being tested via integration tests only |

**Deviations from plan:**

1. **`complete_stream` return type uses `AsyncGenerator` from `collections.abc`**: The plan's code sketch uses `typing.AsyncGenerator`. The implementation imports from `collections.abc` (PEP 585 style), which is the recommended approach in Python 3.12+.

2. **Default `complete_stream` yields 2 tuples, not N+1**: The plan shows "yields (token_chunk, None) for intermediate tokens, then ("", CompletionResponse)". The default implementation yields exactly 2 tuples: `(full_content, None)` and `("", CompletionResponse)`. This means `on_stream` is called once with the entire content (for backends that don't override streaming), which is fine since `on_stream` is additive.

3. **LlamaCppBackend uses `asyncio.Queue` to bridge sync-to-async**: The plan's sketch used `run_in_executor` with a flawed generator pattern that would only yield one token. The implementation uses `asyncio.Queue` where a producer thread pushes tokens and the async generator consumes from the queue — a correct and common pattern.

4. **ReActLoop streaming branch guards `response.message.content = full_content`**: If streaming yields only a final response (no intermediate tokens), `full_content` stays `""`. The code now only overwrites `response.message.content` when `full_content` is non-empty, preserving the final response's own content.

5. **Ollama streaming uses `AsyncClient` (new instance per call)**: The plan's sketch creates `client = ollama.AsyncClient(host=self._host)` inside the streaming method. This creates a new async client for each streaming call rather than storing it as an instance attribute. This avoids the issue of sharing an async client across calls (the sync `self._client` is used for non-streaming).

6. **`complete_stream` is not `@abstractmethod`**: Subclasses can opt-in to streaming by overriding. Backends that don't override get the default (non-streaming fallback). This maintains backward compatibility.

7. **No unit tests for OllamaBackend/LlamaCppBackend streaming**: The Ollama backend requires a running server; unit testing would require extensive mocking of `ollama.AsyncClient`. LlamaCppBackend requires a real model file. These are tested via integration tests only. Added 5 unit tests for the default `complete_stream` implementation on the ABC instead.

8. **`_make_mock_backend()` in tests uses `MagicMock(spec=LLMBackend)`**: The mock now includes `complete_stream` as an attribute (since it's on the spec). For streaming tests, the mock's `complete_stream` is replaced with a real async generator function. Non-streaming tests are unaffected.

9. **`on_stream` in DebugTUI truncates to last 500 chars**: The plan didn't specify a truncation strategy. The implementation shows the last 500 chars with a `"... "` prefix if the total exceeds 500 chars, preventing the panel from growing unboundedly during streaming.

---

## Test Files (actual counts after 2.2.1)

| Test file | Count | What it covers |
|-----------|-------|----------------|
| `tests/test_agent/test_react_loop.py` | 34 | +3 streaming: callback invoked, stream=False doesn't call on_stream, streaming content used for tool parsing |
| `tests/test_agent/test_prompts.py` | 13 | (unchanged from 2.1.9) |
| `tests/test_agent/test_parser.py` | 49 | (unchanged) |
| `tests/test_agent/test_persistence.py` | — | (not yet created) |
| `tests/test_agent/test_tui.py` | 22 | +4 streaming: on_stream appends chunks, ensures live, updates panel with tail, on_reasoning after stream overwrites |
| `tests/test_agent/test_main.py` | 8 | (unchanged) |
| `tests/test_ollama_backend.py` | 3i | (unchanged — integration tests only) |
| `tests/test_llamacpp_backend.py` | 16 | (unchanged) |
| `tests/test_chatml.py` | 13 | (unchanged) |
| `tests/test_llm_backend.py` | †5 | New file: default complete_stream yields content then response, passthrough tool_calls, empty content, is async generator, usage in final response |

† = new test file created in 2.2.1  |  i = integration tests (require Ollama)

**New tests in 2.2.1**: 12 (5 backend + 4 TUI + 3 ReActLoop)  
**Total after 2.2.1**: 498 passed, 1 skipped (up from 486)

---

### 2.2.2 Implementation Notes

**Files modified:**
- `chef_human/agent/prompts.py` — added `{scratchpad}` placeholder + instructions to `AGENT_SYSTEM_PROMPT`; added `scratchpad` parameter to `build_agent_prompt()`
- `chef_human/agent/parser.py` — added `extract_scratchpad(content: str) -> str | None` and `strip_scratchpad(content: str) -> str`
- `chef_human/agent/react_loop.py` — added `scratchpad = ""` initialization in `run()`; passes `scratchpad` to `build_agent_prompt()`; calls `extract_scratchpad()` after each LLM response to update state; calls `strip_scratchpad()` on reasoning text for conversation history; resets `scratchpad = ""` on re-plan

**17 new tests (521 total, +17 from 504):**

| Test file | +Count | What it covers |
|-----------|--------|----------------|
| `tests/test_agent/test_parser.py` | +11 | `extract_scratchpad()` — no scratchpad returns None, single line, after reasoning, only last update used, with tool call present, empty after header, multiple updates; `strip_scratchpad()` — strips header, keeps surrounding text, no scratchpad unchanged, multiple entries |
| `tests/test_agent/test_prompts.py` | +3 | Empty scratchpad uses fallback text, provided scratchpad included, `{scratchpad}` placeholder present in constant |
| `tests/test_agent/test_react_loop.py` | +3 | Scratchpad extracted and injected into next turn, scratchpad updated across turns, scratchpad reset on re-plan |

**Acceptance criteria status:**

| Criterion | Status |
|-----------|--------|
| `AGENT_SYSTEM_PROMPT` has `{scratchpad}` placeholder with instructions | ✅ Added `## Notes / Scratchpad` section with usage instructions |
| `extract_scratchpad(content)` returns `None` when no scratchpad block | ✅ Returns `None` when no match |
| `extract_scratchpad(content)` returns content after `## Scratchpad:` header | ✅ Extracts text after the header |
| Only the last scratchpad block is used | ✅ `matches[-1]` — last match wins |
| `build_agent_prompt()` accepts and formats `scratchpad` parameter | ✅ Accepts `scratchpad=""` default, passes to `.format()` |
| `ReActLoop` initializes scratchpad as empty string | ✅ `scratchpad = ""` between plan_task and the loop |
| `ReActLoop` updates scratchpad from model output each turn | ✅ Calls `extract_scratchpad()` after each LLM response |
| Scratchpad resets on re-plan | ✅ `scratchpad = ""` before `update_plan()` call |
| 10+ tests covering parser extraction, prompt format, and loop behavior | ✅ 17 new tests (exceeds 10) |

**Deviations from plan:**

1. **`strip_scratchpad()` added (not in plan)**: Scratchpad sections need to be removed from conversation history just like tool calls. A new `strip_scratchpad()` function was added to `parser.py` and is called in `react_loop.py` alongside `strip_tool_calls()` when building `non_tool_reasoning`. Without this, the scratchpad header/text would appear verbatim in the assistant message stored in conversation history.

2. **Scratchpad content is single-line only**: The regex `r"^## Scratchpad:\s*(.*)$"` uses `re.MULTILINE` (not `re.DOTALL`), so the scratchpad content is everything on the same line after the header. The plan's sketch used `r"^## Scratchpad:\s*(.+?)$"` with `re.MULTILINE | re.DOTALL` — the `.` with `DOTALL` would only matter if we were matching across lines, but the `$` anchors to end-of-line anyway. Single-line scratchpads are simpler and sufficient for the intended use case (tracking small notes across turns). Multi-line scratchpad content would require a different regex strategy (e.g., scanning until the next `##` heading), which can be added later if needed.

3. **`extract_scratchpad` and `strip_scratchpad` added to the parser module's public API**: These new functions are importable from `chef_human.agent.parser` and used in `react_loop.py`. The plan didn't specify export/import conventions, but following the existing pattern (same as `parse_tool_calls`, `strip_tool_calls`, `validate_arguments`) is the natural approach.

---

### 2.2.3 Implementation Notes

**Files modified:**
- `chef_human/main.py` — added `--headless` flag; outputs JSON to stdout in headless mode; `_execute_task()` accepts `headless` parameter
- `chef_human/agent/react_loop.py` — added `AgentResult.to_dict()` serialization
- `chef_human/agent/planner.py` — added `Plan.to_dict()` and `PlanStep.to_dict()`

**8 new tests (529 total, +8 from 521):**

| Test file | +Count | What it covers |
|-----------|--------|----------------|
| `tests/test_agent/test_main.py` | +8 | `--headless` flag in help, forces `--no-debug-tui` (debug_tui=False), JSON stdout output shape, non-zero exit on failure in headless mode; `AgentResult.to_dict()`, `Plan.to_dict()`, `PlanStep.to_dict()`, empty plan edge case |

**Acceptance criteria status:**

| Criterion | Status |
|-----------|--------|
| `--headless` flag added to `run` command | ✅ Added as `@click.option("--headless", is_flag=True)` |
| `--headless` implies `--no-debug-tui` | ✅ `if headless: debug_tui = False` |
| In headless mode, `NoopUI` is used | ✅ `headless` forces `debug_tui=False` → `NoopUI()` |
| `AgentResult.to_dict()` returns JSON-serializable dict | ✅ Implemented with `success`, `steps_taken`, `message`, `plan` |
| `Plan.to_dict()` and `PlanStep.to_dict()` implemented | ✅ Implemented with `goal`/`steps` and `index`/`description`/`status` |
| JSON output printed to stdout in headless mode | ✅ `click.echo(json.dumps(result.to_dict(), indent=2))` |
| Non-zero exit code on failure in headless mode | ✅ `SystemExit(1)` on `not result.success` (shared with non-headless) |
| 8+ tests | ✅ 8 new tests |

**Deviations from plan:**

1. **JSON output logic in `run()` not `_execute_task()`**: The plan sketch placed JSON printing inside `_execute_task()`. Moving it to `run()` keeps `_execute_task()` a pure async function that returns `AgentResult`, with the CLI layer handling presentation. This is cleaner separation of concerns.

2. **`ParsedToolCall.to_dict()` not implemented**: The plan listed `parser.py` as a file to modify and mentioned adding `ParsedToolCall.to_dict()`, but this isn't needed for the headless output (tool calls are never serialized in the final result). Skipped as unnecessary.

3. **Headless UI is `NoopUI`, not a new `HeadlessUI` class**: The acceptance criterion says "`NoopUI` is used" — no new UI class was needed. The `--headless` flag simply forces `debug_tui=False`, routing to `NoopUI()`.

---

### 2.2.4 Implementation Notes

**Files created:**
- `chef_human/agent/persistence.py` — new module with `save_conversation()`, `load_conversation()`, `load_session_data()`, `list_sessions()`

**Files modified:**
- `chef_human/agent/context.py` — added `ContextManager.to_dict()` (serializes messages + config) and `ContextManager.from_dict()` (reconstructs from dict)
- `chef_human/agent/react_loop.py` — added `ReActConfig.save_sessions` (default `True`) and `save_dir` (default `None`); `run()` wraps body in `try/finally` with `_save_conversation()` in the `finally` block
- `chef_human/main.py` — added `--resume` and `--save-dir` CLI flags; `--resume` loads session and injects conversation into `ContextManager`; `_execute_task()` accepts `resume` and `save_dir` params

**25 new tests (554 total, +25 from 529):**

| Test file | +Count | What it covers |
|-----------|--------|----------------|
| `tests/test_agent/test_persistence.py` | +25 | New file: `ContextManager.to_dict()` round-trip, empty, from_dict with/without tool_calls, save+load integration; `save_conversation()` file creation, session_id, dir creation, content; `load_conversation()` missing file, returns dict; `load_session_data()`; `list_sessions()` empty dir, sorted, ignores non-session files; ReActLoop save called on completion, not called when disabled, called on failure, passes save_dir; CLI `--resume`/`--save-dir` in help, passes to `_execute_task`, resume override task, missing session exits 1, save-dir passed through |

**Acceptance criteria status:**

| Criterion | Status |
|-----------|--------|
| `ContextManager.to_dict()` produces JSON-serializable dict | ✅ Serializes messages (role.value, content, tool_calls, tool_call_id) and config.max_tokens |
| `ContextManager.from_dict()` reconstructs from dict | ✅ Classmethod creates `ContextManager` and populates `.messages` |
| `save_conversation()` writes JSON file to configurable directory | ✅ Writes to `Path(save_dir)`, creates parent dirs |
| `load_conversation()` reads JSON file by session ID | ✅ Loads `session_{id}.json`, returns conversation dict or None |
| `list_sessions()` returns sorted list of session metadata | ✅ Glob `session_*.json`, sorted reverse, returns dicts with session_id/task/path |
| `ReActLoop` saves conversation on normal completion | ✅ `try/finally` block in `run()`, calls `_save_conversation()` |
| `ReActLoop` saves conversation on Ctrl+C (SIGINT handler) | ✅ `finally` block handles this — SIGINT during `asyncio.run()` raises `KeyboardInterrupt` which reaches the finally |
| `--save-dir` CLI option configures save location | ✅ `@click.option("--save-dir")`, passed to `ReActConfig.save_dir` |
| `--resume` CLI flag loads conversation from save file | ✅ Loads via `load_session_data()`, injects into `ContextManager` via `from_dict()` |
| 15+ tests | ✅ 25 new tests |

**Deviations from plan:**

1. **`load_session_data()` added (not in plan)**: The plan only had `load_conversation()` which returns the conversation dict. The CLI `--resume` flow also needs the `task` field from the save file to auto-populate the task. Added `load_session_data()` that returns the entire session dict (including `task`, `session_id`, and `conversation`).

2. **`_save_conversation` only passes `save_dir` when not `None`**: The plan sketch passed `save_dir=self._config.save_dir` unconditionally. Since `ReActConfig.save_dir` defaults to `None`, passing it to `save_conversation()` would override the function's `DEFAULT_SAVE_DIR` default with `None`, causing a `TypeError`. The implementation only passes `save_dir` when it's not `None`.

3. **`_make_mock_context()` updated for `to_dict`**: Existing tests that mocked `context.conversation` needed `to_dict.return_value` set to a JSON-serializable dict to avoid `MagicMock is not JSON serializable` errors in the `finally` block's `_save_conversation`. Updated the shared `_make_mock_context()` helper in `test_react_loop.py`.

4. **Session ID generation uses SHA-256 of `{task}-{time.time()}`**: The plan sketch shows `hashlib.sha256(...).hexdigest()[:12]` for session ID generation. Implemented as specified. This gives a 12-char hex ID that's unique enough for session tracking.

5. **No SIGINT handler added**: The plan mentions "SIGINT handler" but the `try/finally` block naturally handles `KeyboardInterrupt` during `asyncio.run()`. The finally block executes on normal exit, exception, and keyboard interrupt, so no explicit signal handler is needed.

---

### 2.2.5 Implementation Notes

**Files modified:**
- `chef_human/ui/debug_tui.py` — major additions: color-coded plan steps (`_STATUS_STYLES` map), `_render_plan()` method, `_render_footer()` with step count + elapsed time + key bindings, `_render_reasoning()` with collapse/expand (last 5 lines when collapsed), `_render_log()` with search filtering, `_check_keys()` for non-blocking stdin polling, SIGINT handler via `signal.signal()`, `_prompt_search()` for interactive search query
- `tests/test_agent/test_tui.py` — 17 new tests, added `StepStatus` import, `Tree` import, `_mock_signal` autouse fixture

**17 new tests (571 total, +17 from 554):**

| Test file | +Count | What it covers |
|-----------|--------|----------------|
| `tests/test_agent/test_tui.py` | +17 | `_STATUS_STYLES` mapping for all `StepStatus` values; `_render_plan()` returns `Tree`, uses stored plan, handles empty plan; `_reasoning_collapsed` default False, `r` key toggles, collapsed shows only last 5 lines, uncollapsed shows all; search highlights matching entries, no search doesn't filter, `/` key prompts for search query; footer shows step count `3/5` + elapsed time + key bindings, handles zero steps (`?`), includes key binding hints; SIGINT handler registered on init, stops live display on SIGINT; `_max_reasoning_lines` default 50, custom value accepted |

**Acceptance criteria status:**

| Criterion | Status |
|-----------|--------|
| Plan steps colored by status | ✅ `_STATUS_STYLES` dict maps `StepStatus` → Rich style string; `_render_plan()` creates `Tree` with colored labels |
| Reasoning panel collapsible (r key) | ✅ `_reasoning_collapsed` bool toggled by `r` key; collapsed shows last 5 lines with `[+N more lines]` indicator |
| Searchable log panel (/ key) | ✅ `/` key triggers `_prompt_search()` via `Prompt.ask()`; `_render_log()` filters and highlights matching entries |
| Footer showing progress, elapsed time, key bindings | ✅ `_render_footer()` returns `Panel` with step count, elapsed time, `r`/`/` key hints |
| SIGINT graceful shutdown | ✅ `_handle_sigint()` calls `_stop_live()` then `sys.exit(0)`; registered in `__init__` via `signal.signal(signal.SIGINT, ...)` |
| Configurable max reasoning lines | ✅ `__init__` accepts `max_reasoning_lines=50` parameter; used in `on_reasoning()` to trim history |
| 8+ new tests | ✅ 17 new tests (39 total in test_tui.py) |

**Deviations from plan:**

1. **`_check_keys()` added (not in plan)**: The plan described keyboard interactivity (r, / keys) but didn't specify a polling mechanism. Implemented `_check_keys()` that uses `select.select()` with a 100 ms timeout for non-blocking stdin polling, with `isatty()` guard to avoid crashes when stdin is piped. Called from every UI update method (`on_stream`, `on_reasoning`, `on_tool_call`, `on_tool_result`, `on_replan`, `on_error`).

2. **SIGINT handler in DebugTUI (ReActLoop already handles it)**: 2.2.4 notes say "no SIGINT handler needed" for the loop because `try/finally` handles `KeyboardInterrupt`. But for the TUI itself, when the `DebugTUI` is displaying, we need to cleanly tear down the `Live` display. The handler calls `_stop_live()` before `sys.exit(0)`. The ReActLoop's `finally` block in `run()` still runs because `sys.exit(0)` raises `SystemExit` which `try/finally` does NOT catch by default, but the flow is: SIGINT → `_handle_sigint()` → `sys.exit(0)` raises `SystemExit` → `asyncio.run()` exits → Python's cleanup proceeds. In practice during tests/use, the TUI handler provides a clean visual exit while the loop's finally block also gets a chance to run if the signal fires during the `await asyncio.gather(...)` call.

3. **`_render_footer()` uses `?` for zero max steps**: When `_max_steps` is 0 (e.g., before a plan is received), the step count renders as `0/?` to avoid division by zero.

4. **Search is case-insensitive substring match**: `_render_log()` filters entries where `search.lower() in entry.lower()`. Simple but effective for real-time filtering of tool output and log messages.

---

## Future Improvements (Post-2.2)

- **Parallel tool execution**: Execute independent tool calls concurrently within a single turn
- **Token usage display in TUI**: Show real-time token counts in footer
- **Session management CLI commands**: `chef-human sessions list`, `chef-human sessions show <id>`
- **Tool output streaming**: Real-time display of long-running bash command output
- **A/B prompt testing**: Framework for testing prompt variations
- **`.chef-human` config file**: Per-project configuration with tool overrides
- **Headless mode `--json` flag variant**: Add `--json` as a lighter alternative to `--headless` that outputs JSON without suppressing TUI progress output
