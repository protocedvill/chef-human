# Usage

## Running the Agent

Once installed (`pip install -e ".[dev]"`) and with Ollama running locally (`ollama serve`, with the
configured model pulled — default `qwen2.5-coder:7b`), run a task directly:

```bash
chef-human "Add a docstring to the parse_config function"
```

`chef-human` with no recognized subcommand defaults to `run`, so the line above is shorthand for:

```bash
chef-human run "Add a docstring to the parse_config function"
```

If you omit the task and stdin is a TTY, you'll be prompted for one interactively; if stdin is piped,
the task is read from stdin instead.

### `chef-human run [TASK]`

| Option | Default | Description |
|---|---|---|
| `--debug-tui` / `--no-debug-tui` | `--debug-tui` | Use the split-pane TUI (default) vs. plain streaming output |
| `--max-steps INT` | `25` | Max agent steps before giving up |
| `--workspace PATH` | current directory | Workspace root the agent operates in (must exist) |
| `--no-stream` | off | Disable streaming LLM output |
| `--headless` | off | No TUI; print the final result as JSON (implies `--no-debug-tui`) |
| `--json` | off | Also print the final result as JSON after normal streaming output |
| `--quiet` | off | Suppress all output except the final result |
| `--model TEXT` | from config | Override the LLM model for this run |
| `--temperature FLOAT` | from config | Override sampling temperature |
| `--config PATH` | `config.toml` | Use an alternate config file |
| `--resume TEXT` | — | Resume a saved session by ID (restores prior conversation history) |
| `--continue TEXT` | — | Alias for `--resume` |
| `--save-dir PATH` | `.chef-human/sessions` | Directory to save/load sessions from |

Examples:

```bash
# Non-interactive, CI-friendly: JSON result, no TUI
chef-human run "Fix the failing test in tests/test_foo.py" --headless

# Point at a different repo, cap steps, use a bigger model
chef-human run "Refactor the auth module" --workspace ../other-repo --max-steps 40 --model deepseek-coder-v2:16b

# Resume where a previous run left off
chef-human run --resume <session_id>
```

Exit code is `1` if the task did not succeed, `0` otherwise.

By default (`--debug-tui`, non-headless), `run` launches inside the split-pane Textual TUI
(file tree, chat/log, diff preview — see `chef-human tui` below) with your task auto-submitted;
the TUI exits automatically once that task finishes and the plain-text result summary prints as
usual. Pass `--no-debug-tui` for the older plain streaming output instead, or `--headless` for
JSON-only output with no TUI at all (the right choice for CI/scripting).

### `chef-human tui`

Explicitly start the split-pane Textual TUI as an interactive, multi-turn session (no task
argument — type tasks in the input box). Takes the same `--max-steps`, `--workspace`, `--model`,
`--temperature`, `--config`, `--resume`/`--continue`, and `--save-dir` options as `run`/`repl`.

```bash
chef-human tui --workspace .
```

Layout: a file tree, top-left (click a file to preview it), with a session stats panel below it
(tasks run, tool calls/errors/replans, cumulative token usage, current status and plan step, and
recent warnings — e.g. a blocked repeated tool call or a step the agent tried to skip); a chat/log
pane showing plan, reasoning, and tool activity, top-right; and a diff/preview pane, bottom-right,
that switches to show the diff the moment a write/edit/patch tool changes something; with an input
bar at the bottom. Destructive shell commands prompt via a modal Approve/Reject dialog instead of
a terminal `y/N` prompt. Click-drag in a log pane to select text and `Ctrl+C`/`Cmd+C` to copy it
to your system clipboard (works in most terminals; notably not macOS Terminal.app — use iTerm2 or
another terminal there). `Ctrl+Q` quits and auto-saves the session, same as `repl`.

### `chef-human repl`

Interactive multi-turn session against the same conversation/workspace context. Takes the same
`--max-steps`, `--workspace`, `--model`, `--temperature`, `--config`, `--resume`/`--continue`, and
`--save-dir` options as `run` (no task argument — you type tasks at the prompt).

```bash
chef-human repl --workspace .
```

In-REPL commands:

| Command | Effect |
|---|---|
| `/exit`, `/quit`, `/q` | Exit (auto-saves the session unless `/save` was already used) |
| `/help` | List commands |
| `/save` | Save the current session now |
| `/clear` | Clear conversation history |
| `/undo` | Undo the last file change |
| `/redo` | Redo the last undone change |
| `/tokens` | Show cumulative prompt/completion token usage |
| `/history` | Show the last 20 messages |

Any other input is sent to the agent as a new task in the same conversation.

### `chef-human session ...`

Manage saved sessions (JSON files under `.chef-human/sessions` by default, override with `--save-dir`):

```bash
chef-human session list
chef-human session show <session_id>
chef-human session export <session_id> --format md   # or json (default)
chef-human session delete <session_id>
```

### `chef-human show-config`

Print the fully-resolved effective configuration (after merging `config.toml`, any project
`.chef-human/config.toml`, and `CHEF_*` environment variables):

```bash
chef-human show-config
chef-human show-config --config other-config.toml
```

## Programmatic Usage

### Basic chat

```python
import asyncio
from chef_human.llm import create_backend
from chef_human.llm.backend import CompletionRequest, Message, Role

async def main():
    backend = create_backend()  # auto-detects backend from config
    resp = await backend.complete(
        CompletionRequest(
            messages=[Message(role=Role.user, content="Write a Python Fibonacci function")],
        )
    )
    print(resp.message.content)

asyncio.run(main())
```

### With tool calling

```python
from chef_human.llm.backend import ToolDefinition, build_system_prompt

tools = [
    ToolDefinition(
        name="read_file",
        description="Read contents of a file",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to file"}
            },
            "required": ["path"],
        },
    )
]

system_prompt = build_system_prompt(tools)
resp = await backend.complete(
    CompletionRequest(
        messages=[
            Message(role=Role.system, content=system_prompt),
            Message(role=Role.user, content="Read src/main.py"),
        ],
        tools=tools,
    )
)
if resp.message.tool_calls:
    print("Tool call:", resp.message.tool_calls)
else:
    print("Response:", resp.message.content)
```

### Backend selection

You can explicitly choose a backend:

```python
from chef_human.llm.ollama_backend import OllamaBackend
backend = OllamaBackend(model="qwen2.5-coder:7b")
```

```python
from chef_human.llm.llamacpp_backend import LlamaCppBackend
backend = LlamaCppBackend(model_path="/path/to/model.gguf", n_gpu_layers=20)
```

### Embeddings

```python
from chef_human.llm.embeddings import EmbeddingsBackend

emb = EmbeddingsBackend()
vec = emb.embed_single("hello world")
print(f"Vector dimension: {len(vec)}")  # 384 for bge-small
```

### Token counting

```python
from chef_human.llm.tokenizer import create_tokenizer

tok = create_tokenizer("qwen2.5-coder:7b")
print(tok.count("Hello, world!"))  # approximate token count
```

---

## Configuration

Settings are loaded from `config.toml` in the current directory by default, or set as environment
variables with the `CHEF_` prefix:

```bash
export CHEF_OLLAMA_MODEL="deepseek-coder-v2:16b"
export CHEF_TEMPERATURE=0.2
```

Load order (later overrides earlier): `config.toml` → nearest ancestor `.chef-human/config.toml` →
`CHEF_*` environment variables → per-run CLI flags (`--model`, `--temperature`, `--config`). Run
`chef-human show-config` to see the fully resolved settings.

See `config.toml` for all available settings.

---

## Project Structure

```
chef-human/
├── chef_human/        # Python package
│   ├── llm/          # LLM backends (Ollama, llama.cpp) & shared protocol
│   ├── agent/         # ReAct loop, planner, context assembly, symbol index, RAG
│   ├── tools/          # File/shell/refactor/undo/etc. tools the agent can call
│   └── ui/             # Debug TUI, streaming, REPL, no-op UI implementations
├── docs/             # Documentation
├── scripts/          # Setup & utility scripts
├── tests/            # Test suite (mirrors chef_human/ layout)
├── config.toml       # User configuration
└── pyproject.toml    # Package metadata
```

---

## Running Tests

```bash
# Unit tests
pytest tests/ -v

# With integration tests (requires Ollama + model)
pytest tests/ -v -m integration

# All tests
pytest tests/ -v
```
