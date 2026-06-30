# Usage

## Running the Agent

Once installed and configured, start chef-human:

```bash
chef-human
```

> **Note**: The CLI entry point is not yet wired. In Phase 1.1, the agent is used programmatically.

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

Settings are loaded from `config.toml` by default. You can create a `.env` file or set environment variables with the `CHEF_` prefix:

```bash
export CHEF_OLLAMA_MODEL="deepseek-coder-v2:16b"
export CHEF_TEMPERATURE=0.2
```

Priority order: environment variable > `.env` file > `config.toml` > defaults.

See `config.toml` for all available settings.

---

## Project Structure

```
chef-human/
├── chef_human/        # Python package
│   ├── llm/          # LLM backends & interfaces
│   ├── agent/        # Agent loop (future)
│   ├── tools/        # Tool implementations (future)
│   └── codebase/     # Code understanding (future)
├── docs/             # Documentation
├── scripts/          # Setup & utility scripts
├── tests/            # Test suite
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
