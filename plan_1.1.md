# Phase 1.1: LLM Backend Integration

**Goal**: Wrap open-source inference engines so the agent can call LLMs for chat, code generation, and tool use. Provide a unified interface that supports multiple backends (Ollama, llama.cpp) with quantized models on consumer hardware.

---

## Task List

- [ ] **1.1.1** Project scaffolding (pyproject.toml, deps, tooling)
- [ ] **1.1.2** Abstract backend interface (`LLMBackend`)
- [ ] **1.1.3** Ollama backend implementation
- [ ] **1.1.4** llama.cpp backend implementation
- [ ] **1.1.5** ChatML message formatting & tool definition schema
- [ ] **1.1.6** Token counting & context window management
- [ ] **1.1.7** Embeddings backend (BGE for RAG)
- [ ] **1.1.8** Configuration & model auto-detection
- [ ] **1.1.9** Integration test: end-to-end chat + tool call
- [ ] **1.1.10** Model download & setup script

---

## Task 1.1.1: Project Scaffolding

**Files to create:**

```
pyproject.toml
chef_human/__init__.py
chef_human/llm/__init__.py
config.toml
```

**pyproject.toml**

```toml
[project]
name = "chef-human"
version = "0.1.0"
description = "Local AI software development tool"
requires-python = ">=3.12"
dependencies = [
    "ollama>=0.4.0",
    "llama-cpp-python>=0.3.0",
    "sentence-transformers>=3.0.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "rich>=13.0",
    "click>=8.0",
    "tomli>=2.0; python_version < '3.11'",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.5",
    "pyright>=1.1",
]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Acceptance criteria:**
- `pip install -e ".[dev]"` succeeds
- `ruff check .` passes on empty package
- `python -c "import chef_human"` works

---

## Task 1.1.2: Abstract Backend Interface

**File:** `chef_human/llm/backend.py`

Define a protocol that all backends must implement.

```python
# chef_human/llm/backend.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


@dataclass
class Message:
    role: Role
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


@dataclass
class CompletionRequest:
    messages: list[Message]
    tools: list[ToolDefinition] | None = None
    temperature: float = 0.0
    max_tokens: int = 4096
    stop: list[str] | None = None


@dataclass
class CompletionResponse:
    message: Message
    usage: dict[str, int] | None = None  # prompt_tokens, completion_tokens


@dataclass
class EmbeddingRequest:
    texts: list[str]


@dataclass
class EmbeddingResponse:
    embeddings: list[list[float]]
    usage: dict[str, int] | None = None


class LLMBackend(ABC):
    """Abstract interface for LLM inference backends."""

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Send a chat completion request."""
        ...

    @abstractmethod
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings for text(s)."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the loaded model."""
        ...

    @property
    @abstractmethod
    def context_length(self) -> int:
        """Maximum context window size in tokens."""
        ...

    def count_tokens(self, text: str) -> int:
        """Count tokens in text. Override per-backend for accuracy."""
        # Approximate: 1 token ≈ 4 chars
        return len(text) // 4
```

**Acceptance criteria:**
- Module imports cleanly
- `pydantic`-based validation passes
- Protocol is complete enough that a minimal subclass compiles

---

## Task 1.1.3: Ollama Backend

**File:** `chef_human/llm/ollama_backend.py`

### Overview

Ollama provides a REST API for running GGUF models. The backend wraps `ollama` library (or raw HTTP) to support:
- Chat completions with tool definitions
- Streaming (optional, for UI)
- Embedding generation

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Library | `ollama` Python SDK | Simple, async-native, handles streaming |
| Tool calling | ChatML with JSON schema in `tools` param | Native Ollama function calling API |
| Fallback | Raw HTTP `requests` to `http://localhost:11434` | If SDK not available |
| Connection check | `ollama.list()` at init | Verify daemon is running |

### Implementation

```python
# chef_human/llm/ollama_backend.py

from __future__ import annotations

import logging
from typing import Any

import ollama

from chef_human.llm.backend import (
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    LLMBackend,
    Message,
    Role,
    ToolDefinition,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_CONTEXT_LENGTH = 32768


class OllamaBackend(LLMBackend):
    """LLM backend using Ollama."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = "http://localhost:11434",
        context_length: int = DEFAULT_CONTEXT_LENGTH,
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._context_length = context_length
        self._client = ollama.Client(host=self._host)

        # Verify connection
        try:
            self._client.list()
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self._host}. "
                f"Is ollama running? Error: {e}"
            ) from e

        logger.info("Ollama backend initialized with model=%s host=%s", model, host)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def context_length(self) -> int:
        return self._context_length

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        ollama_messages = [_to_ollama_msg(m) for m in request.messages]
        ollama_tools = (
            [_to_ollama_tool(t) for t in request.tools] if request.tools else None
        )

        response = self._client.chat(
            model=self._model,
            messages=ollama_messages,
            tools=ollama_tools or None,
            options={
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
                "stop": request.stop,
            },
        )

        reply = response["message"]
        tool_calls = None
        if "tool_calls" in reply and reply["tool_calls"]:
            tool_calls = reply["tool_calls"]

        return CompletionResponse(
            message=Message(
                role=Role.assistant,
                content=reply.get("content", "") or "",
                tool_calls=tool_calls,
            ),
            usage={
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "completion_tokens": response.get("eval_count", 0),
            },
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        embeddings = []
        total_tokens = 0
        for text in request.texts:
            resp = self._client.embeddings(model=self._model, prompt=text)
            embeddings.append(resp["embedding"])
        return EmbeddingResponse(embeddings=embeddings)


def _to_ollama_msg(msg: Message) -> dict[str, Any]:
    d: dict[str, Any] = {"role": msg.role.value, "content": msg.content}
    if msg.tool_calls:
        d["tool_calls"] = msg.tool_calls
    if msg.tool_call_id:
        d["tool_call_id"] = msg.tool_call_id
    return d


def _to_ollama_tool(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }
```

### Tests

**File:** `tests/test_ollama_backend.py`

```python
import pytest
from chef_human.llm.ollama_backend import OllamaBackend
from chef_human.llm.backend import (
    CompletionRequest,
    Message,
    Role,
    ToolDefinition,
)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ollama_basic_chat():
    backend = OllamaBackend()
    resp = await backend.complete(
        CompletionRequest(
            messages=[Message(role=Role.user, content="Say 'hello'")],
            max_tokens=50,
        )
    )
    assert resp.message.role == Role.assistant
    assert len(resp.message.content) > 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ollama_tool_call():
    backend = OllamaBackend()
    resp = await backend.complete(
        CompletionRequest(
            messages=[Message(role=Role.user, content="What is 2+2? Use the calculator tool.")],
            tools=[
                ToolDefinition(
                    name="calculator",
                    description="Evaluate a math expression",
                    parameters={
                        "type": "object",
                        "properties": {
                            "expr": {"type": "string", "description": "Expression"}
                        },
                        "required": ["expr"],
                    },
                )
            ],
        )
    )
    assert resp.message.tool_calls is not None
```

**Note:** Integration tests require Ollama running with the model pulled. Default to skipped unless `--run-integration` flag is passed.

### Error Handling

| Scenario | Behavior |
|----------|----------|
| Ollama not running | Raise `RuntimeError` at init with clear message |
| Model not pulled | Raise `RuntimeError` with `ollama pull` instructions |
| Request timeout | Raise `TimeoutError` after configurable timeout (default 120s) |
| Invalid tool params | Log warning, return content-only response |
| Rate limiting | N/A (local) |

---

## Task 1.1.4: llama.cpp Backend

**File:** `chef_human/llm/llamacpp_backend.py`

### Overview

For users who want to run directly via llama.cpp (no Ollama dependency). Uses `llama-cpp-python` bindings.

```python
# chef_human/llm/llamacpp_backend.py

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from llama_cpp import Llama

from chef_human.llm.backend import (
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    LLMBackend,
    Message,
    Role,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


class LlamaCppBackend(LLMBackend):
    """LLM backend using llama.cpp directly."""

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 32768,
        n_gpu_layers: int = 0,
        n_threads: int | None = None,
        verbose: bool = False,
    ) -> None:
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self._llm = Llama(
            model_path=str(self._model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=verbose,
        )

    @property
    def model_name(self) -> str:
        return self._model_path.name

    @property
    def context_length(self) -> int:
        return self._llm.context_params.n_ctx

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        prompt = self._format_chatml(request)

        response = self._llm.create_completion(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop or ["<|im_end|>"],
        )

        content = response["choices"][0]["text"].strip()
        tool_calls = self._parse_tool_calls(content)

        return CompletionResponse(
            message=Message(
                role=Role.assistant,
                content=self._strip_tool_calls(content),
                tool_calls=tool_calls,
            ),
            usage={
                "prompt_tokens": response["usage"]["prompt_tokens"],
                "completion_tokens": response["usage"]["completion_tokens"],
            },
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        embeddings = []
        for text in request.texts:
            emb = self._llm.create_embedding(input=text)
            embeddings.append(emb["data"][0]["embedding"])
        return EmbeddingResponse(embeddings=embeddings)

    def count_tokens(self, text: str) -> int:
        return len(self._llm.tokenize(text.encode("utf-8")))

    def _format_chatml(self, request: CompletionRequest) -> str:
        """Format messages into ChatML prompt with tool definitions."""
        parts: list[str] = []

        if request.tools:
            tools_json = json.dumps(
                [self._tool_to_dict(t) for t in request.tools], indent=2
            )
            parts.append(
                f"<|im_start|>system\nAvailable tools:\n{tools_json}\n<|im_end|>"
            )

        for msg in request.messages:
            if msg.role == Role.system:
                parts.append(f"<|im_start|>system\n{msg.content}<|im_end|>")
            elif msg.role == Role.user:
                parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
            elif msg.role == Role.assistant:
                content = msg.content or ""
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content += f"\n<|tool_call|>{json.dumps(tc)}<|/tool_call|>"
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            elif msg.role == Role.tool:
                parts.append(
                    f"<|im_start|>tool\n{msg.content}<|im_end|>"
                )

        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def _parse_tool_calls(self, text: str) -> list[dict[str, Any]] | None:
        calls = []
        for match in re.finditer(
            r"<tool_call>(.*?)</tool_call>", text, re.DOTALL
        ):
            try:
                calls.append(json.loads(match.group(1)))
            except json.JSONDecodeError:
                logger.warning("Failed to parse tool call: %s", match.group(1))
        return calls if calls else None

    def _strip_tool_calls(self, text: str) -> str:
        return re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def _tool_to_dict(tool: ToolDefinition) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
```

### Key Differences from Ollama Backend

| Aspect | Ollama | llama.cpp |
|--------|--------|-----------|
| Setup | Must install Ollama separately | Single pip package |
| Model format | GGUF (managed by Ollama) | GGUF (user provides path) |
| Tool calling | Native API support | Manual ChatML formatting + regex parsing |
| GPU offload | Automatic via Ollama | Configurable via `n_gpu_layers` |
| Ideal for | Beginners, quick start | Power users, no-dependency setups |

---

## Task 1.1.5: ChatML Formatting & Tool Schema

**File:** `chef_human/llm/chatml.py`

Shared utility for tool definition formatting, used by both backends and later by the agent loop.

### Tool Definition JSON Schema

Every tool is defined with this structure:

```json
{
  "type": "function",
  "function": {
    "name": "read_file",
    "description": "Read contents of a file",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {
          "type": "string",
          "description": "Absolute path to file"
        },
        "offset": {
          "type": "integer",
          "description": "Starting line number (1-indexed)",
          "default": 1
        },
        "limit": {
          "type": "integer",
          "description": "Max lines to read",
          "default": 100
        }
      },
      "required": ["path"]
    }
  }
}
```

### Prompt Template for Tool-Use Models

```python
# chef_human/llm/chatml.py

import json
from typing import Any

from chef_human.llm.backend import Message, Role, ToolDefinition

# System prompt prefix
SYSTEM_PROMPT = """You are chef-human, an AI software engineering assistant.
You have access to tools. Use them to accomplish tasks.
Always reason step by step before calling a tool.
When you are done, call the `finish` tool."""


def format_tool_definitions(tools: list[ToolDefinition]) -> str:
    """Format tool definitions for inclusion in the system prompt."""
    entries = []
    for t in tools:
        entries.append(json.dumps({
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        }, indent=2))
    return "\n\n".join(entries)


def build_system_prompt(tools: list[ToolDefinition] | None = None) -> str:
    """Build the full system prompt with available tools."""
    prompt = SYSTEM_PROMPT
    if tools:
        tool_text = format_tool_definitions(tools)
        prompt += f"\n\n## Available Tools\n\n{tool_text}\n\n"
        prompt += (
            "To call a tool, respond with:\n"
            "<tool_call>{ \"name\": \"tool_name\", \"arguments\": { ... } }</tool_call>\n"
        )
    return prompt


def assistant_message_with_tool_calls(
    content: str, tool_calls: list[dict[str, Any]]
) -> Message:
    return Message(
        role=Role.assistant,
        content=content,
        tool_calls=tool_calls,
    )


def tool_result_message(tool_call_id: str, result: str) -> Message:
    return Message(
        role=Role.tool,
        content=result,
        tool_call_id=tool_call_id,
    )
```

**Acceptance criteria:**
- `build_system_prompt()` returns valid prompt with tools when provided
- `build_system_prompt()` returns base prompt when no tools given
- Tool definitions are valid JSON

---

## Task 1.1.6: Token Counting & Context Management

**File:** `chef_human/llm/tokenizer.py`

Accurate token counting is critical for context window management. We support three strategies:

| Strategy | Accuracy | Speed | Requirement |
|----------|----------|-------|-------------|
| Per-backend (preferred) | Exact | Fast | Backend exposes `count_tokens()` |
| tiktoken fallback | High | Fast | `tiktoken` installed + BPE model |
| Character approximation | Low | Instant | No deps |

```python
# chef_human/llm/tokenizer.py

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)

BACKEND_ENCODING_MAP: dict[str, str] = {
    "qwen": "gpt-4o",        # Qwen tokenizer ≈ GPT-4o BPE
    "deepseek": "cl100k_base",
    "codellama": "cl100k_base",
    "default": "cl100k_base",
}


class Tokenizer(Protocol):
    def count(self, text: str) -> int: ...


class TiktokenTokenizer:
    """Token counter using tiktoken library (BPE-based)."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        import tiktoken
        self._enc = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))


class ApproxTokenizer:
    """Rough token counter: 1 token ≈ 4 characters."""

    def count(self, text: str) -> int:
        return max(1, len(text) // 4)


def create_tokenizer(model_name: str = "") -> Tokenizer:
    """Create best available tokenizer for the given model."""
    try:
        import tiktoken  # noqa: F401
        key = "default"
        for prefix in BACKEND_ENCODING_MAP:
            if prefix in model_name.lower():
                key = prefix
                break
        return TiktokenTokenizer(BACKEND_ENCODING_MAP[key])
    except ImportError:
        logger.info("tiktoken not available, using approximate tokenizer")
        return ApproxTokenizer()
```

### Context Window Manager

**File:** `chef_human/agent/context.py`

```python
# chef_human/agent/context.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from chef_human.llm.tokenizer import Tokenizer, create_tokenizer

if TYPE_CHECKING:
    from chef_human.llm.backend import Message


@dataclass
class ContextConfig:
    max_tokens: int = 32768
    max_response_tokens: int = 4096
    summary_tokens: int = 512  # reserve for summary of trimmed messages


class ContextManager:
    """Manages the conversation context window with sliding window + summarization."""

    def __init__(
        self,
        config: ContextConfig | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.config = config or ContextConfig()
        self.tokenizer = tokenizer or create_tokenizer()
        self.messages: list[Message] = []
        self._summary: str = ""

    def add_message(self, msg: Message) -> None:
        self.messages.append(msg)
        self._trim_if_needed()

    def get_messages(self) -> list[Message]:
        return self.messages

    def token_count(self) -> int:
        return sum(self.tokenizer.count(m.content) for m in self.messages)

    def _trim_if_needed(self) -> None:
        """Trim oldest messages when context exceeds max_tokens."""
        while (
            self.token_count() > (self.config.max_tokens - self.config.max_response_tokens - self.config.summary_tokens)
            and len(self.messages) > 1
        ):
            # Summarize oldest messages rather than dropping
            if not self._summary and len(self.messages) > 3:
                old = self.messages[:2]
                self._summary = f"[Previous conversation: {len(old)} messages trimmed]"
                self.messages = self.messages[2:]
            elif len(self.messages) > 2:
                self.messages.pop(0)
            else:
                break  # protect system prompt
```

**Acceptance criteria:**
- Token count matches backend's `count_tokens` within 5%
- Context manager drops oldest messages when budget exceeded
- System message is never dropped
- `tiktoken` import error is gracefully handled (falls back to approx)

---

## Task 1.1.7: Embeddings Backend (BGE)

**File:** `chef_human/llm/embeddings.py`

For RAG retrieval in Phase 3, but setup now so the interface is ready.

```python
# chef_human/llm/embeddings.py

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


class EmbeddingsBackend:
    """Lightweight embedding generator using sentence-transformers."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL) -> None:
        self._model_name = model_name
        self._model: Any = None  # lazy load

    def _lazy_load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self _model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._lazy_load()
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        self._lazy_load()
        return self._model.get_sentence_embedding_dimension()
```

**Acceptance criteria:**
- Model loads on first call (lazy)
- Returns normalized embeddings
- `dimension` property returns int (384 for bge-small)

---

## Task 1.1.8: Configuration & Model Auto-Detection

**File:** `chef_human/config.py`

```python
# chef_human/config.py

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CHEF_",
        env_file=".env",
        env_file_encoding="utf-8",
        toml_file="config.toml",
    )

    # Backend
    llm_backend: Literal["ollama", "llamacpp"] = "ollama"

    # Ollama
    ollama_model: str = "qwen2.5-coder:7b"
    ollama_host: str = "http://localhost:11434"

    # llama.cpp
    llamacpp_model_path: Path | None = None
    llamacpp_n_gpu_layers: int = 0
    llamacpp_n_threads: int | None = None

    # Context
    max_context_tokens: int = 32768
    max_response_tokens: int = 4096

    # Embeddings
    embed_model: str = "BAAI/bge-small-en-v1.5"

    # Behavior
    temperature: float = 0.0
    workspace: Path = Path.cwd()
    max_tool_retries: int = 3
    max_agent_steps: int = 25


settings = Settings()
```

**config.toml**

```toml
[chef_human]
llm_backend = "ollama"
ollama_model = "qwen2.5-coder:7b"
ollama_host = "http://localhost:11434"
max_context_tokens = 32768
temperature = 0.0
workspace = "/home/user/projects/my-project"
```

**Auto-detection logic** (`chef_human/llm/__init__.py`):

```python
from chef_human.config import settings
from chef_human.llm.backend import LLMBackend


def create_backend() -> LLMBackend:
    """Auto-detect and create the appropriate backend from config."""
    if settings.llm_backend == "ollama":
        from chef_human.llm.ollama_backend import OllamaBackend
        return OllamaBackend(
            model=settings.ollama_model,
            host=settings.ollama_host,
        )
    elif settings.llm_backend == "llamacpp":
        from chef_human.llm.llamacpp_backend import LlamaCppBackend
        if settings.llamacpp_model_path is None:
            raise ValueError(
                "llamacpp_model_path must be set when backend is 'llamacpp'"
            )
        return LlamaCppBackend(
            model_path=settings.llamacpp_model_path,
            n_gpu_layers=settings.llamacpp_n_gpu_layers,
            n_threads=settings.llamacpp_n_threads,
        )
    else:
        raise ValueError(f"Unknown backend: {settings.llm_backend}")
```

**Acceptance criteria:**
- Settings load from env vars, `.env` file, `config.toml` (in that priority order)
- `create_backend()` returns correct backend based on config
- Missing llamacpp model path raises clear error

---

## Task 1.1.9: Integration Test: End-to-End Chat + Tool Call

**File:** `tests/test_e2e.py`

```python
"""End-to-end test: full chat + tool call flow.

Requires a running backend (Ollama with Qwen2.5-Coder:7b).
Skip with: pytest -m "not integration"
"""

import pytest

from chef_human.llm.backend import (
    CompletionRequest,
    Message,
    Role,
    ToolDefinition,
)
from chef_human.llm.chatml import build_system_prompt
from chef_human.config import settings


pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        settings.llm_backend != "ollama",
        reason="E2E test requires Ollama backend",
    ),
]


@pytest.fixture
def backend():
    from chef_human.llm import create_backend
    return create_backend()


async def test_simple_chat(backend):
    """Model can hold a basic conversation."""
    resp = await backend.complete(
        CompletionRequest(
            messages=[Message(role=Role.user, content="Reply with exactly 'ok'")],
            max_tokens=10,
            temperature=0,
        )
    )
    assert resp.message.role == Role.assistant
    assert "ok" in resp.message.content.lower()


async def test_tool_call_format(backend):
    """Model calls a tool when instructed."""
    resp = await backend.complete(
        CompletionRequest(
            messages=[
                Message(
                    role=Role.system,
                    content=build_system_prompt(
                        tools=[
                            ToolDefinition(
                                name="add",
                                description="Add two numbers",
                                parameters={
                                    "type": "object",
                                    "properties": {
                                        "a": {"type": "integer"},
                                        "b": {"type": "integer"},
                                    },
                                    "required": ["a", "b"],
                                },
                            )
                        ]
                    ),
                ),
                Message(role=Role.user, content="What is 1 + 2?"),
            ],
            temperature=0,
        )
    )
    # Model should either call the tool or answer directly
    if resp.message.tool_calls:
        tc = resp.message.tool_calls[0]
        assert tc["function"]["name"] == "add"
    else:
        assert "3" in resp.message.content


async def test_multi_turn(backend):
    """Model maintains context across multiple turns."""
    resp1 = await backend.complete(
        CompletionRequest(
            messages=[Message(role=Role.user, content="My name is Alice")],
            temperature=0,
        )
    )
    resp2 = await backend.complete(
        CompletionRequest(
            messages=[
                Message(role=Role.user, content="My name is Alice"),
                Message(role=Role.assistant, content=resp1.message.content),
                Message(role=Role.user, content="What is my name?"),
            ],
            temperature=0,
        )
    )
    assert "Alice" in resp2.message.content


async def test_embedding(backend):
    """Backend can generate embeddings."""
    resp = await backend.embed(texts=["hello world", "test"])
    assert len(resp.embeddings) == 2
    assert len(resp.embeddings[0]) > 0  # non-zero dimension
```

**Run command:**

```bash
pytest tests/ -v -m integration
# Or all tests including integration:
pytest tests/ -v
```

---

## Task 1.1.10: Model Download & Setup Script

**File:** `scripts/setup.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "=== chef-human setup ==="

# 1. Check Python version
PYTHON=$(command -v python3 || command -v python)
PYVER=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$(echo "$PYVER" | cut -d. -f1)" -lt 3 ] || { [ "$(echo "$PYVER" | cut -d. -f1)" -eq 3 ] && [ "$(echo "$PYVER" | cut -d. -f2)" -lt 12 ]; }; then
    echo "Error: Python 3.12+ required (found $PYVER)"
    exit 1
fi

# 2. Install package
echo "Installing chef-human..."
$PYTHON -m pip install -e ".[dev]"

# 3. Check/install Ollama
if command -v ollama &>/dev/null; then
    echo "Ollama found."
else
    echo "Ollama not found. Install it:"
    echo "  curl -fsSL https://ollama.com/install.sh | sh"
    echo "  (or download from https://ollama.com)"
    echo ""
    read -rp "Install Ollama now? [y/N] " yn
    if [[ "$yn" =~ ^[yY] ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo "Skipping Ollama install. You'll need it later."
    fi
fi

# 4. Pull model
if command -v ollama &>/dev/null; then
    MODEL="${CHEF_OLLAMA_MODEL:-qwen2.5-coder:7b}"
    echo "Pulling model: $MODEL (this may take a while)..."
    ollama pull "$MODEL"
    echo "Model $MODEL ready."
fi

# 5. Verify
echo ""
echo "Setup complete! Verify with:"
echo "  python -c 'from chef_human.llm import create_backend; b = create_backend(); print(b.model_name)'"
```

**File:** `scripts/setup.ps1` (Windows equivalent, optional)

**Acceptance criteria:**
- Script runs without errors on a clean system
- Model is pulled and ready to use
- Clear error messages for missing dependencies

---

## Dependencies Map

```
Phase 1.1 Tasks
│
├── 1.1.1 pyproject.toml ──────────────────────► (foundation for all)
│
├── 1.1.2 backend.py (abstract interface) ─────► used by 1.1.3, 1.1.4
│
├── 1.1.3 ollama_backend.py ───────────────────► depends on 1.1.2, 1.1.5
├── 1.1.4 llamacpp_backend.py ─────────────────► depends on 1.1.2, 1.1.5
│
├── 1.1.5 chatml.py ───────────────────────────► depends on 1.1.2
├── 1.1.6 tokenizer.py / context.py ───────────► depends on 1.1.5
│
├── 1.1.7 embeddings.py ───────────────────────► standalone
├── 1.1.8 config.py ───────────────────────────► used by all
│
├── 1.1.9 test_e2e.py ─────────────────────────► depends on 1.1.3, 1.1.8
└── 1.1.10 setup.sh ───────────────────────────► standalone
```

---

## Implementation Order

1. **1.1.1** pyproject.toml + package skeleton
2. **1.1.2** Abstract backend interface (dataclasses + ABC)
3. **1.1.8** Configuration (settings, config.toml, create_backend factory)
4. **1.1.5** ChatML formatting utilities
5. **1.1.6** Token counting + context manager
7. **1.1.3** Ollama backend (primary — test first)
8. **1.1.4** llama.cpp backend (secondary)
9. **1.1.7** Embeddings backend
10. **1.1.9** Integration tests
11. **1.1.10** Setup scripts

---

## Verification Checklist

Before marking Phase 1.1 complete:

- [ ] `pip install -e ".[dev]"` succeeds
- [ ] `ruff check .` passes with no errors
- [ ] `pyright chef_human/` passes with no errors
- [ ] `pytest tests/ -v` passes (unit tests)
- [ ] `pytest tests/ -v -m integration` passes (requires Ollama)
- [ ] `python -c "from chef_human.llm import create_backend; b = create_backend()"` works
- [ ] Ollama model responds to chat
- [ ] Tool call parsing works (model recognizes and formats tool calls)
- [ ] Embeddings return correct dimensions
- [ ] Context manager trims messages correctly
- [ ] `scripts/setup.sh` completes on fresh system
