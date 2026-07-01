from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
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
    parameters: dict[str, Any]


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
    usage: dict[str, int] | None = None


@dataclass
class EmbeddingRequest:
    texts: list[str]


@dataclass
class EmbeddingResponse:
    embeddings: list[list[float]]
    usage: dict[str, int] | None = None


class LLMBackend(ABC):
    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        ...

    async def complete_stream(
        self, request: CompletionRequest
    ) -> AsyncGenerator[tuple[str, CompletionResponse | None], None]:
        """Default streaming: yields entire response as a single chunk.

        Subclasses (OllamaBackend, LlamaCppBackend) should override
        for true token-by-token streaming.

        Yields (token_chunk, None) for intermediate tokens, then
        ("", CompletionResponse) for the final response.
        """
        resp = await self.complete(request)
        yield resp.message.content, None
        yield "", resp

    @abstractmethod
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @property
    @abstractmethod
    def context_length(self) -> int:
        ...

    def count_tokens(self, text: str) -> int:
        return len(text) // 4
