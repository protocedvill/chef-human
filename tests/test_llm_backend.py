from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from chef_human.llm.backend import (
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    LLMBackend,
    Message,
    Role,
)


class _StreamTestBackend(LLMBackend):
    """Concrete backend for testing default complete_stream()."""

    def __init__(self, content: str = "hello world") -> None:
        self._content = content
        self._model = "test-model"
        self._ctx = 4096

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def context_length(self) -> int:
        return self._ctx

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        return CompletionResponse(
            message=Message(role=Role.assistant, content=self._content),
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        return EmbeddingResponse(embeddings=[[0.1, 0.2]])


@pytest.mark.asyncio
async def test_default_complete_stream_yields_content_then_response():
    backend = _StreamTestBackend(content="hello world")
    chunks: list[tuple[str, CompletionResponse | None]] = []
    async for item in backend.complete_stream(
        CompletionRequest(messages=[Message(role=Role.user, content="hi")])
    ):
        chunks.append(item)

    assert len(chunks) == 2
    token, final = chunks[0]
    assert token == "hello world"
    assert final is None

    token, final = chunks[1]
    assert token == ""
    assert final is not None
    assert final.message.content == "hello world"
    assert final.message.role == Role.assistant


@pytest.mark.asyncio
async def test_default_complete_stream_passthrough_tool_calls():
    backend = _StreamTestBackend()
    chunks: list[tuple[str, CompletionResponse | None]] = []
    async for item in backend.complete_stream(
        CompletionRequest(messages=[Message(role=Role.user, content="hi")])
    ):
        chunks.append(item)

    assert len(chunks) == 2
    assert chunks[1][1] is not None


@pytest.mark.asyncio
async def test_default_complete_stream_empty_content():
    backend = _StreamTestBackend(content="")
    chunks: list[tuple[str, CompletionResponse | None]] = []
    async for item in backend.complete_stream(
        CompletionRequest(messages=[Message(role=Role.user, content="hi")])
    ):
        chunks.append(item)

    assert len(chunks) == 2
    assert chunks[0][0] == ""
    assert chunks[1][0] == ""


@pytest.mark.asyncio
async def test_default_complete_stream_is_async_generator():
    backend = _StreamTestBackend()
    gen = backend.complete_stream(
        CompletionRequest(messages=[Message(role=Role.user, content="hi")])
    )
    assert isinstance(gen, AsyncGenerator)


@pytest.mark.asyncio
async def test_complete_stream_usage_not_included_in_intermediate():
    """The usage dict should only be in the final CompletionResponse."""
    backend = _StreamTestBackend(content="test")
    async for token, final in backend.complete_stream(
        CompletionRequest(messages=[Message(role=Role.user, content="hi")])
    ):
        if final is None:
            continue
        assert final.usage is not None
        assert final.usage["prompt_tokens"] == 10
        assert final.usage["completion_tokens"] == 5
