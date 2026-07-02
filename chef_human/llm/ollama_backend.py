from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncGenerator
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
)
from chef_human.llm.chatml import tool_to_dict

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_CONTEXT_LENGTH = 32768


class OllamaBackend(LLMBackend):
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
        self._async_client = ollama.AsyncClient(host=self._host)

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
            [tool_to_dict(t) for t in request.tools] if request.tools else None
        )

        response = await self._async_client.chat(
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

        tool_calls: list[dict[str, Any]] | None = None
        if "tool_calls" in reply and reply["tool_calls"]:
            tool_calls = reply["tool_calls"]
        else:
            tool_calls = _parse_tool_calls_from_content(reply.get("content", ""))

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

    async def complete_stream(
        self, request: CompletionRequest
    ) -> AsyncGenerator[tuple[str, CompletionResponse | None], None]:
        ollama_messages = [_to_ollama_msg(m) for m in request.messages]
        ollama_tools = (
            [tool_to_dict(t) for t in request.tools] if request.tools else None
        )

        stream = await self._async_client.chat(
            model=self._model,
            messages=ollama_messages,
            tools=ollama_tools or None,
            options={
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
                "stop": request.stop,
            },
            stream=True,
        )

        full_content = ""
        tool_calls: list[dict[str, Any]] | None = None
        final_usage: dict[str, int] | None = None

        async for chunk in stream:
            if "message" not in chunk:
                continue
            msg = chunk["message"]
            content_token = msg.get("content", "") or ""
            if content_token:
                full_content += content_token
                yield content_token, None

            # Last chunk may carry tool_calls
            if "tool_calls" in msg and msg["tool_calls"]:
                tool_calls = msg["tool_calls"]

            # Capture usage from final chunk
            if chunk.get("done"):
                final_usage = {
                    "prompt_tokens": chunk.get("prompt_eval_count", 0),
                    "completion_tokens": chunk.get("eval_count", 0),
                }

        if tool_calls is None:
            tool_calls = _parse_tool_calls_from_content(full_content)

        msg = Message(
            role=Role.assistant,
            content=full_content,
            tool_calls=tool_calls,
        )
        yield "", CompletionResponse(message=msg, usage=final_usage)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        embeddings = []
        for text in request.texts:
            resp = await self._async_client.embeddings(model=self._model, prompt=text)
            embeddings.append(resp["embedding"])
        return EmbeddingResponse(embeddings=embeddings)


def _to_ollama_msg(msg: Message) -> dict[str, Any]:
    d: dict[str, Any] = {"role": msg.role.value, "content": msg.content}
    if msg.tool_calls:
        d["tool_calls"] = msg.tool_calls
    if msg.tool_call_id:
        d["tool_call_id"] = msg.tool_call_id
    return d


def _parse_tool_calls_from_content(content: str) -> list[dict[str, Any]] | None:
    calls: list[dict[str, Any]] = []

    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
        try:
            calls.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            logger.warning("Failed to parse <tool_call>: %s", match.group(1))

    if calls:
        return calls

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "name" in parsed:
            calls.append(parsed)
            return calls
    except json.JSONDecodeError:
        pass

    return None
