from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any, Literal

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

DEFAULT_MODEL = "qwen3.8:27b"
DEFAULT_CONTEXT_LENGTH = 32768

# Every CompletionRequest in this codebase asks for temperature=0.0 (greedy
# decoding), which is fine for non-thinking calls but actively breaks
# thinking ones: Qwen's own docs warn greedy decoding under think mode
# "can lead to performance degradation and endless repetitions", and that
# is exactly what was observed live -- the model getting stuck re-deriving
# the same write-vs-edit tool choice for its entire token budget without
# ever emitting a decision. Switching to these sampling params (Qwen's
# documented thinking-mode recommendation) reproducibly fixed it in
# isolation: the identical prompt went from exhausting a 16384-token budget
# with zero output to a clean ~4300-token completion with a single pass
# over the same decision. Applied unconditionally whenever a call actually
# goes out with think enabled, overriding whatever temperature the caller
# specified -- callers have no way to know this constraint exists, so it's
# not reasonable to expect every one of them to opt in individually.
_THINK_SAMPLING_OVERRIDES: dict[str, Any] = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
}


class OllamaBackend(LLMBackend):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = "http://localhost:11434",
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        think: bool | Literal["low", "medium", "high"] = False,
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._context_length = context_length
        self._think: bool | Literal["low", "medium", "high"] = think
        # Set once a live call proves this model rejects the `think` option
        # entirely (Ollama returns a hard 400, not a silent no-op) -- lets a
        # single backend instance keep working for the rest of its life
        # instead of failing on every call. Real-world trigger: routing a
        # non-thinking-capable model (e.g. a small/fast one) to a caller that
        # sets a global `think` default meant for thinking models.
        self._think_unsupported = False
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

    async def _chat(self, **kwargs: Any) -> Any:
        """Wraps AsyncClient.chat with graceful degradation for models that
        reject `think` outright (Ollama returns a hard 400, not a silent
        no-op) -- downgrades to think=False and retries once, remembering
        the downgrade for the rest of this backend instance's life so it
        doesn't re-fail on every subsequent call."""
        think = False if self._think_unsupported else self._think
        try:
            return await self._async_client.chat(
                think=think, **self._with_sampling_overrides(kwargs, think)
            )
        except ollama.ResponseError as exc:
            if think is False or "does not support thinking" not in str(exc):
                # Either already sending think=False (retrying identically
                # would just fail the same way again) or an unrelated error.
                raise
            logger.warning(
                "Model %s does not support the `think` option; disabling it for "
                "this backend instance and retrying.",
                self._model,
            )
            self._think_unsupported = True
            return await self._async_client.chat(
                think=False, **self._with_sampling_overrides(kwargs, False)
            )

    @staticmethod
    def _with_sampling_overrides(
        kwargs: dict[str, Any], think: bool | Literal["low", "medium", "high"]
    ) -> dict[str, Any]:
        if not think:
            return kwargs
        options = dict(kwargs.get("options") or {})
        options.update(_THINK_SAMPLING_OVERRIDES)
        return {**kwargs, "options": options}

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        ollama_messages = [_to_ollama_msg(m) for m in request.messages]
        ollama_tools = (
            [tool_to_dict(t) for t in request.tools] if request.tools else None
        )

        response = await self._chat(
            model=self._model,
            messages=ollama_messages,
            tools=ollama_tools or None,
            options={
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
                "stop": request.stop,
                # Pin the served context window to the length this backend
                # advertises. Without it, Ollama uses each model's own
                # default -- and when the request exceeds it, Ollama
                # silently prunes messages server-side (dropping the user
                # turn), which the qwen3.8 renderer rejects with a fatal
                # "no user query found in messages" 500 (ollama issues
                # #17778/#17754). Explicit num_ctx makes the window the
                # caller budgets against and the window the server enforces
                # the same thing.
                "num_ctx": self._context_length,
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
            thinking=reply.get("thinking") or None,
        )

    async def complete_stream(
        self, request: CompletionRequest
    ) -> AsyncGenerator[tuple[str, CompletionResponse | None], None]:
        ollama_messages = [_to_ollama_msg(m) for m in request.messages]
        ollama_tools = (
            [tool_to_dict(t) for t in request.tools] if request.tools else None
        )

        stream = await self._chat(
            model=self._model,
            messages=ollama_messages,
            tools=ollama_tools or None,
            options={
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
                "stop": request.stop,
                # See complete() -- keeps the served window aligned with
                # the caller's budget so server-side pruning can never
                # silently drop the user turn.
                "num_ctx": self._context_length,
            },
            stream=True,
        )

        full_content = ""
        full_thinking = ""
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
            full_thinking += msg.get("thinking", "") or ""

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
        yield "", CompletionResponse(
            message=msg, usage=final_usage, thinking=full_thinking or None
        )

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
            parsed = json.loads(match.group(1))
            normalized = _normalize_tool_call(parsed)
            if normalized is not None:
                calls.append(normalized)
        except json.JSONDecodeError:
            logger.warning("Failed to parse <tool_call>: %s", match.group(1))

    if calls:
        return calls

    candidates = [content.strip()]
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
    )
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        normalized = _normalize_tool_call(parsed)
        if normalized is not None:
            calls.append(normalized)
            return calls

    return None


def _normalize_tool_call(value: object) -> dict[str, Any] | None:
    """Return Ollama's native tool-call shape for supported JSON variants."""
    if not isinstance(value, dict):
        return None
    function = value.get("function")
    if isinstance(function, dict) and function.get("name"):
        return value
    name = value.get("name")
    if not isinstance(name, str) or not name:
        return None
    return {
        "function": {
            "name": name,
            "arguments": value.get("arguments", {}),
        }
    }
