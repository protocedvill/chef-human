"""Non-integration unit tests for OllamaBackend that don't hit a real model
-- only the constructor's connectivity check needs a live Ollama server
(present in this dev environment); _async_client.chat itself is mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock

import ollama
import pytest

from chef_human.llm.backend import CompletionRequest, Message, Role
from chef_human.llm.ollama_backend import OllamaBackend


def _think_unsupported_error() -> ollama.ResponseError:
    return ollama.ResponseError('"some-model" does not support thinking', 400)


@pytest.mark.asyncio
async def test_think_unsupported_error_falls_back_to_no_thinking(monkeypatch):
    backend = OllamaBackend(model="qwen2.5-coder:7b", think="low")

    calls: list[dict] = []

    async def fake_chat(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise _think_unsupported_error()
        return {"message": {"content": "ok"}, "prompt_eval_count": 1, "eval_count": 1}

    monkeypatch.setattr(backend._async_client, "chat", AsyncMock(side_effect=fake_chat))

    response = await backend.complete(
        CompletionRequest(messages=[Message(role=Role.user, content="hi")])
    )

    assert response.message.content == "ok"
    assert len(calls) == 2
    assert calls[0]["think"] == "low"
    assert calls[1]["think"] is False
    # the backend remembers the downgrade for subsequent calls
    assert backend._think_unsupported is True


@pytest.mark.asyncio
async def test_subsequent_calls_skip_the_failed_think_attempt(monkeypatch):
    backend = OllamaBackend(model="qwen2.5-coder:7b", think="low")

    calls: list[dict] = []

    async def fake_chat(**kwargs):
        calls.append(kwargs)
        if kwargs.get("think"):
            raise _think_unsupported_error()
        return {"message": {"content": "ok"}, "prompt_eval_count": 1, "eval_count": 1}

    monkeypatch.setattr(backend._async_client, "chat", AsyncMock(side_effect=fake_chat))

    await backend.complete(CompletionRequest(messages=[Message(role=Role.user, content="hi")]))
    calls.clear()
    await backend.complete(CompletionRequest(messages=[Message(role=Role.user, content="again")]))

    # second call goes straight to think=False, no failed attempt first
    assert len(calls) == 1
    assert calls[0]["think"] is False


@pytest.mark.asyncio
async def test_unrelated_response_errors_are_not_swallowed(monkeypatch):
    backend = OllamaBackend(model="qwen2.5-coder:7b", think="low")

    async def fake_chat(**kwargs):
        raise ollama.ResponseError("model not found", 404)

    monkeypatch.setattr(backend._async_client, "chat", AsyncMock(side_effect=fake_chat))

    with pytest.raises(ollama.ResponseError, match="model not found"):
        await backend.complete(CompletionRequest(messages=[Message(role=Role.user, content="hi")]))


@pytest.mark.asyncio
async def test_complete_captures_thinking_when_present(monkeypatch):
    backend = OllamaBackend(model="qwen2.5-coder:7b", think="low")

    async def fake_chat(**kwargs):
        return {
            "message": {"content": "ok", "thinking": "reasoning about the task"},
            "prompt_eval_count": 1,
            "eval_count": 1,
        }

    monkeypatch.setattr(backend._async_client, "chat", AsyncMock(side_effect=fake_chat))

    response = await backend.complete(
        CompletionRequest(messages=[Message(role=Role.user, content="hi")])
    )

    assert response.thinking == "reasoning about the task"


@pytest.mark.asyncio
async def test_complete_thinking_is_none_when_absent(monkeypatch):
    backend = OllamaBackend(model="qwen2.5-coder:7b", think=False)

    async def fake_chat(**kwargs):
        return {"message": {"content": "ok"}, "prompt_eval_count": 1, "eval_count": 1}

    monkeypatch.setattr(backend._async_client, "chat", AsyncMock(side_effect=fake_chat))

    response = await backend.complete(
        CompletionRequest(messages=[Message(role=Role.user, content="hi")])
    )

    assert response.thinking is None


@pytest.mark.asyncio
async def test_complete_stream_accumulates_thinking_across_chunks(monkeypatch):
    backend = OllamaBackend(model="qwen2.5-coder:7b", think="low")

    async def fake_stream():
        yield {"message": {"content": "", "thinking": "step one. "}}
        yield {"message": {"content": "hello", "thinking": "step two."}}
        yield {
            "message": {"content": ""},
            "done": True,
            "prompt_eval_count": 1,
            "eval_count": 1,
        }

    async def fake_chat(**kwargs):
        return fake_stream()

    monkeypatch.setattr(backend._async_client, "chat", AsyncMock(side_effect=fake_chat))

    final_response = None
    async for _token, response in backend.complete_stream(
        CompletionRequest(messages=[Message(role=Role.user, content="hi")])
    ):
        if response is not None:
            final_response = response

    assert final_response is not None
    assert final_response.thinking == "step one. step two."


@pytest.mark.asyncio
async def test_think_calls_override_greedy_sampling(monkeypatch):
    """Greedy decoding (temperature=0, the default every CompletionRequest
    uses) is documented by Qwen to cause endless repetition loops under
    think mode, reproduced live in this session -- thinking calls must
    always go out with the non-greedy override, regardless of what
    temperature the caller asked for."""
    backend = OllamaBackend(model="qwen2.5-coder:7b", think="low")

    calls: list[dict] = []

    async def fake_chat(**kwargs):
        calls.append(kwargs)
        return {"message": {"content": "ok"}, "prompt_eval_count": 1, "eval_count": 1}

    monkeypatch.setattr(backend._async_client, "chat", AsyncMock(side_effect=fake_chat))

    await backend.complete(
        CompletionRequest(
            messages=[Message(role=Role.user, content="hi")], temperature=0.0
        )
    )

    assert len(calls) == 1
    options = calls[0]["options"]
    assert options["temperature"] == 0.6
    assert options["top_p"] == 0.95
    assert options["top_k"] == 20
    assert options["min_p"] == 0.0


@pytest.mark.asyncio
async def test_non_think_calls_keep_caller_temperature(monkeypatch):
    backend = OllamaBackend(model="qwen2.5-coder:7b", think=False)

    calls: list[dict] = []

    async def fake_chat(**kwargs):
        calls.append(kwargs)
        return {"message": {"content": "ok"}, "prompt_eval_count": 1, "eval_count": 1}

    monkeypatch.setattr(backend._async_client, "chat", AsyncMock(side_effect=fake_chat))

    await backend.complete(
        CompletionRequest(
            messages=[Message(role=Role.user, content="hi")], temperature=0.0
        )
    )

    assert len(calls) == 1
    assert calls[0]["options"]["temperature"] == 0.0
    assert "top_p" not in calls[0]["options"]


@pytest.mark.asyncio
async def test_fallback_after_think_unsupported_drops_sampling_override(monkeypatch):
    """Once a model proves it rejects `think` outright, the retry -- and
    every call after it -- must not carry the thinking-mode sampling
    override either, since it's no longer a thinking call."""
    backend = OllamaBackend(model="some-model", think="low")

    calls: list[dict] = []

    async def fake_chat(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise _think_unsupported_error()
        return {"message": {"content": "ok"}, "prompt_eval_count": 1, "eval_count": 1}

    monkeypatch.setattr(backend._async_client, "chat", AsyncMock(side_effect=fake_chat))

    await backend.complete(
        CompletionRequest(
            messages=[Message(role=Role.user, content="hi")], temperature=0.0
        )
    )

    assert len(calls) == 2
    assert calls[0]["options"]["temperature"] == 0.6  # the failed thinking attempt
    assert calls[1]["options"]["temperature"] == 0.0  # fallback, caller's own value


@pytest.mark.asyncio
async def test_no_retry_when_think_already_false(monkeypatch):
    backend = OllamaBackend(model="qwen2.5-coder:7b", think=False)

    calls: list[dict] = []

    async def fake_chat(**kwargs):
        calls.append(kwargs)
        raise _think_unsupported_error()

    monkeypatch.setattr(backend._async_client, "chat", AsyncMock(side_effect=fake_chat))

    with pytest.raises(ollama.ResponseError):
        await backend.complete(CompletionRequest(messages=[Message(role=Role.user, content="hi")]))

    # think was already False, so there's nothing to downgrade to -- fails once, not looped
    assert len(calls) == 1
