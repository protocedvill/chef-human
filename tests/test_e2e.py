"""End-to-end test: full chat + tool call flow.

Requires a running Ollama server with Qwen2.5-Coder:7b.
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
    pytest.mark.integration,
    pytest.mark.skipif(
        settings.llm_backend != "ollama",
        reason="E2E test requires Ollama backend",
    ),
]


def _ollama_reachable() -> bool:
    try:
        import ollama
        client = ollama.Client(host=settings.ollama_host)
        client.list()
        return True
    except Exception:
        return False


@pytest.fixture
def backend():
    if not _ollama_reachable():
        pytest.skip("Ollama server is not reachable")
    from chef_human.llm import create_backend
    return create_backend()


async def test_simple_chat(backend):
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
    if resp.message.tool_calls:
        tc = resp.message.tool_calls[0]
        assert tc["function"]["name"] == "add"
    else:
        assert "3" in resp.message.content


async def test_multi_turn(backend):
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
    from chef_human.llm.backend import EmbeddingRequest

    resp = await backend.embed(EmbeddingRequest(texts=["hello world", "test"]))
    assert len(resp.embeddings) == 2
    assert len(resp.embeddings[0]) > 0
