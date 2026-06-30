import pytest

import ollama

from chef_human.llm.backend import (
    CompletionRequest,
    EmbeddingRequest,
    Message,
    Role,
    ToolDefinition,
)
from chef_human.llm.ollama_backend import OllamaBackend


def ollama_supports_embeddings() -> bool:
    try:
        client = ollama.Client()
        client.embeddings(model="qwen2.5-coder:7b", prompt="test")
        return True
    except ollama.ResponseError as e:
        if "does not support embeddings" in str(e):
            return False
        raise


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
            messages=[
                Message(role=Role.user, content="What is 2+2? Use the calculator tool.")
            ],
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
    tc = resp.message.tool_calls[0]
    name = tc.get("function", {}).get("name", "") or tc.get("name", "")
    assert "calculator" in name


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ollama_embedding():
    if not ollama_supports_embeddings():
        pytest.skip("Ollama server does not support embeddings (start with --embeddings)")
    backend = OllamaBackend()
    resp = await backend.embed(EmbeddingRequest(texts=["hello world", "test"]))
    assert len(resp.embeddings) == 2
    assert len(resp.embeddings[0]) > 0
