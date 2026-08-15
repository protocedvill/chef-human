from __future__ import annotations

import pytest

from chef_human.llm.ollama_backend import _parse_tool_calls_from_content


@pytest.mark.parametrize(
    ("content", "expected_name"),
    [
        ('{"name": "read", "arguments": {"path": "README.md"}}', "read"),
        (
            '```json\n{"name": "add", "arguments": {"a": 1, "b": 2}}\n```',
            "add",
        ),
        (
            'I will use a tool.\n```\n{"name": "write", "arguments": {}}\n```',
            "write",
        ),
        (
            '<tool_call>{"name": "list", "arguments": {}}</tool_call>',
            "list",
        ),
        (
            '{"function": {"name": "grep", "arguments": {"pattern": "TODO"}}}',
            "grep",
        ),
    ],
)
def test_parse_tool_call_content_variants(content: str, expected_name: str) -> None:
    calls = _parse_tool_calls_from_content(content)

    assert calls is not None
    call = calls[0]
    assert call["function"]["name"] == expected_name


@pytest.mark.parametrize("content", ["", "ordinary prose", "```json\n[1, 2]\n```"])
def test_non_tool_content_is_not_parsed(content: str) -> None:
    assert _parse_tool_calls_from_content(content) is None
