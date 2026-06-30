from __future__ import annotations

import json
from typing import Any

from chef_human.llm.backend import Message, Role, ToolDefinition

SYSTEM_PROMPT = """You are chef-human, an AI software engineering assistant.
You have access to tools. Use them to accomplish tasks.
Always reason step by step before calling a tool.
When you are done, call the `finish` tool."""


def tool_to_dict(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def format_tool_definitions(tools: list[ToolDefinition]) -> str:
    entries: list[str] = []
    for t in tools:
        entries.append(
            json.dumps(
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
                indent=2,
            )
        )
    return "\n\n".join(entries)


def build_system_prompt(tools: list[ToolDefinition] | None = None) -> str:
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
