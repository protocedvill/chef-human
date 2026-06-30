import json

from chef_human.llm.backend import ToolDefinition
from chef_human.llm.chatml import (
    SYSTEM_PROMPT,
    assistant_message_with_tool_calls,
    build_system_prompt,
    format_tool_definitions,
    tool_result_message,
    tool_to_dict,
)


class TestToolToDict:
    def test_converts_tool_definition(self):
        tool = ToolDefinition(
            name="read_file",
            description="Read a file",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        result = tool_to_dict(tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "read_file"
        assert result["function"]["description"] == "Read a file"
        assert "path" in json.dumps(result)

    def test_includes_parameters(self):
        tool = ToolDefinition(
            name="calc",
            description="Calculate",
            parameters={"type": "object", "properties": {"x": {"type": "integer"}}},
        )
        result = tool_to_dict(tool)
        params = result["function"]["parameters"]
        assert params["type"] == "object"


class TestFormatToolDefinitions:
    def test_single_tool(self):
        tools = [
            ToolDefinition(
                name="tool_a",
                description="Does A",
                parameters={"type": "object", "properties": {}},
            )
        ]
        result = format_tool_definitions(tools)
        assert "tool_a" in result
        assert "Does A" in result
        # Should be valid JSON
        json.loads(result)

    def test_multiple_tools(self):
        tools = [
            ToolDefinition(name="a", description="A", parameters={}),
            ToolDefinition(name="b", description="B", parameters={}),
        ]
        result = format_tool_definitions(tools)
        assert "a" in result
        assert "b" in result

    def test_empty_list(self):
        assert format_tool_definitions([]) == ""


class TestBuildSystemPrompt:
    def test_returns_base_prompt_without_tools(self):
        result = build_system_prompt()
        assert result == SYSTEM_PROMPT

    def test_returns_base_prompt_with_empty_list(self):
        result = build_system_prompt(tools=[])
        assert result == SYSTEM_PROMPT

    def test_appends_tool_section_with_tools(self):
        tools = [
            ToolDefinition(
                name="my_tool",
                description="My test tool",
                parameters={"type": "object", "properties": {}},
            )
        ]
        result = build_system_prompt(tools)
        assert result.startswith(SYSTEM_PROMPT)
        assert "## Available Tools" in result
        assert "my_tool" in result
        assert "My test tool" in result
        assert "<tool_call>" in result

    def test_prompt_ends_with_format_instruction(self):
        tools = [
            ToolDefinition(
                name="t",
                description="test",
                parameters={"type": "object", "properties": {}},
            )
        ]
        result = build_system_prompt(tools)
        assert result.strip().endswith("</tool_call>")


class TestAssistantMessageWithToolCalls:
    def test_creates_message_with_content_and_calls(self):
        msg = assistant_message_with_tool_calls(
            "I'll use the tool now.",
            [{"function": {"name": "foo", "arguments": "{}"}}],
        )
        assert msg.role.value == "assistant"
        assert msg.content == "I'll use the tool now."
        assert len(msg.tool_calls) == 1

    def test_empty_content(self):
        msg = assistant_message_with_tool_calls("", [])
        assert msg.content == ""
        assert msg.tool_calls == []


class TestToolResultMessage:
    def test_creates_tool_result(self):
        msg = tool_result_message("call_123", '{"result": 42}')
        assert msg.role.value == "tool"
        assert msg.content == '{"result": 42}'
        assert msg.tool_call_id == "call_123"

    def test_empty_result(self):
        msg = tool_result_message("call_456", "")
        assert msg.content == ""
        assert msg.tool_call_id == "call_456"
