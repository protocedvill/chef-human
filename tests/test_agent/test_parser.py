from __future__ import annotations

from chef_human.agent.parser import (
    ParsedToolCall,
    ToolCallParseError,
    _is_ollama_tool_call,
    _is_tool_call_object,
    _parse_single_call,
    parse_tool_calls,
    strip_tool_calls,
    validate_arguments,
)


class TestToolCallParseError:
    def test_message(self):
        exc = ToolCallParseError("parse failed")
        assert str(exc) == "parse failed"
        assert exc.raw_content == ""

    def test_with_raw_content(self):
        exc = ToolCallParseError("bad json", raw_content="{bad}")
        assert exc.raw_content == "{bad}"


class TestParsedToolCall:
    def test_fields(self):
        tc = ParsedToolCall(name="read", arguments={"path": "x.py"}, raw='{"name":"read","arguments":{"path":"x.py"}}')
        assert tc.name == "read"
        assert tc.arguments == {"path": "x.py"}
        assert tc.raw


class TestIsToolCallObject:
    def test_valid(self):
        assert _is_tool_call_object({"name": "read", "arguments": {}}) is True

    def test_missing_name(self):
        assert _is_tool_call_object({"arguments": {}}) is False

    def test_missing_arguments(self):
        assert _is_tool_call_object({"name": "read"}) is False

    def test_empty(self):
        assert _is_tool_call_object({}) is False


class TestIsOllamaToolCall:
    def test_valid(self):
        assert _is_ollama_tool_call({"function": {"name": "read"}}) is True

    def test_function_not_dict(self):
        assert _is_ollama_tool_call({"function": "not_a_dict"}) is False

    def test_missing_function(self):
        assert _is_ollama_tool_call({}) is False


class TestParseSingleCall:
    def test_direct_format(self):
        result = _parse_single_call(
            {"name": "read", "arguments": {"path": "x.py"}},
            raw='{"name":"read","arguments":{"path":"x.py"}}',
        )
        assert result is not None
        assert result.name == "read"
        assert result.arguments == {"path": "x.py"}

    def test_ollama_format(self):
        result = _parse_single_call(
            {"function": {"name": "read", "arguments": {"path": "x.py"}}},
            raw='{"function":{"name":"read","arguments":{"path":"x.py"}}}',
        )
        assert result is not None
        assert result.name == "read"
        assert result.arguments == {"path": "x.py"}

    def test_arguments_as_string(self):
        result = _parse_single_call(
            {"name": "read", "arguments": '{"path": "x.py"}'},
            raw='{"name":"read","arguments":"{\\"path\\": \\"x.py\\"}"}',
        )
        assert result is not None
        assert result.name == "read"
        assert result.arguments == {"path": "x.py"}

    def test_arguments_as_invalid_string(self):
        result = _parse_single_call(
            {"name": "read", "arguments": "not-json"},
            raw='{"name":"read","arguments":"not-json"}',
        )
        assert result is not None
        assert result.arguments == {"raw": "not-json"}

    def test_missing_name(self):
        result = _parse_single_call({"arguments": {}}, raw="{}")
        assert result is None

    def test_empty_name(self):
        result = _parse_single_call({"name": "", "arguments": {}}, raw='{"name":""}')
        assert result is None

    def test_ollama_format_with_string_args(self):
        result = _parse_single_call(
            {"function": {"name": "bash", "arguments": '{"command": "ls"}'}},
            raw="...",
        )
        assert result is not None
        assert result.name == "bash"
        assert result.arguments == {"command": "ls"}


class TestParseToolCalls:
    def test_empty_content(self):
        assert parse_tool_calls("") == []

    def test_no_tool_calls(self):
        assert parse_tool_calls("Just some text without any tool calls") == []

    def test_single_tool_call_tag(self):
        content = 'Before <tool_call>{"name": "read", "arguments": {"path": "x.py"}}</tool_call> After'
        calls = parse_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].name == "read"
        assert calls[0].arguments == {"path": "x.py"}

    def test_multiple_tool_call_tags(self):
        content = (
            '<tool_call>{"name": "read", "arguments": {"path": "a.py"}}</tool_call>'
            ' text '
            '<tool_call>{"name": "write", "arguments": {"path": "b.py", "content": "x"}}</tool_call>'
        )
        calls = parse_tool_calls(content)
        assert len(calls) == 2
        assert calls[0].name == "read"
        assert calls[1].name == "write"

    def test_tool_call_tag_ollama_format(self):
        content = '<tool_call>{"function": {"name": "bash", "arguments": {"command": "ls"}}}</tool_call>'
        calls = parse_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].name == "bash"
        assert calls[0].arguments == {"command": "ls"}

    def test_json_code_block_direct(self):
        content = 'Reasoning...\n```json\n{"name": "grep", "arguments": {"pattern": "foo"}}\n```\nMore text.'
        calls = parse_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].name == "grep"
        assert calls[0].arguments == {"pattern": "foo"}

    def test_json_code_block_ollama(self):
        content = '```json\n{"function": {"name": "glob", "arguments": {"pattern": "*.py"}}}\n```'
        calls = parse_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].name == "glob"

    def test_code_block_without_json_prefix(self):
        content = '```\n{"name": "ls", "arguments": {}}\n```'
        calls = parse_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].name == "ls"

    def test_bare_json_object_fallback(self):
        content = 'I think we should read the file. {"name": "read", "arguments": {"path": "x.py"}}'
        calls = parse_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].name == "read"

    def test_no_duplicate_across_formats(self):
        """When same call appears in both tag and code block, deduplicate."""
        content = (
            '<tool_call>{"name": "read", "arguments": {"path": "x.py"}}</tool_call>'
            ' ```json\n{"name": "read", "arguments": {"path": "x.py"}}\n```'
        )
        calls = parse_tool_calls(content)
        assert len(calls) == 1

    def test_malformed_json_in_tag(self):
        content = '<tool_call>{bad json}</tool_call>'
        calls = parse_tool_calls(content)
        assert calls == []

    def test_malformed_json_in_code_block(self):
        content = '```json\n{not valid}\n```'
        calls = parse_tool_calls(content)
        assert calls == []

    def test_mixed_content_with_multiple_formats(self):
        content = (
            'First let me check.\n'
            '<tool_call>{"name": "read", "arguments": {"path": "a.py"}}</tool_call>\n'
            'Now I see it.\n'
            '```json\n{"name": "write", "arguments": {"path": "b.py", "content": "data"}}\n```\n'
            'Done.'
        )
        calls = parse_tool_calls(content)
        assert len(calls) == 2
        names = [c.name for c in calls]
        assert "read" in names
        assert "write" in names

    def test_non_tool_json_in_code_block_not_parsed(self):
        """Code blocks with JSON that isn't a tool call are ignored."""
        content = '```json\n{"key": "value"}\n```'
        calls = parse_tool_calls(content)
        assert calls == []

    def test_bare_non_tool_json_ignored(self):
        content = 'Some text {"key": "value"} more text'
        calls = parse_tool_calls(content)
        assert calls == []


class TestValidateArguments:
    def test_valid_no_required(self):
        tc = ParsedToolCall(name="test", arguments={"x": 1}, raw="")
        params = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        }
        assert validate_arguments(tc, params) == []

    def test_missing_required_arg(self):
        tc = ParsedToolCall(name="test", arguments={}, raw="")
        params = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        errors = validate_arguments(tc, params)
        assert len(errors) == 1
        assert "path" in errors[0]

    def test_unknown_argument(self):
        tc = ParsedToolCall(name="test", arguments={"unknown": "val"}, raw="")
        params = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        }
        errors = validate_arguments(tc, params)
        assert len(errors) == 1
        assert "unknown" in errors[0]

    def test_wrong_type(self):
        tc = ParsedToolCall(name="test", arguments={"count": "not_a_number"}, raw="")
        params = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        }
        errors = validate_arguments(tc, params)
        assert len(errors) == 1
        assert "integer" in errors[0]
        assert "str" in errors[0]

    def test_none_value_skips_type_check(self):
        tc = ParsedToolCall(name="test", arguments={"path": None}, raw="")
        params = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        errors = validate_arguments(tc, params)
        assert errors == []

    def test_boolean_type_validation(self):
        tc = ParsedToolCall(name="test", arguments={"flag": "yes"}, raw="")
        params = {
            "type": "object",
            "properties": {"flag": {"type": "boolean"}},
        }
        errors = validate_arguments(tc, params)
        assert len(errors) == 1

    def test_array_type_validation(self):
        tc = ParsedToolCall(name="test", arguments={"items": "not_a_list"}, raw="")
        params = {
            "type": "object",
            "properties": {"items": {"type": "array"}},
        }
        errors = validate_arguments(tc, params)
        assert len(errors) == 1

    def test_multiple_errors(self):
        tc = ParsedToolCall(
            name="test",
            arguments={"unknown_arg": 42, "count": "bad"},
            raw="",
        )
        params = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["path", "count"],
        }
        errors = validate_arguments(tc, params)
        # missing 'path', unknown 'unknown_arg', wrong type 'count'
        assert len(errors) >= 2

    def test_no_properties(self):
        tc = ParsedToolCall(name="test", arguments={"x": 1}, raw="")
        params = {"type": "object"}
        assert validate_arguments(tc, params) == []

    def test_number_type_accepts_int_and_float(self):
        tc = ParsedToolCall(name="test", arguments={"val": 42}, raw="")
        params = {
            "type": "object",
            "properties": {"val": {"type": "number"}},
        }
        assert validate_arguments(tc, params) == []

        tc2 = ParsedToolCall(name="test", arguments={"val": 3.14}, raw="")
        assert validate_arguments(tc2, params) == []


class TestStripToolCalls:
    def test_removes_tool_call_tags(self):
        content = 'Reasoning <tool_call>{"name":"read"}</tool_call> done'
        result = strip_tool_calls(content)
        assert result == "Reasoning done"

    def test_removes_multiple_tags(self):
        content = (
            '<tool_call>{"name":"read"}</tool_call>'
            ' middle '
            '<tool_call>{"name":"write"}</tool_call>'
            ' end'
        )
        result = strip_tool_calls(content)
        assert result == "middle end"

    def test_removes_json_code_blocks(self):
        content = 'Reasoning\n```json\n{"name":"read"}\n```\ndone'
        result = strip_tool_calls(content)
        assert result == "Reasoning\ndone"

    def test_removes_code_blocks_without_json_prefix(self):
        content = 'Text\n```\n{"name":"read"}\n```\nmore'
        result = strip_tool_calls(content)
        assert result == "Text\nmore"

    def test_no_tool_calls(self):
        content = "Just plain text with no calls"
        result = strip_tool_calls(content)
        assert result == "Just plain text with no calls"

    def test_empty_content(self):
        assert strip_tool_calls("") == ""

    def test_combined_tags_and_blocks(self):
        content = (
            '<tool_call>{"name":"a"}</tool_call>'
            ' text '
            '```json\n{"name":"b"}\n```'
            ' end'
        )
        result = strip_tool_calls(content)
        assert result == "text end"
