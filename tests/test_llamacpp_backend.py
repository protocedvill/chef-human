import pytest

from chef_human.llm.backend import (
    CompletionRequest,
    Message,
    Role,
    ToolDefinition,
)
from chef_human.llm.llamacpp_backend import LlamaCppBackend


class TestFormatChatML:
    def test_basic_user_message(self):
        req = CompletionRequest(
            messages=[Message(role=Role.user, content="hello")]
        )
        result = LlamaCppBackend.format_chatml(req)
        expected = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        assert result == expected

    def test_system_message(self):
        req = CompletionRequest(
            messages=[Message(role=Role.system, content="you are a bot")]
        )
        result = LlamaCppBackend.format_chatml(req)
        expected = "<|im_start|>system\nyou are a bot<|im_end|>\n<|im_start|>assistant\n"
        assert result == expected

    def test_assistant_tool_calls_in_message(self):
        req = CompletionRequest(
            messages=[
                Message(
                    role=Role.assistant,
                    content="",
                    tool_calls=[{"function": {"name": "foo", "arguments": "{}"}}],
                )
            ]
        )
        result = LlamaCppBackend.format_chatml(req)
        assert "<|tool_call|>" in result
        assert "foo" in result

    def test_tool_result_message(self):
        req = CompletionRequest(
            messages=[
                Message(role=Role.tool, content='{"result": 42}', tool_call_id="call_1")
            ]
        )
        result = LlamaCppBackend.format_chatml(req)
        expected = "<|im_start|>tool\n{\"result\": 42}<|im_end|>\n<|im_start|>assistant\n"
        assert result == expected

    def test_tools_in_system_prompt(self):
        req = CompletionRequest(
            messages=[Message(role=Role.user, content="do it")],
            tools=[
                ToolDefinition(
                    name="read",
                    description="read a file",
                    parameters={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                )
            ],
        )
        result = LlamaCppBackend.format_chatml(req)
        assert "Available tools:" in result
        assert "read" in result
        assert "read a file" in result

    def test_multi_turn(self):
        req = CompletionRequest(
            messages=[
                Message(role=Role.user, content="hi"),
                Message(role=Role.assistant, content="hello!"),
                Message(role=Role.user, content="how are you?"),
            ]
        )
        result = LlamaCppBackend.format_chatml(req)
        assert result.count("<|im_start|>") == 4
        assert result.count("<|im_end|>") == 3


class TestParseToolCalls:
    def test_no_tool_calls(self):
        assert LlamaCppBackend.parse_tool_calls("hello world") is None

    def test_single_tool_call(self):
        text = '<tool_call>{"name": "foo", "arguments": {"x": 1}}</tool_call>'
        result = LlamaCppBackend.parse_tool_calls(text)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "foo"

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>{"name": "a"}</tool_call>'
            'text between'
            '<tool_call>{"name": "b"}</tool_call>'
        )
        result = LlamaCppBackend.parse_tool_calls(text)
        assert result is not None
        assert len(result) == 2

    def test_invalid_json_in_tool_call(self):
        text = '<tool_call>not json</tool_call>'
        result = LlamaCppBackend.parse_tool_calls(text)
        assert result is None

    def test_mixed_content(self):
        text = (
            "I'll look up the file.\n"
            '<tool_call>{"name": "read", "arguments": {"path": "x.py"}}</tool_call>'
            "\nNow I have the contents."
        )
        result = LlamaCppBackend.parse_tool_calls(text)
        assert result is not None
        assert result[0]["name"] == "read"


class TestStripToolCalls:
    def test_strip_removes_tags(self):
        text = 'hello <tool_call>{"x": 1}</tool_call> world'
        result = LlamaCppBackend.strip_tool_calls(text)
        assert result == "hello  world"

    def test_strip_no_tags(self):
        text = "hello world"
        result = LlamaCppBackend.strip_tool_calls(text)
        assert result == "hello world"

    def test_strip_multiple_tags(self):
        text = 'a<tool_call>{"x":1}</tool_call>b<tool_call>{"y":2}</tool_call>c'
        result = LlamaCppBackend.strip_tool_calls(text)
        assert result == "abc"

class TestBackendInit:
    def test_raises_filenotefound_for_missing_model(self):
        with pytest.raises(FileNotFoundError, match="Model not found"):
            LlamaCppBackend(model_path="/nonexistent/model.gguf")

    def test_raises_import_error_without_llamacpp(self, tmp_path):
        dummy = tmp_path / "dummy.gguf"
        dummy.write_text("not a real model")
        with pytest.raises(ImportError, match="llama-cpp-python"):
            LlamaCppBackend(model_path=str(dummy), n_ctx=64)
