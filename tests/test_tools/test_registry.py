from __future__ import annotations

from typing import Any


from chef_human.tools.registry import ToolResult, ToolRegistry


class FakeTool:
    name = "test_tool"
    description = "A test tool"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
    }

    async def run(self, **kwargs: Any) -> ToolResult:
        return ToolResult(output=f"ran with {kwargs}")


class TestToolResult:
    def test_defaults(self):
        r = ToolResult()
        assert r.success is True
        assert r.output == ""
        assert r.error is None

    def test_custom_values(self):
        r = ToolResult(success=False, output="oops", error="something broke")
        assert r.success is False
        assert r.output == "oops"
        assert r.error == "something broke"


class TestToolRegistry:
    def test_starts_empty(self):
        reg = ToolRegistry()
        assert reg.list_tools() == []

    def test_register_and_get(self):
        reg = ToolRegistry()
        tool = FakeTool()
        reg.register(tool)
        assert reg.get("test_tool") is tool

    def test_get_returns_none_for_unknown(self):
        reg = ToolRegistry()
        assert reg.get("nonexistent") is None

    def test_list_tools_sorted(self):
        reg = ToolRegistry()
        tool_b = FakeTool()
        tool_b.name = "b_tool"
        tool_a = FakeTool()
        tool_a.name = "a_tool"
        reg.register(tool_b)
        reg.register(tool_a)
        assert reg.list_tools() == ["a_tool", "b_tool"]

    def test_get_definitions_returns_tool_definitions(self):
        reg = ToolRegistry()
        tool = FakeTool()
        tool.name = "adder"
        tool.description = "Adds numbers"
        tool.parameters = {"type": "object", "properties": {"a": {"type": "integer"}}}
        reg.register(tool)
        from chef_human.llm.backend import ToolDefinition
        defs = reg.get_definitions()
        assert len(defs) == 1
        assert isinstance(defs[0], ToolDefinition)
        assert defs[0].name == "adder"
        assert defs[0].description == "Adds numbers"
        assert defs[0].parameters["properties"]["a"]["type"] == "integer"

    def test_register_replaces_existing(self):
        reg = ToolRegistry()
        t1 = FakeTool()
        t2 = FakeTool()
        reg.register(t1)
        reg.register(t2)
        assert reg.get("test_tool") is t2

    def test_tool_protocol_structural(self):
        tool = FakeTool()
        assert isinstance(tool, FakeTool)
        assert hasattr(tool, "name")
        assert hasattr(tool, "run")
