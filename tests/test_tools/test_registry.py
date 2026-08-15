from __future__ import annotations

from typing import Any

from chef_human.agent.workspace import WorkspaceManager
from chef_human.tools import create_tool_registry

from chef_human.tools.registry import (
    TOOL_POLICIES,
    MutationScope,
    ReadRequirement,
    ToolResult,
    ToolRegistry,
)


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


class TestBuiltInToolPolicies:
    def test_every_registered_tool_is_classified(self, tmp_path):
        registry = create_tool_registry(WorkspaceManager(tmp_path))
        assert set(registry.list_tools()) <= set(TOOL_POLICIES)

    def test_every_builtin_mutator_has_one_read_strategy(self):
        mutators = {
            name: policy for name, policy in TOOL_POLICIES.items() if policy.mutates
        }
        assert set(mutators) == {
            "bash",
            "edit",
            "lint_fix",
            "patch",
            "redo",
            "refactor_symbol",
            "undo",
            "write",
        }
        assert all(
            policy.read_requirement != ReadRequirement.none
            for policy in mutators.values()
        )
        assert {
            name
            for name, policy in mutators.items()
            if policy.mutation_scope == MutationScope.path
        } == {"write", "edit", "patch"}
        assert all(
            mutators[name].read_requirement == ReadRequirement.explicit_path
            for name in ("write", "edit", "patch")
        )
