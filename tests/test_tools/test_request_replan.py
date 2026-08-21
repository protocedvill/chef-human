from __future__ import annotations

from pathlib import Path

from chef_human.agent.workspace import WorkspaceManager
from chef_human.tools import create_tool_registry
from chef_human.tools.registry import TOOL_POLICIES, ToolResult
from chef_human.tools.replan import RequestReplanTool


class TestRequestReplanToolContract:
    async def test_ack_is_structured_control_result(self):
        tool = RequestReplanTool()
        result = await tool.run(
            reason="protocol is in firmware/, not host/",
            evidence_summary="firmware/usb.c:42 defines the real request codes",
        )
        assert isinstance(result, ToolResult)
        assert result.success is True
        # Machine-distinct: the loop branches on this flag, not on prose.
        assert result.control is True
        payload: dict = result.control_payload or {}
        assert isinstance(result.control_payload, dict)
        assert payload["action"] == "replan_requested"

    async def test_ack_carries_the_model_arguments(self):
        tool = RequestReplanTool()
        result = await tool.run(
            reason="the step's premise is wrong",
            evidence_summary="src/app.py:10 shows a different API",
        )
        payload = result.control_payload
        assert payload is not None
        assert payload["reason"] == "the step's premise is wrong"
        assert payload["evidence_summary"] == "src/app.py:10 shows a different API"

    async def test_ack_output_is_readable_and_mentions_reason(self):
        tool = RequestReplanTool()
        result = await tool.run(
            reason="file moved",
            evidence_summary="ls shows the module now lives in core/",
        )
        assert "file moved" in result.output
        assert result.error is None


class TestRequestReplanToolRegistration:
    def test_registered_in_default_registry(self, tmp_path: Path):
        registry = create_tool_registry(WorkspaceManager(tmp_path))
        assert "request_replan" in registry.list_tools()
        assert isinstance(registry.get("request_replan"), RequestReplanTool)

    def test_classified_as_non_mutating_control_tool(self):
        # A control-plane action: it must have an explicit policy entry and
        # must NOT be treated as an ordinary mutating repo tool.
        assert "request_replan" in TOOL_POLICIES
        assert TOOL_POLICIES["request_replan"].mutates is False

    def test_argument_contract(self, tmp_path: Path):
        registry = create_tool_registry(WorkspaceManager(tmp_path))
        tool = registry.get("request_replan")
        assert tool is not None
        params = tool.parameters
        assert params["type"] == "object"
        assert params["properties"]["reason"]["type"] == "string"
        assert params["properties"]["evidence_summary"]["type"] == "string"
        assert set(params["required"]) == {"reason", "evidence_summary"}


class TestRequestReplanToolAsyncDefaults:
    async def test_missing_evidence_summary_defaults_to_empty(self):
        tool = RequestReplanTool()
        result = await tool.run(reason="premise wrong")
        payload = result.control_payload
        assert result.control is True
        assert payload is not None
        assert payload["evidence_summary"] == ""
