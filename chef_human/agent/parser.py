from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class ToolCallParseError(Exception):
    def __init__(self, message: str, raw_content: str = "") -> None:
        self.raw_content = raw_content
        super().__init__(message)


@dataclass
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]
    raw: str


def parse_tool_calls(content: str) -> list[ParsedToolCall]:
    calls: list[ParsedToolCall] = []

    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
        raw = match.group(1).strip()
        try:
            data = json.loads(raw)
            parsed = _parse_single_call(data, raw)
            if parsed:
                calls.append(parsed)
        except json.JSONDecodeError:
            logger.warning("Failed to parse <tool_call> JSON: %s", raw[:100])
            continue

    for match in re.finditer(
        r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL
    ):
        raw = match.group(1).strip()
        try:
            data = json.loads(raw)
            if _is_tool_call_object(data) or _is_ollama_tool_call(data):
                parsed = _parse_single_call(data, raw)
                if parsed:
                    if not any(
                        p.name == parsed.name and p.arguments == parsed.arguments
                        for p in calls
                    ):
                        calls.append(parsed)
        except json.JSONDecodeError:
            continue

    if not calls:
        for raw in _extract_json_objects(content):
            try:
                data = json.loads(raw)
                if _is_tool_call_object(data) or _is_ollama_tool_call(data):
                    parsed = _parse_single_call(data, raw)
                    if parsed:
                        calls.append(parsed)
            except json.JSONDecodeError:
                continue

    return calls


def _is_tool_call_object(data: dict[str, Any]) -> bool:
    return "name" in data and "arguments" in data


def _is_ollama_tool_call(data: dict[str, Any]) -> bool:
    return isinstance(data.get("function"), dict)


def _extract_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            depth = 1
            j = i + 1
            while j < len(text) and depth > 0:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            if depth == 0:
                objects.append(text[i:j])
            i = j
        else:
            i += 1
    return objects


def _parse_single_call(
    data: dict[str, Any], raw: str
) -> ParsedToolCall | None:
    if "function" in data:
        func = data["function"]
        if isinstance(func, dict):
            name = func.get("name", "")
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            if name:
                return ParsedToolCall(name=name, arguments=args, raw=raw)

    name = data.get("name", "")
    args = data.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"raw": args}
    if name:
        return ParsedToolCall(name=name, arguments=args, raw=raw)

    return None


def validate_arguments(
    tool_call: ParsedToolCall,
    parameters: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    props = parameters.get("properties")
    required = parameters.get("required", [])

    for req in required:
        if req not in tool_call.arguments:
            errors.append(f"Missing required argument: '{req}'")

    if props is None:
        return errors

    for key, value in tool_call.arguments.items():
        if key not in props:
            errors.append(f"Unknown argument: '{key}'")
            continue
        schema = props[key]
        expected_type = schema.get("type")
        if expected_type and value is not None:
            type_map: dict[str, type | tuple[type, ...]] = {
                "string": str,
                "integer": int,
                "number": (int, float),
                "boolean": bool,
                "array": list,
                "object": dict,
            }
            py_type = type_map.get(expected_type)
            if py_type and not isinstance(value, py_type):
                errors.append(
                    f"Argument '{key}': expected {expected_type}, got {type(value).__name__}"
                )

    return errors


_SCRATCH_PATTERN = re.compile(
    r"^## Scratchpad:\s*(.*)$",
    re.MULTILINE,
)


def extract_scratchpad(content: str) -> str | None:
    """Extract the last scratchpad update from model output.

    The model writes:
        ## Scratchpad: <content>

    Only the last occurrence is used; previous ones are overwritten.
    Returns None if no scratchpad block is found.
    """
    matches = list(_SCRATCH_PATTERN.finditer(content))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def strip_scratchpad(content: str) -> str:
    """Remove ## Scratchpad: lines from content."""
    return _SCRATCH_PATTERN.sub("", content).strip()


def looks_like_tool_call(content: str) -> bool:
    """Check if content appears to attempt a tool call, even if unparseable."""
    if "<tool_call" in content:
        return True
    if re.search(r'"name"\s*:', content) and re.search(r'"arguments"\s*:', content):
        return True
    if re.search(r'```(?:json)?\s*\n?\s*\{', content):
        return True
    return False


def format_parse_error(content: str, detail: str = "") -> str:
    """Produce a human-readable parse error message for the LLM."""
    snippet = content.strip()[:200]
    msg = "Error: Failed to parse tool call from your output.\n"
    if detail:
        msg += f"Reason: {detail}\n"
    msg += f"Your output was:\n{snippet}"
    if len(content) > 200:
        msg += "..."
    msg += (
        "\n\nFix the format and try again. "
        "Use <tool_call>{{\"name\": \"tool_name\", \"arguments\": {{...}}}}</tool_call>."
    )
    return msg


def strip_tool_calls(content: str) -> str:
    content = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL)
    content = re.sub(
        r"```(?:json)?\s*\n?.*?\n?```\n?", "", content, flags=re.DOTALL
    )
    content = re.sub(r" +", " ", content)
    content = re.sub(r"\n{2,}", "\n", content)
    return content.strip()
