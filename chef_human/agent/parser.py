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


def strip_tool_calls(content: str) -> str:
    content = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL)
    content = re.sub(
        r"```(?:json)?\s*\n?.*?\n?```\n?", "", content, flags=re.DOTALL
    )
    content = re.sub(r" +", " ", content)
    content = re.sub(r"\n{2,}", "\n", content)
    return content.strip()
