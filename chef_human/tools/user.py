from __future__ import annotations

import logging
import sys
from typing import Any

from chef_human.tools.registry import ToolResult

logger = logging.getLogger(__name__)


class AskUserTool:
    name = "ask_user"
    description = "Ask the user a question when you need clarification or approval"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "Question to ask the user"},
        },
        "required": ["question"],
    }

    async def run(self, question: str) -> ToolResult:
        logger.info("User asked: %s", question)
        if not sys.stdin.isatty():
            return ToolResult(
                output="[no-tty] Cannot ask user in non-interactive mode. Continuing without answer."
            )
        print(f"\n[Agent asks]: {question}")
        print("[Type your response, or 'skip' to continue without answering]: ", end="", flush=True)
        try:
            response = sys.stdin.readline().strip()
        except (EOFError, KeyboardInterrupt):
            response = ""

        if not response or response.lower() == "skip":
            return ToolResult(output="User skipped the question")

        return ToolResult(output=response)


class FinishTool:
    name = "finish"
    description = "Signal that the task is complete"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "Brief summary of what was accomplished", "default": ""},
        },
    }

    async def run(self, summary: str = "") -> ToolResult:
        msg = "Task complete"
        if summary:
            msg += f": {summary}"
        return ToolResult(output=msg)
