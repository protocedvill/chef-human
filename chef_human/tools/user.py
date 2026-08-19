from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from chef_human.tools.registry import ToolResult

logger = logging.getLogger(__name__)


class AskUserTool:
    name = "ask_user"
    description = (
        "Ask the user only for a genuine unresolved design decision or ambiguous "
        "requirement; never ask them to perform or confirm the current plan step"
    )
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
                output=(
                    "[no-tty] Cannot ask user in non-interactive mode. No answer "
                    "is coming; do not ask again. Proceed using the current plan "
                    "and your best judgment."
                )
            )
        print(f"\n[Agent asks]: {question}")
        print("[Type your response, or 'skip' to continue without answering]: ", end="", flush=True)
        try:
            # Run the blocking read in a thread so it doesn't stall the
            # event loop -- other UI surfaces (e.g. the Textual TUI) run
            # concurrent asyncio tasks that would otherwise freeze while
            # waiting on stdin here.
            response = (await asyncio.to_thread(sys.stdin.readline)).strip()
        except (EOFError, KeyboardInterrupt):
            response = ""

        if not response or response.lower() == "skip":
            return ToolResult(
                output=(
                    "User skipped the question -- no answer is coming. Do not "
                    "ask this or a similar question again. Proceed using your "
                    "own best judgment based on the current plan step and "
                    "available context."
                )
            )

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
