from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum

from chef_human.agent.prompts import PLANNER_SYSTEM_PROMPT
from chef_human.llm.backend import (
    CompletionRequest,
    LLMBackend,
    Message,
    Role,
)


class StepStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    failed = "failed"
    skipped = "skipped"


@dataclass
class PlanStep:
    index: int
    description: str
    status: StepStatus = StepStatus.pending

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "description": self.description,
            "status": self.status.value,
        }


@dataclass
class Plan:
    goal: str
    steps: list[PlanStep] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
        }


class Planner:
    """Generates and updates structured plans for the ReAct loop."""

    def __init__(self, llm_backend: LLMBackend) -> None:
        self._llm = llm_backend

    async def generate_plan(self, task: str, repo_context: str = "") -> Plan:
        messages = [
            Message(role=Role.system, content=PLANNER_SYSTEM_PROMPT),
        ]
        if repo_context:
            messages.append(
                Message(role=Role.system, content=f"## Project Context\n\n{repo_context}")
            )
        messages.append(Message(role=Role.user, content=f"Task: {task}"))

        response = await self._llm.complete(
            CompletionRequest(messages=messages, temperature=0.0, max_tokens=2048)
        )

        steps = self._parse_steps(response.message.content)
        return Plan(goal=task, steps=steps)

    async def update_plan(self, plan: Plan, failure_context: str) -> Plan:
        messages = [
            Message(
                role=Role.system,
                content=PLANNER_SYSTEM_PROMPT
                + "\n\nThe previous plan had a failure. Revise the remaining steps.",
            ),
            Message(
                role=Role.user,
                content=f"Original goal: {plan.goal}\n\n"
                f"Current progress:\n{self._format_plan(plan)}\n\n"
                f"Failure context:\n{failure_context}\n\n"
                f"Output a revised JSON array of remaining steps.",
            ),
        ]
        response = await self._llm.complete(
            CompletionRequest(messages=messages, temperature=0.0, max_tokens=2048)
        )
        steps = self._parse_steps(response.message.content)

        revised = Plan(goal=plan.goal)
        for s in plan.steps:
            if s.status == StepStatus.completed:
                revised.steps.append(s)
        for s in steps:
            if not any(
                existing.description == s.description
                for existing in revised.steps
            ):
                s.index = len(revised.steps) + 1
                revised.steps.append(s)
        return revised

    def _parse_steps(self, content: str) -> list[PlanStep]:
        array_match = re.search(r"\[.*\]", content, re.DOTALL)
        if array_match:
            try:
                data = json.loads(array_match.group(0))
            except json.JSONDecodeError:
                return [
                    PlanStep(index=i + 1, description=s)
                    for i, s in enumerate(content.strip().split("\n"))
                    if s.strip()
                ]
        else:
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                return [
                    PlanStep(index=i + 1, description=s)
                    for i, s in enumerate(content.strip().split("\n"))
                    if s.strip()
                ]

        if isinstance(data, list):
            if all(isinstance(item, str) for item in data):
                return [
                    PlanStep(index=i + 1, description=item)
                    for i, item in enumerate(data)
                ]
            elif all(isinstance(item, dict) for item in data):
                return [
                    PlanStep(
                        index=i + 1,
                        description=item.get("description", str(item)),
                    )
                    for i, item in enumerate(data)
                ]
        return [PlanStep(index=1, description=str(data))]

    @staticmethod
    def format_plan_for_prompt(plan: Plan) -> str:
        lines = ["## Plan", ""]
        for step in plan.steps:
            marker = {
                StepStatus.pending: "[ ]",
                StepStatus.in_progress: "[→]",
                StepStatus.completed: "[✓]",
                StepStatus.failed: "[✗]",
                StepStatus.skipped: "[-]",
            }[step.status]
            lines.append(f"{marker} Step {step.index}: {step.description}")
        return "\n".join(lines)

    @staticmethod
    def _format_plan(plan: Plan) -> str:
        return Planner.format_plan_for_prompt(plan)
