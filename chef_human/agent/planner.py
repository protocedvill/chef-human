from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from chef_human.agent.prompts import PLANNER_SYSTEM_PROMPT, build_verify_prompt
from chef_human.llm.backend import (
    CompletionRequest,
    CompletionResponse,
    LLMBackend,
    Message,
    Role,
)

logger = logging.getLogger(__name__)

_VERIFIER_REPAIR_PROMPT = """Your previous response did not follow the required format.

Previous invalid response:
{invalid_response}

Re-emit the same judgment using exactly two lines and nothing else:
VERDICT: COMPLETE, PARTIAL, or NOT_COMPLETE
REASON: <one short sentence>"""


class StepStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    failed = "failed"
    skipped = "skipped"


class StepVerdict(str, Enum):
    complete = "complete"
    partial = "partial"
    not_complete = "not_complete"


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

    def current_step(self) -> PlanStep | None:
        """The step that should be worked on right now: the first step (in
        order) that isn't yet completed. Returns None once every step is
        completed."""
        return next((s for s in self.steps if s.status == StepStatus.pending), None)

    def unresolved_steps(self) -> list[PlanStep]:
        """Steps that prevent a plan from being reported as complete."""
        return [s for s in self.steps if s.status != StepStatus.completed]

    def is_complete(self) -> bool:
        return not self.unresolved_steps()


class Planner:
    """Generates and updates structured plans for the ReAct loop."""

    _STEP_PREFIX_RE = re.compile(r"^step\s+\d+\s*[:.\-]\s*", re.IGNORECASE)
    _FILE_TARGET_RE = re.compile(r"[`'\"]?([\w\-./]+\.\w{1,10})[`'\"]?")
    _ENV_SETUP_RE = re.compile(
        r"\b(?:install|set up|setup|configure|create)\b.*\b(?:python|pip|dependency|dependencies|"
        r"virtualenv|venv|environment|requirements)\b",
        re.IGNORECASE,
    )
    _EDITOR_MECHANICS_RE = re.compile(
        r"\b(?:open\b.*\b(?:editor|nano|vim|vi|emacs)\b|save and close|close the file|"
        r"create an empty file|touch\s+[\w./-]+)\b",
        re.IGNORECASE,
    )
    _TASK_SETUP_RE = re.compile(
        r"\b(?:install|setup|set up|configure|bootstrap|venv|virtualenv|requirements|dependency|"
        r"dependencies|python)\b",
        re.IGNORECASE,
    )
    _CONDITIONAL_CREATE_RE = re.compile(
        r"^\s*(create|write|add|generate)\b.+\bif it does not exist\b",
        re.IGNORECASE,
    )
    _OPTIONAL_EXPLORE_RE = re.compile(
        r"^\s*(?:explore|inspect|read|check)\b.+\bif any\b",
        re.IGNORECASE,
    )
    _DO_NOT_MODIFY_TESTS_RE = re.compile(
        r"\bdo not modify\b.+\btests?\b|\bdo not modify the specification or tests\b",
        re.IGNORECASE,
    )
    _WRITE_TESTS_RE = re.compile(
        r"\b(?:write|add|create|update|modify)\b.+\btests?\b",
        re.IGNORECASE,
    )

    def __init__(self, llm_backend: LLMBackend) -> None:
        self._llm = llm_backend
        # Set by ReActLoop so planning/verification LLM calls (which happen
        # on a separate call path from the main reasoning loop) are counted
        # towards the same running token total and live UI display -- see
        # ReActLoop._record_usage.
        self.on_usage: Callable[[int, int], None] | None = None
        # Set by ReActLoop so the UI can show what Ollama is currently doing
        # even during planner-side calls (generate_plan/verify_step/
        # update_plan), which otherwise happen invisibly between the
        # main-loop's own on_reasoning_start/on_reasoning pair.
        self.on_llm_start: Callable[[str], None] | None = None
        self.on_llm_end: Callable[[], None] | None = None

    async def _complete(
        self, request: CompletionRequest, activity: str = "planning"
    ) -> CompletionResponse:
        if self.on_llm_start is not None:
            self.on_llm_start(activity)
        try:
            response = await self._llm.complete(request)
        finally:
            if self.on_llm_end is not None:
                self.on_llm_end()
        if response.usage and self.on_usage is not None:
            self.on_usage(
                response.usage.get("prompt_tokens", 0),
                response.usage.get("completion_tokens", 0),
            )
        return response

    async def generate_plan(
        self,
        task: str,
        repo_context: str = "",
        planning_facts: dict[str, bool] | None = None,
    ) -> Plan:
        messages = [
            Message(role=Role.system, content=PLANNER_SYSTEM_PROMPT),
        ]
        if repo_context:
            messages.append(
                Message(role=Role.system, content=f"## Project Context\n\n{repo_context}")
            )
        messages.append(Message(role=Role.user, content=f"Task: {task}"))

        response = await self._complete(
            CompletionRequest(messages=messages, temperature=0.0, max_tokens=2048),
            activity="planning",
        )

        steps = self._normalize_steps(
            task,
            self._parse_steps(response.message.content),
            planning_facts=planning_facts,
        )
        return Plan(goal=task, steps=steps)

    async def verify_step(
        self,
        plan: Plan,
        step: PlanStep,
        evidence: str,
        *,
        recent_history: str = "",
        finish_summary: str = "",
    ) -> tuple[StepVerdict, str]:
        """Check whether `step` was actually accomplished, based on the
        evidence (tool results or reasoning text) from the turn that
        appeared to finish it -- instead of assuming any non-failing turn
        means the current step is done."""
        prompt = build_verify_prompt(
            goal=plan.goal,
            step=step.description,
            evidence=evidence,
            recent_history=recent_history,
            finish_summary=finish_summary,
        )
        response = await self._complete(
            CompletionRequest(
                messages=[Message(role=Role.user, content=prompt)],
                temperature=0.0,
                max_tokens=100,
            ),
            activity="verifying step",
        )
        verdict, reason = self._parse_verdict(response.message.content)
        if reason != "Could not parse verifier response":
            return verdict, reason

        logger.debug(
            "Raw verifier response could not be parsed: %r",
            response.message.content,
        )
        repair_prompt = _VERIFIER_REPAIR_PROMPT.format(
            invalid_response=response.message.content.strip()
            or "(empty response)",
        )
        repaired = await self._complete(
            CompletionRequest(
                messages=[Message(role=Role.user, content=repair_prompt)],
                temperature=0.0,
                max_tokens=100,
            ),
            activity="repairing verifier response",
        )
        repaired_verdict, repaired_reason = self._parse_verdict(
            repaired.message.content
        )
        if repaired_reason == "Could not parse verifier response":
            logger.debug(
                "Verifier repair response could not be parsed: %r",
                repaired.message.content,
            )
        return repaired_verdict, repaired_reason

    @staticmethod
    def _parse_verdict(content: str) -> tuple[StepVerdict, str]:
        text = content.strip()
        reason_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE)
        reason = reason_match.group(1).strip() if reason_match else ""
        verdict_match = re.search(
            r"^\s*VERDICT:\s*(NOT_COMPLETE|NOT COMPLETE|PARTIAL|COMPLETE)\s*$",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        verdict_text = verdict_match.group(1).upper() if verdict_match else ""
        if not verdict_text:
            bare_lines = [line.strip() for line in text.splitlines() if line.strip()]
            if bare_lines:
                first_line = bare_lines[0].upper()
                if first_line in {"NOT_COMPLETE", "NOT COMPLETE", "PARTIAL", "COMPLETE"}:
                    verdict_text = first_line
                    if not reason and len(bare_lines) > 1:
                        reason = " ".join(bare_lines[1:]).strip()
        if verdict_text in {"NOT_COMPLETE", "NOT COMPLETE"}:
            return StepVerdict.not_complete, reason
        if verdict_text == "PARTIAL":
            return StepVerdict.partial, reason
        if verdict_text == "COMPLETE":
            return StepVerdict.complete, reason
        return StepVerdict.not_complete, reason or "Could not parse verifier response"

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
        response = await self._complete(
            CompletionRequest(messages=messages, temperature=0.0, max_tokens=2048),
            activity="replanning",
        )
        steps = self._normalize_steps(plan.goal, self._parse_steps(response.message.content))

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

    @classmethod
    def _clean_description(cls, description: str) -> str:
        return cls._STEP_PREFIX_RE.sub("", description.strip())

    @classmethod
    def _normalize_steps(
        cls,
        task: str,
        steps: list[PlanStep],
        *,
        planning_facts: dict[str, bool] | None = None,
    ) -> list[PlanStep]:
        """Remove low-value plan noise that traps smaller local models.

        This is deliberately conservative: if normalization would erase every
        step, the original cleaned plan is kept instead."""
        allow_setup_steps = bool(cls._TASK_SETUP_RE.search(task))
        protected_tests = bool(cls._DO_NOT_MODIFY_TESTS_RE.search(task))
        planning_facts = {
            path.casefold(): exists for path, exists in (planning_facts or {}).items()
        }
        normalized: list[PlanStep] = []
        seen_descriptions: set[str] = set()

        for step in steps:
            description = cls._clean_description(step.description)
            if not description:
                continue
            if not allow_setup_steps and cls._ENV_SETUP_RE.search(description):
                continue
            if cls._EDITOR_MECHANICS_RE.search(description):
                continue
            if protected_tests and cls._WRITE_TESTS_RE.search(description):
                continue

            description = cls._resolve_conditional_step(description, planning_facts)
            if not description:
                continue
            key = description.casefold()
            if key in seen_descriptions:
                continue
            seen_descriptions.add(key)
            normalized.append(
                PlanStep(index=len(normalized) + 1, description=description)
            )

        if normalized:
            return normalized

        return [
            PlanStep(index=i + 1, description=cls._clean_description(step.description))
            for i, step in enumerate(steps)
            if cls._clean_description(step.description)
        ]

    @classmethod
    def _resolve_conditional_step(
        cls,
        description: str,
        planning_facts: dict[str, bool],
    ) -> str:
        if not planning_facts:
            return description

        file_targets = [m.group(1) for m in cls._FILE_TARGET_RE.finditer(description)]
        if not file_targets:
            return description

        target = file_targets[0]
        exists = planning_facts.get(target.casefold())
        if exists is None:
            return description

        if cls._CONDITIONAL_CREATE_RE.search(description):
            if exists:
                return ""
            return re.sub(
                r"\s+if it does not exist\b",
                "",
                description,
                flags=re.IGNORECASE,
            ).strip()

        if cls._OPTIONAL_EXPLORE_RE.search(description) and not exists:
            return ""

        return description

    def _parse_steps(self, content: str) -> list[PlanStep]:
        array_match = re.search(r"\[.*\]", content, re.DOTALL)
        if array_match:
            try:
                data = json.loads(array_match.group(0))
            except json.JSONDecodeError:
                return [
                    PlanStep(index=i + 1, description=self._clean_description(s))
                    for i, s in enumerate(content.strip().split("\n"))
                    if s.strip()
                ]
        else:
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                return [
                    PlanStep(index=i + 1, description=self._clean_description(s))
                    for i, s in enumerate(content.strip().split("\n"))
                    if s.strip()
                ]

        if isinstance(data, list):
            if all(isinstance(item, str) for item in data):
                return [
                    PlanStep(index=i + 1, description=self._clean_description(item))
                    for i, item in enumerate(data)
                ]
            elif all(isinstance(item, dict) for item in data):
                return [
                    PlanStep(
                        index=i + 1,
                        description=self._clean_description(item.get("description", str(item))),
                    )
                    for i, item in enumerate(data)
                ]
        return [PlanStep(index=1, description=self._clean_description(str(data)))]

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
