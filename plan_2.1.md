# Phase 2.1: ReAct Loop — Reasoning, Planning & Tool Execution

**Goal**: Implement a structured ReAct (Reasoning + Acting) agent loop that takes a user task, generates a high-level plan, and iterates through reasoning → tool calling → observation cycles until completion. Integrate self-correction, structured output parsing, and a Rich-based debug TUI for development.

**Prerequisites**: Phases 1.1 (LLM backends), 1.2 (context manager), and 1.3 (tool layer) complete.

---

## Task List

- [x] **2.1.1** Planner module (`Planner`, `Plan`, `PlanStep` dataclasses)
- [ ] **2.1.2** Structured output parser (tool call extraction, JSON validation, fallback)
- [ ] **2.1.3** Core ReAct loop (`ReActLoop` orchestrator)
- [ ] **2.1.4** Self-correction & retry logic
- [ ] **2.1.5** Destructive operation approval gate
- [ ] **2.1.6** Rich-based debug TUI (test GUI)
- [ ] **2.1.7** CLI entry point (`main.py`, `click`)
- [ ] **2.1.8** Integration tests & factory update
- [ ] **2.1.9** System prompt design & agent message templates

---

## Architecture Overview

```
User Task
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                    CLI (main.py)                      │
│  ┌──────────────────────────────────────────────┐   │
│  │              Debug TUI (rich)                 │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │   │
│  │  │  Plan    │ │ Reason   │ │ Tool Calls   │ │   │
│  │  │  Panel   │ │  Panel   │ │  & Results   │ │   │
│  │  └──────────┘ └──────────┘ └──────────────┘ │   │
│  └──────────────────────────────────────────────┘   │
│                      │                               │
│                      ▼                               │
│  ┌──────────────────────────────────────────────┐   │
│  │              ReActLoop                         │   │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────┐ │   │
│  │  │ Planner  │  │ Parser   │  │ Retry/     │ │   │
│  │  │          │  │          │  │ Correction │ │   │
│  │  └────┬─────┘  └────┬─────┘  └─────┬──────┘ │   │
│  │       │              │              │         │   │
│  │       ▼              ▼              ▼         │   │
│  │  ┌──────────────────────────────────────┐    │   │
│  │  │    LLM Backend + ToolRegistry +      │    │   │
│  │  │    ContextAssembler                   │    │   │
│  │  └──────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Loop Flow

```
┌─────────────────────────────────────────────────────────┐
│ 1. User submits task                                     │
│    ↓                                                     │
│ 2. Planner generates structured plan (separate LLM call) │
│    ↓                                                     │
│ 3. For each step in plan (or until finish):              │
│    ┌───────────────────────────────────────────────┐    │
│    │ a. Assemble context (system + tools + plan +   │    │
│    │    conversation + file context + repo map)     │    │
│    │ b. Send to LLM → model reasons about next step │    │
│    │ c. Parse tool calls from model response        │    │
│    │ d. Validate arguments against JSON schema      │    │
│    │ e. If destructive: ask user approval           │    │
│    │ f. Execute tool, record result in conversation │    │
│    │ g. If error: retry (up to N times), else       │    │
│    │    re-plan or escalate to user                 │    │
│    │ h. If finish tool called → exit loop           │    │
│    │ i. If max steps exceeded → stop with warning   │    │
│    └───────────────────────────────────────────────┘    │
│    ↑                                                    │
│    └─────── loop until finish or max steps ────────────┘│
│                                                         │
│ 4. Return final result summary                          │
└─────────────────────────────────────────────────────────┘
```

---

## Task 2.1.1: Planner Module

**File:** `chef_human/agent/planner.py`

A dedicated module that generates a structured, multi-step plan from a user task using a separate LLM call. The plan is injected into the system prompt of the main loop, guiding execution.

### Data Structures

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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


@dataclass
class Plan:
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
```

### Planner Implementation

```python
PLANNER_SYSTEM_PROMPT = """You are a planning assistant. Given a software engineering task,
break it down into a series of concrete, actionable steps. Each step should be:
1. Specific enough that an AI agent can execute it with available tools
2. Ordered logically (dependencies before dependents)
3. Self-contained (each step has a clear completion criterion)

Output a JSON array of step objects with "description" fields.
Do NOT include any other text before or after the JSON array.
Output ONLY the JSON array, nothing else."""


class Planner:
    """Generates and updates structured plans for the ReAct loop."""

    def __init__(self, llm_backend: LLMBackend) -> None:
        self._llm = llm_backend

    async def generate_plan(self, task: str, repo_context: str = "") -> Plan:
        """Generate a plan from a user task description."""
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
        """Re-plan when a step fails."""
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

        # Merge: keep completed steps, replace failed+future with revised steps
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
        """Parse JSON array of steps from LLM response."""
        import json
        import re

        # Try to extract JSON array from the response
        array_match = re.search(r"\[.*\]", content, re.DOTALL)
        if array_match:
            try:
                data = json.loads(array_match.group(0))
            except json.JSONDecodeError:
                return [PlanStep(index=1, description=f"Step {i+1}: {s}")
                        for i, s in enumerate(content.strip().split("\n")) if s.strip()]
        else:
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                return [PlanStep(index=1, description=f"Step {i+1}: {s}")
                        for i, s in enumerate(content.strip().split("\n")) if s.strip()]

        if isinstance(data, list):
            if all(isinstance(item, str) for item in data):
                return [PlanStep(index=i + 1, description=item) for i, item in enumerate(data)]
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
        """Format plan as text for inclusion in the ReAct loop's system prompt."""
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
```

### Acceptance Criteria

- `generate_plan()` returns a `Plan` with correctly parsed steps from LLM response
- Handles JSON array, line-by-line, and malformed responses gracefully (fallback parsing)
- `update_plan()` merges completed steps with revised remaining steps
- `format_plan_for_prompt()` produces clean markdown-style plan text
- `_parse_steps()` handles: `["a","b"]`, `[{"description":"a"},{"description":"b"}]`, plain text lines, empty response

---

## Task 2.1.2: Structured Output Parser

**File:** `chef_human/agent/parser.py`

Extracts and validates tool calls from unstructured LLM output. Supports multiple formats and provides detailed error messages for self-correction.

```python
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class ToolCallParseError(Exception):
    """Raised when a tool call cannot be parsed from model output."""
    def __init__(self, message: str, raw_content: str = "") -> None:
        self.raw_content = raw_content
        super().__init__(message)


@dataclass
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]
    raw: str  # original raw text for reference


def parse_tool_calls(content: str) -> list[ParsedToolCall]:
    """Parse all tool calls from model response content.

    Supported formats (tried in order):
    1. <tool_call>{...}</tool_call> XML tags
    2. ```json { "name": "...", "arguments": {...} } ``` code blocks
    3. Plain JSON objects with "name" and "arguments" keys
    4. Ollama-style: {"function": {"name": "...", "arguments": {...}}}
    """
    calls: list[ParsedToolCall] = []

    # Format 1: <tool_call> tags
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

    # Format 2: JSON code blocks with tool call structure
    for match in re.finditer(
        r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL
    ):
        raw = match.group(1).strip()
        try:
            data = json.loads(raw)
            if _is_tool_call_object(data) or _is_ollama_tool_call(data):
                parsed = _parse_single_call(data, raw)
                if parsed:
                    # Avoid duplicates with Format 1
                    if not any(p.name == parsed.name and p.arguments == parsed.arguments for p in calls):
                        calls.append(parsed)
        except json.JSONDecodeError:
            continue

    # Format 3: Any JSON object in the content with tool structure
    if not calls:
        for match in re.finditer(r"\{[^{}]*\}", content):
            raw = match.group(0)
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


def _parse_single_call(data: dict[str, Any], raw: str) -> ParsedToolCall | None:
    """Parse a single tool call from various JSON formats."""
    # Ollama format: {"function": {"name": "...", "arguments": {...}}}
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

    # Direct format: {"name": "...", "arguments": {...}}
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
    """Validate parsed arguments against JSON Schema parameters.

    Returns list of error messages (empty if valid).
    """
    errors: list[str] = []
    props = parameters.get("properties", {})
    required = parameters.get("required", [])

    for req in required:
        if req not in tool_call.arguments:
            errors.append(f"Missing required argument: '{req}'")

    for key, value in tool_call.arguments.items():
        if key not in props:
            errors.append(f"Unknown argument: '{key}'")
            continue
        schema = props[key]
        expected_type = schema.get("type")
        if expected_type and value is not None:
            type_map = {
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
    """Remove tool call markup from content, leaving only natural language."""
    content = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL)
    content = re.sub(
        r"```(?:json)?\s*\n?.*?\n?```", "", content, flags=re.DOTALL
    )
    return content.strip()
```

### Acceptance Criteria

- `parse_tool_calls()` extracts calls from `<tool_call>` tags, JSON code blocks, and bare JSON
- `validate_arguments()` checks required params, type mismatches, unknown keys
- Handles malformed JSON gracefully (log warning, continue)
- `strip_tool_calls()` removes all tool markup, leaving clean reasoning text
- Correctly parses both direct `{"name": "read", "arguments": ...}` and Ollama `{"function": {"name": ...}}` formats

---

## Task 2.1.3: Core ReAct Loop

**File:** `chef_human/agent/react_loop.py`

The central orchestrator that ties together planning, context, LLM, tools, and user interaction.

### Configuration

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReActConfig:
    max_steps: int = 25
    max_retries_per_step: int = 3
    temperature: float = 0.0
    max_tokens_per_response: int = 4096
    require_approval_for_destructive: bool = True
    stream: bool = False  # stream model output to UI
```

### ReActLoop Implementation

```python
class ReActLoop:
    """The main agent loop: plan → reason → act → observe → repeat."""

    def __init__(
        self,
        llm_backend: LLMBackend,
        tool_registry: ToolRegistry,
        context_assembler: ContextAssembler,
        planner: Planner,
        config: ReActConfig | None = None,
        ui: ReActUI | None = None,  # optional UI for TUI mode
    ) -> None:
        self._llm = llm_backend
        self._tools = tool_registry
        self._context = context_assembler
        self._planner = planner
        self._config = config or ReActConfig()
        self._ui = ui or NoopUI()

    async def run(self, task: str) -> AgentResult:
        """Execute a task from start to finish."""
        self._ui.on_start(task)
        steps_taken = 0
        retries = 0
        plan = await self._plan_task(task)

        self._context.conversation.add_message(
            Message(role=Role.user, content=task)
        )

        while steps_taken < self._config.max_steps:
            # 1. Build system prompt with plan + tool definitions
            system_prompt = build_agent_system_prompt(
                plan=plan,
                tool_defs=self._tools.get_definitions(),
            )

            # 2. Assemble full context
            messages = self._context.assemble(
                system_prompt=system_prompt,
                tool_definitions="",  # tools already in system prompt
            )

            # 3. Get LLM response
            self._ui.on_reasoning_start()
            response = await self._llm.complete(
                CompletionRequest(
                    messages=messages,
                    tools=self._tools.get_definitions(),
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens_per_response,
                )
            )
            self._ui.on_reasoning(response.message.content)

            # 4. Parse tool calls
            tool_calls = parse_tool_calls(response.message.content)
            non_tool_reasoning = strip_tool_calls(response.message.content)

            # 5. Record assistant response in conversation
            assistant_msg = Message(
                role=Role.assistant,
                content=non_tool_reasoning,
                tool_calls=[
                    {"function": {"name": tc.name, "arguments": tc.arguments}}
                    for tc in tool_calls
                ],
            )
            self._context.conversation.add_message(assistant_msg)

            # 6. Handle no tool calls (model just responded with text)
            if not tool_calls:
                steps_taken += 1
                retries = 0
                # Check if finish was implied in text
                if self._detect_finish(non_tool_reasoning):
                    return self._make_result(
                        plan=plan,
                        steps_taken=steps_taken,
                        message=non_tool_reasoning,
                    )
                continue

            # 7. Execute each tool call
            all_success = True
            tool_results: list[str] = []

            for tc in tool_calls:
                self._ui.on_tool_call(tc)

                # Find tool in registry
                tool = self._tools.get(tc.name)
                if tool is None:
                    result = self._make_tool_error(
                        f"Unknown tool: '{tc.name}'. Available: {', '.join(self._tools.list_tools())}"
                    )
                    self._ui.on_tool_result(tc.name, result)
                    tool_results.append(result)
                    all_success = False
                    continue

                # Validate arguments
                errors = validate_arguments(tc.arguments, tool.parameters)
                if errors:
                    error_msg = f"Invalid arguments for {tc.name}: {'; '.join(errors)}"
                    result = self._make_tool_error(error_msg)
                    self._ui.on_tool_result(tc.name, result)
                    tool_results.append(result)
                    all_success = False
                    continue

                # Check approval for destructive operations (if enabled)
                if self._config.require_approval_for_destructive and tc.name == "bash":
                    if self._is_destructive_command(tc.arguments.get("command", "")):
                        approved = await self._request_approval(tc)
                        if not approved:
                            result = self._make_tool_error(
                                "Command rejected by user: destructive operation requires approval"
                            )
                            self._ui.on_tool_result(tc.name, result)
                            tool_results.append(result)
                            # Don't count as error, just skip
                            continue

                # Execute the tool
                try:
                    tool_result = await tool.run(**tc.arguments)
                except Exception as exc:
                    result = self._make_tool_error(f"Execution error: {exc}")
                    self._ui.on_tool_result(tc.name, result)
                    tool_results.append(result)
                    all_success = False
                    continue

                # Check for finish signal
                if tc.name == "finish":
                    self._ui.on_tool_result(tc.name, tool_result)
                    return self._make_result(
                        plan=plan,
                        steps_taken=steps_taken,
                        message=tool_result.output,
                    )

                result = tool_result.output if tool_result.success else tool_result.error or ""
                if not tool_result.success:
                    result = f"Error: {tool_result.error}\nOutput: {tool_result.output}"
                    all_success = False

                self._ui.on_tool_result(tc.name, result)
                tool_results.append(result)

            # 8. Record tool results in conversation
            for result_text in tool_results:
                self._context.conversation.add_message(
                    Message(role=Role.tool, content=result_text)
                )

            steps_taken += 1

            # 9. Handle failures — retry or re-plan
            if not all_success:
                retries += 1
                if retries >= self._config.max_retries_per_step:
                    self._ui.on_replan()
                    plan = await self._planner.update_plan(
                        plan,
                        failure_context="\n".join(tool_results),
                    )
                    retries = 0
            else:
                retries = 0
                # Mark current plan step as completed
                self._mark_step_completed(plan, tool_results)

        # Max steps exceeded
        return self._make_result(
            plan=plan,
            steps_taken=steps_taken,
            message="Max steps exceeded. The task may be incomplete.",
            success=False,
        )

    async def _plan_task(self, task: str) -> Plan:
        """Generate a plan and notify UI."""
        self._ui.on_planning_start()
        repo_context = self._get_repo_context()
        plan = await self._planner.generate_plan(task, repo_context=repo_context)
        self._ui.on_plan(plan)
        return plan

    def _get_repo_context(self) -> str:
        """Get a brief repository context string for the planner."""
        try:
            tree = self._context._repo_map.generate_tree()
            return tree[:1000]  # keep it short for the planner call
        except Exception:
            return ""

    def _mark_step_completed(self, plan: Plan, results: list[str]) -> None:
        """Mark the next pending step as in_progress and then completed."""
        for step in plan.steps:
            if step.status == StepStatus.pending:
                step.status = StepStatus.in_progress
                step.status = StepStatus.completed
                break

    def _detect_finish(self, content: str) -> bool:
        """Heuristic detection of finish in free-text response."""
        triggers = [
            "task is complete",
            "i have finished",
            "all done",
            "finished the task",
        ]
        return any(t in content.lower() for t in triggers)

    def _is_destructive_command(self, command: str) -> bool:
        """Check if a bash command is destructive (mirrors BashTool logic)."""
        from chef_human.tools.shell import DESTRUCTIVE_PREFIXES
        stripped = command.strip()
        for prefix in DESTRUCTIVE_PREFIXES:
            if stripped.startswith(prefix):
                return True
        return False

    async def _request_approval(self, tool_call: ParsedToolCall) -> bool:
        """Ask user to approve a destructive operation."""
        cmd = tool_call.arguments.get("command", "")
        print(f"\n[!] Destructive operation requested: {cmd}")
        response = input("Approve? (y/N): ").strip().lower()
        return response in ("y", "yes")

    @staticmethod
    def _make_tool_error(message: str) -> str:
        return f"Error: {message}"

    @staticmethod
    def _make_result(
        plan: Plan,
        steps_taken: int,
        message: str,
        success: bool = True,
    ) -> AgentResult:
        return AgentResult(
            plan=plan,
            steps_taken=steps_taken,
            message=message,
            success=success,
        )


@dataclass
class AgentResult:
    plan: Plan
    steps_taken: int
    message: str
    success: bool = True
```

### System Prompt Builder

Included in `react_loop.py` (or `chef_human/agent/prompts.py`):

```python
AGENT_SYSTEM_PROMPT = """You are chef-human, an AI software engineering assistant.
You have access to tools. Use them to accomplish the user's task.

Guidelines:
1. Follow the plan step by step. Complete each step before moving to the next.
2. Think step by step before calling any tools.
3. When you call a tool, output it as:
   <tool_call>{ "name": "tool_name", "arguments": { ... } }</tool_call>
4. After each tool result, analyze the result and decide what to do next.
5. If a tool fails, try to fix the error and retry. If you can't fix it, explain the issue.
6. When the entire task is done, call the `finish` tool with a summary.
7. Do NOT call finish until all plan steps are complete."""


def build_agent_system_prompt(
    plan: Plan | None = None,
    tool_defs: list[ToolDefinition] | None = None,
) -> str:
    prompt = AGENT_SYSTEM_PROMPT

    if tool_defs:
        tool_text = format_tool_definitions(tool_defs)
        prompt += f"\n\n## Available Tools\n\n{tool_text}"

    if plan:
        prompt += f"\n\n{Planner.format_plan_for_prompt(plan)}"

    return prompt
```

### Acceptance Criteria

- `run()` completes a full task end-to-end when given a working LLM backend
- Generates a plan before starting execution
- For each step: reasons, calls tools, observes results
- Detects `finish` tool call and terminates cleanly
- Handles unknown tools gracefully (error message in conversation, not crash)
- Tracks step count and enforces `max_steps`
- Record all messages (assistant reasoning, tool calls, tool results) in context

---

## Task 2.1.4: Self-Correction & Retry Logic

Integrated into `ReActLoop` (Task 2.1.3), but designed as a distinct subsystem.

### Behavior Matrix

| Scenario | Count as? | Action |
|----------|-----------|--------|
| Tool call with invalid JSON | Retryable error | `max_retries_per_step` — retry, add error message to conversation |
| Tool call with missing args | Retryable error | Same as above — model sees validation error, fixes call |
| Tool execution failure (bad path, timeout) | Retryable error | Same — model sees error output, retries with corrected input |
| Tool call with unknown tool name | Retryable error | Model sees available tools list, fixes call |
| Consecutive failures in same step | Non-retryable | After `max_retries_per_step` (default 3), trigger re-plan |
| Re-plan also fails | Escalate | Loop terminates with partial result, error message shown to user |

### Retry Logic (pseudocode integrated in loop)

```python
# Inside the per-step loop:
retries = 0
all_success = True

for tc in tool_calls:
    result = await execute_tool(tc)
    if not result.success:
        all_success = False
        record_error_in_conversation(result.error)

# After executing all tool calls in this turn:
if not all_success:
    retries += 1
    if retries >= max_retries_per_step:
        # Re-plan: update the plan based on what went wrong
        plan = await planner.update_plan(plan, failure_context=error_summary)
        retries = 0  # reset for the new plan
else:
    retries = 0
    mark_step_completed()
```

### Acceptance Criteria

- Tool validation errors are fed back to model as tool result messages
- Model can self-correct from error messages (tested with bad paths, missing arguments)
- Re-plan triggers after N consecutive failures
- Re-plan preserves completed steps, replaces failed + remaining steps
- Loop does not infinite-retry on fundamentally impossible tasks

---

## Task 2.1.5: Destructive Operation Approval Gate

**Location:** Integrated into `ReActLoop._request_approval()`

### Flow

```
Model calls BashTool with "rm -rf build/"
    │
    ▼
ReActLoop receives ParsedToolCall(name="bash", arguments={"command": "rm -rf build/"})
    │
    ▼
Check: is this destructive?
    └─ Yes (matches DESTRUCTIVE_PREFIXES from shell.py)
        │
        ▼
    Prompt user: "[!] Destructive operation requested: rm -rf build/"
                 "Approve? (y/N): "
        │
        ├─ User types "y" or "yes" → execute normally
        │
        └─ User types anything else → skip with error:
            "Command rejected by user: destructive operation requires approval"
```

### Configuration

The approval gate can be disabled via `ReActConfig.require_approval_for_destructive = False`
for automated/CI usage where no user is available to approve.

### Acceptance Criteria

- Destructive commands (`rm`, `mv`, `>`, etc.) trigger approval prompt
- Non-destructive commands (`ls`, `grep`, `read`, etc.) pass through without prompt
- Approved commands execute normally with full result returned
- Rejected commands return structured error message to model
- Gate can be disabled via config flag

---

## Task 2.1.6: Rich-Based Debug TUI

**Directory:** `chef_human/ui/`
**Files:**
- `chef_human/ui/__init__.py`
- `chef_human/ui/debug_tui.py` — Rich-based debug panel
- `chef_human/ui/protocol.py` — `ReActUI` protocol/interface

### Protocol

```python
# chef_human/ui/protocol.py

from __future__ import annotations

from typing import Protocol


class ReActUI(Protocol):
    """Interface for UI components. Implementations can be TUI, CLI, or noop."""

    def on_start(self, task: str) -> None: ...
    def on_planning_start(self) -> None: ...
    def on_plan(self, plan: Plan) -> None: ...
    def on_reasoning_start(self) -> None: ...
    def on_reasoning(self, text: str) -> None: ...
    def on_tool_call(self, call: ParsedToolCall) -> None: ...
    def on_tool_result(self, tool_name: str, result: str) -> None: ...
    def on_replan(self) -> None: ...
    def on_error(self, message: str) -> None: ...
```

### Debug TUI Layout

```
┌──────────────────────────────────────────────────────────┐
│  chef-human  │  Step 3/25  │  Task: Fix bug in parser   │
├──────────────────────┬───────────────────────────────────┤
│  PLAN                │  REASONING                        │
│  [✓] Step 1: Read   │  The error occurs because the     │
│  [✓] Step 2: Reprod │  parser fails on nested JSON.     │
│  [→] Step 3: Fix    │  Let me look at the regex...      │
│  [ ] Step 4: Test   │                                   │
│  [ ] Step 5: Commit │                                   │
├──────────────────────┴───────────────────────────────────┤
│  TOOL CALLS                                              │
│  ▶ grep pattern="regex" → Found at parser.py:42          │
│  ▶ read path="parser.py" offset=40 limit=20              │
│  ✓ read returned 20 lines                                │
├──────────────────────────────────────────────────────────┤
│  LOG                                                     │
│  [14:32:01] Plan generated: 5 steps                      │
│  [14:32:05] Model reasoning: "I need to find the bug"    │
│  [14:32:07] Executing tool: grep                         │
└──────────────────────────────────────────────────────────┘
```

### Implementation Sketch

```python
# chef_human/ui/debug_tui.py

from __future__ import annotations

import time
from datetime import datetime

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree


class DebugTUI:
    """Rich-based debug terminal UI for the ReAct loop."""

    def __init__(self) -> None:
        self.console = Console()
        self.layout = Layout()
        self._setup_layout()
        self._plan_tree: Tree | None = None
        self._reasoning_text = ""
        self._tool_calls: list[tuple[str, str, str]] = []  # (icon, name, detail)
        self._log_entries: list[str] = []
        self._step_count = 0
        self._max_steps = 0

    def _setup_layout(self) -> None:
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        self.layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1),
        )
        self.layout["left"].split_column(
            Layout(name="plan_panel"),
            Layout(name="tool_panel"),
        )
        self.layout["right"].split_column(
            Layout(name="reasoning_panel"),
            Layout(name="log_panel"),
        )

    def _update_header(self, task: str) -> None:
        header = Panel(
            f"[bold cyan]chef-human[/] | Step {self._step_count}/{self._max_steps} | Task: {task[:60]}",
            style="white on dark_blue",
        )
        self.layout["header"].update(header)

    def on_start(self, task: str) -> None:
        self._log(f"Task started: {task}")
        self._update_header(task)

    def on_planning_start(self) -> None:
        self._log("Planning...")

    def on_plan(self, plan: Plan) -> None:
        self._max_steps = len(plan.steps)
        self._plan_tree = Tree("[bold]Plan[/]")
        for step in plan.steps:
            self._plan_tree.add(f"[ ] Step {step.index}: {step.description}")
        self.layout["plan_panel"].update(Panel(self._plan_tree, title="Plan"))
        self._log(f"Plan generated: {len(plan.steps)} steps")

    def on_reasoning_start(self) -> None:
        self._reasoning_text = ""
        self.layout["reasoning_panel"].update(
            Panel("Thinking...", title="Reasoning")
        )

    def on_reasoning(self, text: str) -> None:
        self._reasoning_text = text
        # Truncate for display
        display = text[:500] + ("..." if len(text) > 500 else "")
        self.layout["reasoning_panel"].update(
            Panel(display, title="Reasoning")
        )
        self._log("Model reasoning received")

    def on_tool_call(self, call: ParsedToolCall) -> None:
        args_str = ", ".join(f"{k}={v}" for k, v in call.arguments.items())
        self._tool_calls.append(("▶", call.name, args_str))
        self._refresh_tool_panel()
        self._log(f"Tool call: {call.name}({args_str})")

    def on_tool_result(self, tool_name: str, result: str) -> None:
        status = "✓" if not result.startswith("Error") else "✗"
        result_preview = result[:80] + ("..." if len(result) > 80 else "")
        self._tool_calls.append((status, tool_name, result_preview))
        self._refresh_tool_panel()
        self._log(f"Tool result: {tool_name} -> {result_preview}")

    def on_replan(self) -> None:
        self._log("[yellow]Re-planning...[/]")

    def on_error(self, message: str) -> None:
        self._log(f"[red]Error: {message}[/]")

    def _refresh_tool_panel(self) -> None:
        table = Table(show_header=False, box=None)
        table.add_column("icon", width=2)
        table.add_column("name", width=15)
        table.add_column("detail", width=60)
        for icon, name, detail in self._tool_calls[-20:]:  # last 20
            table.add_row(icon, name, detail)
        self.layout["tool_panel"].update(Panel(table, title="Tool Calls"))

    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._log_entries.append(f"[{timestamp}] {message}")
        log_text = "\n".join(self._log_entries[-10:])  # last 10
        self.layout["log_panel"].update(Panel(log_text, title="Log"))

    def display_final(self, result: AgentResult) -> None:
        self.console.print("\n[bold]=== Task Complete ===[/]")
        self.console.print(f"Steps taken: {result.steps_taken}")
        self.console.print(f"Success: {result.success}")
        self.console.print(f"Message: {result.message}")


class NoopUI:
    """No-op UI for headless/automated mode."""

    def on_start(self, task: str) -> None: ...
    def on_planning_start(self) -> None: ...
    def on_plan(self, plan: Plan) -> None: ...
    def on_reasoning_start(self) -> None: ...
    def on_reasoning(self, text: str) -> None: ...
    def on_tool_call(self, call: ParsedToolCall) -> None: ...
    def on_tool_result(self, tool_name: str, result: str) -> None: ...
    def on_replan(self) -> None: ...
    def on_error(self, message: str) -> None: ...
```

### Acceptance Criteria

- TUI renders plan progress with checkmarks/arrows for step status
- Tool calls and results appear in real-time as they execute
- Model reasoning text is displayed (truncated to fit)
- Log panel shows chronological event history
- NoopUI can be used as a drop-in replacement for automated testing
- Layout is responsive (resizes with terminal)
- TUI exits cleanly on completion or Ctrl+C

---

## Task 2.1.7: CLI Entry Point

**File:** `chef_human/main.py`

```python
from __future__ import annotations

import asyncio
import logging

import click

from chef_human.agent import create_context_assembler
from chef_human.agent.planner import Planner
from chef_human.agent.react_loop import ReActConfig, ReActLoop, ReActLoop
from chef_human.config import settings
from chef_human.llm import create_backend
from chef_human.tools import create_tool_registry
from chef_human.ui.debug_tui import DebugTUI


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.argument("task", required=False)
@click.option("--debug-tui", is_flag=True, default=True, help="Enable debug TUI")
@click.option("--max-steps", type=int, default=25, help="Max agent steps")
@click.option("--workspace", type=click.Path(exists=True), help="Workspace directory")
@click.option("--no-stream", is_flag=True, help="Disable streaming output")
def run(
    task: str | None,
    debug_tui: bool,
    max_steps: int,
    workspace: str | None,
    no_stream: bool,
) -> None:
    """Run chef-human on a task."""
    if not task:
        # Interactive mode: prompt for task
        task = click.prompt("Task", default="")
        if not task:
            click.echo("No task provided. Exiting.")
            return

    result = asyncio.run(_execute_task(
        task=task,
        debug_tui=debug_tui,
        max_steps=max_steps,
        workspace=workspace,
        stream=not no_stream,
    ))

    click.echo(f"\n{'='*40}")
    click.echo(f"Result: {'SUCCESS' if result.success else 'FAILURE'}")
    click.echo(f"Steps: {result.steps_taken}")
    click.echo(f"Message: {result.message}")
    if not result.success:
        raise SystemExit(1)


async def _execute_task(
    task: str,
    debug_tui: bool = True,
    max_steps: int = 25,
    workspace: str | None = None,
    stream: bool = True,
) -> AgentResult:
    logging.basicConfig(level=logging.WARNING)

    backend = create_backend()
    context = create_context_assembler()
    tool_registry = create_tool_registry(context._workspace)
    planner = Planner(llm_backend=backend)
    config = ReActConfig(
        max_steps=max_steps,
        stream=stream,
    )
    ui = DebugTUI() if debug_tui else NoopUI()

    loop = ReActLoop(
        llm_backend=backend,
        tool_registry=tool_registry,
        context_assembler=context,
        planner=planner,
        config=config,
        ui=ui,
    )

    return await loop.run(task)


if __name__ == "__main__":
    cli()
```

### pyproject.toml Entry Point

```toml
[project.scripts]
chef-human = "chef_human.main:cli"
```

### Acceptance Criteria

- `chef-human run "list files in current directory"` executes end-to-end
- `--max-steps` limits loop iterations (test with infinite loop scenario)
- `--no-stream` disables streaming output
- Debug TUI renders when `--debug-tui` (default on)
- Non-zero exit code on failure
- Interactive mode prompts for task when none provided as argument

---

## Task 2.1.8: Integration Tests & Factory Update

### Factory Update

**File:** `chef_human/agent/__init__.py` — add `create_agent()` factory:

```python
def create_agent(
    debug_tui: bool = False,
    max_steps: int = 25,
) -> tuple[ReActLoop, ContextAssembler]:
    """Create a fully-wired ReActLoop from config defaults."""
    from chef_human.llm import create_backend
    from chef_human.tools import create_tool_registry
    from chef_human.agent.planner import Planner
    from chef_human.agent.react_loop import ReActConfig, ReActLoop
    from chef_human.ui.debug_tui import DebugTUI
    from chef_human.ui.protocol import NoopUI

    backend = create_backend()
    context = create_context_assembler()
    tool_registry = create_tool_registry(context._workspace)
    planner = Planner(llm_backend=backend)
    config = ReActConfig(max_steps=max_steps)
    ui = DebugTUI() if debug_tui else NoopUI()

    loop = ReActLoop(
        llm_backend=backend,
        tool_registry=tool_registry,
        context_assembler=context,
        planner=planner,
        config=config,
        ui=ui,
    )
    return loop, context
```

### Test Files

| Test file | Test count | What it covers |
|-----------|-----------|----------------|
| `tests/test_agent/test_planner.py` | ~15 | Plan dataclass, `generate_plan()` parsing (JSON array, plain text, malformed), `update_plan()` merge, `format_plan_for_prompt()`, StepStatus enum |
| `tests/test_agent/test_parser.py` | ~20 | `parse_tool_calls()` with `<tool_call>` tags, code blocks, bare JSON, Ollama format, malformed input, empty input, duplicate detection; `validate_arguments()` required/optional/type/unknown; `strip_tool_calls()` |
| `tests/test_agent/test_react_loop.py` | ~25 | Mock LLM + mock tools: basic flow, plan generation, tool call execution, tool result recording, finish detection, unknown tool, invalid args, max steps exceeded, re-plan on failure, destructive approval gate, step counter |
| `tests/test_agent/test_tui.py` | ~10 | Protocol interface check, NoopUI all methods, DebugTUI layout, message formatting, on_start/on_plan/on_reasoning/on_tool_call/on_tool_result |

### Acceptance Criteria

- [ ] All unit tests pass without requiring a running LLM backend (mocked)
- [ ] Integration test with real (or mocked) backend validates full loop
- [ ] `create_agent()` factory wires all components without error
- [ ] Parser handles all 4 tool call formats correctly
- [ ] Planner falls back gracefully on malformed LLM responses
- [ ] ReAct loop detects and handles all error scenarios

---

## Task 2.1.9: System Prompt Design & Agent Message Templates

**File:** `chef_human/agent/prompts.py` (shared prompt constants)

### Prompt Components

```python
# chef_human/agent/prompts.py

from __future__ import annotations

from chef_human.agent.planner import Plan
from chef_human.llm.chatml import format_tool_definitions
from chef_human.llm.backend import ToolDefinition


# Base system prompt for the planner (separate LLM call)
PLANNER_SYSTEM_PROMPT = """You are a planning assistant for a software engineering AI.
Given a user's task, break it down into a series of concrete steps.

Rules:
- Each step must be actionable with the available tools (read, write, edit, grep, glob, ls, bash)
- Steps should be ordered by dependency
- Each step should have a clear completion criterion
- Output ONLY a JSON array of strings, e.g. ["Step 1", "Step 2", "Step 3"]
- Do NOT include any explanation or markdown — just the JSON array"""

# Base system prompt for the ReAct loop agent
AGENT_SYSTEM_PROMPT = """You are chef-human, an AI software engineering assistant.
You have access to tools that let you read, write, and search files, run commands, and ask the user.

## How to use tools
To call a tool, output:
<tool_call>{ "name": "tool_name", "arguments": { "arg1": "value1" } }</tool_call>

After each tool result, analyze it and decide the next action.
When ALL steps of the plan are complete, call the `finish` tool.

## Guidelines
- Follow the plan. Complete steps in order.
- Reason step-by-step before calling tools.
- If a tool fails, read the error and fix your approach.
- After 3 consecutive failures, the system will re-plan automatically.
- Do not call finish until all plan steps are done.

## Project Structure
{repo_map}

## Plan
{plan_text}

## Available Tools
{tool_definitions}"""


AGENT_FINISH_PROMPT = """
The task is now complete. Summarize what was accomplished:
- What changes were made
- What files were affected
- Any important decisions or trade-offs"""


def build_planner_prompt(task: str, repo_context: str = "") -> str:
    """Build the prompt for the planner LLM call."""
    prompt = PLANNER_SYSTEM_PROMPT
    if repo_context:
        prompt += f"\n\nProject context:\n{repo_context}"
    prompt += f"\n\nUser task: {task}"
    return prompt


def build_agent_prompt(
    plan: Plan,
    tool_defs: list[ToolDefinition],
    repo_map: str = "",
) -> str:
    """Build the full system prompt for the agent loop."""
    plan_text = Plan.format_plan_for_prompt(plan)
    tool_text = format_tool_definitions(tool_defs)

    return AGENT_SYSTEM_PROMPT.format(
        repo_map=repo_map or "(no project context loaded)",
        plan_text=plan_text,
        tool_definitions=tool_text,
    )
```

### Acceptance Criteria

- Prompts are clear, tested with model (format renders correctly)
- `build_agent_prompt()` correctly formats plan + tools + repo map
- Planner prompt produces valid JSON arrays in practice
- Tool calling instructions are unambiguous
- Finish-detection logic works with model's natural language

---

## Dependencies Map

```
2.1.1 planner.py ─────────────► 1.1 backend.py, 1.1.5 chatml.py
2.1.2 parser.py ──────────────► stdlib (json, re, dataclasses)
2.1.3 react_loop.py ──────────► 2.1.1 planner.py, 2.1.2 parser.py,
                                 1.3 registry.py, 1.2.4 context.py,
                                 1.1 backend.py
2.1.4 (integrated in 2.1.3)   ► same as 2.1.3
2.1.5 (integrated in 2.1.3)   ► 1.3.3 shell.py (DESTRUCTIVE_PREFIXES)
2.1.6 debug_tui.py ───────────► rich (existing dep), 2.1.1 plan types, 2.1.2 parser types
2.1.7 main.py ────────────────► click (existing dep), all 2.1.x modules
2.1.8 tests ──────────────────► all 2.1.x modules
2.1.9 prompts.py ─────────────► 1.1.5 chatml.py, 2.1.1 planner.py
```

---

## Implementation Order

1. **2.1.2** Parser (no external deps, easiest to test first) — `chef_human/agent/parser.py`
2. **2.1.1** Planner (depends on backend, but testable with mock) — `chef_human/agent/planner.py`
3. **2.1.9** Prompts (shared text, no logic) — `chef_human/agent/prompts.py`
4. **2.1.3** ReAct loop (core — depends on 2.1.1, 2.1.2, 2.1.9)
5. **2.1.4** Self-correction (integrated in loop body)
6. **2.1.5** Approval gate (integrated in loop body)
7. **2.1.6** Debug TUI (independent, can be parallel with 2.1.3–2.1.5)
8. **2.1.7** CLI entry point (wires everything together)
9. **2.1.8** Tests + factory update (final validation)

---

## Changes & Deviations Tracking

### 2.1.1 Implementation Notes

**Files created:**
- `chef_human/agent/planner.py` — Planner + data structures (127 lines)
- `tests/test_agent/test_planner.py` — 30 tests (312 lines)
- `chef_human/agent/__init__.py` — updated with `Plan`, `PlanStep`, `Planner`, `StepStatus` exports

**30 tests pass covering:**
| Test class | Tests | Coverage |
|-----------|-------|----------|
| `TestStepStatus` | 2 | Enum values, string enum behavior |
| `TestPlanStep` | 3 | Default status, custom status, equality (dataclass auto-generated) |
| `TestPlan` | 2 | Creation, steps list |
| `TestParseSteps` | 11 | JSON array of strings, objects (with/without `description` key), extra text around JSON, plain text lines, empty lines, malformed JSON, empty content, single object, whitespace padding, mixed list items |
| `TestFormatPlanForPrompt` | 4 | Empty plan, single step, mixed statuses, all 5 markers present |
| `TestGeneratePlan` | 4 | Basic, with repo context, correct messages, repo context in messages |
| `TestUpdatePlan` | 4 | Merge completed steps, skip duplicates, no completed steps, failure context sent |

**Important observations and fixes:**

1. **`test_parse_steps` fallback with unknown dict keys**: When dict items lack a `"description"` key, the fallback uses `str(item)` which produces `"{'name': 'Read file'}"` — awkward but functional. In practice, the planner prompt instructs the model to output `"description"` fields, so this edge case shouldn't occur in normal operation.

2. **Mock test pattern**: Tests use a `_make_mock_backend()` helper that creates a `MagicMock` with an `AsyncMock` for `complete()`. The `complete()` method takes a `CompletionRequest` as a **positional argument** (not kwargs), so mock verification uses `call_args.args[0].messages`. The plan's test sketches assumed `call_args.kwargs["messages"]` — corrected.

3. **`json` import needed in tests**: The plan's test skeleton omitted the `import json` needed by `_make_mock_backend()`. Added in implementation.

4. **`assert call_args is not None` guards**: Required to pass pyright's `reportOptionalMemberAccess`. Two tests in the plan sketch were missing this guard.

5. **JSON array regex edge case**: The regex `r"\[.*\]"` with `re.DOTALL` matches from the first `[` to the last `]`, which could be overly greedy with multiple arrays or `]` inside strings. Not a practical concern for the planner prompt's expected output, but worth noting if parsing accuracy becomes an issue later.

6. **No `__all__` in `planner.py`**: Follows the same convention as `backend.py` and other modules — exports are managed via the package `__init__.py`.

Key areas to watch (remaining tasks):

1. **Mock-friendly testing**: All loop tests should work without a real LLM. Use `unittest.mock` for `LLMBackend.complete()`.

2. **Tool call format compatibility**: The model might output tool calls in unexpected formats. The parser must be resilient.

3. **Approval gate UX in TUI**: If the debug TUI is active, destructive operation prompts should appear within the TUI, not break out to raw stdin.

4. **Streaming vs. non-streaming**: The current `LLMBackend.complete()` doesn't support streaming. If streaming is desired for the TUI, both backends need streaming support. This could be a follow-up.

5. **Max steps vs. real work**: Some tasks genuinely need more than 25 steps. The `max_steps` config should be generous by default (50?) or user-configurable.

6. **`context._workspace` access**: The factory accesses `context._workspace` (private). Consider making `workspace` a public property on `ContextAssembler`.

7. **`ReActLoop` needs `ContextAssembler`'s conversation manager**: Currently `ContextAssembler` wraps `ContextManager` (conversation). The loop needs to add messages to the conversation. Ensure `ContextAssembler` exposes its `conversation` attribute.

---

## Future Improvements (Post-2.1)

- **Streaming model output**: Add streaming support to backends so TUI shows reasoning token-by-token.
- **Parallel tool execution**: Batch multiple independent tool calls in a single turn.
- **Conversation save/load**: Persist conversation state to disk for resumption.
- **Cost/token tracking**: Track per-step token usage and total cost.
- **Agent memory**: Long-term memory across sessions (separate from conversation context).
- **`agent_scratchpad`**: Let the model maintain a scratchpad of notes across turns.
- **CLI `--headless` mode**: Run without TUI, output structured JSON result.
- **TUI improvements**: Color-coded plan steps, expandable reasoning panels, searchable log.
