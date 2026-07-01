from __future__ import annotations

import logging
from dataclasses import dataclass

from chef_human.agent.context import ContextAssembler
from chef_human.agent.parser import (
    ParsedToolCall,
    parse_tool_calls,
    strip_tool_calls,
    validate_arguments,
)
from chef_human.agent.planner import Plan, Planner, StepStatus
from chef_human.agent.prompts import build_agent_prompt
from chef_human.agent.retry import RetryAction, RetryManager
from chef_human.llm.backend import (
    CompletionRequest,
    LLMBackend,
    Message,
    Role,
)
from chef_human.tools.registry import ToolRegistry
from chef_human.ui.protocol import NoopUI, ReActUI

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    plan: Plan
    steps_taken: int
    message: str
    success: bool = True


@dataclass
class ReActConfig:
    max_steps: int = 25
    max_retries_per_step: int = 3
    max_replans: int = 1
    temperature: float = 0.0
    max_tokens_per_response: int = 4096
    require_approval_for_destructive: bool = True
    stream: bool = False


class ReActLoop:
    def __init__(
        self,
        llm_backend: LLMBackend,
        tool_registry: ToolRegistry,
        context_assembler: ContextAssembler,
        planner: Planner,
        config: ReActConfig | None = None,
        ui: ReActUI | None = None,
    ) -> None:
        self._llm = llm_backend
        self._tools = tool_registry
        self._context = context_assembler
        self._planner = planner
        self._config = config or ReActConfig()
        self._ui = ui or NoopUI()

    async def run(self, task: str) -> AgentResult:
        self._ui.on_start(task)
        steps_taken = 0
        plan = await self._plan_task(task)
        retry_mgr = RetryManager(
            max_retries_per_step=self._config.max_retries_per_step,
            max_replans=self._config.max_replans,
        )

        self._context.conversation.add_message(
            Message(role=Role.user, content=task)
        )

        while steps_taken < self._config.max_steps:
            system_prompt = build_agent_prompt(
                plan=plan,
                tool_defs=self._tools.get_definitions(),
            )
            messages = self._context.assemble(
                system_prompt=system_prompt,
                tool_definitions="",
            )
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

            tool_calls = parse_tool_calls(response.message.content)
            non_tool_reasoning = strip_tool_calls(response.message.content)

            assistant_msg = Message(
                role=Role.assistant,
                content=non_tool_reasoning,
                tool_calls=[
                    {"function": {"name": tc.name, "arguments": tc.arguments}}
                    for tc in tool_calls
                ],
            )
            self._context.conversation.add_message(assistant_msg)

            if not tool_calls:
                steps_taken += 1
                action = retry_mgr.record_iteration(True, [])
                if action == RetryAction.STEP_COMPLETED:
                    self._mark_step_completed(plan, [])
                if self._detect_finish(non_tool_reasoning):
                    return self._make_result(
                        plan=plan,
                        steps_taken=steps_taken,
                        message=non_tool_reasoning,
                    )
                continue

            all_success = True
            tool_results: list[str] = []

            for tc in tool_calls:
                self._ui.on_tool_call(tc)

                tool = self._tools.get(tc.name)
                if tool is None:
                    result = self._make_tool_error(
                        f"Unknown tool: '{tc.name}'. Available: {', '.join(self._tools.list_tools())}"
                    )
                    self._ui.on_tool_result(tc.name, result)
                    tool_results.append(result)
                    all_success = False
                    continue

                errors = validate_arguments(tc, tool.parameters)
                if errors:
                    error_msg = f"Invalid arguments for {tc.name}: {'; '.join(errors)}"
                    result = self._make_tool_error(error_msg)
                    self._ui.on_tool_result(tc.name, result)
                    tool_results.append(result)
                    all_success = False
                    continue

                if (
                    self._config.require_approval_for_destructive
                    and tc.name == "bash"
                ):
                    if self._is_destructive_command(tc.arguments.get("command", "")):
                        approved = await self._request_approval(tc)
                        if not approved:
                            result = self._make_tool_error(
                                "Command rejected by user: destructive operation requires approval"
                            )
                            self._ui.on_tool_result(tc.name, result)
                            tool_results.append(result)
                            continue

                try:
                    tool_result = await tool.run(**tc.arguments)
                except Exception as exc:
                    result = self._make_tool_error(f"Execution error: {exc}")
                    self._ui.on_tool_result(tc.name, result)
                    tool_results.append(result)
                    all_success = False
                    continue

                if tc.name == "finish":
                    finish_msg = tool_result.output if tool_result.success else tool_result.error or ""
                    self._ui.on_tool_result(tc.name, finish_msg)
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

            for result_text in tool_results:
                self._context.conversation.add_message(
                    Message(role=Role.tool, content=result_text)
                )

            steps_taken += 1
            action = retry_mgr.record_iteration(all_success, tool_results)

            if action == RetryAction.STEP_COMPLETED:
                self._mark_step_completed(plan, tool_results)
            elif action == RetryAction.REPLAN:
                self._ui.on_replan()
                plan = await self._planner.update_plan(
                    plan,
                    failure_context="\n".join(tool_results),
                )
                retry_mgr.on_replan()
            elif action == RetryAction.ESCALATE:
                return self._make_result(
                    plan=plan,
                    steps_taken=steps_taken,
                    message="The task could not be completed despite re-planning. "
                            "The agent encountered persistent failures.",
                    success=False,
                )

        return self._make_result(
            plan=plan,
            steps_taken=steps_taken,
            message="Max steps exceeded. The task may be incomplete.",
            success=False,
        )

    async def _plan_task(self, task: str) -> Plan:
        self._ui.on_planning_start()
        repo_context = self._get_repo_context()
        plan = await self._planner.generate_plan(task, repo_context=repo_context)
        self._ui.on_plan(plan)
        return plan

    def _get_repo_context(self) -> str:
        try:
            tree = self._context._repo_map.generate_tree()
            return tree[:1000]
        except Exception:
            return ""

    def _mark_step_completed(self, plan: Plan, results: list[str]) -> None:
        for step in plan.steps:
            if step.status == StepStatus.pending:
                step.status = StepStatus.in_progress
                step.status = StepStatus.completed
                break

    def _detect_finish(self, content: str) -> bool:
        triggers = [
            "task is complete",
            "i have finished",
            "all done",
            "finished the task",
        ]
        return any(t in content.lower() for t in triggers)

    def _is_destructive_command(self, command: str) -> bool:
        from chef_human.tools.shell import DESTRUCTIVE_PREFIXES
        stripped = command.strip()
        for prefix in DESTRUCTIVE_PREFIXES:
            if stripped.startswith(prefix):
                return True
        return False

    async def _request_approval(self, tool_call: ParsedToolCall) -> bool:
        result = await self._ui.on_approval_request(tool_call)
        if result is not None:
            return result
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
