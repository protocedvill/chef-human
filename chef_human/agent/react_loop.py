from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass


from chef_human.agent.context import ContextAssembler
from chef_human.agent.linter import format_lint_result, run_lint
from chef_human.agent.parser import (
    ParsedToolCall,
    extract_scratchpad,
    parse_tool_calls,
    strip_scratchpad,
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
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "steps_taken": self.steps_taken,
            "message": self.message,
            "plan": self.plan.to_dict(),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }


@dataclass
class ReActConfig:
    max_steps: int = 25
    max_retries_per_step: int = 3
    max_replans: int = 1
    temperature: float = 0.0
    max_tokens_per_response: int = 4096
    require_approval_for_destructive: bool = True
    stream: bool = False
    save_sessions: bool = True
    save_dir: str | None = None
    lint_after_write: bool = True


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
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

    async def run(self, task: str) -> AgentResult:
        self._ui.on_start(task)
        steps_taken = 0
        plan = await self._plan_task(task)
        scratchpad = ""
        retry_mgr = RetryManager(
            max_retries_per_step=self._config.max_retries_per_step,
            max_replans=self._config.max_replans,
        )

        self._context.conversation.add_message(
            Message(role=Role.user, content=task)
        )

        try:
            while steps_taken < self._config.max_steps:
                system_prompt = build_agent_prompt(
                    plan=plan,
                    tool_defs=self._tools.get_definitions(),
                    scratchpad=scratchpad,
                )
                messages = self._context.assemble(
                    system_prompt=system_prompt,
                    tool_definitions="",
                )
                self._ui.on_reasoning_start()
                if self._config.stream:
                    full_content = ""
                    async for token, final_response in self._llm.complete_stream(
                        CompletionRequest(
                            messages=messages,
                            tools=self._tools.get_definitions(),
                            temperature=self._config.temperature,
                            max_tokens=self._config.max_tokens_per_response,
                        )
                    ):
                        if final_response is not None:
                            response = final_response
                        else:
                            full_content += token
                            self._ui.on_stream(token)
                    if full_content:
                        response.message.content = full_content
                else:
                    response = await self._llm.complete(
                        CompletionRequest(
                            messages=messages,
                            tools=self._tools.get_definitions(),
                            temperature=self._config.temperature,
                            max_tokens=self._config.max_tokens_per_response,
                        )
                    )
                self._ui.on_reasoning(response.message.content)

                if response.usage:
                    self._total_prompt_tokens += response.usage.get("prompt_tokens", 0)
                    self._total_completion_tokens += response.usage.get("completion_tokens", 0)

                new_scratchpad = extract_scratchpad(response.message.content)
                if new_scratchpad is not None:
                    scratchpad = new_scratchpad
                    logger.debug("Scratchpad updated: %s", scratchpad[:100])

                tool_calls = parse_tool_calls(response.message.content)
                non_tool_reasoning = strip_scratchpad(response.message.content)
                non_tool_reasoning = strip_tool_calls(non_tool_reasoning)

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
                finish_call: tuple[ParsedToolCall, object] | None = None
                parallel_candidates: list[tuple[ParsedToolCall, object]] = []

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

                    if tc.name == "finish":
                        finish_call = (tc, tool)
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

                    parallel_candidates.append((tc, tool))

                if parallel_candidates:
                    coros = [tool.run(**tc.arguments) for tc, tool in parallel_candidates]
                    gathered = await asyncio.gather(*coros, return_exceptions=True)

                    for (tc, _tool), tool_result in zip(parallel_candidates, gathered):
                        if isinstance(tool_result, Exception):
                            result = self._make_tool_error(f"Execution error: {tool_result}")
                            self._ui.on_tool_result(tc.name, result)
                            tool_results.append(result)
                            all_success = False
                            continue

                        if not tool_result.success:
                            result = f"Error: {tool_result.error}\nOutput: {tool_result.output}"
                            all_success = False
                        else:
                            result = tool_result.output

                        self._ui.on_tool_result(tc.name, result)
                        tool_results.append(result)

                        if (
                            self._config.lint_after_write
                            and tool_result.success
                            and tc.name in ("write", "edit")
                        ):
                            file_path = tc.arguments.get("path", "")
                            lint_output = run_lint(file_path)
                            if lint_output:
                                lint_result = format_lint_result(lint_output)
                                tool_results.append(lint_result)

                if finish_call is not None:
                    tc, tool = finish_call
                    try:
                        finish_result = await tool.run(**tc.arguments)
                    except Exception as exc:
                        result = self._make_tool_error(f"Execution error: {exc}")
                        self._ui.on_tool_result(tc.name, result)
                        tool_results.append(result)
                        all_success = False
                    else:
                        finish_msg = finish_result.output if finish_result.success else finish_result.error or ""
                        self._ui.on_tool_result(tc.name, finish_msg)
                        return self._make_result(
                            plan=plan,
                            steps_taken=steps_taken,
                            message=finish_result.output,
                        )

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
                    scratchpad = ""
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
        finally:
            self._save_conversation(task)

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

    def _make_result(
        self,
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
            total_prompt_tokens=self._total_prompt_tokens,
            total_completion_tokens=self._total_completion_tokens,
        )

    def _save_conversation(self, task: str) -> None:
        if not self._config.save_sessions:
            return
        from chef_human.agent.persistence import save_conversation
        conv = self._context.conversation.to_dict()
        if self._config.save_dir is not None:
            save_conversation(conv, task=task, save_dir=self._config.save_dir)
        else:
            save_conversation(conv, task=task)
