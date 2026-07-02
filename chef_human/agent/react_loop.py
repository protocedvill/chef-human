from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path


from chef_human.agent.context import ContextAssembler
from chef_human.agent.linter import annotate_diff_with_lint, format_lint_result, run_lint
from chef_human.agent.parser import (
    ParsedToolCall,
    extract_scratchpad_entries,
    format_parse_error,
    looks_like_tool_call,
    parse_tool_calls,
    strip_scratchpad,
    strip_tool_calls,
    validate_arguments,
)
from chef_human.agent.planner import Plan, Planner, StepStatus, StepVerdict
from chef_human.agent.prompts import build_agent_prompt
from chef_human.agent.retry import RetryAction, RetryManager
from chef_human.agent.scratchpad import Scratchpad
from chef_human.llm.backend import (
    CompletionRequest,
    LLMBackend,
    Message,
    Role,
)
from chef_human.tools.registry import ToolRegistry
from chef_human.ui.protocol import NoopUI, ReActUI

logger = logging.getLogger(__name__)

# Steps matching these don't produce a durable artifact to point to as
# evidence -- asking a small model "was this really done?" about a read/
# investigate-style step is prone to false negatives, which just prompts it
# to redo the same read over and over.
_INVESTIGATIVE_KEYWORDS = (
    "read", "identify", "check", "review", "analyz", "examine", "inspect",
    "explore", "understand", "look at", "list", "find", "search", "locate",
)

# Generic "what should I do" questions that ignore an active plan step
# entirely, rather than asking about a genuine ambiguity.
_VAGUE_ASK_USER_PATTERNS = (
    "what would you like to do next",
    "what should i do next",
    "what do you want me to do",
    "what would you like me to do",
    "what's next",
    "what next",
)


def _rollback_file(path: str, content: str) -> None:
    """Restore a file to its pre-write content."""
    Path(path).write_text(content)


def _looks_investigative(description: str) -> bool:
    lowered = description.lower()
    return any(kw in lowered for kw in _INVESTIGATIVE_KEYWORDS)


def _is_vague_next_step_question(question: str) -> bool:
    q = question.lower().strip().rstrip("?")
    return any(p in q for p in _VAGUE_ASK_USER_PATTERNS)


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
    tool_timeout: float = 60.0
    require_read_before_edit: bool = True
    block_vague_ask_user: bool = True
    require_plan_complete_to_finish: bool = True


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
        # Planner's own LLM calls (plan-building, step verification) run on
        # a separate call path from the main reasoning loop below -- wire
        # them into the same running total and live UI updates.
        self._planner.on_usage = self._record_usage

    def _record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        self._ui.on_token_usage(prompt_tokens, completion_tokens)

    async def run(self, task: str) -> AgentResult:
        logger.info("Task started: %s", task[:200])
        self._ui.on_start(task)
        steps_taken = 0
        plan = await self._plan_task(task)
        logger.info(
            "Plan generated: %d step(s): %s",
            len(plan.steps),
            [s.description for s in plan.steps],
        )
        scratchpad = Scratchpad()
        last_call_signature: str | None = None
        files_read: set[str] = set()
        retry_mgr = RetryManager(
            max_retries_per_step=self._config.max_retries_per_step,
            max_replans=self._config.max_replans,
        )

        self._context.conversation.add_message(
            Message(role=Role.user, content=task)
        )

        try:
            while steps_taken < self._config.max_steps:
                current = plan.current_step()
                logger.debug(
                    "Turn starting: steps_taken=%d/%d, current step=%r",
                    steps_taken, self._config.max_steps,
                    current.description if current else "(none -- all complete)",
                )
                system_prompt = build_agent_prompt(
                    plan=plan,
                    tool_defs=self._tools.get_definitions(),
                    scratchpad=scratchpad.render(),
                )
                messages = self._context.assemble(
                    system_prompt=system_prompt,
                )
                self._ui.on_reasoning_start()
                llm_start = time.monotonic()
                logger.debug("LLM call starting (stream=%s, %d messages)", self._config.stream, len(messages))
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
                logger.debug(
                    "LLM call finished in %.1fs (%d chars, usage=%s)",
                    time.monotonic() - llm_start,
                    len(response.message.content),
                    response.usage,
                )
                self._ui.on_reasoning(response.message.content)

                if response.usage:
                    self._record_usage(
                        response.usage.get("prompt_tokens", 0),
                        response.usage.get("completion_tokens", 0),
                    )

                new_entries = extract_scratchpad_entries(response.message.content)
                if new_entries:
                    scratchpad.add_lines(new_entries)
                    logger.debug("Scratchpad gained %d new entr%s", len(new_entries), "y" if len(new_entries) == 1 else "ies")

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
                    parse_error = None
                    if looks_like_tool_call(response.message.content):
                        parse_error = format_parse_error(
                            response.message.content,
                            detail="Could not extract valid tool call JSON",
                        )
                        logger.warning("Response looked like a tool call but failed to parse")
                        self._context.conversation.add_message(
                            Message(role=Role.tool, content=parse_error)
                        )
                    else:
                        logger.debug("No tool calls this turn (plain reasoning only)")

                    steps_taken += 1
                    if parse_error:
                        action = retry_mgr.record_iteration(1, 1, [parse_error])
                    else:
                        action = retry_mgr.record_iteration(0, 0, [])

                    if action == RetryAction.STEP_COMPLETED:
                        verify_feedback = await self._verify_and_mark_step(
                            plan, non_tool_reasoning
                        )
                        if verify_feedback:
                            self._ui.on_tool_result("plan-check", verify_feedback)
                            self._context.conversation.add_message(
                                Message(role=Role.tool, content=verify_feedback)
                            )

                    if self._detect_finish(non_tool_reasoning) and not parse_error:
                        logger.info("Task finished via finish-phrase detection after %d step(s)", steps_taken)
                        return self._make_result(
                            plan=plan,
                            steps_taken=steps_taken,
                            message=non_tool_reasoning,
                        )
                    continue

                call_signature = json.dumps(
                    sorted([[tc.name, tc.arguments] for tc in tool_calls], key=lambda x: x[0]),
                    sort_keys=True,
                    default=str,
                )
                is_repeat_call = call_signature == last_call_signature
                last_call_signature = call_signature

                total_calls = len(tool_calls)
                failed_calls = 0
                tool_results: list[str] = []
                finish_call: tuple[ParsedToolCall, object] | None = None
                parallel_candidates: list[tuple[ParsedToolCall, object]] = []

                for tc in tool_calls:
                    logger.debug("Tool call: %s(%s)", tc.name, tc.arguments)
                    self._ui.on_tool_call(tc)

                    tool = self._tools.get(tc.name)
                    if tool is None:
                        result = self._make_tool_error(
                            f"Unknown tool: '{tc.name}'. Available: {', '.join(self._tools.list_tools())}"
                        )
                        self._ui.on_tool_result(tc.name, result)
                        tool_results.append(result)
                        failed_calls += 1
                        continue

                    errors = validate_arguments(tc, tool.parameters)
                    if errors:
                        error_msg = f"Invalid arguments for {tc.name}: {'; '.join(errors)}"
                        result = self._make_tool_error(error_msg)
                        self._ui.on_tool_result(tc.name, result)
                        tool_results.append(result)
                        failed_calls += 1
                        continue

                    if tc.name == "finish":
                        if self._config.require_plan_complete_to_finish:
                            current = plan.current_step()
                            if current is not None:
                                logger.info(
                                    "Blocked premature finish: unfinished step %r remains",
                                    current.description,
                                )
                                result = self._make_tool_error(
                                    "You cannot finish yet -- there's an unfinished plan step: "
                                    f"'{current.description}'. Complete it (and any remaining "
                                    "steps) before calling finish. A tool result claiming work "
                                    "is done is not evidence -- you must actually perform the "
                                    "step (e.g. write/edit the relevant files)."
                                )
                                self._ui.on_tool_result(tc.name, result)
                                tool_results.append(result)
                                failed_calls += 1
                                continue
                        finish_call = (tc, tool)
                        continue

                    if tc.name == "ask_user":
                        current = plan.current_step()
                        question = tc.arguments.get("question", "")
                        if (
                            self._config.block_vague_ask_user
                            and current is not None
                            and _is_vague_next_step_question(question)
                        ):
                            logger.info("Blocked vague ask_user: %r", question[:150])
                            result = self._make_tool_error(
                                "Don't ask what to do next -- there's an active plan step: "
                                f"'{current.description}'. Work on it directly instead of "
                                "asking; only use ask_user for a genuine, specific ambiguity."
                            )
                            self._ui.on_tool_result(tc.name, result)
                            tool_results.append(result)
                            failed_calls += 1
                            continue

                        # Route through the active UI rather than AskUserTool.run()'s
                        # own sys.stdin.readline() -- under the Textual TUI, Textual
                        # owns the terminal in raw mode, so a plain blocking stdin
                        # read there both hangs the whole event loop and can never
                        # actually receive the answer (no visible prompt, nothing
                        # forwarded to the tool's file descriptor). Each UI decides
                        # how it collects an answer (a proper modal for the TUI,
                        # print+stdin for terminal-based UIs).
                        logger.info("ask_user: %r", question[:200])
                        answer = await self._ui.on_ask_user(question)
                        logger.info("ask_user answer: %r", answer[:200])
                        self._ui.on_tool_result(tc.name, answer)
                        tool_results.append(answer)
                        continue

                    if (
                        self._config.require_approval_for_destructive
                        and tc.name == "bash"
                    ):
                        if self._is_destructive_command(tc.arguments.get("command", "")):
                            logger.info("Requesting approval for destructive command: %s", tc.arguments.get("command", "")[:200])
                            approved = await self._request_approval(tc)
                            logger.info("Approval result: %s", approved)
                            if not approved:
                                result = self._make_tool_error(
                                    "Command rejected by user: destructive operation requires approval"
                                )
                                self._ui.on_tool_result(tc.name, result)
                                tool_results.append(result)
                                failed_calls += 1
                                continue

                    if (
                        self._config.require_read_before_edit
                        and tc.name in ("write", "edit")
                    ):
                        read_key = self._unread_existing_file(tc.arguments.get("path", ""), files_read)
                        if read_key is not None:
                            logger.info("Blocked %s of unread file: %r", tc.name, read_key)
                            result = self._make_tool_error(
                                f"You haven't read '{read_key}' yet this session. Read it "
                                "first with the `read` tool so this change is based on the "
                                "file's actual current content, not a guess."
                            )
                            self._ui.on_tool_result(tc.name, result)
                            tool_results.append(result)
                            failed_calls += 1
                            continue

                    parallel_candidates.append((tc, tool))

                if parallel_candidates:
                    # Capture original file content for write/edit tools (for rollback)
                    pre_write_content: dict[str, str | None] = {}
                    if self._config.lint_after_write:
                        for tc, _tool in parallel_candidates:
                            if tc.name in ("write", "edit"):
                                path = tc.arguments.get("path", "")
                                pre_write_content[path] = self._capture_file_content(path)

                    coros = [
                        asyncio.wait_for(
                            tool.run(**tc.arguments),
                            timeout=self._config.tool_timeout,
                        )
                        for tc, tool in parallel_candidates
                    ]
                    dispatch_start = time.monotonic()
                    gathered = await asyncio.gather(*coros, return_exceptions=True)
                    logger.debug(
                        "Dispatched %d tool call(s) in %.1fs: %s",
                        len(parallel_candidates),
                        time.monotonic() - dispatch_start,
                        [tc.name for tc, _ in parallel_candidates],
                    )

                    for (tc, _tool), tool_result in zip(parallel_candidates, gathered):
                        if isinstance(tool_result, Exception):
                            result = self._make_tool_error(f"Execution error: {tool_result}")
                            self._ui.on_tool_result(tc.name, result)
                            tool_results.append(result)
                            failed_calls += 1
                            continue

                        if not tool_result.success:
                            result = f"Error: {tool_result.error}\nOutput: {tool_result.output}"
                            failed_calls += 1
                        else:
                            result = tool_result.output

                        self._ui.on_tool_result(tc.name, result)
                        tool_results.append(result)

                        # Track files the model has now seen the content of --
                        # via an explicit read, or because it just wrote/edited
                        # them itself (so a later edit to the same file isn't
                        # blocked by the read-before-edit guard below).
                        if tool_result.success and tc.name in ("read", "write", "edit"):
                            seen_path = tc.arguments.get("path", "")
                            if seen_path:
                                try:
                                    files_read.add(str(self._context.workspace.resolve(seen_path)))
                                except Exception:
                                    pass

                        if (
                            self._config.lint_after_write
                            and tool_result.success
                            and tc.name in ("write", "edit")
                        ):
                            file_path = tc.arguments.get("path", "")
                            lint_output = run_lint(file_path)
                            if lint_output:
                                # If lint has actual errors, roll back the file
                                original = pre_write_content.get(file_path)
                                if original is not None:
                                    _rollback_file(file_path, original)
                                    rollback_msg = (
                                        f"\n[rollback] Lint errors detected — "
                                        f"file '{file_path}' restored to pre-write state."
                                    )
                                else:
                                    rollback_msg = (
                                        f"\n[rollback] Lint errors detected — "
                                        f"file '{file_path}' was new, cannot restore."
                                    )
                                # Annotate the last tool result's diff if present
                                last_idx = len(tool_results) - 1
                                if last_idx >= 0 and "```diff" in tool_results[last_idx]:
                                    annotated = annotate_diff_with_lint(
                                        tool_results[last_idx], lint_output
                                    )
                                    if annotated:
                                        tool_results[last_idx] = annotated
                                # Append lint output with rollback note
                                lint_result = format_lint_result(lint_output) + rollback_msg
                                tool_results.append(lint_result)
                                failed_calls += 1

                if finish_call is not None:
                    tc, tool = finish_call
                    try:
                        finish_result = await asyncio.wait_for(
                            tool.run(**tc.arguments),
                            timeout=self._config.tool_timeout,
                        )
                    except asyncio.TimeoutError:
                        result = self._make_tool_error(
                            f"Tool '{tc.name}' timed out after {self._config.tool_timeout}s"
                        )
                        self._ui.on_tool_result(tc.name, result)
                        tool_results.append(result)
                        failed_calls += 1
                    except Exception as exc:
                        result = self._make_tool_error(f"Execution error: {exc}")
                        self._ui.on_tool_result(tc.name, result)
                        tool_results.append(result)
                        failed_calls += 1
                    else:
                        finish_msg = finish_result.output if finish_result.success else finish_result.error or ""
                        self._ui.on_tool_result(tc.name, finish_msg)
                        logger.info("Task finished via finish tool after %d step(s)", steps_taken)
                        return self._make_result(
                            plan=plan,
                            steps_taken=steps_taken,
                            message=finish_result.output,
                        )

                if is_repeat_call and finish_call is None:
                    nudge = (
                        "You just repeated the exact same tool call with identical "
                        "arguments as the previous step — it produced no new "
                        "information. Stop repeating it. Either take a concrete "
                        "action that makes progress (e.g. write or edit a file) or "
                        "call `finish` if the task is actually already complete."
                    )
                    self._ui.on_tool_result("repeat-guard", nudge)
                    tool_results.append(nudge)
                    failed_calls += 1

                for result_text in tool_results:
                    self._context.conversation.add_message(
                        Message(role=Role.tool, content=result_text)
                    )

                steps_taken += 1
                action = retry_mgr.record_iteration(total_calls, failed_calls, tool_results)

                if action == RetryAction.STEP_COMPLETED:
                    verify_feedback = await self._verify_and_mark_step(
                        plan, "\n".join(tool_results), has_tool_evidence=True
                    )
                    if verify_feedback:
                        self._ui.on_tool_result("plan-check", verify_feedback)
                        self._context.conversation.add_message(
                            Message(role=Role.tool, content=verify_feedback)
                        )
                elif action == RetryAction.REPLAN:
                    logger.info("Replanning after repeated failures (step %d)", steps_taken)
                    self._ui.on_replan()
                    # Note: the scratchpad is deliberately NOT reset here --
                    # it's the agent's accumulated working memory (decisions,
                    # files touched, assumptions, open questions) and is
                    # exactly what the next attempt needs, not something to
                    # discard just because this attempt failed.
                    plan = await self._planner.update_plan(
                        plan,
                        failure_context="\n".join(tool_results),
                    )
                    retry_mgr.on_replan()
                elif action == RetryAction.ESCALATE:
                    logger.warning("Escalating: persistent failures despite re-planning (step %d)", steps_taken)
                    return self._make_result(
                        plan=plan,
                        steps_taken=steps_taken,
                        message="The task could not be completed despite re-planning. "
                                "The agent encountered persistent failures.",
                        success=False,
                    )

            logger.warning("Max steps (%d) exceeded", self._config.max_steps)
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

    async def _verify_and_mark_step(
        self, plan: Plan, evidence: str, has_tool_evidence: bool = False
    ) -> str | None:
        """Ask the planner to verify the current pending step is actually
        done before advancing it, instead of assuming any non-failing turn
        finished it. Returns feedback to show the model if the step isn't
        really finished yet, or None if it was marked complete."""
        step = plan.current_step()
        if step is None:
            return None

        step.status = StepStatus.in_progress

        if has_tool_evidence and _looks_investigative(step.description):
            # Read/identify/check-style steps have no artifact beyond "the
            # tool ran and returned real output" -- that's sufficient
            # evidence; skip the extra (failure-prone) LLM judgment call.
            logger.debug("Step %r auto-completed (investigative, has tool evidence)", step.description)
            step.status = StepStatus.completed
            return None

        verdict, reason = await self._planner.verify_step(plan, step, evidence)
        logger.debug("Step %r verification verdict: %s (%s)", step.description, verdict.value, reason)
        if verdict == StepVerdict.complete:
            step.status = StepStatus.completed
            return None

        step.status = StepStatus.pending
        return (
            f"Step {step.index} ('{step.description}') is not fully done yet "
            f"({verdict.value}): {reason or 'insufficient evidence in the tool results'}. "
            "Keep working on this step before moving on."
        )

    def _detect_finish(self, content: str) -> bool:
        triggers = [
            "task is complete",
            "i have finished",
            "all done",
            "finished the task",
        ]
        return any(t in content.lower() for t in triggers)

    def _unread_existing_file(self, path: str, files_read: set[str]) -> str | None:
        """Returns the canonical path if `path` refers to an existing file
        that hasn't been read (or written/edited) yet this session, else
        None -- either it's a new file being created, it's already known,
        or the path couldn't be resolved (in which case we don't block;
        the tool itself will report the problem)."""
        if not path:
            return None
        try:
            resolved = self._context.workspace.resolve(path)
        except Exception:
            return None
        key = str(resolved)
        if key in files_read:
            return None
        if not resolved.exists():
            return None
        return key

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
    def _capture_file_content(path: str) -> str | None:
        """Read file content before write/edit for potential rollback."""
        p = Path(path)
        if p.exists():
            try:
                return p.read_text()
            except OSError:
                return None
        return None

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
