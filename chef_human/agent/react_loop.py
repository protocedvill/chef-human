from __future__ import annotations

import asyncio
import json
import logging
import re
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
    CompletionResponse,
    LLMBackend,
    Message,
    Role,
)
from chef_human.tools.registry import (
    TOOL_POLICIES,
    ReadRequirement,
    Tool,
    ToolRegistry,
    get_tool_policy,
)
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

# Read-only tools that can actually produce "the codebase was investigated"
# evidence. The auto-complete bypass below is only safe when the turn's
# tool calls include at least one of these -- otherwise a step whose
# description merely *sounds* investigative (matches _INVESTIGATIVE_KEYWORDS)
# could be auto-completed by any unrelated successful turn, e.g. a turn
# whose only tool call was `ask_user` (which "succeeds" and produces no
# evidence about the codebase at all).
_INVESTIGATIVE_TOOL_NAMES = (
    "read", "ls", "ls_tree", "grep", "glob",
    "lookup_symbol", "goto_definition", "find_references", "view_diff",
)

# Matches a filename token a step description might name, e.g. "Create a
# new file named 'hello_world.py'" -> "hello_world.py".
_FILE_TARGET_RE = re.compile(r"[`'\"]?([\w\-./]+\.\w{1,10})[`'\"]?")

# If a step's description contains one of these verbs *and* names a
# specific file, "is that file created" is objectively checkable (does it
# exist on disk, is it non-empty) -- no LLM judgment needed. Steps that
# merely mention a filename without one of these (e.g. "Test that
# hello_world.py runs correctly") still need real judgment, so they're
# deliberately excluded.
_FILE_CREATION_VERBS = ("create", "write", "make", "add", "generate")


def _looks_like_file_creation_step(description: str) -> set[str]:
    """Filenames this step names, if the step also reads like "create/write
    a file" -- empty set otherwise. A false negative here just means normal
    LLM verification runs (today's behavior), so this is purely additive,
    never a regression."""
    lowered = description.lower()
    if not any(v in lowered for v in _FILE_CREATION_VERBS):
        return set()
    return {m.group(1) for m in _FILE_TARGET_RE.finditer(description)}


def _file_facts(files_written_this_turn: dict[str, bool]) -> str:
    """One line per file touched this turn, checked directly against disk
    right now -- not inferred from a tool's success-message wording (which
    is what caused a step like "create a new file named X" to flip between
    complete/not-complete depending on whether the model happened to call
    `write` or `edit` that turn, even though the file's actual content on
    disk was correct either way)."""
    lines = []
    for path_str, existed_before in files_written_this_turn.items():
        p = Path(path_str)
        exists = p.exists()
        line_count = len(p.read_text(errors="replace").splitlines()) if exists else 0
        lines.append(
            f"- {p.name}: exists={exists}, lines={line_count}, "
            f"newly_created_this_turn={exists and not existed_before}"
        )
    return "\n".join(lines)

# Matches doc-like filenames mentioned in a task string (e.g. "implement
# plan.md") so their content can be read and fed to the planner -- without
# this, the planner only ever sees the task text plus a directory tree, and
# has no way to know what a referenced design/plan document actually says.
_DOC_FILENAME_RE = re.compile(
    r"\b[\w\-./\\]+\.(?:md|markdown|txt|rst|adoc|org)\b", re.IGNORECASE
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

# Yes/no permission-, confirmation-, or status-checking phrasing for a step
# that's already scheduled in the plan -- these aren't design decisions,
# they're stalling. A genuine design question offers a choice ("which X",
# "what should this be called", "how should this handle Y"); these ask for
# approval to do (or confirmation that someone else already did) what's
# already been planned. Unanchored (not just at the start of the question)
# since real phrasing routinely prefixes these with context, e.g. "The file
# wasn't found. Do you want me to create it?" or "Which tool would you like
# me to use next?".
_PERMISSION_SEEKING_RE = re.compile(
    r"\b(?:do you want|would you like|should i|can i|could i|may i|shall i|"
    r"have you (?:completed|finished|already done)|"
    r"is it (?:ok(?:ay)?|fine|alright) (?:to|if)|is that (?:ok(?:ay)?|fine))\b",
    re.IGNORECASE,
)


def _rollback_file(path: Path, content: str) -> None:
    """Restore a file to its pre-write content."""
    path.write_text(content)


def _looks_investigative(description: str) -> bool:
    lowered = description.lower()
    return any(kw in lowered for kw in _INVESTIGATIVE_KEYWORDS)


# Phrases indicating the model itself believes it deliberately introduced a
# security problem -- self-reported, not detected by scanning code, since
# that's the failure mode observed: the model wrote a hardcoded-credential
# check and then wrote a `finish` summary calling it out as intentional.
_SELF_REPORTED_VULNERABILITY_RE = re.compile(
    r"\bintentional(?:ly)?\b[^.]{0,80}\b"
    r"(vulnerab\w*|insecur\w*|backdoor\w*|hardcoded credential\w*|security (?:flaw|issue|bug)\w*)\b",
    re.IGNORECASE,
)


def _looks_like_self_reported_vulnerability(summary: str) -> str | None:
    match = _SELF_REPORTED_VULNERABILITY_RE.search(summary)
    return match.group(0) if match else None


# Tools whose success can change the repo's file layout -- the cached
# repo map (see ContextAssembler.invalidate_repo_map_cache) is only stale
# after one of these actually succeeds. Public (no leading underscore) so
# UI implementations can reuse it too, e.g. to know when to refresh a file
# tree widget -- see TuiUI.on_tool_result.
FILE_MUTATING_TOOLS = tuple(
    name for name, policy in TOOL_POLICIES.items() if policy.mutates
)


def _is_vague_next_step_question(question: str) -> bool:
    q = question.lower().strip().rstrip("?")
    return any(p in q for p in _VAGUE_ASK_USER_PATTERNS)


def _is_low_value_ask_user_question(question: str) -> bool:
    """True for "what next"-style vagueness or yes/no permission-seeking
    about a step the plan has already scheduled -- neither is a genuine
    design decision worth interrupting the user for."""
    q = question.strip()
    return _is_vague_next_step_question(q) or bool(_PERMISSION_SEEKING_RE.search(q))


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
    # When on, ask_user calls are answered immediately with an instruction
    # to proceed on the agent's own judgment instead of prompting the user
    # -- for a "hands off, let it guess" run.
    disable_ask_user: bool = False


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
        self._planner.on_llm_start = self._ui.on_llm_start
        self._planner.on_llm_end = self._ui.on_llm_end

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
        # Accumulates step-verification rejection reasons across retries of
        # the *same* step, separately from RetryManager.tool_results -- the
        # first record_iteration() call each turn resets tool_results
        # whenever the turn's tool calls succeeded, which would otherwise
        # wipe out prior verify-failure reasons before a later REPLAN could
        # see them.
        verify_failure_history: list[str] = []
        retry_mgr = RetryManager(
            max_retries_per_step=self._config.max_retries_per_step,
            max_replans=self._config.max_replans,
        )
        # Counts consecutive turns that started with the plan already fully
        # complete (current_step() is None) where the model still didn't
        # call `finish`. The prompt already tells it to ("All steps are
        # complete -- call `finish`.") but a weak local model routinely
        # ignores that and just keeps calling arbitrary tools instead --
        # observed repeating the exact same read-only tool call turn after
        # turn once nothing was left to do. Once the *system's own*
        # bookkeeping says the plan is done, we don't need the model's
        # permission to stop -- see the auto-finish check below.
        turns_with_plan_complete_no_finish = 0

        self._context.conversation.add_message(
            Message(role=Role.user, content=task)
        )

        try:
            while steps_taken < self._config.max_steps:
                current = plan.current_step()
                plan_was_complete_at_turn_start = plan.is_complete()
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
                self._ui.on_llm_start("reasoning")
                llm_start = time.monotonic()
                logger.debug("LLM call starting (stream=%s, %d messages)", self._config.stream, len(messages))
                response: CompletionResponse | None = None
                try:
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
                        if response is None:
                            raise RuntimeError("LLM stream ended without a final response")
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
                finally:
                    self._ui.on_llm_end()
                if response is None:
                    raise RuntimeError("LLM stream ended without a final response")
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
                            verify_failure_history.append(verify_feedback)
                            action = retry_mgr.record_iteration(1, 1, [verify_feedback])
                        else:
                            verify_failure_history.clear()

                    if (
                        self._detect_finish(non_tool_reasoning)
                        and not parse_error
                        and plan.is_complete()
                    ):
                        logger.info("Task finished via finish-phrase detection after %d step(s)", steps_taken)
                        return self._make_result(
                            plan=plan,
                            steps_taken=steps_taken,
                            message=non_tool_reasoning,
                        )
                    if action == RetryAction.REPLAN:
                        self._ui.on_replan()
                        plan = await self._planner.update_plan(
                            plan,
                            failure_context="\n".join(verify_failure_history),
                        )
                        retry_mgr.on_replan()
                        verify_failure_history.clear()
                    elif action == RetryAction.ESCALATE:
                        return self._make_result(
                            plan=plan,
                            steps_taken=steps_taken,
                            message=(
                                "The task could not be completed because step "
                                "verification repeatedly failed."
                            ),
                            success=False,
                        )
                    continue

                # ask_user is excluded from the repeat signature -- it's not
                # a no-op action even when the question text repeats, since
                # each call round-trips to the user and can get a genuinely
                # new (if unhelpful) answer. Without this, a legitimately
                # re-asked clarifying question gets misidentified as a
                # stalled repeat and triggers the "stop repeating tool
                # calls" nudge below.
                _signature_calls = [tc for tc in tool_calls if tc.name != "ask_user"]
                call_signature = (
                    json.dumps(
                        sorted(
                            ((tc.name, tc.arguments) for tc in _signature_calls),
                            key=lambda item: item[0],
                        ),
                        sort_keys=True,
                        default=str,
                    )
                    if _signature_calls
                    else None
                )
                is_repeat_call = call_signature is not None and call_signature == last_call_signature
                last_call_signature = call_signature

                total_calls = len(tool_calls)
                failed_calls = 0
                tool_results: list[str] = []
                # resolved path -> did the file exist before this turn's write/
                # edit/patch call ran. Used to objectively verify file-creation
                # steps instead of relying on a tool's success-message wording.
                files_written_this_turn: dict[str, bool] = {}
                finish_call: tuple[ParsedToolCall, Tool] | None = None
                parallel_candidates: list[tuple[ParsedToolCall, Tool]] = []

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
                            unresolved = plan.unresolved_steps()
                            if unresolved:
                                current = unresolved[0]
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
                        flagged = _looks_like_self_reported_vulnerability(
                            str(tc.arguments.get("summary", ""))
                        )
                        if flagged:
                            logger.warning(
                                "Blocked finish: summary self-reports an intentional "
                                "vulnerability: %r", flagged
                            )
                            result = self._make_tool_error(
                                "You cannot finish yet -- your summary describes "
                                f"intentionally introducing a security problem ({flagged!r}). "
                                "Do not write deliberately insecure code (hardcoded "
                                "credentials, injection flaws, backdoors, etc.) unless the "
                                "user explicitly asked for exactly that. Fix the issue "
                                "properly, then call finish again."
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

                        if self._config.disable_ask_user:
                            answer = (
                                "Auto mode is enabled -- ask_user is disabled for this "
                                "session. Make your own best-guess decision and proceed; "
                                "do not ask again."
                            )
                            logger.info("ask_user suppressed (auto mode): %r", question[:150])
                            self._ui.on_tool_result(tc.name, answer)
                            tool_results.append(answer)
                            continue

                        if (
                            self._config.block_vague_ask_user
                            and current is not None
                            and _is_low_value_ask_user_question(question)
                        ):
                            logger.info("Blocked low-value ask_user: %r", question[:150])
                            result = self._make_tool_error(
                                "Don't ask what to do next, or for permission/confirmation "
                                "to do the current plan step -- there's an active plan step: "
                                f"'{current.description}', and it's already authorized. Just "
                                "do it. Only use ask_user for a genuine design decision you "
                                "can't resolve yourself -- e.g. a real choice between two "
                                "valid approaches, a naming/schema choice, or an ambiguous "
                                "requirement -- not to ask permission or 'is this ok?'."
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

                    parallel_candidates.append((tc, tool))

                if parallel_candidates:
                    # Capture original file content for write/edit/patch tools --
                    # used both for lint-failure rollback and (unconditionally,
                    # not just when lint_after_write is on) to know whether a
                    # file existed *before* this turn, for the objective
                    # file-creation facts below.
                    pre_write_content: dict[str, str | None] = {}
                    dispatch_start = time.monotonic()
                    # Preserve model order. This makes read+edit in one turn
                    # well-defined and prevents two mutating calls from
                    # racing on the same path. Tool-level async work remains
                    # asynchronous; only a single response batch is
                    # serialized at this safety boundary.
                    for tc, tool in parallel_candidates:
                        policy = get_tool_policy(tc.name)
                        if (
                            self._config.require_read_before_edit
                            and policy.read_requirement == ReadRequirement.explicit_path
                            and policy.path_argument is not None
                        ):
                            path = str(tc.arguments.get(policy.path_argument, ""))
                            read_key = self._unread_existing_file(path, files_read)
                            if read_key is not None:
                                logger.info("Blocked %s of unread file: %r", tc.name, read_key)
                                result = self._make_tool_error(
                                    f"You haven't read '{read_key}' yet this session. Read it "
                                    "first with the `read` tool. Calls in one model response "
                                    "run in order, so `read` immediately followed by this "
                                    "mutation is allowed."
                                )
                                self._ui.on_tool_result(tc.name, result)
                                tool_results.append(result)
                                failed_calls += 1
                                continue

                        if tc.name in ("write", "edit", "patch"):
                            write_path = str(tc.arguments.get("path", ""))
                            pre_write_content[write_path] = self._capture_file_content(
                                write_path
                            )

                        try:
                            tool_result = await asyncio.wait_for(
                                tool.run(**tc.arguments),
                                timeout=self._config.tool_timeout,
                            )
                        except Exception as exc:
                            result = self._make_tool_error(f"Execution error: {exc}")
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

                        if tool_result.success and tc.name in FILE_MUTATING_TOOLS:
                            self._context.invalidate_repo_map_cache()

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

                        if tool_result.success and tc.name in ("write", "edit", "patch"):
                            written_path = tc.arguments.get("path", "")
                            if written_path:
                                try:
                                    resolved_written = str(self._context.workspace.resolve(written_path))
                                    files_written_this_turn[resolved_written] = (
                                        pre_write_content.get(written_path) is not None
                                    )
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
                                resolved_file = self._context.workspace.resolve(file_path)
                                if original is not None:
                                    _rollback_file(resolved_file, original)
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

                    logger.debug(
                        "Dispatched %d ordered tool call(s) in %.1fs: %s",
                        len(parallel_candidates),
                        time.monotonic() - dispatch_start,
                        [tc.name for tc, _ in parallel_candidates],
                    )

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
                    # This nudge is an extra synthetic failure on top of
                    # whatever the turn's real tool calls did -- bump
                    # total_calls to match, so failed_calls can never exceed
                    # it and wrongly make retry.py's failed_calls < total_calls
                    # PARTIAL_SUCCESS check unreachable for a turn that had a
                    # genuinely succeeding call alongside the repeat.
                    total_calls += 1
                    failed_calls += 1

                if (
                    plan_was_complete_at_turn_start
                    and finish_call is None
                    # Only counts turns that didn't fail -- if the model is
                    # busy failing/retrying against a broken tool call, let
                    # the normal retry/replan/escalate machinery handle
                    # that; this check is specifically for "the model keeps
                    # doing things that work, just never finish".
                    and failed_calls == 0
                ):
                    turns_with_plan_complete_no_finish += 1
                    if turns_with_plan_complete_no_finish >= 2:
                        logger.info(
                            "All plan steps were already complete but the model "
                            "didn't call finish for %d turn(s) in a row -- "
                            "finishing on its behalf.",
                            turns_with_plan_complete_no_finish,
                        )
                        return self._make_result(
                            plan=plan,
                            steps_taken=steps_taken,
                            message=(
                                "All plan steps were completed. The agent kept "
                                "calling tools instead of finish afterwards, so "
                                "the task was concluded automatically."
                            ),
                        )
                else:
                    turns_with_plan_complete_no_finish = 0

                for result_text in tool_results:
                    self._context.conversation.add_message(
                        Message(role=Role.tool, content=result_text)
                    )

                steps_taken += 1
                action = retry_mgr.record_iteration(total_calls, failed_calls, tool_results)

                if action == RetryAction.STEP_COMPLETED:
                    # A turn "succeeding" only means none of its tool calls
                    # failed -- it says nothing about whether any of them
                    # actually investigated anything. Require at least one
                    # genuinely read-only tool call this turn before letting
                    # the investigative-step bypass fire (see
                    # _INVESTIGATIVE_TOOL_NAMES), otherwise e.g. a turn
                    # whose only call was `ask_user` could silently
                    # auto-complete a "read the codebase" step that was
                    # never actually worked on.
                    has_real_investigative_evidence = any(
                        tc.name in _INVESTIGATIVE_TOOL_NAMES for tc in tool_calls
                    )
                    verify_feedback = await self._verify_and_mark_step(
                        plan,
                        "\n".join(tool_results),
                        has_tool_evidence=has_real_investigative_evidence,
                        files_written_this_turn=files_written_this_turn,
                    )
                    if verify_feedback:
                        self._ui.on_tool_result("plan-check", verify_feedback)
                        self._context.conversation.add_message(
                            Message(role=Role.tool, content=verify_feedback)
                        )
                        verify_failure_history.append(verify_feedback)
                        # A non-investigative step whose tool calls keep
                        # "succeeding" but keeps failing LLM verification
                        # would otherwise reset to pending forever -- feed
                        # it into the same failure counter that drives
                        # replanning/escalation for actual tool failures,
                        # so it's bounded by max_retries_per_step too.
                        action = retry_mgr.record_iteration(1, 1, [verify_feedback])
                    else:
                        verify_failure_history.clear()

                if action == RetryAction.REPLAN:
                    logger.info("Replanning after repeated failures (step %d)", steps_taken)
                    self._ui.on_replan()
                    # Prefer the accumulated step-verification rejection
                    # reasons when that's what triggered this replan -- the
                    # tool calls themselves looked successful (that's why
                    # verification ran at all), so raw tool_results here
                    # would tell the planner nothing about why the step was
                    # actually rejected. Otherwise fall back to
                    # RetryManager's own accumulated failure history across
                    # this step's retries, not just this turn's tool_results.
                    failure_context = (
                        "\n".join(verify_failure_history)
                        if verify_failure_history
                        else "\n".join(retry_mgr.tool_results)
                    )
                    # Note: the scratchpad is deliberately NOT reset here --
                    # it's the agent's accumulated working memory (decisions,
                    # files touched, assumptions, open questions) and is
                    # exactly what the next attempt needs, not something to
                    # discard just because this attempt failed.
                    plan = await self._planner.update_plan(
                        plan,
                        failure_context=failure_context,
                    )
                    retry_mgr.on_replan()
                    verify_failure_history.clear()
                elif action == RetryAction.ESCALATE:
                    logger.warning("Escalating: persistent failures despite re-planning (step %d)", steps_taken)
                    return self._make_result(
                        plan=plan,
                        steps_taken=steps_taken,
                        message="The task could not be completed despite re-planning. "
                                "The agent encountered persistent failures.",
                        success=False,
                    )

                self._ui.on_plan_progress(plan)

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
        doc_context = self._get_referenced_document_contents(task)
        if doc_context:
            repo_context = f"{repo_context}\n\n{doc_context}" if repo_context else doc_context
        plan = await self._planner.generate_plan(task, repo_context=repo_context)
        self._ui.on_plan(plan)
        return plan

    def _get_repo_context(self) -> str:
        try:
            tree = self._context._repo_map.generate_tree()
            return tree[:1000]
        except Exception:
            return ""

    def _get_referenced_document_contents(
        self, task: str, max_chars_per_file: int = 4000
    ) -> str:
        """Read the contents of any doc-like file (plan.md, spec.txt, ...)
        named in the task string, so the planner sees what the document
        actually says instead of guessing from its filename. Without this,
        a task like "implement the program described in plan.md" is planned
        purely from those words, and the model has been observed to
        hallucinate unrelated steps (e.g. planning a "hello_world.py" for a
        doc that actually describes a ledger program)."""
        workspace = self._context.workspace
        sections: list[str] = []
        for match in _DOC_FILENAME_RE.finditer(task):
            filename = match.group(0)
            try:
                path = workspace.resolve(filename)
                if not path.is_file():
                    continue
                content = path.read_text(errors="replace")[:max_chars_per_file]
            except Exception:
                continue
            sections.append(f"### Contents of {filename}\n\n{content}")
        return "\n\n".join(sections)

    async def _verify_and_mark_step(
        self,
        plan: Plan,
        evidence: str,
        has_tool_evidence: bool = False,
        files_written_this_turn: dict[str, bool] | None = None,
    ) -> str | None:
        """Ask the planner to verify the current pending step is actually
        done before advancing it, instead of assuming any non-failing turn
        finished it. Returns feedback to show the model if the step isn't
        really finished yet, or None if it was marked complete."""
        step = plan.current_step()
        if step is None:
            return None

        step.status = StepStatus.in_progress
        files_written_this_turn = files_written_this_turn or {}

        if has_tool_evidence and _looks_investigative(step.description):
            # Read/identify/check-style steps have no artifact beyond "the
            # tool ran and returned real output" -- that's sufficient
            # evidence; skip the extra (failure-prone) LLM judgment call.
            logger.debug("Step %r auto-completed (investigative, has tool evidence)", step.description)
            step.status = StepStatus.completed
            return None

        target_names = _looks_like_file_creation_step(step.description)
        if target_names:
            for path_str in files_written_this_turn:
                p = Path(path_str)
                if p.name in target_names and p.exists() and p.stat().st_size > 0:
                    # Objectively checked against disk, right now -- not
                    # inferred from a tool's success-message wording (which
                    # is what caused this exact step type to flip-flop
                    # between complete/not-complete depending on whether
                    # the model called `write` or `edit` that turn).
                    logger.debug(
                        "Step %r auto-completed (objective: %s exists on disk)",
                        step.description, path_str,
                    )
                    step.status = StepStatus.completed
                    return None

        facts = _file_facts(files_written_this_turn)
        if facts:
            evidence = (
                "Objective facts (checked directly against disk, not from "
                f"tool output text):\n{facts}\n\nEvidence from this turn:\n{evidence}"
            )

        try:
            verdict, reason = await self._planner.verify_step(plan, step, evidence)
        except Exception as exc:
            # Unlike tool-call dispatch (which wraps every call in
            # try/except so a backend hiccup becomes a recorded failure,
            # not a crash), this LLM call had no such guard -- a transient
            # error here used to propagate uncaught out of run(), aborting
            # the whole task, while also leaving the step stuck
            # `in_progress` (current_step() only matches `pending`, so it'd
            # be silently skipped by any future call). Treat it like a
            # failed verification instead: give the model feedback and let
            # the normal retry/replan machinery handle it.
            logger.warning("Step verification failed with an exception: %s", exc)
            step.status = StepStatus.pending
            return (
                f"Step {step.index} ('{step.description}') could not be verified "
                f"due to an error ({exc}). Keep working on this step; it will be "
                "re-checked next turn."
            )
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

    def _capture_file_content(self, path: str) -> str | None:
        """Read file content before write/edit for potential rollback."""
        try:
            p = self._context.workspace.resolve(path)
        except Exception:
            return None
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
