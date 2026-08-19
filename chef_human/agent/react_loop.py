from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque


from chef_human.agent.context import ContextAssembler
from chef_human.agent.linter import (
    annotate_diff_with_lint,
    format_lint_result,
    run_lint,
    syntax_error,
)
from chef_human.agent.parser import (
    ParsedToolCall,
    extract_scratchpad_entries,
    format_parse_error,
    looks_like_tool_call,
    parse_native_tool_calls,
    parse_tool_calls,
    strip_scratchpad,
    strip_tool_calls,
    validate_arguments,
)
from chef_human.agent.planner import Plan, PlanNode, Planner, StepStatus, StepVerdict
from chef_human.agent.prompts import build_agent_prompt
from chef_human.agent.retry import RetryAction, RetryManager
from chef_human.agent.scratchpad import Scratchpad
from chef_human.agent.workspace import WorkspaceManager
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
_FILE_MUTATION_VERBS = (
    *_FILE_CREATION_VERBS,
    "implement", "edit", "update", "modify", "change", "refactor",
)
_DOC_LIKE_SUFFIXES = (".md", ".markdown", ".txt", ".rst", ".adoc", ".org")
_EXECUTION_STEP_KEYWORDS = ("run", "test", "verify", "execute", "check")
_DIRECT_READ_PREFIX_RE = re.compile(r"^\s*read\b", re.IGNORECASE)


def _looks_like_file_creation_step(description: str) -> set[str]:
    """Filenames this step names, if the step also reads like "create/write
    a file" -- empty set otherwise. A false negative here just means normal
    LLM verification runs (today's behavior), so this is purely additive,
    never a regression."""
    lowered = description.lower()
    if not any(v in lowered for v in _FILE_CREATION_VERBS):
        return set()
    return {m.group(1) for m in _FILE_TARGET_RE.finditer(description)}


def _looks_like_file_mutation_step(description: str) -> set[str]:
    """Filenames this step names, if the step reads like it should leave a
    durable file artifact behind. Used to prevent reasoning-only turns from
    hallucinating progress on implementation/edit steps."""
    lowered = description.lower()
    if not any(v in lowered for v in _FILE_MUTATION_VERBS):
        return set()
    return {m.group(1) for m in _FILE_TARGET_RE.finditer(description)}


def _named_step_files(description: str) -> list[str]:
    seen: dict[str, None] = {}
    for match in _FILE_TARGET_RE.finditer(description):
        seen[match.group(1)] = None
    return list(seen)


def _read_file_names(files_read: set[str]) -> set[str]:
    """Basenames of the files read (or written/edited) so far this session,
    from the loop's set of resolved workspace paths. Basenames are used
    because step descriptions name files by their short path; if two files
    in different directories share a basename this errs toward treating the
    step as satisfied, which is the safe direction for an investigative
    step (better a loose read than a false-negative re-read loop)."""
    names: set[str] = set()
    for path_str in files_read:
        try:
            names.add(Path(path_str).name)
        except Exception:
            continue
    return names


def _file_facts(
    files_written: dict[str, bool],
    *,
    created_label: str = "newly_created_this_turn",
) -> str:
    """One line per file touched this turn, checked directly against disk
    right now -- not inferred from a tool's success-message wording (which
    is what caused a step like "create a new file named X" to flip between
    complete/not-complete depending on whether the model happened to call
    `write` or `edit` that turn, even though the file's actual content on
    disk was correct either way)."""
    lines = []
    for path_str, existed_before in files_written.items():
        p = Path(path_str)
        exists = p.exists()
        line_count = len(p.read_text(errors="replace").splitlines()) if exists else 0
        lines.append(
            f"- {p.name}: exists={exists}, lines={line_count}, "
            f"{created_label}={exists and not existed_before}"
        )
    return "\n".join(lines)


# How many lines of a file's current content get injected into the step
# verifier's evidence so it can give *specific* feedback (quote the actual
# offending lines) instead of vague language like "duplicate code". Read
# fresh from disk on every verification so the verifier sees the real
# state -- without this it only sees the turn's tool-result diffs and had
# to guess at the current file, which produced generic reasons that sent
# the agent chasing phantom problems (observed: verifier saying "duplicate
# lines need to be removed" about an already-deduplicated file).
_VERIFY_FILE_MAX_LINES = 300


def _step_candidate_files(
    workspace: WorkspaceManager,
    written_paths: list[str],
    named_files: list[str],
) -> list[str]:
    """Resolve the files a step is about to existing paths on disk.
    `written_paths` are already-resolved workspace paths; `named_files` are
    short names from the step description that are resolved against the
    workspace (and dropped if they don't exist -- the filename regex
    over-matches on method references like `Inventory.remove`)."""
    candidates: list[str] = []
    seen: set[str] = set()
    for path_str in written_paths:
        try:
            p = Path(path_str)
            if p.exists():
                key = str(p.resolve())
                if key not in seen:
                    seen.add(key)
                    candidates.append(str(p))
        except Exception:
            continue
    for name in named_files:
        try:
            resolved = workspace.resolve(name)
            if resolved.exists():
                key = str(resolved.resolve())
                if key not in seen:
                    seen.add(key)
                    candidates.append(str(resolved))
        except Exception:
            continue
    return candidates


def _step_file_contents(
    workspace: WorkspaceManager,
    written_paths: list[str],
    named_files: list[str],
    rolled_back_content: dict[str, str] | None = None,
) -> str:
    """Current content of the files a step is about, read from disk right
    now, for the verifier's evidence. `rolled_back_content` maps resolved
    paths to content that was written this turn and then restored by the
    lint-after-write rollback -- for those files the *attempted* content is
    shown instead of the restored pre-write state, so the verifier can cite
    the exact offending lines (the model has to fix what it actually wrote,
    not what the file looked like before). Returns "" when nothing readable
    is found."""
    rolled_back_content = rolled_back_content or {}
    sections: list[str] = []
    for path_str in _step_candidate_files(workspace, written_paths, named_files):
        key = str(Path(path_str).resolve())
        if key in rolled_back_content:
            content = rolled_back_content[key]
            label = " (attempted write that was rolled back after lint errors)"
        else:
            try:
                content = Path(path_str).read_text(errors="replace")
            except Exception:
                continue
            label = ""
        content_lines = content.splitlines()
        truncated = len(content_lines) > _VERIFY_FILE_MAX_LINES
        if truncated:
            content_lines = content_lines[:_VERIFY_FILE_MAX_LINES]
        body = "\n".join(content_lines)
        if truncated:
            body += f"\n... (file truncated at {_VERIFY_FILE_MAX_LINES} lines)"
        sections.append(f"--- {Path(path_str).name}{label} ---\n{body}")
    return "\n\n".join(sections)


def _step_python_syntax_errors(
    workspace: WorkspaceManager,
    written_paths: list[str],
    named_files: list[str],
    rolled_back_content: dict[str, str] | None = None,
) -> list[str]:
    """Deterministic syntax check (compile-based, no external linter needed)
    over the Python files a step is about. Returns ruff-style error lines for
    any file that does not compile. The LLM verifier has proven unreliable at
    noticing broken indentation even when shown the file contents -- a file
    that does not even parse can never be a complete step. For files that
    were rolled back after lint errors, the *attempted* content is checked
    (the on-disk state has been restored and would hide the problem)."""
    rolled_back_content = rolled_back_content or {}
    errors: list[str] = []
    for path_str in _step_candidate_files(workspace, written_paths, named_files):
        if Path(path_str).suffix != ".py":
            continue
        key = str(Path(path_str).resolve())
        if key in rolled_back_content:
            source = rolled_back_content[key]
        else:
            try:
                source = Path(path_str).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
        result = syntax_error(source, path_str)
        if result:
            errors.append(result)
    return errors


def _looks_like_execution_step(description: str) -> bool:
    lowered = description.lower()
    return any(keyword in lowered for keyword in _EXECUTION_STEP_KEYWORDS)


def _execution_step_feedback(step: PlanNode) -> str:
    named_files = _named_step_files(step.description)
    lowered = step.description.lower()
    if named_files and "python" in lowered:
        target = named_files[0]
        return (
            f"Step {step.index} ('{step.description}') still needs real execution evidence. "
            "Reasoning alone does not count for a run/verify step; use the `bash` tool now "
            f"to run `python {target}` and capture the output in this turn."
        )
    if named_files:
        targets = ", ".join(f"`{name}`" for name in named_files)
        return (
            f"Step {step.index} ('{step.description}') still needs real execution evidence. "
            "Reasoning alone does not count for a run/verify step; use a tool like `bash` "
            f"to run or test {targets} in this turn."
        )
    return (
        f"Step {step.index} ('{step.description}') still needs real execution evidence. "
        "Reasoning alone does not count for a run/verify step; use a tool like `bash` to "
        "run the command or test in this turn."
    )


def _file_mutation_step_feedback(step: PlanNode, target_names: set[str]) -> str:
    targets = ", ".join(f"`{name}`" for name in sorted(target_names))
    return (
        f"Step {step.index} ('{step.description}') still needs real file-change evidence. "
        "Reasoning alone does not count for a create/implement/edit step; use a mutating "
        f"tool like `write`/`edit` on {targets} in this turn."
    )


def _primary_mutation_target(step: PlanNode, target_names: set[str]) -> str:
    ordered = [name for name in _named_step_files(step.description) if name in target_names]
    for name in ordered:
        if not name.lower().endswith(_DOC_LIKE_SUFFIXES):
            return name
    if ordered:
        return ordered[0]
    return sorted(target_names)[0]


def _file_mutation_no_tool_feedback(
    step: PlanNode,
    target_names: set[str],
    *,
    consecutive_stalls: int,
) -> str:
    target = _primary_mutation_target(step, target_names)
    example = (
        '<tool_call>{ "name": "write", "arguments": '
        f'{{ "path": "{target}", "content": "..." }}'
        "}</tool_call>"
    )
    prefix = (
        f"Step {step.index} ('{step.description}') still needs real file-change evidence. "
    )
    if consecutive_stalls >= 2:
        return (
            prefix
            + 
            "You already tried reasoning without a tool on this same step. "
            "Do not explain the plan again. Your next response must contain a single "
            f"`write`/`edit` tool call for `{target}`. Example for `{target}`:\n{example}"
        )
    return (
        prefix
        +
        "Reasoning alone does not count for a create/implement/edit step. "
        f"Respond by calling `write`/`edit` on `{target}` now. Example for `{target}`:\n{example}"
    )


def _direct_read_targets(description: str) -> list[str]:
    if not _DIRECT_READ_PREFIX_RE.search(description):
        return []
    seen: dict[str, None] = {}
    for match in _FILE_TARGET_RE.finditer(description):
        seen[match.group(1)] = None
    return list(seen)

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
    r"have you (?:(?:already )?(?:completed|finished|done|run|executed|tested|"
    r"verified|checked|created|written|edited|read|installed|configured))|"
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
class EscalationRecord:
    """A node that exhausted its retry/replan budget mid-run. The default
    fallback (mark failed, keep executing the rest of the tree) is already
    applied by the time this is created -- it exists purely so a human can
    act on the escalation *after* the fact, via `ReActLoop.resolve_escalation`,
    rather than the run blocking on them while it happens. See ticket 06
    (mid-execution-escalation-ux) in `.scratch/planning-tree/issues/`."""

    node_id: str
    description: str
    message: str


@dataclass
class AgentResult:
    plan: Plan
    steps_taken: int
    message: str
    success: bool = True
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    escalations: list[EscalationRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "steps_taken": self.steps_taken,
            "message": self.message,
            "plan": self.plan.to_dict(),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "escalations": [
                {"node_id": e.node_id, "description": e.description, "message": e.message}
                for e in self.escalations
            ],
        }


@dataclass
class StepEvidence:
    # path -> whether the file already existed before the first successful
    # write/edit recorded for this step
    files_written: dict[str, bool] = None
    successful_commands: list[str] = None

    def __post_init__(self) -> None:
        if self.files_written is None:
            self.files_written = {}
        if self.successful_commands is None:
            self.successful_commands = []

    def merge_turn(
        self,
        files_written_this_turn: dict[str, bool],
        successful_commands_this_turn: list[str],
    ) -> None:
        for path_str, existed_before in files_written_this_turn.items():
            previous = self.files_written.get(path_str)
            # Preserve the earliest write fact: once a file was observed as
            # newly created for this step, later rewrites should not erase it.
            self.files_written[path_str] = (
                existed_before if previous is None else previous and existed_before
            )
        for command in successful_commands_this_turn:
            if command and command not in self.successful_commands:
                self.successful_commands.append(command)


@dataclass
class ReActConfig:
    max_steps: int = 25
    # Consecutive failing turns tolerated before the loop gives up on the
    # current attempt and either replans or escalates. Deliberately larger
    # than the old 3: a small local model mid-failure frequently recovers
    # given a couple more turns (each turn's feedback is cheap), and a
    # premature replan burns the single replan budget -- observed spiraling
    # from "blocked write" straight into replan-then-escalate without ever
    # retrying. Tune down for faster, noisier runs; tune up to give the
    # model more room before re-planning.
    max_retries_per_step: int = 5
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
    # Safety cap on `_run_pre_execution_review`'s loop -- see that method's
    # docstring. A real human reviewer will never hit this; it only bounds
    # a misbehaving/un-stubbed UI double.
    _MAX_PLAN_REVIEW_ITERATIONS = 25

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
        self._recent_verification_events: deque[str] = deque(maxlen=8)
        self._planning_facts: dict[str, bool] = {}
        self._step_evidence: dict[str, StepEvidence] = {}
        self._mutation_no_tool_stalls: dict[str, int] = {}
        # The node (leaf or branch) whose verification most recently failed
        # -- set in _verify_and_mark_step/_process_rollups/the mutation-guard
        # block, right before returning failure feedback, so a subsequent
        # REPLAN action can scope replan_subtree to that exact node instead
        # of guessing from plan.current_leaf() (wrong once the failing node
        # is a branch: by rollup-rejection time its leaves already report
        # complete, so current_leaf() has already moved past it).
        self._last_failed_node: PlanNode | None = None
        # Nodes that exhausted their retry/replan budget mid-run, in headless
        # or interactive mode alike -- both apply the same mark-failed-and-
        # continue default (see `_handle_escalation`). Kept so a human can
        # review and act on them retroactively via `resolve_escalation` once
        # the run has surfaced them, rather than the run blocking on a
        # decision no one is there (headless) or ready (interactive) to make
        # the instant it happens.
        self._escalations: list[EscalationRecord] = []
        # RetryManager for the run currently in progress -- lives on the
        # instance (not a `_execute` local) so `resolve_escalation`'s resume
        # reuses it instead of resetting every node's failure/replan counters.
        # Reset to None at the top of every `run()`.
        self._retry_mgr: RetryManager | None = None

    def _record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        self._ui.on_token_usage(prompt_tokens, completion_tokens)

    def _record_verification_event(self, label: str, content: str) -> None:
        text = content.strip()
        if not text:
            return
        self._recent_verification_events.append(f"{label}: {text[:1000]}")

    def _recent_verification_history(self) -> str:
        if not self._recent_verification_events:
            return ""
        return "\n\n".join(self._recent_verification_events)

    def _has_recent_non_finish_evidence(self) -> bool:
        for entry in self._recent_verification_events:
            if entry.startswith("tool-call:finish:") or entry.startswith("finish-rejected:"):
                continue
            return True
        return False

    @staticmethod
    def _step_evidence_key(step: PlanNode) -> str:
        return step.node_id

    _ROOT_RETRY_NODE_ID = "__root__"

    def _retry_node_id(self, node: PlanNode | None) -> str:
        """Resolves the node_id RetryManager should key this turn's outcome
        under. Prefers the node actually being worked this turn; falls back
        to whichever node most recently failed verification, then to a
        stable sentinel for turns with no specific step in play (e.g. the
        plan is already complete and only the deferred `finish` call is
        outstanding)."""
        if node is not None:
            return node.node_id
        if self._last_failed_node is not None:
            return self._last_failed_node.node_id
        return self._ROOT_RETRY_NODE_ID

    def _handle_escalation(
        self, plan: Plan, node: PlanNode | None, steps_taken: int, message: str
    ) -> AgentResult | None:
        """Applies an ESCALATE verdict for `node`. Headless and interactive
        modes apply the identical default fallback: mark the node failed and
        keep executing the rest of the tree -- returns None so the caller
        keeps looping. `ReActLoop` dispatch is already sequential (one leaf
        at a time), so there's no real concurrency for a blocking prompt to
        let something "keep running in the background" past; the default
        already applies before a human could see a prompt anyway. A
        notification is surfaced via `self._ui.on_escalation` and recorded in
        `self._escalations` so the human can act on it retroactively --
        `resolve_escalation` edits/redecomposes/rejects after the fact. Only
        when no specific node can be identified (nothing to mark failed and
        continue past) does the whole run still terminate."""
        target = node or self._last_failed_node
        if target is not None:
            target.status = StepStatus.failed
            logger.warning(
                "Node %s exhausted its retry/replan budget; marking failed "
                "and continuing on the rest of the tree",
                target.node_id,
            )
            record = EscalationRecord(
                node_id=target.node_id, description=target.description, message=message
            )
            self._escalations.append(record)
            self._ui.on_escalation(target, message)
            self._last_failed_node = None
            return None
        return self._make_result(
            plan=plan,
            steps_taken=steps_taken,
            message=message,
            success=False,
        )

    def _step_evidence_for(self, step: PlanNode) -> StepEvidence:
        key = self._step_evidence_key(step)
        evidence = self._step_evidence.get(key)
        if evidence is None:
            evidence = StepEvidence()
            self._step_evidence[key] = evidence
        return evidence

    def _discard_subtree_evidence(self, node: PlanNode) -> None:
        """Ahead of a subtree replan: permanently drop every descendant's
        evidence bucket (their node_ids are about to stop existing in the
        tree) and reset `node`'s own bucket to empty (its goal survives the
        replan, but the evidence for how it was previously attempted does
        not)."""

        def walk(n: PlanNode) -> None:
            for child in n.children:
                self._step_evidence.pop(self._step_evidence_key(child), None)
                self._mutation_no_tool_stalls.pop(self._step_evidence_key(child), None)
                walk(child)

        walk(node)
        self._step_evidence[self._step_evidence_key(node)] = StepEvidence()
        self._mutation_no_tool_stalls.pop(self._step_evidence_key(node), None)

    def _rebuild_ancestor_evidence(self, node: PlanNode) -> None:
        """After a subtree replan, recompute every ancestor above `node`
        from scratch as the union of its *current* descendants' evidence
        buckets -- rather than trying to subtract the discarded entries out
        of the additive propagation, this just recomputes from what's
        actually still in the tree, so no stale propagated entry from a
        discarded attempt can survive in an ancestor's bucket."""
        ancestor = node.parent
        while ancestor is not None:
            rebuilt = StepEvidence()

            def collect(n: PlanNode) -> None:
                entry = self._step_evidence.get(self._step_evidence_key(n))
                if entry is not None:
                    rebuilt.merge_turn(entry.files_written, entry.successful_commands)
                for child in n.children:
                    collect(child)

            for child in ancestor.children:
                collect(child)
            self._step_evidence[self._step_evidence_key(ancestor)] = rebuilt
            ancestor = ancestor.parent

    async def _replan_failing_node(self, plan: Plan, failure_context: str) -> Plan:
        """Scope a replan to whichever node most recently failed
        verification (leaf or rollup), preserving its node_id and touching
        no sibling/ancestor node elsewhere in the tree. Falls back to a
        whole-plan replan only when no specific node can be identified."""
        target = self._last_failed_node or plan.current_leaf()
        if target is None:
            return await self._planner.update_plan(plan, failure_context=failure_context)
        self._discard_subtree_evidence(target)
        target.status = StepStatus.pending
        await self._planner.replan_subtree(plan, target, failure_context)
        self._rebuild_ancestor_evidence(target)
        self._last_failed_node = None
        return plan

    def _note_mutation_no_tool_stall(self, step: PlanNode) -> int:
        key = self._step_evidence_key(step)
        next_count = self._mutation_no_tool_stalls.get(key, 0) + 1
        self._mutation_no_tool_stalls[key] = next_count
        return next_count

    def _clear_mutation_no_tool_stall(self, step: PlanNode) -> None:
        self._mutation_no_tool_stalls.pop(self._step_evidence_key(step), None)

    async def run(self, task: str) -> AgentResult:
        logger.info("Task started: %s", task[:200])
        self._ui.on_start(task)
        self._retry_mgr = None
        plan = await self._plan_task(task)
        logger.info(
            "Plan generated: %d step(s): %s",
            len(plan.steps),
            [s.description for s in plan.steps],
        )
        review_result = await self._run_pre_execution_review(plan)
        if review_result is not None:
            return review_result
        return await self._execute(plan, task)

    async def _run_pre_execution_review(self, plan: Plan) -> AgentResult | None:
        """One-time whole-tree human approval pass, run once after generation
        and before any execution begins -- distinct from `resolve_escalation`
        (ticket 06), which acts retroactively on a node mid/post-execution.
        `self._ui.on_plan_review` defaults (via `review_plan_via_stdin`) to
        immediate approval whenever stdin isn't a tty, so headless/benchmark
        runs never block here. Loops so a reviewer can apply several actions
        (edit/mark-as-leaf/redecompose) before finally approving or
        rejecting; each loop re-shows the tree reflecting prior edits. Capped
        at `_MAX_PLAN_REVIEW_ITERATIONS` iterations so a UI double that
        returns something other than a real `PlanReviewAction` (e.g. an
        un-stubbed `MagicMock(spec=NoopUI)` in a test, whose `.kind`
        auto-generates a mock attribute matching nothing below) can't spin
        this loop forever -- falls back to auto-approve past the cap."""
        for _ in range(self._MAX_PLAN_REVIEW_ITERATIONS):
            # Not a bare `await self._ui.on_plan_review(plan)`: many existing
            # UI test doubles are plain `MagicMock()`/`MagicMock(spec=NoopUI)`
            # instances that predate this method and return a non-awaitable
            # `MagicMock` from it -- `isawaitable` lets a real async UI
            # implementation await normally while a mock double degrades to
            # treating the call's return value directly as the action (which
            # the iteration cap above still bounds even when that's nonsense).
            call_result = self._ui.on_plan_review(plan)
            action = await call_result if inspect.isawaitable(call_result) else call_result
            kind = getattr(action, "kind", None) if action is not None else "approve"
            if kind == "approve":
                return None
            if kind == "reject":
                return self._make_result(
                    plan=plan,
                    steps_taken=0,
                    message="Plan rejected by user before execution began.",
                    success=False,
                )
            if kind not in {"edit", "mark_leaf", "redecompose"}:
                self._ui.on_error(f"Plan review: unrecognized action {kind!r}")
                continue

            node_id = getattr(action, "node_id", None)
            target = plan.find_node(node_id) if node_id else None
            if target is None:
                self._ui.on_error(f"Plan review: no node with id {node_id!r} in this plan")
                continue

            if kind == "edit":
                description = getattr(action, "description", None)
                if description:
                    target.description = description
                target.flagged = False
            elif kind == "mark_leaf":
                target.set_children([])
                target.flagged = False
            elif kind == "redecompose":
                guidance = getattr(action, "guidance", None) or ""
                await self._planner.replan_subtree(
                    plan, target, guidance, is_failure=False
                )
                target.flagged = False

        logger.warning(
            "Pre-execution plan review did not resolve to approve/reject after "
            "%d iterations; auto-approving.",
            self._MAX_PLAN_REVIEW_ITERATIONS,
        )
        return None

    async def resolve_escalation(
        self,
        plan: Plan,
        task: str,
        node_id: str,
        action: str,
        description: str | None = None,
        guidance: str | None = None,
    ) -> AgentResult:
        """Acts, after the fact, on a node recorded in `self._escalations`
        (an ESCALATE verdict that was already resolved via the automatic
        mark-failed-and-continue default -- see `_handle_escalation`). This
        is the human's retroactive override, per ticket 06
        (mid-execution-escalation-ux): `action` is one of "edit" (change the
        node's description and retry it), "redecompose" (ask the planner to
        retry the node's subtree with `guidance`), or "reject" (abort the
        whole run). "edit"/"redecompose" reverse the already-applied default
        by un-marking the node failed and resuming execution from the
        current plan state -- everything else in the tree (other completed
        nodes, their evidence) is untouched.

        `plan` and `task` should be the ones returned by/passed into the
        original `run()` call this escalation came from; the plan is mutated
        and re-executed in place, so passing a different plan than the one
        the escalation actually belongs to would silently look up the wrong
        node."""
        target = plan.find_node(node_id)
        if target is None:
            raise ValueError(f"No node with id {node_id!r} in this plan")
        self._escalations = [e for e in self._escalations if e.node_id != node_id]

        if action == "reject":
            return self._make_result(
                plan=plan,
                steps_taken=0,
                message=f"Run rejected by user at node {node_id} ({target.description!r}).",
                success=False,
            )
        if action == "edit":
            if description is not None:
                target.description = description
            target.status = StepStatus.pending
        elif action == "redecompose":
            self._discard_subtree_evidence(target)
            target.status = StepStatus.pending
            await self._planner.replan_subtree(plan, target, guidance or "")
            self._rebuild_ancestor_evidence(target)
        else:
            raise ValueError(f"Unknown escalation action: {action!r}")

        return await self._execute(plan, task, resume=True)

    async def _execute(self, plan: Plan, task: str, resume: bool = False) -> AgentResult:
        steps_taken = 0
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
        # Persisted on the instance (not a local, as it used to be) so a
        # `resolve_escalation` resume reuses the same RetryManager rather
        # than resetting every node's failure/replan counters back to zero
        # -- see ticket 06. `run()` always starts a fresh one; a resume
        # keeps whatever `run()` already built.
        if self._retry_mgr is None:
            self._retry_mgr = RetryManager(
                max_retries_per_step=self._config.max_retries_per_step,
                max_replans=self._config.max_replans,
            )
        retry_mgr = self._retry_mgr
        # Counts consecutive turns that started with the plan already fully
        # complete (current_leaf() is None) where the model still didn't
        # call `finish`. The prompt already tells it to ("All steps are
        # complete -- call `finish`.") but a weak local model routinely
        # ignores that and just keeps calling arbitrary tools instead --
        # observed repeating the exact same read-only tool call turn after
        # turn once nothing was left to do. Once the *system's own*
        # bookkeeping says the plan is done, we don't need the model's
        # permission to stop -- see the auto-finish check below.
        turns_with_plan_complete_no_finish = 0

        if not resume:
            self._context.conversation.add_message(
                Message(role=Role.user, content=task)
            )

        try:
            while steps_taken < self._config.max_steps:
                current = plan.current_leaf()
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

                tool_calls = (
                    parse_native_tool_calls(response.message.tool_calls)
                    if response.message.tool_calls
                    else parse_tool_calls(response.message.content)
                )
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
                    auto_tool_results: list[str] = []
                    auto_read_success = False
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
                        current = plan.current_leaf()
                        # Investigative steps are excluded from the
                        # reasoning-only mutation stall guard: a step like
                        # "explore ... the current implementation of
                        # inventory.py" contains the substring "implement"
                        # (via "implementation") and would otherwise be
                        # misrouted into "call write/edit on inventory.py"
                        # feedback -- it's a read step, not a write step.
                        mutation_targets = (
                            _looks_like_file_mutation_step(current.description)
                            if current is not None
                            and not _looks_investigative(current.description)
                            else set()
                        )
                        direct_read_targets = (
                            _direct_read_targets(current.description)
                            if current is not None
                            else []
                        )
                        if direct_read_targets:
                            read_tool = self._tools.get("read")
                            if read_tool is not None:
                                for target in direct_read_targets:
                                    try:
                                        read_result = await asyncio.wait_for(
                                            read_tool.run(path=target),
                                            timeout=self._config.tool_timeout,
                                        )
                                    except Exception as exc:
                                        read_output = self._make_tool_error(
                                            f"Execution error: {exc}"
                                        )
                                    else:
                                        if read_result.success:
                                            read_output = read_result.output
                                            auto_read_success = True
                                            try:
                                                files_read.add(
                                                    str(self._context.workspace.resolve(target))
                                                )
                                            except Exception:
                                                pass
                                        else:
                                            read_output = (
                                                f"Error: {read_result.error}\n"
                                                f"Output: {read_result.output}"
                                            )
                                    self._ui.on_tool_result("read", read_output)
                                    auto_tool_results.append(read_output)
                                    self._context.conversation.add_message(
                                        Message(role=Role.tool, content=read_output)
                                    )

                        if current is not None and mutation_targets:
                            self._last_failed_node = current
                            stall_count = self._note_mutation_no_tool_stall(current)
                            mutation_feedback = _file_mutation_no_tool_feedback(
                                current,
                                mutation_targets,
                                consecutive_stalls=stall_count,
                            )
                            self._ui.on_tool_result("mutation-guard", mutation_feedback)
                            self._context.conversation.add_message(
                                Message(role=Role.tool, content=mutation_feedback)
                            )
                            verify_failure_history.append(mutation_feedback)
                            action = retry_mgr.record_iteration(
                                current.node_id, 1, 1, [mutation_feedback]
                            )
                            steps_taken += 1
                            if action == RetryAction.REPLAN:
                                self._ui.on_replan()
                                plan = await self._replan_failing_node(
                                    plan,
                                    "\n".join(verify_failure_history),
                                )
                                retry_mgr.on_replan(current.node_id)
                                verify_failure_history.clear()
                            elif action == RetryAction.ESCALATE:
                                result = self._handle_escalation(
                                    plan,
                                    current,
                                    steps_taken,
                                    "The task could not be completed because step "
                                    "verification repeatedly failed.",
                                )
                                if result is not None:
                                    return result
                            continue

                    steps_taken += 1
                    node_id = self._retry_node_id(current)
                    if parse_error:
                        action = retry_mgr.record_iteration(node_id, 1, 1, [parse_error])
                    else:
                        verify_feedback = await self._verify_and_mark_step(
                            plan,
                            "\n".join(auto_tool_results) if auto_tool_results else non_tool_reasoning,
                            has_tool_evidence=auto_read_success,
                            files_read=files_read,
                        )
                        if verify_feedback:
                            self._ui.on_tool_result("plan-check", verify_feedback)
                            self._context.conversation.add_message(
                                Message(role=Role.tool, content=verify_feedback)
                            )
                            verify_failure_history.append(verify_feedback)
                            action = retry_mgr.record_iteration(node_id, 1, 1, [verify_feedback])
                        else:
                            verify_failure_history.clear()
                            action = retry_mgr.record_iteration(node_id, 0, 0, [])

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
                        plan = await self._replan_failing_node(
                            plan,
                            "\n".join(verify_failure_history),
                        )
                        retry_mgr.on_replan(node_id)
                        verify_failure_history.clear()
                    elif action == RetryAction.ESCALATE:
                        result = self._handle_escalation(
                            plan,
                            current,
                            steps_taken,
                            "The task could not be completed because step "
                            "verification repeatedly failed.",
                        )
                        if result is not None:
                            return result
                    continue

                # Interactive ask_user calls are excluded because a repeated
                # question can receive a genuinely new answer. In autonomous
                # mode no answer can arrive, so include them and let the repeat
                # guard detect a stalled model.
                _signature_calls = [
                    tc
                    for tc in tool_calls
                    if tc.name != "ask_user" or self._config.disable_ask_user
                ]
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
                # resolved path -> content the model wrote this turn that the
                # lint-after-write rollback restored. Threaded into step
                # verification so the verifier still sees the attempted
                # (broken) content and the syntax check catches it.
                rolled_back_content: dict[str, str] = {}
                successful_commands_this_turn: list[str] = []
                finish_call: tuple[ParsedToolCall, Tool] | None = None
                deferred_finish_call: tuple[ParsedToolCall, Tool] | None = None
                parallel_candidates: list[tuple[ParsedToolCall, Tool]] = []
                current = plan.current_leaf()
                # Captured now, before `current` can be reassigned below (the
                # deferred-finish branch reassigns it to an unresolved step
                # purely for messaging) -- this is the node whose retry
                # pressure this turn's outcome should count against.
                retry_node = current
                node_id = self._retry_node_id(retry_node)
                if current is not None and _looks_like_file_mutation_step(current.description):
                    self._clear_mutation_no_tool_stall(current)

                for tc in tool_calls:
                    logger.debug("Tool call: %s(%s)", tc.name, tc.arguments)
                    self._ui.on_tool_call(tc)
                    self._record_verification_event(
                        f"tool-call:{tc.name}",
                        json.dumps(tc.arguments, sort_keys=True, default=str),
                    )

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
                        current = plan.current_leaf()
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
                            failed_calls += 1
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
                        if answer.startswith("[no-tty]"):
                            failed_calls += 1
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

                if (
                    finish_call is not None
                    and self._config.require_plan_complete_to_finish
                    and plan.unresolved_steps()
                ):
                    if parallel_candidates:
                        # A model may bundle substantive work and finish in
                        # one response. Run and verify the work before judging
                        # the terminal request; rejecting finish now would make
                        # the whole turn a partial failure and skip verification.
                        deferred_finish_call = finish_call
                        finish_call = None
                        logger.debug(
                            "Deferring finish until %d substantive tool call(s) "
                            "have run and the current step has been verified",
                            len(parallel_candidates),
                        )
                    else:
                        logger.info("Finish requested with unresolved step(s); asking verifier first")

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
                            if tc.name == "bash":
                                successful_commands_this_turn.append(
                                    str(tc.arguments.get("command", ""))
                                )

                        self._ui.on_tool_result(tc.name, result)
                        tool_results.append(result)
                        self._record_verification_event(f"tool:{tc.name}", result)

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
                                    # Capture what the model actually wrote so
                                    # the verifier can still cite the exact
                                    # offending lines after the restore.
                                    try:
                                        broken_content = Path(resolved_file).read_text(errors="replace")
                                    except Exception:
                                        broken_content = None
                                    if broken_content is not None:
                                        rolled_back_content[str(resolved_file)] = broken_content
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
                                # Append lint output with rollback note. This is
                                # *not* counted as a failed_calls: verification
                                # below still runs so the model gets the specific
                                # syntax feedback and the step cannot slip through
                                # as completed. (failed_calls > 0 skips that
                                # whole path, which is what turned a rolled-back
                                # write into a silent repeat loop in the
                                # inventory benchmark.)
                                lint_result = format_lint_result(lint_output) + rollback_msg
                                tool_results.append(lint_result)

                    logger.debug(
                        "Dispatched %d ordered tool call(s) in %.1fs: %s",
                        len(parallel_candidates),
                        time.monotonic() - dispatch_start,
                        [tc.name for tc, _ in parallel_candidates],
                    )

                if finish_call is not None:
                    tc, tool = finish_call
                    unresolved = plan.unresolved_steps()
                    if (
                        self._config.require_plan_complete_to_finish
                        and unresolved
                    ):
                        finish_summary = str(tc.arguments.get("summary", ""))
                        finish_feedback = await self._verify_finish_request(
                            plan,
                            immediate_evidence="\n".join(tool_results),
                            finish_summary=finish_summary,
                            files_written_this_turn=files_written_this_turn,
                            successful_commands_this_turn=successful_commands_this_turn,
                            files_read=files_read,
                            rolled_back_content=rolled_back_content,
                        )
                        if finish_feedback is not None:
                            self._ui.on_tool_result("finish", finish_feedback)
                            tool_results.append(finish_feedback)
                            self._record_verification_event("finish-rejected", finish_feedback)
                            failed_calls += 1
                            finish_call = None
                    if finish_call is not None:
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
                    self._record_verification_event("tool-result", result_text)

                steps_taken += 1
                if failed_calls == 0:
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
                        successful_commands_this_turn=successful_commands_this_turn,
                        files_read=files_read,
                        rolled_back_content=rolled_back_content,
                    )
                    if verify_feedback:
                        self._ui.on_tool_result("plan-check", verify_feedback)
                        self._context.conversation.add_message(
                            Message(role=Role.tool, content=verify_feedback)
                        )
                        verify_failure_history.append(verify_feedback)
                        action = retry_mgr.record_iteration(node_id, 1, 1, [verify_feedback])
                    else:
                        verify_failure_history.clear()
                        action = retry_mgr.record_iteration(
                            node_id, total_calls, failed_calls, tool_results
                        )
                else:
                    action = retry_mgr.record_iteration(
                        node_id, total_calls, failed_calls, tool_results
                    )

                if deferred_finish_call is not None:
                    unresolved = plan.unresolved_steps()
                    if unresolved:
                        current = unresolved[0]
                        logger.info(
                            "Deferred finish remains premature: unfinished step %r remains",
                            current.description,
                        )
                        result = self._make_tool_error(
                            self._unfinished_plan_message(current.description)
                        )
                        self._ui.on_tool_result("finish", result)
                        self._context.conversation.add_message(
                            Message(role=Role.tool, content=result)
                        )
                    else:
                        tc, tool = deferred_finish_call
                        try:
                            finish_result = await asyncio.wait_for(
                                tool.run(**tc.arguments),
                                timeout=self._config.tool_timeout,
                            )
                        except asyncio.TimeoutError:
                            result = self._make_tool_error(
                                f"Tool '{tc.name}' timed out after "
                                f"{self._config.tool_timeout}s"
                            )
                            self._ui.on_tool_result(tc.name, result)
                            self._context.conversation.add_message(
                                Message(role=Role.tool, content=result)
                            )
                            action = retry_mgr.record_iteration(node_id, 1, 1, [result])
                        except Exception as exc:
                            result = self._make_tool_error(f"Execution error: {exc}")
                            self._ui.on_tool_result(tc.name, result)
                            self._context.conversation.add_message(
                                Message(role=Role.tool, content=result)
                            )
                            action = retry_mgr.record_iteration(node_id, 1, 1, [result])
                        else:
                            finish_msg = (
                                finish_result.output
                                if finish_result.success
                                else finish_result.error or ""
                            )
                            self._ui.on_tool_result(tc.name, finish_msg)
                            logger.info(
                                "Task finished via deferred finish tool after %d step(s)",
                                steps_taken,
                            )
                            return self._make_result(
                                plan=plan,
                                steps_taken=steps_taken,
                                message=finish_result.output,
                            )

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
                        else "\n".join(retry_mgr.tool_results(node_id))
                    )
                    # Note: the scratchpad is deliberately NOT reset here --
                    # it's the agent's accumulated working memory (decisions,
                    # files touched, assumptions, open questions) and is
                    # exactly what the next attempt needs, not something to
                    # discard just because this attempt failed.
                    plan = await self._replan_failing_node(plan, failure_context)
                    retry_mgr.on_replan(node_id)
                    verify_failure_history.clear()
                elif action == RetryAction.ESCALATE:
                    logger.warning("Escalating: persistent failures despite re-planning (step %d)", steps_taken)
                    result = self._handle_escalation(
                        plan,
                        retry_node,
                        steps_taken,
                        "The task could not be completed despite re-planning. "
                        "The agent encountered persistent failures.",
                    )
                    if result is not None:
                        return result

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
        planning_facts = self._collect_planning_facts(task, doc_context)
        self._planning_facts = dict(planning_facts)
        if planning_facts:
            facts_text = "\n".join(
                f"- {path}: {'exists' if exists else 'missing'}"
                for path, exists in sorted(planning_facts.items())
            )
            planning_context = f"### Planning Facts\n\n{facts_text}"
            repo_context = (
                f"{repo_context}\n\n{planning_context}" if repo_context else planning_context
            )
        plan = await self._planner.generate_plan(
            task,
            repo_context=repo_context,
            planning_facts=planning_facts,
        )
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

    def _collect_planning_facts(
        self,
        task: str,
        doc_context: str = "",
        *,
        max_files: int = 16,
    ) -> dict[str, bool]:
        """Resolve simple file-existence questions before planning.

        This avoids plans that defer obvious conditions into vague steps like
        "create X if it does not exist" or "inspect X (if any)" when the
        workspace state is already cheaply knowable."""
        workspace = self._context.workspace
        candidates: dict[str, None] = {}
        for source in (task, doc_context):
            for match in _FILE_TARGET_RE.finditer(source):
                candidates[match.group(1)] = None
                if len(candidates) >= max_files:
                    break
            if len(candidates) >= max_files:
                break

        facts: dict[str, bool] = {}
        for filename in candidates:
            try:
                facts[filename] = bool(workspace.resolve(filename).exists())
            except Exception:
                continue
        return facts

    async def _process_rollups(self, plan: Plan) -> str | None:
        """After a leaf is marked complete, walk any ancestor branches whose
        children are now all complete and run a rollup verification on each
        before marking the branch itself complete -- cascading upward as
        long as each rollup passes. A branch whose children are all
        individually complete can still fail its own rollup (a flawed
        decomposition can leave the sub-goal uncovered); on the first
        rejection, stop and return feedback for the model instead of
        continuing to roll up higher branches that depend on it."""
        while True:
            ready = plan.ready_rollup_branches()
            if not ready:
                self._last_failed_node = None
                return None
            branch = ready[0]
            evidence = self._rollup_evidence(branch)
            children_summary = "\n".join(
                f"- {child.description}: {child.status.value}"
                + (f" -- {child.last_verdict_reason}" if child.last_verdict_reason else "")
                for child in branch.children
            )
            try:
                verdict, reason = await self._planner.verify_rollup(
                    plan, branch, evidence, children_summary=children_summary
                )
            except Exception as exc:
                logger.warning("Branch rollup verification failed with an exception: %s", exc)
                # Repoint at this branch, not whatever leaf/branch was last
                # recorded (e.g. the leaf that just completed and triggered
                # this rollup) -- otherwise a subsequent REPLAN would target
                # the wrong node and discard already-correct work.
                self._last_failed_node = branch
                return (
                    f"Branch '{branch.description}' could not be verified due to an "
                    f"error ({exc}); it will be re-checked next turn."
                )
            logger.debug(
                "Branch %r rollup verdict: %s (%s)", branch.description, verdict.value, reason
            )
            if verdict != StepVerdict.complete:
                branch.last_verdict_reason = reason
                self._last_failed_node = branch
                return (
                    f"Branch '{branch.description}' is not fully done yet ({verdict.value}): "
                    f"{reason or 'insufficient evidence that the sub-goal was achieved'}. "
                    "All of its individual steps were marked complete, but the sub-goal as a "
                    "whole is not -- the decomposition may have missed something; keep "
                    "working on this part of the plan."
                )
            branch.status = StepStatus.completed
            branch.last_verdict_reason = reason
            # This branch is done -- clear the stale reference so a later
            # rollup failure higher in the tree (or the exception path
            # above, on a *different* branch) doesn't get attributed back
            # to this now-completed one.
            self._last_failed_node = None

    def _rollup_evidence(self, branch: PlanNode) -> str:
        """Ground-truth evidence for a branch's own sub-goal: current
        contents of any files its description names, read directly from
        disk -- the same "current file contents are ground truth" framing
        leaf verification uses, not inferred from any child's tool-result
        wording."""
        named_files = _named_step_files(branch.description)
        contents = _step_file_contents(
            self._context.workspace, written_paths=[], named_files=named_files
        )
        if contents:
            return f"Current file contents (read directly from disk for verification):\n{contents}"
        return ""

    async def _verify_and_mark_step(
        self,
        plan: Plan,
        evidence: str,
        has_tool_evidence: bool = False,
        files_written_this_turn: dict[str, bool] | None = None,
        successful_commands_this_turn: list[str] | None = None,
        finish_summary: str = "",
        step_override: PlanNode | None = None,
        files_read: set[str] | None = None,
        rolled_back_content: dict[str, str] | None = None,
    ) -> str | None:
        """Ask the planner to verify the current pending step is actually
        done before advancing it, instead of assuming any non-failing turn
        finished it. Returns feedback to show the model if the step isn't
        really finished yet, or None if it was marked complete."""
        step = step_override or plan.current_leaf()
        if step is None:
            return None

        step.status = StepStatus.in_progress
        # Set unconditionally, ahead of every failure-feedback return point
        # below (including the exception path) -- corrected to point at a
        # branch instead if _process_rollups goes on to reject one, or
        # cleared if verification (leaf and every ready rollup) fully
        # succeeds. See the attribute's docstring in __init__.
        self._last_failed_node = step
        files_written_this_turn = files_written_this_turn or {}
        successful_commands_this_turn = successful_commands_this_turn or []
        files_read = files_read or set()
        step_evidence = self._step_evidence_for(step)
        step_evidence.merge_turn(files_written_this_turn, successful_commands_this_turn)
        # Propagate upward at write time so every ancestor branch's bucket
        # always reflects everything that happened under it, without a
        # separate read-time aggregation step at rollup-verification time.
        ancestor = step.parent
        while ancestor is not None:
            self._step_evidence_for(ancestor).merge_turn(
                files_written_this_turn, successful_commands_this_turn
            )
            ancestor = ancestor.parent
        accumulated_files_written = step_evidence.files_written
        accumulated_commands = step_evidence.successful_commands

        # Deterministic guard: a step whose involved .py file does not even
        # parse can never be complete. The LLM verifier has been observed
        # marking a syntactically-broken file "complete" (run 4 of the
        # inventory benchmark: an IndentationError at column 0 got a complete
        # verdict), so the verifier's verdict is overridden below when this
        # check fails. The attempted (rolled-back) content is still shown to
        # the verifier so it can produce a specific fix directive.
        syntax_errors = _step_python_syntax_errors(
            self._context.workspace,
            written_paths=list(accumulated_files_written),
            named_files=_named_step_files(step.description),
            rolled_back_content=rolled_back_content,
        )

        # When a step names specific files (e.g. "Explore inventory.py and
        # test_inventory.py"), auto-completing on *any* read-only tool evidence
        # is too loose -- the model can read a different file and the step
        # gets marked done without ever looking at the ones it named. For
        # such steps, check against the files actually read this session and
        # route through the LLM verifier (with objective read facts) instead
        # of auto-completing. Steps that name no specific file still
        # auto-complete on any real read-only evidence -- there's nothing
        # specific to validate, and the extra LLM judgment is what caused the
        # re-read loop (see _INVESTIGATIVE_KEYWORDS).
        investigative_named_files: list[str] = []
        investigative_read_names: set[str] = set()
        investigative_needs_verifier = False

        if _looks_investigative(step.description):
            investigative_named_files = _named_step_files(step.description)
            if investigative_named_files:
                investigative_read_names = _read_file_names(files_read)
                # The filename regex over-matches on method/attribute
                # references in step descriptions (e.g. "Inventory.add" in
                # "Identify the missing methods `Inventory.add` and
                # `Inventory.remove` in inventory.py"). Only enforce reads
                # for names that correspond to real files on disk (or files
                # already read this session); requiring a read of a
                # non-existent "file" is an impossible loop.
                existing_named_files = []
                for name in investigative_named_files:
                    if name in investigative_read_names:
                        existing_named_files.append(name)
                        continue
                    try:
                        if self._context.workspace.resolve(name).exists():
                            existing_named_files.append(name)
                    except Exception:
                        continue
                investigative_named_files = existing_named_files
            if investigative_named_files:
                unread = [
                    name
                    for name in investigative_named_files
                    if name not in investigative_read_names
                ]
                if unread:
                    step.status = StepStatus.pending
                    targets = ", ".join(f"`{name}`" for name in unread)
                    return (
                        f"Step {step.index} ('{step.description}') named {targets}, but "
                        f"{'that file has' if len(unread) == 1 else 'those files have'} "
                        "not been read yet this session; the step still needs real tool "
                        "evidence. Use the `read` tool on "
                        f"{targets} in this turn -- reading other files does not "
                        "satisfy this step."
                    )
                if not has_tool_evidence:
                    # The named files were read at some earlier point this
                    # session, but this turn produced no read-only tool call --
                    # reasoning alone still isn't fresh evidence.
                    step.status = StepStatus.pending
                    targets = ", ".join(f"`{name}`" for name in investigative_named_files)
                    return (
                        f"Step {step.index} ('{step.description}') still needs real tool "
                        "evidence. Reasoning alone does not count for an investigative "
                        f"step; use a read-only tool like `read`/`grep`/`ls` on {targets} "
                        "in this turn."
                    )
                # All named files were read and this turn produced real tool
                # evidence: validate via the verifier instead of auto-completing.
                investigative_needs_verifier = True
            elif has_tool_evidence or (
                _looks_like_execution_step(step.description)
                and successful_commands_this_turn
            ):
                # Generic read/identify/check-style step (no specific files
                # named) has no artifact beyond "the tool ran and returned
                # real output" -- that's sufficient evidence; skip the extra
                # (failure-prone) LLM judgment call. A description like "run
                # the tests to identify which one is failing" matches both
                # this branch (via "identify") and _looks_like_execution_step
                # (via "run"/"test"), but bash isn't in
                # _INVESTIGATIVE_TOOL_NAMES -- without the added clause, a
                # step phrased that way rejects every turn that actually
                # reruns the tests, no matter how many times it passes, until
                # the retry budget escalates (seen reproducibly on the
                # scroll_grid_navigation_repair benchmark case). Only widens
                # this already-unconditional "no named files" fallback, and
                # only for steps whose wording independently reads as
                # execution too -- a purely investigative step gets no new
                # leniency, and the named-files branch above is untouched.
                logger.debug(
                    "Step %r auto-completed (investigative, has tool evidence)",
                    step.description,
                )
                step.status = StepStatus.completed
                step.last_verdict_reason = "auto-completed: investigative step had tool evidence"
                return await self._process_rollups(plan)
            else:
                step.status = StepStatus.pending
                return (
                    f"Step {step.index} ('{step.description}') still needs real tool "
                    "evidence. Reasoning alone does not count for an investigative "
                    "step; use a relevant read-only tool in this turn."
                )

        target_names = _looks_like_file_creation_step(step.description)
        if target_names and not investigative_needs_verifier:
            for target_name in target_names:
                initially_existed = self._planning_facts.get(target_name)
                if initially_existed is not False:
                    continue
                try:
                    target_path = self._context.workspace.resolve(target_name)
                except Exception:
                    continue
                if target_path.exists() and target_path.stat().st_size > 0:
                    logger.debug(
                        "Step %r auto-completed (objective: %s was missing at plan time and now exists on disk)",
                        step.description,
                        target_name,
                    )
                    step.status = StepStatus.completed
                    step.last_verdict_reason = f"auto-completed: {target_name} now exists on disk"
                    return await self._process_rollups(plan)
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
                    step.last_verdict_reason = f"auto-completed: {path_str} exists on disk"
                    return await self._process_rollups(plan)

        mutation_targets = _looks_like_file_mutation_step(step.description)
        if (
            mutation_targets
            and not accumulated_files_written
            and not investigative_needs_verifier
        ):
            step.status = StepStatus.pending
            return _file_mutation_step_feedback(step, mutation_targets)

        if (
            _looks_like_execution_step(step.description)
            and successful_commands_this_turn
            and not investigative_needs_verifier
        ):
            logger.debug(
                "Step %r auto-completed (objective: successful command(s) %s)",
                step.description,
                successful_commands_this_turn,
            )
            step.status = StepStatus.completed
            step.last_verdict_reason = (
                f"auto-completed: successful command(s) {successful_commands_this_turn}"
            )
            return await self._process_rollups(plan)

        if (
            _looks_like_execution_step(step.description)
            and not has_tool_evidence
            and not investigative_needs_verifier
        ):
            step.status = StepStatus.pending
            return _execution_step_feedback(step)

        evidence_sections: list[str] = []
        if investigative_needs_verifier:
            evidence_sections.append(
                "Files read so far this session (checked directly by the system):\n"
                + "\n".join(f"- {name}" for name in sorted(investigative_read_names))
            )
        accumulated_facts = _file_facts(
            accumulated_files_written,
            created_label="created_during_step",
        )
        if accumulated_facts:
            evidence_sections.append(
                "Objective facts (checked directly against disk, accumulated across "
                f"turns for this step):\n{accumulated_facts}"
            )
        if accumulated_commands:
            commands_text = "\n".join(f"- {command}" for command in accumulated_commands)
            evidence_sections.append(
                "Successful commands accumulated for this step:\n"
                f"{commands_text}"
            )
        step_contents = _step_file_contents(
            self._context.workspace,
            written_paths=list(accumulated_files_written),
            named_files=_named_step_files(step.description),
            rolled_back_content=rolled_back_content,
        )
        if step_contents:
            evidence_sections.append(
                "Current file contents (read directly from disk for verification):\n"
                + step_contents
            )
        if syntax_errors:
            evidence_sections.append(
                "Deterministic syntax check (the system compiled these files "
                "directly; a file listed here does not parse, so the step cannot "
                "be verified complete regardless of the model's judgment):\n"
                + "\n".join(syntax_errors)
            )
        if evidence_sections:
            evidence = "\n\n".join(
                [
                    *evidence_sections,
                    f"Immediate evidence from this turn:\n{evidence}",
                ]
            )

        try:
            logger.debug("VERIFIER_DEBUG evidence sent:\n%s", evidence)
            logger.debug(
                "VERIFIER_DEBUG recent_history:\n%s", self._recent_verification_history()
            )
            logger.debug("VERIFIER_DEBUG finish_summary:\n%s", finish_summary)
            verdict, reason = await self._planner.verify_step(
                plan,
                step,
                evidence,
                recent_history=self._recent_verification_history(),
                finish_summary=finish_summary,
            )
        except Exception as exc:
            # Unlike tool-call dispatch (which wraps every call in
            # try/except so a backend hiccup becomes a recorded failure,
            # not a crash), this LLM call had no such guard -- a transient
            # error here used to propagate uncaught out of run(), aborting
            # the whole task, while also leaving the step stuck
            # `in_progress` (current_leaf() only matches `pending`, so it'd
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
        if verdict == StepVerdict.complete and syntax_errors:
            # Deterministic override: the verifier (and the model) may judge a
            # broken file "complete"; a file that does not even parse cannot be.
            step.status = StepStatus.pending
            details = "; ".join(syntax_errors)
            return (
                f"Step {step.index} ('{step.description}') is not fully done yet: a "
                f"Python file this step involves does not parse (checked directly, "
                f"not inferred): {details}. Fix the syntax error before this step "
                "can be considered done."
            )
        if verdict == StepVerdict.complete:
            step.status = StepStatus.completed
            step.last_verdict_reason = reason
            return await self._process_rollups(plan)

        step.status = StepStatus.pending
        return (
            f"Step {step.index} ('{step.description}') is not fully done yet "
            f"({verdict.value}): {reason or 'insufficient evidence in the tool results'}. "
            "Keep working on this step before moving on."
        )

    async def _verify_finish_request(
        self,
        plan: Plan,
        *,
        immediate_evidence: str,
        finish_summary: str,
        files_written_this_turn: dict[str, bool],
        successful_commands_this_turn: list[str],
        files_read: set[str] | None = None,
        rolled_back_content: dict[str, str] | None = None,
    ) -> str | None:
        unresolved = plan.unresolved_steps()
        if not unresolved:
            return None

        if (
            not immediate_evidence.strip()
            and not files_written_this_turn
            and not successful_commands_this_turn
            and not self._has_recent_non_finish_evidence()
        ):
            current = unresolved[0]
            return self._unfinished_plan_message(current.description)

        for step in unresolved:
            original_status = step.status
            try:
                step.status = StepStatus.pending
                feedback = await self._verify_and_mark_step(
                    plan,
                    immediate_evidence,
                    has_tool_evidence=False,
                    files_written_this_turn=files_written_this_turn,
                    successful_commands_this_turn=successful_commands_this_turn,
                    finish_summary=finish_summary,
                    step_override=step,
                    files_read=files_read,
                    rolled_back_content=rolled_back_content,
                )
                if feedback is not None:
                    return (
                        "Finish request rejected after verification. "
                        f"{feedback}"
                    )
            finally:
                if step.status != StepStatus.completed:
                    step.status = original_status
        return None

    def _detect_finish(self, content: str) -> bool:
        triggers = [
            "task is complete",
            "i have finished",
            "all done",
            "finished the task",
        ]
        return any(t in content.lower() for t in triggers)

    @staticmethod
    def _unfinished_plan_message(description: str) -> str:
        return (
            "You cannot finish yet -- there's an unfinished plan step: "
            f"'{description}'. Complete it (and any remaining steps) before "
            "calling finish. A tool result claiming work is done is not evidence "
            "-- you must actually perform the step (e.g. write/edit the relevant "
            "files)."
        )

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
            escalations=list(self._escalations),
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
