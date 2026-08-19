from __future__ import annotations

import ast
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from chef_human.agent.prompts import (
    PLANNER_SYSTEM_PROMPT,
    build_rollup_verify_prompt,
    build_verify_prompt,
)
from chef_human.llm.backend import (
    CompletionRequest,
    CompletionResponse,
    LLMBackend,
    Message,
    Role,
)

logger = logging.getLogger(__name__)

# Sentinel distinct from any valid parsed value (including None), used by
# Planner._loads_step_data to signal "neither JSON nor Python-literal parsing
# worked" without colliding with a legitimately parsed `null`/`None`.
_UNPARSEABLE = object()

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


class _ConvergenceTracker:
    """Purely observational per-subtree instrumentation for decomposition
    that isn't converging to leaves quickly. Tracks max expansion depth and
    total node count reached while generating one subtree (one
    `generate_plan`/`replan_subtree` call) and logs a single warning the
    first time either crosses its threshold -- never stops or alters
    generation, and is entirely separate from `RetryManager`'s per-node
    failure/replan counters (decomposition has no notion of "failure")."""

    def __init__(self, subtree_label: str) -> None:
        self._subtree_label = subtree_label
        self._max_depth = 0
        self._node_count = 0
        self._warned = False

    def record(self, *, depth: int, new_nodes: int) -> None:
        self._max_depth = max(self._max_depth, depth)
        self._node_count += new_nodes
        if self._warned:
            return
        if (
            self._max_depth >= Planner._SLOW_CONVERGENCE_DEPTH
            or self._node_count >= Planner._SLOW_CONVERGENCE_NODE_COUNT
        ):
            self._warned = True
            logger.warning(
                "Decomposition of subtree %r has not converged to leaves after "
                "depth=%d, node_count=%d (thresholds: depth>=%d, nodes>=%d) -- "
                "generation is continuing unmodified; this is observational only.",
                self._subtree_label,
                self._max_depth,
                self._node_count,
                Planner._SLOW_CONVERGENCE_DEPTH,
                Planner._SLOW_CONVERGENCE_NODE_COUNT,
            )


@dataclass
class PlanNode:
    """A node in a recursive plan tree.

    Identity is a stable `node_id` (assigned once) rather than description
    text or position -- evidence in `ReActLoop` is now keyed by `node_id`,
    not by description string (see CLAUDE.md for the bug that keying used to
    cause). `update_plan()` still only carries a node forward across a
    replan for steps already `StepStatus.completed`; a replan that reworks
    the wording of a still-pending/in-progress step gets a fresh `node_id`
    and an empty evidence bucket exactly as before this change -- fixing
    that is out of scope for this ticket (see the "Known, not-yet-fixed
    follow-up" note in CLAUDE.md and the per-node-evidence-and-replan-scope
    ticket in `.scratch/planning-tree-adr/`).
    `index` is purely a cosmetic display ordinal, not an identity.

    A node with no children is a leaf and must correspond to exactly one
    tool call; a node with children is a branch. Only depth-1 trees (a root
    with only leaf children) are produced today.
    """

    description: str
    status: StepStatus = StepStatus.pending
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    index: int = 0
    children: list["PlanNode"] = field(default_factory=list)
    # Generation-time hint only (like `index`, not identity, not persisted
    # via `to_dict()`): whether the LLM asked for this node to be
    # decomposed further. Consulted once, right after the node is created,
    # to decide whether to issue a further expansion call for it -- a node
    # is only actually a leaf/branch based on whether `children` ends up
    # populated, not based on this flag.
    requested_branch: bool = False
    # Not identity, not persisted -- set by `set_children()` whenever a node
    # gets children, so rollup verification can walk from a just-completed
    # leaf up to its ancestor branches. Excluded from `__eq__`/`repr` since
    # it would otherwise make the dataclass-generated equality/repr recurse
    # through parent<->child cycles.
    parent: "PlanNode | None" = field(default=None, repr=False, compare=False)
    # Reason text from this node's own last verification (leaf step verifier
    # or branch rollup verifier), kept so an ancestor's rollup check can show
    # each child's verdict/reason as supporting context. Not identity, not
    # persisted.
    last_verdict_reason: str = field(default="", repr=False, compare=False)
    # Generation-time hint (like `requested_branch`, not identity, not
    # persisted): the planner marked this node as one it was uncertain how
    # to decompose. Surfaced inline in the pre-execution whole-tree review
    # pass (ticket 07) so a human reviewer knows where to look; cleared when
    # a review action resolves the node (edit, mark-as-leaf, redecompose).
    flagged: bool = field(default=False, repr=False, compare=False)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def set_children(self, children: list["PlanNode"]) -> None:
        """Assigns `children` and links each child's `.parent` back to this
        node -- the only way `children` should be assigned once a node needs
        rollup verification to walk parent pointers."""
        self.children = children
        for child in children:
            child.parent = self

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "description": self.description,
            "status": self.status.value,
        }


class Plan:
    """The outer container: a goal plus a root `PlanNode`.

    `steps` is kept as a property (not a plain field) over the root's
    children, both so `Plan(goal=..., steps=[...])` construction still works
    the way flat-plan callers expect, and so `plan.steps` stays valid --
    today's trees are always depth-1 (a root with only leaf children), and
    that coincidence is what makes `steps` and `root.children` the same
    list."""

    def __init__(
        self,
        goal: str,
        steps: list[PlanNode] | None = None,
        root: PlanNode | None = None,
    ) -> None:
        self.goal = goal
        self.root = root if root is not None else PlanNode(description="root")
        if steps is not None:
            self.root.set_children(steps)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Plan):
            return NotImplemented
        return self.goal == other.goal and self.root == other.root

    def to_dict(self) -> dict:
        """Serializes as the same {goal, steps} shape flat plans used --
        preserves the `chef-human run --headless --json` output contract.
        Only depth-1 trees are produced today, so this is lossless; a real
        decomposition ticket will need to widen this shape."""
        return {
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
        }

    @property
    def steps(self) -> list[PlanNode]:
        """The root's immediate children. Only depth-1 trees are produced
        today, so this coincides with the flat leaf list; a future
        real-decomposition ticket will need to stop relying on that."""
        return self.root.children

    @steps.setter
    def steps(self, value: list[PlanNode]) -> None:
        self.root.set_children(value)

    def _leaves(self) -> list[PlanNode]:
        """Every leaf node, in DFS pre-order."""
        leaves: list[PlanNode] = []

        def walk(node: PlanNode) -> None:
            if node.is_leaf:
                if node is not self.root:
                    leaves.append(node)
                return
            for child in node.children:
                walk(child)

        walk(self.root)
        return leaves

    def current_leaf(self) -> PlanNode | None:
        """The leaf that should be worked on right now: the first leaf (DFS
        pre-order) that isn't yet completed. Returns None once every leaf is
        completed."""
        return next((n for n in self._leaves() if n.status == StepStatus.pending), None)

    def unresolved_steps(self) -> list[PlanNode]:
        """Leaves that prevent a plan from being reported as complete."""
        return [n for n in self._leaves() if n.status != StepStatus.completed]

    def _branches(self) -> list[PlanNode]:
        """Every non-root node with children, post-order -- deepest branches
        first, so a caller processing this list in order naturally resolves
        a branch's rollup before its parent's. The root itself is excluded:
        it represents the plan as a whole, which is already gated by
        `unresolved_steps()`/the finish flow, not by branch rollup."""
        branches: list[PlanNode] = []

        def walk(node: PlanNode) -> None:
            for child in node.children:
                walk(child)
            if not node.is_leaf and node is not self.root:
                branches.append(node)

        walk(self.root)
        return branches

    def ready_rollup_branches(self) -> list[PlanNode]:
        """Branches whose children are all complete but that haven't
        themselves passed rollup verification yet, post-order. A tree shape
        alone is never proof of completion -- each of these still needs a
        rollup verification call before it can be marked complete."""
        return [
            branch
            for branch in self._branches()
            if branch.status != StepStatus.completed
            and all(child.status == StepStatus.completed for child in branch.children)
        ]

    def find_node(self, node_id: str) -> PlanNode | None:
        """Looks up a node anywhere in the tree by its stable `node_id`,
        including the root and branches -- used by retroactive escalation
        handling (`ReActLoop.resolve_escalation`), which only has a node_id
        recorded from an earlier turn, not a live reference into this
        particular `Plan` instance."""
        if self.root.node_id == node_id:
            return self.root

        def walk(node: PlanNode) -> PlanNode | None:
            for child in node.children:
                if child.node_id == node_id:
                    return child
                found = walk(child)
                if found is not None:
                    return found
            return None

        return walk(self.root)

    def is_complete(self) -> bool:
        return not self.unresolved_steps() and all(
            branch.status == StepStatus.completed for branch in self._branches()
        )


class Planner:
    """Generates and updates structured plans for the ReAct loop."""

    # Purely observational: a subtree that keeps requesting further
    # decomposition past this many expansion levels (or this many
    # descendant nodes) is logged as slow-converging. Never changes
    # generation behavior -- no forced leaf, no hard stop -- and is
    # deliberately independent of RetryManager's per-node failure/replan
    # counters, since decomposition has no notion of "failure" and is a
    # distinct phase from execution.
    _SLOW_CONVERGENCE_DEPTH = 5
    _SLOW_CONVERGENCE_NODE_COUNT = 25

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
        plan = Plan(goal=task)
        await self._expand_node(
            plan.root,
            ancestors=[],
            task=task,
            repo_context=repo_context,
            planning_facts=planning_facts,
            is_root=True,
            depth=0,
            convergence=_ConvergenceTracker(subtree_label=task),
        )
        return plan

    async def _expand_node(
        self,
        node: PlanNode,
        *,
        ancestors: list[str],
        task: str,
        repo_context: str,
        planning_facts: dict[str, bool] | None,
        is_root: bool,
        depth: int,
        convergence: "_ConvergenceTracker",
    ) -> None:
        """Decompose `node` into its immediate children via one LLM call,
        then recurse into any child the LLM marked as a branch. `ancestors`
        is the chain of descriptions from the root goal down through every
        ancestor to (but not including) `node` itself, so a deeply nested
        expansion call doesn't drift from the overall task. There is no hard
        depth cap here -- convergence relies on the LLM eventually emitting
        a leaf for every child; `convergence` is purely observational
        instrumentation logging a warning if that doesn't happen quickly,
        never a behavior change."""
        messages = self._build_expand_messages(
            task=task,
            repo_context=repo_context,
            ancestors=ancestors,
            node=node,
            is_root=is_root,
        )
        response = await self._complete(
            CompletionRequest(messages=messages, temperature=0.0, max_tokens=2048),
            activity="planning",
        )
        children = self._normalize_steps(
            task,
            self._parse_steps(response.message.content),
            planning_facts=planning_facts,
        )
        node.set_children(children)
        convergence.record(depth=depth + 1, new_nodes=len(children))

        child_ancestors = ancestors if is_root else ancestors + [node.description]
        for child in children:
            if child.requested_branch:
                await self._expand_node(
                    child,
                    ancestors=child_ancestors,
                    task=task,
                    repo_context=repo_context,
                    planning_facts=planning_facts,
                    is_root=False,
                    depth=depth + 1,
                    convergence=convergence,
                )

    @staticmethod
    def _build_expand_messages(
        *,
        task: str,
        repo_context: str,
        ancestors: list[str],
        node: PlanNode,
        is_root: bool,
    ) -> list[Message]:
        messages = [Message(role=Role.system, content=PLANNER_SYSTEM_PROMPT)]
        if repo_context:
            messages.append(
                Message(role=Role.system, content=f"## Project Context\n\n{repo_context}")
            )
        if is_root:
            messages.append(Message(role=Role.user, content=f"Task: {task}"))
            return messages

        chain = "\n".join(f"- {d}" for d in ([task] + ancestors))
        messages.append(
            Message(
                role=Role.user,
                content=(
                    f"Overall goal: {task}\n\n"
                    f"Ancestor chain (root goal down to this sub-goal):\n{chain}\n\n"
                    "Break the following sub-goal down into its own immediate next steps "
                    "(do not re-plan the overall goal, only this sub-goal):\n"
                    f"{node.description}"
                ),
            )
        )
        return messages

    async def replan_subtree(
        self,
        plan: Plan,
        node: PlanNode,
        failure_context: str,
        *,
        is_failure: bool = True,
    ) -> None:
        """Regenerate `node`'s own children -- either in response to a
        failure (the leaf itself failing verification, or a branch getting
        rejected by rollup verification; `is_failure=True`, the default) or
        proactively, steered by human guidance from the pre-execution
        whole-tree review pass on a node that was never executed
        (`is_failure=False`, see `ReActLoop._run_pre_execution_review`'s
        "redecompose" action) -- the prompt framing differs accordingly so
        the LLM isn't told a never-run node "failed". `node` keeps its own
        `node_id` (its goal is unchanged, only how to achieve it is
        reconsidered); only its descendants are discarded and replaced. The
        replan call only ever sees/produces `node`'s own descendants --
        siblings and ancestors elsewhere in the tree are untouched by this
        call."""
        ancestors = self._ancestor_descriptions(node)
        messages = self._build_replan_messages(
            goal=plan.goal,
            ancestors=ancestors,
            node=node,
            failure_context=failure_context,
            is_failure=is_failure,
        )
        response = await self._complete(
            CompletionRequest(messages=messages, temperature=0.0, max_tokens=2048),
            activity="replanning",
        )
        children = self._normalize_steps(
            plan.goal, self._parse_steps(response.message.content)
        )
        node.set_children(children)
        convergence = _ConvergenceTracker(subtree_label=node.description)
        convergence.record(depth=1, new_nodes=len(children))

        child_ancestors = ancestors + [node.description]
        for child in children:
            if child.requested_branch:
                await self._expand_node(
                    child,
                    ancestors=child_ancestors,
                    task=plan.goal,
                    repo_context="",
                    planning_facts=None,
                    is_root=False,
                    depth=1,
                    convergence=convergence,
                )

    @staticmethod
    def _ancestor_descriptions(node: PlanNode) -> list[str]:
        """The chain of descriptions from (but not including) the true root
        down through every ancestor to (but not including) `node` itself."""
        chain: list[str] = []
        ancestor = node.parent
        while ancestor is not None and ancestor.parent is not None:
            chain.append(ancestor.description)
            ancestor = ancestor.parent
        chain.reverse()
        return chain

    @staticmethod
    def _build_replan_messages(
        *,
        goal: str,
        ancestors: list[str],
        node: PlanNode,
        failure_context: str,
        is_failure: bool = True,
    ) -> list[Message]:
        if is_failure:
            system_suffix = (
                "\n\nA part of the plan failed. Revise just this sub-goal's own "
                "next steps; do not touch anything outside this sub-goal."
            )
            situation_line = "This sub-goal needs to be redone (its previous attempt failed):"
            context_label = "Failure context"
        else:
            system_suffix = (
                "\n\nA human reviewer wants this sub-goal decomposed differently before "
                "execution starts (it has not been attempted yet -- nothing failed). "
                "Revise just this sub-goal's own next steps, steered by their guidance; "
                "do not touch anything outside this sub-goal."
            )
            situation_line = (
                "This sub-goal has not been executed yet; a human reviewer asked for a "
                "different decomposition before execution starts:"
            )
            context_label = "Reviewer guidance"

        messages = [
            Message(role=Role.system, content=PLANNER_SYSTEM_PROMPT + system_suffix)
        ]
        chain = "\n".join(f"- {d}" for d in ([goal] + ancestors))
        messages.append(
            Message(
                role=Role.user,
                content=(
                    f"Overall goal: {goal}\n\n"
                    f"Ancestor chain (root goal down to this sub-goal):\n{chain}\n\n"
                    f"{situation_line}\n"
                    f"{node.description}\n\n"
                    f"{context_label}:\n{failure_context}\n\n"
                    "Break this sub-goal down into fresh immediate next steps "
                    "(do not re-plan anything outside this sub-goal)."
                ),
            )
        )
        return messages

    async def verify_step(
        self,
        plan: Plan,
        step: PlanNode,
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
        return await self._verdict_from_prompt(prompt, activity="verifying step")

    async def verify_rollup(
        self,
        plan: Plan,
        branch: PlanNode,
        evidence: str,
        *,
        children_summary: str = "",
    ) -> tuple[StepVerdict, str]:
        """Check whether a branch's own sub-goal was actually achieved, now
        that every one of its children individually reports complete. The
        tree shape alone (all children complete) is never proof by itself --
        `evidence` (ground-truth repo/file state for the branch's own goal)
        is the primary signal; `children_summary` (each child's own verdict/
        reason) is supporting context only, mirroring how leaf verification
        treats current file contents as ground truth over a tool's own
        success wording."""
        prompt = build_rollup_verify_prompt(
            goal=plan.goal,
            branch=branch.description,
            evidence=evidence,
            children_summary=children_summary,
        )
        return await self._verdict_from_prompt(prompt, activity="verifying branch rollup")

    async def _verdict_from_prompt(
        self, prompt: str, *, activity: str
    ) -> tuple[StepVerdict, str]:
        response = await self._complete(
            CompletionRequest(
                messages=[Message(role=Role.user, content=prompt)],
                temperature=0.0,
                max_tokens=100,
            ),
            activity=activity,
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
        revised_steps: list[PlanNode] = []
        for s in plan.steps:
            if s.status == StepStatus.completed:
                revised_steps.append(s)
        for s in steps:
            if not any(
                existing.description == s.description
                for existing in revised_steps
            ):
                s.index = len(revised_steps) + 1
                revised_steps.append(s)
        revised.steps = revised_steps
        return revised

    @classmethod
    def _clean_description(cls, description: str) -> str:
        return cls._STEP_PREFIX_RE.sub("", description.strip())

    @classmethod
    def _normalize_steps(
        cls,
        task: str,
        steps: list[PlanNode],
        *,
        planning_facts: dict[str, bool] | None = None,
    ) -> list[PlanNode]:
        """Remove low-value plan noise that traps smaller local models.

        This is deliberately conservative: if normalization would erase every
        step, the original cleaned plan is kept instead."""
        allow_setup_steps = bool(cls._TASK_SETUP_RE.search(task))
        protected_tests = bool(cls._DO_NOT_MODIFY_TESTS_RE.search(task))
        planning_facts = {
            path.casefold(): exists for path, exists in (planning_facts or {}).items()
        }
        normalized: list[PlanNode] = []
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
                PlanNode(
                    description=description,
                    index=len(normalized) + 1,
                    requested_branch=step.requested_branch,
                    flagged=step.flagged,
                )
            )

        if normalized:
            return normalized

        return [
            PlanNode(
                description=cls._clean_description(step.description),
                index=i + 1,
                requested_branch=step.requested_branch,
                flagged=step.flagged,
            )
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

    @staticmethod
    def _loads_step_data(text: str) -> Any:
        """Parse a plan array/object out of `text`, tolerating both valid JSON
        and a Python-repr-style literal (single-quoted keys/strings, True/
        False/None) that some models emit instead. Returns _UNPARSEABLE if
        neither parse succeeds, so callers can fall back to line-splitting
        instead of silently swallowing the whole literal into one step's
        description (see the parser bug this replaced: a single-line
        Python-repr list previously became one step whose description was
        the raw `"{'description': ..., 'type': 'leaf'}"` text)."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return _UNPARSEABLE

    def _parse_steps(self, content: str) -> list[PlanNode]:
        array_match = re.search(r"\[.*\]", content, re.DOTALL)
        if array_match:
            data = self._loads_step_data(array_match.group(0))
            if data is _UNPARSEABLE:
                return [
                    PlanNode(description=self._clean_description(s), index=i + 1)
                    for i, s in enumerate(content.strip().split("\n"))
                    if s.strip()
                ]
        else:
            data = self._loads_step_data(content)
            if data is _UNPARSEABLE:
                return [
                    PlanNode(description=self._clean_description(s), index=i + 1)
                    for i, s in enumerate(content.strip().split("\n"))
                    if s.strip()
                ]

        if isinstance(data, dict):
            # A model that emits a single step often skips the enclosing
            # array and just returns that one step's object directly. Treat
            # it as a one-item list instead of falling through to the
            # str(data) fallback below, which would stringify the dict as
            # Python repr (single-quoted) and hand that raw text back as one
            # step's description.
            data = [data]

        if isinstance(data, list) and all(isinstance(item, (str, dict)) for item in data):
            steps = []
            for i, item in enumerate(data):
                if isinstance(item, str) and item.strip().startswith("{"):
                    # Some models double-encode: a JSON array of strings,
                    # where each string is itself a stringified step object
                    # (e.g. '{"description": "...", "type": "leaf"}' or the
                    # Python-repr equivalent) rather than a real JSON object
                    # in the array. Unwrap it the same way, instead of
                    # taking the raw "{'description': ...}" text as the
                    # step's description verbatim.
                    unwrapped = self._loads_step_data(item)
                    if isinstance(unwrapped, dict):
                        item = unwrapped
                if isinstance(item, dict):
                    description = self._clean_description(item.get("description", str(item)))
                    requested_branch = str(item.get("type", "leaf")).strip().lower() == "branch"
                    flagged = bool(item.get("uncertain", False))
                else:
                    description = self._clean_description(item)
                    requested_branch = False
                    flagged = False
                steps.append(
                    PlanNode(
                        description=description,
                        index=i + 1,
                        requested_branch=requested_branch,
                        flagged=flagged,
                    )
                )
            return steps
        return [PlanNode(description=self._clean_description(str(data)), index=1)]

    _STATUS_MARKERS = {
        StepStatus.pending: "[ ]",
        StepStatus.in_progress: "[→]",
        StepStatus.completed: "[✓]",
        StepStatus.failed: "[✗]",
        StepStatus.skipped: "[-]",
    }

    @staticmethod
    def _active_path_ids(plan: Plan) -> set[str]:
        """Node ids of the current leaf plus every ancestor up to (and
        including) the root. Empty once the plan has no current leaf (all
        steps complete)."""
        current = plan.current_leaf()
        if current is None:
            return set()
        ids: set[str] = set()
        node: PlanNode | None = current
        while node is not None:
            ids.add(node.node_id)
            node = node.parent
        return ids

    @classmethod
    def format_plan_for_prompt(cls, plan: Plan) -> str:
        """Tree-aware render for the main-loop prompt: the active path
        (current leaf + its ancestor chain) is expanded, along with the
        siblings at each level of that path (shown in full one-line detail
        but not recursed into) -- every other subtree, completed or not yet
        reached, collapses to a single line with no descendants shown. Only
        ever recursing into active-path nodes keeps the render bounded by
        path depth * branching factor rather than growing with total tree
        size."""
        lines = ["## Plan", ""]
        active_path = cls._active_path_ids(plan)

        def render(node: PlanNode, depth: int) -> None:
            indent = "  " * depth
            for child in node.children:
                marker = cls._STATUS_MARKERS[child.status]
                lines.append(f"{indent}{marker} Step {child.index}: {child.description}")
                if child.node_id in active_path:
                    render(child, depth + 1)

        render(plan.root, 0)
        return "\n".join(lines)

    @classmethod
    def format_full_tree(cls, plan: Plan) -> str:
        """Renders the entire plan tree, unabridged, with flagged nodes
        (`PlanNode.flagged` -- ones the planner was uncertain how to
        decompose) marked inline rather than listed separately, and each
        node's `node_id` shown so a reviewer can address it. Used for the
        one-time pre-execution whole-tree human review pass (ticket 07),
        never for the bounded per-turn prompt render (`format_plan_for_prompt`
        above) -- this is only ever called once, before any execution, on a
        tree that hasn't grown from replans yet."""
        lines = [f"## Plan: {plan.goal}", ""]

        def render(node: PlanNode, depth: int) -> None:
            indent = "  " * depth
            for child in node.children:
                marker = cls._STATUS_MARKERS[child.status]
                flag = " ⚑" if child.flagged else ""
                lines.append(
                    f"{indent}{marker} [{child.node_id}] {child.description}{flag}"
                )
                render(child, depth + 1)

        render(plan.root, 0)
        return "\n".join(lines)

    @staticmethod
    def _format_plan(plan: Plan) -> str:
        return Planner.format_plan_for_prompt(plan)
