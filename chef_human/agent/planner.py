from __future__ import annotations

import ast
import json
import logging
import re
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from chef_human.agent.prompts import (
    PLANNER_SYSTEM_PROMPT,
    build_atomicity_check_prompt,
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
    cause). `update_plan()` carries a node forward across a whole-plan replan
    either because it's already `StepStatus.completed`, or because the LLM's
    revised step tagged `continues_node_id` naming a still-non-completed
    prior node (see docs/adr/0001-evidence-carry-forward-across-whole-plan-
    replan.md) -- an untagged or unmatched step gets a fresh `node_id` and an
    empty evidence bucket, same as before that decision.
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
    # Generation-time hint (like `requested_branch`, not identity, not
    # persisted): the atomicity check's own REASON text when it reclassified
    # this node from leaf to branch -- it already names the distinct pieces
    # bundled into the step (e.g. "bundles subscribe, publish, retries, and
    # dead-lettering"), so the breakdown call expanding this node is told
    # what to split by instead of re-deriving that structure from scratch.
    # Empty for a node marked "branch" directly by the generation call.
    atomicity_reason: str = field(default="", repr=False, compare=False)
    # Generation-time hint only (not identity, not persisted): set from a
    # revised step's `continues_node_id` tag during `update_plan()`'s
    # whole-plan replan, naming the prior node this step continues. Consumed
    # once by `update_plan()` to decide node_id reuse, then irrelevant.
    continues_node_id: str | None = field(default=None, repr=False, compare=False)
    # Generation-time declared kind ("leaf", "branch", or "checkpoint"),
    # parsed/preserved the same way `requested_branch`/`flagged` already
    # are. Unlike `requested_branch` (a yes/no expand-now signal consumed
    # once by `_expand_node`), a checkpoint's "branch-ness" is deferred:
    # `requested_branch` stays False for a checkpoint so `_expand_node`
    # does not recurse into it at generation time, but `declared_type`
    # records that it should still be decomposed later, lazily, the first
    # time execution reaches it (see `Planner.expand_checkpoint`).
    declared_type: str = field(default="leaf", repr=False, compare=False)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def is_checkpoint(self) -> bool:
        return self.declared_type == "checkpoint"

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
        """Nodes that prevent a plan from being reported as complete:
        incomplete leaves, plus any branch (including a checkpoint) whose
        own rollup verification hasn't passed even though its children have
        all completed. Without the branch half of this, `is_complete()`
        (which does check branches) and `unresolved_steps()` disagreed --
        a checkpoint whose rollup verdict came back `not_complete` left its
        children all `completed` with nothing pending, so this returned
        `[]` and the `finish` tool's gate (keyed on this, not
        `is_complete()`) let the agent finish before the checkpoint's
        continuation into implementation steps ever fired. Observed live:
        a real run finished after pure exploration with zero files
        changed."""
        unresolved = [n for n in self._leaves() if n.status != StepStatus.completed]
        unresolved.extend(b for b in self._branches() if b.status != StepStatus.completed)
        return unresolved

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

    # Budget for calls that generate a plan's step JSON (initial expansion,
    # sub-goal breakdown, and the whole-tree-review redecompose path) --
    # deliberately generous. A "thinking" model (Settings.ollama_think /
    # OllamaBackend(think=True)) spends tokens on its reasoning trace before
    # ever emitting the JSON array, and those reasoning tokens count against
    # the same completion budget as the answer. At the previous 2048 cap, a
    # complex task's plan-generation call was observed to spend the entire
    # budget thinking and return empty content -- silently producing a
    # zero-step plan with no error surfaced anywhere. 8192 was validated
    # against a medium-complexity task with think=True: a real multi-level,
    # non-redundant decomposition completed well within budget.
    _PLANNING_MAX_TOKENS = 8192

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
        if response.thinking:
            logger.debug("LLM thinking (%s): %s", activity, response.thinking)
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
            CompletionRequest(messages=messages, temperature=0.0, max_tokens=self._PLANNING_MAX_TOKENS),
            activity="planning",
        )
        children = self._normalize_steps(
            task,
            self._parse_steps(response.message.content),
            planning_facts=planning_facts,
        )
        # Attach children to the tree before classifying them: the
        # atomicity check needs real .parent/.children links to see nearby
        # tree structure, which only exist once set_children() has run.
        node.set_children(children)
        if node.atomicity_reason and len(children) == 1:
            # `node` itself is a branch only because the atomicity check
            # reclassified it (not because the generation call explicitly
            # tagged it "branch" -- that case is trusted as-is, however many
            # children it yields). A breakdown of an atomicity-reclassified
            # node that produces exactly one child achieved nothing
            # structurally -- it can only be the same scope restated (a real
            # split of a bundled step always yields multiple pieces).
            # Observed without this guard: a flagged node's breakdown call
            # kept regenerating a single reworded paraphrase of itself,
            # which the atomicity check flagged again, forever (e.g. "Write
            # notify.py implementing the Bus class with subscribe..." <->
            # "Create notify.py with the Bus class implementation
            # including..."). Force it to stay a leaf rather than running it
            # through another atomicity check that would just repeat the
            # cycle.
            children[0].requested_branch = False
        else:
            await self._classify_children(task, children)
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

    _ATOMICITY_TREE_CONTEXT_LIMIT = 40

    async def _classify_children(self, task: str, children: list[PlanNode]) -> None:
        """Independently double-check every child the generation call left
        as a leaf, via one dedicated LLM call per leaf. The call that wrote
        a step's wording is not a reliable judge of whether that step is
        actually atomic -- it has every incentive to describe its own output
        as done and actionable, the same bias that makes self-grading
        unreliable elsewhere in this codebase. A step already marked
        "branch" is trusted as-is (it already gets decomposed further); this
        only exists to catch the false-negative case, a step that bundles
        multiple pieces of work but was left as "leaf" anyway.

        Callers must attach `children` to the tree (`node.set_children(...)`)
        before calling this -- each check needs real `.parent`/`.children`
        links to see nearby tree structure, not just the step's own
        wording. Judging a step in total isolation from the rest of the
        tree let it recommend a "breakdown" that just recreated an
        equivalent step one level up, forever (e.g. "ls" being broken down
        into "list current directory contents", whose own atomicity check
        then broke it back down into "ls") -- there was nothing in the
        step's own wording to reveal that loop, only the surrounding tree
        structure could."""
        for child in children:
            if child.requested_branch or child.is_checkpoint:
                # A checkpoint is deliberately left undecomposed at
                # generation time (see `expand_checkpoint`) -- running it
                # through the atomicity check here would judge a sub-goal
                # that has no children yet against the same "does this
                # bundle multiple pieces" rubric a leaf gets, and could
                # reclassify it into an ordinary branch, defeating the
                # lazy-expansion point of declaring it a checkpoint at all.
                continue
            nearby = self._collect_nearby_nodes(child, limit=self._ATOMICITY_TREE_CONTEXT_LIMIT)
            tree_context = self._render_nearby_tree(child, nearby)
            needs_breakdown, reason = await self._check_atomicity(
                task, child.description, tree_context
            )
            if needs_breakdown:
                logger.debug(
                    "Atomicity check reclassified leaf as branch: %r (%s)",
                    child.description,
                    reason,
                )
                child.requested_branch = True
                child.atomicity_reason = reason

    @staticmethod
    def _collect_nearby_nodes(node: PlanNode, *, limit: int) -> list[PlanNode]:
        """BFS outward from `node`, treating the tree as undirected (a
        node's parent and children both count as neighbors), collecting up
        to `limit` nodes in nearest-first order -- ancestors, siblings,
        cousins, and descendants alike, not just the straight-line ancestor
        chain a normal expansion call sees. `node` itself is included
        first."""
        visited = {node.node_id}
        collected = [node]
        queue: deque[PlanNode] = deque([node])
        while queue and len(collected) < limit:
            current = queue.popleft()
            neighbors: list[PlanNode] = list(current.children)
            if current.parent is not None:
                neighbors.append(current.parent)
            for neighbor in neighbors:
                if neighbor.node_id in visited:
                    continue
                visited.add(neighbor.node_id)
                collected.append(neighbor)
                queue.append(neighbor)
                if len(collected) >= limit:
                    break
        return collected

    @staticmethod
    def _render_nearby_tree(node: PlanNode, nearby: list[PlanNode]) -> str:
        """Renders `nearby` (from `_collect_nearby_nodes`) as an indented
        tree, marking `node` inline. A node whose parent fell outside the
        BFS ball becomes a top-level entry in the render (the ball's
        boundary), not nested under a parent that isn't shown."""
        nearby_ids = {n.node_id for n in nearby}
        roots = [n for n in nearby if n.parent is None or n.parent.node_id not in nearby_ids]
        lines: list[str] = []

        def label(n: PlanNode) -> str:
            text = "(overall task)" if n.description == "root" else n.description
            return f"{text}  <-- the step being checked" if n.node_id == node.node_id else text

        def render(n: PlanNode, depth: int) -> None:
            lines.append("  " * depth + f"- {label(n)}")
            for child in n.children:
                if child.node_id in nearby_ids:
                    render(child, depth + 1)

        for root in roots:
            render(root, 0)
        return "\n".join(lines)

    async def _check_atomicity(self, goal: str, step: str, tree_context: str) -> tuple[bool, str]:
        prompt = build_atomicity_check_prompt(goal=goal, step=step, tree_context=tree_context)
        response = await self._complete(
            CompletionRequest(
                messages=[Message(role=Role.user, content=prompt)],
                temperature=0.0,
                max_tokens=self._PLANNING_MAX_TOKENS,
            ),
            activity="checking step atomicity",
        )
        return self._parse_atomicity_verdict(response.message.content)

    @staticmethod
    def _parse_atomicity_verdict(content: str) -> tuple[bool, str]:
        """Returns (needs_breakdown, reason). Fails open (False, i.e. trust
        the leaf as-is) on anything unparseable, rather than retrying like
        step verification does -- an unparseable atomicity check should not
        block planning altogether, and leaving a genuinely-too-broad step as
        a leaf is a recoverable failure (the agent still attempts it; a
        subsequent replan can still split it), not a silent data loss."""
        text = content.strip()
        reason_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE)
        reason = reason_match.group(1).strip() if reason_match else ""
        verdict_match = re.search(
            r"^\s*VERDICT:\s*(ATOMIC|NEEDS_BREAKDOWN|NEEDS BREAKDOWN)\s*$",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        verdict_text = (
            verdict_match.group(1).upper().replace(" ", "_") if verdict_match else ""
        )
        if not verdict_text:
            bare_lines = [line.strip() for line in text.splitlines() if line.strip()]
            if bare_lines:
                first_line = bare_lines[0].upper().replace(" ", "_")
                if first_line in {"ATOMIC", "NEEDS_BREAKDOWN"}:
                    verdict_text = first_line
                    if not reason and len(bare_lines) > 1:
                        reason = " ".join(bare_lines[1:]).strip()
        return verdict_text == "NEEDS_BREAKDOWN", reason

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
        why_line = (
            f"\n\nWhy this needs breaking down: {node.atomicity_reason}"
            if node.atomicity_reason
            else ""
        )
        messages.append(
            Message(
                role=Role.user,
                content=(
                    f"Overall goal: {task}\n\n"
                    f"Ancestor chain (root goal down to this sub-goal):\n{chain}\n\n"
                    "Break the following sub-goal down into its own immediate next steps "
                    "(do not re-plan the overall goal, only this sub-goal):\n"
                    f"{node.description}"
                    f"{why_line}\n\n"
                    "This must produce at least two steps, each covering a genuinely "
                    "different piece of the sub-goal above -- a different responsibility, "
                    "feature, or concern, not the same scope restated in different words. "
                    "If the reason above names specific bundled pieces (e.g. "
                    "\"subscribe, publish, retries, and dead-lettering\"), split along "
                    "exactly those lines: one step per named piece. A response with only "
                    "one step, or where every step still describes the same overall scope "
                    "as the sub-goal itself, is not a real breakdown and will be rejected."
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
            CompletionRequest(messages=messages, temperature=0.0, max_tokens=self._PLANNING_MAX_TOKENS),
            activity="replanning",
        )
        children = self._normalize_steps(
            plan.goal, self._parse_steps(response.message.content)
        )
        node.set_children(children)
        if node.atomicity_reason and len(children) == 1:
            # See the matching guard in _expand_node: a single-child
            # breakdown of an atomicity-reclassified node is not a real
            # decomposition.
            children[0].requested_branch = False
        else:
            await self._classify_children(plan.goal, children)
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

    @staticmethod
    def _demote_pass_through_checkpoint(children: list[PlanNode]) -> None:
        """Structural guard (not left to prompt wording alone): a
        decomposition of exactly one child that is itself another
        checkpoint, with no other real work alongside it, achieves nothing
        -- it would just chain straight through to another deferred
        sub-goal. Force that lone child back to an ordinary leaf instead of
        letting it re-defer indefinitely."""
        if len(children) == 1 and children[0].is_checkpoint:
            logger.debug(
                "Demoting pass-through checkpoint (sole child, no other work): %r",
                children[0].description,
            )
            children[0].declared_type = "leaf"
            children[0].requested_branch = False

    async def expand_checkpoint(self, plan: Plan, checkpoint: PlanNode) -> None:
        """Lazily decomposes `checkpoint` into real children, the first
        time execution reaches it -- reusing the ordinary `_expand_node`
        pipeline a generation-time branch would go through. Nothing is
        speculatively generated past a checkpoint before this is called."""
        ancestors = self._ancestor_descriptions(checkpoint)
        convergence = _ConvergenceTracker(subtree_label=checkpoint.description)
        await self._expand_node(
            checkpoint,
            ancestors=ancestors,
            task=plan.goal,
            repo_context="",
            planning_facts=None,
            is_root=False,
            depth=len(ancestors) + 1,
            convergence=convergence,
        )
        self._demote_pass_through_checkpoint(checkpoint.children)
        if not checkpoint.children:
            # Should not happen in practice (the expansion call always
            # yields at least the parsed-fallback single step), but a
            # checkpoint stuck with zero children would look like an
            # unexpanded checkpoint forever and loop `expand_checkpoint`
            # every turn -- fail safe by treating it as an ordinary leaf.
            logger.warning(
                "Checkpoint %r expanded to zero children; treating it as "
                "an ordinary leaf instead",
                checkpoint.description,
            )
            checkpoint.declared_type = "leaf"

    async def continue_from_checkpoint(
        self, plan: Plan, checkpoint: PlanNode, evidence: str
    ) -> list[PlanNode]:
        """Success-framed planning call: fires once `checkpoint`'s own
        rollup verification has passed, to plan the concrete next steps
        using what its children actually discovered -- distinct from
        `update_plan`, which is framed around recovering from a failure.
        Returned nodes are brand-new (no `continues_node_id`) and are not
        yet attached to the tree; the caller is responsible for splicing
        them in as siblings and, once attached, running them through
        `expand_spliced_steps` for classification/further expansion."""
        ancestors = self._ancestor_descriptions(checkpoint)
        chain = "\n".join(f"- {d}" for d in ([plan.goal] + ancestors))
        messages = [
            Message(
                role=Role.system,
                content=PLANNER_SYSTEM_PROMPT
                + "\n\nA checkpoint sub-goal has just been completed and verified -- "
                "nothing failed. Use what its work actually discovered to plan the "
                "concrete next steps that continue the plan from here.",
            ),
            Message(
                role=Role.user,
                content=(
                    f"Overall goal: {plan.goal}\n\n"
                    f"Ancestor chain (root goal down to this checkpoint):\n{chain}\n\n"
                    f"This checkpoint's sub-goal (now complete): {checkpoint.description}\n\n"
                    f"What the checkpoint's own work actually discovered:\n{evidence or '(no evidence recorded)'}\n\n"
                    "Using what was learned above, output a revised JSON array of the "
                    "concrete next steps that continue the plan from here. Do not repeat "
                    "exploration that is already done above -- plan real, grounded steps "
                    "(implementation steps where the evidence above supports them), not "
                    "another round of the same exploration."
                ),
            ),
        ]
        response = await self._complete(
            CompletionRequest(messages=messages, temperature=0.0, max_tokens=self._PLANNING_MAX_TOKENS),
            activity="continuing from checkpoint",
        )
        steps = self._normalize_steps(plan.goal, self._parse_steps(response.message.content))
        for step in steps:
            step.continues_node_id = None
        return steps

    async def expand_spliced_steps(self, plan: Plan, steps: list[PlanNode]) -> None:
        """Runs freshly-spliced continuation steps (already attached to the
        tree with real `.parent` links by the caller) through the same
        atomicity-check/classification pipeline any other generated step
        goes through, recursing into any that get classified as an
        ordinary branch. A spliced step declared a further checkpoint is
        left alone here (it stays lazily unexpanded, same as any other
        checkpoint) -- except a lone spliced checkpoint with no sibling
        work, which the pass-through guard demotes."""
        self._demote_pass_through_checkpoint(steps)
        await self._classify_children(plan.goal, steps)
        convergence = _ConvergenceTracker(subtree_label="checkpoint continuation")
        for step in steps:
            if step.requested_branch and not step.is_checkpoint:
                ancestors = self._ancestor_descriptions(step)
                await self._expand_node(
                    step,
                    ancestors=ancestors,
                    task=plan.goal,
                    repo_context="",
                    planning_facts=None,
                    is_root=False,
                    depth=len(ancestors) + 1,
                    convergence=convergence,
                )

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
                max_tokens=self._PLANNING_MAX_TOKENS,
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
                max_tokens=self._PLANNING_MAX_TOKENS,
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
        non_completed = {
            s.node_id: s for s in plan.steps if s.status != StepStatus.completed
        }
        continuation_hint = ""
        if non_completed:
            lines = [
                f"- [{node_id}] {node.description}" for node_id, node in non_completed.items()
            ]
            continuation_hint = (
                "\n\nThese steps have not completed yet:\n" + "\n".join(lines) +
                "\n\nIf a revised step is essentially the same underlying work as one of these "
                '(even reworded), tag it with "continues_node_id": "<that id>" so its prior '
                'progress carries forward. Omit the tag (or use null) for a genuinely new step.'
            )
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
                f"Failure context:\n{failure_context}"
                f"{continuation_hint}\n\n"
                f"Output a revised JSON array of remaining steps.",
            ),
        ]
        response = await self._complete(
            CompletionRequest(messages=messages, temperature=0.0, max_tokens=self._PLANNING_MAX_TOKENS),
            activity="replanning",
        )
        steps = self._normalize_steps(plan.goal, self._parse_steps(response.message.content))

        revised = Plan(goal=plan.goal)
        revised_steps: list[PlanNode] = []
        for s in plan.steps:
            if s.status == StepStatus.completed:
                revised_steps.append(s)
        claimed_ids: set[str] = set()
        for s in steps:
            if not any(
                existing.description == s.description
                for existing in revised_steps
            ):
                continued = non_completed.get(s.continues_node_id or "")
                if continued is not None and continued.node_id not in claimed_ids:
                    claimed_ids.add(continued.node_id)
                    s.node_id = continued.node_id
                    s.status = StepStatus.pending
                elif s.continues_node_id:
                    logger.debug(
                        "update_plan: continues_node_id %r on step %r did not match an "
                        "unclaimed non-completed node -- treating as a new node",
                        s.continues_node_id,
                        s.description,
                    )
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
                    continues_node_id=step.continues_node_id,
                    declared_type=step.declared_type,
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
                continues_node_id=step.continues_node_id,
                declared_type=step.declared_type,
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
                    type_str = str(item.get("type", "leaf")).strip().lower()
                    declared_type = type_str if type_str in {"leaf", "branch", "checkpoint"} else "leaf"
                    requested_branch = declared_type == "branch"
                    flagged = bool(item.get("uncertain", False))
                    continues_node_id = item.get("continues_node_id") or None
                    if not isinstance(continues_node_id, str):
                        continues_node_id = None
                else:
                    description = self._clean_description(item)
                    requested_branch = False
                    flagged = False
                    continues_node_id = None
                    declared_type = "leaf"
                steps.append(
                    PlanNode(
                        description=description,
                        index=i + 1,
                        requested_branch=requested_branch,
                        flagged=flagged,
                        continues_node_id=continues_node_id,
                        declared_type=declared_type,
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
