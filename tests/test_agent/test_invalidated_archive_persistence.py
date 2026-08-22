"""Ticket 15: prove the plan-model half of evidence-driven subtree replan
independently of loop orchestration.

The feature spec (.scratch/evidence-driven-subtree-replan/spec.md) splits the
invalidation mechanism across tickets 04/05/06 (status, off-tree archive,
compatible persistence) and the loop tickets that wire them into `ReActLoop`
(ticket 10 and beyond). This file proves the model-level contract that those
loop tickets build on, at the plan-model seam:

- the `invalidated` status behaves coherently *inside* an invalidated
  subtree (node invalidated, live children cleared by the loop, rebuilt
  children installed in place) -- distinct from `failed`, `skipped`, and
  `pending` throughout;
- discarded subtrees are archived off-tree and stay out of every live-tree
  view, even across the in-place rebuild and a later whole-plan replan;
- serialization/loading round-trips the new metadata losslessly while
  documents without it (pre-feature, or from a consumer that strips
  unknown keys) load exactly as before -- including the JSON file path
  benchmark replay actually uses.

These tests deliberately stop at the Plan/PlanNode/serialization seam:
gating, budgets, cooldowns, and planner-call shape are proven at the
loop/planner seams by sibling test tickets (14/16/17).
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from chef_human.agent.planner import Plan, PlanNode, Planner, StepStatus
from chef_human.llm.backend import CompletionResponse, Message, Role


def _make_mock_backend(steps: list[dict]) -> MagicMock:
    """A mock LLM backend whose `complete` returns `steps` as plan JSON,
    matching the shape `test_planner.py` uses for whole-plan replan tests.
    Each dict is one step: `description` plus optional `continues_node_id`."""
    payload = json.dumps(steps)
    mock = MagicMock()
    mock.complete = AsyncMock(
        return_value=CompletionResponse(
            message=Message(role=Role.assistant, content=payload)
        )
    )
    return mock


def _discarded_branch() -> tuple[PlanNode, list[PlanNode]]:
    """An invalidated branch whose live children have been cleared (as the
    evidence-driven replan does) plus the exact children that were
    discarded. The discarded children keep a mix of statuses, including a
    completed leaf: completed work from a discarded subtree is *not*
    preserved automatically -- it survives only if the rebuilt subtree
    explicitly continues it (spec: "Continuation of prior valid work")."""
    branch = PlanNode(
        index=1, description="Build the transport layer", status=StepStatus.invalidated
    )
    discarded = [
        PlanNode(
            index=1,
            description="Assume a TCP socket protocol",
            status=StepStatus.completed,
        ),
        PlanNode(
            index=2,
            description="Write the socket framing code",
            status=StepStatus.failed,
        ),
    ]
    return branch, discarded


class TestInvalidatedSubtreeCoherence:
    """Ticket 15 / bullet 1: with an invalidated subtree installed
    (node `invalidated`, live children cleared, rebuilt children in
    place), every plan operation that inspects status stays coherent."""

    def test_in_place_rebuild_is_executable_again(self):
        """After the loop archives a subtree and installs the rebuilt
        children in place, the same node is workable again: its new
        children are the live current work, while the old structure is
        only archive metadata."""
        branch, discarded = _discarded_branch()
        plan = Plan(goal="g", steps=[branch])
        plan.archive_subtree(branch, discarded, reason="r", evidence_summary="e")
        # Ticket 10's replacement step: install the rebuilt subtree under
        # the SAME node.
        branch.set_children(
            [
                PlanNode(
                    index=1,
                    description="Read the real protocol definition",
                    status=StepStatus.pending,
                ),
            ]
        )

        current = plan.current_leaf()
        assert current is not None
        assert current.description == "Read the real protocol definition"
        # The plan is not complete while the rebuilt work is pending, and
        # the rebuilt leaf -- not any archived descendant -- is the
        # unresolved work.
        assert not plan.is_complete()
        assert current in plan.unresolved_steps()
        assert all(
            n.description not in {d.description for d in discarded}
            for n in plan.unresolved_steps()
        )

    def test_invalidated_is_distinct_from_every_other_status_in_subtree_state(self):
        """The invalidated node never collapses into `failed` (it did not
        fail to execute -- its assumptions were disproven), `skipped`
        (budget), `pending` (not yet worked), or `completed`. Distinctness
        holds for identity and equality, and survives serialization as a
        plain string old code can still read."""
        node = PlanNode(index=1, description="Superseded", status=StepStatus.invalidated)
        for other in (
            StepStatus.failed,
            StepStatus.skipped,
            StepStatus.pending,
            StepStatus.in_progress,
            StepStatus.completed,
        ):
            assert node.status is not other
            assert node.status != other
        assert node.to_dict()["status"] == "invalidated"
        assert StepStatus(node.to_dict()["status"]) is StepStatus.invalidated
        # A node that merely failed keeps its failed identity: the two
        # stories -- "execution failed" vs "superseded by better evidence"
        # -- remain separately expressible in one plan.
        failed = PlanNode(index=1, description="Superseded", status=StepStatus.failed)
        assert node != failed

    @pytest.mark.asyncio
    async def test_invalidated_node_is_offered_for_continuation_not_auto_preserved(self):
        """Whole-plan replan contract for an invalidated node: it is not
        completed work, so it is *not* silently preserved into the
        revised plan (unlike `completed` steps), but it IS offered for
        continuation -- a revised step tagging it with
        `continues_node_id` carries its stable identity forward and the
        continued step starts fresh (`pending`), so the rebuilt subtree
        does not inherit the superseded node's status."""
        branch, discarded = _discarded_branch()
        plan = Plan(goal="g", steps=[branch])
        plan.archive_subtree(branch, discarded, reason="r")

        # Untagged: the invalidated node is dropped, like any other
        # non-completed node the planner did not continue.
        dropped = await Planner(
            _make_mock_backend([{"description": "Fresh transport work"}])
        ).update_plan(plan, failure_context="x")
        assert dropped.find_node(branch.node_id) is None
        assert [s.description for s in dropped.steps] == ["Fresh transport work"]

        # Tagged: the identity carries forward and the step resets to
        # pending -- the invalidated status is not smuggled into the
        # rebuilt work.
        continued = await Planner(
            _make_mock_backend(
                [
                    {
                        "description": "Build the real transport layer",
                        "continues_node_id": branch.node_id,
                    }
                ]
            )
        ).update_plan(plan, failure_context="x")
        survivor = continued.find_node(branch.node_id)
        assert survivor is not None
        assert survivor.status is StepStatus.pending
        # The off-tree archive rode along with the revised plan either
        # way, so the discard history survives the whole-plan replan.
        assert branch.node_id in continued.archived_nodes

    def test_invalidation_is_scoped_to_the_invalidated_subtree_only(self):
        """Siblings of an invalidated branch are untouched: their status
        and workability are decided by their own content, not by a
        neighboring branch's invalidation."""
        branch, discarded = _discarded_branch()
        sibling = PlanNode(
            index=2, description="Unrelated work", status=StepStatus.pending
        )
        plan = Plan(goal="g", steps=[branch, sibling])
        plan.archive_subtree(branch, discarded, reason="r")

        assert sibling.status is StepStatus.pending
        assert plan.current_leaf() is sibling
        # The archive records exactly one node, keyed by the invalidated
        # node's identity -- the sibling never lands in it.
        assert list(plan.archived_nodes) == [branch.node_id]
        assert plan.find_node(sibling.node_id) is sibling


class TestOffTreeArchiveModel:
    """Ticket 15 / bullet 2: discarded subtrees are archived off-tree
    rather than left in the live executable tree."""

    def test_archived_structure_is_unreachable_from_live_tree_views(self):
        """Every live-tree view -- leaves, branches, unresolved work, node
        lookup, current work -- must be blind to the archived descendants,
        and mutating the live node afterwards must not reach the archive
        (detached snapshot)."""
        branch, discarded = _discarded_branch()
        plan = Plan(goal="g", steps=[branch])
        entry = plan.archive_subtree(branch, discarded, reason="r")

        assert branch.children == []
        # The whole live tree (every branch and leaf under the root) is
        # blind to the archived descendants...
        assert all(
            n.description not in {d.description for d in discarded}
            for n in _all_nodes(plan)
        )
        # ...and the archived nodes are not findable by identity either.
        assert all(plan.find_node(d.node_id) is None for d in discarded)
        # The live node's own identity stays live (the replacement is
        # installed under it); its descendants do not.
        assert plan.find_node(branch.node_id) is branch
        assert plan.current_leaf() is None
        # Detached: later live mutation cannot rewrite archived history.
        discarded[0].description = "mutated later"
        assert entry.children[0].description == "Assume a TCP socket protocol"

    def test_archive_is_keyed_by_stable_identity_across_description_changes(self):
        """The archive is keyed by `node_id`, never by description or
        index: re-archiving the same node after its wording changed
        appends to the *same* key's history instead of creating a
        parallel entry."""
        branch, first = _discarded_branch()
        plan = Plan(goal="g", steps=[branch])
        plan.archive_subtree(branch, first, reason="first")
        # The rebuilt node keeps its identity; the loop may reword it.
        branch.description = "Build the real transport layer"
        plan.archive_subtree(
            branch,
            [PlanNode(index=1, description="second generation")],
            reason="second",
        )
        assert list(plan.archived_nodes) == [branch.node_id]
        entry = plan.archived_nodes[branch.node_id]
        assert len(entry.history) == 1
        assert entry.history[0].reason == "first"

    def test_archive_is_metadata_not_live_state(self):
        """The archive is observability data, not plan state: two plans
        over the same live tree are equal even when one carries archived
        history (the archive is deliberately excluded from `__eq__`), and
        a plan with no archives serializes with no archive key at all."""
        branch, discarded = _discarded_branch()
        with_archive = Plan(goal="g", steps=[branch])
        with_archive.archive_subtree(branch, discarded, reason="r")
        # Same goal, same root (and therefore the same live tree): the
        # archive must not let the plans differ.
        without_archive = Plan(goal="g", root=with_archive.root)
        assert with_archive == without_archive
        assert without_archive.archived_nodes == {}

        plain = Plan(goal="g", steps=[PlanNode(index=1, description="do it")])
        assert "archived_subtrees" not in plain.to_dict()

    @pytest.mark.asyncio
    async def test_archive_survives_whole_plan_replan_and_stays_off_tree(self):
        """`update_plan` builds a fresh Plan object; the archived history
        must ride along so the later evidence-driven rebuild of the same
        node keeps its full discard history -- and the carried archive
        must remain off-tree in the revised plan (reached only via
        `archived_nodes`, never through the live tree)."""
        branch, discarded = _discarded_branch()
        plan = Plan(goal="g", steps=[branch])
        plan.archive_subtree(branch, discarded, reason="r")
        plan.archive_subtree(
            branch,
            [PlanNode(index=1, description="second generation")],
            reason="second",
        )

        revised = await Planner(
            _make_mock_backend([{"description": "Fresh step"}])
        ).update_plan(plan, failure_context="x")

        entry = revised.archived_nodes[branch.node_id]
        assert entry.reason == "second"
        assert entry.history[0].reason == "r"
        # Off-tree in the revised plan too: the live tree of the revised
        # plan shows only the fresh step; nothing archived is reachable
        # through it.
        assert [s.description for s in revised.steps] == ["Fresh step"]
        archived_descriptions = {d.description for d in discarded} | {
            "second generation"
        }
        assert all(n.description not in archived_descriptions for n in _all_nodes(revised))


class TestBackwardCompatiblePersistence:
    """Ticket 15 / bullet 3: serialization/loading stays
    backward-compatible while preserving the new metadata when present."""

    def test_pre_feature_document_loads_and_still_serializes_without_new_keys(self):
        """A document produced before the invalidation feature -- no
        `node_id` on steps, no `archived_subtrees` key -- loads safely and
        the reloaded plan serializes back into the same legacy shape (plus
        the additive per-step `node_id`, which old consumers ignore)."""
        legacy = {
            "goal": "g",
            "steps": [
                {"index": 1, "description": "Read the README", "status": "completed"},
                {
                    "index": 2,
                    "description": "Write the code",
                    "status": "pending",
                    "type": "branch",
                    "children": [{"index": 1, "description": "Implement the helper"}],
                },
            ],
        }
        plan = Plan.from_dict(legacy)
        assert [s.description for s in plan.steps] == ["Read the README", "Write the code"]
        assert plan.archived_nodes == {}
        reserialized = plan.to_dict()
        assert "archived_subtrees" not in reserialized
        assert reserialized["goal"] == "g"
        # Legacy readers keep working: the fields they rely on are intact,
        # and the only addition is the optional per-step node_id.
        assert [s["description"] for s in reserialized["steps"]] == [
            s["description"] for s in legacy["steps"]
        ]
        for step in reserialized["steps"]:
            assert step["description"]
            assert step["status"]
            assert "node_id" in step

    def test_consumer_that_strips_new_keys_loads_the_old_view(self):
        """A consumer that does not know the new metadata -- e.g. it
        filters a payload down to the fields it recognizes before loading
        -- sees exactly the legacy plan, with no crash and no phantom
        archive entries."""
        branch, discarded = _discarded_branch()
        plan = Plan(goal="g", steps=[branch])
        plan.archive_subtree(branch, discarded, reason="r")
        payload = plan.to_dict()

        known = {"index", "description", "status", "type", "children"}
        stripped = {
            "goal": payload["goal"],
            "steps": [
                {k: v for k, v in step.items() if k in known}
                for step in payload["steps"]
            ],
        }
        legacy_plan = Plan.from_dict(stripped)
        assert [s.description for s in legacy_plan.steps] == [branch.description]
        assert legacy_plan.steps[0].status is StepStatus.invalidated
        assert legacy_plan.archived_nodes == {}

    @pytest.mark.asyncio
    async def test_completed_descendant_of_discarded_subtree_is_not_preserved_automatically(self):
        """The discarded subtree held a *completed* leaf. Whole-plan
        replans preserve completed nodes automatically, but only from the
        *live* tree -- the discarded copy lives only in the archive, which
        is off-tree metadata. So the completed descendant must not be
        resurrected into the revised plan *automatically* (spec:
        "Continuation of prior valid work"; the explicit-continuation path
        that lets it survive is ticket 09's, proven by ticket 16)."""
        branch, discarded = _discarded_branch()
        completed_id = discarded[0].node_id
        plan = Plan(goal="g", steps=[branch])
        plan.archive_subtree(branch, discarded, reason="r")

        revised = await Planner(
            _make_mock_backend([{"description": "Fresh transport work"}])
        ).update_plan(plan, failure_context="x")

        assert revised.find_node(completed_id) is None
        assert "Assume a TCP socket protocol" not in [
            n.description for n in _all_nodes(revised)
        ]
        # ...but it is still inspectable in the archive, where the
        # completed status is retained.
        assert [n.status for n in revised.archived_nodes[branch.node_id].children] == [
            StepStatus.completed,
            StepStatus.failed,
        ]

    def test_full_metadata_round_trip_preserves_everything(self):
        """When the metadata *is* present, the round trip is lossless:
        the invalidated status comes back as the enum, the invalidated
        node keeps its stable identity, the archive keeps its keying,
        structure, statuses, reason/evidence context, and multi-generation
        history -- and the loaded plan is executable (parent links
        intact)."""
        branch, first = _discarded_branch()
        plan = Plan(goal="g", steps=[branch])
        plan.archive_subtree(
            branch,
            first,
            reason="tcp framing disproved",
            evidence_summary="protocol.py:42",
        )
        second_gen = [PlanNode(index=1, description="udp gen")]
        plan.archive_subtree(branch, second_gen, reason="udp also stale")
        # Rebuild the subtree in place, with one completed descendant.
        branch.set_children(
            [
                PlanNode(
                    index=1, description="Real protocol work",
                    status=StepStatus.completed,
                ),
                PlanNode(index=2, description="Next step", status=StepStatus.pending),
            ]
        )

        reloaded = Plan.from_dict(plan.to_dict())

        live = reloaded.find_node(branch.node_id)
        assert live is not None
        assert live.status is StepStatus.invalidated
        assert [c.description for c in live.children] == [
            "Real protocol work",
            "Next step",
        ]
        assert live.children[0].parent is live
        assert reloaded.current_leaf() is live.children[1]

        original = plan.archived_nodes[branch.node_id]
        entry = reloaded.archived_nodes[branch.node_id]
        assert entry.node_id == branch.node_id
        assert entry.reason == "udp also stale"
        assert [n.node_id for n in entry.children] == [
            n.node_id for n in original.children
        ]
        assert [n.node_id for n in entry.children] == [
            n.node_id for n in second_gen
        ]
        assert entry.history[0].reason == "tcp framing disproved"
        assert entry.history[0].evidence_summary == "protocol.py:42"
        assert [n.status for n in entry.history[0].children] == [
            StepStatus.completed,
            StepStatus.failed,
        ]
        # A second serialization of the reloaded plan is the identical
        # document -- nothing is added, dropped, or reordered on load.
        assert reloaded.to_dict() == plan.to_dict()

    def test_round_trip_through_json_file_is_stable(self):
        """The same round trip through an actual JSON file (the path
        benchmark replay and plan persistence use) is lossless: metadata
        in, metadata out, byte-identical document on the way back."""
        branch, discarded = _discarded_branch()
        plan = Plan(goal="g", steps=[branch])
        plan.archive_subtree(branch, discarded, reason="r", evidence_summary="e")
        branch.set_children([PlanNode(index=1, description="Rebuilt")])

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "plan.json")
            with open(path, "w") as f:
                json.dump(plan.to_dict(), f)
            with open(path) as f:
                reloaded = Plan.from_dict(json.load(f))

        assert reloaded.to_dict() == plan.to_dict()
        assert reloaded.find_node(branch.node_id) is not None
        assert reloaded.archived_nodes[branch.node_id].reason == "r"

    def test_unknown_status_string_degrades_to_pending_not_error(self):
        """A document from a *newer* build may carry a status this build
        does not know: the loader degrades that node to `pending` instead
        of raising, and the rest of the plan (including the archive)
        loads normally."""
        payload = {
            "goal": "g",
            "steps": [
                {
                    "index": 1,
                    "description": "future status",
                    "status": "superseded_by_future_feature",
                    "node_id": "a" * 32,
                },
                {
                    "index": 2,
                    "description": "ok",
                    "status": "completed",
                    "node_id": "b" * 32,
                },
            ],
            "archived_subtrees": {
                "a" * 32: {"node_id": "a" * 32, "reason": "r"},
            },
        }
        plan = Plan.from_dict(payload)
        assert plan.steps[0].status is StepStatus.pending
        assert plan.steps[1].status is StepStatus.completed
        assert list(plan.archived_nodes) == ["a" * 32]


def _all_nodes(plan: Plan) -> list[PlanNode]:
    """Every non-root node of the live tree -- the reachable surface a
    live-tree consumer can see."""
    out: list[PlanNode] = []

    def walk(node: PlanNode) -> None:
        for child in node.children:
            out.append(child)
            walk(child)

    walk(plan.root)
    return out
