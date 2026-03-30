"""Extract task-relevant facts from the structured 2D world state."""

from __future__ import annotations

from typing import List, Optional

from vcomt2d.sim.entities import WorldState
from .models import SceneFacts, TaskType


class SceneInterpretationError(RuntimeError):
    pass


def interpret_scene(task_type: TaskType, world_state: WorldState) -> SceneFacts:
    if task_type == TaskType.T1_DOOR_WEDGE_PASS_THROUGH:
        door = _first_or_none(world_state.doors)
        goal_region_id = world_state.task_facts.get("goal_region_id") or _first_goal_by_label(world_state, "goal_room")
        wait_region_id = _first_goal_by_label(world_state, "door_wait")
        if door is None:
            raise SceneInterpretationError("no_valid_door_found")
        if goal_region_id is None:
            raise SceneInterpretationError("no_goal_region_found")
        return SceneFacts(task_type=task_type, relevant_door_id=door.door_id, goal_region_id=goal_region_id, wait_region_id=wait_region_id, topology_notes=[link.via for link in world_state.topology])

    if task_type == TaskType.T2_HERDING_CORRALLING:
        ball = _first_object_of_type(world_state, "ball")
        goal_region_id = world_state.task_facts.get("goal_region_id") or _first_goal_by_label(world_state, "goal_area")
        staging_ids = [goal.region_id for goal in world_state.goals if goal.semantic_label in {"staging", "funnel"}]
        if ball is None:
            raise SceneInterpretationError("target_ball_not_found")
        if goal_region_id is None:
            raise SceneInterpretationError("no_goal_region_found")
        return SceneFacts(task_type=task_type, target_object_id=ball.object_id, goal_region_id=goal_region_id, staging_region_ids=staging_ids)

    if task_type == TaskType.T4_COLLABORATIVE_SEARCH_CONVERGE:
        target_object_id = world_state.task_facts.get("target_object_id") or _find_search_target(world_state)
        search_region_ids = world_state.task_facts.get("search_region_ids") or [goal.region_id for goal in world_state.goals if goal.semantic_label == "search_sector"]
        target_region_id = world_state.task_facts.get("target_region_id") or _first_goal_by_label(world_state, "target_region")
        if target_object_id is None:
            raise SceneInterpretationError("target_object_not_found")
        if len(search_region_ids) < 2:
            raise SceneInterpretationError("insufficient_search_regions")
        if target_region_id is None:
            raise SceneInterpretationError("no_target_region_found")
        return SceneFacts(task_type=task_type, target_object_id=target_object_id, search_region_ids=search_region_ids, goal_region_id=target_region_id)

    if task_type == TaskType.T6_RELAY_DELIVERY:
        target_object = _first_object_of_type(world_state, "box")
        target_object_id = world_state.task_facts.get("target_object_id") or (target_object.object_id if target_object is not None else None)
        goal_region_id = world_state.task_facts.get("goal_region_id") or _first_goal_by_label(world_state, "goal_region")
        handoff_region_id = world_state.task_facts.get("handoff_region_id") or _first_goal_by_label(world_state, "handoff_region")
        if target_object_id is None:
            raise SceneInterpretationError("target_object_not_found")
        if goal_region_id is None:
            raise SceneInterpretationError("no_goal_region_found")
        if handoff_region_id is None:
            raise SceneInterpretationError("no_handoff_region_found")
        return SceneFacts(task_type=task_type, target_object_id=target_object_id, goal_region_id=goal_region_id, handoff_region_id=handoff_region_id)

    raise SceneInterpretationError(f"unsupported_task_type:{task_type}")


def _first_or_none(items):
    return items[0] if items else None


def _first_object_of_type(world_state: WorldState, object_type: str):
    for obj in world_state.objects:
        if obj.object_type == object_type:
            return obj
    return None


def _first_goal_by_label(world_state: WorldState, semantic_label: str) -> Optional[str]:
    for goal in world_state.goals:
        if goal.semantic_label == semantic_label:
            return goal.region_id
    return world_state.goals[0].region_id if world_state.goals else None


def _find_search_target(world_state: WorldState) -> Optional[str]:
    for obj in world_state.objects:
        if "target" in obj.semantic_tags or obj.color == "red":
            return obj.object_id
    return None
