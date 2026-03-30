"""Deterministic capability-aware role assignment heuristics."""

from __future__ import annotations

from .heuristics import (
    choose_door_roles,
    choose_herding_roles,
    choose_relay_roles,
    choose_search_roles,
)
from .models import RoleAssignment, SceneFacts, TaskType


def assign_roles(scene_facts: SceneFacts, world_state) -> RoleAssignment:
    if scene_facts.task_type == TaskType.T1_DOOR_WEDGE_PASS_THROUGH:
        roles, rationale, metrics = choose_door_roles(
            world_state,
            scene_facts.relevant_door_id,
            scene_facts.goal_region_id,
            scene_facts.wait_region_id,
        )
        return RoleAssignment(
            task_type=scene_facts.task_type,
            roles=roles,
            rationale=f"{rationale} Metrics={metrics}",
        )

    if scene_facts.task_type == TaskType.T2_HERDING_CORRALLING:
        roles, rationale, metrics = choose_herding_roles(world_state, scene_facts.target_object_id, scene_facts.goal_region_id)
        return RoleAssignment(
            task_type=scene_facts.task_type,
            roles=roles,
            rationale=f"{rationale} Metrics={metrics}",
        )

    if scene_facts.task_type == TaskType.T4_COLLABORATIVE_SEARCH_CONVERGE:
        roles, rationale, metrics = choose_search_roles(world_state, scene_facts.search_region_ids, scene_facts.goal_region_id)
        return RoleAssignment(
            task_type=scene_facts.task_type,
            roles=roles,
            rationale=f"{rationale} Metrics={metrics}",
        )

    if scene_facts.task_type == TaskType.T6_RELAY_DELIVERY:
        roles, rationale, metrics = choose_relay_roles(
            world_state,
            scene_facts.target_object_id,
            scene_facts.handoff_region_id,
            scene_facts.goal_region_id,
        )
        return RoleAssignment(
            task_type=scene_facts.task_type,
            roles=roles,
            rationale=f"{rationale} Metrics={metrics}",
        )

    raise ValueError(f"Unsupported task type for role assignment: {scene_facts.task_type}")
