"""Deterministic capability-aware role assignment heuristics."""

from __future__ import annotations

from typing import Dict, List, Tuple

from vcomt2d.core.types import Pose2D, distance_pose, distance_xy
from .models import RoleAssignment, SceneFacts, TaskType


def assign_roles(scene_facts: SceneFacts, world_state) -> RoleAssignment:
    robots = {robot.robot_id: robot for robot in world_state.robots}
    if scene_facts.task_type == TaskType.T1_DOOR_WEDGE_PASS_THROUGH:
        door = world_state.door_map()[scene_facts.relevant_door_id]
        holder = sorted(robots.keys(), key=lambda robot_id: (distance_xy(robots[robot_id].pose.xy(), door.pose.xy()), robot_id))[0]
        passer = [robot_id for robot_id in robots if robot_id != holder][0]
        return RoleAssignment(
            task_type=scene_facts.task_type,
            roles={"holder": holder, "passer": passer, "follower": holder},
            rationale=f"Door holder is the robot closest to {scene_facts.relevant_door_id}; the other robot becomes passer.",
        )

    if scene_facts.task_type == TaskType.T2_HERDING_CORRALLING:
        ball = world_state.object_map()[scene_facts.target_object_id]
        goal = world_state.goal_map()[scene_facts.goal_region_id]
        pusher = sorted(robots.keys(), key=lambda robot_id: (distance_xy(robots[robot_id].pose.xy(), ball.pose.xy()), robot_id))[0]
        blocker = [robot_id for robot_id in robots if robot_id != pusher][0]
        return RoleAssignment(
            task_type=scene_facts.task_type,
            roles={"pusher": pusher, "blocker": blocker},
            rationale=f"Pusher is the robot closest to {scene_facts.target_object_id}; blocker is the partner positioned to close the goal-side escape angle toward {scene_facts.goal_region_id}.",
        )

    if scene_facts.task_type == TaskType.T4_COLLABORATIVE_SEARCH_CONVERGE:
        search_regions = scene_facts.search_region_ids[:2]
        assignments = _best_two_region_assignment(world_state, search_regions)
        target_region = world_state.goal_map()[scene_facts.goal_region_id]
        finder = sorted(assignments.keys(), key=lambda robot_id: (distance_xy(world_state.goal_map()[assignments[robot_id]].center, target_region.center), robot_id))[0]
        converger = [robot_id for robot_id in assignments if robot_id != finder][0]
        return RoleAssignment(
            task_type=scene_facts.task_type,
            roles={"finder": finder, "converger": converger, "search_assignments": assignments},
            rationale="Search sectors are assigned by minimum total travel distance; the robot whose sector is closest to the expected target region becomes the primary finder.",
        )

    if scene_facts.task_type == TaskType.T6_RELAY_DELIVERY:
        obj = world_state.object_map()[scene_facts.target_object_id]
        goal = world_state.goal_map()[scene_facts.goal_region_id]
        starter = sorted(robots.keys(), key=lambda robot_id: (distance_xy(robots[robot_id].pose.xy(), obj.pose.xy()), robot_id))[0]
        finisher = sorted(robots.keys(), key=lambda robot_id: (distance_xy(robots[robot_id].pose.xy(), goal.center), robot_id))[0]
        if finisher == starter:
            finisher = [robot_id for robot_id in robots if robot_id != starter][0]
        return RoleAssignment(
            task_type=scene_facts.task_type,
            roles={"starter": starter, "finisher": finisher},
            rationale=f"Starter is selected by proximity to {scene_facts.target_object_id}; finisher is selected by proximity to {scene_facts.goal_region_id}, with tie-breaking to preserve collaboration.",
        )

    raise ValueError(f"Unsupported task type for role assignment: {scene_facts.task_type}")


def _best_two_region_assignment(world_state, region_ids: List[str]) -> Dict[str, str]:
    robots = [robot.robot_id for robot in world_state.robots]
    a, b = region_ids[0], region_ids[1]
    cost_1 = _assignment_cost(world_state, {robots[0]: a, robots[1]: b})
    cost_2 = _assignment_cost(world_state, {robots[0]: b, robots[1]: a})
    return {robots[0]: a, robots[1]: b} if cost_1 <= cost_2 else {robots[0]: b, robots[1]: a}


def _assignment_cost(world_state, assignment: Dict[str, str]) -> float:
    total = 0.0
    for robot_id, region_id in assignment.items():
        total += distance_xy(world_state.robot_map()[robot_id].pose.xy(), world_state.goal_map()[region_id].center)
    return total

