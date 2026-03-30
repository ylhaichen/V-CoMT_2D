"""Reusable task heuristics for role assignment and task realism."""

from __future__ import annotations

from itertools import permutations
from typing import Dict, Iterable, List, Sequence, Tuple

from vcomt2d.core.types import Vec2, clamp, distance_xy, midpoint


def choose_door_roles(world_state, door_id: str, goal_region_id: str, wait_region_id: str | None) -> Tuple[Dict[str, str], str, Dict[str, float]]:
    door_xy = world_state.door_map()[door_id].pose.xy()
    goal_xy = world_state.goal_map()[goal_region_id].center
    wait_xy = world_state.goal_map()[wait_region_id].center if wait_region_id and wait_region_id in world_state.goal_map() else door_xy
    robot_ids = sorted(robot.robot_id for robot in world_state.robots)
    scored: List[Tuple[float, str, str, Dict[str, float]]] = []

    for holder in robot_ids:
        passer = next(robot_id for robot_id in robot_ids if robot_id != holder)
        holder_pose = world_state.robot_map()[holder].pose.xy()
        passer_pose = world_state.robot_map()[passer].pose.xy()
        holder_cost = distance_xy(holder_pose, door_xy) + 0.25 * distance_xy(holder_pose, goal_xy)
        passer_cost = 0.8 * distance_xy(passer_pose, wait_xy) + 0.35 * distance_xy(passer_pose, goal_xy)
        total = holder_cost + passer_cost
        scored.append(
            (
                total,
                holder,
                passer,
                {
                    "holder_cost": round(holder_cost, 3),
                    "passer_cost": round(passer_cost, 3),
                    "total_cost": round(total, 3),
                },
            )
        )

    best_total, holder, passer, metrics = sorted(scored, key=lambda item: (item[0], item[1], item[2]))[0]
    rationale = (
        f"Door roles minimize a joint staging cost to the door and target room; "
        f"{holder} is the lowest-cost holder and {passer} is the corresponding passer."
    )
    metrics["joint_cost"] = round(best_total, 3)
    return {"holder": holder, "passer": passer, "follower": holder}, rationale, metrics


def choose_herding_roles(world_state, object_id: str, goal_region_id: str) -> Tuple[Dict[str, object], str, Dict[str, float]]:
    obj_xy = world_state.object_map()[object_id].pose.xy()
    goal_xy = world_state.goal_map()[goal_region_id].center
    direction = _unit(goal_xy[0] - obj_xy[0], goal_xy[1] - obj_xy[1])
    lateral = (-direction[1], direction[0])
    ideal_push = _offset(obj_xy, direction, -1.1)
    block_candidates = [
        _add(_offset(obj_xy, direction, 1.1), _scale(lateral, 0.9)),
        _add(_offset(obj_xy, direction, 1.1), _scale(lateral, -0.9)),
    ]
    robot_ids = sorted(robot.robot_id for robot in world_state.robots)
    scored: List[Tuple[float, str, str, Vec2, Dict[str, float]]] = []

    for pusher, blocker in permutations(robot_ids, 2):
        pusher_pose = world_state.robot_map()[pusher].pose.xy()
        blocker_pose = world_state.robot_map()[blocker].pose.xy()
        best_block_target = min(block_candidates, key=lambda target: distance_xy(blocker_pose, target))
        pusher_cost = distance_xy(pusher_pose, ideal_push) + 0.35 * distance_xy(pusher_pose, obj_xy)
        blocker_cost = distance_xy(blocker_pose, best_block_target) + 0.2 * distance_xy(best_block_target, goal_xy)
        total = pusher_cost + blocker_cost
        scored.append(
            (
                total,
                pusher,
                blocker,
                best_block_target,
                {
                    "pusher_cost": round(pusher_cost, 3),
                    "blocker_cost": round(blocker_cost, 3),
                    "total_cost": round(total, 3),
                },
            )
        )

    best_total, pusher, blocker, block_target, metrics = sorted(scored, key=lambda item: (item[0], item[1], item[2]))[0]
    funnel_target = _add(block_target, _scale(direction, 0.55))
    roles = {
        "pusher": pusher,
        "blocker": blocker,
        "setup_targets": {
            pusher: [round(ideal_push[0], 3), round(ideal_push[1], 3)],
            blocker: [round(block_target[0], 3), round(block_target[1], 3)],
        },
        "funnel_target": [round(funnel_target[0], 3), round(funnel_target[1], 3)],
    }
    rationale = (
        f"Herding roles minimize approach cost to a push-behind pose and a goal-side funnel pose; "
        f"{pusher} becomes pusher and {blocker} becomes blocker."
    )
    metrics["joint_cost"] = round(best_total, 3)
    return roles, rationale, metrics


def choose_search_roles(world_state, search_region_ids: Sequence[str], target_region_id: str) -> Tuple[Dict[str, object], str, Dict[str, float]]:
    robot_ids = sorted(robot.robot_id for robot in world_state.robots)
    target_xy = world_state.goal_map()[target_region_id].center
    scored: List[Tuple[float, Dict[str, str], Dict[str, float]]] = []

    for region_a, region_b in permutations(search_region_ids, 2):
        assignment = {robot_ids[0]: region_a, robot_ids[1]: region_b}
        travel = sum(distance_xy(world_state.robot_map()[robot_id].pose.xy(), world_state.goal_map()[region_id].center) for robot_id, region_id in assignment.items())
        spread = distance_xy(world_state.goal_map()[region_a].center, world_state.goal_map()[region_b].center)
        target_bias = min(distance_xy(world_state.goal_map()[region_a].center, target_xy), distance_xy(world_state.goal_map()[region_b].center, target_xy))
        total = travel + 0.15 * target_bias - 0.2 * spread
        scored.append(
            (
                total,
                assignment,
                {
                    "travel_cost": round(travel, 3),
                    "spread_bonus": round(spread, 3),
                    "target_bias": round(target_bias, 3),
                    "total_cost": round(total, 3),
                },
            )
        )

    _, assignments, metrics = sorted(scored, key=lambda item: (item[0], tuple(sorted(item[1].items()))))[0]
    finder = sorted(
        assignments.keys(),
        key=lambda robot_id: (
            distance_xy(world_state.goal_map()[assignments[robot_id]].center, target_xy),
            distance_xy(world_state.robot_map()[robot_id].pose.xy(), world_state.goal_map()[assignments[robot_id]].center),
            robot_id,
        ),
    )[0]
    converger = next(robot_id for robot_id in robot_ids if robot_id != finder)
    roles = {"finder": finder, "converger": converger, "search_assignments": assignments}
    rationale = (
        f"Search roles minimize joint travel while keeping sectors distinct; "
        f"{finder} is assigned the sector closest to the expected target region and {converger} covers the complementary sector."
    )
    return roles, rationale, metrics


def choose_relay_handoff_region(world_state, object_id: str, goal_region_id: str, preferred_region_id: str | None = None) -> Tuple[str | None, str, Dict[str, float]]:
    object_xy = world_state.object_map()[object_id].pose.xy()
    goal_xy = world_state.goal_map()[goal_region_id].center
    midpoint_xy = midpoint(object_xy, goal_xy)
    candidates = [goal for goal in world_state.goals if goal.region_id != goal_region_id]
    if not candidates:
        return None, "No candidate handoff region is available in the current world state.", {}

    scored: List[Tuple[float, str, Dict[str, float]]] = []
    for region in candidates:
        directness = distance_xy(object_xy, region.center) + distance_xy(region.center, goal_xy)
        midpoint_offset = distance_xy(region.center, midpoint_xy)
        semantic_bonus = -0.4 if region.semantic_label == "handoff_region" else 0.0
        preferred_bonus = -0.2 if region.region_id == preferred_region_id else 0.0
        total = directness + 0.35 * midpoint_offset + semantic_bonus + preferred_bonus
        scored.append(
            (
                total,
                region.region_id,
                {
                    "directness_cost": round(directness, 3),
                    "midpoint_offset": round(midpoint_offset, 3),
                    "total_cost": round(total, 3),
                },
            )
        )

    _, region_id, metrics = sorted(scored, key=lambda item: (item[0], item[1]))[0]
    rationale = f"Handoff region {region_id} is selected as the best midpoint-like transfer region between payload and goal."
    return region_id, rationale, metrics


def choose_relay_roles(world_state, object_id: str, handoff_region_id: str, goal_region_id: str) -> Tuple[Dict[str, object], str, Dict[str, float]]:
    robot_ids = sorted(robot.robot_id for robot in world_state.robots)
    object_xy = world_state.object_map()[object_id].pose.xy()
    handoff_xy = world_state.goal_map()[handoff_region_id].center
    goal_xy = world_state.goal_map()[goal_region_id].center
    scored: List[Tuple[float, str, str, Dict[str, float]]] = []

    for starter, finisher in permutations(robot_ids, 2):
        starter_pose = world_state.robot_map()[starter].pose.xy()
        finisher_pose = world_state.robot_map()[finisher].pose.xy()
        starter_cost = distance_xy(starter_pose, object_xy) + 0.55 * distance_xy(starter_pose, handoff_xy)
        finisher_cost = distance_xy(finisher_pose, handoff_xy) + 0.85 * distance_xy(finisher_pose, goal_xy)
        total = starter_cost + finisher_cost
        scored.append(
            (
                total,
                starter,
                finisher,
                {
                    "starter_cost": round(starter_cost, 3),
                    "finisher_cost": round(finisher_cost, 3),
                    "total_cost": round(total, 3),
                },
            )
        )

    best_total, starter, finisher, metrics = sorted(scored, key=lambda item: (item[0], item[1], item[2]))[0]
    clearance_position = compute_relay_clearance_position(world_state, starter, handoff_region_id, goal_region_id)
    roles = {
        "starter": starter,
        "finisher": finisher,
        "clearance_position": [round(clearance_position[0], 3), round(clearance_position[1], 3)],
    }
    rationale = (
        f"Relay roles minimize joint cost from payload to handoff and from handoff to goal; "
        f"{starter} starts at the payload and {finisher} is best placed to finish toward the goal."
    )
    metrics["joint_cost"] = round(best_total, 3)
    return roles, rationale, metrics


def compute_relay_clearance_position(world_state, robot_id: str, handoff_region_id: str, goal_region_id: str) -> Vec2:
    handoff_xy = world_state.goal_map()[handoff_region_id].center
    goal_xy = world_state.goal_map()[goal_region_id].center
    starter_xy = world_state.robot_map()[robot_id].pose.xy()
    direction = _unit(goal_xy[0] - handoff_xy[0], goal_xy[1] - handoff_xy[1])
    lateral = (-direction[1], direction[0])
    candidate_positions = [
        _clamp_point(world_state.bounds, _add(_offset(handoff_xy, direction, -1.1), _scale(lateral, 1.0))),
        _clamp_point(world_state.bounds, _add(_offset(handoff_xy, direction, -1.1), _scale(lateral, -1.0))),
    ]
    return min(candidate_positions, key=lambda xy: distance_xy(starter_xy, xy))


def relay_handoff_quality(world_state, object_id: str, goal_region_id: str, handoff_region_id: str) -> Dict[str, float]:
    object_xy = world_state.object_map()[object_id].pose.xy()
    goal_xy = world_state.goal_map()[goal_region_id].center
    handoff_xy = world_state.goal_map()[handoff_region_id].center
    direct = distance_xy(object_xy, goal_xy)
    via_handoff = distance_xy(object_xy, handoff_xy) + distance_xy(handoff_xy, goal_xy)
    midpoint_offset = distance_xy(handoff_xy, midpoint(object_xy, goal_xy))
    return {
        "direct_distance": round(direct, 3),
        "via_handoff_distance": round(via_handoff, 3),
        "midpoint_offset": round(midpoint_offset, 3),
        "detour_ratio": round(via_handoff / direct, 3) if direct > 0.0 else 1.0,
    }


def _unit(dx: float, dy: float) -> Vec2:
    norm = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    return (dx / norm, dy / norm)


def _scale(vector: Vec2, scalar: float) -> Vec2:
    return (vector[0] * scalar, vector[1] * scalar)


def _add(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] + b[0], a[1] + b[1])


def _offset(origin: Vec2, direction: Vec2, distance: float) -> Vec2:
    return (origin[0] + direction[0] * distance, origin[1] + direction[1] * distance)


def _clamp_point(bounds: Sequence[float], point: Vec2) -> Vec2:
    return (clamp(point[0], 0.0, bounds[0]), clamp(point[1], 0.0, bounds[1]))
