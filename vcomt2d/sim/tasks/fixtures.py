"""Deterministic world-state fixtures for supported task families."""

from __future__ import annotations

from vcomt2d.core.types import Pose2D
from vcomt2d.sim.entities import DoorState, ObjectState, ObstacleState, RegionState, RobotState, TopologyLink, WorldState


def door_task_world() -> WorldState:
    return WorldState(
        world_id="door_demo_world",
        bounds=(12.0, 10.0),
        robots=[
            RobotState("robot_a", Pose2D(1.0, 4.5)),
            RobotState("robot_b", Pose2D(2.0, 2.0)),
        ],
        doors=[DoorState("door_main", Pose2D(5.0, 5.0), is_open=False, wedgeable=True, passable=False, connects_regions=("room_start", "room_goal"))],
        goals=[
            RegionState("door_wait", (4.2, 4.2), 0.6, "door_wait"),
            RegionState("room_goal", (9.2, 5.0), 1.1, "goal_room"),
        ],
        topology=[TopologyLink("room_start", "room_goal", "door_main")],
        task_facts={"goal_region_id": "room_goal"},
    )


def herding_task_world() -> WorldState:
    return WorldState(
        world_id="herding_demo_world",
        bounds=(12.0, 12.0),
        robots=[
            RobotState("robot_a", Pose2D(2.0, 2.0)),
            RobotState("robot_b", Pose2D(3.0, 9.0)),
        ],
        objects=[ObjectState("ball_1", "ball", Pose2D(6.0, 6.0), color="blue", semantic_tags=["ball"], radius=0.35)],
        goals=[
            RegionState("goal_corner", (10.2, 10.2), 1.0, "goal_area"),
            RegionState("flank_left", (5.0, 7.0), 0.5, "staging"),
            RegionState("flank_right", (7.0, 5.0), 0.5, "staging"),
            RegionState("funnel_point", (8.0, 7.5), 0.7, "funnel"),
        ],
        obstacles=[ObstacleState("obstacle_1", (7.5, 7.0), 0.7)],
        task_facts={"target_object_id": "ball_1", "goal_region_id": "goal_corner"},
    )


def search_task_world() -> WorldState:
    return WorldState(
        world_id="search_demo_world",
        bounds=(12.0, 12.0),
        robots=[
            RobotState("robot_a", Pose2D(1.0, 1.0)),
            RobotState("robot_b", Pose2D(1.0, 10.0)),
        ],
        objects=[ObjectState("box_red", "box", Pose2D(10.0, 9.0), color="red", semantic_tags=["target", "search_target"], radius=0.45)],
        goals=[
            RegionState("search_sector_a", (3.0, 9.0), 0.8, "search_sector"),
            RegionState("search_sector_b", (8.0, 3.0), 0.8, "search_sector"),
            RegionState("target_region", (10.0, 9.0), 1.0, "target_region"),
        ],
        task_facts={"target_object_id": "box_red", "search_region_ids": ["search_sector_a", "search_sector_b"], "target_region_id": "target_region"},
    )


def relay_task_world() -> WorldState:
    return WorldState(
        world_id="relay_demo_world",
        bounds=(12.0, 12.0),
        robots=[
            RobotState("robot_a", Pose2D(2.0, 9.0)),
            RobotState("robot_b", Pose2D(8.0, 5.0)),
        ],
        objects=[ObjectState("box_1", "box", Pose2D(3.0, 8.5), color="gray", semantic_tags=["payload"], radius=0.45)],
        goals=[
            RegionState("handoff_mid", (6.0, 6.0), 0.9, "handoff_region"),
            RegionState("far_goal", (10.0, 2.0), 1.1, "goal_region"),
        ],
        task_facts={"target_object_id": "box_1", "handoff_region_id": "handoff_mid", "goal_region_id": "far_goal"},
    )

