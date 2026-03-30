"""Structured 2D world-state schema for planning and execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from vcomt2d.core.types import Pose2D, Vec2, to_serializable


@dataclass
class RobotState:
    robot_id: str
    pose: Pose2D
    current_status: str = "idle"
    carrying: Optional[str] = None
    pushing: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return to_serializable(self)


@dataclass
class ObjectState:
    object_id: str
    object_type: str
    pose: Pose2D
    state_flags: Dict[str, Any] = field(default_factory=dict)
    color: Optional[str] = None
    semantic_tags: List[str] = field(default_factory=list)
    movable: bool = True
    radius: float = 0.35

    def to_dict(self) -> Dict[str, Any]:
        return to_serializable(self)


@dataclass
class DoorState:
    door_id: str
    pose: Pose2D
    is_open: bool = False
    wedgeable: bool = True
    passable: bool = False
    connects_regions: Tuple[str, str] = ("room_a", "room_b")
    radius: float = 0.45

    def to_dict(self) -> Dict[str, Any]:
        return to_serializable(self)


@dataclass
class RegionState:
    region_id: str
    center: Vec2
    radius: float
    semantic_label: str

    def to_dict(self) -> Dict[str, Any]:
        return to_serializable(self)


@dataclass
class ObstacleState:
    obstacle_id: str
    center: Vec2
    radius: float

    def to_dict(self) -> Dict[str, Any]:
        return to_serializable(self)


@dataclass
class TopologyLink:
    from_region: str
    to_region: str
    via: str

    def to_dict(self) -> Dict[str, Any]:
        return to_serializable(self)


@dataclass
class WorldState:
    world_id: str
    bounds: Vec2
    robots: List[RobotState]
    objects: List[ObjectState] = field(default_factory=list)
    doors: List[DoorState] = field(default_factory=list)
    goals: List[RegionState] = field(default_factory=list)
    obstacles: List[ObstacleState] = field(default_factory=list)
    topology: List[TopologyLink] = field(default_factory=list)
    task_facts: Dict[str, Any] = field(default_factory=dict)

    def robot_map(self) -> Dict[str, RobotState]:
        return {robot.robot_id: robot for robot in self.robots}

    def object_map(self) -> Dict[str, ObjectState]:
        return {obj.object_id: obj for obj in self.objects}

    def door_map(self) -> Dict[str, DoorState]:
        return {door.door_id: door for door in self.doors}

    def goal_map(self) -> Dict[str, RegionState]:
        return {goal.region_id: goal for goal in self.goals}

    def obstacle_map(self) -> Dict[str, ObstacleState]:
        return {obstacle.obstacle_id: obstacle for obstacle in self.obstacles}

    def to_dict(self) -> Dict[str, Any]:
        return to_serializable(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "WorldState":
        return WorldState(
            world_id=data["world_id"],
            bounds=tuple(data["bounds"]),
            robots=[
                RobotState(robot_id=item["robot_id"], pose=Pose2D.from_dict(item["pose"]), current_status=item.get("current_status", "idle"), carrying=item.get("carrying"), pushing=item.get("pushing"))
                for item in data.get("robots", [])
            ],
            objects=[
                ObjectState(
                    object_id=item["object_id"],
                    object_type=item["object_type"],
                    pose=Pose2D.from_dict(item["pose"]),
                    state_flags=item.get("state_flags", {}),
                    color=item.get("color"),
                    semantic_tags=item.get("semantic_tags", []),
                    movable=item.get("movable", True),
                    radius=item.get("radius", 0.35),
                )
                for item in data.get("objects", [])
            ],
            doors=[
                DoorState(
                    door_id=item["door_id"],
                    pose=Pose2D.from_dict(item["pose"]),
                    is_open=item.get("is_open", False),
                    wedgeable=item.get("wedgeable", True),
                    passable=item.get("passable", False),
                    connects_regions=tuple(item.get("connects_regions", ("room_a", "room_b"))),
                    radius=item.get("radius", 0.45),
                )
                for item in data.get("doors", [])
            ],
            goals=[RegionState(region_id=item["region_id"], center=tuple(item["center"]), radius=item["radius"], semantic_label=item["semantic_label"]) for item in data.get("goals", [])],
            obstacles=[ObstacleState(obstacle_id=item["obstacle_id"], center=tuple(item["center"]), radius=item["radius"]) for item in data.get("obstacles", [])],
            topology=[TopologyLink(from_region=item["from_region"], to_region=item["to_region"], via=item["via"]) for item in data.get("topology", [])],
            task_facts=data.get("task_facts", {}),
        )

