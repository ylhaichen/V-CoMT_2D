"""Minimal deterministic 2D execution world."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from vcomt2d.core.types import Pose2D, clamp, distance_pose, distance_xy, midpoint, to_serializable
from .entities import DoorState, ObjectState, RegionState, RobotState, WorldState


@dataclass
class SimWorld:
    """Runtime wrapper around WorldState with blackboard and event log."""

    state: WorldState
    tick: int = 0
    blackboard: Dict[str, Any] = field(default_factory=dict)
    event_log: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.blackboard.setdefault("signals", [])
        self.blackboard.setdefault("flags", {})
        self.blackboard.setdefault("sync_points", {})

    def robots(self) -> Dict[str, RobotState]:
        return self.state.robot_map()

    def objects(self) -> Dict[str, ObjectState]:
        return self.state.object_map()

    def doors(self) -> Dict[str, DoorState]:
        return self.state.door_map()

    def goals(self) -> Dict[str, RegionState]:
        return self.state.goal_map()

    def move_robot_towards(self, robot_id: str, target_xy, speed: float = 0.8) -> bool:
        robot_map = self.robots()
        robot = robot_map[robot_id]
        reached = self._move_pose_towards(robot.pose, target_xy, speed)
        robot_map[robot_id] = RobotState(
            robot_id=robot.robot_id,
            pose=reached["pose"],
            current_status=robot.current_status,
            carrying=robot.carrying,
            pushing=robot.pushing,
        )
        self.state.robots = list(robot_map.values())
        return reached["done"]

    def move_object_towards(self, object_id: str, target_xy, speed: float = 0.6) -> bool:
        object_map = self.objects()
        obj = object_map[object_id]
        reached = self._move_pose_towards(obj.pose, target_xy, speed)
        object_map[object_id] = ObjectState(
            object_id=obj.object_id,
            object_type=obj.object_type,
            pose=reached["pose"],
            state_flags=dict(obj.state_flags),
            color=obj.color,
            semantic_tags=list(obj.semantic_tags),
            movable=obj.movable,
            radius=obj.radius,
        )
        self.state.objects = list(object_map.values())
        return reached["done"]

    def set_flag(self, flag: str, value: Any = True) -> None:
        self.blackboard.setdefault("flags", {})[flag] = value

    def get_flag(self, flag: str) -> Any:
        return self.blackboard.setdefault("flags", {}).get(flag)

    def add_signal(self, sender: str, message: str) -> None:
        self.blackboard.setdefault("signals", []).append({"sender": sender, "message": message, "tick": self.tick})
        self.event_log.append(f"t={self.tick}: {sender} signaled {message}")

    def has_signal(self, message: str, from_robot: Optional[str] = None) -> bool:
        return any(signal["message"] == message and (from_robot is None or signal["sender"] == from_robot) for signal in self.blackboard.get("signals", []))

    def set_sync(self, sync_key: str, robot_id: str) -> bool:
        sync_state = self.blackboard.setdefault("sync_points", {}).setdefault(sync_key, set())
        sync_state.add(robot_id)
        return len(sync_state) >= len(self.state.robots)

    def region_contains_robot(self, region_id: str, robot_id: str) -> bool:
        region = self.goals()[region_id]
        robot = self.robots()[robot_id]
        return distance_xy(region.center, robot.pose.xy()) <= region.radius

    def region_contains_object(self, region_id: str, object_id: str) -> bool:
        region = self.goals()[region_id]
        obj = self.objects()[object_id]
        return distance_xy(region.center, obj.pose.xy()) <= region.radius

    def snapshot(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "world_state": self.state.to_dict(),
            "blackboard": to_serializable(self.blackboard),
            "events": list(self.event_log[-12:]),
        }

    def advance_tick(self) -> None:
        self.tick += 1

    def _move_pose_towards(self, pose: Pose2D, target_xy, speed: float) -> Dict[str, Any]:
        dist = distance_xy(pose.xy(), target_xy)
        if dist <= speed:
            return {"pose": Pose2D(target_xy[0], target_xy[1], pose.theta), "done": True}
        dx = (target_xy[0] - pose.x) / dist
        dy = (target_xy[1] - pose.y) / dist
        width, height = self.state.bounds
        next_pose = Pose2D(clamp(pose.x + dx * speed, 0.0, width), clamp(pose.y + dy * speed, 0.0, height), pose.theta)
        return {"pose": next_pose, "done": False}
