"""Deterministic FSM executor for demo validation and animation traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from vcomt2d.core.logging import ExecutionTrace, TraceFrame
from vcomt2d.core.skills import SkillName
from vcomt2d.core.types import Pose2D, distance_xy
from vcomt2d.sim.backend import SimBackend
from vcomt2d.sim.world import SimWorld
from .schema import ActionSpec, ConditionSpec, FSMPlan, StateSpec
from .validator import validate_fsm


@dataclass
class ExecutionConfig:
    max_ticks: int = 120
    default_move_speed: float = 0.8
    default_push_speed: float = 0.55


@dataclass
class ExecutionResult:
    success: bool
    status: str
    terminal_state: str
    trace: ExecutionTrace
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "terminal_state": self.terminal_state,
            "trace": self.trace.to_dict(),
            "failure_reason": self.failure_reason,
        }


class FSMExecutor:
    """Simple executor that interprets the shared skill vocabulary."""

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.backend = SimBackend()

    def execute(self, request_id: str, task_type: str, plan: FSMPlan, world_state) -> ExecutionResult:
        validation = validate_fsm(plan)
        if not validation.valid:
            trace = ExecutionTrace(request_id=request_id, task_type=task_type, status="validation_failed")
            trace.events.append({"type": "validation_failed", "errors": validation.to_dict()["errors"]})
            return ExecutionResult(False, "validation_failed", "N/A", trace, failure_reason="plan_validation_failed")

        world = self.backend.create_world(world_state)
        state_map = plan.state_map()
        current = state_map[plan.initial_state]
        trace = ExecutionTrace(request_id=request_id, task_type=task_type, status="running")
        state_tick = 0

        while world.tick < self.config.max_ticks:
            if current.terminal:
                trace.status = current.status or "failure"
                trace.terminal_state = current.state_id
                return ExecutionResult(current.status == "success", current.status or "failure", current.state_id, trace)

            actions_done, action_events = self._apply_state_actions(world, current, state_tick)
            self._update_semantics(world, current)
            reason, next_state_id = self._evaluate_transitions(world, current, state_tick, actions_done)
            trace.frames.append(
                TraceFrame(
                    tick=world.tick,
                    state_id=current.state_id,
                    robot_actions={
                        "robot_a": current.robot_a.to_dict() if current.robot_a else {},
                        "robot_b": current.robot_b.to_dict() if current.robot_b else {},
                    },
                    world_snapshot=world.snapshot(),
                    transition_reason=reason,
                    next_state=next_state_id,
                    events=action_events,
                )
            )
            if next_state_id is not None:
                current = state_map[next_state_id]
                state_tick = 0
            else:
                state_tick += 1
            world.advance_tick()

        trace.status = "failure"
        trace.terminal_state = "TIMEOUT"
        trace.events.append({"type": "global_timeout", "tick": world.tick})
        return ExecutionResult(False, "failure", "TIMEOUT", trace, failure_reason="global_timeout")

    def _apply_state_actions(self, world: SimWorld, state: StateSpec, state_tick: int) -> Tuple[bool, List[str]]:
        events: List[str] = []
        done_a = self._execute_action(world, "robot_a", state.robot_a, state_tick, state.state_id, events)
        done_b = self._execute_action(world, "robot_b", state.robot_b, state_tick, state.state_id, events)
        return done_a and done_b, events

    def _execute_action(self, world: SimWorld, robot_id: str, action: ActionSpec, state_tick: int, state_id: str, events: List[str]) -> bool:
        if action.skill == SkillName.MOVE_TO.value:
            target_xy = self._resolve_target_xy(world, action.params)
            done = world.move_robot_towards(robot_id, target_xy, speed=action.params.get("speed", self.config.default_move_speed))
            return done
        if action.skill == SkillName.RETREAT.value:
            target_xy = self._resolve_target_xy(world, action.params)
            return world.move_robot_towards(robot_id, target_xy, speed=action.params.get("speed", self.config.default_move_speed))
        if action.skill == SkillName.HOLD_POSITION.value:
            required_ticks = action.params.get("ticks", 1)
            if state_tick + 1 >= required_ticks:
                events.append(f"{robot_id} completed HOLD_POSITION")
            return state_tick + 1 >= required_ticks
        if action.skill == SkillName.FOLLOW.value:
            target_robot = action.params["target_robot"]
            target_pose = world.robots()[target_robot].pose.xy()
            return world.move_robot_towards(robot_id, target_pose, speed=action.params.get("speed", self.config.default_move_speed))
        if action.skill == SkillName.SYNCHRONIZE.value:
            sync_key = f"{state_id}:{action.params.get('sync_key', 'sync')}"
            done = world.set_sync(sync_key, robot_id)
            return done
        if action.skill == SkillName.SIGNAL.value:
            message = action.params["message"]
            if not world.has_signal(message, from_robot=robot_id):
                world.add_signal(robot_id, message)
            return True
        if action.skill == SkillName.WAIT_UNTIL.value:
            return self._evaluate_condition(world, action.params["condition"], state_tick, actions_done=False)
        if action.skill == SkillName.TRACK_OBJECT.value:
            object_id = action.params["object_id"]
            object_map = world.objects()
            obj = object_map[object_id]
            done = world.move_robot_towards(robot_id, obj.pose.xy(), speed=action.params.get("speed", self.config.default_move_speed))
            if distance_xy(world.robots()[robot_id].pose.xy(), obj.pose.xy()) <= action.params.get("detection_radius", 0.9):
                obj.state_flags["found"] = True
                object_map[object_id] = obj
                world.state.objects = list(object_map.values())
            return done or obj.state_flags.get("found", False)
        if action.skill == SkillName.PUSH.value:
            object_id = action.params["object_id"]
            obj = world.objects()[object_id]
            robot = world.robots()[robot_id]
            if distance_xy(robot.pose.xy(), obj.pose.xy()) > 1.0:
                return world.move_robot_towards(robot_id, obj.pose.xy(), speed=self.config.default_move_speed)
            target_xy = self._resolve_target_xy(world, action.params)
            object_done = world.move_object_towards(object_id, target_xy, speed=action.params.get("speed", self.config.default_push_speed))
            world.move_robot_towards(robot_id, world.objects()[object_id].pose.xy(), speed=self.config.default_push_speed)
            return object_done
        raise ValueError(f"Unsupported skill at execution time: {action.skill}")

    def _evaluate_transitions(self, world: SimWorld, state: StateSpec, state_tick: int, actions_done: bool) -> Tuple[str, Optional[str]]:
        for transition in state.transitions:
            if self._evaluate_condition(world, transition.condition, state_tick, actions_done):
                return transition.notes or transition.condition.kind, transition.to
        return "running", None

    def _evaluate_condition(self, world: SimWorld, condition: ConditionSpec | Dict[str, Any], state_tick: int, actions_done: bool) -> bool:
        spec = condition if isinstance(condition, ConditionSpec) else ConditionSpec(kind=condition["kind"], args=condition.get("args", {}))
        if spec.kind == "timeout":
            return state_tick + 1 >= int(spec.args["ticks"])
        if spec.kind == "all_actions_done":
            return actions_done
        if spec.kind == "robot_in_region":
            return world.region_contains_robot(spec.args["region_id"], spec.args["robot"])
        if spec.kind == "both_in_region":
            return all(world.region_contains_robot(spec.args["region_id"], robot_id) for robot_id in world.robots())
        if spec.kind == "object_in_region":
            return world.region_contains_object(spec.args["region_id"], spec.args["object_id"])
        if spec.kind == "signal_received":
            return world.has_signal(spec.args["message"], spec.args.get("from_robot"))
        if spec.kind == "object_found":
            return bool(world.objects()[spec.args["object_id"]].state_flags.get("found"))
        if spec.kind == "handoff_ready":
            return world.region_contains_object(spec.args["region_id"], spec.args["object_id"])
        if spec.kind == "flag_true":
            return bool(world.get_flag(spec.args["flag"]))
        return False

    def _resolve_target_xy(self, world: SimWorld, params: Dict[str, Any]):
        if "target_position" in params:
            return tuple(params["target_position"])
        if "target_id" in params:
            target_id = params["target_id"]
            if target_id in world.goals():
                return world.goals()[target_id].center
            if target_id in world.doors():
                return world.doors()[target_id].pose.xy()
            if target_id in world.objects():
                return world.objects()[target_id].pose.xy()
        raise KeyError("Action params require target_position or target_id")

    def _update_semantics(self, world: SimWorld, state: StateSpec) -> None:
        if state.state_id == "S1_HOLD_DOOR":
            holder_id = state.notes.split("holder=")[-1] if "holder=" in state.notes else "robot_a"
            if world.robots()[holder_id]:
                world.set_flag("door_held", True)
                for door in world.state.doors:
                    door.passable = True
                    door.is_open = True
        if state.state_id == "S1_FIRST_PUSH":
            handoff_regions = [goal.region_id for goal in world.state.goals if goal.semantic_label == "handoff_region"]
            movable_objects = [obj.object_id for obj in world.state.objects if obj.movable]
            for region_id in handoff_regions:
                for object_id in movable_objects:
                    if world.region_contains_object(region_id, object_id):
                        world.set_flag("handoff_ready", True)
