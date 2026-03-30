"""Semantic sanity checks for task-specific FSM quality."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from vcomt2d.core.types import distance_xy, midpoint
from vcomt2d.fsm.schema import FSMPlan, StateSpec
from .models import PlanningRequest, RoleAssignment, SceneFacts, SemanticCheckResult, SemanticIssue, TaskType
from .role_assignment import assign_roles


def _unique(items: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


@dataclass
class SemanticRepairResult:
    plan: FSMPlan
    roles: RoleAssignment
    scene_facts: SceneFacts
    applied_repairs: List[str] = field(default_factory=list)


class SemanticSanityChecker:
    """Task-aware semantic validation that complements structural FSM checks."""

    def check(self, task_type: TaskType, plan: FSMPlan, world_state, scene_facts: SceneFacts, roles: RoleAssignment) -> SemanticCheckResult:
        handler = getattr(self, f"_check_{task_type.name.lower()}", None)
        if handler is None:
            return SemanticCheckResult(
                passed=False,
                errors=[SemanticIssue("semantic_checker_missing", f"No semantic checker is implemented for task {task_type.value}.", repairable=False)],
            )
        return handler(plan, world_state, scene_facts, roles)

    def _check_t1_door_wedge_pass_through(self, plan: FSMPlan, world_state, scene_facts: SceneFacts, roles: RoleAssignment) -> SemanticCheckResult:
        errors: List[SemanticIssue] = []
        state_map = plan.state_map()
        door = world_state.door_map().get(scene_facts.relevant_door_id)
        goal = world_state.goal_map().get(scene_facts.goal_region_id)
        expected_roles = assign_roles(scene_facts, world_state).roles

        if door is None:
            errors.append(SemanticIssue("door_missing", "Referenced door does not exist in world_state.", repairable=False))
        elif not door.wedgeable:
            errors.append(SemanticIssue("door_not_wedgeable", "Door task references a non-wedgeable door.", repairable=False))
        if goal is None:
            errors.append(SemanticIssue("goal_region_missing", "Door task requires a valid target-side goal region.", repairable=False))

        hold_state = state_map.get("S1_HOLD_DOOR")
        pass_state = state_map.get("S2_PASS_PARTNER")
        follow_state = state_map.get("S3_FOLLOW")
        if hold_state is None or pass_state is None or follow_state is None:
            errors.append(SemanticIssue("door_state_pattern_invalid", "Door task FSM is missing hold/pass/follow states.", repairable=True, suggested_repair="resynthesize_task_template"))
            return self._result(errors)

        holder = self._robot_with_skill(hold_state, "HOLD_POSITION")
        waiter = self._robot_with_skill(hold_state, "WAIT_UNTIL")
        mover = self._robot_with_skill(pass_state, "MOVE_TO", target_id=scene_facts.goal_region_id)
        if holder is None or waiter is None or mover is None:
            errors.append(SemanticIssue("door_role_actions_missing", "Door task must include one holder, one waiter, and one pass-through mover.", repairable=True, suggested_repair="resynthesize_task_template"))
        else:
            if holder == waiter:
                errors.append(SemanticIssue("door_contradictory_role", "The same robot cannot both hold the door and wait for it to be held.", repairable=True, suggested_repair="reassign_door_roles"))
            if holder != expected_roles["holder"]:
                errors.append(SemanticIssue("door_wrong_holder", "Door holder is not the geometrically preferred robot.", repairable=True, suggested_repair="reassign_door_roles"))
            if mover != waiter:
                errors.append(SemanticIssue("door_pass_order_inconsistent", "The robot that waited for the door is not the one passing through first.", repairable=True, suggested_repair="relink_pass_sequence"))

        if not self._has_transition(pass_state, "S3_FOLLOW", kind="robot_in_region", robot=expected_roles["passer"], region_id=scene_facts.goal_region_id):
            errors.append(SemanticIssue("door_pass_condition_invalid", "Door pass-through state must transition when the passer reaches the target side.", repairable=True, suggested_repair="relink_pass_sequence", state_id="S2_PASS_PARTNER"))
        if not self._has_transition(follow_state, "S_DONE", kind="both_in_region", region_id=scene_facts.goal_region_id):
            errors.append(SemanticIssue("door_done_condition_invalid", "Door completion must require both robots reaching the target-side region.", repairable=True, suggested_repair="restore_terminal_condition", state_id="S3_FOLLOW"))

        return self._result(errors, debug_info={"expected_roles": expected_roles, "inferred_holder": holder, "inferred_waiter": waiter})

    def _check_t2_herding_corralling(self, plan: FSMPlan, world_state, scene_facts: SceneFacts, roles: RoleAssignment) -> SemanticCheckResult:
        errors: List[SemanticIssue] = []
        state_map = plan.state_map()
        target_obj = world_state.object_map().get(scene_facts.target_object_id)
        goal = world_state.goal_map().get(scene_facts.goal_region_id)
        expected_roles = assign_roles(scene_facts, world_state).roles

        if target_obj is None:
            errors.append(SemanticIssue("herding_object_missing", "Herding task requires a valid target object.", repairable=False))
        elif not target_obj.movable:
            errors.append(SemanticIssue("herding_object_immovable", "Herding target must be movable.", repairable=False))
        if goal is None:
            errors.append(SemanticIssue("herding_goal_missing", "Herding task requires a goal region.", repairable=False))

        setup_state = state_map.get("S0_FLANK_SETUP")
        push_state = state_map.get("S2_PUSH_AND_FUNNEL")
        if setup_state is None or push_state is None:
            errors.append(SemanticIssue("herding_state_pattern_invalid", "Herding task must include setup and push phases.", repairable=True, suggested_repair="resynthesize_task_template"))
            return self._result(errors)

        pusher = self._robot_with_skill(push_state, "PUSH", object_id=scene_facts.target_object_id)
        blocker = [robot_id for robot_id in ("robot_a", "robot_b") if robot_id != pusher][0] if pusher else None
        if pusher is None:
            errors.append(SemanticIssue("herding_no_pusher", "Herding push phase must assign one robot to PUSH the target object.", repairable=True, suggested_repair="reassign_herding_roles"))
        elif pusher != expected_roles["pusher"]:
            errors.append(SemanticIssue("herding_wrong_pusher", "Assigned herding pusher is not the geometrically preferred robot.", repairable=True, suggested_repair="reassign_herding_roles"))

        setup_targets = [self._action_target_id(getattr(setup_state, robot_id)) for robot_id in ("robot_a", "robot_b")]
        if None in setup_targets or setup_targets[0] == setup_targets[1]:
            errors.append(SemanticIssue("herding_setup_not_partitioned", "Herding setup should place robots at differentiated staging targets before pushing.", repairable=True, suggested_repair="rebuild_setup_phase"))

        if target_obj is not None and goal is not None and blocker is not None:
            blocker_action = getattr(push_state, blocker)
            if blocker_action is not None and "target_id" in blocker_action.params:
                blocker_target = self._resolve_target_xy(world_state, blocker_action)
            else:
                blocker_target = self._resolve_target_xy(world_state, getattr(setup_state, blocker))
            pusher_target = self._resolve_target_xy(world_state, getattr(setup_state, pusher))
            goal_vector = (goal.center[0] - target_obj.pose.x, goal.center[1] - target_obj.pose.y)
            blocker_vector = (blocker_target[0] - target_obj.pose.x, blocker_target[1] - target_obj.pose.y)
            pusher_vector = (pusher_target[0] - target_obj.pose.x, pusher_target[1] - target_obj.pose.y)
            blocker_alignment = goal_vector[0] * blocker_vector[0] + goal_vector[1] * blocker_vector[1]
            pusher_alignment = goal_vector[0] * pusher_vector[0] + goal_vector[1] * pusher_vector[1]
            if blocker_alignment <= pusher_alignment:
                errors.append(SemanticIssue("herding_blocker_wrong_side", "Blocker/funnel robot is not positioned on the target-side constraint role.", repairable=True, suggested_repair="reassign_herding_roles"))

        if not self._has_transition(push_state, "S_DONE", kind="object_in_region", object_id=scene_facts.target_object_id, region_id=scene_facts.goal_region_id):
            errors.append(SemanticIssue("herding_done_condition_invalid", "Herding completion must depend on the object entering the goal region.", repairable=True, suggested_repair="restore_terminal_condition", state_id="S2_PUSH_AND_FUNNEL"))

        return self._result(errors, debug_info={"expected_roles": expected_roles, "inferred_pusher": pusher})

    def _check_t4_collaborative_search_converge(self, plan: FSMPlan, world_state, scene_facts: SceneFacts, roles: RoleAssignment) -> SemanticCheckResult:
        errors: List[SemanticIssue] = []
        state_map = plan.state_map()
        target_obj = world_state.object_map().get(scene_facts.target_object_id)
        goal = world_state.goal_map().get(scene_facts.goal_region_id)
        expected_roles = assign_roles(scene_facts, world_state).roles

        if target_obj is None:
            errors.append(SemanticIssue("search_target_missing", "Search task requires a target object.", repairable=False))
        if goal is None:
            errors.append(SemanticIssue("search_goal_missing", "Search task requires a converge region.", repairable=False))

        split_state = state_map.get("S0_SPLIT_SEARCH")
        track_state = state_map.get("S1_TRACK_TARGET")
        signal_state = state_map.get("S2_SIGNAL_FOUND")
        converge_state = state_map.get("S3_CONVERGE")
        if split_state is None or track_state is None or signal_state is None or converge_state is None:
            errors.append(SemanticIssue("search_state_pattern_invalid", "Search task must include split, track, signal, and converge phases.", repairable=True, suggested_repair="resynthesize_task_template"))
            return self._result(errors)

        split_targets = [self._action_target_id(getattr(split_state, robot_id)) for robot_id in ("robot_a", "robot_b")]
        if split_targets[0] == split_targets[1]:
            errors.append(SemanticIssue("search_same_region", "Both robots are assigned the same initial search region.", repairable=True, suggested_repair="reassign_search_regions", state_id="S0_SPLIT_SEARCH"))
        if any(target_id not in scene_facts.search_region_ids for target_id in split_targets if target_id is not None):
            errors.append(SemanticIssue("search_invalid_region", "Initial search targets must be drawn from the declared search regions.", repairable=True, suggested_repair="reassign_search_regions", state_id="S0_SPLIT_SEARCH"))

        finder = self._robot_with_skill(track_state, "TRACK_OBJECT", object_id=scene_facts.target_object_id)
        signaler = self._robot_with_skill(signal_state, "SIGNAL", message="target_found")
        waiter = self._robot_with_skill(signal_state, "WAIT_UNTIL")
        converger = self._robot_with_skill(converge_state, "MOVE_TO", target_id=scene_facts.goal_region_id)
        if finder is None or signaler is None or waiter is None or converger is None:
            errors.append(SemanticIssue("search_coordination_missing", "Search task must include explicit find/report/converge coordination.", repairable=True, suggested_repair="restore_search_coordination"))
        else:
            if finder != expected_roles["finder"]:
                errors.append(SemanticIssue("search_wrong_finder", "Primary finder is not aligned with the deterministic search heuristic.", repairable=True, suggested_repair="reassign_search_roles"))
            if signaler != finder:
                errors.append(SemanticIssue("search_reporter_mismatch", "The robot that finds the object should also report the discovery.", repairable=True, suggested_repair="restore_search_coordination"))
            if waiter != expected_roles["converger"] or converger != expected_roles["converger"]:
                errors.append(SemanticIssue("search_converger_mismatch", "Non-finder robot is not the one waiting for the report and converging afterward.", repairable=True, suggested_repair="reassign_search_roles"))

        if not self._has_transition(track_state, "S2_SIGNAL_FOUND", kind="object_found", object_id=scene_facts.target_object_id):
            errors.append(SemanticIssue("search_found_transition_invalid", "Search tracking must transition on object_found semantics.", repairable=True, suggested_repair="restore_search_coordination", state_id="S1_TRACK_TARGET"))
        if not self._has_transition(converge_state, "S_DONE", kind="both_in_region", region_id=scene_facts.goal_region_id):
            errors.append(SemanticIssue("search_done_condition_invalid", "Search completion must require both robots converging to the target region.", repairable=True, suggested_repair="restore_terminal_condition", state_id="S3_CONVERGE"))

        return self._result(errors, debug_info={"expected_roles": expected_roles, "split_targets": split_targets})

    def _check_t6_relay_delivery(self, plan: FSMPlan, world_state, scene_facts: SceneFacts, roles: RoleAssignment) -> SemanticCheckResult:
        errors: List[SemanticIssue] = []
        state_map = plan.state_map()
        target_obj = world_state.object_map().get(scene_facts.target_object_id)
        goal = world_state.goal_map().get(scene_facts.goal_region_id)
        handoff = world_state.goal_map().get(scene_facts.handoff_region_id) if scene_facts.handoff_region_id else None
        expected_roles = assign_roles(scene_facts, world_state).roles

        if target_obj is None:
            errors.append(SemanticIssue("relay_object_missing", "Relay task requires a valid payload object.", repairable=False))
        elif not target_obj.movable:
            errors.append(SemanticIssue("relay_object_immovable", "Relay payload must be movable.", repairable=False))
        if goal is None:
            errors.append(SemanticIssue("relay_goal_missing", "Relay task requires a goal region.", repairable=False))
        if handoff is None:
            errors.append(SemanticIssue("relay_handoff_missing", "Relay task requires a handoff region or a sensible inferred equivalent.", repairable=False))
        if target_obj is not None and goal is not None and handoff is not None:
            direct = distance_xy(target_obj.pose.xy(), goal.center)
            via_handoff = distance_xy(target_obj.pose.xy(), handoff.center) + distance_xy(handoff.center, goal.center)
            mid = midpoint(target_obj.pose.xy(), goal.center)
            if via_handoff > direct * 1.6 or distance_xy(handoff.center, mid) > direct * 0.55:
                errors.append(SemanticIssue("relay_handoff_implausible", "Handoff region is not plausibly between source and destination.", repairable=False))

        first_push = state_map.get("S1_FIRST_PUSH")
        handoff_sync = state_map.get("S2_HANDOFF_SYNC")
        second_push = state_map.get("S3_SECOND_PUSH")
        if first_push is None or handoff_sync is None or second_push is None:
            errors.append(SemanticIssue("relay_state_pattern_invalid", "Relay task must include first push, handoff, and second push phases.", repairable=True, suggested_repair="resynthesize_task_template"))
            return self._result(errors)

        starter = self._robot_with_skill(first_push, "PUSH", object_id=scene_facts.target_object_id, target_id=scene_facts.handoff_region_id)
        finisher = self._robot_with_skill(second_push, "PUSH", object_id=scene_facts.target_object_id, target_id=scene_facts.goal_region_id)
        signaler = self._robot_with_skill(handoff_sync, "SIGNAL", message="handoff_ready")
        waiter = self._robot_with_skill(handoff_sync, "WAIT_UNTIL")
        supporter = [robot_id for robot_id in ("robot_a", "robot_b") if robot_id != finisher][0] if finisher else None

        if starter is None or finisher is None or signaler is None or waiter is None:
            errors.append(SemanticIssue("relay_coordination_missing", "Relay task must contain explicit handoff and signal-based coordination.", repairable=True, suggested_repair="restore_relay_handoff"))
        else:
            if starter == finisher:
                errors.append(SemanticIssue("relay_not_a_relay", "Relay plan assigns both push phases to the same robot.", repairable=True, suggested_repair="reassign_relay_roles"))
            if starter != expected_roles["starter"]:
                errors.append(SemanticIssue("relay_wrong_starter", "Starter is not the preferred robot nearest the payload.", repairable=True, suggested_repair="reassign_relay_roles"))
            if finisher != expected_roles["finisher"]:
                errors.append(SemanticIssue("relay_wrong_finisher", "Finisher is not the preferred robot nearest the goal.", repairable=True, suggested_repair="reassign_relay_roles"))
            if signaler != starter or waiter != finisher:
                errors.append(SemanticIssue("relay_handoff_roles_inconsistent", "Starter should announce handoff and finisher should wait for that signal.", repairable=True, suggested_repair="restore_relay_handoff"))
            if getattr(second_push, supporter).skill not in {"RETREAT", "HOLD_POSITION"}:
                errors.append(SemanticIssue("relay_support_role_useless", "Non-finisher must explicitly yield or clear space during second push.", repairable=True, suggested_repair="restore_relay_handoff"))

        if not self._has_transition(first_push, "S2_HANDOFF_SYNC", kind="handoff_ready", region_id=scene_facts.handoff_region_id, object_id=scene_facts.target_object_id):
            errors.append(SemanticIssue("relay_handoff_condition_invalid", "Relay handoff must be triggered by the payload reaching the handoff region.", repairable=True, suggested_repair="restore_relay_handoff", state_id="S1_FIRST_PUSH"))
        if not self._has_transition(second_push, "S_DONE", kind="object_in_region", object_id=scene_facts.target_object_id, region_id=scene_facts.goal_region_id):
            errors.append(SemanticIssue("relay_done_condition_invalid", "Relay completion must depend on the payload reaching the goal region.", repairable=True, suggested_repair="restore_terminal_condition", state_id="S3_SECOND_PUSH"))

        return self._result(errors, debug_info={"expected_roles": expected_roles, "inferred_starter": starter, "inferred_finisher": finisher})

    def _result(self, errors: List[SemanticIssue], warnings: Optional[List[SemanticIssue]] = None, debug_info: Optional[Dict[str, object]] = None) -> SemanticCheckResult:
        warning_list = warnings or []
        suggested = _unique([issue.suggested_repair for issue in errors if issue.suggested_repair])
        return SemanticCheckResult(passed=not errors, warnings=warning_list, errors=errors, suggested_repairs=suggested, debug_info=debug_info or {})

    def _robot_with_skill(self, state: StateSpec, skill: str, **param_filters) -> Optional[str]:
        matches: List[str] = []
        for robot_id in ("robot_a", "robot_b"):
            action = getattr(state, robot_id)
            if action is None or action.skill != skill:
                continue
            if all(action.params.get(key) == value for key, value in param_filters.items()):
                matches.append(robot_id)
        return matches[0] if len(matches) == 1 else None

    def _has_transition(self, state: StateSpec, to: str, kind: str, **arg_filters) -> bool:
        for item in state.transitions:
            if item.to != to or item.condition.kind != kind:
                continue
            if all(item.condition.args.get(key) == value for key, value in arg_filters.items()):
                return True
        return False

    def _action_target_id(self, action) -> Optional[str]:
        if action is None:
            return None
        return action.params.get("target_id")

    def _resolve_target_xy(self, world_state, action) -> Tuple[float, float]:
        if action is None:
            raise KeyError("Missing action")
        if "target_position" in action.params:
            return tuple(action.params["target_position"])
        target_id = action.params.get("target_id")
        if target_id in world_state.goal_map():
            return world_state.goal_map()[target_id].center
        if target_id in world_state.door_map():
            return world_state.door_map()[target_id].pose.xy()
        if target_id in world_state.object_map():
            return world_state.object_map()[target_id].pose.xy()
        raise KeyError(f"Unknown target_id {target_id}")


class SemanticRepairer:
    """Explicit semantic repair path based on deterministic re-synthesis."""

    def repair(
        self,
        request: PlanningRequest,
        plan: FSMPlan,
        semantic_result: SemanticCheckResult,
        scene_facts: SceneFacts,
        roles: RoleAssignment,
        builder: Callable,
    ) -> SemanticRepairResult:
        repairable_errors = [issue for issue in semantic_result.errors if issue.repairable]
        if not repairable_errors:
            return SemanticRepairResult(plan=plan, roles=roles, scene_facts=scene_facts, applied_repairs=[])

        refreshed_roles = assign_roles(scene_facts, request.world_state)
        refreshed_reasoning = (
            f"{plan.reasoning_summary} Semantic repair re-synthesized the plan using deterministic scene facts and role heuristics. "
            f"Applied repairs={semantic_result.suggested_repairs}."
        )
        repaired_plan = builder(request.user_instruction, refreshed_reasoning, scene_facts, refreshed_roles, request.planning_config)
        applied_repairs = [f"semantic::{repair}" for repair in semantic_result.suggested_repairs] or ["semantic::resynthesize_task_template"]
        return SemanticRepairResult(plan=repaired_plan, roles=refreshed_roles, scene_facts=scene_facts, applied_repairs=applied_repairs)
