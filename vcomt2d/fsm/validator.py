"""Strict structural validator for multi-robot FSM plans."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Set

from vcomt2d.core.skills import VALID_SKILLS
from .schema import FSMPlan


VALID_CONDITIONS = {
    "timeout",
    "all_actions_done",
    "robot_in_region",
    "both_in_region",
    "object_in_region",
    "signal_received",
    "object_found",
    "handoff_ready",
    "flag_true",
}


@dataclass
class ValidationError:
    code: str
    message: str
    state_id: str | None = None
    path: str | None = None

    def to_dict(self) -> Dict[str, str | None]:
        return {"code": self.code, "message": self.message, "state_id": self.state_id, "path": self.path}


@dataclass
class ValidationResult:
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {"valid": self.valid, "errors": [error.to_dict() for error in self.errors]}


def validate_fsm(plan: FSMPlan) -> ValidationResult:
    errors: List[ValidationError] = []
    state_map = plan.state_map()

    if not plan.initial_state:
        errors.append(ValidationError("missing_initial_state", "FSM initial_state is empty", path="fsm.initial_state"))
    if plan.initial_state and plan.initial_state not in state_map:
        errors.append(ValidationError("missing_initial_state", "FSM initial_state does not reference an existing state", path="fsm.initial_state"))

    state_ids = [state.state_id for state in plan.states]
    duplicates = {state_id for state_id in state_ids if state_ids.count(state_id) > 1}
    for duplicate in sorted(duplicates):
        errors.append(ValidationError("duplicate_state_id", f"State '{duplicate}' is defined multiple times", state_id=duplicate))

    terminal_statuses = {state.status for state in plan.states if state.terminal and state.status}
    if "success" not in terminal_statuses:
        errors.append(ValidationError("missing_success_terminal", "FSM must contain a success terminal state"))
    if "failure" not in terminal_statuses:
        errors.append(ValidationError("missing_failure_terminal", "FSM must contain a failure terminal state"))

    for state in plan.states:
        if not state.state_id:
            errors.append(ValidationError("invalid_state_name", "State name must be non-empty"))
        _validate_state(state, state_map, errors)

    if plan.initial_state in state_map:
        reachable = _reachable_states(plan.initial_state, state_map)
        for state_id in sorted(state_map.keys() - reachable):
            errors.append(ValidationError("orphan_state", f"State '{state_id}' is unreachable from initial_state", state_id=state_id))

    return ValidationResult(valid=not errors, errors=errors)


def _validate_state(state, state_map, errors):
    if state.terminal:
        if state.status not in {"success", "failure"}:
            errors.append(ValidationError("invalid_terminal_status", "Terminal state status must be 'success' or 'failure'", state_id=state.state_id))
        return

    if state.robot_a is None:
        errors.append(ValidationError("missing_robot_action", "robot_a action missing", state_id=state.state_id, path=f"{state.state_id}.robot_a"))
    if state.robot_b is None:
        errors.append(ValidationError("missing_robot_action", "robot_b action missing", state_id=state.state_id, path=f"{state.state_id}.robot_b"))

    for robot_key in ("robot_a", "robot_b"):
        action = getattr(state, robot_key)
        if action is None:
            continue
        if action.skill not in VALID_SKILLS:
            errors.append(ValidationError("invalid_skill", f"Skill '{action.skill}' is not supported", state_id=state.state_id, path=f"{state.state_id}.{robot_key}.skill"))
        if not isinstance(action.params, dict):
            errors.append(ValidationError("invalid_params", "Action params must be a dict", state_id=state.state_id, path=f"{state.state_id}.{robot_key}.params"))
        else:
            _validate_action_params(state.state_id, robot_key, action.skill, action.params, errors)

    if not state.transitions:
        errors.append(ValidationError("missing_transition", "Non-terminal state requires at least one transition", state_id=state.state_id))
        return

    has_timeout = False
    for transition in state.transitions:
        if transition.to not in state_map:
            errors.append(ValidationError("missing_target_state", f"Transition target '{transition.to}' does not exist", state_id=state.state_id))
        if transition.condition.kind not in VALID_CONDITIONS:
            errors.append(ValidationError("invalid_condition", f"Condition '{transition.condition.kind}' is not supported", state_id=state.state_id))
        if transition.condition.kind == "timeout":
            has_timeout = True
            if "ticks" not in transition.condition.args:
                errors.append(ValidationError("missing_timeout_ticks", "timeout condition requires ticks", state_id=state.state_id))
    if not has_timeout:
        errors.append(ValidationError("missing_timeout_transition", "Non-terminal state requires timeout coverage", state_id=state.state_id))


def _validate_action_params(state_id, robot_key, skill, params, errors):
    if skill in {"MOVE_TO", "RETREAT"} and not ({"target_id", "target_position"} & set(params.keys())):
        errors.append(ValidationError("invalid_params", f"{skill} requires target_id or target_position", state_id=state_id, path=f"{state_id}.{robot_key}.params"))
    if skill == "PUSH" and not {"object_id", "target_id"} <= set(params.keys()):
        errors.append(ValidationError("invalid_params", "PUSH requires object_id and target_id", state_id=state_id, path=f"{state_id}.{robot_key}.params"))
    if skill == "FOLLOW" and "target_robot" not in params:
        errors.append(ValidationError("invalid_params", "FOLLOW requires target_robot", state_id=state_id, path=f"{state_id}.{robot_key}.params"))
    if skill == "SIGNAL" and "message" not in params:
        errors.append(ValidationError("invalid_params", "SIGNAL requires message", state_id=state_id, path=f"{state_id}.{robot_key}.params"))
    if skill == "WAIT_UNTIL" and "condition" not in params:
        errors.append(ValidationError("invalid_params", "WAIT_UNTIL requires condition", state_id=state_id, path=f"{state_id}.{robot_key}.params"))
    if skill == "TRACK_OBJECT" and "object_id" not in params:
        errors.append(ValidationError("invalid_params", "TRACK_OBJECT requires object_id", state_id=state_id, path=f"{state_id}.{robot_key}.params"))


def _reachable_states(initial_state: str, state_map: Dict[str, object]) -> Set[str]:
    visited: Set[str] = set()
    queue = deque([initial_state])
    while queue:
        state_id = queue.popleft()
        if state_id in visited or state_id not in state_map:
            continue
        visited.add(state_id)
        for transition in state_map[state_id].transitions:
            queue.append(transition.to)
    return visited
