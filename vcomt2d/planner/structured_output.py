"""Shared prompt, schema, and parsing helpers for model-based planner backends."""

from __future__ import annotations

import json
from typing import Any, Dict

from vcomt2d.core.skills import VALID_SKILLS


SUPPORTED_CONDITIONS = [
    "timeout",
    "all_actions_done",
    "robot_in_region",
    "both_in_region",
    "object_in_region",
    "signal_received",
    "object_found",
    "handoff_ready",
    "flag_true",
]


def _nullable(schema: Dict[str, Any]) -> Dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


CONDITION_ARGS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ticks", "robot", "region_id", "object_id", "message", "from_robot", "flag"],
    "properties": {
        "ticks": _nullable({"type": "integer"}),
        "robot": _nullable({"type": "string"}),
        "region_id": _nullable({"type": "string"}),
        "object_id": _nullable({"type": "string"}),
        "message": _nullable({"type": "string"}),
        "from_robot": _nullable({"type": "string"}),
        "flag": _nullable({"type": "string"}),
    },
}


CONDITION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "args"],
    "properties": {
        "kind": {"type": "string"},
        "args": CONDITION_ARGS_SCHEMA,
    },
}


ACTION_PARAMS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "target_id",
        "target_position",
        "object_id",
        "ticks",
        "speed",
        "target_robot",
        "sync_key",
        "message",
        "condition",
        "detection_radius",
    ],
    "properties": {
        "target_id": _nullable({"type": "string"}),
        "target_position": _nullable(
            {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "number"},
            }
        ),
        "object_id": _nullable({"type": "string"}),
        "ticks": _nullable({"type": "integer"}),
        "speed": _nullable({"type": "number"}),
        "target_robot": _nullable({"type": "string"}),
        "sync_key": _nullable({"type": "string"}),
        "message": _nullable({"type": "string"}),
        "condition": _nullable(CONDITION_SCHEMA),
        "detection_radius": _nullable({"type": "number"}),
    },
}


ACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["skill", "params"],
    "properties": {
        "skill": {"type": "string"},
        "params": ACTION_PARAMS_SCHEMA,
    },
}


TRANSITION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["to", "condition", "notes"],
    "properties": {
        "to": {"type": "string"},
        "condition": CONDITION_SCHEMA,
        "notes": {"type": "string"},
    },
}


PLANNER_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["task_description", "reasoning_summary", "fsm"],
    "properties": {
        "task_description": {"type": "string"},
        "reasoning_summary": {"type": "string"},
        "fsm": {
            "type": "object",
            "additionalProperties": False,
            "required": ["initial_state", "states"],
            "properties": {
                "initial_state": {"type": "string"},
                "states": {
                    "type": "array",
                    "minItems": 3,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["state_id", "robot_a", "robot_b", "transitions", "terminal", "status", "notes"],
                        "properties": {
                            "state_id": {"type": "string"},
                            "robot_a": _nullable(ACTION_SCHEMA),
                            "robot_b": _nullable(ACTION_SCHEMA),
                            "transitions": {
                                "type": "array",
                                "items": TRANSITION_SCHEMA,
                            },
                            "terminal": {"type": "boolean"},
                            "status": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "notes": {"type": "string"},
                        },
                    },
                },
            },
        },
    },
}


def build_planner_system_prompt() -> str:
    return (
        "You are a planner backend for V-CoMT_2D. "
        "Produce a multi-robot FSM plan for exactly two robots: robot_a and robot_b. "
        f"Allowed skills: {sorted(VALID_SKILLS)}. "
        f"Allowed condition kinds: {SUPPORTED_CONDITIONS}. "
        "Return only a JSON object that matches the supplied schema. "
        "Every non-terminal state must include actions for both robots, at least one success-path transition, and explicit timeout coverage to S_FAIL. "
        "Use readable state names such as S0_INIT, S1_APPROACH, S_DONE, and S_FAIL. "
        "Always include terminal success and failure states. "
        "Plan for collaboration: even if one robot is primary, the second robot must have a meaningful support role. "
        "Do not emit prose outside the JSON response."
    )


def build_planner_user_payload(context) -> Dict[str, Any]:
    return {
        "request_id": context.request.request_id,
        "instruction": context.request.user_instruction,
        "task_type_hint": None if context.intent.task_type is None else context.intent.task_type.value,
        "intent_keywords": context.intent.matched_keywords,
        "scene_facts_hint": context.scene_facts.to_dict(),
        "role_assignment_hint": context.roles.to_dict(),
        "world_state": context.request.world_state.to_dict(),
        "fsm_requirements": {
            "must_include_initial_state": True,
            "must_include_success_terminal": True,
            "must_include_failure_terminal": True,
            "must_assign_both_robots": True,
            "must_include_timeout_coverage": True,
            "must_use_allowed_skills_only": True,
            "must_be_json_serializable": True,
        },
        "completion_requirements": {
            "T1_Door_Wedge_Pass_Through": "Both robots must reach the target-side region after a plausible hold/pass/follow sequence.",
            "T2_Herding_Corralling": "The movable object must end in the goal region after setup plus push/funnel coordination.",
            "T4_Collaborative_Search_Converge": "The target object must be found and both robots must converge to the target region.",
            "T6_Relay_Delivery": "The payload must reach the far goal through a real relay with a handoff phase.",
        },
    }


def prune_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: prune_nulls(inner) for key, inner in value.items() if inner is not None}
    if isinstance(value, list):
        return [prune_nulls(item) for item in value]
    return value


def parse_candidate_plan_text(text: str) -> Dict[str, Any]:
    payload = text.strip()
    if payload.startswith("```"):
        lines = payload.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        payload = "\n".join(lines).strip()

    try:
        return prune_nulls(json.loads(payload))
    except json.JSONDecodeError:
        start = payload.find("{")
        end = payload.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return prune_nulls(json.loads(payload[start : end + 1]))
            except json.JSONDecodeError:
                pass
        if start == -1:
            raise
        repaired = _balance_json_payload(payload[start:])
        return prune_nulls(json.loads(repaired))


def _balance_json_payload(payload: str) -> str:
    stack: list[str] = []
    result: list[str] = []
    in_string = False
    escape = False
    opener_to_closer = {"{": "}", "[": "]"}
    closers = set(opener_to_closer.values())

    for char in payload:
        result.append(char)
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char in opener_to_closer:
            stack.append(opener_to_closer[char])
            continue
        if char in closers:
            if stack and char == stack[-1]:
                stack.pop()
            else:
                # Leave unexpected closers in place; the follow-up json.loads
                # will reject truly malformed output.
                continue

    balanced = "".join(result).rstrip()
    while True:
        updated = balanced.replace(",}", "}").replace(",]", "]")
        if updated == balanced:
            break
        balanced = updated
    return balanced + "".join(reversed(stack))
