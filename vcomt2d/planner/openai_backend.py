"""Real OpenAI GPT planner backend using the Responses API."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from vcomt2d.core.skills import VALID_SKILLS
from vcomt2d.fsm.schema import FSMPlan

from .backends import PlannerBackend, PlannerBackendError
from .models import BackendPlanCandidate, PlanningContext
from .openai_config import OpenAIPlannerConfig


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
                            "robot_a": {
                                "anyOf": [
                                    {"type": "null"},
                                    {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": ["skill", "params"],
                                        "properties": {
                                            "skill": {"type": "string"},
                                            "params": {"type": "object", "additionalProperties": True},
                                        },
                                    },
                                ]
                            },
                            "robot_b": {
                                "anyOf": [
                                    {"type": "null"},
                                    {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": ["skill", "params"],
                                        "properties": {
                                            "skill": {"type": "string"},
                                            "params": {"type": "object", "additionalProperties": True},
                                        },
                                    },
                                ]
                            },
                            "transitions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["to", "condition", "notes"],
                                    "properties": {
                                        "to": {"type": "string"},
                                        "condition": {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "required": ["kind", "args"],
                                            "properties": {
                                                "kind": {"type": "string"},
                                                "args": {"type": "object", "additionalProperties": True},
                                            },
                                        },
                                        "notes": {"type": "string"},
                                    },
                                },
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


class OpenAIGPTPlannerBackend(PlannerBackend):
    """Planner backend backed by OpenAI Responses API + Structured Outputs."""

    backend_name = "gpt"

    def __init__(self, config: Optional[OpenAIPlannerConfig] = None, client: Any = None):
        self.config = config
        self._client = client

    def build_candidate(self, context: PlanningContext) -> BackendPlanCandidate:
        runtime_config = self.config or OpenAIPlannerConfig.from_planning_config(context.request.planning_config)
        client = self._get_client(runtime_config)
        system_prompt = self._build_system_prompt()
        user_payload = self._build_user_payload(context)
        request_payload = self._build_request_payload(runtime_config, system_prompt, user_payload)
        response = client.responses.create(**request_payload)
        raw_response = self._response_to_dict(response)
        output_text = self._extract_output_text(response, raw_response)
        if not output_text:
            raise PlannerBackendError("openai_response_missing_text")

        try:
            candidate_plan = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise PlannerBackendError(f"openai_response_not_json:{exc}") from exc

        try:
            plan = FSMPlan.from_dict(candidate_plan)
        except Exception as exc:  # noqa: BLE001 - surface parser failure through planner backend error
            raise PlannerBackendError(f"openai_candidate_parse_failed:{exc}") from exc

        return BackendPlanCandidate(
            backend_name=self.backend_name,
            model_name=runtime_config.model,
            plan=plan,
            debug_info={
                "api": "responses",
                "model_name": runtime_config.model,
                "reasoning_effort": runtime_config.reasoning_effort,
                "system_prompt": system_prompt,
                "request_payload": request_payload,
                "prompt_payload": user_payload,
                "raw_response": raw_response,
                "candidate_plan": candidate_plan,
            },
        )

    def _get_client(self, runtime_config: OpenAIPlannerConfig):
        if self._client is not None:
            return self._client
        if not runtime_config.api_key:
            raise PlannerBackendError("openai_api_key_missing")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised by mocked tests instead
            raise PlannerBackendError("openai_package_not_installed") from exc
        self._client = OpenAI(
            api_key=runtime_config.api_key,
            timeout=runtime_config.timeout_seconds,
            max_retries=runtime_config.retry_count,
        )
        return self._client

    def _build_system_prompt(self) -> str:
        return (
            "You are the GPT planner backend for V-CoMT_2D. "
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

    def _build_user_payload(self, context: PlanningContext) -> Dict[str, Any]:
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

    def _build_request_payload(self, runtime_config: OpenAIPlannerConfig, system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "model": runtime_config.model,
            "reasoning": {"effort": runtime_config.reasoning_effort},
            "max_output_tokens": runtime_config.max_output_tokens,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": json.dumps(user_payload, indent=2)}],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "vcomt2d_fsm_plan",
                    "strict": True,
                    "schema": PLANNER_RESPONSE_SCHEMA,
                }
            },
        }

    def _response_to_dict(self, response: Any) -> Dict[str, Any]:
        if hasattr(response, "model_dump"):
            try:
                return response.model_dump(mode="json")
            except TypeError:
                return response.model_dump()
        if hasattr(response, "to_dict"):
            return response.to_dict()
        if isinstance(response, dict):
            return response
        return {"repr": repr(response), "output_text": getattr(response, "output_text", None)}

    def _extract_output_text(self, response: Any, raw_response: Dict[str, Any]) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text
        for output in raw_response.get("output", []):
            for content in output.get("content", []):
                if content.get("type") in {"output_text", "text"} and content.get("text"):
                    return content["text"]
        return ""
