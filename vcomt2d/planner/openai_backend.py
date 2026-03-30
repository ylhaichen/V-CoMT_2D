"""Real OpenAI GPT planner backend using the Responses API."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from vcomt2d.core.types import to_serializable
from vcomt2d.fsm.schema import FSMPlan

from .backends import PlannerBackend, PlannerBackendError
from .models import BackendPlanCandidate, PlanningContext
from .openai_config import OpenAIPlannerConfig
from .structured_output import PLANNER_RESPONSE_SCHEMA, build_planner_system_prompt, build_planner_user_payload, parse_candidate_plan_text


class OpenAIGPTPlannerBackend(PlannerBackend):
    """Planner backend backed by OpenAI Responses API + Structured Outputs."""

    backend_name = "gpt"

    def __init__(self, config: Optional[OpenAIPlannerConfig] = None, client: Any = None):
        self.config = config
        self._client = client

    def build_candidate(self, context: PlanningContext) -> BackendPlanCandidate:
        runtime_config = self.config or OpenAIPlannerConfig.from_planning_config(context.request.planning_config)
        client = self._get_client(runtime_config)
        system_prompt = build_planner_system_prompt()
        user_payload = build_planner_user_payload(context)
        request_payload = self._build_request_payload(runtime_config, system_prompt, user_payload)
        backend_debug = {
            "api": "responses",
            "model_name": runtime_config.model,
            "reasoning_effort": runtime_config.reasoning_effort,
            "system_prompt": system_prompt,
            "request_payload": request_payload,
            "prompt_payload": user_payload,
        }

        try:
            response = client.responses.create(**request_payload)
        except Exception as exc:  # noqa: BLE001 - backend must translate SDK failures into planner failures
            error_payload = self._exception_to_debug(exc)
            backend_debug["api_error"] = error_payload
            raise PlannerBackendError(self._classify_api_error(exc, error_payload), debug_info=backend_debug) from exc

        raw_response = self._response_to_dict(response)
        output_text = self._extract_output_text(response, raw_response)
        if not output_text:
            backend_debug["raw_response"] = raw_response
            raise PlannerBackendError("openai_response_missing_text", debug_info=backend_debug)

        try:
            candidate_plan = parse_candidate_plan_text(output_text)
        except json.JSONDecodeError as exc:
            backend_debug["raw_response"] = raw_response
            backend_debug["response_text"] = output_text
            raise PlannerBackendError(f"openai_response_not_json:{exc}", debug_info=backend_debug) from exc

        try:
            plan = FSMPlan.from_dict(candidate_plan)
        except Exception as exc:  # noqa: BLE001 - surface parser failure through planner backend error
            backend_debug["raw_response"] = raw_response
            backend_debug["candidate_plan"] = candidate_plan
            raise PlannerBackendError(f"openai_candidate_parse_failed:{exc}", debug_info=backend_debug) from exc

        return BackendPlanCandidate(
            backend_name=self.backend_name,
            model_name=runtime_config.model,
            plan=plan,
            debug_info={**backend_debug, "raw_response": raw_response, "candidate_plan": candidate_plan},
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

    def _classify_api_error(self, exc: Exception, error_payload: Dict[str, Any]) -> str:
        exc_type = exc.__class__.__name__
        code = str(error_payload.get("code") or "").lower()
        message = str(error_payload.get("message") or exc).lower()

        if exc_type == "RateLimitError" and ("insufficient_quota" in code or "insufficient_quota" in message):
            return "openai_insufficient_quota"
        if exc_type == "RateLimitError":
            return "openai_rate_limited"
        if exc_type == "AuthenticationError":
            return "openai_authentication_failed"
        if exc_type == "PermissionDeniedError":
            return "openai_permission_denied"
        if exc_type == "BadRequestError":
            return "openai_bad_request"
        if exc_type in {"APITimeoutError", "TimeoutError"}:
            return "openai_request_timed_out"
        if exc_type in {"APIConnectionError", "APIStatusError"}:
            return "openai_request_failed"
        if "insufficient_quota" in code or "insufficient_quota" in message:
            return "openai_insufficient_quota"
        return "openai_request_failed"

    def _exception_to_debug(self, exc: Exception) -> Dict[str, Any]:
        body = getattr(exc, "body", None)
        message = str(exc)
        code = None
        error_type = None

        if isinstance(body, dict):
            error_payload = body.get("error", body)
            if isinstance(error_payload, dict):
                code = error_payload.get("code")
                error_type = error_payload.get("type")
                message = str(error_payload.get("message", message))

        return {
            "exception_type": exc.__class__.__name__,
            "message": message,
            "status_code": getattr(exc, "status_code", None),
            "code": code,
            "error_type": error_type,
            "body": to_serializable(body),
        }
