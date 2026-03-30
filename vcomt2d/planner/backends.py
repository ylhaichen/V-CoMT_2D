"""Planner backends for deterministic and future LLM/VLM integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict

from .models import BackendPlanCandidate, PlanningContext, TaskType
from .templates.door_wedge import build_door_plan
from .templates.herding import build_herding_plan
from .templates.relay_delivery import build_relay_plan
from .templates.search_converge import build_search_plan


TEMPLATE_BUILDERS: Dict[TaskType, Callable] = {
    TaskType.T1_DOOR_WEDGE_PASS_THROUGH: build_door_plan,
    TaskType.T2_HERDING_CORRALLING: build_herding_plan,
    TaskType.T4_COLLABORATIVE_SEARCH_CONVERGE: build_search_plan,
    TaskType.T6_RELAY_DELIVERY: build_relay_plan,
}


class PlannerBackendError(RuntimeError):
    """Raised when a planner backend cannot produce a candidate plan."""

    def __init__(self, message: str, *, debug_info: Dict[str, object] | None = None):
        super().__init__(message)
        self.debug_info = debug_info or {}


class PlannerBackend(ABC):
    """Backend interface for candidate plan generation."""

    backend_name = "unknown"

    @abstractmethod
    def build_candidate(self, context: PlanningContext) -> BackendPlanCandidate:
        raise NotImplementedError


class DeterministicTemplateBackend(PlannerBackend):
    """Reference symbolic backend used as the current planning baseline."""

    backend_name = "deterministic_template"

    def build_candidate(self, context: PlanningContext) -> BackendPlanCandidate:
        builder = TEMPLATE_BUILDERS.get(context.intent.task_type)
        if builder is None:
            raise PlannerBackendError(f"unsupported_task_type:{context.intent.task_type}")
        plan = builder(
            context.request.user_instruction,
            context.reasoning_summary,
            context.scene_facts,
            context.roles,
            context.request.planning_config,
        )
        return BackendPlanCandidate(
            backend_name=self.backend_name,
            plan=plan,
            debug_info={"task_type": context.intent.task_type.value},
        )


class StubLLMPlannerBackend(PlannerBackend):
    """Stub insertion point for future LLM/VLM-generated candidate plans."""

    backend_name = "llm_stub"

    def __init__(self, fallback_backend: PlannerBackend | None = None):
        self.fallback_backend = fallback_backend

    def build_candidate(self, context: PlanningContext) -> BackendPlanCandidate:
        if self.fallback_backend is not None:
            candidate = self.fallback_backend.build_candidate(context)
            candidate.backend_name = self.backend_name
            candidate.debug_info["fallback_backend"] = self.fallback_backend.backend_name
            candidate.debug_info["stub_note"] = "LLM/VLM backend not configured; deterministic fallback was used."
            return candidate
        raise PlannerBackendError("llm_backend_not_configured")
