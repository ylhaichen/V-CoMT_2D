"""Top-level planner entrypoints."""

from __future__ import annotations

from vcomt2d.core.status import PlannerMode

from .backends import DeterministicTemplateBackend, PlannerBackend, StubLLMPlannerBackend
from .base import Planner
from .models import PlanningRequest, PlanningResult
from .pipeline import PlannerPipeline


class DeterministicPlanner(Planner):
    """Stable planner contract backed by the planner pipeline."""

    def __init__(self, backend: PlannerBackend | None = None):
        self.backend = backend or DeterministicTemplateBackend()
        self.pipeline = PlannerPipeline(generation_backend=self.backend)

    def plan(self, request: PlanningRequest) -> PlanningResult:
        return self.pipeline.run(request)


def planner_from_mode(planner_mode: str) -> Planner:
    """Factory hook for future backend selection without changing callers."""

    if planner_mode == PlannerMode.DETERMINISTIC.value:
        return DeterministicPlanner()
    if planner_mode in {"llm_stub", "vlm_stub"}:
        return DeterministicPlanner(backend=StubLLMPlannerBackend(fallback_backend=DeterministicTemplateBackend()))
    return DeterministicPlanner()
