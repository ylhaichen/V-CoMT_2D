from vcomt2d.fsm.schema import FSMPlan, StateSpec
from vcomt2d.planner.backends import PlannerBackend, PlannerBackendError, StubLLMPlannerBackend
from vcomt2d.planner.fsm_builder import action, condition, terminal, transition
from vcomt2d.planner.main import DeterministicPlanner, planner_from_mode
from vcomt2d.planner.models import BackendPlanCandidate, PlanningContext, PlanningRequest
from vcomt2d.sim.tasks.fixtures import relay_task_world


class FailingBackend(PlannerBackend):
    backend_name = "failing_backend"

    def build_candidate(self, context: PlanningContext) -> BackendPlanCandidate:
        raise PlannerBackendError("backend_runtime_failure", debug_info={"model_name": "fake-model"})


class InvalidPlanBackend(PlannerBackend):
    backend_name = "invalid_plan_backend"

    def build_candidate(self, context: PlanningContext) -> BackendPlanCandidate:
        plan = FSMPlan(
            task_description="invalid relay plan",
            reasoning_summary="broken",
            initial_state="S0",
            states=[],
        )
        return BackendPlanCandidate(backend_name=self.backend_name, model_name="fake-model", plan=plan, debug_info={"model_name": "fake-model"})


def test_llm_stub_mode_uses_deterministic_fallback():
    planner = planner_from_mode("llm_stub")
    result = planner.plan(
        PlanningRequest(
            request_id="relay_stub",
            user_instruction="Move the box to the far goal",
            world_state=relay_task_world(),
        )
    )
    assert result.success is True
    assert result.debug_info["backend"] == "llm_stub"
    assert result.debug_info["backend_debug"]["generation"]["fallback_backend"] == "deterministic_template"


def test_unconfigured_backend_returns_structured_failure():
    planner = DeterministicPlanner(backend=StubLLMPlannerBackend())
    result = planner.plan(
        PlanningRequest(
            request_id="relay_stub_fail",
            user_instruction="Move the box to the far goal",
            world_state=relay_task_world(),
        )
    )
    assert result.success is False
    assert result.failure_reason == "llm_backend_not_configured"
    assert result.plan is None


def test_backend_failure_resynthesizes_with_deterministic_template():
    planner = DeterministicPlanner(backend=FailingBackend())
    result = planner.plan(
        PlanningRequest(
            request_id="relay_backend_fail",
            user_instruction="Move the box to the far goal",
            world_state=relay_task_world(),
        )
    )
    assert result.success is True
    assert "backend::resynthesize_task_template" in result.debug_info["applied_repairs"]
    assert result.debug_info["backend"] == "failing_backend"
    assert result.debug_info["backend_debug"]["final_candidate_backend"] == "deterministic_template"


def test_invalid_plan_resynthesizes_after_structural_validation_failure():
    planner = DeterministicPlanner(backend=InvalidPlanBackend())
    result = planner.plan(
        PlanningRequest(
            request_id="relay_invalid_plan",
            user_instruction="Move the box to the far goal",
            world_state=relay_task_world(),
        )
    )
    assert result.success is True
    assert "structural::resynthesize_task_template" in result.debug_info["applied_repairs"]
    assert result.debug_info["backend"] == "invalid_plan_backend"
    assert result.debug_info["backend_debug"]["final_candidate_backend"] == "deterministic_template"
