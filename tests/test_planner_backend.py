from vcomt2d.planner.backends import StubLLMPlannerBackend
from vcomt2d.planner.main import DeterministicPlanner, planner_from_mode
from vcomt2d.planner.models import PlanningRequest
from vcomt2d.sim.tasks.fixtures import relay_task_world


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
    assert result.debug_info["backend_debug"]["fallback_backend"] == "deterministic_template"


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
