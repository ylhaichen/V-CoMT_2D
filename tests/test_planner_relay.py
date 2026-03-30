from vcomt2d.planner.models import PlanningRequest
from vcomt2d.planner.main import DeterministicPlanner
from vcomt2d.sim.tasks.fixtures import relay_task_world


def test_planner_returns_valid_relay_plan():
    planner = DeterministicPlanner()
    result = planner.plan(PlanningRequest(request_id="relay", user_instruction="Move the box to the far goal", world_state=relay_task_world()))
    assert result.success is True
    assert result.validation["valid"] is True
    assert result.semantic_validation["passed"] is True
    assert result.plan.initial_state == "S0_INIT"
    assert any(state.status == "success" for state in result.plan.states if state.terminal)

