from vcomt2d.planner.models import PlanningRequest
from vcomt2d.planner.main import DeterministicPlanner
from vcomt2d.sim.tasks.fixtures import herding_task_world


def test_planner_returns_valid_herding_plan():
    planner = DeterministicPlanner()
    result = planner.plan(PlanningRequest(request_id="herding", user_instruction="Push the ball into the corner", world_state=herding_task_world()))
    assert result.success is True
    assert result.validation["valid"] is True
    assert result.semantic_validation["passed"] is True
    assert result.plan.initial_state == "S0_FLANK_SETUP"
    assert any(state.status == "success" for state in result.plan.states if state.terminal)
