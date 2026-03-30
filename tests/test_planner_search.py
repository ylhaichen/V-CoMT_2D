from vcomt2d.planner.models import PlanningRequest
from vcomt2d.planner.main import DeterministicPlanner
from vcomt2d.sim.tasks.fixtures import search_task_world


def test_planner_returns_valid_search_plan():
    planner = DeterministicPlanner()
    result = planner.plan(PlanningRequest(request_id="search", user_instruction="Find the red box and meet there", world_state=search_task_world()))
    assert result.success is True
    assert result.validation["valid"] is True
    assert result.semantic_validation["passed"] is True
    assert result.plan.initial_state == "S0_SPLIT_SEARCH"
    assert any(state.status == "failure" for state in result.plan.states if state.terminal)
