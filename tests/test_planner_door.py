from vcomt2d.planner.models import PlanningRequest
from vcomt2d.planner.main import DeterministicPlanner
from vcomt2d.sim.tasks.fixtures import door_task_world


def test_planner_returns_valid_door_plan():
    planner = DeterministicPlanner()
    result = planner.plan(PlanningRequest(request_id="door", user_instruction="Both of you get into the next room", world_state=door_task_world()))
    assert result.success is True
    assert result.validation["valid"] is True
    assert result.semantic_validation["passed"] is True
    assert result.plan.initial_state == "S0_APPROACH"
    statuses = {state.status for state in result.plan.states if state.terminal}
    assert {"success", "failure"} <= statuses
    for state in result.plan.states:
        if not state.terminal:
            assert state.robot_a is not None
            assert state.robot_b is not None
