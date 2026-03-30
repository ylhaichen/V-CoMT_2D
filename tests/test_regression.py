from vcomt2d.planner.main import DeterministicPlanner
from vcomt2d.planner.models import PlanningRequest
from vcomt2d.sim.tasks.fixtures import herding_task_world


def test_unsupported_instruction_returns_structured_failure():
    planner = DeterministicPlanner()
    result = planner.plan(PlanningRequest(request_id="bad", user_instruction="Dance around the map", world_state=herding_task_world()))
    assert result.success is False
    assert result.failure_reason == "unsupported_instruction"
    assert result.plan is None


def test_missing_object_returns_structured_failure():
    planner = DeterministicPlanner()
    world = herding_task_world()
    world.objects = []
    result = planner.plan(PlanningRequest(request_id="missing_ball", user_instruction="Push the ball into the corner", world_state=world))
    assert result.success is False
    assert result.failure_reason == "target_ball_not_found"
