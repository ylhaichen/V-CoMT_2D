from copy import deepcopy

from vcomt2d.fsm.validator import validate_fsm
from vcomt2d.planner.main import DeterministicPlanner
from vcomt2d.planner.models import PlanningRequest, TaskType
from vcomt2d.planner.role_assignment import assign_roles
from vcomt2d.planner.scene_interpreter import interpret_scene
from vcomt2d.planner.semantic_checks import SemanticRepairer, SemanticSanityChecker
from vcomt2d.planner.templates.door_wedge import build_door_plan
from vcomt2d.planner.templates.relay_delivery import build_relay_plan
from vcomt2d.planner.templates.search_converge import build_search_plan
from vcomt2d.sim.tasks.fixtures import door_task_world, herding_task_world, relay_task_world, search_task_world


def _build_valid_plan(task_type, instruction, world):
    planner = DeterministicPlanner()
    result = planner.plan(PlanningRequest(request_id="check", user_instruction=instruction, world_state=world))
    assert result.success is True
    return result.plan


def test_good_plans_pass_semantic_checks():
    checker = SemanticSanityChecker()
    cases = [
        (TaskType.T1_DOOR_WEDGE_PASS_THROUGH, "Both of you get into the next room", door_task_world()),
        (TaskType.T2_HERDING_CORRALLING, "Push the ball into the corner", herding_task_world()),
        (TaskType.T4_COLLABORATIVE_SEARCH_CONVERGE, "Find the red box and meet there", search_task_world()),
        (TaskType.T6_RELAY_DELIVERY, "Move the box to the far goal", relay_task_world()),
    ]
    for task_type, instruction, world in cases:
        scene_facts = interpret_scene(task_type, world)
        roles = assign_roles(scene_facts, world)
        plan = _build_valid_plan(task_type, instruction, world)
        semantic = checker.check(task_type, plan, world, scene_facts, roles)
        assert semantic.passed is True


def test_semantically_wrong_door_plan_fails():
    checker = SemanticSanityChecker()
    world = door_task_world()
    task_type = TaskType.T1_DOOR_WEDGE_PASS_THROUGH
    scene_facts = interpret_scene(task_type, world)
    roles = assign_roles(scene_facts, world)
    plan = _build_valid_plan(task_type, "Both of you get into the next room", world)

    state_map = plan.state_map()
    hold_state = state_map["S1_HOLD_DOOR"]
    pass_state = state_map["S2_PASS_PARTNER"]
    hold_state.robot_a, hold_state.robot_b = hold_state.robot_b, hold_state.robot_a
    pass_state.robot_a, pass_state.robot_b = pass_state.robot_b, pass_state.robot_a

    assert validate_fsm(plan).valid is True
    semantic = checker.check(task_type, plan, world, scene_facts, roles)
    codes = {issue.code for issue in semantic.errors}
    assert semantic.passed is False
    assert "door_wrong_holder" in codes or "door_pass_order_inconsistent" in codes


def test_repairable_search_semantic_issue_gets_corrected():
    checker = SemanticSanityChecker()
    repairer = SemanticRepairer()
    world = search_task_world()
    task_type = TaskType.T4_COLLABORATIVE_SEARCH_CONVERGE
    scene_facts = interpret_scene(task_type, world)
    roles = assign_roles(scene_facts, world)
    request = PlanningRequest(request_id="search_semantic_fix", user_instruction="Find the red box and meet there", world_state=world)
    plan = _build_valid_plan(task_type, request.user_instruction, world)

    split_state = plan.state_map()["S0_SPLIT_SEARCH"]
    split_state.robot_b.params["target_id"] = split_state.robot_a.params["target_id"]

    semantic = checker.check(task_type, plan, world, scene_facts, roles)
    assert semantic.passed is False
    assert any(issue.code == "search_same_region" for issue in semantic.errors)

    repaired = repairer.repair(request, plan, semantic, scene_facts, roles, build_search_plan)
    repaired_semantic = checker.check(task_type, repaired.plan, world, scene_facts, repaired.roles)
    assert repaired.applied_repairs
    assert validate_fsm(repaired.plan).valid is True
    assert repaired_semantic.passed is True


def test_relay_plan_without_handoff_logic_fails():
    checker = SemanticSanityChecker()
    world = relay_task_world()
    task_type = TaskType.T6_RELAY_DELIVERY
    scene_facts = interpret_scene(task_type, world)
    roles = assign_roles(scene_facts, world)
    plan = _build_valid_plan(task_type, "Move the box to the far goal", world)

    first_push = plan.state_map()["S1_FIRST_PUSH"]
    second_push = plan.state_map()["S3_SECOND_PUSH"]
    first_push.robot_a = deepcopy(second_push.robot_b)
    first_push.robot_b = deepcopy(second_push.robot_a)
    first_push.transitions[0].condition.kind = "all_actions_done"
    first_push.transitions[0].condition.args = {}

    semantic = checker.check(task_type, plan, world, scene_facts, roles)
    codes = {issue.code for issue in semantic.errors}
    assert semantic.passed is False
    assert "relay_coordination_missing" in codes or "relay_handoff_condition_invalid" in codes or "relay_not_a_relay" in codes


def test_herding_plan_with_bad_blocker_side_fails():
    checker = SemanticSanityChecker()
    world = herding_task_world()
    task_type = TaskType.T2_HERDING_CORRALLING
    scene_facts = interpret_scene(task_type, world)
    roles = assign_roles(scene_facts, world)
    plan = _build_valid_plan(task_type, "Push the ball into the corner", world)

    setup_state = plan.state_map()["S0_FLANK_SETUP"]
    setup_state.robot_b.params["target_id"] = setup_state.robot_a.params["target_id"]

    semantic = checker.check(task_type, plan, world, scene_facts, roles)
    codes = {issue.code for issue in semantic.errors}
    assert semantic.passed is False
    assert "herding_setup_not_partitioned" in codes or "herding_blocker_wrong_side" in codes


def test_unrecoverable_semantic_issue_returns_clean_planning_failure():
    planner = DeterministicPlanner()
    world = door_task_world()
    world.doors[0].wedgeable = False

    result = planner.plan(PlanningRequest(request_id="door_unrecoverable", user_instruction="Both of you get into the next room", world_state=world))
    assert result.success is False
    assert result.failure_reason == "semantic_sanity_failed"
    assert result.plan is None
    assert any(error["code"] == "door_not_wedgeable" for error in result.semantic_validation["errors"])
