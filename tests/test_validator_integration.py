from vcomt2d.fsm.schema import FSMPlan, StateSpec
from vcomt2d.fsm.validator import validate_fsm
from vcomt2d.planner.fsm_builder import action, condition, terminal, transition


def _valid_plan():
    return FSMPlan(
        task_description="demo",
        reasoning_summary="demo",
        initial_state="S0",
        states=[
            StateSpec(
                state_id="S0",
                robot_a=action("MOVE_TO", target_id="door_main"),
                robot_b=action("MOVE_TO", target_id="door_main"),
                transitions=[
                    transition("S_DONE", condition("all_actions_done")),
                    transition("S_FAIL", condition("timeout", ticks=5)),
                ],
            ),
            terminal("S_DONE", "success"),
            terminal("S_FAIL", "failure"),
        ],
    )


def test_valid_fsm_passes():
    assert validate_fsm(_valid_plan()).valid is True


def test_missing_timeout_fails():
    plan = _valid_plan()
    plan.states[0].transitions = [transition("S_DONE", condition("all_actions_done"))]
    validation = validate_fsm(plan)
    assert validation.valid is False
    assert any(error.code == "missing_timeout_transition" for error in validation.errors)


def test_missing_robot_action_fails():
    plan = _valid_plan()
    plan.states[0].robot_b = None
    validation = validate_fsm(plan)
    assert validation.valid is False
    assert any(error.code == "missing_robot_action" for error in validation.errors)


def test_unreachable_state_fails():
    plan = _valid_plan()
    plan.states.append(terminal("S_ORPHAN", "success"))
    validation = validate_fsm(plan)
    assert validation.valid is False
    assert any(error.code == "orphan_state" for error in validation.errors)


def test_missing_terminal_fails():
    plan = _valid_plan()
    plan.states = plan.states[:-1]
    validation = validate_fsm(plan)
    assert validation.valid is False
    assert any(error.code == "missing_failure_terminal" for error in validation.errors)

