from vcomt2d.fsm.schema import FSMPlan, StateSpec
from vcomt2d.fsm.validator import validate_fsm
from vcomt2d.planner.config import PlanningConfig
from vcomt2d.planner.fsm_builder import action, condition, state, transition
from vcomt2d.planner.repair import PlanRepairer


def test_repair_fixes_common_structural_errors():
    malformed = FSMPlan(
        task_description="broken",
        reasoning_summary="broken",
        initial_state="missing",
        states=[
            StateSpec(
                state_id="S0",
                robot_a=action("move", target_id="door_main"),
                robot_b=None,
                transitions=[transition("ghost", condition("all_actions_done"))],
            )
        ],
    )
    initial_validation = validate_fsm(malformed)
    assert initial_validation.valid is False

    repairer = PlanRepairer(PlanningConfig())
    repaired = repairer.repair(malformed, initial_validation).plan
    repaired_validation = validate_fsm(repaired)
    assert repaired_validation.valid is True

