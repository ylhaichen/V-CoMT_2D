"""Door wedge and pass-through template."""

from __future__ import annotations

from vcomt2d.core.skills import SkillName
from vcomt2d.planner.fsm_builder import action, condition, plan, state, terminal, transition


def build_door_plan(instruction: str, reasoning_summary: str, scene_facts, roles, config):
    holder = roles.roles["holder"]
    passer = roles.roles["passer"]
    robot_actions = {
        holder: {
            "S0_APPROACH": action(SkillName.MOVE_TO.value, target_id=scene_facts.relevant_door_id),
            "S1_HOLD_DOOR": action(SkillName.HOLD_POSITION.value, ticks=2),
            "S2_PASS_PARTNER": action(SkillName.HOLD_POSITION.value, ticks=2),
            "S3_FOLLOW": action(SkillName.MOVE_TO.value, target_id=scene_facts.goal_region_id),
        },
        passer: {
            "S0_APPROACH": action(SkillName.MOVE_TO.value, target_id=scene_facts.wait_region_id or scene_facts.relevant_door_id),
            "S1_HOLD_DOOR": action(SkillName.WAIT_UNTIL.value, condition={"kind": "flag_true", "args": {"flag": "door_held"}}),
            "S2_PASS_PARTNER": action(SkillName.MOVE_TO.value, target_id=scene_facts.goal_region_id),
            "S3_FOLLOW": action(SkillName.HOLD_POSITION.value, ticks=1),
        },
    }
    states = [
        state(
            "S0_APPROACH",
            robot_actions["robot_a"]["S0_APPROACH"],
            robot_actions["robot_b"]["S0_APPROACH"],
            [
                transition("S1_HOLD_DOOR", condition("all_actions_done"), "robots reached door staging"),
                transition("S_FAIL", condition("timeout", ticks=config.default_timeout_ticks["approach"]), "approach timeout"),
            ],
        ),
        state(
            "S1_HOLD_DOOR",
            robot_actions["robot_a"]["S1_HOLD_DOOR"],
            robot_actions["robot_b"]["S1_HOLD_DOOR"],
            [
                transition("S2_PASS_PARTNER", condition("flag_true", flag="door_held"), "door is wedged and passable"),
                transition("S_FAIL", condition("timeout", ticks=config.default_timeout_ticks["hold"]), "failed to hold door"),
            ],
            notes=f"holder={holder}",
        ),
        state(
            "S2_PASS_PARTNER",
            robot_actions["robot_a"]["S2_PASS_PARTNER"],
            robot_actions["robot_b"]["S2_PASS_PARTNER"],
            [
                transition("S3_FOLLOW", condition("robot_in_region", robot=passer, region_id=scene_facts.goal_region_id), "partner cleared doorway"),
                transition("S_FAIL", condition("timeout", ticks=config.default_timeout_ticks["hold"]), "pass-through timeout"),
            ],
        ),
        state(
            "S3_FOLLOW",
            robot_actions["robot_a"]["S3_FOLLOW"],
            robot_actions["robot_b"]["S3_FOLLOW"],
            [
                transition("S_DONE", condition("both_in_region", region_id=scene_facts.goal_region_id), "both robots reached target room"),
                transition("S_FAIL", condition("timeout", ticks=config.default_timeout_ticks["converge"]), "follow timeout"),
            ],
        ),
        terminal("S_DONE", "success"),
        terminal("S_FAIL", "failure"),
    ]
    return plan(task_description=instruction, reasoning_summary=reasoning_summary, initial_state="S0_APPROACH", states=states)

