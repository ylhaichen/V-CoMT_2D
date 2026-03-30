"""Relay delivery template."""

from __future__ import annotations

from vcomt2d.core.skills import SkillName
from vcomt2d.planner.fsm_builder import action, condition, plan, state, terminal, transition


def build_relay_plan(instruction: str, reasoning_summary: str, scene_facts, roles, config):
    starter = roles.roles["starter"]
    finisher = roles.roles["finisher"]
    clearance_position = roles.roles.get("clearance_position", [4.0, 8.0])
    robot_actions = {
        starter: {
            "S0_INIT": action(SkillName.MOVE_TO.value, target_id=scene_facts.target_object_id),
            "S1_FIRST_PUSH": action(SkillName.PUSH.value, object_id=scene_facts.target_object_id, target_id=scene_facts.handoff_region_id),
            "S2_HANDOFF_SYNC": action(SkillName.SIGNAL.value, message="handoff_ready"),
            "S3_SECOND_PUSH": action(SkillName.RETREAT.value, target_position=clearance_position),
        },
        finisher: {
            "S0_INIT": action(SkillName.MOVE_TO.value, target_id=scene_facts.handoff_region_id),
            "S1_FIRST_PUSH": action(SkillName.WAIT_UNTIL.value, condition={"kind": "handoff_ready", "args": {"region_id": scene_facts.handoff_region_id, "object_id": scene_facts.target_object_id}}),
            "S2_HANDOFF_SYNC": action(SkillName.WAIT_UNTIL.value, condition={"kind": "signal_received", "args": {"message": "handoff_ready", "from_robot": starter}}),
            "S3_SECOND_PUSH": action(SkillName.PUSH.value, object_id=scene_facts.target_object_id, target_id=scene_facts.goal_region_id),
        },
    }
    states = [
        state(
            "S0_INIT",
            robot_actions["robot_a"]["S0_INIT"],
            robot_actions["robot_b"]["S0_INIT"],
            [
                transition("S1_FIRST_PUSH", condition("all_actions_done"), "starter reached payload and finisher reached handoff"),
                transition("S_FAIL", condition("timeout", ticks=config.default_timeout_ticks["approach"]), "initial positioning timeout"),
            ],
        ),
        state(
            "S1_FIRST_PUSH",
            robot_actions["robot_a"]["S1_FIRST_PUSH"],
            robot_actions["robot_b"]["S1_FIRST_PUSH"],
            [
                transition("S2_HANDOFF_SYNC", condition("handoff_ready", region_id=scene_facts.handoff_region_id, object_id=scene_facts.target_object_id), "payload arrived at handoff"),
                transition("S_FAIL", condition("timeout", ticks=config.default_timeout_ticks["push"]), "first push timeout"),
            ],
        ),
        state(
            "S2_HANDOFF_SYNC",
            robot_actions["robot_a"]["S2_HANDOFF_SYNC"],
            robot_actions["robot_b"]["S2_HANDOFF_SYNC"],
            [
                transition("S3_SECOND_PUSH", condition("all_actions_done"), "handoff acknowledgement complete"),
                transition("S_FAIL", condition("timeout", ticks=config.default_timeout_ticks["handoff"]), "handoff sync timeout"),
            ],
        ),
        state(
            "S3_SECOND_PUSH",
            robot_actions["robot_a"]["S3_SECOND_PUSH"],
            robot_actions["robot_b"]["S3_SECOND_PUSH"],
            [
                transition("S_DONE", condition("object_in_region", object_id=scene_facts.target_object_id, region_id=scene_facts.goal_region_id), "payload delivered"),
                transition("S_FAIL", condition("timeout", ticks=config.default_timeout_ticks["push"]), "second push timeout"),
            ],
        ),
        terminal("S_DONE", "success"),
        terminal("S_FAIL", "failure"),
    ]
    return plan(task_description=instruction, reasoning_summary=reasoning_summary, initial_state="S0_INIT", states=states)
