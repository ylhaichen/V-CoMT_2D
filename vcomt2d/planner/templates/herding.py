"""Herding / corralling template."""

from __future__ import annotations

from vcomt2d.core.skills import SkillName
from vcomt2d.planner.fsm_builder import action, condition, plan, state, terminal, transition


def build_herding_plan(instruction: str, reasoning_summary: str, scene_facts, roles, config):
    pusher = roles.roles["pusher"]
    blocker = roles.roles["blocker"]
    staging = scene_facts.staging_region_ids
    flank_left = staging[0] if len(staging) >= 1 else scene_facts.goal_region_id
    flank_right = staging[1] if len(staging) >= 2 else flank_left
    funnel_point = staging[2] if len(staging) >= 3 else scene_facts.goal_region_id
    robot_actions = {
        pusher: {
            "S0_FLANK_SETUP": action(SkillName.MOVE_TO.value, target_id=flank_left),
            "S1_SYNCHRONIZE": action(SkillName.SYNCHRONIZE.value, sync_key="herd_ready"),
            "S2_PUSH_AND_FUNNEL": action(SkillName.PUSH.value, object_id=scene_facts.target_object_id, target_id=scene_facts.goal_region_id),
        },
        blocker: {
            "S0_FLANK_SETUP": action(SkillName.MOVE_TO.value, target_id=flank_right),
            "S1_SYNCHRONIZE": action(SkillName.SYNCHRONIZE.value, sync_key="herd_ready"),
            "S2_PUSH_AND_FUNNEL": action(SkillName.MOVE_TO.value, target_id=funnel_point),
        },
    }
    states = [
        state(
            "S0_FLANK_SETUP",
            robot_actions["robot_a"]["S0_FLANK_SETUP"],
            robot_actions["robot_b"]["S0_FLANK_SETUP"],
            [
                transition("S1_SYNCHRONIZE", condition("all_actions_done"), "robots reached flanking points"),
                transition("S_FAIL", condition("timeout", ticks=config.default_timeout_ticks["approach"]), "flank setup timeout"),
            ],
        ),
        state(
            "S1_SYNCHRONIZE",
            robot_actions["robot_a"]["S1_SYNCHRONIZE"],
            robot_actions["robot_b"]["S1_SYNCHRONIZE"],
            [
                transition("S2_PUSH_AND_FUNNEL", condition("all_actions_done"), "robots synchronized"),
                transition("S_FAIL", condition("timeout", ticks=config.default_timeout_ticks["hold"]), "sync timeout"),
            ],
        ),
        state(
            "S2_PUSH_AND_FUNNEL",
            robot_actions["robot_a"]["S2_PUSH_AND_FUNNEL"],
            robot_actions["robot_b"]["S2_PUSH_AND_FUNNEL"],
            [
                transition("S_DONE", condition("object_in_region", object_id=scene_facts.target_object_id, region_id=scene_facts.goal_region_id), "ball corralled into target"),
                transition("S_FAIL", condition("timeout", ticks=config.default_timeout_ticks["push"]), "push timeout"),
            ],
        ),
        terminal("S_DONE", "success"),
        terminal("S_FAIL", "failure"),
    ]
    return plan(task_description=instruction, reasoning_summary=reasoning_summary, initial_state="S0_FLANK_SETUP", states=states)
