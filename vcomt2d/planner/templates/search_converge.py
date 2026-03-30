"""Collaborative search and converge template."""

from __future__ import annotations

from vcomt2d.core.skills import SkillName
from vcomt2d.planner.fsm_builder import action, condition, plan, state, terminal, transition


def build_search_plan(instruction: str, reasoning_summary: str, scene_facts, roles, config):
    finder = roles.roles["finder"]
    converger = roles.roles["converger"]
    assignments = roles.roles["search_assignments"]
    robot_actions = {
        finder: {
            "S0_SPLIT_SEARCH": action(SkillName.MOVE_TO.value, target_id=assignments[finder]),
            "S1_TRACK_TARGET": action(SkillName.TRACK_OBJECT.value, object_id=scene_facts.target_object_id),
            "S2_SIGNAL_FOUND": action(SkillName.SIGNAL.value, message="target_found"),
            "S3_CONVERGE": action(SkillName.MOVE_TO.value, target_id=scene_facts.goal_region_id),
        },
        converger: {
            "S0_SPLIT_SEARCH": action(SkillName.MOVE_TO.value, target_id=assignments[converger]),
            "S1_TRACK_TARGET": action(SkillName.WAIT_UNTIL.value, condition={"kind": "object_found", "args": {"object_id": scene_facts.target_object_id}}),
            "S2_SIGNAL_FOUND": action(SkillName.WAIT_UNTIL.value, condition={"kind": "signal_received", "args": {"message": "target_found", "from_robot": finder}}),
            "S3_CONVERGE": action(SkillName.MOVE_TO.value, target_id=scene_facts.goal_region_id),
        },
    }
    states = [
        state(
            "S0_SPLIT_SEARCH",
            robot_actions["robot_a"]["S0_SPLIT_SEARCH"],
            robot_actions["robot_b"]["S0_SPLIT_SEARCH"],
            [
                transition("S1_TRACK_TARGET", condition("all_actions_done"), "search sectors assigned"),
                transition("S_FAIL", condition("timeout", ticks=config.default_timeout_ticks["search"]), "failed to reach search sectors"),
            ],
        ),
        state(
            "S1_TRACK_TARGET",
            robot_actions["robot_a"]["S1_TRACK_TARGET"],
            robot_actions["robot_b"]["S1_TRACK_TARGET"],
            [
                transition("S2_SIGNAL_FOUND", condition("object_found", object_id=scene_facts.target_object_id), "finder located target"),
                transition("S_FAIL", condition("timeout", ticks=config.default_timeout_ticks["search"]), "search timeout"),
            ],
        ),
        state(
            "S2_SIGNAL_FOUND",
            robot_actions["robot_a"]["S2_SIGNAL_FOUND"],
            robot_actions["robot_b"]["S2_SIGNAL_FOUND"],
            [
                transition("S3_CONVERGE", condition("all_actions_done"), "signal received by partner"),
                transition("S_FAIL", condition("timeout", ticks=config.default_timeout_ticks["hold"]), "signal exchange timeout"),
            ],
        ),
        state(
            "S3_CONVERGE",
            robot_actions["robot_a"]["S3_CONVERGE"],
            robot_actions["robot_b"]["S3_CONVERGE"],
            [
                transition("S_DONE", condition("both_in_region", region_id=scene_facts.goal_region_id), "both robots converged to target"),
                transition("S_FAIL", condition("timeout", ticks=config.default_timeout_ticks["converge"]), "converge timeout"),
            ],
        ),
        terminal("S_DONE", "success"),
        terminal("S_FAIL", "failure"),
    ]
    return plan(task_description=instruction, reasoning_summary=reasoning_summary, initial_state="S0_SPLIT_SEARCH", states=states)
