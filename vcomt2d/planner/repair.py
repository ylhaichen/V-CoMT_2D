"""Bounded deterministic FSM repair logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from vcomt2d.core.skills import SkillName, normalize_skill_name
from vcomt2d.fsm.schema import ActionSpec, FSMPlan, StateSpec, TransitionSpec
from vcomt2d.fsm.validator import ValidationResult
from .config import PlanningConfig
from .fsm_builder import condition, terminal, transition


@dataclass
class RepairResult:
    plan: FSMPlan
    applied_repairs: List[str] = field(default_factory=list)


class PlanRepairer:
    def __init__(self, config: PlanningConfig):
        self.config = config

    def repair(self, plan: FSMPlan, validation: ValidationResult) -> RepairResult:
        state_map = plan.state_map()
        applied: List[str] = []
        error_codes = [error.code for error in validation.errors]

        if "missing_success_terminal" in error_codes and "S_DONE" not in state_map:
            plan.states.append(terminal("S_DONE", "success", notes="Added by repair"))
            applied.append("added_success_terminal")
            state_map = plan.state_map()

        if "missing_failure_terminal" in error_codes and "S_FAIL" not in state_map:
            plan.states.append(terminal("S_FAIL", "failure", notes="Added by repair"))
            applied.append("added_failure_terminal")
            state_map = plan.state_map()

        if "missing_initial_state" in error_codes and plan.states:
            candidate = next((state.state_id for state in plan.states if not state.terminal), plan.states[0].state_id)
            plan.initial_state = candidate
            applied.append("relinked_initial_state")

        for state in list(plan.states):
            if state.terminal:
                continue
            if state.robot_a is None:
                state.robot_a = ActionSpec(skill=SkillName.HOLD_POSITION.value, params={"ticks": 1})
                applied.append(f"filled_robot_a:{state.state_id}")
            if state.robot_b is None:
                state.robot_b = ActionSpec(skill=SkillName.HOLD_POSITION.value, params={"ticks": 1})
                applied.append(f"filled_robot_b:{state.state_id}")
            for robot_key in ("robot_a", "robot_b"):
                action = getattr(state, robot_key)
                normalized = normalize_skill_name(action.skill)
                if normalized != action.skill:
                    action.skill = normalized
                    applied.append(f"normalized_skill:{state.state_id}:{robot_key}")
            if not any(item.condition.kind == "timeout" for item in state.transitions):
                state.transitions.append(transition("S_FAIL", condition("timeout", ticks=self.config.default_timeout_ticks.get("approach", 8)), notes="Added timeout by repair"))
                applied.append(f"added_timeout:{state.state_id}")
            for item in state.transitions:
                if item.condition.kind == "timeout" and "ticks" not in item.condition.args:
                    item.condition.args["ticks"] = self.config.default_timeout_ticks.get("approach", 8)
                    applied.append(f"filled_timeout_ticks:{state.state_id}")

        state_ids = {state.state_id for state in plan.states}
        for state in plan.states:
            if state.terminal:
                continue
            for item in state.transitions:
                if item.to not in state_ids:
                    item.to = "S_FAIL"
                    applied.append(f"relinked_missing_target:{state.state_id}")

        success_referenced = any(item.to == "S_DONE" for state in plan.states if not state.terminal for item in state.transitions)
        if "S_DONE" in {state.state_id for state in plan.states} and not success_referenced:
            first_non_terminal = next((state for state in plan.states if not state.terminal), None)
            if first_non_terminal is not None:
                non_timeout = [item for item in first_non_terminal.transitions if item.condition.kind != "timeout"]
                timeout_items = [item for item in first_non_terminal.transitions if item.condition.kind == "timeout"]
                if non_timeout:
                    non_timeout[0].to = "S_DONE"
                    applied.append(f"relinked_success_path:{first_non_terminal.state_id}")
                else:
                    non_timeout.append(transition("S_DONE", condition("all_actions_done"), notes="Added success path by repair"))
                    applied.append(f"added_success_path:{first_non_terminal.state_id}")
                first_non_terminal.transitions = non_timeout + timeout_items

        if "orphan_state" in error_codes:
            referenced = {plan.initial_state}
            frontier = [plan.initial_state]
            while frontier:
                current = frontier.pop()
                current_state = plan.state_map().get(current)
                if current_state is None:
                    continue
                for item in current_state.transitions:
                    if item.to not in referenced:
                        referenced.add(item.to)
                        frontier.append(item.to)
            plan.states = [state for state in plan.states if state.state_id in referenced]
            applied.append("removed_orphans")

        return RepairResult(plan=plan, applied_repairs=applied)
