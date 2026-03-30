"""Utilities for consistent FSM construction."""

from __future__ import annotations

from typing import Any, Dict, List

from vcomt2d.fsm.schema import ActionSpec, ConditionSpec, FSMPlan, StateSpec, TransitionSpec


def action(skill: str, **params) -> ActionSpec:
    return ActionSpec(skill=skill, params=params)


def condition(kind: str, **args) -> ConditionSpec:
    return ConditionSpec(kind=kind, args=args)


def transition(to: str, condition_spec: ConditionSpec, notes: str = "") -> TransitionSpec:
    return TransitionSpec(to=to, condition=condition_spec, notes=notes)


def state(state_id: str, robot_a: ActionSpec, robot_b: ActionSpec, transitions: List[TransitionSpec], notes: str = "") -> StateSpec:
    return StateSpec(state_id=state_id, robot_a=robot_a, robot_b=robot_b, transitions=transitions, terminal=False, status=None, notes=notes)


def terminal(state_id: str, status: str, notes: str = "") -> StateSpec:
    return StateSpec(state_id=state_id, transitions=[], terminal=True, status=status, notes=notes)


def plan(task_description: str, reasoning_summary: str, initial_state: str, states: List[StateSpec]) -> FSMPlan:
    return FSMPlan(task_description=task_description, reasoning_summary=reasoning_summary, initial_state=initial_state, states=states)

