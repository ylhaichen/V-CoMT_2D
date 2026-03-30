"""JSON-serializable FSM schema used by planner and executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from vcomt2d.core.types import to_serializable


@dataclass
class ConditionSpec:
    kind: str
    args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_serializable(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ConditionSpec":
        return ConditionSpec(kind=data["kind"], args=data.get("args", {}))


@dataclass
class ActionSpec:
    skill: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_serializable(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ActionSpec":
        return ActionSpec(skill=data["skill"], params=data.get("params", {}))


@dataclass
class TransitionSpec:
    to: str
    condition: ConditionSpec
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"to": self.to, "condition": self.condition.to_dict(), "notes": self.notes}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TransitionSpec":
        return TransitionSpec(to=data["to"], condition=ConditionSpec.from_dict(data["condition"]), notes=data.get("notes", ""))


@dataclass
class StateSpec:
    state_id: str
    robot_a: Optional[ActionSpec] = None
    robot_b: Optional[ActionSpec] = None
    transitions: List[TransitionSpec] = field(default_factory=list)
    terminal: bool = False
    status: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "robot_a": None if self.robot_a is None else self.robot_a.to_dict(),
            "robot_b": None if self.robot_b is None else self.robot_b.to_dict(),
            "transitions": [transition.to_dict() for transition in self.transitions],
            "terminal": self.terminal,
            "status": self.status,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "StateSpec":
        return StateSpec(
            state_id=data["state_id"],
            robot_a=None if data.get("robot_a") is None else ActionSpec.from_dict(data["robot_a"]),
            robot_b=None if data.get("robot_b") is None else ActionSpec.from_dict(data["robot_b"]),
            transitions=[TransitionSpec.from_dict(item) for item in data.get("transitions", [])],
            terminal=data.get("terminal", False),
            status=data.get("status"),
            notes=data.get("notes", ""),
        )


@dataclass
class FSMPlan:
    task_description: str
    reasoning_summary: str
    initial_state: str
    states: List[StateSpec]

    def state_map(self) -> Dict[str, StateSpec]:
        return {state.state_id: state for state in self.states}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_description": self.task_description,
            "reasoning_summary": self.reasoning_summary,
            "fsm": {
                "initial_state": self.initial_state,
                "states": [state.to_dict() for state in self.states],
            },
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "FSMPlan":
        fsm = data["fsm"]
        return FSMPlan(
            task_description=data["task_description"],
            reasoning_summary=data["reasoning_summary"],
            initial_state=fsm["initial_state"],
            states=[StateSpec.from_dict(item) for item in fsm["states"]],
        )

