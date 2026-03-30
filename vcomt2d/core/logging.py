"""Trace models shared by executor and visualization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .types import to_serializable


@dataclass
class TraceFrame:
    tick: int
    state_id: str
    robot_actions: Dict[str, Dict[str, Any]]
    world_snapshot: Dict[str, Any]
    transition_reason: str
    next_state: Optional[str]
    events: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return to_serializable(self)


@dataclass
class ExecutionTrace:
    request_id: str
    task_type: str
    status: str
    frames: List[TraceFrame] = field(default_factory=list)
    terminal_state: Optional[str] = None
    events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return to_serializable(self)

