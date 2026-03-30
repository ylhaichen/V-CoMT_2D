"""Planner request/response and intermediate reasoning models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from vcomt2d.core.status import PlannerMode
from vcomt2d.core.types import to_serializable
from vcomt2d.fsm.schema import FSMPlan
from vcomt2d.sim.entities import WorldState
from .config import PlanningConfig


class TaskType(str, Enum):
    T1_DOOR_WEDGE_PASS_THROUGH = "T1_Door_Wedge_Pass_Through"
    T2_HERDING_CORRALLING = "T2_Herding_Corralling"
    T4_COLLABORATIVE_SEARCH_CONVERGE = "T4_Collaborative_Search_Converge"
    T6_RELAY_DELIVERY = "T6_Relay_Delivery"


@dataclass
class PlanningRequest:
    request_id: str
    user_instruction: str
    world_state: WorldState
    planning_config: PlanningConfig = field(default_factory=PlanningConfig)
    history: List[Dict[str, Any]] = field(default_factory=list)
    retry_count: int = 0
    planner_mode: str = PlannerMode.DETERMINISTIC.value

    def to_dict(self) -> Dict[str, Any]:
        return to_serializable(self)


@dataclass
class IntentParseResult:
    task_type: Optional[TaskType]
    confidence: float
    matched_keywords: List[str] = field(default_factory=list)
    inferred_entities: Dict[str, str] = field(default_factory=dict)
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return to_serializable(self)


@dataclass
class SceneFacts:
    task_type: TaskType
    relevant_door_id: Optional[str] = None
    target_object_id: Optional[str] = None
    goal_region_id: Optional[str] = None
    wait_region_id: Optional[str] = None
    search_region_ids: List[str] = field(default_factory=list)
    handoff_region_id: Optional[str] = None
    staging_region_ids: List[str] = field(default_factory=list)
    topology_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return to_serializable(self)


@dataclass
class RoleAssignment:
    task_type: TaskType
    roles: Dict[str, Any]
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return to_serializable(self)


@dataclass
class PlanningContext:
    request: PlanningRequest
    intent: IntentParseResult
    scene_facts: SceneFacts
    roles: RoleAssignment
    reasoning_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return to_serializable(self)


@dataclass
class BackendPlanCandidate:
    backend_name: str
    plan: FSMPlan
    model_name: Optional[str] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = to_serializable(self)
        payload["plan"] = self.plan.to_dict()
        return payload


@dataclass
class SemanticIssue:
    code: str
    message: str
    repairable: bool = False
    suggested_repair: Optional[str] = None
    state_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return to_serializable(self)


@dataclass
class SemanticCheckResult:
    passed: bool
    warnings: List[SemanticIssue] = field(default_factory=list)
    errors: List[SemanticIssue] = field(default_factory=list)
    suggested_repairs: List[str] = field(default_factory=list)
    debug_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_serializable(self)


@dataclass
class PlanningResult:
    success: bool
    task_type: Optional[str]
    reasoning_summary: str
    plan: Optional[FSMPlan]
    validation: Dict[str, Any]
    semantic_validation: Dict[str, Any]
    retries_used: int
    failure_reason: Optional[str] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = to_serializable(self)
        if self.plan is not None:
            payload["plan"] = self.plan.to_dict()
        return payload
