"""Evaluation request, result, and batch summary models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from vcomt2d.core.types import to_serializable


@dataclass
class EvalRequest:
    tasks: List[str]
    runs_per_task: int = 1
    enable_animation: bool = False
    output_dir: str = "outputs/eval"
    save_logs: bool = True
    planner_mode: str = "deterministic"
    model_name: Optional[str] = None
    reasoning_effort: Optional[str] = None
    use_scene_image: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return to_serializable(self)


@dataclass
class EvalArtifactPaths:
    run_dir: str
    request_json: Optional[str] = None
    prompt_json: Optional[str] = None
    raw_response_json: Optional[str] = None
    candidate_plan_json: Optional[str] = None
    final_plan_json: Optional[str] = None
    validation_json: Optional[str] = None
    semantic_sanity_json: Optional[str] = None
    plan_json: Optional[str] = None
    trace_json: Optional[str] = None
    summary_json: Optional[str] = None
    animation: Optional[str] = None
    logs_txt: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return to_serializable(self)


@dataclass
class EvalRunSummary:
    run_id: str
    task_key: str
    task_family: Optional[str]
    instruction: str
    backend_name: str
    model_name: Optional[str]
    planner_success: bool
    validation_passed: bool
    semantic_passed: bool
    execution_success: bool
    retries_used: int
    repairs_applied: List[str]
    step_count: int
    duration_seconds: float
    failure_reason: Optional[str]
    artifacts: EvalArtifactPaths

    def to_dict(self) -> Dict[str, Any]:
        return to_serializable(self)


@dataclass
class EvalBatchSummary:
    request: EvalRequest
    total_runs: int
    successful_runs: int
    failed_runs: int
    per_task: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    runs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return to_serializable(self)
