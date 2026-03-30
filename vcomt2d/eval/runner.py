"""Batch evaluation runner for planner/executor experiments."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from vcomt2d.fsm.executor import ExecutionResult, FSMExecutor
from vcomt2d.planner.base import Planner
from vcomt2d.planner.main import planner_from_mode
from vcomt2d.planner.models import PlanningRequest, PlanningResult
from vcomt2d.sim.tasks.catalog import TASK_SCENARIOS
from vcomt2d.viz.export import save_animation_with_fallback
from .artifacts import prepare_run_dir, write_logs, write_plan, write_summary, write_trace
from .models import EvalArtifactPaths, EvalBatchSummary, EvalRequest, EvalRunSummary


class EvaluationHarness:
    """Deterministic harness for batch planning/execution experiments."""

    def __init__(self, planner: Optional[Planner] = None, executor: Optional[FSMExecutor] = None):
        self.planner = planner
        self.executor = executor or FSMExecutor()

    def run_batch(self, request: EvalRequest) -> EvalBatchSummary:
        summaries: List[EvalRunSummary] = []
        for task_key in request.tasks:
            if task_key not in TASK_SCENARIOS:
                raise ValueError(f"Unknown task key '{task_key}'. Valid tasks: {sorted(TASK_SCENARIOS)}")
            for run_index in range(request.runs_per_task):
                summaries.append(self.run_single(task_key=task_key, run_index=run_index, request=request))

        batch_summary = EvalBatchSummary(
            request=request,
            total_runs=len(summaries),
            successful_runs=sum(1 for item in summaries if item.execution_success and item.planner_success),
            failed_runs=sum(1 for item in summaries if not (item.execution_success and item.planner_success)),
            per_task=self._aggregate_by_task(summaries),
            runs=[item.to_dict() for item in summaries],
        )
        batch_path = Path(request.output_dir) / "batch_summary.json"
        batch_path.parent.mkdir(parents=True, exist_ok=True)
        batch_path.write_text(json.dumps(batch_summary.to_dict(), indent=2) + "\n", encoding="utf-8")
        return batch_summary

    def run_single(self, task_key: str, run_index: int, request: EvalRequest) -> EvalRunSummary:
        scenario = TASK_SCENARIOS[task_key]
        run_id = f"run_{run_index:03d}"
        run_dir = prepare_run_dir(request.output_dir, task_key, run_id)
        artifact_paths = EvalArtifactPaths(
            run_dir=str(run_dir),
            plan_json=str(run_dir / "plan.json"),
            trace_json=str(run_dir / "trace.json"),
            summary_json=str(run_dir / "summary.json"),
            logs_txt=str(run_dir / "logs.txt") if request.save_logs else None,
        )
        started = time.perf_counter()

        planning_request = PlanningRequest(
            request_id=f"{task_key}_{run_id}",
            user_instruction=scenario.instruction,
            world_state=scenario.fixture_builder(),
            planner_mode=request.planner_mode,
        )
        planner = self.planner or planner_from_mode(request.planner_mode)
        planning_result: PlanningResult = planner.plan(planning_request)
        execution_result: Optional[ExecutionResult] = None
        if planning_result.success and planning_result.plan is not None:
            execution_result = self.executor.execute(planning_request.request_id, planning_result.task_type, planning_result.plan, scenario.fixture_builder())

        write_plan(run_dir, planning_result.plan)
        write_trace(run_dir, None if execution_result is None else execution_result.trace)

        if request.enable_animation and execution_result is not None:
            animation_stem = str(run_dir / "animation")
            animation_artifact = save_animation_with_fallback(
                execution_result.trace,
                animation_stem,
                fps=planning_request.planning_config.animation_fps,
            )
            artifact_paths.animation = animation_artifact.saved_path
        if request.save_logs:
            write_logs(run_dir, self._build_log_lines(planning_result, execution_result))

        repairs_applied = planning_result.debug_info.get("applied_repairs", [])
        duration = time.perf_counter() - started
        step_count = len(execution_result.trace.frames) if execution_result is not None else 0
        failure_reason = planning_result.failure_reason or (execution_result.failure_reason if execution_result is not None else None)

        summary = EvalRunSummary(
            run_id=run_id,
            task_key=task_key,
            task_family=planning_result.task_type,
            instruction=scenario.instruction,
            planner_success=planning_result.success,
            validation_passed=bool(planning_result.validation.get("valid", False)),
            semantic_passed=bool(planning_result.semantic_validation.get("passed", False)),
            execution_success=bool(execution_result.success) if execution_result is not None else False,
            retries_used=planning_result.retries_used,
            repairs_applied=list(repairs_applied),
            step_count=step_count,
            duration_seconds=round(duration, 6),
            failure_reason=failure_reason,
            artifacts=artifact_paths,
        )
        write_summary(run_dir, summary)
        return summary

    def _build_log_lines(self, planning_result: PlanningResult, execution_result: Optional[ExecutionResult]) -> List[str]:
        lines = [
            f"planner_success={planning_result.success}",
            f"task_type={planning_result.task_type}",
            f"validation_passed={planning_result.validation.get('valid', False)}",
            f"semantic_passed={planning_result.semantic_validation.get('passed', False)}",
            f"retries_used={planning_result.retries_used}",
            f"failure_reason={planning_result.failure_reason}",
            f"repairs_applied={planning_result.debug_info.get('applied_repairs', [])}",
        ]
        if execution_result is not None:
            lines.extend(
                [
                    f"execution_success={execution_result.success}",
                    f"execution_status={execution_result.status}",
                    f"execution_failure_reason={execution_result.failure_reason}",
                    f"step_count={len(execution_result.trace.frames)}",
                ]
            )
        return lines

    def _aggregate_by_task(self, summaries: List[EvalRunSummary]) -> Dict[str, Dict[str, object]]:
        per_task: Dict[str, Dict[str, object]] = {}
        for item in summaries:
            task_summary = per_task.setdefault(
                item.task_key,
                {
                    "task_family": item.task_family,
                    "runs": 0,
                    "planner_successes": 0,
                    "execution_successes": 0,
                    "validation_passes": 0,
                    "semantic_passes": 0,
                    "failures": [],
                },
            )
            task_summary["runs"] += 1
            task_summary["planner_successes"] += int(item.planner_success)
            task_summary["execution_successes"] += int(item.execution_success)
            task_summary["validation_passes"] += int(item.validation_passed)
            task_summary["semantic_passes"] += int(item.semantic_passed)
            if item.failure_reason:
                task_summary["failures"].append({"run_id": item.run_id, "failure_reason": item.failure_reason})
        return per_task
