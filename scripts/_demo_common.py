"""Shared helpers for planner and animation demo scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vcomt2d.fsm.executor import FSMExecutor
from vcomt2d.planner.main import DeterministicPlanner
from vcomt2d.planner.models import PlanningRequest
from vcomt2d.sim.tasks.fixtures import door_task_world, herding_task_world, relay_task_world, search_task_world
from vcomt2d.viz.export import save_animation_with_fallback


TASK_DEMOS: Dict[str, Tuple[str, Callable]] = {
    "door": ("Both of you get into the next room", door_task_world),
    "herding": ("Push the ball into the corner", herding_task_world),
    "search": ("Find the red box and meet there", search_task_world),
    "relay": ("Move the box to the far goal", relay_task_world),
}


def run_planning_demo(task_name: str) -> None:
    instruction, fixture_builder = TASK_DEMOS[task_name]
    planner = DeterministicPlanner()
    request = PlanningRequest(request_id=f"{task_name}_demo", user_instruction=instruction, world_state=fixture_builder())
    result = planner.plan(request)
    print(f"== {task_name.upper()} Planning Demo ==")
    print("reasoning_summary:")
    print(result.reasoning_summary)
    print("validation:")
    print(json.dumps(result.validation, indent=2))
    print("semantic_validation:")
    print(json.dumps(result.semantic_validation, indent=2))
    print("plan:")
    print(json.dumps(result.plan.to_dict() if result.plan else result.to_dict(), indent=2))


def run_animation_demo(task_name: str) -> None:
    instruction, fixture_builder = TASK_DEMOS[task_name]
    planner = DeterministicPlanner()
    request = PlanningRequest(request_id=f"{task_name}_demo", user_instruction=instruction, world_state=fixture_builder())
    result = planner.plan(request)
    if not result.success or result.plan is None:
        raise SystemExit(f"Planning failed: {result.failure_reason}")
    execution = FSMExecutor().execute(request.request_id, result.task_type, result.plan, fixture_builder())
    output_stem = Path("outputs/animations") / f"{task_name}_demo"
    artifact = save_animation_with_fallback(execution.trace, str(output_stem), fps=request.planning_config.animation_fps)
    print(f"== {task_name.upper()} Animation Demo ==")
    print(f"execution_status: {execution.status}")
    print(f"saved_path: {artifact.saved_path}")
    print(f"format: {artifact.format}")
    print(f"message: {artifact.message}")
