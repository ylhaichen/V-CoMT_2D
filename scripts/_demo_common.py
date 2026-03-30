"""Shared helpers for planner and animation demo scripts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vcomt2d.eval.models import EvalRequest
from vcomt2d.eval.runner import EvaluationHarness
from vcomt2d.planner.main import planner_from_mode
from vcomt2d.planner.models import PlanningRequest
from vcomt2d.planner.config import PlanningConfig
from vcomt2d.sim.tasks.catalog import TASK_SCENARIOS


def build_demo_parser(description: str, default_output_dir: str = "outputs/demos") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--backend", default="deterministic", choices=["deterministic", "gpt", "llm_stub", "vlm_stub"], help="Planner backend to use.")
    parser.add_argument("--model", default=None, help="Optional backend model override.")
    parser.add_argument("--reasoning-effort", default=None, help="Optional reasoning effort override.")
    parser.add_argument("--output-dir", default=default_output_dir, help="Directory used for saved demo artifacts.")
    return parser


def run_planning_demo(task_name: str, backend: str = "deterministic", model: str | None = None, reasoning_effort: str | None = None) -> None:
    scenario = TASK_SCENARIOS[task_name]
    planner = planner_from_mode(backend)
    request = PlanningRequest(
        request_id=f"{task_name}_demo",
        user_instruction=scenario.instruction,
        world_state=scenario.fixture_builder(),
        planner_mode=backend,
        planning_config=PlanningConfig(openai_model=model, openai_reasoning_effort=reasoning_effort),
    )
    result = planner.plan(request)
    print(f"== {task_name.upper()} Planning Demo ==")
    print(f"backend: {backend}")
    print(f"model: {result.debug_info.get('model_name')}")
    print("reasoning_summary:")
    print(result.reasoning_summary)
    print("validation:")
    print(json.dumps(result.validation, indent=2))
    print("semantic_validation:")
    print(json.dumps(result.semantic_validation, indent=2))
    print("plan:")
    print(json.dumps(result.plan.to_dict() if result.plan else result.to_dict(), indent=2))


def run_animation_demo(
    task_name: str,
    backend: str = "deterministic",
    model: str | None = None,
    reasoning_effort: str | None = None,
    output_dir: str = "outputs/demos",
) -> None:
    harness = EvaluationHarness()
    batch = harness.run_batch(
        EvalRequest(
            tasks=[task_name],
            runs_per_task=1,
            enable_animation=True,
            output_dir=output_dir,
            planner_mode=backend,
            model_name=model,
            reasoning_effort=reasoning_effort,
        )
    )
    summary = batch.runs[0]
    print(f"== {task_name.upper()} Animation Demo ==")
    print(f"backend: {backend}")
    print(f"model: {summary.get('model_name')}")
    print(f"execution_success: {summary['execution_success']}")
    print(f"saved_path: {summary['artifacts']['animation']}")
    print(f"summary_path: {summary['artifacts']['summary_json']}")
