"""Batch evaluation entrypoint."""

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
from vcomt2d.core.status import PlannerMode
from vcomt2d.sim.tasks.catalog import TASK_SCENARIOS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run batch planner/executor evaluation for V-CoMT_2D.")
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASK_SCENARIOS.keys()), default=sorted(TASK_SCENARIOS.keys()))
    parser.add_argument("--runs", type=int, default=1, help="Number of deterministic runs per task.")
    parser.add_argument("--output-dir", default="outputs/eval", help="Directory used for evaluation artifacts.")
    parser.add_argument("--save-animation", "--animation", dest="animation", action="store_true", help="Save animation artifacts for each run.")
    parser.add_argument("--no-logs", action="store_true", help="Skip logs.txt generation.")
    parser.add_argument(
        "--backend",
        "--planner-mode",
        dest="planner_mode",
        default=PlannerMode.DETERMINISTIC.value,
        choices=[
            PlannerMode.DETERMINISTIC.value,
            PlannerMode.GPT.value,
            PlannerMode.QWEN_VL.value,
            PlannerMode.LLM_STUB.value,
            PlannerMode.VLM_STUB.value,
            PlannerMode.SCRIPTED.value,
            PlannerMode.REPAIR.value,
        ],
        help="Planner backend mode used for candidate generation.",
    )
    parser.add_argument("--model", default=None, help="Override the backend model name, for example gpt-5.4.")
    parser.add_argument("--reasoning-effort", default=None, help="Override backend reasoning effort, for example low, medium, or high.")
    parser.add_argument("--scene-image", action="store_true", help="Render the planner input scene as an image for backends that support image-conditioned planning.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    request = EvalRequest(
        tasks=list(args.tasks),
        runs_per_task=args.runs,
        enable_animation=args.animation,
        output_dir=args.output_dir,
        save_logs=not args.no_logs,
        planner_mode=args.planner_mode,
        model_name=args.model,
        reasoning_effort=args.reasoning_effort,
        use_scene_image=args.scene_image,
    )
    summary = EvaluationHarness().run_batch(request)
    print(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    main()
