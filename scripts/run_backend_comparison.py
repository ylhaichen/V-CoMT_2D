"""Compare planner backends on the same task set."""

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
    parser = argparse.ArgumentParser(description="Run backend comparison for V-CoMT_2D.")
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASK_SCENARIOS.keys()), default=sorted(TASK_SCENARIOS.keys()))
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per backend and task.")
    parser.add_argument("--output-dir", default="outputs/eval", help="Directory used for comparison artifacts.")
    parser.add_argument("--save-animation", action="store_true", help="Save animations for all compared runs.")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=[PlannerMode.DETERMINISTIC.value, PlannerMode.QWEN_VL.value],
        choices=[
            PlannerMode.DETERMINISTIC.value,
            PlannerMode.GPT.value,
            PlannerMode.QWEN_VL.value,
            PlannerMode.LLM_STUB.value,
            PlannerMode.VLM_STUB.value,
        ],
        help="Backends to compare.",
    )
    parser.add_argument("--model", default=None, help="Optional model override for model-based backends, for example gpt-5.4 or Qwen/Qwen2.5-VL-3B-Instruct.")
    parser.add_argument("--reasoning-effort", default=None, help="Optional reasoning effort override for GPT.")
    parser.add_argument("--scene-image", action="store_true", help="Render planner input images for local VLM backends.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    harness = EvaluationHarness()
    comparison = {}

    for backend in args.backends:
        request = EvalRequest(
            tasks=list(args.tasks),
            runs_per_task=args.runs,
            enable_animation=args.save_animation,
            output_dir=args.output_dir,
            planner_mode=backend,
            model_name=args.model if backend in {PlannerMode.GPT.value, PlannerMode.QWEN_VL.value} else None,
            reasoning_effort=args.reasoning_effort if backend == "gpt" else None,
            use_scene_image=args.scene_image if backend == PlannerMode.QWEN_VL.value else False,
        )
        summary = harness.run_batch(request)
        comparison[backend] = summary.to_dict()

    comparison_path = Path(args.output_dir) / "comparison_summary.json"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    print(f"comparison_summary: {comparison_path}")


if __name__ == "__main__":
    main()
