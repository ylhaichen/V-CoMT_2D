"""Compare deterministic and GPT planner backends on the same task set."""

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
from vcomt2d.sim.tasks.catalog import TASK_SCENARIOS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic vs GPT backend comparison for V-CoMT_2D.")
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASK_SCENARIOS.keys()), default=sorted(TASK_SCENARIOS.keys()))
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per backend and task.")
    parser.add_argument("--output-dir", default="outputs/eval", help="Directory used for comparison artifacts.")
    parser.add_argument("--save-animation", action="store_true", help="Save animations for all compared runs.")
    parser.add_argument("--model", default=None, help="Optional GPT model override, for example gpt-5.4.")
    parser.add_argument("--reasoning-effort", default=None, help="Optional GPT reasoning effort override.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    harness = EvaluationHarness()
    comparison = {}

    for backend in ("deterministic", "gpt"):
        request = EvalRequest(
            tasks=list(args.tasks),
            runs_per_task=args.runs,
            enable_animation=args.save_animation,
            output_dir=args.output_dir,
            planner_mode=backend,
            model_name=args.model if backend == "gpt" else None,
            reasoning_effort=args.reasoning_effort if backend == "gpt" else None,
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
