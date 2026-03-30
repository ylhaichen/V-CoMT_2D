import json
from pathlib import Path

from vcomt2d.eval.models import EvalRequest
from vcomt2d.eval.runner import EvaluationHarness
from vcomt2d.planner.models import PlanningResult


def test_eval_harness_creates_artifacts(tmp_path):
    output_dir = tmp_path / "eval_outputs"
    harness = EvaluationHarness()
    batch = harness.run_batch(EvalRequest(tasks=["door"], runs_per_task=1, output_dir=str(output_dir), enable_animation=False))

    run_dir = output_dir / "deterministic_template" / "door" / "run_000"
    assert run_dir.exists()
    assert (run_dir / "request.json").exists()
    assert (run_dir / "plan.json").exists()
    assert (run_dir / "final_plan.json").exists()
    assert (run_dir / "validation.json").exists()
    assert (run_dir / "semantic_sanity.json").exists()
    assert (run_dir / "trace.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "logs.txt").exists()
    assert (output_dir / "deterministic_template" / "batch_summary.json").exists()
    assert batch.total_runs == 1


def test_eval_summary_schema_contains_expected_fields(tmp_path):
    output_dir = tmp_path / "eval_outputs"
    harness = EvaluationHarness()
    harness.run_batch(EvalRequest(tasks=["search"], runs_per_task=1, output_dir=str(output_dir)))

    summary_path = output_dir / "deterministic_template" / "search" / "run_000" / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_keys = {
        "run_id",
        "task_key",
        "task_family",
        "instruction",
        "backend_name",
        "model_name",
        "planner_success",
        "validation_passed",
        "semantic_passed",
        "execution_success",
        "retries_used",
        "repairs_applied",
        "step_count",
        "duration_seconds",
        "failure_reason",
        "artifacts",
    }
    assert expected_keys <= set(payload)


def test_batch_runner_small_fixture_set(tmp_path):
    output_dir = tmp_path / "eval_outputs"
    harness = EvaluationHarness()
    batch = harness.run_batch(EvalRequest(tasks=["door", "relay"], runs_per_task=1, output_dir=str(output_dir)))

    assert batch.total_runs == 2
    assert set(batch.per_task) == {"door", "relay"}
    assert (output_dir / "deterministic_template" / "batch_summary.json").exists()


def test_failed_runs_still_produce_useful_summaries(tmp_path):
    class FailingPlanner:
        def plan(self, request):
            return PlanningResult(
                success=False,
                task_type=None,
                reasoning_summary="Forced planner failure for harness testing.",
                plan=None,
                validation={"valid": False, "errors": []},
                semantic_validation={"passed": False, "warnings": [], "errors": [], "suggested_repairs": [], "debug_info": {}},
                retries_used=0,
                failure_reason="forced_failure",
                debug_info={"applied_repairs": [], "backend": "failing_planner"},
            )

    output_dir = tmp_path / "eval_outputs"
    harness = EvaluationHarness(planner=FailingPlanner())
    batch = harness.run_batch(EvalRequest(tasks=["door"], runs_per_task=1, output_dir=str(output_dir)))

    summary_path = output_dir / "failing_planner" / "door" / "run_000" / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert batch.failed_runs == 1
    assert payload["backend_name"] == "failing_planner"
    assert payload["planner_success"] is False
    assert payload["execution_success"] is False
    assert payload["failure_reason"] == "forced_failure"
    assert payload["artifacts"]["summary_json"].endswith("summary.json")


def test_animation_paths_recorded_when_enabled(tmp_path):
    output_dir = tmp_path / "eval_outputs"
    harness = EvaluationHarness()
    harness.run_batch(EvalRequest(tasks=["door"], runs_per_task=1, output_dir=str(output_dir), enable_animation=True))

    summary_path = output_dir / "deterministic_template" / "door" / "run_000" / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    animation_path = payload["artifacts"]["animation"]
    assert animation_path is not None
    assert Path(animation_path).exists()
    assert animation_path.endswith((".mp4", ".gif"))
