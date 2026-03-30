import shutil

from vcomt2d.fsm.executor import FSMExecutor
from vcomt2d.planner.main import DeterministicPlanner
from vcomt2d.planner.models import PlanningRequest
from vcomt2d.sim.tasks.fixtures import door_task_world
from vcomt2d.viz.export import AnimationExportError, save_animation_mp4, save_animation_with_fallback


def _door_trace():
    planner = DeterministicPlanner()
    result = planner.plan(PlanningRequest(request_id="door_anim", user_instruction="Both of you get into the next room", world_state=door_task_world()))
    execution = FSMExecutor().execute("door_anim", result.task_type, result.plan, door_task_world())
    return execution.trace


def test_demo_execution_produces_frames():
    trace = _door_trace()
    assert len(trace.frames) > 0


def test_mp4_export_or_clear_failure(tmp_path):
    trace = _door_trace()
    output_path = tmp_path / "door.mp4"
    if shutil.which("ffmpeg"):
        saved = save_animation_mp4(trace, str(output_path), fps=2)
        assert output_path.exists()
        assert saved.endswith(".mp4")
    else:
        try:
            save_animation_mp4(trace, str(output_path), fps=2)
        except AnimationExportError as exc:
            assert "ffmpeg" in str(exc).lower()
        else:
            raise AssertionError("Expected clear ffmpeg error when encoder is missing")


def test_animation_export_fallback_writes_artifact(tmp_path):
    trace = _door_trace()
    artifact = save_animation_with_fallback(trace, str(tmp_path / "door_anim"), fps=2)
    assert artifact.saved_path.endswith((".mp4", ".gif"))
    assert "saved" in artifact.message.lower()

