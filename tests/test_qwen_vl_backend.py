import json
from pathlib import Path

from vcomt2d.eval.models import EvalRequest
from vcomt2d.eval.runner import EvaluationHarness
from vcomt2d.planner.intent_parser import parse_intent
from vcomt2d.planner.main import DeterministicPlanner, planner_from_mode
from vcomt2d.planner.models import PlanningContext, PlanningRequest, TaskType
from vcomt2d.planner.qwen_vl_backend import QwenVLPlannerBackend
from vcomt2d.planner.qwen_vl_config import QwenVLPlannerConfig
from vcomt2d.planner.role_assignment import assign_roles
from vcomt2d.planner.scene_interpreter import interpret_scene
from vcomt2d.sim.tasks.fixtures import door_task_world


class FakeQwenEngine:
    engine_name = "fake_qwen_engine"

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "messages": [{"role": "user", "content": [{"type": "text", "text": "fake"}]}],
            "chat_text": "fake chat text",
            "output_text": json.dumps(self.payload),
            "raw_response": {"generated_text": json.dumps(self.payload)},
        }


class FailingQwenEngine:
    engine_name = "fake_qwen_engine"

    def generate(self, **kwargs):
        raise RuntimeError("CUDA out of memory while generating")


def _valid_door_payload():
    return {
        "task_description": "Both of you get into the next room",
        "reasoning_summary": "Door task with a holder and a passer.",
        "fsm": {
            "initial_state": "S0_APPROACH",
            "states": [
                {
                    "state_id": "S0_APPROACH",
                    "robot_a": {"skill": "MOVE_TO", "params": {"target_id": "door_main"}},
                    "robot_b": {"skill": "MOVE_TO", "params": {"target_id": "door_wait"}},
                    "transitions": [
                        {"to": "S1_HOLD_DOOR", "condition": {"kind": "all_actions_done", "args": {}}, "notes": "approach complete"},
                        {"to": "S_FAIL", "condition": {"kind": "timeout", "args": {"ticks": 8}}, "notes": "timeout"},
                    ],
                    "terminal": False,
                    "status": None,
                    "notes": "",
                },
                {
                    "state_id": "S1_HOLD_DOOR",
                    "robot_a": {"skill": "HOLD_POSITION", "params": {"ticks": 2}},
                    "robot_b": {"skill": "WAIT_UNTIL", "params": {"condition": {"kind": "flag_true", "args": {"flag": "door_held"}}}},
                    "transitions": [
                        {"to": "S2_PASS_PARTNER", "condition": {"kind": "flag_true", "args": {"flag": "door_held"}}, "notes": "held"},
                        {"to": "S_FAIL", "condition": {"kind": "timeout", "args": {"ticks": 5}}, "notes": "timeout"},
                    ],
                    "terminal": False,
                    "status": None,
                    "notes": "holder=robot_a",
                },
                {
                    "state_id": "S2_PASS_PARTNER",
                    "robot_a": {"skill": "HOLD_POSITION", "params": {"ticks": 2}},
                    "robot_b": {"skill": "MOVE_TO", "params": {"target_id": "room_goal"}},
                    "transitions": [
                        {"to": "S3_FOLLOW", "condition": {"kind": "robot_in_region", "args": {"robot": "robot_b", "region_id": "room_goal"}}, "notes": "passed"},
                        {"to": "S_FAIL", "condition": {"kind": "timeout", "args": {"ticks": 5}}, "notes": "timeout"},
                    ],
                    "terminal": False,
                    "status": None,
                    "notes": "",
                },
                {
                    "state_id": "S3_FOLLOW",
                    "robot_a": {"skill": "FOLLOW", "params": {"target_robot": "robot_b"}},
                    "robot_b": {"skill": "HOLD_POSITION", "params": {"ticks": 1}},
                    "transitions": [
                        {"to": "S_DONE", "condition": {"kind": "both_in_region", "args": {"region_id": "room_goal"}}, "notes": "done"},
                        {"to": "S_FAIL", "condition": {"kind": "timeout", "args": {"ticks": 8}}, "notes": "timeout"},
                    ],
                    "terminal": False,
                    "status": None,
                    "notes": "",
                },
                {"state_id": "S_DONE", "robot_a": None, "robot_b": None, "transitions": [], "terminal": True, "status": "success", "notes": ""},
                {"state_id": "S_FAIL", "robot_a": None, "robot_b": None, "transitions": [], "terminal": True, "status": "failure", "notes": ""},
            ],
        },
    }


def _door_context():
    world = door_task_world()
    request = PlanningRequest(request_id="door_qwen", user_instruction="Both of you get into the next room", world_state=world, planner_mode="qwen_vl")
    intent = parse_intent(request.user_instruction)
    scene_facts = interpret_scene(TaskType.T1_DOOR_WEDGE_PASS_THROUGH, world)
    roles = assign_roles(scene_facts, world)
    return PlanningContext(
        request=request,
        intent=intent,
        scene_facts=scene_facts,
        roles=roles,
        reasoning_summary="Door task context for Qwen backend testing.",
    )


def test_qwen_config_loads_from_env(monkeypatch):
    monkeypatch.setenv("QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    monkeypatch.setenv("QWEN_VL_MAX_NEW_TOKENS", "1024")
    monkeypatch.setenv("QWEN_VL_TEMPERATURE", "0.1")
    monkeypatch.setenv("QWEN_VL_LOAD_IN_4BIT", "true")
    monkeypatch.setenv("QWEN_VL_USE_SCENE_IMAGE", "true")

    config = QwenVLPlannerConfig.from_env()
    assert config.model == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert config.max_new_tokens == 1024
    assert config.temperature == 0.1
    assert config.load_in_4bit is True
    assert config.use_scene_image is True


def test_qwen_backend_builds_candidate_and_optional_scene_image():
    engine = FakeQwenEngine(_valid_door_payload())
    backend = QwenVLPlannerBackend(
        config=QwenVLPlannerConfig(model="Qwen/Qwen2.5-VL-3B-Instruct", use_scene_image=True),
        engine=engine,
    )

    candidate = backend.build_candidate(_door_context())

    assert candidate.backend_name == "qwen_vl"
    assert candidate.model_name == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert candidate.plan.initial_state == "S0_APPROACH"
    assert engine.calls
    scene_image_path = engine.calls[0]["scene_image_path"]
    assert scene_image_path is not None
    assert Path(scene_image_path).exists()
    assert candidate.debug_info["scene_image_path"] == scene_image_path


def test_qwen_engine_resolves_local_snapshot_path(tmp_path, monkeypatch):
    cache_root = tmp_path / ".cache" / "huggingface" / "hub"
    repo_dir = cache_root / "models--Qwen--Qwen2.5-VL-3B-Instruct"
    snapshot_dir = repo_dir / "snapshots" / "revision123"
    snapshot_dir.mkdir(parents=True)
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text("revision123\n", encoding="utf-8")

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    engine = QwenVLPlannerBackend().engine

    resolved = engine._resolve_model_source(
        QwenVLPlannerConfig(
            model="Qwen/Qwen2.5-VL-3B-Instruct",
            local_files_only=True,
        )
    )

    assert resolved == str(snapshot_dir)


def test_planner_from_mode_returns_qwen_backend():
    planner = planner_from_mode("qwen_vl")
    assert isinstance(planner, DeterministicPlanner)
    assert planner.backend.backend_name == "qwen_vl"


def test_qwen_eval_run_saves_replayable_artifacts(tmp_path):
    planner = DeterministicPlanner(
        backend=QwenVLPlannerBackend(
            config=QwenVLPlannerConfig(model="Qwen/Qwen2.5-VL-3B-Instruct"),
            engine=FakeQwenEngine(_valid_door_payload()),
        )
    )
    harness = EvaluationHarness(planner=planner)
    output_dir = tmp_path / "eval_outputs"

    harness.run_batch(EvalRequest(tasks=["door"], runs_per_task=1, output_dir=str(output_dir), planner_mode="qwen_vl"))

    run_dir = output_dir / "qwen_vl" / "door" / "run_000"
    assert (run_dir / "request.json").exists()
    assert (run_dir / "prompt.json").exists()
    assert (run_dir / "raw_response.json").exists()
    assert (run_dir / "candidate_plan.json").exists()
    assert (run_dir / "final_plan.json").exists()
    assert (run_dir / "summary.json").exists()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["backend_name"] == "qwen_vl"
    assert summary["model_name"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert summary["planner_success"] is True


def test_qwen_eval_run_records_runtime_error_details(tmp_path):
    planner = DeterministicPlanner(
        backend=QwenVLPlannerBackend(
            config=QwenVLPlannerConfig(model="Qwen/Qwen2.5-VL-3B-Instruct"),
            engine=FailingQwenEngine(),
        )
    )
    harness = EvaluationHarness(planner=planner)
    output_dir = tmp_path / "eval_outputs"

    batch = harness.run_batch(EvalRequest(tasks=["door"], runs_per_task=1, output_dir=str(output_dir), planner_mode="qwen_vl"))

    run_dir = output_dir / "qwen_vl" / "door" / "run_000"
    assert batch.successful_runs == 1
    prompt = json.loads((run_dir / "prompt.json").read_text(encoding="utf-8"))
    logs = (run_dir / "logs.txt").read_text(encoding="utf-8")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["planner_success"] is True
    assert "backend::resynthesize_task_template" in summary["repairs_applied"]
    assert prompt["backend_error"]["exception_type"] == "RuntimeError"
    assert "CUDA out of memory" in prompt["backend_error"]["message"]
    assert "backend_error=" in logs
