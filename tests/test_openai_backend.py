import json

from vcomt2d.eval.models import EvalRequest
from vcomt2d.eval.runner import EvaluationHarness
from vcomt2d.planner.intent_parser import parse_intent
from vcomt2d.planner.main import DeterministicPlanner, planner_from_mode
from vcomt2d.planner.models import PlanningContext, PlanningRequest, TaskType
from vcomt2d.planner.openai_backend import OpenAIGPTPlannerBackend
from vcomt2d.planner.openai_config import OpenAIPlannerConfig
from vcomt2d.planner.role_assignment import assign_roles
from vcomt2d.planner.scene_interpreter import interpret_scene
from vcomt2d.sim.tasks.fixtures import door_task_world


class FakeResponse:
    def __init__(self, payload):
        self.output_text = json.dumps(payload)
        self._payload = {
            "id": "resp_test",
            "output": [{"content": [{"type": "output_text", "text": self.output_text}]}],
        }

    def model_dump(self, mode="json"):
        return self._payload


class FakeResponsesAPI:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self.payload)


class FakeOpenAIClient:
    def __init__(self, payload):
        self.responses = FakeResponsesAPI(payload)


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
    request = PlanningRequest(request_id="door_gpt", user_instruction="Both of you get into the next room", world_state=world, planner_mode="gpt")
    intent = parse_intent(request.user_instruction)
    scene_facts = interpret_scene(TaskType.T1_DOOR_WEDGE_PASS_THROUGH, world)
    roles = assign_roles(scene_facts, world)
    return PlanningContext(
        request=request,
        intent=intent,
        scene_facts=scene_facts,
        roles=roles,
        reasoning_summary="Door task context for GPT backend testing.",
    )


def test_openai_config_loads_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "high")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "33")
    monkeypatch.setenv("OPENAI_RETRY_COUNT", "4")
    monkeypatch.setenv("OPENAI_MAX_OUTPUT_TOKENS", "2048")

    config = OpenAIPlannerConfig.from_env()
    assert config.api_key == "test-key"
    assert config.model == "gpt-5.4"
    assert config.reasoning_effort == "high"
    assert config.timeout_seconds == 33.0
    assert config.retry_count == 4
    assert config.max_output_tokens == 2048


def test_openai_backend_builds_structured_request_and_parses_plan():
    client = FakeOpenAIClient(_valid_door_payload())
    backend = OpenAIGPTPlannerBackend(
        config=OpenAIPlannerConfig(api_key="test-key", model="gpt-5.4", reasoning_effort="medium", timeout_seconds=30.0, retry_count=1, max_output_tokens=2048),
        client=client,
    )

    candidate = backend.build_candidate(_door_context())

    assert candidate.backend_name == "gpt"
    assert candidate.model_name == "gpt-5.4"
    assert candidate.plan.initial_state == "S0_APPROACH"
    assert client.responses.calls
    request_payload = client.responses.calls[0]
    assert request_payload["model"] == "gpt-5.4"
    assert request_payload["reasoning"]["effort"] == "medium"
    assert request_payload["text"]["format"]["type"] == "json_schema"
    assert request_payload["text"]["format"]["strict"] is True


def test_planner_from_mode_returns_gpt_backend():
    planner = planner_from_mode("gpt")
    assert isinstance(planner, DeterministicPlanner)
    assert planner.backend.backend_name == "gpt"


def test_gpt_eval_run_saves_replayable_artifacts(tmp_path):
    client = FakeOpenAIClient(_valid_door_payload())
    planner = DeterministicPlanner(
        backend=OpenAIGPTPlannerBackend(
            config=OpenAIPlannerConfig(api_key="test-key", model="gpt-5.4", reasoning_effort="medium", timeout_seconds=30.0, retry_count=1, max_output_tokens=2048),
            client=client,
        )
    )
    harness = EvaluationHarness(planner=planner)
    output_dir = tmp_path / "eval_outputs"

    harness.run_batch(EvalRequest(tasks=["door"], runs_per_task=1, output_dir=str(output_dir)))

    run_dir = output_dir / "gpt" / "door" / "run_000"
    assert (run_dir / "request.json").exists()
    assert (run_dir / "prompt.json").exists()
    assert (run_dir / "raw_response.json").exists()
    assert (run_dir / "candidate_plan.json").exists()
    assert (run_dir / "final_plan.json").exists()
    assert (run_dir / "validation.json").exists()
    assert (run_dir / "semantic_sanity.json").exists()
    assert (run_dir / "trace.json").exists()
    assert (run_dir / "summary.json").exists()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["backend_name"] == "gpt"
    assert summary["model_name"] == "gpt-5.4"
    assert summary["planner_success"] is True
