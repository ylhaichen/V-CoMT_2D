from vcomt2d.planner.structured_output import parse_candidate_plan_text


def test_parse_candidate_plan_text_recovers_truncated_fenced_json():
    text = """```json
{
  "task_description": "Door task",
  "reasoning_summary": "holder and passer",
  "fsm": {
    "initial_state": "S0",
    "states": [
      {
        "state_id": "S0",
        "robot_a": {"skill": "MOVE_TO", "params": {"target_id": "door_main"}},
        "robot_b": {"skill": "HOLD_POSITION", "params": {"ticks": 1}},
        "transitions": [
          {"to": "S_DONE", "condition": {"kind": "all_actions_done", "args": {}}, "notes": "done"},
          {"to": "S_FAIL", "condition": {"kind": "timeout", "args": {"ticks": 8}}, "notes": "timeout"}
        ],
        "terminal": false,
        "status": null,
        "notes": ""
      },
      {"state_id": "S_DONE", "robot_a": null, "robot_b": null, "transitions": [], "terminal": true, "status": "success", "notes": ""},
      {"state_id": "S_FAIL", "robot_a": null, "robot_b": null, "transitions": [], "terminal": true, "status": "failure", "notes": ""}
    ]
  }
```"""

    parsed = parse_candidate_plan_text(text)

    assert parsed["task_description"] == "Door task"
    assert parsed["fsm"]["initial_state"] == "S0"
    assert len(parsed["fsm"]["states"]) == 3
