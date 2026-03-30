import pytest

from vcomt2d.planner.intent_parser import parse_intent
from vcomt2d.planner.models import TaskType


@pytest.mark.parametrize(
    ("instruction", "expected_task"),
    [
        ("Both of you get into the next room", TaskType.T1_DOOR_WEDGE_PASS_THROUGH),
        ("Push the ball into the corner", TaskType.T2_HERDING_CORRALLING),
        ("Find the red box and meet there", TaskType.T4_COLLABORATIVE_SEARCH_CONVERGE),
        ("Move the box to the far goal", TaskType.T6_RELAY_DELIVERY),
    ],
)
def test_parse_intent_supported_prompts(instruction, expected_task):
    result = parse_intent(instruction)
    assert result.task_type == expected_task
    assert result.confidence > 0.0


def test_parse_intent_unsupported_prompt():
    result = parse_intent("Please sing a song")
    assert result.task_type is None
    assert result.failure_reason == "unsupported_instruction"

