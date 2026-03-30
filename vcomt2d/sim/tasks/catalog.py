"""Canonical task catalog shared by demos and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from .fixtures import door_task_world, herding_task_world, relay_task_world, search_task_world


@dataclass(frozen=True)
class TaskScenario:
    task_name: str
    instruction: str
    fixture_builder: Callable


TASK_SCENARIOS: Dict[str, TaskScenario] = {
    "door": TaskScenario("T1_Door_Wedge_Pass_Through", "Both of you get into the next room", door_task_world),
    "herding": TaskScenario("T2_Herding_Corralling", "Push the ball into the corner", herding_task_world),
    "search": TaskScenario("T4_Collaborative_Search_Converge", "Find the red box and meet there", search_task_world),
    "relay": TaskScenario("T6_Relay_Delivery", "Move the box to the far goal", relay_task_world),
}

