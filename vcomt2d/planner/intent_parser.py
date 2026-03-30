"""Deterministic intent parser for supported task families."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .models import IntentParseResult, TaskType


TASK_KEYWORDS: Dict[TaskType, List[str]] = {
    TaskType.T1_DOOR_WEDGE_PASS_THROUGH: ["door", "next room", "through the door", "get through", "pass through", "together"],
    TaskType.T2_HERDING_CORRALLING: ["ball", "target area", "corner", "corral", "push the ball"],
    TaskType.T4_COLLABORATIVE_SEARCH_CONVERGE: ["find", "search", "red box", "target", "meet there"],
    TaskType.T6_RELAY_DELIVERY: ["deliver", "far corner", "far goal", "move the box", "relay", "object"],
}


def parse_intent(instruction: str) -> IntentParseResult:
    normalized = " ".join(instruction.lower().strip().split())
    scores: List[Tuple[TaskType, int, List[str]]] = []
    for task_type, keywords in TASK_KEYWORDS.items():
        matches = [keyword for keyword in keywords if keyword in normalized]
        scores.append((task_type, len(matches), matches))
    scores.sort(key=lambda item: (-item[1], item[0].value))

    best_task, score, matches = scores[0]
    if score == 0:
        return IntentParseResult(task_type=None, confidence=0.0, failure_reason="unsupported_instruction")

    inferred_entities: Dict[str, str] = {}
    if "red box" in normalized:
        inferred_entities["target_object_hint"] = "red box"
    if "ball" in normalized:
        inferred_entities["target_object_hint"] = "ball"
    if "door" in normalized or "room" in normalized:
        inferred_entities["door_required"] = "true"

    return IntentParseResult(task_type=best_task, confidence=min(1.0, 0.55 + 0.15 * score), matched_keywords=matches, inferred_entities=inferred_entities)

