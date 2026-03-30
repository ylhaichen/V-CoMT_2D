"""Shared skill vocabulary and helpers."""

from __future__ import annotations

from enum import Enum
from typing import Dict, Set


class SkillName(str, Enum):
    MOVE_TO = "MOVE_TO"
    PUSH = "PUSH"
    RETREAT = "RETREAT"
    HOLD_POSITION = "HOLD_POSITION"
    FOLLOW = "FOLLOW"
    SYNCHRONIZE = "SYNCHRONIZE"
    SIGNAL = "SIGNAL"
    WAIT_UNTIL = "WAIT_UNTIL"
    TRACK_OBJECT = "TRACK_OBJECT"


VALID_SKILLS: Set[str] = {skill.value for skill in SkillName}

SKILL_ALIASES: Dict[str, str] = {
    "move": SkillName.MOVE_TO.value,
    "move_to": SkillName.MOVE_TO.value,
    "goto": SkillName.MOVE_TO.value,
    "hold": SkillName.HOLD_POSITION.value,
    "wait": SkillName.WAIT_UNTIL.value,
    "sync": SkillName.SYNCHRONIZE.value,
    "track": SkillName.TRACK_OBJECT.value,
}


def normalize_skill_name(skill_name: str) -> str:
    canonical = skill_name.strip().upper()
    if canonical in VALID_SKILLS:
        return canonical
    lowered = skill_name.strip().lower()
    return SKILL_ALIASES.get(lowered, skill_name)

