"""Shared status constants."""

from __future__ import annotations

from enum import Enum


class PlannerMode(str, Enum):
    DETERMINISTIC = "deterministic"
    SCRIPTED = "scripted"
    REPAIR = "repair"


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    VALIDATION_FAILED = "validation_failed"
    PLANNING_FAILED = "planning_failed"

