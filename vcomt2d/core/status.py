"""Shared status constants."""

from __future__ import annotations

from enum import Enum


class PlannerMode(str, Enum):
    DETERMINISTIC = "deterministic"
    GPT = "gpt"
    QWEN_VL = "qwen_vl"
    SCRIPTED = "scripted"
    REPAIR = "repair"
    LLM_STUB = "llm_stub"
    VLM_STUB = "vlm_stub"


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    VALIDATION_FAILED = "validation_failed"
    PLANNING_FAILED = "planning_failed"
