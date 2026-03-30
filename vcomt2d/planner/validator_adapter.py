"""Planner-facing adapter around FSM validation."""

from __future__ import annotations

from typing import Dict

from vcomt2d.fsm.schema import FSMPlan
from vcomt2d.fsm.validator import ValidationResult, validate_fsm


def validate_candidate(plan: FSMPlan) -> ValidationResult:
    return validate_fsm(plan)


def validation_to_debug(validation: ValidationResult) -> Dict[str, object]:
    return validation.to_dict()

