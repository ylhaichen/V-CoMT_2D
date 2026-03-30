"""Top-level deterministic planner orchestration."""

from __future__ import annotations

from typing import Callable, Dict

from .base import Planner
from .intent_parser import parse_intent
from .models import PlanningRequest, PlanningResult, TaskType
from .repair import PlanRepairer
from .role_assignment import assign_roles
from .scene_interpreter import SceneInterpretationError, interpret_scene
from .semantic_checks import SemanticRepairer, SemanticSanityChecker
from .templates.door_wedge import build_door_plan
from .templates.herding import build_herding_plan
from .templates.relay_delivery import build_relay_plan
from .templates.search_converge import build_search_plan
from .validator_adapter import validate_candidate, validation_to_debug


TEMPLATE_BUILDERS: Dict[TaskType, Callable] = {
    TaskType.T1_DOOR_WEDGE_PASS_THROUGH: build_door_plan,
    TaskType.T2_HERDING_CORRALLING: build_herding_plan,
    TaskType.T4_COLLABORATIVE_SEARCH_CONVERGE: build_search_plan,
    TaskType.T6_RELAY_DELIVERY: build_relay_plan,
}


class DeterministicPlanner(Planner):
    """Deterministic symbolic planner with validation and repair."""

    def __init__(self):
        self.semantic_checker = SemanticSanityChecker()
        self.semantic_repairer = SemanticRepairer()

    def plan(self, request: PlanningRequest) -> PlanningResult:
        intent = parse_intent(request.user_instruction)
        if intent.task_type is None:
            return PlanningResult(
                success=False,
                task_type=None,
                reasoning_summary="Instruction could not be mapped to a supported task family.",
                plan=None,
                validation={"valid": False, "errors": []},
                semantic_validation={"passed": False, "warnings": [], "errors": [], "suggested_repairs": [], "debug_info": {}},
                retries_used=0,
                failure_reason=intent.failure_reason or "unsupported_instruction",
                debug_info={"intent": intent.to_dict()},
            )

        try:
            scene_facts = interpret_scene(intent.task_type, request.world_state)
        except SceneInterpretationError as exc:
            return PlanningResult(
                success=False,
                task_type=intent.task_type.value,
                reasoning_summary="Planner identified the task family but could not extract the required scene facts.",
                plan=None,
                validation={"valid": False, "errors": []},
                semantic_validation={"passed": False, "warnings": [], "errors": [], "suggested_repairs": [], "debug_info": {}},
                retries_used=0,
                failure_reason=str(exc),
                debug_info={"intent": intent.to_dict()},
            )

        roles = assign_roles(scene_facts, request.world_state)
        reasoning_summary = self._build_reasoning_summary(intent, scene_facts, roles)
        builder = TEMPLATE_BUILDERS[intent.task_type]
        candidate = builder(request.user_instruction, reasoning_summary, scene_facts, roles, request.planning_config)
        validation = validate_candidate(candidate)
        repairer = PlanRepairer(request.planning_config)
        retries = 0
        applied_repairs = []
        semantic_result = {"passed": False, "warnings": [], "errors": [], "suggested_repairs": [], "debug_info": {}}

        while not validation.valid and request.planning_config.allow_repair and retries < request.planning_config.max_retries:
            repair_result = repairer.repair(candidate, validation)
            candidate = repair_result.plan
            applied_repairs.extend(repair_result.applied_repairs)
            validation = validate_candidate(candidate)
            retries += 1

        if not validation.valid:
            return PlanningResult(
                success=False,
                task_type=intent.task_type.value,
                reasoning_summary=reasoning_summary,
                plan=None if request.planning_config.strict_validation else candidate,
                validation=validation_to_debug(validation),
                semantic_validation=semantic_result,
                retries_used=retries,
                failure_reason="plan_validation_failed",
                debug_info={
                    "intent": intent.to_dict(),
                    "scene_facts": scene_facts.to_dict(),
                    "roles": roles.to_dict(),
                    "applied_repairs": applied_repairs,
                },
            )

        if request.planning_config.enable_semantic_checks:
            semantic_validation = self.semantic_checker.check(intent.task_type, candidate, request.world_state, scene_facts, roles)
            semantic_result = semantic_validation.to_dict()
            semantic_retries = 0
            while not semantic_validation.passed and request.planning_config.allow_repair and semantic_retries < request.planning_config.max_retries:
                semantic_repair = self.semantic_repairer.repair(request, candidate, semantic_validation, scene_facts, roles, builder)
                if not semantic_repair.applied_repairs:
                    break
                candidate = semantic_repair.plan
                roles = semantic_repair.roles
                scene_facts = semantic_repair.scene_facts
                applied_repairs.extend(semantic_repair.applied_repairs)
                reasoning_summary = self._build_reasoning_summary(intent, scene_facts, roles)
                candidate.reasoning_summary = reasoning_summary
                validation = validate_candidate(candidate)
                if not validation.valid:
                    while not validation.valid and request.planning_config.allow_repair and retries < request.planning_config.max_retries:
                        repair_result = repairer.repair(candidate, validation)
                        candidate = repair_result.plan
                        applied_repairs.extend(repair_result.applied_repairs)
                        validation = validate_candidate(candidate)
                        retries += 1
                    if not validation.valid:
                        return PlanningResult(
                            success=False,
                            task_type=intent.task_type.value,
                            reasoning_summary=reasoning_summary,
                            plan=None if request.planning_config.strict_validation else candidate,
                            validation=validation_to_debug(validation),
                            semantic_validation=semantic_result,
                            retries_used=retries + semantic_retries,
                            failure_reason="plan_validation_failed_after_semantic_repair",
                            debug_info={
                                "intent": intent.to_dict(),
                                "scene_facts": scene_facts.to_dict(),
                                "roles": roles.to_dict(),
                                "applied_repairs": applied_repairs,
                            },
                        )
                semantic_validation = self.semantic_checker.check(intent.task_type, candidate, request.world_state, scene_facts, roles)
                semantic_result = semantic_validation.to_dict()
                semantic_retries += 1
                retries += 1

            if not semantic_validation.passed:
                return PlanningResult(
                    success=False,
                    task_type=intent.task_type.value,
                    reasoning_summary=reasoning_summary,
                    plan=None if request.planning_config.strict_validation else candidate,
                    validation=validation_to_debug(validation),
                    semantic_validation=semantic_result,
                    retries_used=retries,
                    failure_reason="semantic_sanity_failed",
                    debug_info={
                        "intent": intent.to_dict(),
                        "scene_facts": scene_facts.to_dict(),
                        "roles": roles.to_dict(),
                        "applied_repairs": applied_repairs,
                    },
                )

        return PlanningResult(
            success=True,
            task_type=intent.task_type.value,
            reasoning_summary=reasoning_summary,
            plan=candidate,
            validation=validation_to_debug(validation),
            semantic_validation=semantic_result,
            retries_used=retries,
            failure_reason=None,
            debug_info={
                "intent": intent.to_dict(),
                "scene_facts": scene_facts.to_dict(),
                "roles": roles.to_dict(),
                "applied_repairs": applied_repairs,
            },
        )

    def _build_reasoning_summary(self, intent, scene_facts, roles) -> str:
        return (
            f"Inferred task={intent.task_type.value} from keywords={intent.matched_keywords}. "
            f"Scene facts: door={scene_facts.relevant_door_id}, object={scene_facts.target_object_id}, "
            f"goal={scene_facts.goal_region_id}, search_regions={scene_facts.search_region_ids}, handoff={scene_facts.handoff_region_id}. "
            f"Role assignment: {roles.roles}. Rationale: {roles.rationale}"
        )
