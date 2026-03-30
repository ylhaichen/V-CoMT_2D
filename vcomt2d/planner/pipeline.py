"""Planner orchestration pipeline with backend generation and sanitize layers."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional

from .backends import DeterministicTemplateBackend, PlannerBackend, PlannerBackendError
from .intent_parser import parse_intent
from .models import PlanningContext, PlanningRequest, PlanningResult
from .repair import PlanRepairer
from .role_assignment import assign_roles
from .scene_interpreter import SceneInterpretationError, interpret_scene
from .semantic_checks import SemanticRepairer, SemanticSanityChecker
from .validator_adapter import validate_candidate, validation_to_debug


class PlannerPipeline:
    """Deterministic planning pipeline around a pluggable candidate backend."""

    def __init__(
        self,
        generation_backend: PlannerBackend,
        repair_backend: Optional[PlannerBackend] = None,
        semantic_checker: Optional[SemanticSanityChecker] = None,
        semantic_repairer: Optional[SemanticRepairer] = None,
    ):
        self.generation_backend = generation_backend
        self.repair_backend = repair_backend or DeterministicTemplateBackend()
        self.semantic_checker = semantic_checker or SemanticSanityChecker()
        self.semantic_repairer = semantic_repairer or SemanticRepairer()

    def run(self, request: PlanningRequest) -> PlanningResult:
        intent = parse_intent(request.user_instruction)
        semantic_result = {"passed": True, "warnings": [], "errors": [], "suggested_repairs": [], "debug_info": {"skipped": True}}

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
        context = PlanningContext(
            request=request,
            intent=intent,
            scene_facts=scene_facts,
            roles=roles,
            reasoning_summary=self._build_reasoning_summary(intent.to_dict(), scene_facts.to_dict(), roles.to_dict()),
        )

        try:
            candidate_bundle = self.generation_backend.build_candidate(context)
        except PlannerBackendError as exc:
            return PlanningResult(
                success=False,
                task_type=intent.task_type.value,
                reasoning_summary=context.reasoning_summary,
                plan=None,
                validation={"valid": False, "errors": []},
                semantic_validation={"passed": False, "warnings": [], "errors": [], "suggested_repairs": [], "debug_info": {}},
                retries_used=0,
                failure_reason=str(exc),
                debug_info={"intent": intent.to_dict(), "scene_facts": scene_facts.to_dict(), "roles": roles.to_dict(), "backend": self.generation_backend.backend_name},
            )

        generation_candidate_bundle = candidate_bundle
        repair_backend_debug: list[Dict[str, object]] = []
        candidate = candidate_bundle.plan
        validation = validate_candidate(candidate)
        repairer = PlanRepairer(request.planning_config)
        retries = 0
        applied_repairs = []

        while not validation.valid and request.planning_config.allow_repair and retries < request.planning_config.max_retries:
            repair_result = repairer.repair(candidate, validation)
            candidate = repair_result.plan
            applied_repairs.extend(repair_result.applied_repairs)
            validation = validate_candidate(candidate)
            retries += 1

        if not validation.valid:
            return self._failure_result(
                request=request,
                context=context,
                validation=validation_to_debug(validation),
                semantic_result=semantic_result,
                retries_used=retries,
                failure_reason="plan_validation_failed",
                applied_repairs=applied_repairs,
                plan=None if request.planning_config.strict_validation else candidate,
                backend_debug=candidate_bundle.debug_info,
            )

        if request.planning_config.enable_semantic_checks:
            semantic_validation = self.semantic_checker.check(context.intent.task_type, candidate, request.world_state, context.scene_facts, context.roles)
            semantic_result = semantic_validation.to_dict()
            semantic_retries = 0

            while not semantic_validation.passed and request.planning_config.allow_repair and semantic_retries < request.planning_config.max_retries:
                repair = self.semantic_repairer.repair(request, context, candidate, semantic_validation, self.repair_backend)
                if not repair.applied_repairs:
                    break
                context = replace(
                    context,
                    scene_facts=repair.scene_facts,
                    roles=repair.roles,
                    reasoning_summary=self._build_reasoning_summary(intent.to_dict(), repair.scene_facts.to_dict(), repair.roles.to_dict()),
                )
                candidate_bundle = self.repair_backend.build_candidate(context)
                repair_backend_debug.append(
                    {
                        "backend": candidate_bundle.backend_name,
                        "model_name": candidate_bundle.model_name,
                        "debug_info": candidate_bundle.debug_info,
                    }
                )
                candidate = candidate_bundle.plan
                applied_repairs.extend(repair.applied_repairs)
                validation = validate_candidate(candidate)
                if not validation.valid:
                    while not validation.valid and request.planning_config.allow_repair and retries < request.planning_config.max_retries:
                        repair_result = repairer.repair(candidate, validation)
                        candidate = repair_result.plan
                        applied_repairs.extend(repair_result.applied_repairs)
                        validation = validate_candidate(candidate)
                        retries += 1
                    if not validation.valid:
                        return self._failure_result(
                            request=request,
                            context=context,
                            validation=validation_to_debug(validation),
                            semantic_result=semantic_result,
                            retries_used=retries + semantic_retries,
                            failure_reason="plan_validation_failed_after_semantic_repair",
                            applied_repairs=applied_repairs,
                            plan=None if request.planning_config.strict_validation else candidate,
                            backend_debug=candidate_bundle.debug_info,
                        )
                semantic_validation = self.semantic_checker.check(context.intent.task_type, candidate, request.world_state, context.scene_facts, context.roles)
                semantic_result = semantic_validation.to_dict()
                semantic_retries += 1
                retries += 1

            if not semantic_validation.passed:
                return self._failure_result(
                    request=request,
                    context=context,
                    validation=validation_to_debug(validation),
                    semantic_result=semantic_result,
                    retries_used=retries,
                    failure_reason="semantic_sanity_failed",
                    applied_repairs=applied_repairs,
                    plan=None if request.planning_config.strict_validation else candidate,
                    backend_debug=candidate_bundle.debug_info,
                )

        return PlanningResult(
            success=True,
            task_type=intent.task_type.value,
            reasoning_summary=context.reasoning_summary,
            plan=candidate,
            validation=validation_to_debug(validation),
            semantic_validation=semantic_result,
            retries_used=retries,
            failure_reason=None,
            debug_info={
                "intent": context.intent.to_dict(),
                "scene_facts": context.scene_facts.to_dict(),
                "roles": context.roles.to_dict(),
                "backend": generation_candidate_bundle.backend_name,
                "model_name": generation_candidate_bundle.model_name,
                "backend_debug": {
                    "generation": generation_candidate_bundle.debug_info,
                    "repair_candidates": repair_backend_debug,
                    "final_candidate_backend": candidate_bundle.backend_name,
                },
                "applied_repairs": applied_repairs,
            },
        )

    def _failure_result(
        self,
        request: PlanningRequest,
        context: PlanningContext,
        validation: Dict[str, object],
        semantic_result: Dict[str, object],
        retries_used: int,
        failure_reason: str,
        applied_repairs: list[str],
        backend_debug: Dict[str, object],
        plan,
    ) -> PlanningResult:
        return PlanningResult(
            success=False,
            task_type=context.intent.task_type.value,
            reasoning_summary=context.reasoning_summary,
            plan=plan,
            validation=validation,
            semantic_validation=semantic_result,
            retries_used=retries_used,
            failure_reason=failure_reason,
            debug_info={
                "intent": context.intent.to_dict(),
                "scene_facts": context.scene_facts.to_dict(),
                "roles": context.roles.to_dict(),
                "backend": self.generation_backend.backend_name,
                "model_name": backend_debug.get("model_name") if isinstance(backend_debug, dict) else None,
                "backend_debug": backend_debug,
                "applied_repairs": applied_repairs,
            },
        )

    def _build_reasoning_summary(self, intent_payload: Dict[str, object], scene_payload: Dict[str, object], roles_payload: Dict[str, object]) -> str:
        return (
            f"Inferred task={intent_payload['task_type']} from keywords={intent_payload.get('matched_keywords', [])}. "
            f"Scene facts: door={scene_payload.get('relevant_door_id')}, object={scene_payload.get('target_object_id')}, "
            f"goal={scene_payload.get('goal_region_id')}, search_regions={scene_payload.get('search_region_ids')}, handoff={scene_payload.get('handoff_region_id')}. "
            f"Role assignment: {roles_payload.get('roles')}. Rationale: {roles_payload.get('rationale')}"
        )
