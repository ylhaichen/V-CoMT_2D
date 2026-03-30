# Planner

## Modules

- `base.py`
  - top-level planner interface
- `models.py`
  - request, response, and reasoning models
- `backends.py`
  - candidate plan generation backends
- `pipeline.py`
  - orchestration for generation, validation, semantic checks, and repair
- `intent_parser.py`
  - keyword-based task-family inference
- `scene_interpreter.py`
  - extraction of task-relevant facts from `WorldState`
- `heuristics.py`
  - reusable geometry-aware heuristics
- `role_assignment.py`
  - deterministic role selection
- `fsm_builder.py`
  - schema-safe state/action/transition helpers
- `repair.py`
  - bounded structural repair
- `semantic_checks.py`
  - semantic sanity checks and deterministic resynthesis
- `templates/`
  - per-task deterministic `FSM` synthesis

## Pipeline

1. parse instruction into a supported task family
2. interpret the structured 2D scene
3. assign collaborative roles using deterministic heuristics
4. build a `PlanningContext`
5. ask the planner backend for a candidate plan
6. run structural validation
7. run structural repair if needed
8. run semantic sanity checks
9. run semantic repair / deterministic resynthesis if needed
10. return `PlanningResult`

## Backends

Current backends:

- `DeterministicTemplateBackend`
  - trusted symbolic baseline used for normal planning
- `StubLLMPlannerBackend`
  - future insertion point for `LLM/VLM` candidate generation

The backend boundary exists so future model-generated candidates can enter the same sanitize / validate / execute stack without changing downstream components.

## Semantic Sanity Layer

`semantic_checks.py` rejects plans that are structurally valid but strategically poor.

Examples:

- `T1`
  - invalid holder choice
  - broken wait / pass order
  - incorrect completion semantics
- `T2`
  - missing setup phase
  - implausible blocker geometry
  - incorrect object-goal completion condition
- `T4`
  - duplicated search sectors
  - missing report / converge coordination
  - incorrect rendezvous semantics
- `T6`
  - implausible handoff region
  - non-relay pseudo-strategy
  - incorrect payload-goal completion condition

Returned result includes:

- `passed`
- `warnings`
- `errors`
- `suggested_repairs`
- `debug_info`

## Repair Strategy

Two repair layers exist:

- structural repair
  - local `FSM` fixes such as timeout coverage, missing terminal states, broken transition targets, and action completion
- semantic repair
  - deterministic resynthesis from trusted scene facts, heuristics, and a symbolic backend

This split keeps the system compatible with future `LLM/VLM` backends while preserving a stable execution contract.

## Adding A New Task

1. extend `TaskType`
2. add instruction patterns in `intent_parser.py`
3. add scene-fact extraction in `scene_interpreter.py`
4. add heuristics in `heuristics.py`
5. add role logic in `role_assignment.py`
6. add a template under `planner/templates/`
7. add semantic checks and repair coverage
8. add fixtures, demos, tests, and evaluation coverage
