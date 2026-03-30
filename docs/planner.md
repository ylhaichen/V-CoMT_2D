# Planner

## Modules

- `base.py`: planner interface
- `models.py`: request / response / reasoning models
- `intent_parser.py`: keyword-based task-family inference
- `scene_interpreter.py`: extract task-relevant facts from `WorldState`
- `role_assignment.py`: deterministic role heuristics
- `fsm_builder.py`: schema-safe state/action/transition helpers
- `repair.py`: bounded structural repair
- `validator_adapter.py`: wraps FSM validator
- `templates/`: per-task FSM synthesis
- `main.py`: orchestration entrypoint

## Validation Flow

1. parse intent
2. interpret scene
3. assign roles
4. synthesize task template
5. validate FSM
6. apply structural repair if needed
7. run semantic sanity checks
8. if semantic issues are repairable, resynthesize deterministically from trusted scene facts / role heuristics
9. revalidate and return structured `PlanningResult`

## Semantic Sanity Layer

`semantic_checks.py` adds task-aware planner quality checks on top of structural validation.

Current checks cover:

- `T1`: valid wedgeable door, plausible holder, waiting/pass order, both robots on target side at completion
- `T2`: movable object, goal region, setup before push, plausible blocker geometry, object-at-goal terminal
- `T4`: differentiated search sectors, explicit found/report step, converger behavior, rendezvous terminal
- `T6`: plausible handoff region, starter/finisher plausibility, actual relay structure, object-at-goal terminal

Returned result includes:

- `passed`
- `warnings`
- `errors`
- `suggested_repairs`
- `debug_info`

## How To Add A New Task

1. extend `TaskType`
2. add keywords in `intent_parser.py`
3. add `SceneFacts` extraction logic
4. add deterministic role heuristic
5. add a template in `planner/templates/`
6. register the template in `planner/main.py`
