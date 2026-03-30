# Planner

## Modules

- `base.py`
  - top-level planner interface
- `models.py`
  - request, response, and reasoning models
- `backends.py`
  - candidate plan generation backends
- `openai_backend.py`
  - real OpenAI GPT backend using the `Responses API`
- `openai_config.py`
  - environment/config loading for live GPT planning
- `qwen_vl_backend.py`
  - local `Qwen2.5-VL-3B-Instruct` backend for 8 GB VRAM-oriented experiments
- `qwen_vl_config.py`
  - local `Qwen` config loading from environment variables and `PlanningConfig`
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
10. if a model backend still cannot provide a usable plan, fall back to deterministic template resynthesis with artifacts preserved
11. return `PlanningResult`

## Backends

Current backends:

- `DeterministicTemplateBackend`
  - trusted symbolic baseline used for normal planning
- `OpenAIGPTPlannerBackend`
  - real GPT-based candidate generator using OpenAI `Responses API`
  - requests `Structured Outputs` with a schema-constrained `FSM` response
- `QwenVLPlannerBackend`
  - local open-source `VLM` backend using `Qwen2.5-VL-3B-Instruct`
  - targets 8 GB VRAM by default through `4-bit` loading when the local runtime is available
  - can optionally attach a rendered planner input scene image
- `StubLLMPlannerBackend`
  - future insertion point for `LLM/VLM` candidate generation

The backend boundary exists so model-generated candidates can enter the same sanitize / validate / execute stack without changing downstream components.

## GPT Backend Notes

The GPT backend:

- uses model `gpt-5.4` by default
- loads credentials from `OPENAI_API_KEY`
- supports model override through config or CLI
- records prompt payloads, raw responses, parsed candidate plans, and repair outputs for replay

The planner still does not trust raw model output. The validator, semantic sanity checks, and repair logic remain the source of truth.

## Qwen Backend Notes

The local `Qwen` backend:

- uses `Qwen/Qwen2.5-VL-3B-Instruct` by default
- is intended as the primary local open-source `VLM` backend
- is configured for practical 8 GB VRAM operation via `bitsandbytes` `4-bit` loading by default
- can operate on `instruction + structured world state` alone
- can optionally add a rendered scene image when `QWEN_VL_USE_SCENE_IMAGE=true` or `--scene-image` is used
- supports reproducible offline runs via `QWEN_VL_LOCAL_FILES_ONLY=1`, which resolves the cached Hugging Face `snapshot` path directly

The runtime path is:

1. build a shared planner prompt and structured payload
2. optionally render a single scene snapshot
3. run local generation through `transformers`
4. parse the candidate JSON
5. run the normal validation / semantic sanity / repair pipeline
6. if the candidate still cannot be used, resynthesize from the deterministic template backend while preserving the original `Qwen` artifacts

This keeps the local `VLM` backend on the same contract as the deterministic and GPT backends.

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
