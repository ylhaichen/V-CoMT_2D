# V-CoMT_2D

V-CoMT_2D is a lightweight 2D research prototype for validating the end-to-end planning pipeline of a hierarchical multi-robot collaboration system.

The project is intentionally planner-first. It focuses on:

- explicit `FSM` contracts
- deterministic and reproducible planning
- structural validation and semantic sanity checks
- repair and retry flows
- execution traces and animation artifacts
- benchmark-style batch evaluation

It is not intended to be a high-fidelity simulator.

## What The Repository Currently Supports

Current pipeline:

`instruction -> intent parsing -> scene interpretation -> role assignment -> planner backend -> FSM synthesis -> structural validation -> semantic sanity checks -> repair / resynthesis -> execution -> trace -> animation / evaluation artifacts`

Implemented components:

- stable planner API through `DeterministicPlanner.plan(request)`
- structured `PlanningRequest` / `PlanningResult`
- JSON-serializable multi-robot `FSMPlan`
- structural `FSM` validator and bounded repair layer
- task-aware semantic sanity checker
- deterministic execution engine for the shared skill vocabulary
- `MP4` export through `ffmpeg`, with explicit `GIF` fallback
- evaluation harness for batch runs and artifact collection

## Supported Tasks

### T1 Door Wedge & Pass-Through

Example prompts:

- `Both of you get into the next room`
- `Get through the door together`

### T2 Herding / Corralling

Example prompts:

- `Push the ball into the corner`
- `Get the ball into that target area`

### T4 Collaborative Search & Converge

Example prompts:

- `Find the red box`
- `Search for the target and meet there`

### T6 Relay Delivery

Example prompts:

- `Move the box to the far goal`
- `Get that object to the far corner`

## Repository Structure

```text
V-CoMT_2D/
├── README.md
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── configs/
│   └── sim2d/
├── docs/
│   ├── architecture.md
│   ├── evaluation.md
│   ├── fsm_schema.md
│   ├── planner.md
│   └── tasks.md
├── outputs/
│   ├── animations/
│   └── eval/
├── scripts/
│   ├── plan_door_demo.py
│   ├── plan_herding_demo.py
│   ├── plan_search_demo.py
│   ├── plan_relay_demo.py
│   ├── animate_door_demo.py
│   ├── animate_herding_demo.py
│   ├── animate_search_demo.py
│   ├── animate_relay_demo.py
│   └── run_eval.py
├── tests/
└── vcomt2d/
    ├── core/
    ├── eval/
    ├── fsm/
    ├── planner/
    │   ├── backends.py
    │   ├── pipeline.py
    │   ├── heuristics.py
    │   └── templates/
    ├── sim/
    │   └── tasks/
    └── viz/
```

## Planner Architecture

The planner contract stays stable while the candidate generation backend can change.

Main planner layers:

- `vcomt2d/planner/base.py`
  - top-level planner interface
- `vcomt2d/planner/backends.py`
  - `DeterministicTemplateBackend`
  - `StubLLMPlannerBackend`
- `vcomt2d/planner/pipeline.py`
  - orchestration for candidate generation, validation, semantic checks, repair, and result packaging
- `vcomt2d/planner/intent_parser.py`
  - maps vague instructions to supported task families
- `vcomt2d/planner/scene_interpreter.py`
  - extracts scene facts from structured `WorldState`
- `vcomt2d/planner/heuristics.py`
  - reusable geometry-aware heuristics for roles, relay handoff selection, and task realism
- `vcomt2d/planner/role_assignment.py`
  - deterministic role assignment built on task heuristics
- `vcomt2d/planner/templates/`
  - per-task deterministic `FSM` synthesis
- `vcomt2d/planner/repair.py`
  - structural `FSM` repair
- `vcomt2d/planner/semantic_checks.py`
  - semantic sanity validation and deterministic resynthesis

### Backend Flow

The backend architecture is ready for future `LLM/VLM` integration:

1. `instruction + world_state` are converted into a `PlanningContext`
2. a planner backend produces a candidate `FSMPlan`
3. structural validation runs
4. structural repair runs if needed
5. semantic sanity checks run
6. semantic repair / resynthesis runs if needed
7. final `PlanningResult` is returned

Today the default backend is deterministic. A stub `LLM/VLM` backend is included as the insertion point for future model-generated candidate plans.

## Evaluation Harness

The repository includes a batch evaluation harness for planner/executor benchmarking.

Main entrypoint:

```bash
python3 scripts/run_eval.py --tasks door herding search relay --runs 1
```

Useful evaluation switches:

- `--planner-mode`
  - select `deterministic`, `llm_stub`, or `vlm_stub`
- `--animation`
  - save per-run animation artifacts
- `--output-dir`
  - override the artifact root directory
- `--no-logs`
  - skip `logs.txt`

What each run records:

- planner success
- structural validation result
- semantic sanity result
- repair and retry count
- execution success
- failure reason
- execution step count
- per-run artifacts

Artifact layout:

```text
outputs/
  eval/
    <task_name>/
      run_000/
        plan.json
        trace.json
        summary.json
        logs.txt
        animation.mp4
    batch_summary.json
```

## Environment Setup

### Prerequisites

Recommended baseline:

- Python `3.10`
- `ffmpeg` for `MP4` export
- Linux, macOS, or Windows

### Option A: Conda / Micromamba

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate vcomt2d
```

Or with `micromamba`:

```bash
micromamba create -f environment.yml
micromamba activate vcomt2d
```

### Option B: Python venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Install ffmpeg

`MP4` export requires `ffmpeg` to be available on `PATH`, or provided via `VCOMT2D_FFMPEG_BINARY`.

Ubuntu / Debian:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

macOS:

```bash
brew install ffmpeg
```

Conda / Micromamba:

```bash
conda install -c conda-forge ffmpeg
```

Verify:

```bash
ffmpeg -version
```

If `ffmpeg` is installed in a non-standard location:

```bash
export VCOMT2D_FFMPEG_BINARY=/absolute/path/to/ffmpeg
```

## Running Planning Demos

```bash
python3 scripts/plan_door_demo.py
python3 scripts/plan_herding_demo.py
python3 scripts/plan_search_demo.py
python3 scripts/plan_relay_demo.py
```

Each planning demo prints:

- reasoning summary
- structural validation result
- semantic validation result
- final `FSM` as JSON

## Running Animation Demos

```bash
python3 scripts/animate_door_demo.py
python3 scripts/animate_herding_demo.py
python3 scripts/animate_search_demo.py
python3 scripts/animate_relay_demo.py
```

If `ffmpeg` is available, the animation is saved as `MP4` under `outputs/animations/`.

If `ffmpeg` is not available, the exporter prints a clear message and saves a `GIF` fallback instead.

## Running Evaluation

Run all tasks once:

```bash
python3 scripts/run_eval.py --runs 1
```

Run only two tasks:

```bash
python3 scripts/run_eval.py --tasks door relay --runs 3
```

Run the future-backend entrypoint with deterministic fallback:

```bash
python3 scripts/run_eval.py --tasks relay --runs 1 --planner-mode llm_stub
```

Enable animation artifacts during evaluation:

```bash
python3 scripts/run_eval.py --tasks search relay --runs 1 --animation
```

Write artifacts to a custom directory:

```bash
python3 scripts/run_eval.py --output-dir /tmp/vcomt_eval --runs 1
```

## Running Tests

Run the full test suite:

```bash
python3 -m pytest -q
```

Important test groups include:

- parser and scene interpretation tests
- role assignment tests
- structural validator tests
- semantic sanity tests
- repair regression tests
- planner task tests
- animation export tests
- evaluation harness tests
- backend / future integration tests

## Animation And MP4 Export

Animation export is implemented in `vcomt2d/viz/export.py`.

Behavior:

- `save_animation_mp4(...)` requires `ffmpeg`
- `save_animation_with_fallback(...)` prefers `MP4`
- if `ffmpeg` is unavailable and fallback is allowed, it writes `GIF`
- exported files are returned as structured `ExportArtifact` metadata

This keeps demo and evaluation workflows deterministic and scriptable.

## Current Limitations

- planner intent parsing is still rule-based
- the `LLM/VLM` backend is a stub, not a real model integration
- physics, collision, and perception are intentionally simplified
- execution semantics are lightweight and designed for pipeline validation, not realism benchmarking
- only four collaborative task families are currently supported

## Future Work

- replace the stub backend with a real `LLM/VLM` planner adapter
- add richer semantic repair strategies beyond deterministic resynthesis
- expand the evaluation harness with larger fixture banks and aggregate metrics
- increase task realism with better topology reasoning and stronger failure semantics
- connect the planner contract to a 3D simulator or robotics middleware layer

## Additional Docs

- `docs/architecture.md`
- `docs/planner.md`
- `docs/evaluation.md`
- `docs/tasks.md`
- `docs/fsm_schema.md`
