# V-CoMT_2D

V-CoMT_2D is a lightweight 2D prototype for validating the end-to-end planning pipeline of a hierarchical multi-robot collaboration system.

It is intentionally optimized for:

- deterministic planning behavior
- explicit FSM contracts
- validation and repair
- debugging and observability
- quick iteration before a future 3D / VLM-backed integration

It is not intended to be a high-fidelity simulator.

## Overview

The current milestone focuses on a robust planner-facing stack:

`instruction -> intent parsing -> scene interpretation -> role assignment -> FSM synthesis -> structural validation -> semantic sanity checks -> repair/retry -> execution -> animation export`

The repository already includes:

- a deterministic planner contract
- a JSON-serializable multi-robot FSM schema
- a structural FSM validator
- a task-aware semantic sanity checker
- bounded structural and semantic repair flows
- a lightweight deterministic executor
- planning demo scripts
- animation demo scripts
- MP4 export when `ffmpeg` is available
- explicit GIF fallback when `ffmpeg` is not available

## Features

- Deterministic symbolic planner with a stable API
- Structured `PlanningRequest` / `PlanningResult`
- Scene-aware role assignment for `robot_a` and `robot_b`
- Multi-state FSM synthesis for supported collaborative tasks
- Structural FSM validation
- Semantic planner sanity checks
- Explicit repair/retry pipeline
- Deterministic execution trace generation
- Matplotlib-based 2D animation rendering
- MP4 export support through `ffmpeg`
- Unit, integration, regression, and export tests

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
├── .gitignore
├── configs/
│   └── sim2d/
│       └── planner_config.json
├── docs/
│   ├── architecture.md
│   ├── planner.md
│   ├── tasks.md
│   └── fsm_schema.md
├── outputs/
│   └── animations/
├── scripts/
│   ├── _demo_common.py
│   ├── plan_door_demo.py
│   ├── plan_herding_demo.py
│   ├── plan_search_demo.py
│   ├── plan_relay_demo.py
│   ├── animate_door_demo.py
│   ├── animate_herding_demo.py
│   ├── animate_search_demo.py
│   └── animate_relay_demo.py
├── tests/
│   ├── test_animation_export.py
│   ├── test_intent_parser.py
│   ├── test_planner_door.py
│   ├── test_planner_herding.py
│   ├── test_planner_relay.py
│   ├── test_planner_search.py
│   ├── test_regression.py
│   ├── test_repair.py
│   ├── test_role_assignment.py
│   ├── test_scene_interpreter.py
│   ├── test_semantic_sanity.py
│   └── test_validator_integration.py
└── vcomt2d/
    ├── core/
    ├── fsm/
    ├── planner/
    │   └── templates/
    ├── sim/
    │   └── tasks/
    └── viz/
```

## Planner Architecture

The planner is modular and intended to remain the long-term contract between a future VLM/LLM planner and the downstream executor.

Key modules:

- `vcomt2d/planner/intent_parser.py`
  - maps natural-language instructions to supported task families
- `vcomt2d/planner/scene_interpreter.py`
  - extracts task-relevant facts from a structured 2D `WorldState`
- `vcomt2d/planner/role_assignment.py`
  - assigns deterministic collaborative roles based on geometry
- `vcomt2d/planner/templates/`
  - one deterministic FSM synthesis module per task family
- `vcomt2d/planner/repair.py`
  - structural repair logic
- `vcomt2d/planner/semantic_checks.py`
  - task-aware semantic sanity checks and semantic repair
- `vcomt2d/planner/main.py`
  - top-level orchestration entrypoint

Top-level API:

```python
from vcomt2d.planner.main import DeterministicPlanner
from vcomt2d.planner.models import PlanningRequest

planner = DeterministicPlanner()
result = planner.plan(request)
```

## Planner Input

The planner consumes a structured `PlanningRequest` containing:

- `request_id`
- `user_instruction`
- `world_state`
- `planning_config`
- `history`
- `retry_count`
- `planner_mode`

`world_state` is serializable and includes:

- robot states
- object states
- door states
- goal / region definitions
- obstacles
- topology links
- task-specific facts

## Planner Output

The planner returns a structured `PlanningResult` containing:

- `success`
- `task_type`
- `reasoning_summary`
- `plan`
- `validation`
- `semantic_validation`
- `retries_used`
- `failure_reason`
- `debug_info`

The `plan` field is an `FSMPlan` with:

- `task_description`
- `reasoning_summary`
- `fsm.initial_state`
- `fsm.states`

Each non-terminal state contains:

- `robot_a` action
- `robot_b` action
- structured transitions
- timeout coverage

Terminal states explicitly encode:

- success
- failure

## Structural Validation

Structural validation checks:

- initial state exists
- state names are unique
- transition targets exist
- non-terminal states contain actions for both robots
- non-terminal states have transitions
- non-terminal states have timeout coverage
- valid terminal success/failure states exist
- skill names are valid
- action parameters are structurally valid
- the graph is reachable from the initial state
- orphan states are rejected

## Semantic Sanity Checks

Structural validity is not enough, so the planner also runs task-aware semantic sanity checks before returning success.

These checks catch plans that are structurally valid but strategically wrong.

Current semantic checks include:

- **T1 Door Wedge & Pass-Through**
  - valid wedgeable door
  - plausible holder selection
  - correct wait / pass / follow ordering
  - terminal condition tied to both robots reaching the target side
- **T2 Herding / Corralling**
  - movable target object exists
  - valid goal region exists
  - setup phase exists before pushing
  - pusher / blocker roles are geometrically plausible
  - terminal condition depends on object-in-goal
- **T4 Collaborative Search & Converge**
  - differentiated search partition
  - explicit found/report coordination
  - partner convergence after discovery
  - terminal condition reflects rendezvous semantics
- **T6 Relay Delivery**
  - valid payload, handoff region, and goal
  - plausible starter / finisher choice
  - actual relay structure, not single-robot pseudo-relay
  - terminal condition depends on payload reaching the goal

Semantic results are returned as a structured object with:

- `passed`
- `warnings`
- `errors`
- `suggested_repairs`
- `debug_info`

## Environment Setup

### Prerequisites

Recommended:

- Python `3.10`
- Linux, macOS, or Windows with a working Python environment
- `ffmpeg` if you want MP4 export

The repository has been tested with:

- Python `3.10.12`
- Ubuntu `22.04`

### Option A: Conda / Micromamba Setup

Create the environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate vcomt2d
```

If you use `micromamba`:

```bash
micromamba create -f environment.yml
micromamba activate vcomt2d
```

Verify the environment:

```bash
python --version
python -m pytest -q
```

### Option B: Python venv Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify the environment:

```bash
python --version
python -m pytest -q
```

## Installing ffmpeg

MP4 export requires `ffmpeg` to be available on `PATH`.

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

### Conda / Micromamba

If you prefer an environment-local installation:

```bash
conda install -c conda-forge ffmpeg
```

or

```bash
micromamba install -n vcomt2d -c conda-forge ffmpeg
```

### macOS

```bash
brew install ffmpeg
```

### Verify ffmpeg

```bash
ffmpeg -version
```

If this command fails, MP4 export will not work and the project will explicitly fall back to GIF export.

## Installing Project Dependencies

If you are not using `environment.yml`, install the Python dependencies manually:

```bash
pip install -r requirements.txt
```

## Running Planning Demos

From the repository root:

```bash
python3 scripts/plan_door_demo.py
python3 scripts/plan_herding_demo.py
python3 scripts/plan_search_demo.py
python3 scripts/plan_relay_demo.py
```

Each planning demo prints:

- planner reasoning summary
- structural validation result
- semantic validation result
- final FSM JSON

## Running Animation Demos

```bash
python3 scripts/animate_door_demo.py
python3 scripts/animate_herding_demo.py
python3 scripts/animate_search_demo.py
python3 scripts/animate_relay_demo.py
```

Generated outputs are saved under:

```text
outputs/animations/
```

## MP4 Export Behavior

The animation export logic prefers MP4 whenever `ffmpeg` is available.

Behavior:

- if `ffmpeg` is found on `PATH`:
  - animation demos save `.mp4`
- if `ffmpeg` is not found:
  - export raises a clear `AnimationExportError`
  - the demo wrapper uses the explicit GIF fallback path

Direct API:

```python
from vcomt2d.viz.export import save_animation_mp4

save_animation_mp4(trace, "outputs/animations/door_demo.mp4", fps=4)
```

Fallback-friendly API:

```python
from vcomt2d.viz.export import save_animation_with_fallback

artifact = save_animation_with_fallback(trace, "outputs/animations/door_demo", fps=4)
print(artifact.saved_path, artifact.format, artifact.message)
```

## Running Tests

Run the full test suite:

```bash
python3 -m pytest -q
```

Current test coverage includes:

- intent parsing
- scene interpretation
- role assignment
- planner generation for all supported tasks
- structural validation
- structural repair
- semantic sanity checks
- regression handling for unsupported or inconsistent inputs
- animation trace generation
- MP4 export behavior and fallback handling

## Current Limitations

- intent parsing is still rule-based
- semantic checks are deterministic heuristics, not learned reasoning
- the executor is a lightweight 2D runtime, not a full robotics simulator
- no realistic dynamics, sensing, or collision modeling
- search targets are available through structured world-state fixtures
- no external LLM/VLM integration yet

## Future Work

- replace deterministic instruction parsing with an LLM/VLM-backed planner frontend
- extend semantic sanity checks with richer world semantics
- connect the planner contract to a future 3D simulator
- add more task templates and richer topology handling
- strengthen execution-time semantic failure detection
- add CI-ready environment automation for MP4 verification

## Additional Documentation

- `docs/architecture.md`
- `docs/planner.md`
- `docs/tasks.md`
- `docs/fsm_schema.md`
