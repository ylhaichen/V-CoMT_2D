# Architecture

`V-CoMT_2D` is organized around a planner-first contract.

Primary data flow:

`instruction + structured world state -> planner pipeline -> FSM plan -> executor -> trace -> animation / evaluation artifacts`

Main packages:

- `vcomt2d/planner/`
  - planner backends, including deterministic and real OpenAI GPT candidate generation, symbolic heuristics, scene interpretation, role assignment, `FSM` synthesis, validation, semantic repair
- `vcomt2d/fsm/`
  - schema, validation, and execution
- `vcomt2d/sim/`
  - structured `WorldState`, deterministic runtime `SimWorld`, and task fixtures
- `vcomt2d/eval/`
  - batch evaluation harness and artifact writers
- `vcomt2d/viz/`
  - rendering, animation assembly, and export helpers

Design intent:

- keep downstream execution contract stable
- isolate candidate plan generation behind a backend boundary
- keep semantic checking and repair independent from any future model backend
- prioritize reproducibility, observability, and evaluation artifacts over visual fidelity
