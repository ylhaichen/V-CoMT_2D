# Architecture

`V-CoMT_2D` 当前以 `planner-first` 组织：

- `vcomt2d/planner/`: deterministic symbolic planner, intent parsing, scene interpretation, role assignment, FSM synthesis, repair
- `vcomt2d/fsm/`: schema, validation, execution
- `vcomt2d/sim/`: structured 2D `WorldState`, runtime `SimWorld`, task fixtures
- `vcomt2d/viz/`: matplotlib renderer, animation builder, MP4/GIF export

数据流：

`instruction + world_state -> planner.plan() -> FSMPlan -> validator -> executor -> trace -> animation export`

