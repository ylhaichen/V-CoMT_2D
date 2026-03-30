# FSM Schema

`FSMPlan`:

- `task_description`
- `reasoning_summary`
- `fsm.initial_state`
- `fsm.states[]`

`StateSpec`:

- `state_id`
- `robot_a`
- `robot_b`
- `transitions`
- `terminal`
- `status`

`TransitionSpec`:

- `to`
- `condition.kind`
- `condition.args`
- `notes`

Supported `condition.kind`:

- `timeout`
- `all_actions_done`
- `robot_in_region`
- `both_in_region`
- `object_in_region`
- `signal_received`
- `object_found`
- `handoff_ready`
- `flag_true`

