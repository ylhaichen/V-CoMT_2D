from vcomt2d.planner.models import TaskType
from vcomt2d.planner.role_assignment import assign_roles
from vcomt2d.planner.scene_interpreter import interpret_scene
from vcomt2d.sim.tasks.fixtures import door_task_world, relay_task_world, search_task_world


def test_role_assignment_door_is_deterministic():
    world = door_task_world()
    facts = interpret_scene(TaskType.T1_DOOR_WEDGE_PASS_THROUGH, world)
    roles = assign_roles(facts, world)
    assert roles.roles["holder"] == "robot_a"
    assert roles.roles["passer"] == "robot_b"


def test_role_assignment_relay_is_deterministic():
    world = relay_task_world()
    facts = interpret_scene(TaskType.T6_RELAY_DELIVERY, world)
    roles = assign_roles(facts, world)
    assert roles.roles["starter"] == "robot_a"
    assert roles.roles["finisher"] == "robot_b"


def test_role_assignment_search_assigns_disjoint_regions():
    world = search_task_world()
    facts = interpret_scene(TaskType.T4_COLLABORATIVE_SEARCH_CONVERGE, world)
    roles = assign_roles(facts, world)
    assignments = roles.roles["search_assignments"]
    assert set(assignments.keys()) == {"robot_a", "robot_b"}
    assert len(set(assignments.values())) == 2

