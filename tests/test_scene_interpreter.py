from vcomt2d.planner.models import TaskType
from vcomt2d.planner.scene_interpreter import interpret_scene
from vcomt2d.sim.entities import RegionState
from vcomt2d.sim.tasks.fixtures import door_task_world, herding_task_world, relay_task_world, search_task_world


def test_scene_interpreter_door():
    facts = interpret_scene(TaskType.T1_DOOR_WEDGE_PASS_THROUGH, door_task_world())
    assert facts.relevant_door_id == "door_main"
    assert facts.goal_region_id == "room_goal"


def test_scene_interpreter_herding():
    facts = interpret_scene(TaskType.T2_HERDING_CORRALLING, herding_task_world())
    assert facts.target_object_id == "ball_1"
    assert facts.goal_region_id == "goal_corner"
    assert len(facts.staging_region_ids) >= 2


def test_scene_interpreter_search():
    facts = interpret_scene(TaskType.T4_COLLABORATIVE_SEARCH_CONVERGE, search_task_world())
    assert facts.target_object_id == "box_red"
    assert len(facts.search_region_ids) == 2
    assert facts.goal_region_id == "target_region"


def test_scene_interpreter_relay():
    facts = interpret_scene(TaskType.T6_RELAY_DELIVERY, relay_task_world())
    assert facts.target_object_id == "box_1"
    assert facts.handoff_region_id == "handoff_mid"
    assert facts.goal_region_id == "far_goal"


def test_scene_interpreter_relay_prefers_plausible_handoff_region():
    world = relay_task_world()
    world.task_facts.pop("handoff_region_id", None)
    world.goals.append(RegionState("bad_handoff", (1.0, 1.0), 0.8, "handoff_region"))

    facts = interpret_scene(TaskType.T6_RELAY_DELIVERY, world)
    assert facts.handoff_region_id == "handoff_mid"
