from vcomt2d.planner.main import DeterministicPlanner
from vcomt2d.sim.tasks.fixtures import door_task_world, herding_task_world, relay_task_world, search_task_world


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests that touch animation export")


def planner():
    return DeterministicPlanner()


FIXTURE_BUILDERS = {
    "door": door_task_world,
    "herding": herding_task_world,
    "search": search_task_world,
    "relay": relay_task_world,
}

