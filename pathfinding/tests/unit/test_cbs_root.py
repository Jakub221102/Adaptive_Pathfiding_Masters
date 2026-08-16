import pytest

from pathfinding.src.algorithms.mapf.cbs import build_cbs_root
from pathfinding.src.algorithms.mapf.conflicts import detect_conflicts
from pathfinding.src.algorithms.mapf.metrics import sum_of_costs
from pathfinding.src.algorithms.mapf.models import (
    EdgeConflict,
    MAPFAgent,
    MAPFScenario,
    VertexConflict,
)
from pathfinding.src.algorithms.mapf.prioritized_planning import plan_prioritized
from pathfinding.src.algorithms.mapf.space_time_astar import find_path
from pathfinding.src.core.models import Position
from pathfinding.tests.helpers import build_grid_map


def _agent(
    agent_id: int,
    start_row: int,
    start_col: int,
    goal_row: int,
    goal_col: int,
) -> MAPFAgent:
    return MAPFAgent(
        agent_id=agent_id,
        start=Position(row=start_row, col=start_col),
        goal=Position(row=goal_row, col=goal_col),
    )


def _scenario(*agents: MAPFAgent) -> MAPFScenario:
    return MAPFScenario(agents=agents)


def test_single_agent_root_has_no_conflicts() -> None:
    grid_map = build_grid_map([[0, 0, 0]])
    scenario = _scenario(_agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=2))

    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=5)

    assert root is not None
    assert root.constraints == ()
    assert len(root.paths) == 1
    assert root.conflicts == ()
    assert root.cost == len(root.paths[0].states) - 1


def test_several_independent_agents_root_has_no_conflicts() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=3),
        _agent(agent_id=1, start_row=1, start_col=0, goal_row=1, goal_col=3),
    )

    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=6)

    assert root is not None
    assert len(root.paths) == 2
    assert root.conflicts == ()
    assert root.cost == sum_of_costs(root.paths)


def test_vertex_conflict_root_is_still_created() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=1, start_col=1, goal_row=0, goal_col=0),
        _agent(agent_id=1, start_row=0, start_col=0, goal_row=0, goal_col=2),
    )

    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=5)

    assert root is not None
    assert len(root.conflicts) > 0
    assert any(isinstance(conflict, VertexConflict) for conflict in root.conflicts)


def test_edge_swap_conflict_root_is_still_created() -> None:
    grid_map = build_grid_map([[0, 0]])
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=1, start_row=0, start_col=1, goal_row=0, goal_col=0),
    )

    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=3)

    assert root is not None
    assert any(isinstance(conflict, EdgeConflict) for conflict in root.conflicts)


def test_root_cost_equals_sum_of_costs() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=4),
        _agent(agent_id=1, start_row=1, start_col=0, goal_row=1, goal_col=2),
        _agent(agent_id=2, start_row=2, start_col=0, goal_row=2, goal_col=1),
    )

    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=8)

    assert root is not None
    assert root.cost == sum_of_costs(root.paths)


def test_root_paths_follow_scenario_order() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=7, start_row=0, start_col=0, goal_row=0, goal_col=2),
        _agent(agent_id=2, start_row=1, start_col=0, goal_row=1, goal_col=2),
        _agent(agent_id=5, start_row=2, start_col=0, goal_row=2, goal_col=2),
    )

    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=6)

    assert root is not None
    assert tuple(path.agent_id for path in root.paths) == (7, 2, 5)


def test_impossible_agent_returns_none() -> None:
    grid_map = build_grid_map([[0, 1]])
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=1, start_row=0, start_col=0, goal_row=0, goal_col=1),
    )

    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=5)

    assert root is None


def test_insufficient_horizon_returns_none() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=4),
        _agent(agent_id=1, start_row=1, start_col=4, goal_row=1, goal_col=0),
    )

    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=3)

    assert root is None


def test_negative_max_timestep_raises_value_error() -> None:
    grid_map = build_grid_map([[0, 0]])
    scenario = _scenario(_agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1))

    with pytest.raises(ValueError, match="max_timestep must be non-negative"):
        build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=-1)


def test_conflict_free_root_is_valid() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=3),
        _agent(agent_id=1, start_row=1, start_col=0, goal_row=1, goal_col=3),
    )

    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=6)

    assert root is not None
    assert root.conflicts == ()


def test_root_does_not_coordinate_like_prioritized_planning() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=1, start_col=1, goal_row=0, goal_col=0),
        _agent(agent_id=1, start_row=0, start_col=0, goal_row=0, goal_col=2),
    )
    max_timestep = 5

    root = build_cbs_root(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
    )
    pp_result = plan_prioritized(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
    )

    assert root is not None
    assert len(root.conflicts) > 0
    assert pp_result.success is True
    assert detect_conflicts(pp_result.paths) == ()
    assert root.paths[1] == find_path(
        grid_map=grid_map,
        agent=scenario.agents[1],
        max_timestep=max_timestep,
    )
