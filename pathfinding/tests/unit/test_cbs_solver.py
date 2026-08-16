import pytest

from pathfinding.src.algorithms.mapf.cbs import build_cbs_root, expand_cbs_node, solve_cbs
from pathfinding.src.algorithms.mapf.conflicts import detect_conflicts
from pathfinding.src.algorithms.mapf.metrics import sum_of_costs
from pathfinding.src.algorithms.mapf.models import (
    EdgeConflict,
    MAPFAgent,
    MAPFResult,
    MAPFScenario,
    VertexConflict,
)
from pathfinding.src.algorithms.mapf.prioritized_planning import plan_prioritized
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


def _assert_successful_solution(result: MAPFResult, scenario: MAPFScenario) -> None:
    assert result.success is True
    assert len(result.paths) == len(scenario.agents)
    assert detect_conflicts(result.paths) == ()
    assert sum_of_costs(result.paths) >= 0

    for path, agent in zip(result.paths, scenario.agents, strict=True):
        assert path.agent_id == agent.agent_id
        assert path.states[0].row == agent.start.row
        assert path.states[0].col == agent.start.col
        assert path.states[-1].row == agent.goal.row
        assert path.states[-1].col == agent.goal.col


def test_single_agent_solution() -> None:
    grid_map = build_grid_map([[0, 0, 0]])
    scenario = _scenario(_agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=2))

    result = solve_cbs(grid_map=grid_map, scenario=scenario, max_timestep=5)

    _assert_successful_solution(result, scenario)


def test_conflict_free_multi_agent_terminates_at_root() -> None:
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

    result = solve_cbs(grid_map=grid_map, scenario=scenario, max_timestep=6)

    _assert_successful_solution(result, scenario)
    assert sum_of_costs(result.paths) == root.cost


def test_vertex_conflict_is_resolved() -> None:
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
    assert any(isinstance(conflict, VertexConflict) for conflict in root.conflicts)

    result = solve_cbs(grid_map=grid_map, scenario=scenario, max_timestep=5)

    _assert_successful_solution(result, scenario)
    assert result.paths != root.paths


def test_edge_swap_conflict_is_resolved() -> None:
    grid_map = build_grid_map(
        [
            [0, 0],
            [0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=1, start_row=0, start_col=1, goal_row=0, goal_col=0),
    )
    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=5)
    assert root is not None
    assert any(isinstance(conflict, EdgeConflict) for conflict in root.conflicts)

    result = solve_cbs(grid_map=grid_map, scenario=scenario, max_timestep=5)

    _assert_successful_solution(result, scenario)


def test_solver_requires_multiple_ct_expansions() -> None:
    grid_map = build_grid_map(
        [
            [0, 0],
            [0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=1, start_row=0, start_col=1, goal_row=0, goal_col=0),
    )
    max_timestep = 5
    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=max_timestep)
    assert root is not None
    assert root.conflicts

    first_children = expand_cbs_node(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        max_timestep=max_timestep,
    )
    assert len(first_children) == 2
    assert all(child.conflicts for child in first_children)

    result = solve_cbs(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
    )

    _assert_successful_solution(result, scenario)


def test_impossible_agent_returns_failure() -> None:
    grid_map = build_grid_map([[0, 1]])
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=1, start_row=0, start_col=0, goal_row=0, goal_col=1),
    )

    result = solve_cbs(grid_map=grid_map, scenario=scenario, max_timestep=5)

    assert result.success is False
    assert result.paths == ()


def test_no_coordinated_solution_within_horizon_returns_failure() -> None:
    grid_map = build_grid_map(
        [
            [0, 0],
            [0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=1, start_row=0, start_col=1, goal_row=0, goal_col=0),
    )
    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=2)
    assert root is not None

    result = solve_cbs(grid_map=grid_map, scenario=scenario, max_timestep=2)

    assert result.success is False
    assert result.paths == ()


def test_insufficient_horizon_for_root_returns_failure() -> None:
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

    result = solve_cbs(grid_map=grid_map, scenario=scenario, max_timestep=3)

    assert result.success is False
    assert result.paths == ()


def test_negative_max_timestep_raises_value_error() -> None:
    grid_map = build_grid_map([[0, 0]])
    scenario = _scenario(_agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1))

    with pytest.raises(ValueError, match="max_timestep must be non-negative"):
        solve_cbs(grid_map=grid_map, scenario=scenario, max_timestep=-1)


def test_solution_preserves_scenario_path_order() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=7, start_row=0, start_col=0, goal_row=0, goal_col=2),
        _agent(agent_id=2, start_row=1, start_col=1, goal_row=0, goal_col=0),
        _agent(agent_id=5, start_row=2, start_col=0, goal_row=2, goal_col=2),
    )

    result = solve_cbs(grid_map=grid_map, scenario=scenario, max_timestep=8)

    _assert_successful_solution(result, scenario)
    assert tuple(path.agent_id for path in result.paths) == (7, 2, 5)


def test_solver_is_deterministic() -> None:
    grid_map = build_grid_map(
        [
            [0, 0],
            [0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=1, start_row=0, start_col=1, goal_row=0, goal_col=0),
    )

    first = solve_cbs(grid_map=grid_map, scenario=scenario, max_timestep=5)
    second = solve_cbs(grid_map=grid_map, scenario=scenario, max_timestep=5)

    assert first == second


def test_known_optimal_sum_of_costs_for_swap_case() -> None:
    grid_map = build_grid_map(
        [
            [0, 0],
            [0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=1, start_row=0, start_col=1, goal_row=0, goal_col=0),
    )
    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=5)
    assert root is not None

    result = solve_cbs(grid_map=grid_map, scenario=scenario, max_timestep=5)

    _assert_successful_solution(result, scenario)
    assert root.cost == 2
    assert sum_of_costs(result.paths) == 4


def test_cbs_cost_is_no_worse_than_prioritized_planning() -> None:
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

    pp_result = plan_prioritized(grid_map=grid_map, scenario=scenario, max_timestep=5)
    cbs_result = solve_cbs(grid_map=grid_map, scenario=scenario, max_timestep=5)

    assert pp_result.success is True
    _assert_successful_solution(cbs_result, scenario)
    assert detect_conflicts(pp_result.paths) == ()
    assert sum_of_costs(cbs_result.paths) <= sum_of_costs(pp_result.paths)


def test_cbs_succeeds_when_prioritized_planning_order_fails() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    agent_a = _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=2)
    agent_b = _agent(agent_id=1, start_row=0, start_col=1, goal_row=0, goal_col=0)
    scenario = _scenario(agent_b, agent_a)

    pp_result = plan_prioritized(grid_map=grid_map, scenario=scenario, max_timestep=3)
    cbs_result = solve_cbs(grid_map=grid_map, scenario=scenario, max_timestep=3)

    assert pp_result.success is False
    assert pp_result.paths == ()
    _assert_successful_solution(cbs_result, scenario)
