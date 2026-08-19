from __future__ import annotations

import pytest

from pathfinding.src.algorithms.mapf.models import AgentPath, MAPFAgent, TimedState
from pathfinding.src.algorithms.mapf.space_time_astar import find_path
from pathfinding.src.core.models import GridMap, Position, Scenario
from pathfinding.src.experiments.mapf_benchmark_instances import (
    evaluate_benchmark_candidate,
)
from pathfinding.src.experiments.mapf_independent_static_path import (
    StaticPathSearchStatus,
    evaluate_candidate_with_independent_paths,
    find_independent_static_path,
    find_independent_static_path_bounded,
    find_independent_static_path_bounded_result,
    independent_path_cost,
    independent_path_excess_moves,
    independent_path_fits_horizon,
    merge_diagnostic_scenario_indices,
    spatial_trajectory,
)
from pathfinding.tests.helpers import build_grid_map


def _agent(
    start_row: int,
    start_col: int,
    goal_row: int,
    goal_col: int,
    agent_id: int = 0,
) -> MAPFAgent:
    return MAPFAgent(
        agent_id=agent_id,
        start=Position(row=start_row, col=start_col),
        goal=Position(row=goal_row, col=goal_col),
    )


def _scenario(
    start_row: int,
    start_col: int,
    goal_row: int,
    goal_col: int,
    *,
    optimal_length: float = 25.0,
    width: int = 5,
    height: int = 3,
) -> Scenario:
    return Scenario(
        map_name="test.map",
        width=width,
        height=height,
        start=Position(row=start_row, col=start_col),
        goal=Position(row=goal_row, col=goal_col),
        optimal_length=optimal_length,
    )


def test_static_straight_path() -> None:
    grid_map = build_grid_map(
        [[0, 0, 0, 0, 0] for _ in range(3)],
        name="straight.map",
    )
    path = find_independent_static_path(grid_map, _agent(1, 0, 1, 4))
    assert path is not None
    assert independent_path_cost(path) == 4
    assert spatial_trajectory(path) == ((1, 0), (1, 1), (1, 2), (1, 3), (1, 4))


def test_static_obstacle_detour() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        name="detour.map",
    )
    path = find_independent_static_path(grid_map, _agent(1, 0, 1, 4))
    assert path is not None
    assert independent_path_cost(path) == 6
    assert path.states[0].row == 1 and path.states[-1].row == 1


def test_static_start_equals_goal() -> None:
    grid_map = build_grid_map([[0]], name="single.map")
    path = find_independent_static_path(grid_map, _agent(0, 0, 0, 0))
    assert path is not None
    assert path.states == (TimedState(row=0, col=0, timestep=0),)
    assert independent_path_cost(path) == 0


def test_static_blocked_start() -> None:
    grid_map = build_grid_map([[1]], name="blocked.map")
    assert find_independent_static_path(grid_map, _agent(0, 0, 0, 0)) is None


def test_static_blocked_goal() -> None:
    grid_map = build_grid_map(
        [
            [0, 1],
            [0, 0],
        ],
        name="blocked_goal.map",
    )
    assert find_independent_static_path(grid_map, _agent(0, 0, 0, 1)) is None


def test_static_unreachable_target() -> None:
    grid_map = build_grid_map(
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ],
        name="split.map",
    )
    assert find_independent_static_path(grid_map, _agent(0, 0, 0, 2)) is None


def test_static_consecutive_timesteps() -> None:
    grid_map = build_grid_map([[0, 0, 0]], name="line.map")
    path = find_independent_static_path(grid_map, _agent(0, 0, 0, 2))
    assert path is not None
    for index in range(len(path.states) - 1):
        assert path.states[index + 1].timestep == path.states[index].timestep + 1


def test_static_only_four_connected_moves() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        name="grid.map",
    )
    path = find_independent_static_path(grid_map, _agent(0, 0, 2, 2))
    assert path is not None
    for current, nxt in zip(path.states, path.states[1:]):
        row_diff = abs(current.row - nxt.row)
        col_diff = abs(current.col - nxt.col)
        assert row_diff + col_diff == 1


def test_static_no_wait_in_nontrivial_path() -> None:
    grid_map = build_grid_map([[0, 0, 0, 0]], name="line.map")
    path = find_independent_static_path(grid_map, _agent(0, 0, 0, 3))
    assert path is not None
    for current, nxt in zip(path.states, path.states[1:]):
        assert (current.row, current.col) != (nxt.row, nxt.col)


def test_static_deterministic_repeated_result() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ],
        name="repeat.map",
    )
    agent = _agent(0, 0, 2, 3)
    first = find_independent_static_path(grid_map, agent)
    second = find_independent_static_path(grid_map, agent)
    assert first == second


@pytest.mark.parametrize(
    "grid_cells,agent,max_timestep",
    [
        (
            [[0, 0, 0, 0, 0] for _ in range(3)],
            _agent(1, 0, 1, 4),
            64,
        ),
        (
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            _agent(1, 0, 1, 4),
            64,
        ),
        (
            [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            _agent(0, 0, 2, 5),
            64,
        ),
        (
            [[0, 0], [0, 0]],
            _agent(0, 0, 1, 1),
            8,
        ),
    ],
)
def test_static_vs_sta_cost_equivalence_on_synthetic_grids(
    grid_cells: list[list[int]],
    agent: MAPFAgent,
    max_timestep: int,
) -> None:
    grid_map = build_grid_map(grid_cells, name="synthetic.map")
    static_path = find_independent_static_path(grid_map, agent)
    sta_path = find_path(
        grid_map=grid_map,
        agent=agent,
        max_timestep=max_timestep,
        constraints=(),
    )

    assert (static_path is None) == (sta_path is None)
    if static_path is None:
        return

    assert sta_path is not None
    assert independent_path_cost(static_path) == independent_path_cost(sta_path)


def test_static_vs_sta_trajectory_may_differ_on_equal_cost_grid() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        name="multi_optimal.map",
    )
    agent = _agent(0, 0, 2, 2)
    static_path = find_independent_static_path(grid_map, agent)
    sta_path = find_path(
        grid_map=grid_map,
        agent=agent,
        max_timestep=16,
        constraints=(),
    )

    assert static_path is not None
    assert sta_path is not None
    assert independent_path_cost(static_path) == independent_path_cost(sta_path)
    # Multiple shortest paths exist; exact trajectory equality is not required.


def test_candidate_interaction_equivalence_on_synthetic_pool() -> None:
    grid_map = build_grid_map(
        [[0 for _ in range(9)] for _ in range(9)],
        name="bench.map",
    )
    scenarios = [
        _scenario(row, 0, row, 8, width=9, height=9, optimal_length=25.0 + row)
        for row in range(9)
    ]
    scenario_indices = (1, 4, 8)

    sta_evaluation = evaluate_benchmark_candidate(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_indices=scenario_indices,
        max_timestep=64,
        precomputed_lookup=None,
    )

    static_paths = []
    for agent_id, scenario_index in enumerate(scenario_indices):
        scenario = scenarios[scenario_index]
        path = find_independent_static_path(
            grid_map,
            MAPFAgent(agent_id=agent_id, start=scenario.start, goal=scenario.goal),
        )
        assert path is not None
        static_paths.append(path)

    static_evaluation = evaluate_candidate_with_independent_paths(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_indices=scenario_indices,
        independent_paths=static_paths,
    )

    assert sta_evaluation is not None
    assert static_evaluation is not None
    assert (
        sta_evaluation.independent_conflict_count
        == static_evaluation.independent_conflict_count
    )
    assert (
        sta_evaluation.vertex_conflict_count
        == static_evaluation.vertex_conflict_count
    )
    assert sta_evaluation.edge_conflict_count == static_evaluation.edge_conflict_count
    assert (
        sta_evaluation.conflicting_agent_pair_count
        == static_evaluation.conflicting_agent_pair_count
    )
    assert sta_evaluation.independent_soc == static_evaluation.independent_soc
    assert sta_evaluation.independent_makespan == static_evaluation.independent_makespan
    assert sta_evaluation.interaction_level == static_evaluation.interaction_level


def test_independent_path_fits_horizon_zero_cost() -> None:
    path = AgentPath(
        agent_id=0,
        states=(TimedState(row=0, col=0, timestep=0),),
    )
    assert independent_path_fits_horizon(path, max_timestep=0)
    assert independent_path_excess_moves(path, max_timestep=0) == 0


def test_independent_path_fits_horizon_exact_limit() -> None:
    states = tuple(
        TimedState(row=0, col=index, timestep=index) for index in range(513)
    )
    path = AgentPath(agent_id=0, states=states)
    assert independent_path_cost(path) == 512
    assert independent_path_fits_horizon(path, max_timestep=512)
    assert independent_path_excess_moves(path, max_timestep=512) == 0


def test_independent_path_exceeds_horizon_by_one() -> None:
    states = tuple(
        TimedState(row=0, col=index, timestep=index) for index in range(514)
    )
    path = AgentPath(agent_id=0, states=states)
    assert independent_path_cost(path) == 513
    assert not independent_path_fits_horizon(path, max_timestep=512)
    assert independent_path_excess_moves(path, max_timestep=512) == 1


def test_independent_path_fits_horizon_negative_max_timestep_raises() -> None:
    path = AgentPath(
        agent_id=0,
        states=(TimedState(row=0, col=0, timestep=0),),
    )
    with pytest.raises(ValueError, match="max_timestep must be non-negative"):
        independent_path_fits_horizon(path, max_timestep=-1)


def test_merge_diagnostic_scenario_indices_deduplicates_special() -> None:
    merged = merge_diagnostic_scenario_indices(
        (50, 51, 1125, 60),
        (1125, 999),
    )
    assert merged == (50, 51, 1125, 60, 999)


def test_bounded_negative_max_cost_raises() -> None:
    grid_map = build_grid_map([[0]], name="single.map")
    with pytest.raises(ValueError, match="max_cost must be non-negative"):
        find_independent_static_path_bounded(
            grid_map,
            _agent(0, 0, 0, 0),
            max_cost=-1,
        )


def test_bounded_start_equals_goal_zero_max_cost() -> None:
    grid_map = build_grid_map([[0]], name="single.map")
    path = find_independent_static_path_bounded(
        grid_map,
        _agent(0, 0, 0, 0),
        max_cost=0,
    )
    assert path is not None
    assert path.states == (TimedState(row=0, col=0, timestep=0),)


def test_bounded_manhattan_short_circuit_skips_astar() -> None:
    grid_map = build_grid_map(
        [[0, 0, 0, 0, 0] for _ in range(3)],
        name="manhattan_skip.map",
    )
    agent = _agent(1, 0, 1, 4)

    import pathfinding.src.experiments.mapf_independent_static_path as static_module

    original_astar = static_module._static_astar

    def raising_astar(*args, **kwargs):
        raise AssertionError("_static_astar must not run when Manhattan exceeds max_cost")

    static_module._static_astar = raising_astar
    try:
        result = find_independent_static_path_bounded_result(
            grid_map,
            agent,
            max_cost=3,
        )
    finally:
        static_module._static_astar = original_astar

    assert result.path is None
    assert result.status == StaticPathSearchStatus.OVER_COST_BOUND
    assert result.expansions == 0


def test_bounded_exact_boundary_semantics() -> None:
    grid_map = build_grid_map(
        [[0, 0, 0, 0, 0] for _ in range(3)],
        name="boundary.map",
    )
    agent = _agent(1, 0, 1, 4)
    unbounded = find_independent_static_path(grid_map, agent)
    assert unbounded is not None
    assert independent_path_cost(unbounded) == 4

    assert find_independent_static_path_bounded(grid_map, agent, max_cost=3) is None
    at_limit = find_independent_static_path_bounded(grid_map, agent, max_cost=4)
    above_limit = find_independent_static_path_bounded(grid_map, agent, max_cost=5)
    assert at_limit == unbounded
    assert above_limit == unbounded


@pytest.mark.parametrize(
    "grid_cells,agent",
    [
        (
            [[0, 0, 0, 0, 0] for _ in range(3)],
            _agent(1, 0, 1, 4),
        ),
        (
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            _agent(1, 0, 1, 4),
        ),
        (
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            _agent(0, 0, 2, 2),
        ),
    ],
)
def test_bounded_matches_unbounded_when_feasible(
    grid_cells: list[list[int]],
    agent: MAPFAgent,
) -> None:
    grid_map = build_grid_map(grid_cells, name="equivalence.map")
    unbounded = find_independent_static_path(grid_map, agent)
    assert unbounded is not None
    cost = independent_path_cost(unbounded)
    assert cost is not None

    bounded = find_independent_static_path_bounded(
        grid_map,
        agent,
        max_cost=cost,
    )
    assert bounded == unbounded


def test_bounded_no_path_on_disconnected_grid() -> None:
    grid_map = build_grid_map(
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ],
        name="split.map",
    )
    result = find_independent_static_path_bounded_result(
        grid_map,
        _agent(0, 0, 0, 2),
        max_cost=64,
    )
    assert result.path is None
    assert result.status == StaticPathSearchStatus.NO_PATH
