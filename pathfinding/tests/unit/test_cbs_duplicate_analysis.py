import pytest

from pathfinding.src.algorithms.mapf.cbs import (
    _constraint_signature,
    _path_set_signature,
    _path_signature,
    build_cbs_root,
    solve_cbs_with_stats,
)
from pathfinding.src.algorithms.mapf.models import (
    AgentPath,
    EdgeConstraint,
    MAPFAgent,
    MAPFScenario,
    TimedState,
    VertexConstraint,
)
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


def _path(agent_id: int, coordinates: list[tuple[int, int, int]]) -> AgentPath:
    return AgentPath(
        agent_id=agent_id,
        states=tuple(
            TimedState(row=row, col=col, timestep=timestep)
            for row, col, timestep in coordinates
        ),
    )


def _distribution_total(distribution: tuple[tuple[int, int], ...]) -> int:
    return sum(count for _, count in distribution)


def test_empty_constraint_signature_is_stable() -> None:
    assert _constraint_signature(()) == frozenset()


def test_constraint_signature_is_order_independent() -> None:
    first = (
        VertexConstraint(agent_id=1, row=2, col=3, timestep=4),
        EdgeConstraint(
            agent_id=2,
            from_row=0,
            from_col=1,
            to_row=0,
            to_col=2,
            timestep=5,
        ),
    )
    second = (first[1], first[0])

    assert _constraint_signature(first) == _constraint_signature(second)


def test_different_constraints_produce_different_signatures() -> None:
    earlier = (
        VertexConstraint(agent_id=1, row=2, col=3, timestep=4),
    )
    later = (
        VertexConstraint(agent_id=1, row=2, col=3, timestep=5),
    )

    assert _constraint_signature(earlier) != _constraint_signature(later)


def test_identical_structural_paths_share_path_signature() -> None:
    coordinates = [(0, 0, 0), (0, 1, 1), (0, 2, 2)]
    first = _path(agent_id=3, coordinates=coordinates)
    second = _path(agent_id=3, coordinates=coordinates)

    assert first is not second
    assert _path_signature(first) == _path_signature(second)


def test_different_timesteps_produce_different_path_signatures() -> None:
    first = _path(agent_id=0, coordinates=[(0, 0, 0), (0, 0, 1), (0, 1, 2)])
    second = _path(agent_id=0, coordinates=[(0, 0, 0), (0, 1, 1)])

    assert _path_signature(first) != _path_signature(second)


def test_different_agent_ids_produce_different_path_signatures() -> None:
    coordinates = [(0, 0, 0), (0, 1, 1)]
    first = _path(agent_id=0, coordinates=coordinates)
    second = _path(agent_id=1, coordinates=coordinates)

    assert _path_signature(first) != _path_signature(second)


def test_conflict_free_root_duplicate_analysis() -> None:
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

    run = solve_cbs_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=6,
    )
    stats = run.stats

    assert stats.generated_ct_nodes == 1
    assert stats.unique_constraint_signatures == 1
    assert stats.duplicate_constraint_signatures == 0
    assert stats.unique_path_signatures == 1
    assert stats.duplicate_path_signatures == 0
    assert stats.generated_cost_distribution == ((6, 1),)
    assert stats.expanded_cost_distribution == ()


def test_root_failure_duplicate_analysis() -> None:
    grid_map = build_grid_map([[0, 1]])
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=1, start_row=0, start_col=0, goal_row=0, goal_col=1),
    )

    run = solve_cbs_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
    )
    stats = run.stats

    assert stats.generated_ct_nodes == 0
    assert stats.unique_constraint_signatures == 0
    assert stats.duplicate_constraint_signatures == 0
    assert stats.unique_path_signatures == 0
    assert stats.duplicate_path_signatures == 0
    assert stats.generated_cost_distribution == ()
    assert stats.expanded_cost_distribution == ()


def test_duplicate_counts_match_generated_minus_unique() -> None:
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

    run = solve_cbs_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
    )
    stats = run.stats

    assert stats.duplicate_constraint_signatures == (
        stats.generated_ct_nodes - stats.unique_constraint_signatures
    )
    assert stats.duplicate_path_signatures == (
        stats.generated_ct_nodes - stats.unique_path_signatures
    )


def test_expansion_limit_zero_records_root_generated_distribution_only() -> None:
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

    run = solve_cbs_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
        max_expanded_nodes=0,
    )
    stats = run.stats

    assert stats.generated_cost_distribution == ((root.cost, 1),)
    assert stats.expanded_cost_distribution == ()


def test_one_expansion_distribution_totals_match_counters() -> None:
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

    run = solve_cbs_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
        max_expanded_nodes=1,
    )
    stats = run.stats

    assert _distribution_total(stats.generated_cost_distribution) == stats.generated_ct_nodes
    assert _distribution_total(stats.expanded_cost_distribution) == stats.expanded_ct_nodes


def test_path_set_signature_preserves_path_order() -> None:
    first_path = _path(agent_id=7, coordinates=[(0, 0, 0), (0, 1, 1)])
    second_path = _path(agent_id=2, coordinates=[(1, 0, 0), (1, 1, 1)])
    ordered = _path_set_signature((first_path, second_path))
    reversed_order = _path_set_signature((second_path, first_path))

    assert ordered != reversed_order
