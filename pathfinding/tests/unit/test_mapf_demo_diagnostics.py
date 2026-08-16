import pytest

from pathfinding.src.algorithms.mapf.demo_diagnostics import (
    count_agents_with_changed_path,
    count_agents_with_wait,
    count_total_wait_steps,
    count_wait_steps,
    paths_have_identical_states,
)
from pathfinding.src.algorithms.mapf.demo_scenario import (
    NON_TRIVIAL_DEMO_SCENARIO_INDICES,
    build_mapf_scenario_from_indices,
    build_non_trivial_demo_scenario,
)
from pathfinding.src.algorithms.mapf.models import AgentPath, TimedState
from pathfinding.src.core.models import GridMap, Position, Scenario
from pathfinding.tests.helpers import build_grid_map


def _path(agent_id: int, coordinates: list[tuple[int, int, int]]) -> AgentPath:
    return AgentPath(
        agent_id=agent_id,
        states=tuple(
            TimedState(row=row, col=col, timestep=timestep)
            for row, col, timestep in coordinates
        ),
    )


def _scenario(
    start_row: int,
    start_col: int,
    goal_row: int,
    goal_col: int,
    *,
    optimal_length: float = 10.0,
) -> Scenario:
    return Scenario(
        map_name="test.map",
        width=5,
        height=5,
        start=Position(row=start_row, col=start_col),
        goal=Position(row=goal_row, col=goal_col),
        optimal_length=optimal_length,
    )


def test_count_wait_steps_returns_zero_without_wait() -> None:
    path = _path(0, [(0, 0, 0), (0, 1, 1), (0, 2, 2)])

    assert count_wait_steps(path) == 0


def test_count_wait_steps_counts_single_wait() -> None:
    path = _path(0, [(0, 0, 0), (0, 0, 1), (0, 1, 2)])

    assert count_wait_steps(path) == 1


def test_count_total_wait_steps_across_multiple_agents() -> None:
    paths = (
        _path(0, [(0, 0, 0), (0, 0, 1), (0, 1, 2)]),
        _path(1, [(1, 0, 0), (1, 1, 1), (1, 1, 2), (1, 2, 3)]),
    )

    assert count_total_wait_steps(paths) == 2


def test_count_agents_with_wait() -> None:
    paths = (
        _path(0, [(0, 0, 0), (0, 1, 1)]),
        _path(1, [(1, 0, 0), (1, 0, 1), (1, 1, 2)]),
    )

    assert count_agents_with_wait(paths) == 1


def test_paths_have_identical_states_for_matching_paths() -> None:
    left = _path(0, [(0, 0, 0), (0, 1, 1)])
    right = _path(0, [(0, 0, 0), (0, 1, 1)])

    assert paths_have_identical_states(left, right) is True


def test_changed_path_counter_detects_wait_insertion() -> None:
    independent = (_path(0, [(0, 0, 0), (0, 1, 1)]),)
    coordinated = (_path(0, [(0, 0, 0), (0, 0, 1), (0, 1, 2)]),)

    assert count_agents_with_changed_path(independent, coordinated) == 1


def test_changed_path_counter_detects_spatial_detour() -> None:
    independent = (_path(0, [(0, 0, 0), (0, 1, 1), (0, 2, 2)]),)
    coordinated = (_path(0, [(0, 0, 0), (1, 0, 1), (1, 1, 2), (0, 2, 3)]),)

    assert count_agents_with_changed_path(independent, coordinated) == 1


def test_changed_path_counter_leaves_identical_agent_unchanged() -> None:
    path = _path(0, [(0, 0, 0), (0, 1, 1)])

    assert count_agents_with_changed_path((path,), (path,)) == 0


def test_build_mapf_scenario_from_indices_returns_requested_agents() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    scenarios = [
        _scenario(0, 0, 0, 4, optimal_length=50.0),
        _scenario(1, 0, 1, 4, optimal_length=55.0),
        _scenario(2, 0, 2, 4, optimal_length=60.0),
    ]

    selection = build_mapf_scenario_from_indices(
        scenarios=scenarios,
        grid_map=grid_map,
        scenario_indices=(0, 1, 2),
    )

    assert len(selection.scenario.agents) == 3
    assert selection.scenario_indices == (0, 1, 2)


def test_build_mapf_scenario_from_indices_preserves_order_and_ids() -> None:
    grid_map = build_grid_map([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    scenarios = [
        _scenario(0, 0, 0, 2, optimal_length=50.0),
        _scenario(1, 0, 1, 2, optimal_length=55.0),
    ]

    selection = build_mapf_scenario_from_indices(
        scenarios=scenarios,
        grid_map=grid_map,
        scenario_indices=(1, 0),
    )

    assert [agent.agent_id for agent in selection.scenario.agents] == [0, 1]
    assert selection.agents[0].scenario_index == 1
    assert selection.agents[1].scenario_index == 0


def test_build_mapf_scenario_from_indices_is_deterministic() -> None:
    grid_map = build_grid_map([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    scenarios = [
        _scenario(0, 0, 0, 2, optimal_length=50.0),
        _scenario(1, 0, 1, 2, optimal_length=55.0),
    ]

    first = build_mapf_scenario_from_indices(
        scenarios=scenarios,
        grid_map=grid_map,
        scenario_indices=(0, 1),
    )
    second = build_mapf_scenario_from_indices(
        scenarios=scenarios,
        grid_map=grid_map,
        scenario_indices=(0, 1),
    )

    assert first == second


def test_build_mapf_scenario_from_indices_enforces_unique_starts_and_goals() -> None:
    grid_map = build_grid_map([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    scenarios = [
        _scenario(0, 0, 0, 2, optimal_length=50.0),
        _scenario(0, 0, 1, 1, optimal_length=55.0),
    ]

    with pytest.raises(ValueError, match="not a valid MAPF agent candidate"):
        build_mapf_scenario_from_indices(
            scenarios=scenarios,
            grid_map=grid_map,
            scenario_indices=(0, 1),
        )


def test_frozen_demo_fixture_indices_match_expected_count() -> None:
    assert len(NON_TRIVIAL_DEMO_SCENARIO_INDICES) == 10


def test_build_non_trivial_demo_scenario_uses_frozen_indices() -> None:
    grid_map = GridMap(
        name="demo.map",
        width=512,
        height=512,
        cells=[[0 for _ in range(512)] for _ in range(512)],
    )
    scenarios = [
        _scenario(index, 0, index, 1, optimal_length=50.0 + index)
        for index in range(max(NON_TRIVIAL_DEMO_SCENARIO_INDICES) + 1)
    ]

    selection = build_non_trivial_demo_scenario(
        scenarios=scenarios,
        grid_map=grid_map,
    )

    assert selection.scenario_indices == NON_TRIVIAL_DEMO_SCENARIO_INDICES
    assert [spec.scenario_index for spec in selection.agents] == list(
        NON_TRIVIAL_DEMO_SCENARIO_INDICES
    )
