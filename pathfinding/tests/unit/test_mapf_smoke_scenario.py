import pytest

from pathfinding.src.algorithms.mapf.smoke_scenario import build_mapf_scenario
from pathfinding.src.core.models import GridMap, Position, Scenario
from pathfinding.tests.helpers import build_grid_map


def _scenario(
    start_row: int,
    start_col: int,
    goal_row: int,
    goal_col: int,
    *,
    map_name: str = "test.map",
) -> Scenario:
    return Scenario(
        map_name=map_name,
        width=5,
        height=5,
        start=Position(row=start_row, col=start_col),
        goal=Position(row=goal_row, col=goal_col),
        optimal_length=1.0,
    )


def test_returns_requested_number_of_agents() -> None:
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
        _scenario(0, 0, 0, 4),
        _scenario(1, 0, 1, 4),
        _scenario(2, 0, 2, 4),
    ]

    mapf_scenario = build_mapf_scenario(
        scenarios=scenarios,
        grid_map=grid_map,
        agent_count=3,
    )

    assert len(mapf_scenario.agents) == 3


def test_agent_ids_are_zero_based_in_selection_order() -> None:
    grid_map = build_grid_map([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    scenarios = [
        _scenario(0, 0, 0, 2),
        _scenario(1, 0, 1, 2),
    ]

    mapf_scenario = build_mapf_scenario(
        scenarios=scenarios,
        grid_map=grid_map,
        agent_count=2,
    )

    assert [agent.agent_id for agent in mapf_scenario.agents] == [0, 1]


def test_scenario_order_follows_candidate_order() -> None:
    grid_map = build_grid_map([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    first = _scenario(0, 0, 0, 2)
    second = _scenario(1, 0, 1, 2)
    third = _scenario(2, 0, 2, 2)

    mapf_scenario = build_mapf_scenario(
        scenarios=[first, second, third],
        grid_map=grid_map,
        agent_count=2,
    )

    assert mapf_scenario.agents[0].start == first.start
    assert mapf_scenario.agents[0].goal == first.goal
    assert mapf_scenario.agents[1].start == second.start
    assert mapf_scenario.agents[1].goal == second.goal


def test_duplicate_starts_are_skipped() -> None:
    grid_map = build_grid_map([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    scenarios = [
        _scenario(0, 0, 0, 2),
        _scenario(0, 0, 1, 1),
        _scenario(1, 0, 1, 2),
    ]

    mapf_scenario = build_mapf_scenario(
        scenarios=scenarios,
        grid_map=grid_map,
        agent_count=2,
    )

    assert mapf_scenario.agents[0].start == scenarios[0].start
    assert mapf_scenario.agents[1].start == scenarios[2].start


def test_duplicate_goals_are_skipped() -> None:
    grid_map = build_grid_map([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    scenarios = [
        _scenario(0, 0, 0, 2),
        _scenario(1, 0, 0, 2),
        _scenario(2, 0, 2, 1),
    ]

    mapf_scenario = build_mapf_scenario(
        scenarios=scenarios,
        grid_map=grid_map,
        agent_count=2,
    )

    assert mapf_scenario.agents[0].goal == scenarios[0].goal
    assert mapf_scenario.agents[1].goal == scenarios[2].goal


def test_start_equals_goal_candidates_are_skipped() -> None:
    grid_map = build_grid_map([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    scenarios = [
        _scenario(0, 0, 0, 0),
        _scenario(0, 0, 0, 2),
        _scenario(1, 0, 1, 2),
    ]

    mapf_scenario = build_mapf_scenario(
        scenarios=scenarios,
        grid_map=grid_map,
        agent_count=2,
    )

    assert mapf_scenario.agents[0].start == scenarios[1].start
    assert mapf_scenario.agents[1].start == scenarios[2].start


def test_row_col_follows_existing_loader_convention() -> None:
    grid_map = GridMap(
        name="movingai.map",
        width=512,
        height=512,
        cells=[[0 for _ in range(512)] for _ in range(512)],
    )
    line = "0 AR0204SR.map 512 512 328 151 330 153 2.82842712"
    parts = line.split()

    scenario = Scenario(
        map_name=parts[1],
        width=int(parts[2]),
        height=int(parts[3]),
        start=Position(row=int(parts[5]), col=int(parts[4])),
        goal=Position(row=int(parts[7]), col=int(parts[6])),
        optimal_length=float(parts[8]),
    )

    mapf_scenario = build_mapf_scenario(
        scenarios=[scenario],
        grid_map=grid_map,
        agent_count=1,
    )

    agent = mapf_scenario.agents[0]
    assert agent.start.row == 151
    assert agent.start.col == 328
    assert agent.goal.row == 153
    assert agent.goal.col == 330


def test_insufficient_valid_candidates_raises_clear_error() -> None:
    grid_map = build_grid_map([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    scenarios = [
        _scenario(0, 0, 0, 2),
        _scenario(1, 0, 1, 2),
    ]

    with pytest.raises(ValueError, match="Could not collect 3 valid MAPF agents"):
        build_mapf_scenario(
            scenarios=scenarios,
            grid_map=grid_map,
            agent_count=3,
        )


def test_non_walkable_positions_are_skipped() -> None:
    grid_map = build_grid_map(
        [
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    scenarios = [
        _scenario(0, 1, 0, 2),
        _scenario(0, 0, 0, 2),
        _scenario(1, 0, 1, 2),
    ]

    mapf_scenario = build_mapf_scenario(
        scenarios=scenarios,
        grid_map=grid_map,
        agent_count=2,
    )

    assert mapf_scenario.agents[0].start == scenarios[1].start
    assert mapf_scenario.agents[1].start == scenarios[2].start
