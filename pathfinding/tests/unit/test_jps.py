import pytest

from pathfinding.src.algorithms.astar import AStar
from pathfinding.src.algorithms.jps import JumpPointSearch
from pathfinding.src.core.models import Position
from pathfinding.tests.helpers import build_grid_map


def test_jps_matches_astar_on_open_map() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )

    start = Position(row=0, col=0)
    goal = Position(row=3, col=3)

    astar_result = AStar().find_path(grid_map, start, goal)
    jps_result = JumpPointSearch().find_path(grid_map, start, goal)

    assert jps_result.found is True
    assert jps_result.path_cost == pytest.approx(astar_result.path_cost)
    assert jps_result.path_length == astar_result.path_length


def test_jps_returns_not_found_when_path_is_blocked() -> None:
    grid_map = build_grid_map(
        [
            [0, 1, 0],
        ]
    )

    result = JumpPointSearch().find_path(
        grid_map=grid_map,
        start=Position(row=0, col=0),
        goal=Position(row=0, col=2),
    )

    assert result.found is False
    assert result.path == []
    assert result.path_cost == 0.0
