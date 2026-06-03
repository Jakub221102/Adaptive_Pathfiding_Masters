import math

import pytest

from pathfinding.src.algorithms.astar import AStar
from pathfinding.src.core.models import Position
from pathfinding.tests.helpers import build_grid_map


def test_astar_finds_straight_path() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
        ]
    )

    result = AStar().find_path(
        grid_map=grid_map,
        start=Position(row=0, col=0),
        goal=Position(row=0, col=2),
    )

    assert result.found is True
    assert result.path_length == 2
    assert result.path_cost == pytest.approx(2.0)
    assert result.path[0] == Position(row=0, col=0)
    assert result.path[-1] == Position(row=0, col=2)


def test_astar_finds_diagonal_path_with_corner_cutting() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )

    result = AStar().find_path(
        grid_map=grid_map,
        start=Position(row=0, col=0),
        goal=Position(row=2, col=2),
    )

    assert result.found is True
    assert result.path_length == 2
    assert result.path_cost == pytest.approx(2 * math.sqrt(2))


def test_astar_returns_not_found_when_path_is_blocked() -> None:
    grid_map = build_grid_map(
        [
            [0, 1, 0],
        ]
    )

    result = AStar().find_path(
        grid_map=grid_map,
        start=Position(row=0, col=0),
        goal=Position(row=0, col=2),
    )

    assert result.found is False
    assert result.path == []
    assert result.path_length == 0
    assert result.path_cost == 0.0
