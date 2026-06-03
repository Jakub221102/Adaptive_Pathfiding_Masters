import math

import pytest

from pathfinding.src.algorithms.astar import AStar
from pathfinding.src.core.models import Position
from pathfinding.tests.helpers import build_grid_map


def test_horizontal_cost() -> None:
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

    assert result.path_cost == pytest.approx(2.0)


def test_vertical_cost() -> None:
    grid_map = build_grid_map(
        [
            [0],
            [0],
            [0],
        ]
    )

    result = AStar().find_path(
        grid_map=grid_map,
        start=Position(row=0, col=0),
        goal=Position(row=2, col=0),
    )

    assert result.path_cost == pytest.approx(2.0)


def test_diagonal_cost() -> None:
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

    assert result.path_cost == pytest.approx(
        2 * math.sqrt(2)
    )
