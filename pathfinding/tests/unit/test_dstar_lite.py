import math

import pytest

from pathfinding.src.algorithms.astar import AStar
from pathfinding.src.algorithms.dstar_lite import DStarLite
from pathfinding.src.core.dynamic_models import DynamicGridMap
from pathfinding.src.core.models import Position
from pathfinding.tests.helpers import build_grid_map


def test_dstar_lite_finds_straight_path() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
        ]
    )

    result = DStarLite().find_path(
        grid_map=grid_map,
        start=Position(row=0, col=0),
        goal=Position(row=0, col=2),
    )

    assert result.found is True
    assert result.path_length == 2
    assert result.path_cost == pytest.approx(2.0)
    assert result.path[0] == Position(row=0, col=0)
    assert result.path[-1] == Position(row=0, col=2)


def test_dstar_lite_matches_astar_path_cost_on_open_map() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )

    start = Position(row=0, col=0)
    goal = Position(row=2, col=2)

    astar_result = AStar().find_path(
        grid_map=grid_map,
        start=start,
        goal=goal,
    )
    dstar_result = DStarLite().find_path(
        grid_map=grid_map,
        start=start,
        goal=goal,
    )

    assert astar_result.found is True
    assert dstar_result.found is True
    assert dstar_result.path_cost == pytest.approx(astar_result.path_cost)


def test_dstar_lite_finds_diagonal_path_with_corner_cutting() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )

    result = DStarLite().find_path(
        grid_map=grid_map,
        start=Position(row=0, col=0),
        goal=Position(row=2, col=2),
    )

    assert result.found is True
    assert result.path_length == 2
    assert result.path_cost == pytest.approx(2 * math.sqrt(2))


def test_dstar_lite_returns_not_found_when_path_is_blocked() -> None:
    grid_map = build_grid_map(
        [
            [0, 1, 0],
        ]
    )

    result = DStarLite().find_path(
        grid_map=grid_map,
        start=Position(row=0, col=0),
        goal=Position(row=0, col=2),
    )

    assert result.found is False
    assert result.path == []
    assert result.path_cost == 0.0


def test_dstar_lite_replan_after_dynamic_block() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    dynamic_map = DynamicGridMap(base_map=grid_map)

    start = Position(row=0, col=0)
    goal = Position(row=0, col=4)

    dstar = DStarLite()
    initial_result = dstar.find_path(
        grid_map=dynamic_map,
        start=start,
        goal=goal,
    )

    assert initial_result.found is True

    dynamic_map.block_cell(Position(row=0, col=2))
    dstar.update_cell(Position(row=0, col=2))

    replanned_result = dstar.replan()

    assert replanned_result.found is True
    assert replanned_result.path[-1] == goal
    assert replanned_result.path_cost >= initial_result.path_cost
