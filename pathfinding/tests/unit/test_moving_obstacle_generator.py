from itertools import combinations

from pathfinding.src.core.dynamic_models import get_occupied_positions
from pathfinding.src.core.moving_obstacle_generator import (
    MovingObstacleGeneratorConfig,
    _obstacle_center,
    generate_moving_obstacles,
)
from pathfinding.src.core.models import Position, Scenario
from pathfinding.tests.helpers import build_grid_map

_VALID_DIRECTIONS = {(1, 0), (-1, 0), (0, 1), (0, -1)}


def _open_grid(rows: int, cols: int) -> list[list[int]]:
    return [[0] * cols for _ in range(rows)]


def test_generator_is_deterministic() -> None:
    grid_map = build_grid_map(_open_grid(50, 50))

    config = MovingObstacleGeneratorConfig(
        seed=42,
        obstacle_count=10,
        min_width=4,
        max_width=8,
        min_height=4,
        max_height=8,
    )

    obstacles_a = generate_moving_obstacles(grid_map=grid_map, config=config)
    obstacles_b = generate_moving_obstacles(grid_map=grid_map, config=config)

    assert obstacles_a == obstacles_b


def test_generator_respects_forbidden_positions() -> None:
    grid_map = build_grid_map(_open_grid(40, 40))
    scenario = Scenario(
        map_name="test_map",
        width=40,
        height=40,
        start=Position(row=5, col=5),
        goal=Position(row=35, col=35),
        optimal_length=100.0,
    )

    forbidden_positions = {
        (scenario.start.row, scenario.start.col),
        (scenario.goal.row, scenario.goal.col),
    }

    obstacles = generate_moving_obstacles(
        grid_map=grid_map,
        config=MovingObstacleGeneratorConfig(
            seed=7,
            obstacle_count=8,
            min_width=3,
            max_width=6,
            min_height=3,
            max_height=6,
        ),
        forbidden_positions=forbidden_positions,
    )

    for obstacle in obstacles:
        occupied = {
            (position.row, position.col)
            for position in get_occupied_positions(obstacle)
        }
        assert occupied.isdisjoint(forbidden_positions)


def test_generator_creates_obstacles_within_map_bounds() -> None:
    grid_map = build_grid_map(_open_grid(60, 80))

    obstacles = generate_moving_obstacles(
        grid_map=grid_map,
        config=MovingObstacleGeneratorConfig(
            seed=123,
            obstacle_count=12,
            min_width=4,
            max_width=10,
            min_height=4,
            max_height=10,
        ),
    )

    for obstacle in obstacles:
        assert obstacle.row >= 0
        assert obstacle.col >= 0
        assert obstacle.row + obstacle.height <= grid_map.height
        assert obstacle.col + obstacle.width <= grid_map.width


def test_generator_creates_valid_directions() -> None:
    grid_map = build_grid_map(_open_grid(50, 50))

    obstacles = generate_moving_obstacles(
        grid_map=grid_map,
        config=MovingObstacleGeneratorConfig(
            seed=99,
            obstacle_count=10,
            min_width=3,
            max_width=7,
            min_height=3,
            max_height=7,
        ),
    )

    for obstacle in obstacles:
        assert (obstacle.delta_row, obstacle.delta_col) in _VALID_DIRECTIONS


def _chebyshev_center_distance(
        obstacle_a,
        obstacle_b,
) -> float:
    center_a = _obstacle_center(obstacle_a)
    center_b = _obstacle_center(obstacle_b)
    return max(
        abs(center_a[0] - center_b[0]),
        abs(center_a[1] - center_b[1]),
    )


def test_generator_respects_min_obstacle_spacing() -> None:
    grid_map = build_grid_map(_open_grid(100, 100))

    obstacles = generate_moving_obstacles(
        grid_map=grid_map,
        config=MovingObstacleGeneratorConfig(
            seed=42,
            obstacle_count=10,
            min_obstacle_spacing=20,
        ),
    )

    for obstacle_a, obstacle_b in combinations(obstacles, 2):
        assert _chebyshev_center_distance(obstacle_a, obstacle_b) >= 20


def test_min_obstacle_spacing_zero_preserves_generation() -> None:
    grid_map = build_grid_map(_open_grid(50, 50))

    obstacles = generate_moving_obstacles(
        grid_map=grid_map,
        config=MovingObstacleGeneratorConfig(
            seed=42,
            obstacle_count=10,
            min_obstacle_spacing=0,
        ),
    )

    assert len(obstacles) > 0


def test_too_large_min_obstacle_spacing_returns_fewer_obstacles() -> None:
    grid_map = build_grid_map(_open_grid(20, 20))

    obstacles = generate_moving_obstacles(
        grid_map=grid_map,
        config=MovingObstacleGeneratorConfig(
            seed=42,
            obstacle_count=10,
            min_obstacle_spacing=100,
        ),
    )

    assert len(obstacles) <= 10
