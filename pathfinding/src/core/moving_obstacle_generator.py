import random

from pydantic import BaseModel, Field

from pathfinding.src.core.dynamic_models import MovingObstacle
from pathfinding.src.core.models import GridMap, Position


class MovingObstacleGeneratorConfig(BaseModel):
    seed: int = 42
    obstacle_count: int = Field(default=10, ge=0)

    min_width: int = Field(default=4, ge=1)
    max_width: int = Field(default=10, ge=1)

    min_height: int = Field(default=4, ge=1)
    max_height: int = Field(default=10, ge=1)

    min_row: int = Field(default=0, ge=0)
    max_row: int | None = None

    min_col: int = Field(default=0, ge=0)
    max_col: int | None = None

    allow_vertical: bool = True
    allow_horizontal: bool = True


def _rectangle_is_valid(
        grid_map: GridMap,
        row: int,
        col: int,
        width: int,
        height: int,
        forbidden_positions: set[tuple[int, int]],
) -> bool:
    if row < 0 or col < 0:
        return False

    if row + height > grid_map.height:
        return False

    if col + width > grid_map.width:
        return False

    for cell_row in range(row, row + height):
        for cell_col in range(col, col + width):
            position = Position(row=cell_row, col=cell_col)
            if not grid_map.is_walkable(position):
                return False
            if (cell_row, cell_col) in forbidden_positions:
                return False

    return True


def _pick_direction(
        rng: random.Random,
        allow_vertical: bool,
        allow_horizontal: bool,
) -> tuple[int, int]:
    if allow_vertical and allow_horizontal:
        if rng.choice([True, False]):
            return rng.choice([(1, 0), (-1, 0)])
        return rng.choice([(0, 1), (0, -1)])

    if allow_vertical:
        return rng.choice([(1, 0), (-1, 0)])

    if allow_horizontal:
        return rng.choice([(0, 1), (0, -1)])

    raise ValueError("At least one movement direction must be allowed.")


def generate_moving_obstacles(
        grid_map: GridMap,
        config: MovingObstacleGeneratorConfig,
        forbidden_positions: set[tuple[int, int]] | None = None,
) -> list[MovingObstacle]:
    forbidden = forbidden_positions or set()
    rng = random.Random(config.seed)

    max_row = config.max_row if config.max_row is not None else grid_map.height - 1
    max_col = config.max_col if config.max_col is not None else grid_map.width - 1

    obstacles: list[MovingObstacle] = []
    max_attempts = config.obstacle_count * 100

    for _ in range(max_attempts):
        if len(obstacles) >= config.obstacle_count:
            break

        width = rng.randint(config.min_width, config.max_width)
        height = rng.randint(config.min_height, config.max_height)

        row_min = config.min_row
        row_max = max_row - height + 1
        col_min = config.min_col
        col_max = max_col - width + 1

        if row_max < row_min or col_max < col_min:
            continue

        row = rng.randint(row_min, row_max)
        col = rng.randint(col_min, col_max)

        if not _rectangle_is_valid(
                grid_map=grid_map,
                row=row,
                col=col,
                width=width,
                height=height,
                forbidden_positions=forbidden,
        ):
            continue

        delta_row, delta_col = _pick_direction(
            rng=rng,
            allow_vertical=config.allow_vertical,
            allow_horizontal=config.allow_horizontal,
        )

        obstacles.append(
            MovingObstacle(
                row=row,
                col=col,
                width=width,
                height=height,
                delta_row=delta_row,
                delta_col=delta_col,
            )
        )

    return obstacles
