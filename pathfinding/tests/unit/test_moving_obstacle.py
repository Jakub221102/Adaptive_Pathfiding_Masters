from unittest.mock import patch

from pathfinding.src.algorithms.dstar_lite import DStarLite
from pathfinding.src.core.dynamic_models import (
    DynamicGridMap,
    MovingObstacle,
    MovingObstacleCollisionPolicy,
    get_occupied_positions,
    move_obstacle,
    move_obstacle_with_agent_collision,
)
from pathfinding.src.core.models import Position
from pathfinding.tests.helpers import build_grid_map


def test_moving_obstacle_moves_in_delta_direction() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    dynamic_map = DynamicGridMap(base_map=grid_map)
    obstacle = MovingObstacle(
        row=1,
        col=1,
        width=2,
        height=2,
        delta_row=1,
        delta_col=0,
    )

    dynamic_map.block_rectangle(
        position=Position(row=obstacle.row, col=obstacle.col),
        width=obstacle.width,
        height=obstacle.height,
    )

    move_obstacle(obstacle=obstacle, dynamic_map=dynamic_map)

    assert obstacle.row == 2
    assert obstacle.col == 1
    assert dynamic_map.is_walkable(Position(row=1, col=1)) is True
    assert dynamic_map.is_walkable(Position(row=2, col=1)) is False
    assert dynamic_map.is_walkable(Position(row=3, col=2)) is False


def test_moving_obstacle_bounces_at_map_boundary() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    dynamic_map = DynamicGridMap(base_map=grid_map)
    obstacle = MovingObstacle(
        row=0,
        col=1,
        width=1,
        height=1,
        delta_row=-1,
        delta_col=0,
    )

    dynamic_map.block_cell(Position(row=0, col=1))

    move_obstacle(obstacle=obstacle, dynamic_map=dynamic_map)

    assert obstacle.row == 1
    assert obstacle.delta_row == 1
    assert dynamic_map.is_walkable(Position(row=0, col=1)) is True
    assert dynamic_map.is_walkable(Position(row=1, col=1)) is False


def test_moving_obstacle_blocks_and_unblocks_cells() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    dynamic_map = DynamicGridMap(base_map=grid_map)
    obstacle = MovingObstacle(
        row=1,
        col=0,
        width=1,
        height=1,
        delta_row=0,
        delta_col=1,
    )

    old_positions = get_occupied_positions(obstacle)
    dynamic_map.block_rectangle(
        position=Position(row=obstacle.row, col=obstacle.col),
        width=obstacle.width,
        height=obstacle.height,
    )

    for position in old_positions:
        assert dynamic_map.is_walkable(position) is False

    affected_positions = move_obstacle(obstacle=obstacle, dynamic_map=dynamic_map)

    for position in old_positions:
        assert dynamic_map.is_walkable(position) is True

    new_positions = get_occupied_positions(obstacle)
    for position in new_positions:
        assert dynamic_map.is_walkable(position) is False

    affected_keys = {(p.row, p.col) for p in affected_positions}
    expected_keys = {
        (p.row, p.col) for p in old_positions + new_positions
    }
    assert affected_keys == expected_keys


def test_dstar_lite_receives_update_cells_after_obstacle_move() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    dynamic_map = DynamicGridMap(base_map=grid_map)
    obstacle = MovingObstacle(
        row=1,
        col=1,
        width=1,
        height=1,
        delta_row=0,
        delta_col=1,
    )

    dstar = DStarLite()
    dstar.find_path(
        grid_map=dynamic_map,
        start=Position(row=0, col=0),
        goal=Position(row=0, col=4),
    )

    dynamic_map.block_cell(Position(row=1, col=1))

    with patch.object(dstar, "update_cells") as update_cells_mock:
        affected_positions = move_obstacle(
            obstacle=obstacle,
            dynamic_map=dynamic_map,
        )
        dstar.update_cells(affected_positions)

    update_cells_mock.assert_called_once()
    called_positions = update_cells_mock.call_args[0][0]
    called_keys = {(p.row, p.col) for p in called_positions}
    affected_keys = {(p.row, p.col) for p in affected_positions}
    assert called_keys == affected_keys


def test_moving_obstacle_pushes_agent_when_collision_possible() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    dynamic_map = DynamicGridMap(base_map=grid_map)
    agent_position = Position(row=0, col=1)
    obstacle = MovingObstacle(
        row=0,
        col=2,
        width=1,
        height=1,
        delta_row=0,
        delta_col=-1,
    )

    dynamic_map.block_cell(Position(row=obstacle.row, col=obstacle.col))

    result = move_obstacle_with_agent_collision(
        obstacle=obstacle,
        dynamic_map=dynamic_map,
        agent_position=agent_position,
        policy=MovingObstacleCollisionPolicy.PUSH_AGENT,
    )

    assert result.collision_detected is True
    assert result.agent_pushed is True
    assert result.new_agent_position is not None
    assert result.new_agent_position != agent_position


def test_moving_obstacle_does_not_push_agent_into_wall() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    dynamic_map = DynamicGridMap(base_map=grid_map)
    agent_position = Position(row=1, col=2)
    wall_position = Position(row=1, col=1)
    obstacle = MovingObstacle(
        row=1,
        col=3,
        width=1,
        height=1,
        delta_row=0,
        delta_col=-1,
    )

    dynamic_map.block_cell(Position(row=obstacle.row, col=obstacle.col))

    result = move_obstacle_with_agent_collision(
        obstacle=obstacle,
        dynamic_map=dynamic_map,
        agent_position=agent_position,
        policy=MovingObstacleCollisionPolicy.PUSH_AGENT,
    )

    assert result.agent_pushed is True
    assert result.new_agent_position is not None
    assert result.new_agent_position != wall_position


def test_moving_obstacle_blocks_when_agent_cannot_be_pushed() -> None:
    grid_map = build_grid_map(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
    )
    dynamic_map = DynamicGridMap(base_map=grid_map)
    agent_position = Position(row=1, col=1)
    obstacle = MovingObstacle(
        row=2,
        col=1,
        width=1,
        height=1,
        delta_row=-1,
        delta_col=0,
    )

    dynamic_map.block_cell(Position(row=obstacle.row, col=obstacle.col))

    result = move_obstacle_with_agent_collision(
        obstacle=obstacle,
        dynamic_map=dynamic_map,
        agent_position=agent_position,
        policy=MovingObstacleCollisionPolicy.PUSH_AGENT,
    )

    assert result.collision_detected is True
    assert result.agent_pushed is False
    assert result.obstacle_blocked is True
    assert obstacle.row == 2
    assert obstacle.col == 1
