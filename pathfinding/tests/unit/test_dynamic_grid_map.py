from pathfinding.src.core.dynamic_models import (
    DynamicGridMap,
    DynamicObstacleEvent,
    DynamicObstacleEventType,
    apply_dynamic_event,
)
from pathfinding.src.core.models import Position
from pathfinding.tests.helpers import build_grid_map


def test_free_cell_in_base_map_is_walkable() -> None:
    grid_map = build_grid_map(
        [
            [0, 0],
            [0, 0],
        ]
    )
    dynamic_map = DynamicGridMap(base_map=grid_map)

    assert dynamic_map.is_walkable(Position(row=0, col=0)) is True
    assert dynamic_map.is_walkable(Position(row=1, col=1)) is True


def test_block_cell_blocks_walkable_cell() -> None:
    grid_map = build_grid_map([[0]])
    dynamic_map = DynamicGridMap(base_map=grid_map)
    position = Position(row=0, col=0)

    dynamic_map.block_cell(position)

    assert dynamic_map.is_walkable(position) is False


def test_unblock_cell_unblocks_blocked_cell() -> None:
    grid_map = build_grid_map([[0]])
    dynamic_map = DynamicGridMap(base_map=grid_map)
    position = Position(row=0, col=0)

    dynamic_map.block_cell(position)
    dynamic_map.unblock_cell(position)

    assert dynamic_map.is_walkable(position) is True


def test_reset_dynamic_changes_clears_modifications() -> None:
    grid_map = build_grid_map(
        [
            [0, 1],
        ]
    )
    dynamic_map = DynamicGridMap(base_map=grid_map)
    free_position = Position(row=0, col=0)
    wall_position = Position(row=0, col=1)

    dynamic_map.block_cell(free_position)
    dynamic_map.unblock_cell(wall_position)
    dynamic_map.reset_dynamic_changes()

    assert dynamic_map.is_walkable(free_position) is True
    assert dynamic_map.is_walkable(wall_position) is False


def test_dynamic_block_overrides_base_walkable_cell() -> None:
    grid_map = build_grid_map([[0]])
    dynamic_map = DynamicGridMap(base_map=grid_map)
    position = Position(row=0, col=0)

    dynamic_map.block_cell(position)

    assert grid_map.is_walkable(position) is True
    assert dynamic_map.is_walkable(position) is False


def test_dynamic_unblock_can_open_base_wall() -> None:
    grid_map = build_grid_map([[1]])
    dynamic_map = DynamicGridMap(base_map=grid_map)
    position = Position(row=0, col=0)

    dynamic_map.unblock_cell(position)

    assert grid_map.is_walkable(position) is False
    assert dynamic_map.is_walkable(position) is True


def test_block_rectangle_blocks_multiple_cells() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    dynamic_map = DynamicGridMap(base_map=grid_map)

    affected_positions = dynamic_map.block_rectangle(
        position=Position(row=0, col=0),
        width=2,
        height=2,
    )

    assert len(affected_positions) == 4
    assert dynamic_map.is_walkable(Position(row=0, col=0)) is False
    assert dynamic_map.is_walkable(Position(row=1, col=1)) is False
    assert dynamic_map.is_walkable(Position(row=0, col=2)) is True


def test_apply_dynamic_event_supports_rectangle_block() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    dynamic_map = DynamicGridMap(base_map=grid_map)

    affected_positions = apply_dynamic_event(
        dynamic_map=dynamic_map,
        event=DynamicObstacleEvent(
            step_index=0,
            position=Position(row=0, col=1),
            event_type=DynamicObstacleEventType.BLOCK,
            width=2,
            height=1,
        ),
    )

    assert len(affected_positions) == 2
    assert dynamic_map.is_walkable(Position(row=0, col=1)) is False
    assert dynamic_map.is_walkable(Position(row=0, col=2)) is False
