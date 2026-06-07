from enum import Enum

from pydantic import BaseModel, Field

from pathfinding.src.core.models import GridMap, Position


class DynamicGridMap(BaseModel):
    base_map: GridMap
    dynamic_blocked: set[tuple[int, int]] = Field(default_factory=set)
    dynamic_unblocked: set[tuple[int, int]] = Field(default_factory=set)

    def is_walkable(self, position: Position) -> bool:
        key = (position.row, position.col)

        if key in self.dynamic_blocked:
            return False

        if key in self.dynamic_unblocked:
            return True

        return self.base_map.is_walkable(position)

    def in_bounds(self, position: Position) -> bool:
        return self.base_map.in_bounds(position)

    @property
    def name(self) -> str:
        return self.base_map.name

    @property
    def width(self) -> int:
        return self.base_map.width

    @property
    def height(self) -> int:
        return self.base_map.height

    @property
    def cells(self) -> list[list[int]]:
        return self.base_map.cells

    def block_cell(self, position: Position) -> None:
        key = (position.row, position.col)
        self.dynamic_unblocked.discard(key)
        self.dynamic_blocked.add(key)

    def unblock_cell(self, position: Position) -> None:
        key = (position.row, position.col)
        self.dynamic_blocked.discard(key)
        self.dynamic_unblocked.add(key)

    def reset_dynamic_changes(self) -> None:
        self.dynamic_blocked.clear()
        self.dynamic_unblocked.clear()

    def block_rectangle(
            self,
            position: Position,
            width: int = 1,
            height: int = 1,
    ) -> list[Position]:
        affected_positions: list[Position] = []

        for cell_position in iter_rectangle_positions(
                position=position,
                width=width,
                height=height,
        ):
            if not self.in_bounds(cell_position):
                continue

            self.block_cell(cell_position)
            affected_positions.append(cell_position)

        return affected_positions

    def unblock_rectangle(
            self,
            position: Position,
            width: int = 1,
            height: int = 1,
    ) -> list[Position]:
        affected_positions: list[Position] = []

        for cell_position in iter_rectangle_positions(
                position=position,
                width=width,
                height=height,
        ):
            if not self.in_bounds(cell_position):
                continue

            self.unblock_cell(cell_position)
            affected_positions.append(cell_position)

        return affected_positions


class DynamicObstacleEventType(str, Enum):
    BLOCK = "block"
    UNBLOCK = "unblock"


class DynamicObstacleEvent(BaseModel):
    step_index: int
    position: Position
    event_type: DynamicObstacleEventType
    width: int = Field(default=1, ge=1)
    height: int = Field(default=1, ge=1)


def iter_rectangle_positions(
        position: Position,
        width: int,
        height: int,
) -> list[Position]:
    positions: list[Position] = []

    for row in range(position.row, position.row + height):
        for col in range(position.col, position.col + width):
            positions.append(Position(row=row, col=col))

    return positions


def apply_dynamic_event(
        dynamic_map: DynamicGridMap,
        event: DynamicObstacleEvent,
) -> list[Position]:
    if event.event_type == DynamicObstacleEventType.BLOCK:
        return dynamic_map.block_rectangle(
            position=event.position,
            width=event.width,
            height=event.height,
        )

    return dynamic_map.unblock_rectangle(
        position=event.position,
        width=event.width,
        height=event.height,
    )


class DynamicReplanningStats(BaseModel):
    algorithm_name: str

    initial_path_found: bool
    final_goal_reached: bool

    replanning_count: int
    total_execution_time_ms: float

    total_path_cost: float
    travelled_steps: int

    dynamic_events_applied: int


def is_path_blocked(
        grid_map: DynamicGridMap,
        path: list[Position],
        current_index: int,
) -> bool:
    for position in path[current_index:]:
        if not grid_map.is_walkable(position):
            return True

    return False
