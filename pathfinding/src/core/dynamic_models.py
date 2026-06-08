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

    def clear_dynamic_block(self, position: Position) -> None:
        key = (position.row, position.col)
        self.dynamic_blocked.discard(key)

    def clear_dynamic_block_rectangle(
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

            self.clear_dynamic_block(cell_position)
            affected_positions.append(cell_position)

        return affected_positions

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


class MovingObstacleCollisionPolicy(str, Enum):
    PUSH_AGENT = "push_agent"
    BLOCK_OBSTACLE = "block_obstacle"


class DynamicReplanningStats(BaseModel):
    algorithm_name: str

    initial_path_found: bool
    final_goal_reached: bool

    replanning_count: int
    total_execution_time_ms: float

    total_path_cost: float
    travelled_steps: int

    dynamic_events_applied: int
    waiting_steps: int = 0
    agent_push_count: int = 0
    obstacle_blocked_count: int = 0
    collision_count: int = 0


def is_path_blocked(
        grid_map: DynamicGridMap,
        path: list[Position],
        current_index: int,
) -> bool:
    for position in path[current_index:]:
        if not grid_map.is_walkable(position):
            return True

    return False


def is_path_blocked_with_lookahead(
        grid_map: DynamicGridMap,
        path: list[Position],
        current_index: int,
        lookahead_steps: int,
) -> bool:
    end_index = min(current_index + lookahead_steps, len(path))

    for position in path[current_index:end_index]:
        if not grid_map.is_walkable(position):
            return True

    return False


def validate_path_walkable(
        grid_map: DynamicGridMap,
        path: list[Position],
) -> bool:
    return all(
        grid_map.in_bounds(position) and grid_map.is_walkable(position)
        for position in path
    )


class MovingObstacle(BaseModel):
    row: int
    col: int

    width: int = 1
    height: int = 1

    delta_row: int
    delta_col: int


def get_occupied_positions(obstacle: MovingObstacle) -> list[Position]:
    return iter_rectangle_positions(
        position=Position(row=obstacle.row, col=obstacle.col),
        width=obstacle.width,
        height=obstacle.height,
    )


def predict_moving_obstacle_positions(
        obstacles: list[MovingObstacle],
        dynamic_map: DynamicGridMap,
        steps: int,
) -> set[tuple[int, int]]:
    if steps <= 0:
        return set()

    simulated = [
        (
            obstacle.row,
            obstacle.col,
            obstacle.width,
            obstacle.height,
            obstacle.delta_row,
            obstacle.delta_col,
        )
        for obstacle in obstacles
    ]
    predicted: set[tuple[int, int]] = set()

    for _ in range(steps):
        next_simulated: list[tuple[int, int, int, int, int, int]] = []

        for row, col, width, height, delta_row, delta_col in simulated:
            new_row = row + delta_row
            new_col = col + delta_col
            new_delta_row = delta_row
            new_delta_col = delta_col

            if not _obstacle_can_occupy(
                    dynamic_map=dynamic_map,
                    row=new_row,
                    col=new_col,
                    width=width,
                    height=height,
            ):
                new_delta_row = -delta_row
                new_delta_col = -delta_col
                new_row = row + new_delta_row
                new_col = col + new_delta_col

            for cell_row in range(new_row, new_row + height):
                for cell_col in range(new_col, new_col + width):
                    predicted.add((cell_row, cell_col))

            next_simulated.append(
                (new_row, new_col, width, height, new_delta_row, new_delta_col)
            )

        simulated = next_simulated

    return predicted


def _obstacle_fits(
        dynamic_map: DynamicGridMap,
        row: int,
        col: int,
        width: int,
        height: int,
) -> bool:
    if row < 0 or col < 0:
        return False

    if row + height > dynamic_map.height:
        return False

    if col + width > dynamic_map.width:
        return False

    return True


def _obstacle_can_occupy(
        dynamic_map: DynamicGridMap,
        row: int,
        col: int,
        width: int,
        height: int,
) -> bool:
    if row < 0 or col < 0:
        return False

    if row + height > dynamic_map.height:
        return False

    if col + width > dynamic_map.width:
        return False

    for cell_row in range(row, row + height):
        for cell_col in range(col, col + width):
            position = Position(row=cell_row, col=cell_col)
            if not dynamic_map.base_map.is_walkable(position):
                return False

    return True


def move_obstacle(
        obstacle: MovingObstacle,
        dynamic_map: DynamicGridMap,
) -> list[Position]:
    old_positions = get_occupied_positions(obstacle)

    for position in old_positions:
        if dynamic_map.in_bounds(position):
            dynamic_map.clear_dynamic_block(position)

    new_row = obstacle.row + obstacle.delta_row
    new_col = obstacle.col + obstacle.delta_col

    if not _obstacle_can_occupy(
            dynamic_map=dynamic_map,
            row=new_row,
            col=new_col,
            width=obstacle.width,
            height=obstacle.height,
    ):
        obstacle.delta_row = -obstacle.delta_row
        obstacle.delta_col = -obstacle.delta_col
        new_row = obstacle.row + obstacle.delta_row
        new_col = obstacle.col + obstacle.delta_col

    obstacle.row = new_row
    obstacle.col = new_col

    new_positions = dynamic_map.block_rectangle(
        position=Position(row=obstacle.row, col=obstacle.col),
        width=obstacle.width,
        height=obstacle.height,
    )

    return old_positions + new_positions


class MovingObstacleMoveResult(BaseModel):
    affected_positions: list[Position]
    obstacle_moved: bool
    collision_detected: bool = False
    agent_pushed: bool = False
    obstacle_blocked: bool = False
    new_agent_position: Position | None = None


def _position_key(position: Position) -> tuple[int, int]:
    return position.row, position.col


def _is_safe_push_position(
        dynamic_map: DynamicGridMap,
        position: Position,
        obstacle_old_positions: list[Position],
        obstacle_new_positions: list[Position],
) -> bool:
    if not dynamic_map.in_bounds(position):
        return False

    position_key = _position_key(position)
    old_keys = {_position_key(p) for p in obstacle_old_positions}
    new_keys = {_position_key(p) for p in obstacle_new_positions}

    if position_key in old_keys or position_key in new_keys:
        return False

    return dynamic_map.is_walkable(position)


def _push_candidate_offsets(push_direction: tuple[int, int]) -> list[tuple[int, int]]:
    delta_row, delta_col = push_direction
    candidates: list[tuple[int, int]] = [(delta_row, delta_col)]

    if delta_row != 0 and delta_col == 0:
        if delta_row > 0:
            candidates.extend([(0, -1), (0, 1), (-1, 0)])
        else:
            candidates.extend([(0, 1), (0, -1), (1, 0)])
    elif delta_col != 0 and delta_row == 0:
        if delta_col > 0:
            candidates.extend([(-1, 0), (1, 0), (0, -1)])
        else:
            candidates.extend([(-1, 0), (1, 0), (0, 1)])
    else:
        perpendicular = [
            (delta_col, -delta_row),
            (-delta_col, delta_row),
            (-delta_row, -delta_col),
        ]
        for offset in perpendicular:
            if offset not in candidates:
                candidates.append(offset)

    return candidates


def find_safe_push_position(
        dynamic_map: DynamicGridMap,
        agent_position: Position,
        obstacle_old_positions: list[Position],
        obstacle_new_positions: list[Position],
        push_direction: tuple[int, int],
) -> Position | None:
    for row_offset, col_offset in _push_candidate_offsets(push_direction):
        candidate = Position(
            row=agent_position.row + row_offset,
            col=agent_position.col + col_offset,
        )
        if _is_safe_push_position(
                dynamic_map=dynamic_map,
                position=candidate,
                obstacle_old_positions=obstacle_old_positions,
                obstacle_new_positions=obstacle_new_positions,
        ):
            return candidate

    return None


def _compute_candidate_position(
        obstacle: MovingObstacle,
        dynamic_map: DynamicGridMap,
) -> tuple[int, int, int, int, bool]:
    delta_row = obstacle.delta_row
    delta_col = obstacle.delta_col
    bounced = False

    new_row = obstacle.row + delta_row
    new_col = obstacle.col + delta_col

    if not _obstacle_can_occupy(
            dynamic_map=dynamic_map,
            row=new_row,
            col=new_col,
            width=obstacle.width,
            height=obstacle.height,
    ):
        delta_row = -delta_row
        delta_col = -delta_col
        new_row = obstacle.row + delta_row
        new_col = obstacle.col + delta_col
        bounced = True

    return new_row, new_col, delta_row, delta_col, bounced


def _rectangle_positions(
        row: int,
        col: int,
        width: int,
        height: int,
) -> list[Position]:
    return iter_rectangle_positions(
        position=Position(row=row, col=col),
        width=width,
        height=height,
    )


def _apply_obstacle_move(
        obstacle: MovingObstacle,
        dynamic_map: DynamicGridMap,
        old_positions: list[Position],
        new_row: int,
        new_col: int,
        delta_row: int,
        delta_col: int,
) -> list[Position]:
    for position in old_positions:
        if dynamic_map.in_bounds(position):
            dynamic_map.clear_dynamic_block(position)

    obstacle.row = new_row
    obstacle.col = new_col
    obstacle.delta_row = delta_row
    obstacle.delta_col = delta_col

    new_positions = dynamic_map.block_rectangle(
        position=Position(row=obstacle.row, col=obstacle.col),
        width=obstacle.width,
        height=obstacle.height,
    )

    return old_positions + new_positions


def _block_obstacle_with_bounce(
        obstacle: MovingObstacle,
        old_positions: list[Position],
        delta_row: int,
        delta_col: int,
) -> list[Position]:
    obstacle.delta_row = delta_row
    obstacle.delta_col = delta_col
    return list(old_positions)


def move_obstacle_with_agent_collision(
        obstacle: MovingObstacle,
        dynamic_map: DynamicGridMap,
        agent_position: Position,
        policy: MovingObstacleCollisionPolicy,
) -> MovingObstacleMoveResult:
    old_positions = get_occupied_positions(obstacle)

    new_row, new_col, delta_row, delta_col, _bounced = _compute_candidate_position(
        obstacle=obstacle,
        dynamic_map=dynamic_map,
    )
    new_positions = _rectangle_positions(
        row=new_row,
        col=new_col,
        width=obstacle.width,
        height=obstacle.height,
    )
    new_position_keys = {_position_key(position) for position in new_positions}
    agent_key = _position_key(agent_position)

    if agent_key not in new_position_keys:
        affected_positions = _apply_obstacle_move(
            obstacle=obstacle,
            dynamic_map=dynamic_map,
            old_positions=old_positions,
            new_row=new_row,
            new_col=new_col,
            delta_row=delta_row,
            delta_col=delta_col,
        )
        return MovingObstacleMoveResult(
            affected_positions=affected_positions,
            obstacle_moved=True,
        )

    if policy == MovingObstacleCollisionPolicy.BLOCK_OBSTACLE:
        affected_positions = _block_obstacle_with_bounce(
            obstacle=obstacle,
            old_positions=old_positions,
            delta_row=delta_row,
            delta_col=delta_col,
        )
        return MovingObstacleMoveResult(
            affected_positions=affected_positions,
            obstacle_moved=False,
            collision_detected=True,
            obstacle_blocked=True,
        )

    push_position = find_safe_push_position(
        dynamic_map=dynamic_map,
        agent_position=agent_position,
        obstacle_old_positions=old_positions,
        obstacle_new_positions=new_positions,
        push_direction=(delta_row, delta_col),
    )

    if push_position is None:
        affected_positions = _block_obstacle_with_bounce(
            obstacle=obstacle,
            old_positions=old_positions,
            delta_row=delta_row,
            delta_col=delta_col,
        )
        return MovingObstacleMoveResult(
            affected_positions=affected_positions,
            obstacle_moved=False,
            collision_detected=True,
            obstacle_blocked=True,
        )

    affected_positions = _apply_obstacle_move(
        obstacle=obstacle,
        dynamic_map=dynamic_map,
        old_positions=old_positions,
        new_row=new_row,
        new_col=new_col,
        delta_row=delta_row,
        delta_col=delta_col,
    )
    affected_positions.append(agent_position)
    affected_positions.append(push_position)

    return MovingObstacleMoveResult(
        affected_positions=affected_positions,
        obstacle_moved=True,
        collision_detected=True,
        agent_pushed=True,
        new_agent_position=push_position,
    )
