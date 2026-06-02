import heapq
import math
import time

from code.src.algorithms.base import PathfindingAlgorithm
from code.src.core.models import GridMap, PathfindingResult, Position
from code.src.core.trace import GridNode, RawAlgorithmStep

Direction = tuple[int, int]


class JumpPointSearch(PathfindingAlgorithm):
    name = "JPS"

    DIRECTIONS: tuple[Direction, ...] = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )

    def find_path(
            self,
            grid_map: GridMap,
            start: Position,
            goal: Position,
    ) -> PathfindingResult:
        result, _ = self._run_jps(
            grid_map=grid_map,
            start=start,
            goal=goal,
            record_steps=False,
        )

        return result

    def find_path_with_steps(
            self,
            grid_map: GridMap,
            start: Position,
            goal: Position,
            step_record_interval: int = 10,
    ) -> tuple[PathfindingResult, list[RawAlgorithmStep]]:
        return self._run_jps(
            grid_map=grid_map,
            start=start,
            goal=goal,
            record_steps=True,
            step_record_interval=step_record_interval,
        )

    def _run_jps(
            self,
            grid_map: GridMap,
            start: Position,
            goal: Position,
            record_steps: bool,
            step_record_interval: int = 10,
    ) -> tuple[PathfindingResult, list[RawAlgorithmStep]]:
        start_time = time.perf_counter()
        self.scanned_nodes = 0

        start_node = (start.row, start.col)
        goal_node = (goal.row, goal.col)

        open_heap: list[tuple[float, int, GridNode]] = []
        open_lookup: set[GridNode] = {start_node}
        closed_lookup: set[GridNode] = set()

        came_from: dict[GridNode, GridNode] = {}
        g_score: dict[GridNode, float] = {
            start_node: 0.0,
        }

        counter = 0
        heapq.heappush(open_heap, (0.0, counter, start_node))

        visited_nodes = 0
        steps: list[RawAlgorithmStep] = []

        while open_heap:
            _, _, current = heapq.heappop(open_heap)

            if current not in open_lookup:
                continue

            open_lookup.remove(current)
            closed_lookup.add(current)
            visited_nodes += 1

            if record_steps and visited_nodes % step_record_interval == 0:
                steps.append(
                    self._build_step(
                        current=current,
                        open_nodes=open_lookup,
                        closed_nodes=closed_lookup,
                    )
                )

            if current == goal_node:
                jump_path = self._reconstruct_path(
                    came_from=came_from,
                    current=current,
                )

                full_path = self._expand_jump_path(jump_path)

                result = self._build_result(
                    path=full_path,
                    found=True,
                    visited_nodes=visited_nodes,
                    start_time=start_time,
                )

                if record_steps:
                    steps.append(
                        self._build_step(
                            current=current,
                            open_nodes=open_lookup,
                            closed_nodes=closed_lookup,
                            path=full_path,
                        )
                    )

                return result, steps

            for jump_point in self._identify_successors(
                    grid_map=grid_map,
                    current=current,
                    goal=goal_node,
                    came_from=came_from,
            ):
                if jump_point in closed_lookup:
                    continue

                tentative_g = (
                        g_score[current]
                        + self._distance(current, jump_point)
                )

                if jump_point not in g_score or tentative_g < g_score[jump_point]:
                    came_from[jump_point] = current
                    g_score[jump_point] = tentative_g

                    counter += 1

                    f_score = tentative_g + self._heuristic(
                        current=jump_point,
                        goal=goal_node,
                    )

                    heapq.heappush(
                        open_heap,
                        (f_score, counter, jump_point),
                    )

                    open_lookup.add(jump_point)

        result = self._build_result(
            path=[],
            found=False,
            visited_nodes=visited_nodes,
            start_time=start_time,
        )

        return result, steps

    def _identify_successors(
            self,
            grid_map: GridMap,
            current: GridNode,
            goal: GridNode,
            came_from: dict[GridNode, GridNode],
    ) -> list[GridNode]:
        successors: list[GridNode] = []

        for direction in self._get_pruned_directions(
                grid_map=grid_map,
                current=current,
                came_from=came_from,
        ):
            jump_point = self._jump(
                grid_map=grid_map,
                current=current,
                direction=direction,
                goal=goal,
            )

            if jump_point is not None:
                successors.append(jump_point)

        return successors

    def _jump(
            self,
            grid_map: GridMap,
            current: GridNode,
            direction: Direction,
            goal: GridNode,
    ) -> GridNode | None:
        row_direction, col_direction = direction
        node = current

        while True:
            if not self._can_move(
                    grid_map=grid_map,
                    current=node,
                    direction=direction,
            ):
                return None

            next_node = (
                node[0] + row_direction,
                node[1] + col_direction,
            )

            self.scanned_nodes += 1

            if next_node == goal:
                return next_node

            if self._has_forced_neighbor(
                    grid_map=grid_map,
                    node=next_node,
                    direction=direction,
            ):
                return next_node

            if self._is_diagonal(direction):
                horizontal_direction = (0, col_direction)
                vertical_direction = (row_direction, 0)

                if (
                        self._jump_straight(
                            grid_map=grid_map,
                            current=next_node,
                            direction=horizontal_direction,
                            goal=goal,
                        )
                        is not None
                ):
                    return next_node

                if (
                        self._jump_straight(
                            grid_map=grid_map,
                            current=next_node,
                            direction=vertical_direction,
                            goal=goal,
                        )
                        is not None
                ):
                    return next_node

            node = next_node

    def _get_pruned_directions(
            self,
            grid_map: GridMap,
            current: GridNode,
            came_from: dict[GridNode, GridNode],
    ) -> list[Direction]:
        if current not in came_from:
            return [
                direction
                for direction in self.DIRECTIONS
                if self._can_move(grid_map, current, direction)
            ]

        parent = came_from[current]
        row_direction, col_direction = self._normalize_direction(current, parent)

        candidates: list[Direction] = []

        if row_direction != 0 and col_direction != 0:
            candidates.extend([
                (row_direction, col_direction),
                (row_direction, 0),
                (0, col_direction),
            ])

            if not self._is_walkable_at(grid_map, current[0] - row_direction, current[1]):
                candidates.append((-row_direction, col_direction))

            if not self._is_walkable_at(grid_map, current[0], current[1] - col_direction):
                candidates.append((row_direction, -col_direction))

        elif row_direction != 0:
            candidates.append((row_direction, 0))

            if not self._is_walkable_at(grid_map, current[0], current[1] - 1):
                candidates.append((row_direction, -1))

            if not self._is_walkable_at(grid_map, current[0], current[1] + 1):
                candidates.append((row_direction, 1))

        elif col_direction != 0:
            candidates.append((0, col_direction))

            if not self._is_walkable_at(grid_map, current[0] - 1, current[1]):
                candidates.append((-1, col_direction))

            if not self._is_walkable_at(grid_map, current[0] + 1, current[1]):
                candidates.append((1, col_direction))

        result: list[Direction] = []
        seen: set[Direction] = set()

        for direction in candidates:
            if direction in seen:
                continue

            if self._can_move(grid_map, current, direction):
                result.append(direction)
                seen.add(direction)

        return result

    def _jump_straight(
            self,
            grid_map: GridMap,
            current: GridNode,
            direction: Direction,
            goal: GridNode,
    ) -> GridNode | None:
        node = current

        while True:
            if not self._can_move(
                    grid_map=grid_map,
                    current=node,
                    direction=direction,
            ):
                return None

            next_node = (
                node[0] + direction[0],
                node[1] + direction[1],
            )

            self.scanned_nodes += 1

            if next_node == goal:
                return next_node

            if self._has_forced_neighbor(
                    grid_map=grid_map,
                    node=next_node,
                    direction=direction,
            ):
                return next_node

            node = next_node

    def _has_forced_neighbor(
            self,
            grid_map: GridMap,
            node: GridNode,
            direction: Direction,
    ) -> bool:
        row, col = node
        row_direction, col_direction = direction

        if self._is_diagonal(direction):
            return (
                    not self._is_walkable_at(
                        grid_map,
                        row - row_direction,
                        col,
                    )
                    and self._is_walkable_at(
                grid_map,
                row - row_direction,
                col + col_direction,
            )
            ) or (
                    not self._is_walkable_at(
                        grid_map,
                        row,
                        col - col_direction,
                    )
                    and self._is_walkable_at(
                grid_map,
                row + row_direction,
                col - col_direction,
            )
            )

        if row_direction != 0:
            return (
                    not self._is_walkable_at(grid_map, row, col - 1)
                    and self._is_walkable_at(
                grid_map,
                row + row_direction,
                col - 1,
            )
            ) or (
                    not self._is_walkable_at(grid_map, row, col + 1)
                    and self._is_walkable_at(
                grid_map,
                row + row_direction,
                col + 1,
            )
            )

        if col_direction != 0:
            return (
                    not self._is_walkable_at(grid_map, row - 1, col)
                    and self._is_walkable_at(
                grid_map,
                row - 1,
                col + col_direction,
            )
            ) or (
                    not self._is_walkable_at(grid_map, row + 1, col)
                    and self._is_walkable_at(
                grid_map,
                row + 1,
                col + col_direction,
            )
            )

        return False

    def _can_move(
            self,
            grid_map: GridMap,
            current: GridNode,
            direction: Direction,
    ) -> bool:
        next_node = (
            current[0] + direction[0],
            current[1] + direction[1],
        )

        return self._is_walkable_at(
            grid_map=grid_map,
            row=next_node[0],
            col=next_node[1],
        )

    @staticmethod
    def _is_walkable_at(
            grid_map: GridMap,
            row: int,
            col: int,
    ) -> bool:
        if row < 0 or col < 0:
            return False

        position = Position(row=row, col=col)

        if not grid_map.in_bounds(position):
            return False

        return grid_map.is_walkable(position)

    @staticmethod
    def _normalize_direction(
            current: GridNode,
            parent: GridNode,
    ) -> Direction:
        row_diff = current[0] - parent[0]
        col_diff = current[1] - parent[1]

        row_direction = 0
        col_direction = 0

        if row_diff > 0:
            row_direction = 1
        elif row_diff < 0:
            row_direction = -1

        if col_diff > 0:
            col_direction = 1
        elif col_diff < 0:
            col_direction = -1

        return row_direction, col_direction

    @staticmethod
    def _is_diagonal(
            direction: Direction,
    ) -> bool:
        return direction[0] != 0 and direction[1] != 0

    def _expand_jump_path(
            self,
            jump_path: list[GridNode],
    ) -> list[Position]:
        if len(jump_path) < 2:
            return [
                Position(row=node[0], col=node[1])
                for node in jump_path
            ]

        full_path: list[Position] = []

        for index in range(len(jump_path) - 1):
            segment = self._expand_segment(
                start=jump_path[index],
                end=jump_path[index + 1],
            )

            if full_path:
                segment = segment[1:]

            full_path.extend(segment)

        return full_path

    @staticmethod
    def _expand_segment(
            start: GridNode,
            end: GridNode,
    ) -> list[Position]:
        path: list[Position] = []

        row_direction = 0
        col_direction = 0

        if end[0] > start[0]:
            row_direction = 1
        elif end[0] < start[0]:
            row_direction = -1

        if end[1] > start[1]:
            col_direction = 1
        elif end[1] < start[1]:
            col_direction = -1

        current = start

        path.append(Position(row=current[0], col=current[1]))

        while current != end:
            current = (
                current[0] + row_direction,
                current[1] + col_direction,
            )

            path.append(Position(row=current[0], col=current[1]))

        return path

    @staticmethod
    def _reconstruct_path(
            came_from: dict[GridNode, GridNode],
            current: GridNode,
    ) -> list[GridNode]:
        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()
        return path

    @staticmethod
    def _heuristic(
            current: GridNode,
            goal: GridNode,
    ) -> float:
        dx = abs(current[0] - goal[0])
        dy = abs(current[1] - goal[1])

        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    @staticmethod
    def _distance(
            first: GridNode,
            second: GridNode,
    ) -> float:
        row_diff = abs(first[0] - second[0])
        col_diff = abs(first[1] - second[1])

        diagonal_steps = min(row_diff, col_diff)
        straight_steps = max(row_diff, col_diff) - diagonal_steps

        return diagonal_steps * math.sqrt(2) + straight_steps

    @staticmethod
    def _calculate_path_cost(
            path: list[Position],
    ) -> float:
        if len(path) < 2:
            return 0.0

        cost = 0.0

        for index in range(len(path) - 1):
            current = path[index]
            next_position = path[index + 1]

            row_diff = abs(current.row - next_position.row)
            col_diff = abs(current.col - next_position.col)

            if row_diff == 1 and col_diff == 1:
                cost += math.sqrt(2)
            else:
                cost += 1.0

        return cost

    def _build_result(
            self,
            path: list[Position],
            found: bool,
            visited_nodes: int,
            start_time: float,
    ) -> PathfindingResult:
        return PathfindingResult(
            algorithm_name=self.name,
            found=found,
            path=path,
            path_length=max(len(path) - 1, 0),
            path_cost=self._calculate_path_cost(path),
            visited_nodes=visited_nodes,
            scanned_nodes=getattr(self, "scanned_nodes", 0),
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
        )

    @staticmethod
    def _build_step(
            current: GridNode | None,
            open_nodes: set[GridNode],
            closed_nodes: set[GridNode],
            path: list[Position] | None = None,
    ) -> RawAlgorithmStep:
        raw_path: tuple[GridNode, ...] = ()

        if path:
            raw_path = tuple(
                (position.row, position.col)
                for position in path
            )

        return RawAlgorithmStep(
            current=current,
            open_nodes=frozenset(open_nodes),
            closed_nodes=frozenset(closed_nodes),
            path=raw_path,
        )
