import heapq
import time

from code.src.algorithms.base import PathfindingAlgorithm
from code.src.core.models import GridMap, PathfindingResult, Position
from code.src.core.trace import GridNode, RawAlgorithmStep


def _get_neighbors(
        grid_map: GridMap,
        node: GridNode,
        allowed_positions: set[GridNode] | None = None,
) -> list[GridNode]:
    row, col = node

    candidates = [
        (row + 1, col),
        (row - 1, col),
        (row, col + 1),
        (row, col - 1),
    ]

    result: list[GridNode] = []

    for candidate_row, candidate_col in candidates:
        candidate = (candidate_row, candidate_col)

        if allowed_positions is not None and candidate not in allowed_positions:
            continue

        position = Position(row=candidate_row, col=candidate_col)

        if grid_map.in_bounds(position) and grid_map.is_walkable(position):
            result.append(candidate)

    return result


class AStar(PathfindingAlgorithm):
    name = "A*"

    def find_path(
            self,
            grid_map: GridMap,
            start: Position,
            goal: Position,
            allowed_positions: set[GridNode] | None = None,
    ) -> PathfindingResult:
        result, _ = self._run_astar(
            grid_map=grid_map,
            start=start,
            goal=goal,
            record_steps=False,
            allowed_positions=allowed_positions,
        )

        return result

    def find_path_with_steps(
            self,
            grid_map: GridMap,
            start: Position,
            goal: Position,
            step_record_interval: int = 10,
            allowed_positions: set[GridNode] | None = None,
    ) -> tuple[PathfindingResult, list[RawAlgorithmStep]]:
        return self._run_astar(
            grid_map=grid_map,
            start=start,
            goal=goal,
            record_steps=True,
            step_record_interval=step_record_interval,
            allowed_positions=allowed_positions,
        )

    def _run_astar(
            self,
            grid_map: GridMap,
            start: Position,
            goal: Position,
            record_steps: bool,
            step_record_interval: int = 10,
            allowed_positions: set[GridNode] | None = None,
    ) -> tuple[PathfindingResult, list[RawAlgorithmStep]]:
        start_time = time.perf_counter()

        start_node = (start.row, start.col)
        goal_node = (goal.row, goal.col)

        if allowed_positions is not None:
            if start_node not in allowed_positions or goal_node not in allowed_positions:
                result = self._build_result(
                    path=[],
                    found=False,
                    visited_nodes=0,
                    start_time=start_time,
                )
                return result, []

        open_heap: list[tuple[float, int, GridNode]] = []
        open_lookup: set[GridNode] = {start_node}
        closed_lookup: set[GridNode] = set()

        counter = 0
        heapq.heappush(open_heap, (0, counter, start_node))

        came_from: dict[GridNode, GridNode] = {}
        g_score: dict[GridNode, float] = {
            start_node: 0
        }

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
                path = self._reconstruct_path(
                    came_from=came_from,
                    current=current,
                )

                if record_steps:
                    steps.append(
                        self._build_step(
                            current=current,
                            open_nodes=open_lookup,
                            closed_nodes=closed_lookup,
                            path=path,
                        )
                    )

                result = self._build_result(
                    path=path,
                    found=True,
                    visited_nodes=visited_nodes,
                    start_time=start_time,
                )

                return result, steps

            for neighbor in self._get_neighbors(
                    grid_map=grid_map,
                    node=current,
                    allowed_positions=allowed_positions,
            ):
                if neighbor in closed_lookup:
                    continue

                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g

                    counter += 1
                    f_score = tentative_g + self._heuristic(neighbor, goal_node)

                    heapq.heappush(
                        open_heap,
                        (f_score, counter, neighbor),
                    )

                    open_lookup.add(neighbor)

        result = self._build_result(
            path=[],
            found=False,
            visited_nodes=visited_nodes,
            start_time=start_time,
        )

        return result, steps

    def _get_neighbors(
            self,
            grid_map: GridMap,
            node: GridNode,
            allowed_positions: set[GridNode] | None = None,
    ) -> list[GridNode]:
        row, col = node

        candidates = [
            (row + 1, col),
            (row - 1, col),
            (row, col + 1),
            (row, col - 1),
        ]

        result: list[GridNode] = []

        for candidate_row, candidate_col in candidates:
            candidate = (candidate_row, candidate_col)

            if allowed_positions is not None and candidate not in allowed_positions:
                continue

            position = Position(row=candidate_row, col=candidate_col)

            if grid_map.in_bounds(position) and grid_map.is_walkable(position):
                result.append(candidate)

        return result

    @staticmethod
    def _heuristic(
            current: GridNode,
            goal: GridNode,
    ) -> float:
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

    @staticmethod
    def _reconstruct_path(
            came_from: dict[GridNode, GridNode],
            current: GridNode,
    ) -> list[Position]:
        path = [Position(row=current[0], col=current[1])]

        while current in came_from:
            current = came_from[current]
            path.append(Position(row=current[0], col=current[1]))

        path.reverse()
        return path

    def _build_result(
            self,
            path: list[Position],
            found: bool,
            visited_nodes: int,
            start_time: float,
    ) -> PathfindingResult:
        end_time = time.perf_counter()

        return PathfindingResult(
            algorithm_name=self.name,
            found=found,
            path=path,
            path_length=max(len(path) - 1, 0),
            visited_nodes=visited_nodes,
            execution_time_ms=(end_time - start_time) * 1000,
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
            raw_path = tuple((position.row, position.col) for position in path)

        return RawAlgorithmStep(
            current=current,
            open_nodes=frozenset(open_nodes),
            closed_nodes=frozenset(closed_nodes),
            path=raw_path,
        )
