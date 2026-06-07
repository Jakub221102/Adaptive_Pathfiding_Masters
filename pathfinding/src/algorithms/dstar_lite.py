import heapq
import math
import time

from pathfinding.src.algorithms.astar import AStar
from pathfinding.src.algorithms.base import PathfindingAlgorithm
from pathfinding.src.core.models import GridMap, PathfindingResult, Position
from pathfinding.src.core.trace import GridNode, RawAlgorithmStep

INF = float("inf")


class DStarLite(PathfindingAlgorithm):
    name = "D* Lite"

    def __init__(self) -> None:
        self._grid_map: GridMap | None = None
        self._start: Position | None = None
        self._goal: Position | None = None
        self._start_node: GridNode | None = None
        self._goal_node: GridNode | None = None
        self._km = 0.0
        self._g: dict[GridNode, float] = {}
        self._rhs: dict[GridNode, float] = {}
        self._open_heap: list[tuple[tuple[float, float], int, GridNode]] = []
        self._open_lookup: set[GridNode] = set()
        self._counter = 0
        self._expanded_nodes = 0
        self._last_steps: list[RawAlgorithmStep] = []
        self._record_steps: bool = False
        self._step_record_interval: int = 10

    def find_path(
            self,
            grid_map: GridMap,
            start: Position,
            goal: Position,
    ) -> PathfindingResult:
        start_time = time.perf_counter()
        self._initialize_search(grid_map=grid_map, start=start, goal=goal)
        self._expanded_nodes = self._compute_shortest_path()
        path = self._extract_path()
        found = bool(path) and path[-1] == goal

        return self._build_result(
            path=path if found else [],
            found=found,
            visited_nodes=self._expanded_nodes,
            start_time=start_time,
        )

    def find_path_with_steps(
            self,
            grid_map: GridMap,
            start: Position,
            goal: Position,
            step_record_interval: int = 10,
    ) -> tuple[PathfindingResult, list[RawAlgorithmStep]]:
        self._record_steps = True
        self._step_record_interval = step_record_interval
        self._last_steps = []
        result = self.find_path(
            grid_map=grid_map,
            start=start,
            goal=goal,
        )
        steps = list(self._last_steps)
        self._record_steps = False
        return result, steps

    def replan_with_steps(
            self,
            step_record_interval: int = 10,
    ) -> tuple[PathfindingResult, list[RawAlgorithmStep]]:
        self._record_steps = True
        self._step_record_interval = step_record_interval
        self._last_steps = []
        result = self.replan()
        steps = list(self._last_steps)
        self._record_steps = False
        return result, steps

    def replan(self) -> PathfindingResult:
        if self._grid_map is None or self._start is None or self._goal is None:
            raise RuntimeError("D* Lite search was not initialized.")

        start_time = time.perf_counter()
        self._expanded_nodes = self._compute_shortest_path()
        path = self._extract_path()
        found = bool(path) and path[-1] == self._goal

        return self._build_result(
            path=path if found else [],
            found=found,
            visited_nodes=self._expanded_nodes,
            start_time=start_time,
        )

    def update_cell(self, position: Position) -> None:
        self.update_cells([position])

    def update_cells(self, positions: list[Position]) -> None:
        if self._grid_map is None or self._goal_node is None:
            return

        affected_nodes: set[GridNode] = set()

        for position in positions:
            node = (position.row, position.col)
            affected_nodes.add(node)

            for predecessor in self._get_pred(node):
                affected_nodes.add(predecessor)

            for successor in self._get_succ(node):
                affected_nodes.add(successor)

        for affected_node in affected_nodes:
            self._update_vertex(affected_node)

    def move_agent(self, new_position: Position) -> None:
        if self._start_node is None:
            raise RuntimeError("D* Lite search was not initialized.")

        new_node = (new_position.row, new_position.col)
        self._km += AStar._movement_cost(self._start_node, new_node)
        self._start = new_position
        self._start_node = new_node

    def _initialize_search(
            self,
            grid_map: GridMap,
            start: Position,
            goal: Position,
    ) -> None:
        self._grid_map = grid_map
        self._start = start
        self._goal = goal
        self._start_node = (start.row, start.col)
        self._goal_node = (goal.row, goal.col)
        self._km = 0.0
        self._g = {}
        self._rhs = {}
        self._open_heap = []
        self._open_lookup = set()
        self._counter = 0
        self._expanded_nodes = 0

        if (
                not grid_map.is_walkable(start)
                or not grid_map.is_walkable(goal)
        ):
            return

        self._rhs[self._goal_node] = 0.0
        self._insert(self._goal_node)

    def _compute_shortest_path(self) -> int:
        if (
                self._grid_map is None
                or self._start_node is None
                or self._goal_node is None
        ):
            return 0

        expanded_nodes = 0

        while self._open_heap:
            start_key = self._calculate_key(self._start_node)
            top_key, top_node = self._top_key()

            if (
                    top_key >= start_key
                    and self._rhs.get(self._start_node, INF) == self._g.get(
                        self._start_node, INF
                    )
            ):
                break

            _, node = self._pop()
            expanded_nodes += 1

            if expanded_nodes % self._step_record_interval == 0:
                self._record_algorithm_step(current=node)

            node_key = self._calculate_key(node)

            if node_key > top_key:
                self._insert(node)
                continue

            if self._g.get(node, INF) > self._rhs.get(node, INF):
                self._g[node] = self._rhs[node]
                for predecessor in self._get_pred(node):
                    self._update_vertex(predecessor)
            else:
                self._g[node] = INF
                self._update_vertex(node)
                for predecessor in self._get_pred(node):
                    self._update_vertex(predecessor)

        self._record_algorithm_step(current=self._start_node)
        return expanded_nodes

    def _extract_path(self) -> list[Position]:
        if (
                self._grid_map is None
                or self._start is None
                or self._goal is None
                or self._start_node is None
                or self._goal_node is None
        ):
            return []

        if self._g.get(self._start_node, INF) == INF:
            return []

        path = [self._start]
        current_node = self._start_node
        visited: set[GridNode] = set()

        while current_node != self._goal_node:
            if current_node in visited:
                return []
            visited.add(current_node)

            successors = self._get_succ(current_node)

            if not successors:
                return []

            next_node = min(
                successors,
                key=lambda successor: AStar._movement_cost(
                    current_node,
                    successor,
                ) + self._g.get(successor, INF),
            )

            if self._g.get(next_node, INF) == INF:
                return []

            current_node = next_node
            path.append(
                Position(row=current_node[0], col=current_node[1])
            )

        return path

    def _update_vertex(self, node: GridNode) -> None:
        if self._goal_node is None:
            return

        if node != self._goal_node:
            self._rhs[node] = self._calculate_rhs(node)

        if node in self._open_lookup:
            self._open_lookup.discard(node)

        if self._g.get(node, INF) != self._rhs.get(node, INF):
            self._insert(node)

    def _calculate_rhs(self, node: GridNode) -> float:
        if self._grid_map is None:
            return INF

        if not self._is_walkable(node):
            return INF

        successors = self._get_succ(node)

        if not successors:
            return INF

        return min(
            AStar._movement_cost(node, successor) + self._g.get(successor, INF)
            for successor in successors
        )

    def _record_algorithm_step(self, current: GridNode | None) -> None:
        if not self._record_steps:
            return

        closed_nodes = {
            node
            for node, value in self._g.items()
            if value < INF
        }

        open_nodes = set(self._open_lookup)

        self._last_steps.append(
            RawAlgorithmStep(
                current=current,
                open_nodes=frozenset(open_nodes),
                closed_nodes=frozenset(closed_nodes),
                path=tuple(),
            )
        )

    def _calculate_key(self, node: GridNode) -> tuple[float, float]:
        if self._start_node is None:
            return (INF, INF)

        min_g_rhs = min(self._g.get(node, INF), self._rhs.get(node, INF))
        heuristic = AStar._heuristic(self._start_node, node)

        return min_g_rhs + heuristic + self._km, min_g_rhs

    def _insert(self, node: GridNode) -> None:
        self._open_lookup.add(node)
        heapq.heappush(
            self._open_heap,
            (self._calculate_key(node), self._counter, node),
        )
        self._counter += 1

    def _top_key(self) -> tuple[tuple[float, float], GridNode]:
        while self._open_heap:
            key, _, node = self._open_heap[0]

            if node not in self._open_lookup:
                heapq.heappop(self._open_heap)
                continue

            return key, node

        return (INF, INF), (-1, -1)

    def _pop(self) -> tuple[tuple[float, float], GridNode]:
        while self._open_heap:
            key, _, node = heapq.heappop(self._open_heap)

            if node not in self._open_lookup:
                continue

            self._open_lookup.discard(node)
            return key, node

        return (INF, INF), (-1, -1)

    def _get_succ(self, node: GridNode) -> list[GridNode]:
        if self._grid_map is None:
            return []

        return self._get_neighbors(node)

    def _get_pred(self, node: GridNode) -> list[GridNode]:
        if self._grid_map is None or not self._is_walkable(node):
            return []

        row, col = node
        predecessors: list[GridNode] = []

        for candidate in self._neighbor_candidates(row, col):
            if not self._is_walkable(candidate):
                continue

            if node in self._get_neighbors(candidate):
                predecessors.append(candidate)

        return predecessors

    def _get_neighbors(self, node: GridNode) -> list[GridNode]:
        if self._grid_map is None:
            return []

        row, col = node
        neighbors: list[GridNode] = []

        for candidate in self._neighbor_candidates(row, col):
            if self._is_walkable(candidate):
                neighbors.append(candidate)

        return neighbors

    @staticmethod
    def _neighbor_candidates(row: int, col: int) -> list[GridNode]:
        return [
            (row + 1, col),
            (row - 1, col),
            (row, col + 1),
            (row, col - 1),
            (row + 1, col + 1),
            (row + 1, col - 1),
            (row - 1, col + 1),
            (row - 1, col - 1),
        ]

    def _is_walkable(self, node: GridNode) -> bool:
        if self._grid_map is None:
            return False

        row, col = node

        if row < 0 or col < 0:
            return False

        if row >= self._grid_map.height or col >= self._grid_map.width:
            return False

        return self._grid_map.is_walkable(
            Position(row=row, col=col)
        )

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
            path_cost=AStar._calculate_path_cost(path),
            visited_nodes=visited_nodes,
            execution_time_ms=(end_time - start_time) * 1000,
        )
