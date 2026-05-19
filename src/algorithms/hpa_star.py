import heapq
import time

from src.algorithms.astar import AStar
from src.algorithms.base import PathfindingAlgorithm
from src.core.hpa_models import (
    AbstractEdge,
    AbstractGraph,
    AbstractNode,
    Cluster,
    Entrance,
)
from src.core.models import GridMap, PathfindingResult, Position
from src.core.trace import RawAlgorithmStep


class HPAStar(PathfindingAlgorithm):
    name = "HPA*"

    def __init__(self, cluster_size: int = 32) -> None:
        self.cluster_size = cluster_size
        self._fallback_astar = AStar()

        self._preprocessed_map_name: str | None = None
        self._clusters: list[Cluster] = []
        self._entrances: list[Entrance] = []
        self._abstract_graph: AbstractGraph | None = None

        self._cluster_by_id: dict[int, Cluster] = {}
        self._cluster_lookup: dict[tuple[int, int], Cluster] = {}
        self._allowed_positions_by_cluster: dict[int, set[tuple[int, int]]] = {}

    def preprocess_map(self, grid_map: GridMap) -> None:
        if self._preprocessed_map_name == grid_map.name:
            return

        self._clusters = self.build_clusters(grid_map)
        self._cluster_by_id = {
            cluster.id: cluster
            for cluster in self._clusters
        }

        self._cluster_lookup = self._build_cluster_lookup(self._clusters)

        self._allowed_positions_by_cluster = {
            cluster.id: self._build_allowed_positions_for_cluster(cluster)
            for cluster in self._clusters
        }

        self._entrances = self.detect_entrances(
            grid_map=grid_map,
            clusters=self._clusters,
        )

        self._abstract_graph = self.build_abstract_graph(
            grid_map=grid_map,
            entrances=self._entrances,
            clusters=self._clusters,
        )

        self._preprocessed_map_name = grid_map.name

    def find_path(
        self,
        grid_map: GridMap,
        start: Position,
        goal: Position,
    ) -> PathfindingResult:
        start_time = time.perf_counter()

        self.preprocess_map(grid_map)

        if self._abstract_graph is None:
            return self._build_hpa_result(
                path=[],
                found=False,
                visited_nodes=0,
                start_time=start_time,
            )

        start_cluster = self._cluster_lookup.get((start.row, start.col))
        goal_cluster = self._cluster_lookup.get((goal.row, goal.col))

        if start_cluster is None or goal_cluster is None:
            return self._build_hpa_result(
                path=[],
                found=False,
                visited_nodes=0,
                start_time=start_time,
            )

        if start_cluster.id == goal_cluster.id:
            result = self._fallback_astar.find_path(
                grid_map=grid_map,
                start=start,
                goal=goal,
                allowed_positions=self._allowed_positions_by_cluster[start_cluster.id],
            )
            result.algorithm_name = self.name
            return result

        query_graph, start_node, goal_node = self._insert_start_and_goal_nodes(
            grid_map=grid_map,
            graph=self._abstract_graph,
            start=start,
            goal=goal,
            start_cluster=start_cluster,
            goal_cluster=goal_cluster,
        )

        abstract_path = self.find_abstract_path(
            graph=query_graph,
            start_node_id=start_node.id,
            goal_node_id=goal_node.id,
        )

        if not abstract_path:
            return self._build_hpa_result(
                path=[],
                found=False,
                visited_nodes=0,
                start_time=start_time,
            )

        refined_path = self._refine_abstract_path(
            grid_map=grid_map,
            abstract_path=abstract_path,
        )

        return self._build_hpa_result(
            path=refined_path,
            found=bool(refined_path),
            visited_nodes=len(abstract_path),
            start_time=start_time,
        )

    def find_path_with_steps(
        self,
        grid_map: GridMap,
        start: Position,
        goal: Position,
        step_record_interval: int = 10,
    ) -> tuple[PathfindingResult, list[RawAlgorithmStep]]:
        result = self.find_path(
            grid_map=grid_map,
            start=start,
            goal=goal,
        )

        return result, []

    def build_clusters(self, grid_map: GridMap) -> list[Cluster]:
        clusters: list[Cluster] = []
        cluster_id = 0

        for row_start in range(0, grid_map.height, self.cluster_size):
            for col_start in range(0, grid_map.width, self.cluster_size):
                clusters.append(
                    Cluster(
                        id=cluster_id,
                        row_start=row_start,
                        row_end=min(row_start + self.cluster_size, grid_map.height),
                        col_start=col_start,
                        col_end=min(col_start + self.cluster_size, grid_map.width),
                    )
                )
                cluster_id += 1

        return clusters

    def detect_entrances(
        self,
        grid_map: GridMap,
        clusters: list[Cluster],
    ) -> list[Entrance]:
        entrances: list[Entrance] = []
        cluster_by_grid_position = self._build_cluster_lookup(clusters)

        for cluster in clusters:
            entrances.extend(
                self._detect_horizontal_neighbor_entrance_groups(
                    grid_map=grid_map,
                    cluster=cluster,
                    cluster_by_grid_position=cluster_by_grid_position,
                )
            )

            entrances.extend(
                self._detect_vertical_neighbor_entrance_groups(
                    grid_map=grid_map,
                    cluster=cluster,
                    cluster_by_grid_position=cluster_by_grid_position,
                )
            )

        return entrances

    def build_abstract_graph(
        self,
        grid_map: GridMap,
        entrances: list[Entrance],
        clusters: list[Cluster],
    ) -> AbstractGraph:
        nodes: list[AbstractNode] = []
        edges: list[AbstractEdge] = []

        node_id = 0

        for entrance in entrances:
            node_a = AbstractNode(
                id=node_id,
                cluster_id=entrance.cluster_a_id,
                position=entrance.position_a,
            )
            node_id += 1

            node_b = AbstractNode(
                id=node_id,
                cluster_id=entrance.cluster_b_id,
                position=entrance.position_b,
            )
            node_id += 1

            nodes.append(node_a)
            nodes.append(node_b)

            edges.append(
                AbstractEdge(
                    from_node_id=node_a.id,
                    to_node_id=node_b.id,
                    cost=1,
                )
            )
            edges.append(
                AbstractEdge(
                    from_node_id=node_b.id,
                    to_node_id=node_a.id,
                    cost=1,
                )
            )

        edges.extend(
            self._build_intra_cluster_edges(
                grid_map=grid_map,
                clusters=clusters,
                nodes=nodes,
            )
        )

        return AbstractGraph(
            nodes=nodes,
            edges=edges,
        )

    def find_abstract_path(
        self,
        graph: AbstractGraph,
        start_node_id: int,
        goal_node_id: int,
    ) -> list[AbstractNode]:
        nodes_by_id = {
            node.id: node
            for node in graph.nodes
        }

        adjacency: dict[int, list[AbstractEdge]] = {}

        for edge in graph.edges:
            adjacency.setdefault(edge.from_node_id, []).append(edge)

        open_heap: list[tuple[float, int, int]] = []
        came_from: dict[int, int] = {}
        g_score: dict[int, float] = {
            start_node_id: 0
        }

        counter = 0
        heapq.heappush(open_heap, (0, counter, start_node_id))

        closed: set[int] = set()

        while open_heap:
            _, _, current_id = heapq.heappop(open_heap)

            if current_id in closed:
                continue

            closed.add(current_id)

            if current_id == goal_node_id:
                return self._reconstruct_abstract_path(
                    came_from=came_from,
                    current_id=current_id,
                    nodes_by_id=nodes_by_id,
                )

            for edge in adjacency.get(current_id, []):
                neighbor_id = edge.to_node_id

                tentative_g = g_score[current_id] + edge.cost

                if neighbor_id not in g_score or tentative_g < g_score[neighbor_id]:
                    came_from[neighbor_id] = current_id
                    g_score[neighbor_id] = tentative_g

                    counter += 1
                    priority = tentative_g + self._abstract_heuristic(
                        nodes_by_id[neighbor_id],
                        nodes_by_id[goal_node_id],
                    )

                    heapq.heappush(
                        open_heap,
                        (priority, counter, neighbor_id),
                    )

        return []

    def _build_intra_cluster_edges(
        self,
        grid_map: GridMap,
        clusters: list[Cluster],
        nodes: list[AbstractNode],
    ) -> list[AbstractEdge]:
        edges: list[AbstractEdge] = []

        nodes_by_cluster: dict[int, list[AbstractNode]] = {}

        for node in nodes:
            nodes_by_cluster.setdefault(node.cluster_id, []).append(node)

        cluster_by_id = {
            cluster.id: cluster
            for cluster in clusters
        }

        for cluster_id, cluster_nodes in nodes_by_cluster.items():
            cluster = cluster_by_id[cluster_id]
            allowed_positions = self._allowed_positions_by_cluster.get(
                cluster.id,
                self._build_allowed_positions_for_cluster(cluster),
            )

            for i in range(len(cluster_nodes)):
                for j in range(i + 1, len(cluster_nodes)):
                    node_a = cluster_nodes[i]
                    node_b = cluster_nodes[j]

                    result = self._fallback_astar.find_path(
                        grid_map=grid_map,
                        start=node_a.position,
                        goal=node_b.position,
                        allowed_positions=allowed_positions,
                    )

                    if not result.found:
                        continue

                    edges.append(
                        AbstractEdge(
                            from_node_id=node_a.id,
                            to_node_id=node_b.id,
                            cost=float(result.path_length),
                        )
                    )
                    edges.append(
                        AbstractEdge(
                            from_node_id=node_b.id,
                            to_node_id=node_a.id,
                            cost=float(result.path_length),
                        )
                    )

        return edges

    def _detect_horizontal_neighbor_entrance_groups(
        self,
        grid_map: GridMap,
        cluster: Cluster,
        cluster_by_grid_position: dict[tuple[int, int], Cluster],
    ) -> list[Entrance]:
        entrances: list[Entrance] = []

        right_col = cluster.col_end
        left_col = cluster.col_end - 1

        if right_col >= grid_map.width:
            return entrances

        current_group: list[tuple[int, Cluster]] = []

        for row in range(cluster.row_start, cluster.row_end):
            neighbor_cluster = cluster_by_grid_position.get((row, right_col))

            if neighbor_cluster is None:
                self._add_horizontal_group_entrance(
                    entrances,
                    current_group,
                    cluster,
                    left_col,
                    right_col,
                )
                current_group = []
                continue

            position_a = Position(row=row, col=left_col)
            position_b = Position(row=row, col=right_col)

            if grid_map.is_walkable(position_a) and grid_map.is_walkable(position_b):
                current_group.append((row, neighbor_cluster))
            else:
                self._add_horizontal_group_entrance(
                    entrances,
                    current_group,
                    cluster,
                    left_col,
                    right_col,
                )
                current_group = []

        self._add_horizontal_group_entrance(
            entrances,
            current_group,
            cluster,
            left_col,
            right_col,
        )

        return entrances

    def _detect_vertical_neighbor_entrance_groups(
        self,
        grid_map: GridMap,
        cluster: Cluster,
        cluster_by_grid_position: dict[tuple[int, int], Cluster],
    ) -> list[Entrance]:
        entrances: list[Entrance] = []

        bottom_row = cluster.row_end
        top_row = cluster.row_end - 1

        if bottom_row >= grid_map.height:
            return entrances

        current_group: list[tuple[int, Cluster]] = []

        for col in range(cluster.col_start, cluster.col_end):
            neighbor_cluster = cluster_by_grid_position.get((bottom_row, col))

            if neighbor_cluster is None:
                self._add_vertical_group_entrance(
                    entrances,
                    current_group,
                    cluster,
                    top_row,
                    bottom_row,
                )
                current_group = []
                continue

            position_a = Position(row=top_row, col=col)
            position_b = Position(row=bottom_row, col=col)

            if grid_map.is_walkable(position_a) and grid_map.is_walkable(position_b):
                current_group.append((col, neighbor_cluster))
            else:
                self._add_vertical_group_entrance(
                    entrances,
                    current_group,
                    cluster,
                    top_row,
                    bottom_row,
                )
                current_group = []

        self._add_vertical_group_entrance(
            entrances,
            current_group,
            cluster,
            top_row,
            bottom_row,
        )

        return entrances

    def _insert_start_and_goal_nodes(
        self,
        grid_map: GridMap,
        graph: AbstractGraph,
        start: Position,
        goal: Position,
        start_cluster: Cluster,
        goal_cluster: Cluster,
    ) -> tuple[AbstractGraph, AbstractNode, AbstractNode]:
        nodes = list(graph.nodes)
        edges = list(graph.edges)

        next_id = max(node.id for node in nodes) + 1 if nodes else 0

        start_node = AbstractNode(
            id=next_id,
            cluster_id=start_cluster.id,
            position=start,
        )

        goal_node = AbstractNode(
            id=next_id + 1,
            cluster_id=goal_cluster.id,
            position=goal,
        )

        nodes.append(start_node)
        nodes.append(goal_node)

        self._connect_node_to_cluster_nodes(
            grid_map=grid_map,
            node=start_node,
            cluster=start_cluster,
            nodes=nodes,
            edges=edges,
        )

        self._connect_node_to_cluster_nodes(
            grid_map=grid_map,
            node=goal_node,
            cluster=goal_cluster,
            nodes=nodes,
            edges=edges,
        )

        return AbstractGraph(nodes=nodes, edges=edges), start_node, goal_node

    def _connect_node_to_cluster_nodes(
        self,
        grid_map: GridMap,
        node: AbstractNode,
        cluster: Cluster,
        nodes: list[AbstractNode],
        edges: list[AbstractEdge],
    ) -> None:
        allowed_positions = self._allowed_positions_by_cluster.get(
            cluster.id,
            self._build_allowed_positions_for_cluster(cluster),
        )

        cluster_nodes = [
            other_node for other_node in nodes
            if other_node.cluster_id == cluster.id
            and other_node.id != node.id
        ]

        for other_node in cluster_nodes:
            result = self._fallback_astar.find_path(
                grid_map=grid_map,
                start=node.position,
                goal=other_node.position,
                allowed_positions=allowed_positions,
            )

            if not result.found:
                continue

            edges.append(
                AbstractEdge(
                    from_node_id=node.id,
                    to_node_id=other_node.id,
                    cost=float(result.path_length),
                )
            )
            edges.append(
                AbstractEdge(
                    from_node_id=other_node.id,
                    to_node_id=node.id,
                    cost=float(result.path_length),
                )
            )

    def _refine_abstract_path(
        self,
        grid_map: GridMap,
        abstract_path: list[AbstractNode],
    ) -> list[Position]:
        if len(abstract_path) < 2:
            return []

        full_path: list[Position] = []

        for index in range(len(abstract_path) - 1):
            current_node = abstract_path[index]
            next_node = abstract_path[index + 1]

            current_cluster = self._cluster_lookup.get(
                (current_node.position.row, current_node.position.col)
            )
            next_cluster = self._cluster_lookup.get(
                (next_node.position.row, next_node.position.col)
            )

            if current_cluster is None or next_cluster is None:
                return []

            if current_cluster.id == next_cluster.id:
                allowed_positions = self._allowed_positions_by_cluster[
                    current_cluster.id
                ]
            else:
                allowed_positions = None

            result = self._fallback_astar.find_path(
                grid_map=grid_map,
                start=current_node.position,
                goal=next_node.position,
                allowed_positions=allowed_positions,
            )

            if not result.found:
                return []

            segment = result.path

            if full_path:
                segment = segment[1:]

            full_path.extend(segment)

        return full_path

    def _build_hpa_result(
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
    def _reconstruct_abstract_path(
        came_from: dict[int, int],
        current_id: int,
        nodes_by_id: dict[int, AbstractNode],
    ) -> list[AbstractNode]:
        path = [nodes_by_id[current_id]]

        while current_id in came_from:
            current_id = came_from[current_id]
            path.append(nodes_by_id[current_id])

        path.reverse()
        return path

    @staticmethod
    def _abstract_heuristic(
        current: AbstractNode,
        goal: AbstractNode,
    ) -> float:
        return (
            abs(current.position.row - goal.position.row)
            + abs(current.position.col - goal.position.col)
        )

    @staticmethod
    def _build_allowed_positions_for_cluster(
        cluster: Cluster,
    ) -> set[tuple[int, int]]:
        return {
            (row, col)
            for row in range(cluster.row_start, cluster.row_end)
            for col in range(cluster.col_start, cluster.col_end)
        }

    @staticmethod
    def _build_cluster_lookup(
        clusters: list[Cluster],
    ) -> dict[tuple[int, int], Cluster]:
        lookup: dict[tuple[int, int], Cluster] = {}

        for cluster in clusters:
            for row in range(cluster.row_start, cluster.row_end):
                for col in range(cluster.col_start, cluster.col_end):
                    lookup[(row, col)] = cluster

        return lookup

    @staticmethod
    def _add_horizontal_group_entrance(
        entrances: list[Entrance],
        group: list[tuple[int, Cluster]],
        cluster: Cluster,
        left_col: int,
        right_col: int,
    ) -> None:
        if not group:
            return

        middle_index = len(group) // 2
        row, neighbor_cluster = group[middle_index]

        entrances.append(
            Entrance(
                cluster_a_id=cluster.id,
                cluster_b_id=neighbor_cluster.id,
                position_a=Position(row=row, col=left_col),
                position_b=Position(row=row, col=right_col),
            )
        )

    @staticmethod
    def _add_vertical_group_entrance(
        entrances: list[Entrance],
        group: list[tuple[int, Cluster]],
        cluster: Cluster,
        top_row: int,
        bottom_row: int,
    ) -> None:
        if not group:
            return

        middle_index = len(group) // 2
        col, neighbor_cluster = group[middle_index]

        entrances.append(
            Entrance(
                cluster_a_id=cluster.id,
                cluster_b_id=neighbor_cluster.id,
                position_a=Position(row=top_row, col=col),
                position_b=Position(row=bottom_row, col=col),
            )
        )