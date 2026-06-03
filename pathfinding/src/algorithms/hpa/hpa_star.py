import math
import time

from pathfinding.src.algorithms.base import PathfindingAlgorithm
from pathfinding.src.algorithms.hpa.abstract_graph_builder import AbstractGraphBuilder
from pathfinding.src.algorithms.hpa.abstract_search import AbstractSearch
from pathfinding.src.algorithms.hpa.cluster_builder import (
    build_allowed_positions_by_cluster,
    build_clusters,
    build_cluster_lookup,
)
from pathfinding.src.algorithms.hpa.entrance_detector import EntranceDetector
from pathfinding.src.algorithms.hpa.hpa_tracer import build_hpa_steps_from_edges
from pathfinding.src.algorithms.hpa.local_path_cache import LocalPathCache
from pathfinding.src.core.hpa_models import (
    AbstractEdge,
    AbstractGraph,
    AbstractNode,
    Cluster,
    Entrance,
    HPAPreprocessingStats,
    HPAQueryStats,
)
from pathfinding.src.core.models import GridMap, PathfindingResult, Position
from pathfinding.src.core.trace import RawAlgorithmStep


class HPAStar(PathfindingAlgorithm):
    name = "HPA*"

    def __init__(
            self,
            cluster_size: int = 32,
            max_entrances_per_cluster_pair: int = 2,
            min_entrance_width: int = 1,
    ) -> None:
        self.cluster_size = cluster_size
        self.max_entrances_per_cluster_pair = max_entrances_per_cluster_pair
        self.min_entrance_width = min_entrance_width

        self._local_path_cache = LocalPathCache()
        self._entrance_detector = EntranceDetector(
            max_entrances_per_cluster_pair=max_entrances_per_cluster_pair,
            min_entrance_width=min_entrance_width,
        )
        self._abstract_graph_builder = AbstractGraphBuilder(
            local_path_cache=self._local_path_cache,
        )
        self._abstract_search = AbstractSearch()

        self._preprocessed_map_name: str | None = None
        self._preprocessing_time_ms: float = 0.0

        self._clusters: list[Cluster] = []
        self._entrances: list[Entrance] = []
        self._abstract_graph: AbstractGraph | None = None

        self._cluster_by_id: dict[int, Cluster] = {}
        self._cluster_lookup: dict[tuple[int, int], Cluster] = {}
        self._allowed_positions_by_cluster: dict[int, set[tuple[int, int]]] = {}

        self._last_query_stats: HPAQueryStats = HPAQueryStats()
        self._last_abstract_edges: list[AbstractEdge] = []
        self._last_refined_path: list[Position] = []

    def preprocess_map(
            self,
            grid_map: GridMap,
    ) -> None:
        if self._preprocessed_map_name == grid_map.name:
            return

        preprocessing_start_time = time.perf_counter()

        self._local_path_cache.clear()

        self._clusters = build_clusters(
            grid_map=grid_map,
            cluster_size=self.cluster_size,
        )

        self._cluster_by_id = {
            cluster.id: cluster
            for cluster in self._clusters
        }

        self._cluster_lookup = build_cluster_lookup(self._clusters)

        self._allowed_positions_by_cluster = build_allowed_positions_by_cluster(
            self._clusters
        )

        self._entrances = self._entrance_detector.detect_entrances(
            grid_map=grid_map,
            clusters=self._clusters,
            cluster_lookup=self._cluster_lookup,
        )

        self._abstract_graph = self._abstract_graph_builder.build_abstract_graph(
            grid_map=grid_map,
            entrances=self._entrances,
            clusters=self._clusters,
            allowed_positions_by_cluster=self._allowed_positions_by_cluster,
        )

        self._preprocessed_map_name = grid_map.name
        self._preprocessing_time_ms = (
                                              time.perf_counter() - preprocessing_start_time
                                      ) * 1000

    def find_path(
            self,
            grid_map: GridMap,
            start: Position,
            goal: Position,
    ) -> PathfindingResult:
        start_time = time.perf_counter()
        self._last_query_stats = HPAQueryStats()
        self._last_abstract_edges = []
        self._last_refined_path = []

        cache_hits_before = self._local_path_cache.hits
        cache_misses_before = self._local_path_cache.misses

        self.preprocess_map(grid_map)

        if self._abstract_graph is None:
            return self._build_result(
                path=[],
                found=False,
                visited_nodes=0,
                start_time=start_time,
            )

        start_cluster = self._cluster_lookup.get((start.row, start.col))
        goal_cluster = self._cluster_lookup.get((goal.row, goal.col))

        if start_cluster is None or goal_cluster is None:
            return self._build_result(
                path=[],
                found=False,
                visited_nodes=0,
                start_time=start_time,
            )

        if start_cluster.id == goal_cluster.id:
            path = self._local_path_cache.find_path(
                grid_map=grid_map,
                start=start,
                goal=goal,
                cluster_id=start_cluster.id,
                allowed_positions=self._allowed_positions_by_cluster[start_cluster.id],
            )

            cache_hits = self._local_path_cache.hits - cache_hits_before
            cache_misses = self._local_path_cache.misses - cache_misses_before

            self._last_query_stats = HPAQueryStats(
                abstract_nodes_visited=0,
                abstract_path_edge_count=0,
                refined_path_length=max(len(path) - 1, 0),
                refined_path_cost=self._calculate_path_cost(path),
                local_path_cache_hits=cache_hits,
                local_path_cache_misses=cache_misses,
                cache_hit_ratio=self._calculate_cache_hit_ratio(
                    hits=cache_hits,
                    misses=cache_misses,
                ),
                query_time_ms=(time.perf_counter() - start_time) * 1000,
            )

            return self._build_result(
                path=path,
                found=bool(path),
                visited_nodes=0,
                start_time=start_time,
            )

        query_graph, start_node, goal_node = self._insert_start_and_goal_nodes(
            grid_map=grid_map,
            graph=self._abstract_graph,
            start=start,
            goal=goal,
            start_cluster=start_cluster,
            goal_cluster=goal_cluster,
        )

        abstract_search_start_time = time.perf_counter()

        abstract_result = self._abstract_search.find_edge_path(
            graph=query_graph,
            start_node_id=start_node.id,
            goal_node_id=goal_node.id,
        )

        abstract_edges = abstract_result.path

        abstract_search_time_ms = (time.perf_counter() - abstract_search_start_time) * 1000

        self._last_abstract_edges = abstract_edges

        if not abstract_edges:
            cache_hits = self._local_path_cache.hits - cache_hits_before
            cache_misses = self._local_path_cache.misses - cache_misses_before

            self._last_query_stats = HPAQueryStats(
                abstract_nodes_visited=abstract_result.visited_node_count,
                abstract_path_edge_count=0,
                refined_path_length=0,
                refined_path_cost=0.0,
                local_path_cache_hits=cache_hits,
                local_path_cache_misses=cache_misses,
                cache_hit_ratio=self._calculate_cache_hit_ratio(
                    hits=cache_hits,
                    misses=cache_misses,
                ),
                abstract_search_time_ms=abstract_search_time_ms,
                query_time_ms=(time.perf_counter() - start_time) * 1000,
            )

            return self._build_result(
                path=[],
                found=False,
                visited_nodes=0,
                start_time=start_time,
            )

        refinement_start_time = time.perf_counter()

        refined_path = self._refine_abstract_edge_path(abstract_edges)
        refinement_time_ms = (time.perf_counter() - refinement_start_time) * 1000
        self._last_refined_path = refined_path

        cache_hits = self._local_path_cache.hits - cache_hits_before
        cache_misses = self._local_path_cache.misses - cache_misses_before

        query_time_ms = (time.perf_counter() - start_time) * 1000

        self._last_query_stats = HPAQueryStats(
            abstract_nodes_visited=abstract_result.visited_node_count,
            abstract_path_edge_count=len(abstract_edges),
            refined_path_length=max(len(refined_path) - 1, 0),
            refined_path_cost=self._calculate_path_cost(refined_path),
            local_path_cache_hits=cache_hits,
            local_path_cache_misses=cache_misses,
            cache_hit_ratio=self._calculate_cache_hit_ratio(
                hits=cache_hits,
                misses=cache_misses,
            ),
            abstract_search_time_ms=abstract_search_time_ms,
            refinement_time_ms=refinement_time_ms,
            query_time_ms=query_time_ms,
        )

        return self._build_result(
            path=refined_path,
            found=bool(refined_path),
            visited_nodes=abstract_result.visited_node_count,
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

        steps = build_hpa_steps_from_edges(
            grid_map=grid_map,
            abstract_edges=self._last_abstract_edges,
            cluster_lookup=self._cluster_lookup,
            corridor_radius=2,
            include_cluster_cells=True,
        )

        return result, steps

    def get_preprocessing_stats(self) -> HPAPreprocessingStats:
        if self._abstract_graph is None:
            return HPAPreprocessingStats(
                local_path_cache_size=self._local_path_cache.size(),
                preprocessing_time_ms=self._preprocessing_time_ms,
            )

        node_count = len(self._abstract_graph.nodes)
        edge_count = len(self._abstract_graph.edges)

        return HPAPreprocessingStats(
            cluster_count=len(self._clusters),
            entrance_count=len(self._entrances),
            abstract_node_count=node_count,
            abstract_edge_count=edge_count,
            graph_density=self._calculate_graph_density(
                node_count=node_count,
                edge_count=edge_count,
            ),
            average_edges_per_node=self._calculate_average_edges_per_node(
                node_count=node_count,
                edge_count=edge_count,
            ),
            local_path_cache_size=self._local_path_cache.size(),
            preprocessing_time_ms=self._preprocessing_time_ms,
        )

    def get_last_query_stats(self) -> HPAQueryStats:
        return self._last_query_stats

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

        self._abstract_graph_builder.connect_node_to_cluster_nodes(
            grid_map=grid_map,
            node=start_node,
            cluster=start_cluster,
            nodes=nodes,
            edges=edges,
            allowed_positions_by_cluster=self._allowed_positions_by_cluster,
        )

        self._abstract_graph_builder.connect_node_to_cluster_nodes(
            grid_map=grid_map,
            node=goal_node,
            cluster=goal_cluster,
            nodes=nodes,
            edges=edges,
            allowed_positions_by_cluster=self._allowed_positions_by_cluster,
        )

        return AbstractGraph(nodes=nodes, edges=edges), start_node, goal_node

    @staticmethod
    def _refine_abstract_edge_path(
            abstract_edges: list[AbstractEdge],
    ) -> list[Position]:
        if not abstract_edges:
            return []

        full_path: list[Position] = []

        for edge in abstract_edges:
            if not edge.path:
                return []

            segment = edge.path

            if full_path:
                segment = segment[1:]

            full_path.extend(segment)

        return full_path

    @staticmethod
    def _calculate_graph_density(
            node_count: int,
            edge_count: int,
    ) -> float:
        if node_count <= 1:
            return 0.0

        return edge_count / (node_count * (node_count - 1))

    @staticmethod
    def _calculate_average_edges_per_node(
            node_count: int,
            edge_count: int,
    ) -> float:
        if node_count == 0:
            return 0.0

        return edge_count / node_count

    @staticmethod
    def _calculate_cache_hit_ratio(
            hits: int,
            misses: int,
    ) -> float:
        total = hits + misses

        if total == 0:
            return 0.0

        return hits / total

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
        end_time = time.perf_counter()

        return PathfindingResult(
            algorithm_name=self.name,
            found=found,
            path=path,
            path_length=max(len(path) - 1, 0),
            path_cost=self._calculate_path_cost(path),
            visited_nodes=visited_nodes,
            execution_time_ms=(end_time - start_time) * 1000,
        )
