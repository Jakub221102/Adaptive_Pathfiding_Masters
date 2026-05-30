import heapq

from pydantic import BaseModel

from src.core.hpa_models import AbstractEdge, AbstractGraph, AbstractNode


class AbstractSearchResult(BaseModel):
    path: list[AbstractEdge]
    visited_node_count: int


class AbstractSearch:
    def find_edge_path(
            self,
            graph: AbstractGraph,
            start_node_id: int,
            goal_node_id: int,
    ) -> AbstractSearchResult:
        nodes_by_id = {
            node.id: node
            for node in graph.nodes
        }

        adjacency: dict[int, list[AbstractEdge]] = {}

        for edge in graph.edges:
            adjacency.setdefault(edge.from_node_id, []).append(edge)

        open_heap: list[tuple[float, int, int]] = []
        came_from_edge: dict[int, AbstractEdge] = {}
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
                return AbstractSearchResult(
                    path=self._reconstruct_edge_path(
                        came_from_edge=came_from_edge,
                        current_id=current_id,
                        start_node_id=start_node_id,
                    ),
                    visited_node_count=len(closed),
                )

            for edge in adjacency.get(current_id, []):
                neighbor_id = edge.to_node_id
                tentative_g = g_score[current_id] + edge.cost

                if neighbor_id not in g_score or tentative_g < g_score[neighbor_id]:
                    came_from_edge[neighbor_id] = edge
                    g_score[neighbor_id] = tentative_g

                    counter += 1

                    priority = tentative_g + self._heuristic(
                        current=nodes_by_id[neighbor_id],
                        goal=nodes_by_id[goal_node_id],
                    )

                    heapq.heappush(open_heap, (priority, counter, neighbor_id))

        return AbstractSearchResult(
            path=[],
            visited_node_count=len(closed),
        )

    @staticmethod
    def _reconstruct_edge_path(
            came_from_edge: dict[int, AbstractEdge],
            current_id: int,
            start_node_id: int,
    ) -> list[AbstractEdge]:
        path: list[AbstractEdge] = []

        while current_id != start_node_id:
            edge = came_from_edge.get(current_id)

            if edge is None:
                return []

            path.append(edge)
            current_id = edge.from_node_id

        path.reverse()
        return path

    @staticmethod
    def _heuristic(
            current: AbstractNode,
            goal: AbstractNode,
    ) -> float:
        return (
                abs(current.position.row - goal.position.row)
                + abs(current.position.col - goal.position.col)
        )
