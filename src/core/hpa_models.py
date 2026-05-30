from pydantic import BaseModel, Field

from src.core.models import Position


class Cluster(BaseModel):
    id: int
    row_start: int = Field(ge=0)
    row_end: int = Field(ge=0)
    col_start: int = Field(ge=0)
    col_end: int = Field(ge=0)

    def contains(self, position: Position) -> bool:
        return (
                self.row_start <= position.row < self.row_end
                and self.col_start <= position.col < self.col_end
        )


class Entrance(BaseModel):
    cluster_a_id: int
    cluster_b_id: int
    position_a: Position
    position_b: Position
    width: int = 1


class AbstractNode(BaseModel):
    id: int
    cluster_id: int
    position: Position


class AbstractEdge(BaseModel):
    from_node_id: int
    to_node_id: int
    cost: float
    path: list[Position] = Field(default_factory=list)


class AbstractGraph(BaseModel):
    nodes: list[AbstractNode]
    edges: list[AbstractEdge]


class HPAPathResult(BaseModel):
    abstract_path: list[AbstractNode]
    refined_path: list[Position]


class HPAPreprocessingStats(BaseModel):
    cluster_count: int = 0
    entrance_count: int = 0

    abstract_node_count: int = 0
    abstract_edge_count: int = 0

    graph_density: float = 0.0
    average_edges_per_node: float = 0.0

    local_path_cache_size: int = 0
    preprocessing_time_ms: float = 0.0


class HPAQueryStats(BaseModel):
    abstract_nodes_visited: int = 0

    abstract_path_edge_count: int = 0
    refined_path_length: int = 0

    local_path_cache_hits: int = 0
    local_path_cache_misses: int = 0
    cache_hit_ratio: float = 0.0

    abstract_search_time_ms: float = 0.0
    refinement_time_ms: float = 0.0
    query_time_ms: float = 0.0
