from pydantic import BaseModel


class BenchmarkResult(BaseModel):
    algorithm: str
    scenario_index: int
    optimal_length: float | None

    found: bool
    path_length: int
    path_cost: float
    visited_nodes: int
    execution_time_ms: float

    preprocessing_time_ms: float | None = None

    abstract_nodes_visited: int | None = None
    abstract_path_edge_count: int | None = None
    refined_path_length: int | None = None
    refined_path_cost: float | None = None
    scanned_nodes: int | None = None

    local_path_cache_hits: int | None = None
    local_path_cache_misses: int | None = None
    cache_hit_ratio: float | None = None

    abstract_search_time_ms: float | None = None
    refinement_time_ms: float | None = None
    query_time_ms: float | None = None
