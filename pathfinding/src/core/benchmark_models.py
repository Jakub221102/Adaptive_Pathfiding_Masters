from pydantic import BaseModel, Field


class DynamicBenchmarkConfig(BaseModel):
    scenario_count: int = Field(default=50, gt=0)
    min_optimal_length: float = Field(default=100, ge=0)

    obstacle_count: int = Field(default=8, ge=0)
    obstacle_seed: int = 42
    path_margin: int = Field(default=30, ge=0)

    min_obstacle_size: int = Field(default=6, ge=1)
    max_obstacle_size: int = Field(default=16, ge=1)
    min_obstacle_spacing: int = Field(default=20, ge=0)

    path_block_lookahead: int = Field(default=40, gt=0)
    wait_when_no_path: bool = True
    max_wait_steps: int = Field(default=30, ge=0)
    max_stuck_steps: int = Field(default=200, ge=0)
    max_simulation_steps: int | None = Field(default=None, gt=0)
    moving_obstacle_prediction_steps: int = Field(default=5, ge=0)

    min_dynamic_interaction_rate: float = Field(default=0.5, ge=0, le=1)
    progress_interval: int = Field(default=10, gt=0)
    algorithm_wall_clock_warning_s: float = Field(default=10.0, gt=0)


class DynamicBenchmarkResult(BaseModel):
    algorithm: str
    scenario_index: int
    seed: int

    found: bool
    final_goal_reached: bool
    initial_path_found: bool

    replanning_count: int
    waiting_steps: int
    collision_count: int
    agent_push_count: int
    obstacle_blocked_count: int

    travelled_steps: int
    total_path_cost: float
    total_execution_time_ms: float
    wall_clock_s: float

    obstacle_count: int
    prediction_steps: int
    path_block_lookahead: int

    start_row: int
    start_col: int
    goal_row: int
    goal_col: int
    optimal_length: float | None = None


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
