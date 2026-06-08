from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from pathfinding.src.core.dynamic_models import MovingObstacleCollisionPolicy
from pathfinding.src.visualization.pygame_models import ViewerMode


class AlgorithmName(str, Enum):
    ASTAR = "astar"
    HPA_STAR = "hpa_star"
    JPS = "jps"
    DSTAR_LITE = "dstar_lite"


class ExperimentConfig(BaseModel):
    map_path: Path
    scen_path: Path

    algorithm: AlgorithmName = AlgorithmName.ASTAR
    viewer_mode: ViewerMode = ViewerMode.STATIC

    scenario_index: int = Field(default=0, ge=0)
    min_optimal_length: float = Field(default=100, ge=0)

    fps: int = Field(default=120, gt=0)
    step_delay_ms: int = Field(default=1, ge=0)
    steps_per_frame: int = Field(default=100, gt=0)
    step_record_interval: int = Field(default=10, gt=0)
    movement_hold_frames: int = Field(default=1, ge=1)
    movement_frame_delay_ms: int = Field(default=1, ge=0)
    movement_steps_per_frame: int = Field(default=1, gt=0)
    obstacle_hold_frames: int = Field(default=3, ge=1)
    obstacle_frame_delay_ms: int = Field(default=1, ge=0)

    path_block_lookahead: int = Field(default=8, gt=0)
    wait_when_no_path: bool = True
    max_wait_steps: int = Field(default=30, ge=0)
    max_stuck_steps: int = Field(default=50, ge=0)
    max_simulation_steps: int | None = Field(default=None, gt=0)
    moving_obstacle_prediction_steps: int = Field(default=0, ge=0)
    moving_obstacle_collision_policy: MovingObstacleCollisionPolicy = (
        MovingObstacleCollisionPolicy.PUSH_AGENT
    )

    show_cluster_overlay: bool = False
    cluster_size: int = Field(default=32, gt=0)
    max_entrances_per_cluster_pair: int = Field(default=2, gt=0)
    min_entrance_width: int = Field(default=1, gt=0)

    show_heatmap_overlay: bool = False
    animated_comparison: bool = False
    comparison_animation_time_ms: int = Field(default=8000, gt=0)

    benchmark_scenarios: int = Field(default=100, gt=0)
    results_dir: Path = Path("Results")
    benchmark_output_file: str = "benchmark_results.csv"
