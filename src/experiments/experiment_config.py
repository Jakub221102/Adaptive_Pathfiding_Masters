from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from src.visualization.pygame_models import ViewerMode


class AlgorithmName(str, Enum):
    ASTAR = "astar"
    HPA_STAR = "hpa_star"


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

    show_cluster_overlay: bool = False
    cluster_size: int = Field(default=32, gt=0)
    show_heatmap_overlay: bool = False