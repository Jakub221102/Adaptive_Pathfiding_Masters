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
    show_cluster_overlay: bool = False

    scenario_index: int = Field(default=0, ge=0)
    min_optimal_length: float = 100

    fps: int = 120
    step_delay_ms: int = 1
    steps_per_frame: int = 100
    step_record_interval: int = 50
