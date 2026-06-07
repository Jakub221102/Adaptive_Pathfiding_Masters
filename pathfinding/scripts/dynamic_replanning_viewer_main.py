from pathlib import Path

from pathfinding.src.core.dynamic_models import (
    DynamicObstacleEvent,
    DynamicObstacleEventType,
)
from pathfinding.src.core.models import Position
from pathfinding.src.experiments.dynamic_replanning_visualization_experiment import (
    DynamicReplanningVisualizationExperiment,
)
from pathfinding.src.experiments.experiment_config import (
    AlgorithmName,
    ExperimentConfig,
)


def main() -> None:
    config = ExperimentConfig(
        map_path=Path("../../Data/bg512-map/AR0204SR.map"),
        scen_path=Path("../../Data/bg512-scen/AR0204SR.map.scen"),
        scenario_index=19,
        min_optimal_length=100,
        algorithm=AlgorithmName.DSTAR_LITE,
        fps=120,
        step_delay_ms=8,
        steps_per_frame=8,
        step_record_interval=8,
        movement_hold_frames=1,
        movement_frame_delay_ms=1,
        movement_steps_per_frame=16,
    )

    events = [
        DynamicObstacleEvent(
            step_index=20,
            position=Position(row=100, col=301),
            event_type=DynamicObstacleEventType.BLOCK,
            width=10,
            height=20,
        )
    ]

    experiment = DynamicReplanningVisualizationExperiment(
        config=config,
        events=events,
    )

    experiment.run()


if __name__ == "__main__":
    main()
