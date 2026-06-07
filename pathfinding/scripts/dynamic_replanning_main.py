from pathlib import Path

from pathfinding.src.core.dynamic_models import (
    DynamicObstacleEvent,
    DynamicObstacleEventType,
)
from pathfinding.src.core.models import Position
from pathfinding.src.experiments.dynamic_replanning_experiment import (
    DynamicReplanningExperiment,
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
    )

    events = [
        DynamicObstacleEvent(
            step_index=20,
            position=Position(row=101, col=302),
            event_type=DynamicObstacleEventType.BLOCK,
        )
    ]

    experiment = DynamicReplanningExperiment(
        config=config,
        events=events,
    )

    experiment.run()


if __name__ == "__main__":
    main()
