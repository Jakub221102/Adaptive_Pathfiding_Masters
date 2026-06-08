from pathlib import Path

from pathfinding.src.core.dynamic_models import MovingObstacle
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
        algorithm=AlgorithmName.ASTAR,
        fps=120,
        step_delay_ms=8,
        steps_per_frame=8,
        step_record_interval=8,
        movement_hold_frames=1,
        movement_frame_delay_ms=1,
        movement_steps_per_frame=10,
        path_block_lookahead=8,
        wait_when_no_path=True,
        max_wait_steps=30,
    )

    moving_obstacles = [
        MovingObstacle(
            row=30,
            col=300,
            width=10,
            height=10,
            delta_row=-5,
            delta_col=0,
        )
    ]

    experiment = DynamicReplanningVisualizationExperiment(
        config=config,
        moving_obstacles=moving_obstacles,
    )

    experiment.run()


if __name__ == "__main__":
    main()
