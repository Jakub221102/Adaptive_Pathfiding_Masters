from pathlib import Path

from src.experiments.experiment_config import AlgorithmName, ExperimentConfig
from src.experiments.pathfinding_experiment import PathfindingExperiment
from src.visualization.pygame_models import ViewerMode


def main() -> None:
    config = ExperimentConfig(
        map_path=Path("Data/bg512-map/AR0204SR.map"),
        scen_path=Path("Data/bg512-scen/AR0204SR.map.scen"),
        algorithm=AlgorithmName.ASTAR,
        viewer_mode = ViewerMode.STATIC,
        scenario_index=400,
        min_optimal_length=100,
        fps=120,
        step_delay_ms=1,
        steps_per_frame=100,
        step_record_interval=10,
    )

    experiment = PathfindingExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()