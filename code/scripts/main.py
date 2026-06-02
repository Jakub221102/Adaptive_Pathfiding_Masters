from pathlib import Path

from code.src.experiments.experiment_config import AlgorithmName, ExperimentConfig
from code.src.experiments.pathfinding_experiment import PathfindingExperiment
from code.src.visualization.pygame_models import ViewerMode


def main() -> None:
    config = ExperimentConfig(
        map_path=Path("../../Data/bg512-map/AR0204SR.map"),
        scen_path=Path("../../Data/bg512-scen/AR0204SR.map.scen"),
        algorithm=AlgorithmName.JPS,
        viewer_mode=ViewerMode.STATIC,
        scenario_index=19,
        min_optimal_length=100,
    )

    experiment = PathfindingExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
