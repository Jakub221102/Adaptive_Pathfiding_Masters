from pathlib import Path

from pathfinding.src.experiments.experiment_config import AlgorithmName, ExperimentConfig
from pathfinding.src.experiments.pathfinding_experiment import PathfindingExperiment
from pathfinding.src.visualization.pygame_models import ViewerMode


def main() -> None:
    config = ExperimentConfig(
        map_path=Path("../../Data/bg512-map/AR0204SR.map"),
        scen_path=Path("../../Data/bg512-scen/AR0204SR.map.scen"),
        algorithm=AlgorithmName.ASTAR,
        viewer_mode=ViewerMode.ANIMATED,
        scenario_index=1,
        min_optimal_length=400,
        show_heatmap_overlay=True,
        step_record_interval=5,
    )

    experiment = PathfindingExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
