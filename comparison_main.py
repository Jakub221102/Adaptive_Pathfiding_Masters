from pathlib import Path

from src.experiments.comparison_experiment import ComparisonExperiment
from src.experiments.experiment_config import AlgorithmName, ExperimentConfig


def main() -> None:
    config = ExperimentConfig(
        map_path=Path("Data/bg512-map/AR0204SR.map"),
        scen_path=Path("Data/bg512-scen/AR0204SR.map.scen"),
        scenario_index=400,
        min_optimal_length=100,
        fps=120,
    )

    experiment = ComparisonExperiment(
        config=config,
        algorithms=[
            AlgorithmName.ASTAR,
            AlgorithmName.HPA_STAR,
        ],
    )

    experiment.run()


if __name__ == "__main__":
    main()