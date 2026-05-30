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
        step_record_interval=10,
        cluster_size=32,
        animated_comparison=True,
        comparison_animation_time_ms=8000,
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