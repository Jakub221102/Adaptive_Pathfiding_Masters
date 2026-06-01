from pathlib import Path

from code.src.experiments.benchmark_experiment import BenchmarkExperiment
from code.src.experiments.experiment_config import AlgorithmName, ExperimentConfig


def main() -> None:
    config = ExperimentConfig(
        map_path=Path("../../Data/bg512-map/AR0204SR.map"),
        scen_path=Path("../../Data/bg512-scen/AR0204SR.map.scen"),
        min_optimal_length=100,
        cluster_size=32,
        benchmark_scenarios=100,
        results_dir=Path("../../Results/HPA_vs_ASTAR"),
        benchmark_output_file=f"cluster_{32}_scenarios_{100}.csv",
    )

    experiment = BenchmarkExperiment(
        config=config,
        algorithms=[
            AlgorithmName.ASTAR,
            AlgorithmName.HPA_STAR,
        ],
    )

    experiment.run()


if __name__ == "__main__":
    main()
