from pathlib import Path

from code.src.experiments.benchmark_experiment import BenchmarkExperiment
from code.src.experiments.experiment_config import AlgorithmName, ExperimentConfig


def main() -> None:
    config = ExperimentConfig(
        map_path=Path("../../Data/bg512-map/AR0204SR.map"),
        scen_path=Path("../../Data/bg512-scen/AR0204SR.map.scen"),
        min_optimal_length=100,
        cluster_size=32,
        max_entrances_per_cluster_pair=2,
        benchmark_scenarios=100,
        results_dir=Path("../../Results/static_algorithms"),
        benchmark_output_file=(
            "astar_hpa_jps_cluster_32_scenarios_100.csv"
        )

    )

    experiment = BenchmarkExperiment(
        config=config,
        algorithms=[
            AlgorithmName.ASTAR,
            AlgorithmName.HPA_STAR,
            AlgorithmName.JPS,
        ],
    )

    experiment.run()


if __name__ == "__main__":
    main()
