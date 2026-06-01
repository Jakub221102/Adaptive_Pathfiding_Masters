from pathlib import Path

from code.src.experiments.benchmark_experiment import BenchmarkExperiment
from code.src.experiments.experiment_config import AlgorithmName, ExperimentConfig


def run_max_entrances_benchmark(
        max_entrances_values: list[int],
        cluster_size: int,
        benchmark_scenarios: int,
) -> None:
    for max_entrances in max_entrances_values:
        config = ExperimentConfig(
            map_path=Path("../../Data/bg512-map/AR0204SR.map"),
            scen_path=Path("../../Data/bg512-scen/AR0204SR.map.scen"),
            min_optimal_length=100,
            cluster_size=cluster_size,
            max_entrances_per_cluster_pair=max_entrances,
            benchmark_scenarios=benchmark_scenarios,
            results_dir=Path("../../Results/max_entrances"),
            benchmark_output_file=(
                f"hpa_cluster_{cluster_size}_entrances_{max_entrances}.csv"
            ),
        )

        experiment = BenchmarkExperiment(
            config=config,
            algorithms=[
                AlgorithmName.HPA_STAR,
            ],
        )

        print(
            f"\nRunning HPA* benchmark for "
            f"cluster_size={cluster_size}, "
            f"max_entrances={max_entrances}"
        )

        experiment.run()


def main() -> None:
    run_max_entrances_benchmark(
        max_entrances_values=[1, 2, 4, 8, 16],
        cluster_size=128,
        benchmark_scenarios=100,
    )


if __name__ == "__main__":
    main()
