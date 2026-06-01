from pathlib import Path

from code.src.experiments.benchmark_experiment import BenchmarkExperiment
from code.src.experiments.experiment_config import AlgorithmName, ExperimentConfig


def run_cluster_size_benchmark(
        cluster_sizes: list[int],
        benchmark_scenarios: int,
) -> None:
    for cluster_size in cluster_sizes:
        config = ExperimentConfig(
            map_path=Path("../../Data/bg512-map/AR0204SR.map"),
            scen_path=Path("../../Data/bg512-scen/AR0204SR.map.scen"),
            min_optimal_length=100,
            cluster_size=cluster_size,
            benchmark_scenarios=benchmark_scenarios,
            results_dir=Path("../../Results/cluster_size"),
            benchmark_output_file=f"hpa_cluster_{cluster_size}.csv",
        )

        experiment = BenchmarkExperiment(
            config=config,
            algorithms=[
                AlgorithmName.HPA_STAR,
            ],
        )

        print(f"\nRunning HPA* benchmark for cluster_size={cluster_size}")
        experiment.run()


def main() -> None:
    run_cluster_size_benchmark(
        cluster_sizes=[16, 32, 64, 128],
        benchmark_scenarios=100,
    )


if __name__ == "__main__":
    main()
