from pathlib import Path

from pathfinding.src.core.benchmark_models import DynamicBenchmarkConfig
from pathfinding.src.experiments.dynamic_benchmark_experiment import (
    DynamicBenchmarkExperiment,
)
from pathfinding.src.experiments.experiment_config import ExperimentConfig


def main() -> None:
    config = ExperimentConfig(
        map_path=Path("../../Data/bg512-map/AR0204SR.map"),
        scen_path=Path("../../Data/bg512-scen/AR0204SR.map.scen"),
        min_optimal_length=100,
        benchmark_scenarios=20,
        results_dir=Path("../../Results/dynamic_algorithms"),
        benchmark_output_file="astar_vs_dstar_dynamic_seed_42.csv",
        path_block_lookahead=20,
        wait_when_no_path=True,
        max_wait_steps=30,
        max_simulation_steps=1500,
        moving_obstacle_prediction_steps=0,
    )

    benchmark_config = DynamicBenchmarkConfig(
        scenario_count=20,
        min_optimal_length=100,
        obstacle_count=4,
        obstacle_seed=42,
        min_obstacle_size=4,
        max_obstacle_size=10,
        path_block_lookahead=20,
        wait_when_no_path=True,
        max_wait_steps=30,
        max_stuck_steps=200,
        max_simulation_steps=1500,
        moving_obstacle_prediction_steps=0,
        progress_interval=1,
        algorithm_wall_clock_warning_s=10.0,
    )

    DynamicBenchmarkExperiment(
        config=config,
        benchmark_config=benchmark_config,
    ).run()


if __name__ == "__main__":
    main()
