import argparse
import json
import time
from pathlib import Path

from pathfinding.src.core.benchmark_models import DynamicBenchmarkConfig
from pathfinding.src.core.dynamic_models import DynamicGridMap
from pathfinding.src.experiments.algorithm_registry import create_algorithm
from pathfinding.src.experiments.dynamic_benchmark_experiment import (
    DynamicBenchmarkExperiment,
    _position_covered_by_obstacles,
    generate_scenario_obstacles,
)
from pathfinding.src.experiments.dynamic_simulation import run_dynamic_simulation
from pathfinding.src.experiments.experiment_config import AlgorithmName, ExperimentConfig
from pathfinding.src.loaders.map_loader import load_moving_ai_map
from pathfinding.src.loaders.scen_loader import load_moving_ai_scenarios
from pathfinding.src.utils.csv_exporter import export_dynamic_benchmark_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug a single dynamic benchmark scenario.",
    )
    parser.add_argument("--scenario-index", type=int, default=10)
    parser.add_argument("--scenario-seed", type=int, default=52)
    parser.add_argument("--obstacle-seed", type=int, default=42)
    parser.add_argument("--obstacle-count", type=int, default=4)
    parser.add_argument("--prediction-steps", type=int, default=0)
    parser.add_argument("--path-block-lookahead", type=int, default=20)
    parser.add_argument("--max-simulation-steps", type=int, default=1500)
    parser.add_argument(
        "--map-path",
        type=Path,
        default=Path("../../Data/bg512-map/AR0204SR.map"),
    )
    parser.add_argument(
        "--scen-path",
        type=Path,
        default=Path("../../Data/bg512-scen/AR0204SR.map.scen"),
    )
    parser.add_argument("--min-optimal-length", type=float, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../../Results/dynamic_algorithms/debug"),
    )
    parser.add_argument("--save-output", action="store_true")
    return parser.parse_args()


def load_scenario(
        map_path: Path,
        scen_path: Path,
        scenario_index: int,
        min_optimal_length: float,
):
    grid_map = load_moving_ai_map(map_path)
    scenarios = load_moving_ai_scenarios(scen_path)
    filtered_scenarios = [
        scenario for scenario in scenarios
        if scenario.optimal_length is not None
           and scenario.optimal_length >= min_optimal_length
    ]

    if scenario_index >= len(filtered_scenarios):
        raise IndexError(
            f"scenario_index={scenario_index} is out of range. "
            f"Available: {len(filtered_scenarios)}"
        )

    return grid_map, filtered_scenarios[scenario_index]


def print_obstacles(obstacles) -> None:
    print("\nGenerated obstacles:", flush=True)
    for index, obstacle in enumerate(obstacles, start=1):
        print(
            f"  Obstacle {index}: row={obstacle.row}, col={obstacle.col}, "
            f"width={obstacle.width}, height={obstacle.height}, "
            f"delta=({obstacle.delta_row}, {obstacle.delta_col})",
            flush=True,
        )


def print_stats(algorithm_label: str, stats, wall_clock_s: float) -> None:
    print(f"\n{algorithm_label} stats:", flush=True)
    print(f"  initial_path_found={stats.initial_path_found}", flush=True)
    print(f"  final_goal_reached={stats.final_goal_reached}", flush=True)
    print(f"  replanning_count={stats.replanning_count}", flush=True)
    print(f"  waiting_steps={stats.waiting_steps}", flush=True)
    print(f"  collision_count={stats.collision_count}", flush=True)
    print(f"  travelled_steps={stats.travelled_steps}", flush=True)
    print(f"  total_path_cost={stats.total_path_cost:.3f}", flush=True)
    print(f"  total_execution_time_ms={stats.total_execution_time_ms:.3f}", flush=True)
    print(f"  wall_clock_s={wall_clock_s:.3f}", flush=True)


def run_algorithm(
        algorithm_name: AlgorithmName,
        grid_map,
        scenario,
        moving_obstacles,
        benchmark_config: DynamicBenchmarkConfig,
        config: ExperimentConfig,
):
    dynamic_map = DynamicGridMap(base_map=grid_map)
    algorithm = create_algorithm(name=algorithm_name)

    wall_clock_start = time.perf_counter()
    stats, _ = run_dynamic_simulation(
        algorithm=algorithm,
        dynamic_map=dynamic_map,
        scenario=scenario,
        events=[],
        moving_obstacles=moving_obstacles,
        record_algorithm_steps=False,
        path_block_lookahead=benchmark_config.path_block_lookahead,
        wait_when_no_path=benchmark_config.wait_when_no_path,
        max_wait_steps=benchmark_config.max_wait_steps,
        max_stuck_steps=benchmark_config.max_stuck_steps,
        max_simulation_steps=benchmark_config.max_simulation_steps,
        moving_obstacle_collision_policy=config.moving_obstacle_collision_policy,
        moving_obstacle_prediction_steps=benchmark_config.moving_obstacle_prediction_steps,
    )
    wall_clock_s = time.perf_counter() - wall_clock_start
    return stats, wall_clock_s


def main() -> None:
    args = parse_args()

    config = ExperimentConfig(
        map_path=args.map_path,
        scen_path=args.scen_path,
        min_optimal_length=args.min_optimal_length,
        benchmark_scenarios=1,
        path_block_lookahead=args.path_block_lookahead,
        wait_when_no_path=True,
        max_wait_steps=30,
        max_simulation_steps=args.max_simulation_steps,
        moving_obstacle_prediction_steps=args.prediction_steps,
    )
    benchmark_config = DynamicBenchmarkConfig(
        scenario_count=1,
        min_optimal_length=args.min_optimal_length,
        obstacle_count=args.obstacle_count,
        obstacle_seed=args.obstacle_seed,
        min_obstacle_size=4,
        max_obstacle_size=10,
        path_block_lookahead=args.path_block_lookahead,
        wait_when_no_path=True,
        max_wait_steps=30,
        max_stuck_steps=200,
        max_simulation_steps=args.max_simulation_steps,
        moving_obstacle_prediction_steps=args.prediction_steps,
    )

    grid_map, scenario = load_scenario(
        map_path=args.map_path,
        scen_path=args.scen_path,
        scenario_index=args.scenario_index,
        min_optimal_length=args.min_optimal_length,
    )

    print("\nDynamic benchmark single-scenario debug:", flush=True)
    print(f"  scenario_index={args.scenario_index}", flush=True)
    print(f"  scenario_seed={args.scenario_seed}", flush=True)
    print(
        f"  start=({scenario.start.row},{scenario.start.col}) "
        f"goal=({scenario.goal.row},{scenario.goal.col})",
        flush=True,
    )
    print(f"  optimal_length={scenario.optimal_length}", flush=True)

    moving_obstacles = generate_scenario_obstacles(
        grid_map=grid_map,
        scenario=scenario,
        scenario_index=args.scenario_index,
        benchmark_config=benchmark_config,
        scenario_seed=args.scenario_seed,
    )
    print_obstacles(moving_obstacles)

    start_blocked = _position_covered_by_obstacles(
        position=scenario.start,
        obstacles=moving_obstacles,
    )
    goal_blocked = _position_covered_by_obstacles(
        position=scenario.goal,
        obstacles=moving_obstacles,
    )
    print(
        f"\nstart_blocked_by_obstacle={start_blocked}, "
        f"goal_blocked_by_obstacle={goal_blocked}",
        flush=True,
    )
    if start_blocked or goal_blocked:
        print(
            f"Warning: scenario_index={args.scenario_index} start or goal "
            "covered by generated obstacle.",
            flush=True,
        )

    experiment = DynamicBenchmarkExperiment(
        config=config,
        benchmark_config=benchmark_config,
    )
    results = []

    for algorithm_name in (AlgorithmName.ASTAR, AlgorithmName.DSTAR_LITE):
        algorithm_label = "A*" if algorithm_name == AlgorithmName.ASTAR else "D* Lite"
        stats, wall_clock_s = run_algorithm(
            algorithm_name=algorithm_name,
            grid_map=grid_map,
            scenario=scenario,
            moving_obstacles=moving_obstacles,
            benchmark_config=benchmark_config,
            config=config,
        )
        print_stats(algorithm_label=algorithm_label, stats=stats, wall_clock_s=wall_clock_s)
        results.append(
            experiment._build_result(
                algorithm_name=algorithm_name,
                scenario_index=args.scenario_index,
                scenario_seed=args.scenario_seed,
                scenario=scenario,
                stats=stats,
                wall_clock_s=wall_clock_s,
            )
        )

    if args.save_output:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_prefix = (
            f"scenario_{args.scenario_index}_seed_{args.scenario_seed}"
        )
        csv_path = args.output_dir / f"{output_prefix}.csv"
        json_path = args.output_dir / f"{output_prefix}.json"

        export_dynamic_benchmark_results(results=results, output_path=csv_path)
        json_path.write_text(
            json.dumps(
                {
                    "scenario_index": args.scenario_index,
                    "scenario_seed": args.scenario_seed,
                    "start_blocked_by_obstacle": start_blocked,
                    "goal_blocked_by_obstacle": goal_blocked,
                    "obstacles": [obstacle.model_dump() for obstacle in moving_obstacles],
                    "results": [result.model_dump() for result in results],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved CSV: {csv_path}", flush=True)
        print(f"Saved JSON: {json_path}", flush=True)


if __name__ == "__main__":
    main()
