import argparse
from pathlib import Path

from pathfinding.src.core.benchmark_models import DynamicBenchmarkConfig
from pathfinding.src.experiments.dynamic_benchmark_experiment import (
    generate_valid_scenario_obstacles,
)
from pathfinding.src.experiments.dynamic_replanning_visualization_experiment import (
    DynamicReplanningVisualizationExperiment,
)
from pathfinding.src.experiments.experiment_config import AlgorithmName, ExperimentConfig
from pathfinding.src.loaders.map_loader import load_moving_ai_map
from pathfinding.src.loaders.scen_loader import load_moving_ai_scenarios

DEFAULT_MAP_PATH = Path("../../Data/bg512-map/AR0204SR.map")
DEFAULT_SCEN_PATH = Path("../../Data/bg512-scen/AR0204SR.map.scen")
DEFAULT_SCENARIO_INDICES = (10, 14)

_ALGORITHM_CHOICES = {
    "astar": AlgorithmName.ASTAR,
    "a*": AlgorithmName.ASTAR,
    "dstar": AlgorithmName.DSTAR_LITE,
    "dstar_lite": AlgorithmName.DSTAR_LITE,
    "d*": AlgorithmName.DSTAR_LITE,
    "d* lite": AlgorithmName.DSTAR_LITE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Animate dynamic benchmark scenarios with the same obstacles "
            "and settings as dynamic_benchmark_main.py."
        ),
    )
    parser.add_argument(
        "--scenario-index",
        type=int,
        nargs="+",
        default=list(DEFAULT_SCENARIO_INDICES),
        help="Benchmark scenario indices to visualize (default: 10 14).",
    )
    parser.add_argument(
        "--algorithm",
        choices=sorted(_ALGORITHM_CHOICES),
        default="astar",
        help="Algorithm to visualize (default: astar).",
    )
    parser.add_argument(
        "--all-algorithms",
        action="store_true",
        help="Run both A* and D* Lite for each selected scenario.",
    )
    parser.add_argument("--map-path", type=Path, default=DEFAULT_MAP_PATH)
    parser.add_argument("--scen-path", type=Path, default=DEFAULT_SCEN_PATH)
    parser.add_argument("--min-optimal-length", type=float, default=100)
    parser.add_argument("--obstacle-seed", type=int, default=42)
    parser.add_argument("--obstacle-count", type=int, default=4)
    parser.add_argument("--prediction-steps", type=int, default=0)
    parser.add_argument("--path-block-lookahead", type=int, default=20)
    parser.add_argument("--max-simulation-steps", type=int, default=1500)
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--step-delay-ms", type=int, default=1)
    parser.add_argument("--steps-per-frame", type=int, default=200)
    parser.add_argument("--step-record-interval", type=int, default=40)
    return parser.parse_args()


def load_benchmark_scenario(
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


def build_benchmark_config(args: argparse.Namespace) -> DynamicBenchmarkConfig:
    return DynamicBenchmarkConfig(
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


def build_visualization_config(
        args: argparse.Namespace,
        algorithm: AlgorithmName,
        scenario_index: int,
) -> ExperimentConfig:
    return ExperimentConfig(
        map_path=args.map_path,
        scen_path=args.scen_path,
        scenario_index=scenario_index,
        min_optimal_length=args.min_optimal_length,
        algorithm=algorithm,
        fps=args.fps,
        step_delay_ms=args.step_delay_ms,
        steps_per_frame=args.steps_per_frame,
        step_record_interval=args.step_record_interval,
        movement_hold_frames=1,
        movement_frame_delay_ms=0,
        movement_steps_per_frame=10,
        obstacle_hold_frames=1,
        obstacle_frame_delay_ms=0,
        path_block_lookahead=args.path_block_lookahead,
        wait_when_no_path=True,
        max_wait_steps=30,
        max_stuck_steps=200,
        max_simulation_steps=args.max_simulation_steps,
        moving_obstacle_prediction_steps=args.prediction_steps,
    )


def algorithm_label(algorithm: AlgorithmName) -> str:
    if algorithm == AlgorithmName.ASTAR:
        return "A*"
    return "D* Lite"


def print_obstacles(obstacles) -> None:
    print("Generated obstacles:", flush=True)
    for index, obstacle in enumerate(obstacles, start=1):
        print(
            f"  Obstacle {index}: row={obstacle.row}, col={obstacle.col}, "
            f"width={obstacle.width}, height={obstacle.height}, "
            f"delta=({obstacle.delta_row}, {obstacle.delta_col})",
            flush=True,
        )


def run_scenario_view(
        args: argparse.Namespace,
        scenario_index: int,
        algorithm: AlgorithmName,
        benchmark_config: DynamicBenchmarkConfig,
) -> None:
    grid_map, scenario = load_benchmark_scenario(
        map_path=args.map_path,
        scen_path=args.scen_path,
        scenario_index=scenario_index,
        min_optimal_length=args.min_optimal_length,
    )

    moving_obstacles, scenario_seed = generate_valid_scenario_obstacles(
        grid_map=grid_map,
        scenario=scenario,
        scenario_index=scenario_index,
        benchmark_config=benchmark_config,
    )

    label = algorithm_label(algorithm)
    print(
        f"\nAnimating benchmark scenario_index={scenario_index}, "
        f"seed={scenario_seed}, algorithm={label}",
        flush=True,
    )
    print(
        f"  start=({scenario.start.row},{scenario.start.col}) "
        f"goal=({scenario.goal.row},{scenario.goal.col}) "
        f"optimal_length={scenario.optimal_length}",
        flush=True,
    )

    if moving_obstacles is None:
        print(
            f"  Skipping scenario_index={scenario_index}: "
            "could not generate obstacles without blocking start or goal.",
            flush=True,
        )
        return

    print_obstacles(moving_obstacles)

    config = build_visualization_config(
        args=args,
        algorithm=algorithm,
        scenario_index=scenario_index,
    )
    experiment = DynamicReplanningVisualizationExperiment(
        config=config,
        moving_obstacles=moving_obstacles,
        grid_map=grid_map,
        scenario=scenario,
    )
    experiment.run()


def resolve_algorithms(args: argparse.Namespace) -> list[AlgorithmName]:
    if args.all_algorithms:
        return [AlgorithmName.ASTAR, AlgorithmName.DSTAR_LITE]
    return [_ALGORITHM_CHOICES[args.algorithm]]


def main() -> None:
    args = parse_args()
    benchmark_config = build_benchmark_config(args)
    algorithms = resolve_algorithms(args)

    print("\nDynamic benchmark scenario viewer:", flush=True)
    print(f"  scenario_indices={args.scenario_index}", flush=True)
    print(
        "  algorithms="
        f"{[algorithm_label(algorithm) for algorithm in algorithms]}",
        flush=True,
    )
    print(f"  obstacle_seed={args.obstacle_seed}", flush=True)

    for scenario_index in args.scenario_index:
        for algorithm in algorithms:
            run_scenario_view(
                args=args,
                scenario_index=scenario_index,
                algorithm=algorithm,
                benchmark_config=benchmark_config,
            )


if __name__ == "__main__":
    main()
