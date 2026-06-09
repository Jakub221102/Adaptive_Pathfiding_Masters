import argparse
import csv
from dataclasses import dataclass
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
DEFAULT_RESULTS_CSV = Path(
    "../../Results/dynamic_algorithms/astar_vs_dstar_dynamic_seed_42.csv",
)
DEFAULT_SCENARIO_INDICES = (10, 14)

_SLOWEST_BY_CHOICES = ("wall_clock", "algorithm_time", "path_cost")

_ALGORITHM_CHOICES = {
    "astar": AlgorithmName.ASTAR,
    "a*": AlgorithmName.ASTAR,
    "dstar": AlgorithmName.DSTAR_LITE,
    "dstar_lite": AlgorithmName.DSTAR_LITE,
    "d*": AlgorithmName.DSTAR_LITE,
    "d* lite": AlgorithmName.DSTAR_LITE,
}

_ALGORITHM_LABELS = {
    AlgorithmName.ASTAR: "A*",
    AlgorithmName.DSTAR_LITE: "D* Lite",
}


@dataclass(frozen=True)
class BenchmarkCsvRow:
    algorithm: str
    scenario_index: int
    seed: int
    found: bool
    final_goal_reached: bool
    initial_path_found: bool
    replanning_count: int
    waiting_steps: int
    collision_count: int
    agent_push_count: int
    obstacle_blocked_count: int
    travelled_steps: int
    total_path_cost: float
    total_execution_time_ms: float
    wall_clock_s: float
    obstacle_count: int
    prediction_steps: int
    path_block_lookahead: int
    start_row: int
    start_col: int
    goal_row: int
    goal_col: int
    optimal_length: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Animate dynamic benchmark scenarios with the same obstacles "
            "and settings as dynamic_benchmark_main.py."
        ),
        epilog=(
            "Examples:\n"
            "  py dynamic_benchmark_scenario_viewer.py 14 astar\n"
            "  py dynamic_benchmark_scenario_viewer.py 14 dstar\n"
            "  py dynamic_benchmark_scenario_viewer.py --scenario-index 14 --algorithm astar\n"
            "  py dynamic_benchmark_scenario_viewer.py --slowest"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        type=int,
        metavar="INDEX",
        help="Benchmark scenario index, e.g. 14 (shorthand for --scenario-index).",
    )
    parser.add_argument(
        "algorithm_choice",
        nargs="?",
        choices=sorted(_ALGORITHM_CHOICES),
        metavar="ALGORITHM",
        help="Algorithm shorthand, e.g. astar or dstar (shorthand for --algorithm).",
    )
    parser.add_argument(
        "--scenario-index",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Benchmark scenario indices to visualize "
            f"(default: {list(DEFAULT_SCENARIO_INDICES)})."
        ),
    )
    parser.add_argument(
        "--slowest",
        action="store_true",
        help=(
            "Visualize the slowest scenario from benchmark CSV "
            "(overrides --scenario-index)."
        ),
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help="Benchmark results CSV used with --slowest.",
    )
    parser.add_argument(
        "--slowest-by",
        choices=_SLOWEST_BY_CHOICES,
        default="wall_clock",
        help="Metric for picking the slowest scenario (default: wall_clock).",
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
    parser.add_argument("--step-record-interval", type=int, default=10)
    parser.add_argument(
        "--with-search-animation",
        action="store_true",
        help=(
            "Record algorithm search steps for richer animation. "
            "May change D* Lite behavior; default reproduces benchmark logic."
        ),
    )
    args = parser.parse_args()
    apply_positional_selection(args)
    return args


def apply_positional_selection(args: argparse.Namespace) -> None:
    if args.scenario is not None:
        args.scenario_index = [args.scenario]

    if args.algorithm_choice is not None:
        args.algorithm = args.algorithm_choice
        args.all_algorithms = False


def _parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _parse_optional_float(value: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def load_benchmark_csv(results_csv: Path) -> list[BenchmarkCsvRow]:
    if not results_csv.exists():
        raise FileNotFoundError(f"Benchmark results CSV not found: {results_csv}")

    rows: list[BenchmarkCsvRow] = []
    with results_csv.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                BenchmarkCsvRow(
                    algorithm=row["algorithm"],
                    scenario_index=int(row["scenario_index"]),
                    seed=int(row["seed"]),
                    found=_parse_bool(row["found"]),
                    final_goal_reached=_parse_bool(row["final_goal_reached"]),
                    initial_path_found=_parse_bool(row["initial_path_found"]),
                    replanning_count=int(row["replanning_count"]),
                    waiting_steps=int(row["waiting_steps"]),
                    collision_count=int(row["collision_count"]),
                    agent_push_count=int(row["agent_push_count"]),
                    obstacle_blocked_count=int(row["obstacle_blocked_count"]),
                    travelled_steps=int(row["travelled_steps"]),
                    total_path_cost=float(row["total_path_cost"]),
                    total_execution_time_ms=float(row["total_execution_time_ms"]),
                    wall_clock_s=float(row["wall_clock_s"]),
                    obstacle_count=int(row["obstacle_count"]),
                    prediction_steps=int(row["prediction_steps"]),
                    path_block_lookahead=int(row["path_block_lookahead"]),
                    start_row=int(row["start_row"]),
                    start_col=int(row["start_col"]),
                    goal_row=int(row["goal_row"]),
                    goal_col=int(row["goal_col"]),
                    optimal_length=_parse_optional_float(row["optimal_length"]),
                )
            )

    if not rows:
        raise ValueError(f"No benchmark rows found in: {results_csv}")

    return rows


def _slowest_sort_key(metric: str):
    if metric == "algorithm_time":
        return lambda row: row.total_execution_time_ms
    if metric == "path_cost":
        return lambda row: row.total_path_cost
    return lambda row: row.wall_clock_s


def select_slowest_benchmark_row(
        rows: list[BenchmarkCsvRow],
        metric: str,
        algorithm: AlgorithmName | None = None,
) -> BenchmarkCsvRow:
    candidates = rows
    if algorithm is not None:
        algorithm_label = _ALGORITHM_LABELS[algorithm]
        candidates = [row for row in rows if row.algorithm == algorithm_label]

    if not candidates:
        raise ValueError("No benchmark rows match the selected algorithm filter.")

    return max(candidates, key=_slowest_sort_key(metric))


def csv_row_to_algorithm(row: BenchmarkCsvRow) -> AlgorithmName:
    normalized = row.algorithm.strip().lower()
    if normalized in {"a*", "astar"}:
        return AlgorithmName.ASTAR
    if normalized in {"d* lite", "dstar", "dstar_lite", "d*"}:
        return AlgorithmName.DSTAR_LITE
    raise ValueError(f"Unsupported algorithm in benchmark CSV: {row.algorithm}")


def find_benchmark_row(
        rows: list[BenchmarkCsvRow],
        scenario_index: int,
        algorithm: AlgorithmName,
) -> BenchmarkCsvRow | None:
    algorithm_label = _ALGORITHM_LABELS[algorithm]
    for row in rows:
        if row.scenario_index == scenario_index and row.algorithm == algorithm_label:
            return row
    return None


def print_benchmark_row_stats(
        row: BenchmarkCsvRow,
        metric: str,
        *,
        from_csv: bool = True,
) -> None:
    metric_label = {
        "wall_clock": "wall_clock_s",
        "algorithm_time": "total_execution_time_ms",
        "path_cost": "total_path_cost",
    }[metric]
    metric_value = _slowest_sort_key(metric)(row)

    source = "saved benchmark CSV" if from_csv else "expected benchmark settings"
    print(f"\nBenchmark result for selected scenario ({source}):", flush=True)
    if from_csv:
        print(
            f"  metric={metric} ({metric_label}={metric_value})",
            flush=True,
        )
    print(
        f"  algorithm={row.algorithm} scenario_index={row.scenario_index} "
        f"seed={row.seed}",
        flush=True,
    )
    print(
        f"  start=({row.start_row},{row.start_col}) "
        f"goal=({row.goal_row},{row.goal_col}) "
        f"optimal_length={row.optimal_length}",
        flush=True,
    )
    print(
        f"  reached={row.final_goal_reached} replans={row.replanning_count} "
        f"waits={row.waiting_steps} collisions={row.collision_count}",
        flush=True,
    )
    print(
        f"  travelled_steps={row.travelled_steps} "
        f"path_cost={row.total_path_cost:.3f}",
        flush=True,
    )
    print(
        f"  algorithm_time={row.total_execution_time_ms:.2f} ms "
        f"wall_clock={row.wall_clock_s:.2f} s",
        flush=True,
    )
    print(
        f"  obstacle_count={row.obstacle_count} "
        f"prediction_steps={row.prediction_steps} "
        f"path_block_lookahead={row.path_block_lookahead}",
        flush=True,
    )


def resolve_view_targets(args: argparse.Namespace) -> tuple[list[int], list[AlgorithmName], BenchmarkCsvRow | None]:
    if args.slowest:
        benchmark_rows = load_benchmark_csv(args.results_csv)
        algorithm_filter = None if args.all_algorithms else _ALGORITHM_CHOICES[args.algorithm]
        slowest_row = select_slowest_benchmark_row(
            rows=benchmark_rows,
            metric=args.slowest_by,
            algorithm=algorithm_filter,
        )

        scenario_indices = [slowest_row.scenario_index]
        if args.all_algorithms:
            algorithms = [AlgorithmName.ASTAR, AlgorithmName.DSTAR_LITE]
        else:
            algorithms = [csv_row_to_algorithm(slowest_row)]
        return scenario_indices, algorithms, slowest_row

    scenario_indices = (
        args.scenario_index
        if args.scenario_index is not None
        else list(DEFAULT_SCENARIO_INDICES)
    )
    algorithms = resolve_algorithms(args)
    return scenario_indices, algorithms, None


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
        min_obstacle_spacing=25,
        path_margin=30,
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
        record_algorithm_steps=args.with_search_animation,
    )


def algorithm_label(algorithm: AlgorithmName) -> str:
    return _ALGORITHM_LABELS[algorithm]


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
        benchmark_rows: list[BenchmarkCsvRow] | None = None,
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

    if benchmark_rows is not None:
        benchmark_row = find_benchmark_row(
            rows=benchmark_rows,
            scenario_index=scenario_index,
            algorithm=algorithm,
        )
        if benchmark_row is not None:
            print_benchmark_row_stats(row=benchmark_row, metric=args.slowest_by)
            if not benchmark_row.final_goal_reached:
                print(
                    "  Note: benchmark marked this run as not reaching the goal. "
                    "Viewer now uses the same simulation limits as the benchmark.",
                    flush=True,
                )
            if args.with_search_animation:
                print(
                    "  Warning: --with-search-animation can change D* Lite behavior "
                    "and may not match benchmark results.",
                    flush=True,
                )
        else:
            print(
                "  No matching benchmark row found in CSV for this scenario/algorithm.",
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
    scenario_indices, algorithms, _ = resolve_view_targets(args)
    benchmark_rows = (
        load_benchmark_csv(args.results_csv)
        if args.results_csv.exists()
        else None
    )

    print("\nDynamic benchmark scenario viewer:", flush=True)
    print(f"  scenario_indices={scenario_indices}", flush=True)
    print(
        "  algorithms="
        f"{[algorithm_label(algorithm) for algorithm in algorithms]}",
        flush=True,
    )
    print(f"  obstacle_seed={args.obstacle_seed}", flush=True)
    print(
        "  benchmark_faithful_simulation="
        f"{not args.with_search_animation}",
        flush=True,
    )
    if args.slowest:
        print(f"  results_csv={args.results_csv}", flush=True)
        print(f"  slowest_by={args.slowest_by}", flush=True)

    for scenario_index in scenario_indices:
        for algorithm in algorithms:
            run_scenario_view(
                args=args,
                scenario_index=scenario_index,
                algorithm=algorithm,
                benchmark_config=benchmark_config,
                benchmark_rows=benchmark_rows,
            )


if __name__ == "__main__":
    main()
