import statistics
import time

from pathfinding.src.algorithms.astar import AStar
from pathfinding.src.core.benchmark_models import (
    DynamicBenchmarkConfig,
    DynamicBenchmarkResult,
)
from pathfinding.src.core.dynamic_models import (
    DynamicGridMap,
    DynamicReplanningStats,
    MovingObstacle,
    get_occupied_positions,
)
from pathfinding.src.core.models import GridMap, Position, Scenario
from pathfinding.src.core.moving_obstacle_generator import (
    MovingObstacleGeneratorConfig,
    generate_moving_obstacles,
)
from pathfinding.src.experiments.algorithm_registry import create_algorithm
from pathfinding.src.experiments.base_experiment import BaseExperiment
from pathfinding.src.experiments.dynamic_simulation import (
    _build_failed_stats,
    run_dynamic_simulation,
)
from pathfinding.src.experiments.experiment_config import AlgorithmName, ExperimentConfig
from pathfinding.src.utils.csv_exporter import export_dynamic_benchmark_results

_ALGORITHM_LABELS = {
    AlgorithmName.ASTAR: "A*",
    AlgorithmName.DSTAR_LITE: "D* Lite",
}

_OBSTACLE_GENERATION_MAX_ATTEMPTS = 10
_OUTLIER_REPORT_LIMIT = 5


def is_dynamic_scenario(stats: DynamicReplanningStats) -> bool:
    return (
            stats.replanning_count > 0
            or stats.waiting_steps > 0
            or stats.collision_count > 0
    )


def is_dynamic_result(result: DynamicBenchmarkResult) -> bool:
    return (
            result.replanning_count > 0
            or result.waiting_steps > 0
            or result.collision_count > 0
    )


def path_bounds_with_margin(
        path: list[Position],
        grid_map: GridMap,
        margin: int,
) -> tuple[int, int, int, int]:
    rows = [position.row for position in path]
    cols = [position.col for position in path]

    min_row = max(0, min(rows) - margin)
    max_row = min(grid_map.height - 1, max(rows) + margin)
    min_col = max(0, min(cols) - margin)
    max_col = min(grid_map.width - 1, max(cols) + margin)

    return min_row, max_row, min_col, max_col


def _position_covered_by_obstacles(
        position: Position,
        obstacles: list[MovingObstacle],
) -> bool:
    position_tuple = (position.row, position.col)

    for obstacle in obstacles:
        for occupied_position in get_occupied_positions(obstacle):
            if (occupied_position.row, occupied_position.col) == position_tuple:
                return True

    return False


def obstacles_block_positions(
        obstacles: list[MovingObstacle],
        start: Position,
        goal: Position,
) -> bool:
    return (
            _position_covered_by_obstacles(position=start, obstacles=obstacles)
            or _position_covered_by_obstacles(position=goal, obstacles=obstacles)
    )


def generate_scenario_obstacles(
        grid_map: GridMap,
        scenario: Scenario,
        scenario_index: int,
        benchmark_config: DynamicBenchmarkConfig,
        *,
        seed_offset: int = 0,
        scenario_seed: int | None = None,
) -> list[MovingObstacle]:
    reference_path = AStar().find_path(
        grid_map=grid_map,
        start=scenario.start,
        goal=scenario.goal,
    ).path

    if reference_path:
        min_row, max_row, min_col, max_col = path_bounds_with_margin(
            path=reference_path,
            grid_map=grid_map,
            margin=benchmark_config.path_margin,
        )
    else:
        min_row = 0
        max_row = grid_map.height - 1
        min_col = 0
        max_col = grid_map.width - 1

    if scenario_seed is None:
        scenario_seed = benchmark_config.obstacle_seed + scenario_index + seed_offset

    forbidden_positions = {
        (scenario.start.row, scenario.start.col),
        (scenario.goal.row, scenario.goal.col),
    }

    return generate_moving_obstacles(
        grid_map=grid_map,
        config=MovingObstacleGeneratorConfig(
            seed=scenario_seed,
            obstacle_count=benchmark_config.obstacle_count,
            min_width=benchmark_config.min_obstacle_size,
            max_width=benchmark_config.max_obstacle_size,
            min_height=benchmark_config.min_obstacle_size,
            max_height=benchmark_config.max_obstacle_size,
            min_row=min_row,
            max_row=max_row,
            min_col=min_col,
            max_col=max_col,
            min_obstacle_spacing=benchmark_config.min_obstacle_spacing,
        ),
        forbidden_positions=forbidden_positions,
    )


def generate_valid_scenario_obstacles(
        grid_map: GridMap,
        scenario: Scenario,
        scenario_index: int,
        benchmark_config: DynamicBenchmarkConfig,
) -> tuple[list[MovingObstacle] | None, int]:
    for attempt in range(_OBSTACLE_GENERATION_MAX_ATTEMPTS):
        obstacles = generate_scenario_obstacles(
            grid_map=grid_map,
            scenario=scenario,
            scenario_index=scenario_index,
            benchmark_config=benchmark_config,
            seed_offset=attempt,
        )

        if obstacles_block_positions(
                obstacles=obstacles,
                start=scenario.start,
                goal=scenario.goal,
        ):
            print(
                f"Warning: scenario_index={scenario_index} start or goal "
                "covered by generated obstacle.",
                flush=True,
            )
            continue

        return obstacles, benchmark_config.obstacle_seed + scenario_index + attempt

    return None, benchmark_config.obstacle_seed + scenario_index


def compute_distribution_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "avg": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }

    sorted_values = sorted(values)
    return {
        "avg": sum(values) / len(values),
        "median": statistics.median(values),
        "p95": percentile(sorted_values, 0.95),
        "max": max(values),
    }


def percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0

    if len(sorted_values) == 1:
        return sorted_values[0]

    index = quantile * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def compute_dynamic_interaction_rate(
        results: list[DynamicBenchmarkResult],
) -> float:
    if not results:
        return 0.0

    dynamic_count = sum(1 for result in results if is_dynamic_result(result))
    return dynamic_count / len(results)


class DynamicBenchmarkExperiment(BaseExperiment):
    def __init__(
            self,
            config: ExperimentConfig,
            benchmark_config: DynamicBenchmarkConfig | None = None,
            *,
            grid_map: GridMap | None = None,
            scenarios: list[Scenario] | None = None,
    ) -> None:
        super().__init__(config)
        self.benchmark_config = benchmark_config or DynamicBenchmarkConfig()
        self._grid_map_override = grid_map
        self._scenarios_override = scenarios

    def run(self) -> list[DynamicBenchmarkResult]:
        if self._grid_map_override is not None and self._scenarios_override is not None:
            grid_map = self._grid_map_override
            scenarios = self._scenarios_override
        else:
            grid_map, scenarios = self._load_filtered_scenarios()

        min_length = max(
            self.config.min_optimal_length,
            self.benchmark_config.min_optimal_length,
        )
        filtered_scenarios = [
            scenario for scenario in scenarios
            if scenario.optimal_length is not None
               and scenario.optimal_length >= min_length
        ]

        if not filtered_scenarios:
            raise ValueError("No scenarios match the selected filter.")

        scenario_count = min(
            self.benchmark_config.scenario_count,
            self.config.benchmark_scenarios,
            len(filtered_scenarios),
        )
        selected_scenarios = filtered_scenarios[:scenario_count]

        self._print_benchmark_start(scenario_count=scenario_count)

        results: list[DynamicBenchmarkResult] = []
        dynamic_scenario_flags: dict[int, bool] = {}

        for scenario_index, scenario in enumerate(selected_scenarios):
            moving_obstacles, scenario_seed = generate_valid_scenario_obstacles(
                grid_map=grid_map,
                scenario=scenario,
                scenario_index=scenario_index,
                benchmark_config=self.benchmark_config,
            )

            if moving_obstacles is None:
                print(
                    f"\nRunning scenario {scenario_index + 1}/{scenario_count}: "
                    f"scenario_index={scenario_index}, seed={scenario_seed}, "
                    f"start=({scenario.start.row},{scenario.start.col}), "
                    f"goal=({scenario.goal.row},{scenario.goal.col}), "
                    f"optimal_length={_format_optimal_length(scenario)}",
                    flush=True,
                )
                print(
                    f"  Skipping scenario_index={scenario_index}: "
                    "obstacles block start or goal after "
                    f"{_OBSTACLE_GENERATION_MAX_ATTEMPTS} attempts",
                    flush=True,
                )
                for algorithm_name in (AlgorithmName.ASTAR, AlgorithmName.DSTAR_LITE):
                    failed_stats = _build_failed_stats(
                        algorithm_name=_ALGORITHM_LABELS[algorithm_name],
                        total_execution_time_ms=0.0,
                    )
                    results.append(
                        self._build_result(
                            algorithm_name=algorithm_name,
                            scenario_index=scenario_index,
                            scenario_seed=scenario_seed,
                            scenario=scenario,
                            stats=failed_stats,
                            wall_clock_s=0.0,
                        )
                    )
                continue

            start_blocked = _position_covered_by_obstacles(
                position=scenario.start,
                obstacles=moving_obstacles,
            )
            goal_blocked = _position_covered_by_obstacles(
                position=scenario.goal,
                obstacles=moving_obstacles,
            )
            if start_blocked or goal_blocked:
                print(
                    f"Warning: scenario_index={scenario_index} start or goal "
                    "covered by generated obstacle.",
                    flush=True,
                )

            self._print_scenario_start(
                scenario_number=scenario_index + 1,
                scenario_count=scenario_count,
                scenario_index=scenario_index,
                scenario_seed=scenario_seed,
                scenario=scenario,
            )

            for algorithm_name in (AlgorithmName.ASTAR, AlgorithmName.DSTAR_LITE):
                algorithm_label = _ALGORITHM_LABELS[algorithm_name]
                print(f"  Running {algorithm_label}...", flush=True)

                wall_clock_start = time.perf_counter()
                try:
                    stats = self._run_algorithm(
                        algorithm_name=algorithm_name,
                        grid_map=grid_map,
                        scenario=scenario,
                        moving_obstacles=moving_obstacles,
                    )
                except Exception as error:
                    wall_clock_s = time.perf_counter() - wall_clock_start
                    print(
                        f"ERROR: {algorithm_label} scenario_index={scenario_index}: "
                        f"{error}",
                        flush=True,
                    )
                    stats = _build_failed_stats(
                        algorithm_name=algorithm_label,
                        total_execution_time_ms=0.0,
                    )
                    self._print_algorithm_result(
                        algorithm_label=algorithm_label,
                        stats=stats,
                        wall_clock_s=wall_clock_s,
                    )
                    self._print_immediate_failure_warning(
                        algorithm_label=algorithm_label,
                        scenario_index=scenario_index,
                        stats=stats,
                    )
                    results.append(
                        self._build_result(
                            algorithm_name=algorithm_name,
                            scenario_index=scenario_index,
                            scenario_seed=scenario_seed,
                            scenario=scenario,
                            stats=stats,
                            wall_clock_s=wall_clock_s,
                        )
                    )
                    continue

                wall_clock_s = time.perf_counter() - wall_clock_start
                self._print_algorithm_result(
                    algorithm_label=algorithm_label,
                    stats=stats,
                    wall_clock_s=wall_clock_s,
                )
                self._print_immediate_failure_warning(
                    algorithm_label=algorithm_label,
                    scenario_index=scenario_index,
                    stats=stats,
                )

                if wall_clock_s > self.benchmark_config.algorithm_wall_clock_warning_s:
                    print(
                        f"Warning: {algorithm_label} on scenario_index="
                        f"{scenario_index} took {wall_clock_s:.1f}s wall-clock",
                        flush=True,
                    )

                if is_dynamic_scenario(stats):
                    dynamic_scenario_flags[scenario_index] = True

                results.append(
                    self._build_result(
                        algorithm_name=algorithm_name,
                        scenario_index=scenario_index,
                        scenario_seed=scenario_seed,
                        scenario=scenario,
                        stats=stats,
                        wall_clock_s=wall_clock_s,
                    )
                )

            completed_scenarios = scenario_index + 1
            if (
                    completed_scenarios % self.benchmark_config.progress_interval == 0
                    or completed_scenarios == scenario_count
            ):
                self._print_progress(
                    completed_scenarios=completed_scenarios,
                    scenario_count=scenario_count,
                    results=results,
                    dynamic_scenario_flags=dynamic_scenario_flags,
                )

        self._print_summary(
            results=results,
            dynamic_scenario_flags=dynamic_scenario_flags,
            scenario_count=scenario_count,
        )
        self._print_outlier_report(results=results)

        output_dir = self.config.results_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        export_dynamic_benchmark_results(
            results=results,
            output_path=output_dir / self.config.benchmark_output_file,
        )

        return results

    def _resolve_max_simulation_steps(self) -> int | None:
        if self.benchmark_config.max_simulation_steps is not None:
            return self.benchmark_config.max_simulation_steps
        return self.config.max_simulation_steps

    def _run_algorithm(
            self,
            algorithm_name: AlgorithmName,
            grid_map: GridMap,
            scenario: Scenario,
            moving_obstacles: list[MovingObstacle],
    ) -> DynamicReplanningStats:
        dynamic_map = DynamicGridMap(base_map=grid_map)
        algorithm = create_algorithm(name=algorithm_name)

        stats, _ = run_dynamic_simulation(
            algorithm=algorithm,
            dynamic_map=dynamic_map,
            scenario=scenario,
            events=[],
            moving_obstacles=moving_obstacles,
            record_algorithm_steps=False,
            path_block_lookahead=self.benchmark_config.path_block_lookahead,
            wait_when_no_path=self.benchmark_config.wait_when_no_path,
            max_wait_steps=self.benchmark_config.max_wait_steps,
            max_stuck_steps=self.benchmark_config.max_stuck_steps,
            max_simulation_steps=self._resolve_max_simulation_steps(),
            moving_obstacle_collision_policy=(
                self.config.moving_obstacle_collision_policy
            ),
            moving_obstacle_prediction_steps=(
                self.benchmark_config.moving_obstacle_prediction_steps
            ),
        )
        return stats

    def _build_result(
            self,
            algorithm_name: AlgorithmName,
            scenario_index: int,
            scenario_seed: int,
            scenario: Scenario,
            stats: DynamicReplanningStats,
            wall_clock_s: float,
    ) -> DynamicBenchmarkResult:
        return DynamicBenchmarkResult(
            algorithm=stats.algorithm_name,
            scenario_index=scenario_index,
            seed=scenario_seed,
            found=stats.initial_path_found,
            final_goal_reached=stats.final_goal_reached,
            initial_path_found=stats.initial_path_found,
            replanning_count=stats.replanning_count,
            waiting_steps=stats.waiting_steps,
            collision_count=stats.collision_count,
            agent_push_count=stats.agent_push_count,
            obstacle_blocked_count=stats.obstacle_blocked_count,
            travelled_steps=stats.travelled_steps,
            total_path_cost=stats.total_path_cost,
            total_execution_time_ms=stats.total_execution_time_ms,
            wall_clock_s=wall_clock_s,
            obstacle_count=self.benchmark_config.obstacle_count,
            prediction_steps=self.benchmark_config.moving_obstacle_prediction_steps,
            path_block_lookahead=self.benchmark_config.path_block_lookahead,
            start_row=scenario.start.row,
            start_col=scenario.start.col,
            goal_row=scenario.goal.row,
            goal_col=scenario.goal.col,
            optimal_length=scenario.optimal_length,
        )

    def _print_benchmark_start(self, scenario_count: int) -> None:
        print("\nDynamic benchmark started:", flush=True)
        print(f"  scenarios={scenario_count}", flush=True)
        print(f"  obstacle_seed={self.benchmark_config.obstacle_seed}", flush=True)
        print(f"  obstacle_count={self.benchmark_config.obstacle_count}", flush=True)
        print(
            "  prediction_steps="
            f"{self.benchmark_config.moving_obstacle_prediction_steps}",
            flush=True,
        )
        print(
            f"  path_block_lookahead={self.benchmark_config.path_block_lookahead}",
            flush=True,
        )
        max_simulation_steps = self._resolve_max_simulation_steps()
        if max_simulation_steps is not None:
            print(f"  max_simulation_steps={max_simulation_steps}", flush=True)

    def _print_scenario_start(
            self,
            scenario_number: int,
            scenario_count: int,
            scenario_index: int,
            scenario_seed: int,
            scenario: Scenario,
    ) -> None:
        print(
            f"\nRunning scenario {scenario_number}/{scenario_count}: "
            f"scenario_index={scenario_index}, seed={scenario_seed}, "
            f"start=({scenario.start.row},{scenario.start.col}), "
            f"goal=({scenario.goal.row},{scenario.goal.col}), "
            f"optimal_length={_format_optimal_length(scenario)}",
            flush=True,
        )

    def _print_algorithm_result(
            self,
            algorithm_label: str,
            stats: DynamicReplanningStats,
            wall_clock_s: float,
    ) -> None:
        print(
            f"  {algorithm_label}: reached={stats.final_goal_reached}, "
            f"replans={stats.replanning_count}, waits={stats.waiting_steps}, "
            f"collisions={stats.collision_count}, "
            f"time={stats.total_execution_time_ms:.2f} ms, "
            f"wall_clock={wall_clock_s:.2f} s",
            flush=True,
        )

    def _print_immediate_failure_warning(
            self,
            algorithm_label: str,
            scenario_index: int,
            stats: DynamicReplanningStats,
    ) -> None:
        if (
                stats.final_goal_reached
                or stats.replanning_count != 0
                or stats.waiting_steps != 0
                or stats.collision_count != 0
                or stats.total_execution_time_ms >= 1.0
        ):
            return

        print(
            f"Warning: {algorithm_label} failed immediately on "
            f"scenario_index={scenario_index}.\n"
            "Possible initial path failure or start/goal blocked by dynamic obstacles.",
            flush=True,
        )
        print(
            f"  initial_path_found={stats.initial_path_found}, "
            f"final_goal_reached={stats.final_goal_reached}, "
            f"travelled_steps={stats.travelled_steps}, "
            f"total_path_cost={stats.total_path_cost:.3f}",
            flush=True,
        )

    def _print_progress(
            self,
            completed_scenarios: int,
            scenario_count: int,
            results: list[DynamicBenchmarkResult],
            dynamic_scenario_flags: dict[int, bool],
    ) -> None:
        progress_percent = completed_scenarios / scenario_count * 100
        progress_bar = _format_progress_bar(
            current=completed_scenarios,
            total=scenario_count,
        )

        print(f"\n[{completed_scenarios}/{scenario_count}] Dynamic benchmark progress", flush=True)
        print(
            f"Progress: {progress_bar} "
            f"{completed_scenarios}/{scenario_count} {progress_percent:.1f}%",
            flush=True,
        )

        for algorithm_name in ("A*", "D* Lite"):
            algorithm_results = [
                result for result in results
                if result.algorithm == algorithm_name
            ]
            goal_reached = sum(
                1 for result in algorithm_results if result.final_goal_reached
            )
            avg_time = _average(
                result.total_execution_time_ms for result in algorithm_results
            )
            avg_replans = _average(
                result.replanning_count for result in algorithm_results
            )
            print(
                f"{algorithm_name}: reached={goal_reached}/{completed_scenarios}, "
                f"avg_time={avg_time:.2f} ms, avg_replans={avg_replans:.2f}",
                flush=True,
            )

        for algorithm_name in ("A*", "D* Lite"):
            algorithm_results = [
                result for result in results
                if result.algorithm == algorithm_name
            ]
            dynamic_rate = compute_dynamic_interaction_rate(algorithm_results) * 100
            print(
                f"{algorithm_name} dynamic interaction rate: {dynamic_rate:.1f}%",
                flush=True,
            )

        dynamic_count = sum(
            1
            for scenario_index, is_dynamic in dynamic_scenario_flags.items()
            if is_dynamic and scenario_index < completed_scenarios
        )
        combined_dynamic_rate = dynamic_count / completed_scenarios * 100
        print(
            f"Combined dynamic interaction rate: {combined_dynamic_rate:.1f}%",
            flush=True,
        )

    def _print_summary(
            self,
            results: list[DynamicBenchmarkResult],
            dynamic_scenario_flags: dict[int, bool],
            scenario_count: int,
    ) -> None:
        print("\nDynamic benchmark summary:", flush=True)

        grouped_results: dict[str, list[DynamicBenchmarkResult]] = {}
        for result in results:
            grouped_results.setdefault(result.algorithm, []).append(result)

        for algorithm_name, algorithm_results in grouped_results.items():
            goal_reached = sum(
                1 for result in algorithm_results if result.final_goal_reached
            )
            goal_reached_rate = goal_reached / len(algorithm_results)

            execution_time_stats = compute_distribution_stats(
                [result.total_execution_time_ms for result in algorithm_results]
            )
            wall_clock_stats = compute_distribution_stats(
                [result.wall_clock_s for result in algorithm_results]
            )

            print(f"\nAlgorithm: {algorithm_name}", flush=True)
            print(f"  scenarios: {len(algorithm_results)}", flush=True)
            print(f"  goal_reached: {goal_reached}", flush=True)
            print(f"  goal_reached_rate: {goal_reached_rate:.3f}", flush=True)
            print(
                "  avg_replanning_count: "
                f"{_average(result.replanning_count for result in algorithm_results):.3f}",
                flush=True,
            )
            print(
                "  avg_waiting_steps: "
                f"{_average(result.waiting_steps for result in algorithm_results):.3f}",
                flush=True,
            )
            print(
                "  avg_collision_count: "
                f"{_average(result.collision_count for result in algorithm_results):.3f}",
                flush=True,
            )
            print(
                "  avg_travelled_steps: "
                f"{_average(result.travelled_steps for result in algorithm_results):.3f}",
                flush=True,
            )
            print(
                "  avg_total_path_cost: "
                f"{_average(result.total_path_cost for result in algorithm_results):.3f}",
                flush=True,
            )
            print(
                "  avg_total_execution_time_ms: "
                f"{execution_time_stats['avg']:.3f}",
                flush=True,
            )
            print(
                "  median_total_execution_time_ms: "
                f"{execution_time_stats['median']:.3f}",
                flush=True,
            )
            print(
                "  p95_total_execution_time_ms: "
                f"{execution_time_stats['p95']:.3f}",
                flush=True,
            )
            print(
                "  max_total_execution_time_ms: "
                f"{execution_time_stats['max']:.3f}",
                flush=True,
            )
            print(
                f"  avg_wall_clock_s: {wall_clock_stats['avg']:.3f}",
                flush=True,
            )
            print(
                f"  median_wall_clock_s: {wall_clock_stats['median']:.3f}",
                flush=True,
            )
            print(
                f"  p95_wall_clock_s: {wall_clock_stats['p95']:.3f}",
                flush=True,
            )
            print(
                f"  max_wall_clock_s: {wall_clock_stats['max']:.3f}",
                flush=True,
            )

        for algorithm_name in ("A*", "D* Lite"):
            algorithm_results = [
                result for result in results
                if result.algorithm == algorithm_name
            ]
            dynamic_rate = compute_dynamic_interaction_rate(algorithm_results)
            print(
                f"\n{algorithm_name} dynamic interaction rate: "
                f"{dynamic_rate * 100:.0f}%",
                flush=True,
            )

        dynamic_count = sum(1 for is_dynamic in dynamic_scenario_flags.values() if is_dynamic)
        combined_dynamic_rate = dynamic_count / scenario_count if scenario_count else 0.0

        print(
            f"Combined dynamic interaction rate: {combined_dynamic_rate * 100:.0f}%",
            flush=True,
        )

        if combined_dynamic_rate < self.benchmark_config.min_dynamic_interaction_rate:
            print(
                "Warning: only "
                f"{combined_dynamic_rate * 100:.0f}% scenarios had dynamic interactions. "
                "Increase obstacle_count, obstacle size, or reduce generation margin.",
                flush=True,
            )

    def _print_outlier_report(self, results: list[DynamicBenchmarkResult]) -> None:
        print("\nSlowest scenarios by wall-clock:", flush=True)
        for result in _top_slowest_results(
                results=results,
                sort_key=lambda item: item.wall_clock_s,
        ):
            _print_outlier_record(result=result)

        print("\nSlowest scenarios by algorithm time:", flush=True)
        for result in _top_slowest_results(
                results=results,
                sort_key=lambda item: item.total_execution_time_ms,
        ):
            _print_outlier_record(result=result)


def _format_optimal_length(scenario: Scenario) -> str:
    if scenario.optimal_length is None:
        return "n/a"
    return f"{scenario.optimal_length:.2f}"


def _average(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _format_progress_bar(current: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"

    filled = int(width * current / total)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _top_slowest_results(
        results: list[DynamicBenchmarkResult],
        sort_key,
        limit: int = _OUTLIER_REPORT_LIMIT,
) -> list[DynamicBenchmarkResult]:
    return sorted(results, key=sort_key, reverse=True)[:limit]


def _print_outlier_record(result: DynamicBenchmarkResult) -> None:
    print(
        f"  {result.algorithm} scenario_index={result.scenario_index} "
        f"seed={result.seed} reached={result.final_goal_reached} "
        f"replans={result.replanning_count} waits={result.waiting_steps} "
        f"collisions={result.collision_count} "
        f"time={result.total_execution_time_ms:.2f} ms "
        f"wall_clock={result.wall_clock_s:.2f} s",
        flush=True,
    )
