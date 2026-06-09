import csv
from pathlib import Path

from pathfinding.src.core.benchmark_models import DynamicBenchmarkConfig, DynamicBenchmarkResult
from pathfinding.src.core.dynamic_models import DynamicGridMap, MovingObstacle, MovingObstacleCollisionPolicy
from pathfinding.src.core.models import Position, Scenario
from pathfinding.src.experiments.algorithm_registry import create_algorithm
from pathfinding.src.experiments.dynamic_benchmark_experiment import (
    DynamicBenchmarkExperiment,
    compute_distribution_stats,
    generate_scenario_obstacles,
)
from pathfinding.src.experiments.dynamic_simulation import run_dynamic_simulation
from pathfinding.src.experiments.experiment_config import AlgorithmName, ExperimentConfig
from pathfinding.tests.helpers import build_grid_map


def _build_test_scenarios(grid_map) -> list[Scenario]:
    return [
        Scenario(
            map_name=grid_map.name,
            width=grid_map.width,
            height=grid_map.height,
            start=Position(row=1, col=1),
            goal=Position(row=1, col=8),
            optimal_length=120.0,
        ),
        Scenario(
            map_name=grid_map.name,
            width=grid_map.width,
            height=grid_map.height,
            start=Position(row=2, col=2),
            goal=Position(row=2, col=9),
            optimal_length=110.0,
        ),
    ]


def test_dynamic_benchmark_result_contains_wall_clock_s() -> None:
    result = DynamicBenchmarkResult(
        algorithm="A*",
        scenario_index=0,
        seed=42,
        found=True,
        final_goal_reached=True,
        initial_path_found=True,
        replanning_count=1,
        waiting_steps=0,
        collision_count=0,
        agent_push_count=0,
        obstacle_blocked_count=0,
        travelled_steps=10,
        total_path_cost=12.0,
        total_execution_time_ms=5.0,
        wall_clock_s=0.25,
        obstacle_count=4,
        prediction_steps=0,
        path_block_lookahead=20,
        start_row=1,
        start_col=1,
        goal_row=1,
        goal_col=8,
    )

    assert result.wall_clock_s == 0.25


def test_benchmark_generates_identical_obstacles_for_both_algorithms() -> None:
    grid_map = build_grid_map(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        name="dynamic_benchmark_obstacle_map",
    )
    scenario = _build_test_scenarios(grid_map)[0]
    benchmark_config = DynamicBenchmarkConfig(
        obstacle_count=4,
        obstacle_seed=42,
        min_obstacle_size=2,
        max_obstacle_size=3,
        path_margin=2,
    )

    obstacles_for_astar = generate_scenario_obstacles(
        grid_map=grid_map,
        scenario=scenario,
        scenario_index=0,
        benchmark_config=benchmark_config,
    )
    obstacles_for_dstar = generate_scenario_obstacles(
        grid_map=grid_map,
        scenario=scenario,
        scenario_index=0,
        benchmark_config=benchmark_config,
    )

    assert obstacles_for_astar == obstacles_for_dstar


def test_dynamic_benchmark_writes_csv(tmp_path: Path) -> None:
    grid_map = build_grid_map(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        name="dynamic_benchmark_csv_map",
    )
    scenarios = _build_test_scenarios(grid_map)

    config = ExperimentConfig(
        map_path=Path("."),
        scen_path=Path("."),
        min_optimal_length=0,
        benchmark_scenarios=2,
        results_dir=tmp_path,
        benchmark_output_file="dynamic_benchmark_test.csv",
    )
    benchmark_config = DynamicBenchmarkConfig(
        scenario_count=2,
        min_optimal_length=0,
        obstacle_count=3,
        obstacle_seed=7,
        min_obstacle_size=2,
        max_obstacle_size=3,
        path_margin=2,
        max_stuck_steps=20,
    )

    DynamicBenchmarkExperiment(
        config=config,
        benchmark_config=benchmark_config,
        grid_map=grid_map,
        scenarios=scenarios,
    ).run()

    output_path = tmp_path / "dynamic_benchmark_test.csv"
    assert output_path.exists()

    with output_path.open(encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    assert len(rows) == 4
    assert "algorithm" in fieldnames
    assert "scenario_index" in fieldnames
    assert "final_goal_reached" in fieldnames
    assert "replanning_count" in fieldnames
    assert "total_execution_time_ms" in fieldnames
    assert "wall_clock_s" in fieldnames

    for row in rows:
        assert float(row["wall_clock_s"]) >= 0.0


def test_run_dynamic_simulation_does_not_mutate_shared_obstacles() -> None:
    grid_map = build_grid_map(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        name="shared_obstacle_map",
    )
    scenario = Scenario(
        map_name=grid_map.name,
        width=grid_map.width,
        height=grid_map.height,
        start=Position(row=1, col=1),
        goal=Position(row=1, col=10),
        optimal_length=9.0,
    )
    obstacles = [
        MovingObstacle(row=2, col=4, width=2, height=1, delta_row=0, delta_col=1),
        MovingObstacle(row=3, col=7, width=1, height=2, delta_row=-1, delta_col=0),
    ]
    initial_state = [
        (obstacle.row, obstacle.col, obstacle.delta_row, obstacle.delta_col)
        for obstacle in obstacles
    ]

    dynamic_map = DynamicGridMap(base_map=grid_map)
    algorithm = create_algorithm(AlgorithmName.ASTAR)
    run_dynamic_simulation(
        algorithm=algorithm,
        dynamic_map=dynamic_map,
        scenario=scenario,
        events=[],
        moving_obstacles=obstacles,
        record_algorithm_steps=False,
        max_simulation_steps=200,
        moving_obstacle_collision_policy=MovingObstacleCollisionPolicy.PUSH_AGENT,
    )

    assert [
        (obstacle.row, obstacle.col, obstacle.delta_row, obstacle.delta_col)
        for obstacle in obstacles
    ] == initial_state


def test_summary_distribution_stats_with_outliers() -> None:
    astar_stats = compute_distribution_stats([10.0, 20.0, 10000.0])
    dstar_stats = compute_distribution_stats([30.0, 40.0, 50.0])

    assert astar_stats["median"] == 20.0
    assert dstar_stats["median"] == 40.0
    assert astar_stats["max"] == 10000.0
    assert dstar_stats["max"] == 50.0
