from pathlib import Path

from pathfinding.src.algorithms.astar import AStar
from pathfinding.src.core.dynamic_models import (
    DynamicGridMap,
    DynamicObstacleEvent,
    DynamicObstacleEventType,
)
from pathfinding.src.core.models import Position, Scenario
from pathfinding.src.experiments.dynamic_replanning_experiment import (
    DynamicReplanningExperiment,
)
from pathfinding.src.experiments.experiment_config import AlgorithmName, ExperimentConfig
from pathfinding.tests.helpers import build_grid_map


def test_dynamic_replanning_reaches_goal_after_obstacle_on_path() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        name="dynamic_replanning_map",
    )

    scenario = Scenario(
        map_name=grid_map.name,
        width=grid_map.width,
        height=grid_map.height,
        start=Position(row=0, col=0),
        goal=Position(row=0, col=4),
    )

    dynamic_map = DynamicGridMap(base_map=grid_map)
    initial_result = AStar().find_path(
        grid_map=dynamic_map,
        start=scenario.start,
        goal=scenario.goal,
    )

    assert initial_result.found is True

    events = [
        DynamicObstacleEvent(
            step_index=1,
            position=Position(row=0, col=2),
            event_type=DynamicObstacleEventType.BLOCK,
        )
    ]

    config = ExperimentConfig(
        map_path=Path("."),
        scen_path=Path("."),
        scenario_index=0,
        min_optimal_length=0,
        algorithm=AlgorithmName.ASTAR,
    )

    stats = DynamicReplanningExperiment(
        config=config,
        events=events,
        grid_map=grid_map,
        scenario=scenario,
    ).run()

    assert stats.initial_path_found is True
    assert stats.final_goal_reached is True
    assert stats.replanning_count >= 1
    assert stats.dynamic_events_applied == 1


def test_dstar_lite_dynamic_replanning_reaches_goal_after_obstacle_on_path() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        name="dynamic_replanning_map",
    )

    scenario = Scenario(
        map_name=grid_map.name,
        width=grid_map.width,
        height=grid_map.height,
        start=Position(row=0, col=0),
        goal=Position(row=0, col=4),
    )

    events = [
        DynamicObstacleEvent(
            step_index=1,
            position=Position(row=0, col=2),
            event_type=DynamicObstacleEventType.BLOCK,
        )
    ]

    config = ExperimentConfig(
        map_path=Path("."),
        scen_path=Path("."),
        scenario_index=0,
        min_optimal_length=0,
        algorithm=AlgorithmName.DSTAR_LITE,
    )

    stats = DynamicReplanningExperiment(
        config=config,
        events=events,
        grid_map=grid_map,
        scenario=scenario,
    ).run()

    assert stats.initial_path_found is True
    assert stats.final_goal_reached is True
    assert stats.replanning_count >= 1
    assert stats.dynamic_events_applied == 1


def test_rectangle_obstacle_triggers_replanning() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        name="rectangle_obstacle_map",
    )

    scenario = Scenario(
        map_name=grid_map.name,
        width=grid_map.width,
        height=grid_map.height,
        start=Position(row=0, col=0),
        goal=Position(row=0, col=5),
    )

    events = [
        DynamicObstacleEvent(
            step_index=1,
            position=Position(row=0, col=2),
            event_type=DynamicObstacleEventType.BLOCK,
            width=2,
            height=1,
        )
    ]

    config = ExperimentConfig(
        map_path=Path("."),
        scen_path=Path("."),
        scenario_index=0,
        min_optimal_length=0,
        algorithm=AlgorithmName.ASTAR,
    )

    stats = DynamicReplanningExperiment(
        config=config,
        events=events,
        grid_map=grid_map,
        scenario=scenario,
    ).run()

    assert stats.initial_path_found is True
    assert stats.final_goal_reached is True
    assert stats.replanning_count >= 1
    assert stats.dynamic_events_applied == 1


def test_agent_waits_for_temporarily_blocked_corridor() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        name="waiting_corridor_map",
    )

    scenario = Scenario(
        map_name=grid_map.name,
        width=grid_map.width,
        height=grid_map.height,
        start=Position(row=0, col=0),
        goal=Position(row=0, col=9),
    )

    events = [
        DynamicObstacleEvent(
            step_index=1,
            position=Position(row=0, col=5),
            event_type=DynamicObstacleEventType.BLOCK,
        ),
        DynamicObstacleEvent(
            step_index=3,
            position=Position(row=0, col=5),
            event_type=DynamicObstacleEventType.UNBLOCK,
        ),
    ]

    config = ExperimentConfig(
        map_path=Path("."),
        scen_path=Path("."),
        scenario_index=0,
        min_optimal_length=0,
        algorithm=AlgorithmName.ASTAR,
        path_block_lookahead=8,
        wait_when_no_path=True,
        max_wait_steps=10,
    )

    stats = DynamicReplanningExperiment(
        config=config,
        events=events,
        grid_map=grid_map,
        scenario=scenario,
    ).run()

    assert stats.initial_path_found is True
    assert stats.final_goal_reached is True
    assert stats.waiting_steps >= 1
    assert stats.dynamic_events_applied == 2
