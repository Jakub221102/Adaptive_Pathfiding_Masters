from pathlib import Path

from pathfinding.src.algorithms.astar import AStar
from pathfinding.src.algorithms.dstar_lite import DStarLite
from pathfinding.src.core.dynamic_models import (
    DynamicGridMap,
    DynamicObstacleEvent,
    DynamicObstacleEventType,
    MovingObstacle,
    MovingObstacleCollisionPolicy,
    validate_path_walkable,
)
from pathfinding.src.experiments.dynamic_simulation import (
    _handle_replanning,
    run_dynamic_simulation,
    with_predicted_obstacle_blocks,
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


def test_dynamic_simulation_handles_moving_obstacle_agent_collision() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        name="moving_obstacle_collision_map",
    )

    scenario = Scenario(
        map_name=grid_map.name,
        width=grid_map.width,
        height=grid_map.height,
        start=Position(row=0, col=1),
        goal=Position(row=0, col=4),
    )

    dynamic_map = DynamicGridMap(base_map=grid_map)
    moving_obstacles = [
        MovingObstacle(
            row=0,
            col=2,
            width=1,
            height=1,
            delta_row=0,
            delta_col=-1,
        )
    ]

    stats, _ = run_dynamic_simulation(
        algorithm=AStar(),
        dynamic_map=dynamic_map,
        scenario=scenario,
        events=[],
        moving_obstacles=moving_obstacles,
        moving_obstacle_collision_policy=MovingObstacleCollisionPolicy.PUSH_AGENT,
    )

    assert stats.initial_path_found is True
    assert stats.collision_count >= 1


def test_dynamic_simulation_records_agent_push_on_collision() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        name="moving_obstacle_push_map",
    )

    scenario = Scenario(
        map_name=grid_map.name,
        width=grid_map.width,
        height=grid_map.height,
        start=Position(row=0, col=1),
        goal=Position(row=0, col=4),
    )

    dynamic_map = DynamicGridMap(base_map=grid_map)
    moving_obstacles = [
        MovingObstacle(
            row=0,
            col=2,
            width=1,
            height=1,
            delta_row=0,
            delta_col=-1,
        )
    ]

    stats, _ = run_dynamic_simulation(
        algorithm=AStar(),
        dynamic_map=dynamic_map,
        scenario=scenario,
        events=[],
        moving_obstacles=moving_obstacles,
        moving_obstacle_collision_policy=MovingObstacleCollisionPolicy.PUSH_AGENT,
    )

    assert stats.collision_count >= 1
    assert stats.agent_push_count >= 1


def test_dynamic_simulation_records_obstacle_blocked_when_push_impossible() -> None:
    grid_map = build_grid_map(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        name="moving_obstacle_block_map",
    )

    scenario = Scenario(
        map_name=grid_map.name,
        width=grid_map.width,
        height=grid_map.height,
        start=Position(row=1, col=1),
        goal=Position(row=1, col=1),
    )

    dynamic_map = DynamicGridMap(base_map=grid_map)
    moving_obstacles = [
        MovingObstacle(
            row=2,
            col=1,
            width=1,
            height=1,
            delta_row=-1,
            delta_col=0,
        )
    ]

    stats, _ = run_dynamic_simulation(
        algorithm=AStar(),
        dynamic_map=dynamic_map,
        scenario=scenario,
        events=[],
        moving_obstacles=moving_obstacles,
        moving_obstacle_collision_policy=MovingObstacleCollisionPolicy.PUSH_AGENT,
    )

    assert stats.collision_count >= 1
    assert stats.agent_push_count == 0
    assert stats.obstacle_blocked_count >= 1


def test_with_predicted_obstacle_blocks_restores_dynamic_blocked() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        name="predicted_blocks_restore_map",
    )
    dynamic_map = DynamicGridMap(base_map=grid_map)
    dynamic_map.dynamic_blocked.add((1, 1))

    obstacle = MovingObstacle(
        row=0,
        col=1,
        width=1,
        height=1,
        delta_row=1,
        delta_col=0,
    )
    algorithm = DStarLite()
    current_position = Position(row=2, col=0)
    goal = Position(row=2, col=2)
    initial_blocked = set(dynamic_map.dynamic_blocked)

    with with_predicted_obstacle_blocks(
            dynamic_map=dynamic_map,
            moving_obstacles=[obstacle],
            prediction_steps=2,
            algorithm=algorithm,
            current_position=current_position,
            goal=goal,
    ):
        assert len(dynamic_map.dynamic_blocked) > len(initial_blocked)

    assert dynamic_map.dynamic_blocked == initial_blocked


def test_handle_replanning_returns_walkable_path_with_prediction() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        name="predicted_replan_walkable_map",
    )
    dynamic_map = DynamicGridMap(base_map=grid_map)
    start = Position(row=1, col=0)
    goal = Position(row=1, col=6)

    obstacle = MovingObstacle(
        row=1,
        col=4,
        width=1,
        height=1,
        delta_row=0,
        delta_col=-1,
    )
    dynamic_map.block_rectangle(
        position=Position(row=obstacle.row, col=obstacle.col),
        width=obstacle.width,
        height=obstacle.height,
    )

    algorithm = DStarLite()
    initial_result, _ = algorithm.find_path_with_steps(
        grid_map=dynamic_map,
        start=start,
        goal=goal,
        step_record_interval=100,
    )

    assert initial_result.found is True

    _, replanned_path, _, _, _ = _handle_replanning(
        algorithm=algorithm,
        dynamic_map=dynamic_map,
        current_position=start,
        goal=goal,
        current_path=initial_result.path,
        current_path_index=0,
        travelled_path=[start],
        step_record_interval=100,
        wait_when_no_path=True,
        waiting_steps=0,
        max_wait_steps=10,
        replanning_count=0,
        keyframes=[],
        record_keyframes=False,
        frame_callback=None,
        moving_obstacles=[obstacle],
        moving_obstacle_prediction_steps=5,
    )

    if replanned_path is not None:
        assert validate_path_walkable(dynamic_map, replanned_path) is True


def test_dstar_lite_dynamic_replanning_with_prediction_reaches_goal() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        name="prediction_goal_map",
    )

    scenario = Scenario(
        map_name=grid_map.name,
        width=grid_map.width,
        height=grid_map.height,
        start=Position(row=1, col=0),
        goal=Position(row=1, col=6),
    )

    dynamic_map = DynamicGridMap(base_map=grid_map)
    moving_obstacles = [
        MovingObstacle(
            row=0,
            col=3,
            width=1,
            height=1,
            delta_row=0,
            delta_col=1,
        )
    ]

    stats, _ = run_dynamic_simulation(
        algorithm=DStarLite(),
        dynamic_map=dynamic_map,
        scenario=scenario,
        events=[],
        moving_obstacles=moving_obstacles,
        moving_obstacle_prediction_steps=5,
        path_block_lookahead=4,
    )

    assert stats.initial_path_found is True
    assert stats.final_goal_reached is True


def test_dynamic_simulation_does_not_deadlock_in_narrow_corridor() -> None:
    grid_map = build_grid_map(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        name="narrow_corridor_deadlock_map",
    )

    scenario = Scenario(
        map_name=grid_map.name,
        width=grid_map.width,
        height=grid_map.height,
        start=Position(row=1, col=1),
        goal=Position(row=1, col=3),
    )

    dynamic_map = DynamicGridMap(base_map=grid_map)
    moving_obstacles = [
        MovingObstacle(
            row=1,
            col=4,
            width=1,
            height=1,
            delta_row=0,
            delta_col=-1,
        )
    ]

    stats, _ = run_dynamic_simulation(
        algorithm=AStar(),
        dynamic_map=dynamic_map,
        scenario=scenario,
        events=[],
        moving_obstacles=moving_obstacles,
        moving_obstacle_collision_policy=MovingObstacleCollisionPolicy.PUSH_AGENT,
        moving_obstacle_prediction_steps=0,
        path_block_lookahead=4,
        max_stuck_steps=25,
    )

    assert stats.initial_path_found is True
    assert stats.travelled_steps > 0
    assert stats.obstacle_blocked_count > 0
    assert stats.waiting_steps <= 30
