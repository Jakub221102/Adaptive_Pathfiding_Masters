from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field

from pathfinding.src.algorithms.astar import AStar
from pathfinding.src.algorithms.dstar_lite import DStarLite
from pathfinding.src.core.dynamic_models import (
    DynamicGridMap,
    DynamicObstacleEvent,
    DynamicReplanningStats,
    MovingObstacle,
    MovingObstacleCollisionPolicy,
    apply_dynamic_event,
    is_path_blocked_with_lookahead,
    move_obstacle_with_agent_collision,
)
from pathfinding.src.core.models import Position, Scenario
from pathfinding.src.core.trace import RawAlgorithmStep


class DynamicSimulationKeyframe(BaseModel):
    frame_type: Literal["search", "move", "obstacle", "path_update"]
    agent_position: Position
    planned_path: list[Position]
    travelled_path: list[Position]
    dynamic_blocked: set[tuple[int, int]] = Field(default_factory=set)
    search_steps: list[RawAlgorithmStep] | None = None
    new_obstacle_positions: list[Position] = Field(default_factory=list)
    status_text: str = ""


class DynamicPlaybackFrame(BaseModel):
    frame_kind: Literal["search", "move", "obstacle", "path_update"] = "move"
    agent_position: Position
    planned_path: list[Position]
    travelled_path: list[Position]
    dynamic_blocked: set[tuple[int, int]] = Field(default_factory=set)
    search_step: RawAlgorithmStep | None = None
    new_obstacle_positions: list[Position] = Field(default_factory=list)
    status_text: str = ""


def run_dynamic_simulation(
        algorithm,
        dynamic_map: DynamicGridMap,
        scenario: Scenario,
        events: list[DynamicObstacleEvent],
        *,
        moving_obstacles: list[MovingObstacle] | None = None,
        record_keyframes: bool = False,
        frame_callback: Callable[[DynamicSimulationKeyframe], None] | None = None,
        step_record_interval: int = 10,
        path_block_lookahead: int = 8,
        wait_when_no_path: bool = True,
        max_wait_steps: int = 30,
        moving_obstacle_collision_policy: MovingObstacleCollisionPolicy = (
            MovingObstacleCollisionPolicy.PUSH_AGENT
        ),
) -> tuple[DynamicReplanningStats, list[DynamicSimulationKeyframe]]:
    current_position = scenario.start
    goal = scenario.goal
    keyframes: list[DynamicSimulationKeyframe] = []
    obstacles = list(moving_obstacles or [])

    for obstacle in obstacles:
        dynamic_map.block_rectangle(
            position=Position(row=obstacle.row, col=obstacle.col),
            width=obstacle.width,
            height=obstacle.height,
        )

    initial_result, initial_steps = _plan_path(
        algorithm=algorithm,
        dynamic_map=dynamic_map,
        start=current_position,
        goal=goal,
        step_record_interval=step_record_interval,
    )

    total_execution_time_ms = initial_result.execution_time_ms

    if not initial_result.found:
        return _build_failed_stats(
            algorithm_name=algorithm.name,
            total_execution_time_ms=total_execution_time_ms,
        ), keyframes

    _emit_keyframe(
        keyframes=keyframes,
        record_keyframes=record_keyframes,
        frame_callback=frame_callback,
        keyframe=_create_search_keyframe(
            agent_position=current_position,
            planned_path=initial_result.path,
            travelled_path=[current_position],
            dynamic_map=dynamic_map,
            search_steps=initial_steps,
            status_text="Initial planning",
        ),
    )

    current_path = initial_result.path
    current_path_index = 0
    replanning_count = 0
    dynamic_events_applied = 0
    travelled_path = [current_position]
    final_goal_reached = False
    waiting_steps = 0
    total_waiting_steps = 0
    collision_count = 0
    agent_push_count = 0
    obstacle_blocked_count = 0

    event_index = 0
    max_steps = dynamic_map.width * dynamic_map.height + max_wait_steps

    for step in range(max_steps):
        (
            current_position,
            step_collision_count,
            step_agent_push_count,
            step_obstacle_blocked_count,
            agent_position_changed,
        ) = _move_moving_obstacles(
            algorithm=algorithm,
            dynamic_map=dynamic_map,
            moving_obstacles=obstacles,
            record_keyframes=record_keyframes,
            frame_callback=frame_callback,
            keyframes=keyframes,
            agent_position=current_position,
            planned_path=current_path[current_path_index:],
            travelled_path=travelled_path,
            step=step,
            collision_policy=moving_obstacle_collision_policy,
        )
        collision_count += step_collision_count
        agent_push_count += step_agent_push_count
        obstacle_blocked_count += step_obstacle_blocked_count

        if agent_position_changed:
            current_path_index = 0

        applied_events, event_index = _apply_events(
            algorithm=algorithm,
            dynamic_map=dynamic_map,
            events=events,
            step=step,
            event_index=event_index,
            record_keyframes=record_keyframes,
            frame_callback=frame_callback,
            keyframes=keyframes,
            agent_position=current_position,
            planned_path=current_path[current_path_index:],
            travelled_path=travelled_path,
        )
        dynamic_events_applied += applied_events

        if current_position == goal:
            final_goal_reached = True
            break

        force_replan = agent_position_changed

        if force_replan or is_path_blocked_with_lookahead(
                dynamic_map,
                current_path,
                current_path_index,
                path_block_lookahead,
        ):
            (
                should_continue,
                replanned_path,
                current_path_index,
                waiting_steps,
                replan_time_ms,
            ) = _handle_replanning(
                algorithm=algorithm,
                dynamic_map=dynamic_map,
                current_position=current_position,
                goal=goal,
                current_path=current_path,
                current_path_index=current_path_index,
                travelled_path=travelled_path,
                step_record_interval=step_record_interval,
                wait_when_no_path=wait_when_no_path,
                waiting_steps=waiting_steps,
                max_wait_steps=max_wait_steps,
                replanning_count=replanning_count,
                keyframes=keyframes,
                record_keyframes=record_keyframes,
                frame_callback=frame_callback,
            )
            replanning_count += 1
            total_execution_time_ms += replan_time_ms

            if not should_continue:
                break

            if replanned_path is None:
                total_waiting_steps += 1
                continue

            current_path = replanned_path
            waiting_steps = 0

        if current_path_index + 1 >= len(current_path):
            break

        next_position = current_path[current_path_index + 1]

        if not dynamic_map.is_walkable(next_position):
            (
                should_continue,
                replanned_path,
                current_path_index,
                waiting_steps,
                replan_time_ms,
            ) = _handle_replanning(
                algorithm=algorithm,
                dynamic_map=dynamic_map,
                current_position=current_position,
                goal=goal,
                current_path=current_path,
                current_path_index=current_path_index,
                travelled_path=travelled_path,
                step_record_interval=step_record_interval,
                wait_when_no_path=wait_when_no_path,
                waiting_steps=waiting_steps,
                max_wait_steps=max_wait_steps,
                replanning_count=replanning_count,
                keyframes=keyframes,
                record_keyframes=record_keyframes,
                frame_callback=frame_callback,
            )
            replanning_count += 1
            total_execution_time_ms += replan_time_ms

            if not should_continue:
                break

            if replanned_path is None:
                total_waiting_steps += 1
                continue

            current_path = replanned_path
            next_position = current_path[current_path_index + 1]
            waiting_steps = 0

        current_position = next_position
        current_path_index += 1
        travelled_path.append(current_position)

        if isinstance(algorithm, DStarLite):
            algorithm.move_agent(current_position)

        _emit_keyframe(
            keyframes=keyframes,
            record_keyframes=record_keyframes,
            frame_callback=frame_callback,
            keyframe=DynamicSimulationKeyframe(
                frame_type="move",
                agent_position=current_position,
                planned_path=current_path[current_path_index:],
                travelled_path=list(travelled_path),
                dynamic_blocked=set(dynamic_map.dynamic_blocked),
                status_text=(
                    f"Step {len(travelled_path) - 1}: "
                    f"agent moves to ({current_position.row}, {current_position.col})"
                ),
            ),
        )

        if current_position == goal:
            final_goal_reached = True
            break

    stats = DynamicReplanningStats(
        algorithm_name=algorithm.name,
        initial_path_found=True,
        final_goal_reached=final_goal_reached,
        replanning_count=replanning_count,
        total_execution_time_ms=total_execution_time_ms,
        total_path_cost=AStar._calculate_path_cost(travelled_path),
        travelled_steps=max(len(travelled_path) - 1, 0),
        dynamic_events_applied=dynamic_events_applied,
        waiting_steps=total_waiting_steps,
        agent_push_count=agent_push_count,
        obstacle_blocked_count=obstacle_blocked_count,
        collision_count=collision_count,
    )
    return stats, keyframes


def flatten_keyframes_to_playback_frames(
        keyframes: list[DynamicSimulationKeyframe],
        *,
        search_steps_per_frame: int = 5,
        obstacle_hold_frames: int = 25,
        move_hold_frames: int = 3,
) -> list[DynamicPlaybackFrame]:
    playback_frames: list[DynamicPlaybackFrame] = []

    for keyframe in keyframes:
        playback_kind: Literal["search", "move", "obstacle", "path_update"] = (
            keyframe.frame_type
        )
        base_frame = DynamicPlaybackFrame(
            frame_kind=playback_kind,
            agent_position=keyframe.agent_position,
            planned_path=keyframe.planned_path,
            travelled_path=keyframe.travelled_path,
            dynamic_blocked=set(keyframe.dynamic_blocked),
            new_obstacle_positions=list(keyframe.new_obstacle_positions),
            status_text=keyframe.status_text,
        )

        if keyframe.frame_type == "search":
            search_steps = keyframe.search_steps or []

            if not search_steps:
                playback_frames.append(base_frame)
                continue

            for step_index in range(0, len(search_steps), search_steps_per_frame):
                search_step = search_steps[step_index]
                playback_frames.append(
                    base_frame.model_copy(
                        update={"search_step": search_step},
                    )
                )

            playback_frames.append(
                base_frame.model_copy(
                    update={"search_step": search_steps[-1]},
                )
            )
            continue

        if keyframe.frame_type == "obstacle":
            for _ in range(obstacle_hold_frames):
                playback_frames.append(base_frame)
            continue

        if keyframe.frame_type == "path_update":
            for _ in range(obstacle_hold_frames):
                playback_frames.append(base_frame)
            continue

        for _ in range(move_hold_frames):
            playback_frames.append(base_frame)

    return playback_frames


def _plan_path(
        algorithm,
        dynamic_map: DynamicGridMap,
        start: Position,
        goal: Position,
        step_record_interval: int,
):
    if isinstance(algorithm, DStarLite):
        return algorithm.find_path_with_steps(
            grid_map=dynamic_map,
            start=start,
            goal=goal,
            step_record_interval=step_record_interval,
        )

    return algorithm.find_path_with_steps(
        grid_map=dynamic_map,
        start=start,
        goal=goal,
        step_record_interval=step_record_interval,
    )


def _handle_replanning(
        algorithm,
        dynamic_map: DynamicGridMap,
        current_position: Position,
        goal: Position,
        current_path: list[Position],
        current_path_index: int,
        travelled_path: list[Position],
        step_record_interval: int,
        wait_when_no_path: bool,
        waiting_steps: int,
        max_wait_steps: int,
        replanning_count: int,
        keyframes: list[DynamicSimulationKeyframe],
        record_keyframes: bool,
        frame_callback: Callable[[DynamicSimulationKeyframe], None] | None,
) -> tuple[bool, list[Position] | None, int, int, float]:
    _emit_replanning_started_keyframe(
        keyframes=keyframes,
        record_keyframes=record_keyframes,
        frame_callback=frame_callback,
        agent_position=current_position,
        planned_path=current_path[current_path_index:],
        travelled_path=travelled_path,
        dynamic_map=dynamic_map,
        replanning_count=replanning_count + 1,
    )

    result, search_steps = _replan_path(
        algorithm=algorithm,
        dynamic_map=dynamic_map,
        start=current_position,
        goal=goal,
        step_record_interval=step_record_interval,
    )

    if result.found:
        _emit_keyframe(
            keyframes=keyframes,
            record_keyframes=record_keyframes,
            frame_callback=frame_callback,
            keyframe=_create_replan_keyframe(
                algorithm=algorithm,
                agent_position=current_position,
                planned_path=result.path,
                travelled_path=travelled_path,
                dynamic_map=dynamic_map,
                search_steps=search_steps,
                replanning_count=replanning_count + 1,
            ),
        )
        return True, result.path, 0, 0, result.execution_time_ms

    if wait_when_no_path and waiting_steps < max_wait_steps:
        _emit_waiting_keyframe(
            keyframes=keyframes,
            record_keyframes=record_keyframes,
            frame_callback=frame_callback,
            agent_position=current_position,
            planned_path=current_path[current_path_index:],
            travelled_path=travelled_path,
            dynamic_map=dynamic_map,
            waiting_steps=waiting_steps + 1,
        )
        return (
            True,
            None,
            current_path_index,
            waiting_steps + 1,
            result.execution_time_ms,
        )

    return False, None, 0, waiting_steps, result.execution_time_ms


def _replan_path(
        algorithm,
        dynamic_map: DynamicGridMap,
        start: Position,
        goal: Position,
        step_record_interval: int,
):
    if isinstance(algorithm, DStarLite):
        return algorithm.replan_with_steps(
            step_record_interval=step_record_interval,
        )

    return algorithm.find_path_with_steps(
        grid_map=dynamic_map,
        start=start,
        goal=goal,
        step_record_interval=step_record_interval,
    )


def _emit_keyframe(
        keyframes: list[DynamicSimulationKeyframe],
        record_keyframes: bool,
        frame_callback: Callable[[DynamicSimulationKeyframe], None] | None,
        keyframe: DynamicSimulationKeyframe,
) -> None:
    if record_keyframes:
        keyframes.append(keyframe)

    if frame_callback is not None:
        frame_callback(keyframe)


def _move_moving_obstacles(
        algorithm,
        dynamic_map: DynamicGridMap,
        moving_obstacles: list[MovingObstacle],
        record_keyframes: bool,
        frame_callback: Callable[[DynamicSimulationKeyframe], None] | None,
        keyframes: list[DynamicSimulationKeyframe],
        agent_position: Position,
        planned_path: list[Position],
        travelled_path: list[Position],
        step: int,
        collision_policy: MovingObstacleCollisionPolicy,
) -> tuple[Position, int, int, int, bool]:
    current_agent_position = agent_position
    collision_count = 0
    agent_push_count = 0
    obstacle_blocked_count = 0
    agent_position_changed = False

    for obstacle in moving_obstacles:
        move_result = move_obstacle_with_agent_collision(
            obstacle=obstacle,
            dynamic_map=dynamic_map,
            agent_position=current_agent_position,
            policy=collision_policy,
        )

        if move_result.collision_detected:
            collision_count += 1

        if move_result.agent_pushed:
            agent_push_count += 1
            agent_position_changed = True
            if move_result.new_agent_position is not None:
                current_agent_position = move_result.new_agent_position
                travelled_path.append(current_agent_position)

                if isinstance(algorithm, DStarLite):
                    algorithm.move_agent(current_agent_position)

        if move_result.obstacle_blocked:
            obstacle_blocked_count += 1

        if isinstance(algorithm, DStarLite):
            algorithm.update_cells(move_result.affected_positions)

        size_label = (
            f"{obstacle.width}x{obstacle.height}"
            if obstacle.width > 1 or obstacle.height > 1
            else "1x1"
        )

        if move_result.agent_pushed:
            _emit_keyframe(
                keyframes=keyframes,
                record_keyframes=record_keyframes,
                frame_callback=frame_callback,
                keyframe=DynamicSimulationKeyframe(
                    frame_type="path_update",
                    agent_position=current_agent_position,
                    planned_path=planned_path,
                    travelled_path=list(travelled_path),
                    dynamic_blocked=set(dynamic_map.dynamic_blocked),
                    new_obstacle_positions=move_result.affected_positions,
                    status_text="Moving obstacle pushed agent",
                ),
            )
        elif move_result.obstacle_blocked:
            _emit_keyframe(
                keyframes=keyframes,
                record_keyframes=record_keyframes,
                frame_callback=frame_callback,
                keyframe=DynamicSimulationKeyframe(
                    frame_type="path_update",
                    agent_position=current_agent_position,
                    planned_path=planned_path,
                    travelled_path=list(travelled_path),
                    dynamic_blocked=set(dynamic_map.dynamic_blocked),
                    new_obstacle_positions=move_result.affected_positions,
                    status_text=(
                        "Moving obstacle blocked to avoid crushing agent"
                    ),
                ),
            )
        else:
            _emit_keyframe(
                keyframes=keyframes,
                record_keyframes=record_keyframes,
                frame_callback=frame_callback,
                keyframe=DynamicSimulationKeyframe(
                    frame_type="obstacle",
                    agent_position=current_agent_position,
                    planned_path=planned_path,
                    travelled_path=list(travelled_path),
                    dynamic_blocked=set(dynamic_map.dynamic_blocked),
                    new_obstacle_positions=move_result.affected_positions,
                    status_text=(
                        f"Moving obstacle ({size_label}) "
                        f"at ({obstacle.row}, {obstacle.col}) step {step}"
                    ),
                ),
            )

    return (
        current_agent_position,
        collision_count,
        agent_push_count,
        obstacle_blocked_count,
        agent_position_changed,
    )


def _apply_events(
        algorithm,
        dynamic_map: DynamicGridMap,
        events: list[DynamicObstacleEvent],
        step: int,
        event_index: int,
        record_keyframes: bool,
        frame_callback: Callable[[DynamicSimulationKeyframe], None] | None,
        keyframes: list[DynamicSimulationKeyframe],
        agent_position: Position,
        planned_path: list[Position],
        travelled_path: list[Position],
) -> tuple[int, int]:
    applied_events = 0

    while event_index < len(events) and events[event_index].step_index == step:
        event = events[event_index]
        affected_positions = apply_dynamic_event(
            dynamic_map=dynamic_map,
            event=event,
        )

        if isinstance(algorithm, DStarLite):
            algorithm.update_cells(affected_positions)

        applied_events += 1
        event_index += 1

        size_label = (
            f"{event.width}x{event.height}"
            if event.width > 1 or event.height > 1
            else "1x1"
        )
        _emit_keyframe(
            keyframes=keyframes,
            record_keyframes=record_keyframes,
            frame_callback=frame_callback,
            keyframe=DynamicSimulationKeyframe(
                frame_type="obstacle",
                agent_position=agent_position,
                planned_path=planned_path,
                travelled_path=list(travelled_path),
                dynamic_blocked=set(dynamic_map.dynamic_blocked),
                new_obstacle_positions=affected_positions,
                status_text=(
                    f"Obstacle {event.event_type.value} ({size_label}) "
                    f"at step {step}"
                ),
            ),
        )

    return applied_events, event_index


def _create_search_keyframe(
        agent_position: Position,
        planned_path: list[Position],
        travelled_path: list[Position],
        dynamic_map: DynamicGridMap,
        search_steps: list[RawAlgorithmStep],
        status_text: str,
) -> DynamicSimulationKeyframe:
    return DynamicSimulationKeyframe(
        frame_type="search",
        agent_position=agent_position,
        planned_path=planned_path,
        travelled_path=travelled_path,
        dynamic_blocked=set(dynamic_map.dynamic_blocked),
        search_steps=search_steps,
        status_text=status_text,
    )


def _emit_waiting_keyframe(
        keyframes: list[DynamicSimulationKeyframe],
        record_keyframes: bool,
        frame_callback: Callable[[DynamicSimulationKeyframe], None] | None,
        agent_position: Position,
        planned_path: list[Position],
        travelled_path: list[Position],
        dynamic_map: DynamicGridMap,
        waiting_steps: int,
) -> None:
    _emit_keyframe(
        keyframes=keyframes,
        record_keyframes=record_keyframes,
        frame_callback=frame_callback,
        keyframe=DynamicSimulationKeyframe(
            frame_type="path_update",
            agent_position=agent_position,
            planned_path=planned_path,
            travelled_path=list(travelled_path),
            dynamic_blocked=set(dynamic_map.dynamic_blocked),
            status_text=(
                f"Waiting for obstacle to clear corridor... "
                f"(step {waiting_steps})"
            ),
        ),
    )


def _emit_replanning_started_keyframe(
        keyframes: list[DynamicSimulationKeyframe],
        record_keyframes: bool,
        frame_callback: Callable[[DynamicSimulationKeyframe], None] | None,
        agent_position: Position,
        planned_path: list[Position],
        travelled_path: list[Position],
        dynamic_map: DynamicGridMap,
        replanning_count: int,
) -> None:
    if not record_keyframes and frame_callback is None:
        return

    _emit_keyframe(
        keyframes=keyframes,
        record_keyframes=record_keyframes,
        frame_callback=frame_callback,
        keyframe=DynamicSimulationKeyframe(
            frame_type="path_update",
            agent_position=agent_position,
            planned_path=planned_path,
            travelled_path=list(travelled_path),
            dynamic_blocked=set(dynamic_map.dynamic_blocked),
            status_text=f"Replanning #{replanning_count}...",
        ),
    )


def _create_replan_keyframe(
        algorithm,
        agent_position: Position,
        planned_path: list[Position],
        travelled_path: list[Position],
        dynamic_map: DynamicGridMap,
        search_steps: list[RawAlgorithmStep],
        replanning_count: int,
) -> DynamicSimulationKeyframe:
    if isinstance(algorithm, DStarLite):
        return DynamicSimulationKeyframe(
            frame_type="search",
            agent_position=agent_position,
            planned_path=planned_path,
            travelled_path=list(travelled_path),
            dynamic_blocked=set(dynamic_map.dynamic_blocked),
            search_steps=search_steps,
            status_text=f"D* Lite replanning #{replanning_count}",
        )

    return DynamicSimulationKeyframe(
        frame_type="search",
        agent_position=agent_position,
        planned_path=planned_path,
        travelled_path=list(travelled_path),
        dynamic_blocked=set(dynamic_map.dynamic_blocked),
        search_steps=search_steps,
        status_text=f"Replanning #{replanning_count}",
    )


def _build_failed_stats(
        algorithm_name: str,
        total_execution_time_ms: float,
) -> DynamicReplanningStats:
    return DynamicReplanningStats(
        algorithm_name=algorithm_name,
        initial_path_found=False,
        final_goal_reached=False,
        replanning_count=0,
        total_execution_time_ms=total_execution_time_ms,
        total_path_cost=0.0,
        travelled_steps=0,
        dynamic_events_applied=0,
        waiting_steps=0,
        agent_push_count=0,
        obstacle_blocked_count=0,
        collision_count=0,
    )
