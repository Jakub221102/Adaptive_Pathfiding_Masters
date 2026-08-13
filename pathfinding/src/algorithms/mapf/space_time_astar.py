from __future__ import annotations

import heapq
from collections.abc import Collection
from dataclasses import dataclass

from pathfinding.src.algorithms.mapf.models import (
    AgentPath,
    Constraint,
    EdgeConstraint,
    MAPFAgent,
    TimedState,
    VertexConstraint,
)
from pathfinding.src.core.models import GridMap, Position

_MOVEMENT_DELTAS: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (0, 0),
)


@dataclass(frozen=True, slots=True)
class _ConstraintIndex:
    vertex: frozenset[tuple[int, int, int]]
    edge: frozenset[tuple[int, int, int, int, int]]


def _manhattan_heuristic(row: int, col: int, goal_row: int, goal_col: int) -> int:
    return abs(row - goal_row) + abs(col - goal_col)


def _is_valid_cell(grid_map: GridMap, row: int, col: int) -> bool:
    if row < 0 or col < 0:
        return False
    if row >= grid_map.height or col >= grid_map.width:
        return False

    position = Position(row=row, col=col)
    return grid_map.is_walkable(position)


def _build_constraint_index(
    agent_id: int,
    constraints: Collection[Constraint],
    max_timestep: int,
) -> _ConstraintIndex:
    vertex: set[tuple[int, int, int]] = set()
    edge: set[tuple[int, int, int, int, int]] = set()

    for constraint in constraints:
        if isinstance(constraint, VertexConstraint):
            if constraint.agent_id != agent_id:
                continue
            if constraint.timestep > max_timestep:
                continue
            vertex.add((constraint.row, constraint.col, constraint.timestep))
            continue

        if constraint.agent_id != agent_id:
            continue
        if constraint.timestep > max_timestep:
            continue
        edge.add(
            (
                constraint.from_row,
                constraint.from_col,
                constraint.to_row,
                constraint.to_col,
                constraint.timestep,
            )
        )

    return _ConstraintIndex(vertex=frozenset(vertex), edge=frozenset(edge))


def _is_vertex_forbidden(
    constraint_index: _ConstraintIndex,
    row: int,
    col: int,
    timestep: int,
) -> bool:
    return (row, col, timestep) in constraint_index.vertex


def _is_edge_forbidden(
    constraint_index: _ConstraintIndex,
    current: TimedState,
    successor: TimedState,
) -> bool:
    return (
        current.row,
        current.col,
        successor.row,
        successor.col,
        successor.timestep,
    ) in constraint_index.edge


def _goal_is_safe(
    constraint_index: _ConstraintIndex,
    goal_row: int,
    goal_col: int,
    arrival_timestep: int,
    max_timestep: int,
) -> bool:
    for future_timestep in range(arrival_timestep + 1, max_timestep + 1):
        if _is_vertex_forbidden(
            constraint_index,
            goal_row,
            goal_col,
            future_timestep,
        ):
            return False

        if (
            goal_row,
            goal_col,
            goal_row,
            goal_col,
            future_timestep,
        ) in constraint_index.edge:
            return False

    return True


def _successors(
    grid_map: GridMap,
    state: TimedState,
    max_timestep: int,
    constraint_index: _ConstraintIndex,
) -> tuple[TimedState, ...]:
    if state.timestep >= max_timestep:
        return ()

    next_timestep = state.timestep + 1
    successors: list[TimedState] = []

    for row_delta, col_delta in _MOVEMENT_DELTAS:
        next_row = state.row + row_delta
        next_col = state.col + col_delta

        if not _is_valid_cell(grid_map, next_row, next_col):
            continue

        successor = TimedState(row=next_row, col=next_col, timestep=next_timestep)

        if _is_vertex_forbidden(constraint_index, next_row, next_col, next_timestep):
            continue

        if _is_edge_forbidden(constraint_index, state, successor):
            continue

        successors.append(successor)

    return tuple(successors)


def _reconstruct_path(
    came_from: dict[TimedState, TimedState],
    current: TimedState,
) -> tuple[TimedState, ...]:
    path = [current]

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()
    return tuple(path)


def find_path(
    grid_map: GridMap,
    agent: MAPFAgent,
    max_timestep: int,
    constraints: Collection[Constraint] = (),
) -> AgentPath | None:
    if max_timestep < 0:
        raise ValueError("max_timestep must be non-negative")

    start = agent.start
    goal = agent.goal
    constraint_index = _build_constraint_index(
        agent_id=agent.agent_id,
        constraints=constraints,
        max_timestep=max_timestep,
    )

    if not _is_valid_cell(grid_map, start.row, start.col):
        return None
    if not _is_valid_cell(grid_map, goal.row, goal.col):
        return None

    if _is_vertex_forbidden(constraint_index, start.row, start.col, 0):
        return None

    goal_row = goal.row
    goal_col = goal.col

    if start.row == goal_row and start.col == goal_col:
        if _goal_is_safe(
            constraint_index,
            goal_row,
            goal_col,
            arrival_timestep=0,
            max_timestep=max_timestep,
        ):
            return AgentPath(
                agent_id=agent.agent_id,
                states=(TimedState(row=start.row, col=start.col, timestep=0),),
            )
        if max_timestep == 0:
            return None

    start_state = TimedState(row=start.row, col=start.col, timestep=0)

    open_heap: list[tuple[int, int, int, int, int]] = []
    counter = 0
    g_score: dict[TimedState, int] = {start_state: 0}
    came_from: dict[TimedState, TimedState] = {}
    closed: set[TimedState] = set()

    initial_h = _manhattan_heuristic(start.row, start.col, goal_row, goal_col)
    heapq.heappush(
        open_heap,
        (initial_h, counter, start_state.timestep, start_state.row, start_state.col),
    )
    counter += 1

    while open_heap:
        _, _, timestep, row, col = heapq.heappop(open_heap)
        current = TimedState(row=row, col=col, timestep=timestep)

        if current in closed:
            continue

        closed.add(current)

        if (
            row == goal_row
            and col == goal_col
            and _goal_is_safe(
                constraint_index,
                goal_row,
                goal_col,
                arrival_timestep=timestep,
                max_timestep=max_timestep,
            )
        ):
            return AgentPath(
                agent_id=agent.agent_id,
                states=_reconstruct_path(came_from, current),
            )

        current_g = g_score[current]

        for successor in _successors(grid_map, current, max_timestep, constraint_index):
            if successor in closed:
                continue

            tentative_g = current_g + 1

            if successor not in g_score or tentative_g < g_score[successor]:
                came_from[successor] = current
                g_score[successor] = tentative_g
                h_score = _manhattan_heuristic(
                    successor.row,
                    successor.col,
                    goal_row,
                    goal_col,
                )
                f_score = tentative_g + h_score
                heapq.heappush(
                    open_heap,
                    (f_score, counter, successor.timestep, successor.row, successor.col),
                )
                counter += 1

    return None
