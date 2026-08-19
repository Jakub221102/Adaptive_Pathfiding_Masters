"""Static 4-connected independent path planner for MAPF benchmark evaluation.

Experimental helper only — not used by production MAPF coordination (PP/CBS).
"""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from pathfinding.src.algorithms.mapf.conflicts import detect_conflicts
from pathfinding.src.algorithms.mapf.metrics import makespan, sum_of_costs
from pathfinding.src.algorithms.mapf.models import AgentPath, MAPFAgent, TimedState
from pathfinding.src.core.models import GridMap, Position, Scenario
from pathfinding.src.experiments.mapf_benchmark_instances import (
    MAPFBenchmarkInstance,
    MAPFInteractionLevel,
    classify_interaction_level,
    count_conflicting_agent_pairs,
    _count_conflict_types,
    _validate_candidate_indices,
)

# Spatial move order matches Space-Time A* (_MOVEMENT_DELTAS) without WAIT.
_SPATIAL_MOVEMENT_DELTAS: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)

SpatialCell = tuple[int, int]


class StaticPathSearchStatus(Enum):
    SUCCESS = "success"
    OVER_COST_BOUND = "over_cost_bound"
    NO_PATH = "no_path"


@dataclass(frozen=True, slots=True)
class BoundedStaticPathResult:
    path: AgentPath | None
    status: StaticPathSearchStatus
    expansions: int = 0


@dataclass(frozen=True, slots=True)
class StaticReachabilityIndex:
    width: int
    height: int
    component_count: int
    walkable_cell_count: int
    _component_ids: tuple[int, ...]

    def component_id_at(self, row: int, col: int) -> int | None:
        if row < 0 or col < 0 or row >= self.height or col >= self.width:
            return None
        component_id = self._component_ids[row * self.width + col]
        if component_id < 0:
            return None
        return component_id


def build_static_reachability_index(grid_map: GridMap) -> StaticReachabilityIndex:
    width = grid_map.width
    height = grid_map.height
    component_ids = [-1] * (width * height)
    walkable_cell_count = 0
    next_component_id = 0

    for row in range(height):
        for col in range(width):
            flat_index = row * width + col
            if component_ids[flat_index] != -1:
                continue

            position = Position(row=row, col=col)
            if not grid_map.is_walkable(position):
                continue

            walkable_cell_count += 1
            component_ids[flat_index] = next_component_id
            queue: deque[SpatialCell] = deque([(row, col)])

            while queue:
                current_row, current_col = queue.popleft()
                for row_delta, col_delta in _SPATIAL_MOVEMENT_DELTAS:
                    next_row = current_row + row_delta
                    next_col = current_col + col_delta
                    if next_row < 0 or next_col < 0:
                        continue
                    if next_row >= height or next_col >= width:
                        continue

                    next_flat_index = next_row * width + next_col
                    if component_ids[next_flat_index] != -1:
                        continue

                    next_position = Position(row=next_row, col=next_col)
                    if not grid_map.is_walkable(next_position):
                        continue

                    walkable_cell_count += 1
                    component_ids[next_flat_index] = next_component_id
                    queue.append((next_row, next_col))

            next_component_id += 1

    return StaticReachabilityIndex(
        width=width,
        height=height,
        component_count=next_component_id,
        walkable_cell_count=walkable_cell_count,
        _component_ids=tuple(component_ids),
    )


def are_spatially_connected(
    index: StaticReachabilityIndex,
    start: Position,
    goal: Position,
) -> bool:
    start_component = index.component_id_at(start.row, start.col)
    if start_component is None:
        return False

    goal_component = index.component_id_at(goal.row, goal.col)
    if goal_component is None:
        return False

    return start_component == goal_component


def _manhattan_heuristic(row: int, col: int, goal_row: int, goal_col: int) -> int:
    return abs(row - goal_row) + abs(col - goal_col)


def _is_valid_cell(grid_map: GridMap, row: int, col: int) -> bool:
    if row < 0 or col < 0:
        return False
    if row >= grid_map.height or col >= grid_map.width:
        return False

    position = Position(row=row, col=col)
    return grid_map.is_walkable(position)


def _is_spatially_reachable(
    grid_map: GridMap,
    start_row: int,
    start_col: int,
    goal_row: int,
    goal_col: int,
) -> bool:
    if (start_row, start_col) == (goal_row, goal_col):
        return True

    visited: set[SpatialCell] = {(start_row, start_col)}
    queue: deque[SpatialCell] = deque([(start_row, start_col)])

    while queue:
        row, col = queue.popleft()
        for row_delta, col_delta in _SPATIAL_MOVEMENT_DELTAS:
            next_row = row + row_delta
            next_col = col + col_delta
            if (next_row, next_col) == (goal_row, goal_col):
                return True
            if (next_row, next_col) in visited:
                continue
            if not _is_valid_cell(grid_map, next_row, next_col):
                continue
            visited.add((next_row, next_col))
            queue.append((next_row, next_col))

    return False


def _static_astar(
    grid_map: GridMap,
    agent: MAPFAgent,
    *,
    max_cost: int | None,
) -> tuple[AgentPath | None, int]:
    start = agent.start
    goal = agent.goal

    if not _is_valid_cell(grid_map, start.row, start.col):
        return None, 0
    if not _is_valid_cell(grid_map, goal.row, goal.col):
        return None, 0

    if start.row == goal.row and start.col == goal.col:
        return (
            AgentPath(
                agent_id=agent.agent_id,
                states=(TimedState(row=start.row, col=start.col, timestep=0),),
            ),
            0,
        )

    goal_row = goal.row
    goal_col = goal.col
    start_cell = (start.row, start.col)

    open_heap: list[tuple[int, int, int, int]] = []
    counter = 0
    g_score: dict[SpatialCell, int] = {start_cell: 0}
    came_from: dict[SpatialCell, SpatialCell] = {}
    closed: set[SpatialCell] = set()
    expansions = 0

    initial_h = _manhattan_heuristic(start.row, start.col, goal_row, goal_col)
    if max_cost is not None and initial_h > max_cost:
        return None, 0

    heapq.heappush(open_heap, (initial_h, counter, start.row, start.col))
    counter += 1

    while open_heap:
        f_score, _, row, col = heapq.heappop(open_heap)
        current = (row, col)

        if max_cost is not None and f_score > max_cost:
            break

        if current in closed:
            continue

        closed.add(current)
        expansions += 1

        if row == goal_row and col == goal_col:
            spatial_path: list[SpatialCell] = [current]
            node = current
            while node in came_from:
                node = came_from[node]
                spatial_path.append(node)
            spatial_path.reverse()

            states = tuple(
                TimedState(row=cell_row, col=cell_col, timestep=timestep)
                for timestep, (cell_row, cell_col) in enumerate(spatial_path)
            )
            return AgentPath(agent_id=agent.agent_id, states=states), expansions

        current_g = g_score[current]
        if max_cost is not None and current_g > max_cost:
            continue

        for row_delta, col_delta in _SPATIAL_MOVEMENT_DELTAS:
            next_row = row + row_delta
            next_col = col + col_delta

            if not _is_valid_cell(grid_map, next_row, next_col):
                continue

            neighbor = (next_row, next_col)
            if neighbor in closed:
                continue

            tentative_g = current_g + 1
            if max_cost is not None and tentative_g > max_cost:
                continue

            h_score = _manhattan_heuristic(next_row, next_col, goal_row, goal_col)
            if max_cost is not None and tentative_g + h_score > max_cost:
                continue

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(
                    open_heap,
                    (tentative_g + h_score, counter, next_row, next_col),
                )
                counter += 1

    return None, expansions


def find_independent_static_path_bounded_result(
    grid_map: GridMap,
    agent: MAPFAgent,
    max_cost: int,
    *,
    reachability_index: StaticReachabilityIndex | None = None,
) -> BoundedStaticPathResult:
    if max_cost < 0:
        raise ValueError("max_cost must be non-negative")

    start = agent.start
    goal = agent.goal

    if not _is_valid_cell(grid_map, start.row, start.col):
        return BoundedStaticPathResult(path=None, status=StaticPathSearchStatus.NO_PATH)
    if not _is_valid_cell(grid_map, goal.row, goal.col):
        return BoundedStaticPathResult(path=None, status=StaticPathSearchStatus.NO_PATH)

    if reachability_index is not None:
        if not are_spatially_connected(reachability_index, start, goal):
            return BoundedStaticPathResult(path=None, status=StaticPathSearchStatus.NO_PATH)

    if start.row == goal.row and start.col == goal.col:
        return BoundedStaticPathResult(
            path=AgentPath(
                agent_id=agent.agent_id,
                states=(TimedState(row=start.row, col=start.col, timestep=0),),
            ),
            status=StaticPathSearchStatus.SUCCESS,
            expansions=0,
        )

    minimum_possible_cost = _manhattan_heuristic(
        start.row,
        start.col,
        goal.row,
        goal.col,
    )
    if minimum_possible_cost > max_cost:
        if reachability_index is not None:
            return BoundedStaticPathResult(
                path=None,
                status=StaticPathSearchStatus.OVER_COST_BOUND,
                expansions=0,
            )
        if _is_spatially_reachable(
            grid_map,
            start.row,
            start.col,
            goal.row,
            goal.col,
        ):
            return BoundedStaticPathResult(
                path=None,
                status=StaticPathSearchStatus.OVER_COST_BOUND,
                expansions=0,
            )
        return BoundedStaticPathResult(
            path=None,
            status=StaticPathSearchStatus.NO_PATH,
            expansions=0,
        )

    path, expansions = _static_astar(grid_map, agent, max_cost=max_cost)
    if path is not None:
        return BoundedStaticPathResult(
            path=path,
            status=StaticPathSearchStatus.SUCCESS,
            expansions=expansions,
        )

    if reachability_index is not None:
        return BoundedStaticPathResult(
            path=None,
            status=StaticPathSearchStatus.OVER_COST_BOUND,
            expansions=expansions,
        )

    if _is_spatially_reachable(
        grid_map,
        start.row,
        start.col,
        goal.row,
        goal.col,
    ):
        return BoundedStaticPathResult(
            path=None,
            status=StaticPathSearchStatus.OVER_COST_BOUND,
            expansions=expansions,
        )

    return BoundedStaticPathResult(
        path=None,
        status=StaticPathSearchStatus.NO_PATH,
        expansions=expansions,
    )


def find_independent_static_path_bounded(
    grid_map: GridMap,
    agent: MAPFAgent,
    max_cost: int,
    *,
    reachability_index: StaticReachabilityIndex | None = None,
) -> AgentPath | None:
    result = find_independent_static_path_bounded_result(
        grid_map=grid_map,
        agent=agent,
        max_cost=max_cost,
        reachability_index=reachability_index,
    )
    if result.status == StaticPathSearchStatus.SUCCESS:
        return result.path
    return None


def find_independent_static_path(
    grid_map: GridMap,
    agent: MAPFAgent,
) -> AgentPath | None:
    path, _ = _static_astar(grid_map, agent, max_cost=None)
    return path


def independent_path_cost(path: AgentPath | None) -> int | None:
    if path is None:
        return None
    return len(path.states) - 1


def independent_path_fits_horizon(path: AgentPath, max_timestep: int) -> bool:
    if max_timestep < 0:
        raise ValueError("max_timestep must be non-negative")
    return independent_path_cost(path) <= max_timestep


def independent_path_excess_moves(path: AgentPath, max_timestep: int) -> int:
    if max_timestep < 0:
        raise ValueError("max_timestep must be non-negative")
    cost = independent_path_cost(path)
    assert cost is not None
    return max(0, cost - max_timestep)


def spatial_trajectory(path: AgentPath) -> tuple[tuple[int, int], ...]:
    return tuple((state.row, state.col) for state in path.states)


def trajectories_equal(path_a: AgentPath, path_b: AgentPath) -> bool:
    return spatial_trajectory(path_a) == spatial_trajectory(path_b)


def select_diagnostic_scenario_indices(
    eligible_indices: Sequence[int],
    *,
    early_count: int,
    middle_count: int,
    late_count: int,
) -> tuple[int, ...]:
    total = early_count + middle_count + late_count
    if total <= 0:
        raise ValueError("diagnostic selection counts must sum to a positive value")
    if len(eligible_indices) < total:
        raise ValueError(
            f"Only {len(eligible_indices)} eligible scenarios; need at least {total}."
        )

    early = eligible_indices[:early_count]
    late = eligible_indices[-late_count:]
    middle_start = max(0, (len(eligible_indices) - middle_count) // 2)
    middle = eligible_indices[middle_start : middle_start + middle_count]
    return tuple(early + middle + late)


def merge_diagnostic_scenario_indices(
    base_indices: Sequence[int],
    special_indices: Sequence[int],
) -> tuple[int, ...]:
    seen: set[int] = set()
    merged: list[int] = []
    for scenario_index in (*base_indices, *special_indices):
        if scenario_index in seen:
            continue
        seen.add(scenario_index)
        merged.append(scenario_index)
    return tuple(merged)


def evaluate_candidate_with_independent_paths(
    grid_map: GridMap,
    scenarios: Sequence[Scenario],
    scenario_indices: Sequence[int],
    independent_paths: Sequence[AgentPath],
) -> MAPFBenchmarkInstance | None:
    if not _validate_candidate_indices(scenario_indices, scenarios, grid_map):
        return None
    if len(independent_paths) != len(scenario_indices):
        raise ValueError("independent_paths length must match scenario_indices")

    conflicts = detect_conflicts(tuple(independent_paths))
    conflict_count = len(conflicts)
    vertex_count, edge_count = _count_conflict_types(conflicts)

    return MAPFBenchmarkInstance(
        instance_id="",
        agent_count=len(scenario_indices),
        interaction_level=classify_interaction_level(conflict_count),
        scenario_indices=tuple(scenario_indices),
        independent_conflict_count=conflict_count,
        conflicting_agent_pair_count=count_conflicting_agent_pairs(conflicts),
        independent_soc=sum_of_costs(independent_paths),
        independent_makespan=makespan(independent_paths),
        vertex_conflict_count=vertex_count,
        edge_conflict_count=edge_count,
    )


def build_independent_paths_from_lookup(
    scenario_indices: Sequence[int],
    lookup: Mapping[int, tuple[TimedState, ...]],
) -> tuple[AgentPath, ...] | None:
    paths: list[AgentPath] = []
    for agent_id, scenario_index in enumerate(scenario_indices):
        states = lookup.get(scenario_index)
        if states is None:
            return None
        paths.append(AgentPath(agent_id=agent_id, states=states))
    return tuple(paths)
