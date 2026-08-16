from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pathfinding.src.algorithms.mapf.metrics import makespan
from pathfinding.src.algorithms.mapf.models import AgentPath, MAPFScenario
from pathfinding.src.core.models import GridMap, Position


@dataclass(frozen=True, slots=True)
class MapViewport:
    min_row: int
    min_col: int
    max_row: int
    max_col: int

    @property
    def width_cells(self) -> int:
        return self.max_col - self.min_col + 1

    @property
    def height_cells(self) -> int:
        return self.max_row - self.min_row + 1


def position_at_timestep(path: AgentPath, timestep: int) -> tuple[int, int]:
    first_timestep = path.states[0].timestep

    if timestep < first_timestep:
        return path.states[0].row, path.states[0].col

    final_state = path.states[-1]
    if timestep >= final_state.timestep:
        return final_state.row, final_state.col

    index = timestep - first_timestep
    state = path.states[index]
    return state.row, state.col


def position_at_timestep_as_position(path: AgentPath, timestep: int) -> Position:
    row, col = position_at_timestep(path, timestep)
    return Position(row=row, col=col)


def animation_max_timestep(paths: Sequence[AgentPath]) -> int:
    return makespan(paths)


def compute_mapf_viewport(
    grid_map: GridMap,
    scenario: MAPFScenario,
    paths: Sequence[AgentPath],
    padding: int = 2,
) -> MapViewport:
    if padding < 0:
        raise ValueError("padding must be non-negative")

    rows: list[int] = []
    cols: list[int] = []

    for agent in scenario.agents:
        rows.extend((agent.start.row, agent.goal.row))
        cols.extend((agent.start.col, agent.goal.col))

    for path in paths:
        for state in path.states:
            rows.append(state.row)
            cols.append(state.col)

    if not rows or not cols:
        return MapViewport(
            min_row=0,
            min_col=0,
            max_row=grid_map.height - 1,
            max_col=grid_map.width - 1,
        )

    min_row = max(0, min(rows) - padding)
    min_col = max(0, min(cols) - padding)
    max_row = min(grid_map.height - 1, max(rows) + padding)
    max_col = min(grid_map.width - 1, max(cols) + padding)

    return MapViewport(
        min_row=min_row,
        min_col=min_col,
        max_row=max_row,
        max_col=max_col,
    )


def calculate_viewport_cell_size(
    viewport: MapViewport,
    max_window_width: int,
    max_window_height: int,
) -> int:
    if max_window_width <= 0 or max_window_height <= 0:
        raise ValueError("window dimensions must be positive")

    width_scale = max_window_width // viewport.width_cells
    height_scale = max_window_height // viewport.height_cells
    return max(min(width_scale, height_scale), 1)
