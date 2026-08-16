from __future__ import annotations

from dataclasses import dataclass

from pathfinding.src.algorithms.mapf.conflicts import detect_conflicts
from pathfinding.src.algorithms.mapf.metrics import sum_of_costs
from pathfinding.src.algorithms.mapf.models import (
    AgentPath,
    Conflict,
    Constraint,
    MAPFScenario,
)
from pathfinding.src.algorithms.mapf.space_time_astar import find_path
from pathfinding.src.core.models import GridMap


@dataclass(frozen=True, slots=True)
class CBSNode:
    constraints: tuple[Constraint, ...]
    paths: tuple[AgentPath, ...]
    cost: int
    conflicts: tuple[Conflict, ...]


def build_cbs_root(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
) -> CBSNode | None:
    if max_timestep < 0:
        raise ValueError("max_timestep must be non-negative")

    paths: list[AgentPath] = []

    for agent in scenario.agents:
        path = find_path(
            grid_map=grid_map,
            agent=agent,
            max_timestep=max_timestep,
            constraints=(),
        )
        if path is None:
            return None

        paths.append(path)

    path_tuple = tuple(paths)
    return CBSNode(
        constraints=(),
        paths=path_tuple,
        cost=sum_of_costs(path_tuple),
        conflicts=detect_conflicts(path_tuple),
    )
