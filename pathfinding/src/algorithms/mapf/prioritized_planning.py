from __future__ import annotations

from pathfinding.src.algorithms.mapf.models import AgentPath, MAPFResult, MAPFScenario
from pathfinding.src.algorithms.mapf.reservations import build_reservation_constraints
from pathfinding.src.algorithms.mapf.space_time_astar import find_path
from pathfinding.src.core.models import GridMap


def plan_prioritized(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
) -> MAPFResult:
    if max_timestep < 0:
        raise ValueError("max_timestep must be non-negative")

    planned_paths: list[AgentPath] = []

    for agent in scenario.agents:
        reservations = build_reservation_constraints(
            planned_paths=planned_paths,
            target_agent_id=agent.agent_id,
            max_timestep=max_timestep,
        )
        path = find_path(
            grid_map=grid_map,
            agent=agent,
            max_timestep=max_timestep,
            constraints=reservations,
        )
        if path is None:
            return MAPFResult(success=False, paths=())

        planned_paths.append(path)

    return MAPFResult(success=True, paths=tuple(planned_paths))
