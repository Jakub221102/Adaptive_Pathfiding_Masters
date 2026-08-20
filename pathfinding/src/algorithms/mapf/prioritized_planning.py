from __future__ import annotations

from dataclasses import dataclass

from pathfinding.src.algorithms.mapf.models import AgentPath, MAPFResult, MAPFScenario
from pathfinding.src.algorithms.mapf.reservations import build_reservation_constraints
from pathfinding.src.algorithms.mapf.space_time_astar import find_path
from pathfinding.src.core.models import GridMap


@dataclass(frozen=True, slots=True)
class PrioritizedPlanningStats:
    low_level_searches: int
    agents_planned: int


@dataclass(frozen=True, slots=True)
class PrioritizedPlanningRunResult:
    result: MAPFResult
    stats: PrioritizedPlanningStats
    termination_reason: str


def _plan_prioritized_internal(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
) -> PrioritizedPlanningRunResult:
    planned_paths: list[AgentPath] = []
    low_level_searches = 0

    for agent in scenario.agents:
        reservations = build_reservation_constraints(
            planned_paths=planned_paths,
            target_agent_id=agent.agent_id,
            max_timestep=max_timestep,
        )
        low_level_searches += 1
        path = find_path(
            grid_map=grid_map,
            agent=agent,
            max_timestep=max_timestep,
            constraints=reservations,
        )
        if path is None:
            return PrioritizedPlanningRunResult(
                result=MAPFResult(success=False, paths=()),
                stats=PrioritizedPlanningStats(
                    low_level_searches=low_level_searches,
                    agents_planned=len(planned_paths),
                ),
                termination_reason="failure",
            )

        planned_paths.append(path)

    return PrioritizedPlanningRunResult(
        result=MAPFResult(success=True, paths=tuple(planned_paths)),
        stats=PrioritizedPlanningStats(
            low_level_searches=low_level_searches,
            agents_planned=len(planned_paths),
        ),
        termination_reason="success",
    )


def plan_prioritized(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
) -> MAPFResult:
    if max_timestep < 0:
        raise ValueError("max_timestep must be non-negative")

    return _plan_prioritized_internal(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
    ).result


def plan_prioritized_with_stats(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
) -> PrioritizedPlanningRunResult:
    if max_timestep < 0:
        raise ValueError("max_timestep must be non-negative")

    return _plan_prioritized_internal(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
    )
