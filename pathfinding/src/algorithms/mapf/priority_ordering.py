from __future__ import annotations

import random

from pathfinding.src.algorithms.mapf.models import MAPFAgent, MAPFScenario
from pathfinding.src.algorithms.mapf.space_time_astar import find_path
from pathfinding.src.core.models import GridMap


def random_priority_order(scenario: MAPFScenario, seed: int) -> MAPFScenario:
    agents = list(scenario.agents)
    rng = random.Random(seed)
    rng.shuffle(agents)
    return MAPFScenario(agents=tuple(agents))


def _independent_path_costs(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
) -> tuple[tuple[int, int, MAPFAgent], ...]:
    if max_timestep < 0:
        raise ValueError("max_timestep must be non-negative")

    ranked: list[tuple[int, int, MAPFAgent]] = []
    for original_index, agent in enumerate(scenario.agents):
        path = find_path(
            grid_map=grid_map,
            agent=agent,
            max_timestep=max_timestep,
            constraints=(),
        )
        if path is None:
            raise ValueError(
                f"no independent path found for agent_id={agent.agent_id} "
                f"at original scenario index {original_index} "
                f"within max_timestep={max_timestep}"
            )

        independent_cost = len(path.states) - 1
        ranked.append((independent_cost, original_index, agent))

    return tuple(ranked)


def shortest_path_first_order(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
) -> MAPFScenario:
    ranked = sorted(
        _independent_path_costs(grid_map, scenario, max_timestep),
        key=lambda item: (item[0], item[1]),
    )
    return MAPFScenario(agents=tuple(agent for _, _, agent in ranked))


def longest_path_first_order(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
) -> MAPFScenario:
    ranked = sorted(
        _independent_path_costs(grid_map, scenario, max_timestep),
        key=lambda item: (-item[0], item[1]),
    )
    return MAPFScenario(agents=tuple(agent for _, _, agent in ranked))
