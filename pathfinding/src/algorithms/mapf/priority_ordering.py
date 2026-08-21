"""Priority orderings for Prioritized Planning.

Conflict-aware orderings (CDF-H, CDF-L, SPF+CD) are static and instance-adaptive:
each order is computed once from independent Space-Time A* paths and their
conflicts before standard PP. They are not online-adaptive during planning.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pathfinding.src.algorithms.mapf.conflicts import detect_conflicts
from pathfinding.src.algorithms.mapf.models import (
    AgentPath,
    Conflict,
    MAPFAgent,
    MAPFScenario,
)
from pathfinding.src.algorithms.mapf.space_time_astar import find_path
from pathfinding.src.core.models import GridMap


def random_priority_order(scenario: MAPFScenario, seed: int) -> MAPFScenario:
    agents = list(scenario.agents)
    rng = random.Random(seed)
    rng.shuffle(agents)
    return MAPFScenario(agents=tuple(agents))


@dataclass(frozen=True, slots=True)
class ConflictAwareOrderingInputs:
    """Independent-path analysis shared by conflict-aware orderings."""

    paths: tuple[AgentPath, ...]
    independent_costs: tuple[int, ...]
    degrees: tuple[int, ...]
    incident_counts: tuple[int, ...]
    conflicts: tuple[Conflict, ...]
    conflict_pair_count: int


def _compute_independent_agent_data(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
) -> tuple[tuple[AgentPath, int, int, MAPFAgent], ...]:
    if max_timestep < 0:
        raise ValueError("max_timestep must be non-negative")

    ranked: list[tuple[AgentPath, int, int, MAPFAgent]] = []
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
        ranked.append((path, independent_cost, original_index, agent))

    return tuple(ranked)


def _independent_path_costs(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
) -> tuple[tuple[int, int, MAPFAgent], ...]:
    return tuple(
        (cost, original_index, agent)
        for _, cost, original_index, agent in _compute_independent_agent_data(
            grid_map=grid_map,
            scenario=scenario,
            max_timestep=max_timestep,
        )
    )


def _agent_id_to_original_index(scenario: MAPFScenario) -> dict[int, int]:
    return {agent.agent_id: index for index, agent in enumerate(scenario.agents)}


def conflict_graph_edges(
    scenario: MAPFScenario,
    conflicts: Sequence[Conflict],
) -> frozenset[tuple[int, int]]:
    id_to_index = _agent_id_to_original_index(scenario)
    edges: set[tuple[int, int]] = set()

    for conflict in conflicts:
        first_index = id_to_index[conflict.agent1_id]
        second_index = id_to_index[conflict.agent2_id]
        if first_index == second_index:
            raise ValueError("conflict must involve two distinct agents")
        edges.add(
            (min(first_index, second_index), max(first_index, second_index))
        )

    return frozenset(edges)


def conflict_degrees_from_edges(
    agent_count: int,
    edges: frozenset[tuple[int, int]],
) -> tuple[int, ...]:
    if agent_count < 0:
        raise ValueError("agent_count must be non-negative")

    degrees = [0] * agent_count
    for first_index, second_index in edges:
        degrees[first_index] += 1
        degrees[second_index] += 1

    return tuple(degrees)


def incident_conflict_counts(
    scenario: MAPFScenario,
    conflicts: Sequence[Conflict],
) -> tuple[int, ...]:
    id_to_index = _agent_id_to_original_index(scenario)
    counts = [0] * len(scenario.agents)

    for conflict in conflicts:
        counts[id_to_index[conflict.agent1_id]] += 1
        counts[id_to_index[conflict.agent2_id]] += 1

    return tuple(counts)


def build_conflict_aware_ordering_inputs(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
) -> ConflictAwareOrderingInputs:
    agent_data = _compute_independent_agent_data(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
    )
    paths = tuple(path for path, _, _, _ in agent_data)
    independent_costs = tuple(cost for _, cost, _, _ in agent_data)
    conflicts = detect_conflicts(paths)
    edges = conflict_graph_edges(scenario, conflicts)
    degrees = conflict_degrees_from_edges(len(scenario.agents), edges)

    return ConflictAwareOrderingInputs(
        paths=paths,
        independent_costs=independent_costs,
        degrees=degrees,
        incident_counts=incident_conflict_counts(scenario, conflicts),
        conflicts=conflicts,
        conflict_pair_count=len(edges),
    )


def _conflict_aware_order(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
    *,
    sort_key: Callable[[int, ConflictAwareOrderingInputs], tuple[int, ...]],
) -> MAPFScenario:
    inputs = build_conflict_aware_ordering_inputs(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
    )
    ranked_indices = sorted(
        range(len(scenario.agents)),
        key=lambda original_index: sort_key(original_index, inputs),
    )
    return MAPFScenario(
        agents=tuple(scenario.agents[index] for index in ranked_indices)
    )


def conflict_degree_first_high_order(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
) -> MAPFScenario:
    """CDF-H: (-degree, independent_cost, original_scenario_index)."""

    return _conflict_aware_order(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
        sort_key=lambda index, inputs: (
            -inputs.degrees[index],
            inputs.independent_costs[index],
            index,
        ),
    )


def conflict_degree_first_low_order(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
) -> MAPFScenario:
    """CDF-L: (degree, independent_cost, original_scenario_index)."""

    return _conflict_aware_order(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
        sort_key=lambda index, inputs: (
            inputs.degrees[index],
            inputs.independent_costs[index],
            index,
        ),
    )


def shortest_path_first_conflict_degree_order(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
) -> MAPFScenario:
    """SPF+CD: (independent_cost, -degree, original_scenario_index)."""

    return _conflict_aware_order(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
        sort_key=lambda index, inputs: (
            inputs.independent_costs[index],
            -inputs.degrees[index],
            index,
        ),
    )


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
