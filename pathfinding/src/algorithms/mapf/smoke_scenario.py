from __future__ import annotations

from collections.abc import Sequence

from pathfinding.src.algorithms.mapf.models import MAPFAgent, MAPFScenario
from pathfinding.src.core.models import GridMap, Scenario


def _position_key(row: int, col: int) -> tuple[int, int]:
    return row, col


def _is_valid_agent_candidate(
    scenario: Scenario,
    grid_map: GridMap,
    used_starts: set[tuple[int, int]],
    used_goals: set[tuple[int, int]],
) -> bool:
    if (
        scenario.start.row == scenario.goal.row
        and scenario.start.col == scenario.goal.col
    ):
        return False

    start_key = _position_key(scenario.start.row, scenario.start.col)
    goal_key = _position_key(scenario.goal.row, scenario.goal.col)

    if start_key in used_starts or goal_key in used_goals:
        return False

    if not grid_map.in_bounds(scenario.start) or not grid_map.is_walkable(scenario.start):
        return False

    if not grid_map.in_bounds(scenario.goal) or not grid_map.is_walkable(scenario.goal):
        return False

    return True


def build_mapf_scenario(
    scenarios: Sequence[Scenario],
    grid_map: GridMap,
    agent_count: int,
) -> MAPFScenario:
    if agent_count <= 0:
        raise ValueError("agent_count must be positive")

    agents: list[MAPFAgent] = []
    used_starts: set[tuple[int, int]] = set()
    used_goals: set[tuple[int, int]] = set()

    for scenario in scenarios:
        if len(agents) >= agent_count:
            break

        if not _is_valid_agent_candidate(
            scenario,
            grid_map,
            used_starts,
            used_goals,
        ):
            continue

        used_starts.add(_position_key(scenario.start.row, scenario.start.col))
        used_goals.add(_position_key(scenario.goal.row, scenario.goal.col))
        agents.append(
            MAPFAgent(
                agent_id=len(agents),
                start=scenario.start,
                goal=scenario.goal,
            )
        )

    if len(agents) < agent_count:
        raise ValueError(
            f"Could not collect {agent_count} valid MAPF agents from "
            f"{len(scenarios)} scenario entries; only {len(agents)} valid candidates found."
        )

    return MAPFScenario(agents=tuple(agents))
