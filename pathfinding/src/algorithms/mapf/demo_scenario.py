from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pathfinding.src.algorithms.mapf.models import AgentPath, MAPFAgent, MAPFScenario
from pathfinding.src.algorithms.mapf.smoke_scenario import _is_valid_agent_candidate
from pathfinding.src.core.models import GridMap, Position, Scenario

# Demo fixture discovered once from AR0204SR.map.scen using a deterministic
# search over consecutive long-path entries. These indices are a visualization
# fixture, not benchmark sampling methodology.
NON_TRIVIAL_DEMO_SCENARIO_INDICES: tuple[int, ...] = (
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
    151,
)
NON_TRIVIAL_DEMO_MIN_OPTIMAL_LENGTH = 50.0
NON_TRIVIAL_DEMO_AGENT_COUNT = len(NON_TRIVIAL_DEMO_SCENARIO_INDICES)


@dataclass(frozen=True, slots=True)
class DemoAgentSpec:
    agent_id: int
    scenario_index: int
    start: Position
    goal: Position
    reference_optimal_length: float | None


@dataclass(frozen=True, slots=True)
class DemoScenarioSelection:
    scenario: MAPFScenario
    agents: tuple[DemoAgentSpec, ...]
    scenario_indices: tuple[int, ...]


def build_mapf_scenario_from_indices(
    scenarios: Sequence[Scenario],
    grid_map: GridMap,
    scenario_indices: Sequence[int],
) -> DemoScenarioSelection:
    if not scenario_indices:
        raise ValueError("scenario_indices must not be empty")

    agents: list[MAPFAgent] = []
    specs: list[DemoAgentSpec] = []
    used_starts: set[tuple[int, int]] = set()
    used_goals: set[tuple[int, int]] = set()

    for agent_id, scenario_index in enumerate(scenario_indices):
        if scenario_index < 0 or scenario_index >= len(scenarios):
            raise ValueError(
                f"scenario index {scenario_index} is out of range for "
                f"{len(scenarios)} loaded scenarios"
            )

        scenario = scenarios[scenario_index]
        if not _is_valid_agent_candidate(
            scenario,
            grid_map,
            used_starts,
            used_goals,
        ):
            raise ValueError(
                f"scenario index {scenario_index} is not a valid MAPF agent "
                f"candidate for agent_id={agent_id}"
            )

        used_starts.add((scenario.start.row, scenario.start.col))
        used_goals.add((scenario.goal.row, scenario.goal.col))
        agents.append(
            MAPFAgent(
                agent_id=agent_id,
                start=scenario.start,
                goal=scenario.goal,
            )
        )
        specs.append(
            DemoAgentSpec(
                agent_id=agent_id,
                scenario_index=scenario_index,
                start=scenario.start,
                goal=scenario.goal,
                reference_optimal_length=scenario.optimal_length,
            )
        )

    indices = tuple(scenario_indices)
    return DemoScenarioSelection(
        scenario=MAPFScenario(agents=tuple(agents)),
        agents=tuple(specs),
        scenario_indices=indices,
    )


def build_non_trivial_demo_scenario(
    scenarios: Sequence[Scenario],
    grid_map: GridMap,
) -> DemoScenarioSelection:
    return build_mapf_scenario_from_indices(
        scenarios=scenarios,
        grid_map=grid_map,
        scenario_indices=NON_TRIVIAL_DEMO_SCENARIO_INDICES,
    )
