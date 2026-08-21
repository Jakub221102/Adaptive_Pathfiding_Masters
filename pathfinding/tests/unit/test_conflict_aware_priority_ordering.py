import pytest

from pathfinding.src.algorithms.mapf.conflicts import detect_conflicts
from pathfinding.src.algorithms.mapf.models import (
    AgentPath,
    EdgeConflict,
    MAPFAgent,
    MAPFScenario,
    TimedState,
    VertexConflict,
)
from pathfinding.src.algorithms.mapf.priority_ordering import (
    build_conflict_aware_ordering_inputs,
    conflict_degree_first_high_order,
    conflict_degree_first_low_order,
    conflict_degrees_from_edges,
    conflict_graph_edges,
    incident_conflict_counts,
    shortest_path_first_conflict_degree_order,
    shortest_path_first_order,
)
from pathfinding.src.algorithms.mapf.prioritized_planning import plan_prioritized
from pathfinding.src.core.models import Position
from pathfinding.tests.helpers import build_grid_map


def _agent(
    agent_id: int,
    start_row: int,
    start_col: int,
    goal_row: int,
    goal_col: int,
) -> MAPFAgent:
    return MAPFAgent(
        agent_id=agent_id,
        start=Position(row=start_row, col=start_col),
        goal=Position(row=goal_row, col=goal_col),
    )


def _scenario(*agents: MAPFAgent) -> MAPFScenario:
    return MAPFScenario(agents=agents)


def _path(agent_id: int, coordinates: list[tuple[int, int, int]]) -> AgentPath:
    return AgentPath(
        agent_id=agent_id,
        states=tuple(
            TimedState(row=row, col=col, timestep=timestep)
            for row, col, timestep in coordinates
        ),
    )


def _original_indices(
    reordered: MAPFScenario,
    original: MAPFScenario,
) -> tuple[int, ...]:
    return tuple(original.agents.index(agent) for agent in reordered.agents)


def test_conflict_graph_edges_deduplicates_pairs_by_original_index() -> None:
    scenario = _scenario(
        _agent(agent_id=9, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=2, start_row=1, start_col=0, goal_row=1, goal_col=1),
    )
    conflicts = (
        VertexConflict(agent1_id=9, agent2_id=2, row=0, col=0, timestep=1),
        VertexConflict(agent1_id=9, agent2_id=2, row=0, col=0, timestep=2),
    )

    edges = conflict_graph_edges(scenario, conflicts)

    assert edges == frozenset({(0, 1)})


def test_edge_conflict_creates_unweighted_graph_edge() -> None:
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=1, start_row=0, start_col=1, goal_row=0, goal_col=0),
    )
    conflicts = (
        EdgeConflict(
            agent1_id=0,
            agent2_id=1,
            agent1_from_row=0,
            agent1_from_col=0,
            agent1_to_row=0,
            agent1_to_col=1,
            agent2_from_row=0,
            agent2_from_col=1,
            agent2_to_row=0,
            agent2_to_col=0,
            timestep=1,
        ),
    )

    edges = conflict_graph_edges(scenario, conflicts)
    degrees = conflict_degrees_from_edges(len(scenario.agents), edges)

    assert edges == frozenset({(0, 1)})
    assert degrees == (1, 1)


def test_incident_conflict_counts_track_events_not_pairs() -> None:
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=1, start_row=1, start_col=0, goal_row=1, goal_col=1),
    )
    conflicts = (
        VertexConflict(agent1_id=0, agent2_id=1, row=0, col=0, timestep=1),
        VertexConflict(agent1_id=0, agent2_id=1, row=0, col=0, timestep=2),
    )

    assert incident_conflict_counts(scenario, conflicts) == (2, 2)
    assert len(conflict_graph_edges(scenario, conflicts)) == 1


def test_cdf_h_sort_matches_frozen_key() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=2),
        _agent(agent_id=1, start_row=2, start_col=0, goal_row=2, goal_col=2),
        _agent(agent_id=2, start_row=0, start_col=1, goal_row=2, goal_col=1),
    )
    max_timestep = 8

    inputs = build_conflict_aware_ordering_inputs(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
    )
    reordered = conflict_degree_first_high_order(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
    )

    expected = tuple(
        sorted(
            range(len(scenario.agents)),
            key=lambda index: (
                -inputs.degrees[index],
                inputs.independent_costs[index],
                index,
            ),
        )
    )
    assert _original_indices(reordered, scenario) == expected


def test_cdf_l_sort_matches_frozen_key() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=2),
        _agent(agent_id=1, start_row=2, start_col=0, goal_row=2, goal_col=2),
        _agent(agent_id=2, start_row=0, start_col=1, goal_row=2, goal_col=1),
    )
    max_timestep = 8

    inputs = build_conflict_aware_ordering_inputs(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
    )
    reordered = conflict_degree_first_low_order(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
    )

    expected = tuple(
        sorted(
            range(len(scenario.agents)),
            key=lambda index: (
                inputs.degrees[index],
                inputs.independent_costs[index],
                index,
            ),
        )
    )
    assert _original_indices(reordered, scenario) == expected


def test_spf_cd_sort_matches_frozen_key() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=2),
        _agent(agent_id=1, start_row=2, start_col=0, goal_row=2, goal_col=2),
        _agent(agent_id=2, start_row=0, start_col=1, goal_row=2, goal_col=1),
    )
    max_timestep = 8

    inputs = build_conflict_aware_ordering_inputs(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
    )
    reordered = shortest_path_first_conflict_degree_order(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
    )

    expected = tuple(
        sorted(
            range(len(scenario.agents)),
            key=lambda index: (
                inputs.independent_costs[index],
                -inputs.degrees[index],
                index,
            ),
        )
    )
    assert _original_indices(reordered, scenario) == expected


def test_cdf_h_tie_breaks_by_original_scenario_index_not_agent_id() -> None:
    grid_map = build_grid_map([[0, 0, 0, 0, 0]])
    agent_a = _agent(agent_id=9, start_row=0, start_col=0, goal_row=0, goal_col=4)
    agent_b = _agent(agent_id=2, start_row=0, start_col=0, goal_row=0, goal_col=1)
    agent_c = _agent(agent_id=8, start_row=0, start_col=0, goal_row=0, goal_col=4)
    scenario = _scenario(agent_a, agent_b, agent_c)
    max_timestep = 6

    reordered = conflict_degree_first_high_order(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
    )

    assert _original_indices(reordered, scenario) == (1, 0, 2)


def test_spf_cd_matches_spf_when_independent_costs_are_unique() -> None:
    grid_map = build_grid_map([[0, 0, 0, 0, 0]])
    scenario = _scenario(
        _agent(agent_id=7, start_row=0, start_col=0, goal_row=0, goal_col=4),
        _agent(agent_id=3, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=5, start_row=0, start_col=0, goal_row=0, goal_col=3),
    )
    max_timestep = 6

    spf = shortest_path_first_order(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
    )
    spf_cd = shortest_path_first_conflict_degree_order(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
    )

    assert spf.agents == spf_cd.agents


def test_conflict_aware_ordering_is_deterministic() -> None:
    grid_map = build_grid_map([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=4),
        _agent(agent_id=1, start_row=1, start_col=0, goal_row=1, goal_col=2),
        _agent(agent_id=2, start_row=0, start_col=0, goal_row=1, goal_col=4),
    )

    first = conflict_degree_first_high_order(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=8,
    )
    second = conflict_degree_first_high_order(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=8,
    )

    assert first.agents == second.agents


def test_conflict_aware_ordering_does_not_mutate_original_scenario() -> None:
    grid_map = build_grid_map([[0, 0, 0]])
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=2),
        _agent(agent_id=1, start_row=0, start_col=0, goal_row=0, goal_col=1),
    )
    original_order = scenario.agents

    conflict_degree_first_high_order(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
    )

    assert scenario.agents == original_order


def test_conflict_aware_ordering_contains_each_original_agent_once() -> None:
    agents = (
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=2),
        _agent(agent_id=1, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=2, start_row=0, start_col=0, goal_row=0, goal_col=3),
    )
    scenario = _scenario(*agents)
    grid_map = build_grid_map([[0, 0, 0, 0]])

    reordered = conflict_degree_first_high_order(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=6,
    )

    assert len(reordered.agents) == len(agents)
    for agent in agents:
        assert agent in reordered.agents


def test_conflict_aware_ordering_raises_when_independent_path_missing() -> None:
    grid_map = build_grid_map([[0, 0, 0]])
    scenario = _scenario(
        _agent(agent_id=4, start_row=0, start_col=0, goal_row=0, goal_col=2),
    )

    with pytest.raises(ValueError, match="agent_id=4"):
        conflict_degree_first_high_order(
            grid_map=grid_map,
            scenario=scenario,
            max_timestep=1,
        )


def test_build_conflict_aware_ordering_inputs_aligns_with_detect_conflicts() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=2),
        _agent(agent_id=1, start_row=1, start_col=0, goal_row=1, goal_col=2),
    )

    inputs = build_conflict_aware_ordering_inputs(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=6,
    )

    assert inputs.conflicts == detect_conflicts(inputs.paths)
    assert inputs.conflict_pair_count == len(
        conflict_graph_edges(scenario, inputs.conflicts)
    )
    assert inputs.degrees == conflict_degrees_from_edges(
        len(scenario.agents),
        conflict_graph_edges(scenario, inputs.conflicts),
    )


def test_cdf_h_works_with_plan_prioritized() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=3),
        _agent(agent_id=1, start_row=1, start_col=0, goal_row=1, goal_col=3),
    )
    reordered = conflict_degree_first_high_order(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=6,
    )

    result = plan_prioritized(grid_map=grid_map, scenario=reordered, max_timestep=6)

    assert result.success is True
    assert len(result.paths) == len(reordered.agents)
    assert detect_conflicts(result.paths) == ()
    for path, agent in zip(result.paths, reordered.agents, strict=True):
        assert path.agent_id == agent.agent_id
