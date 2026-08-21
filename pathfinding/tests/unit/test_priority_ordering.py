import pytest

from pathfinding.src.algorithms.mapf.conflicts import detect_conflicts
from pathfinding.src.algorithms.mapf.models import MAPFAgent, MAPFScenario
from pathfinding.src.algorithms.mapf.priority_ordering import (
    random_priority_order,
    shortest_path_first_order,
)
from pathfinding.src.algorithms.mapf.prioritized_planning import plan_prioritized
from pathfinding.src.algorithms.mapf.space_time_astar import find_path
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


def test_same_seed_produces_identical_ordering() -> None:
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=2),
        _agent(agent_id=1, start_row=1, start_col=0, goal_row=1, goal_col=2),
        _agent(agent_id=2, start_row=2, start_col=0, goal_row=2, goal_col=2),
    )

    first = random_priority_order(scenario, seed=42)
    second = random_priority_order(scenario, seed=42)

    assert tuple(agent.agent_id for agent in first.agents) == tuple(
        agent.agent_id for agent in second.agents
    )


def test_reordered_scenario_contains_each_agent_once() -> None:
    agents = (
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=1, start_row=0, start_col=1, goal_row=0, goal_col=0),
        _agent(agent_id=2, start_row=1, start_col=0, goal_row=1, goal_col=1),
    )
    scenario = _scenario(*agents)

    reordered = random_priority_order(scenario, seed=7)

    assert len(reordered.agents) == len(agents)
    assert {agent.agent_id for agent in reordered.agents} == {agent.agent_id for agent in agents}
    for agent in agents:
        assert agent in reordered.agents


def test_original_scenario_is_unchanged() -> None:
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=2),
        _agent(agent_id=1, start_row=1, start_col=0, goal_row=1, goal_col=2),
    )
    original_order = scenario.agents

    random_priority_order(scenario, seed=99)

    assert scenario.agents == original_order


def test_random_ordering_works_with_plan_prioritized() -> None:
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
    reordered = random_priority_order(scenario, seed=123)

    result = plan_prioritized(grid_map=grid_map, scenario=reordered, max_timestep=6)

    assert result.success is True
    assert len(result.paths) == len(reordered.agents)
    assert detect_conflicts(result.paths) == ()
    for path, agent in zip(result.paths, reordered.agents, strict=True):
        assert path.agent_id == agent.agent_id


def _independent_cost(
    grid_map,
    agent: MAPFAgent,
    max_timestep: int,
) -> int:
    path = find_path(
        grid_map=grid_map,
        agent=agent,
        max_timestep=max_timestep,
    )
    assert path is not None
    return len(path.states) - 1


def test_shortest_path_first_orders_by_ascending_independent_cost() -> None:
    grid_map = build_grid_map([[0, 0, 0, 0, 0]])
    scenario = _scenario(
        _agent(agent_id=7, start_row=0, start_col=0, goal_row=0, goal_col=4),
        _agent(agent_id=3, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=5, start_row=0, start_col=0, goal_row=0, goal_col=3),
    )
    max_timestep = 6

    reordered = shortest_path_first_order(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
    )

    costs = [
        _independent_cost(grid_map, agent, max_timestep)
        for agent in reordered.agents
    ]
    assert costs == sorted(costs)
    assert tuple(agent.agent_id for agent in reordered.agents) == (3, 5, 7)


def test_shortest_path_first_tie_breaks_by_original_scenario_index() -> None:
    grid_map = build_grid_map([[0, 0, 0, 0, 0]])
    agent_a = _agent(agent_id=9, start_row=0, start_col=0, goal_row=0, goal_col=4)
    agent_b = _agent(agent_id=2, start_row=0, start_col=0, goal_row=0, goal_col=1)
    agent_c = _agent(agent_id=8, start_row=0, start_col=0, goal_row=0, goal_col=4)
    scenario = _scenario(agent_a, agent_b, agent_c)
    max_timestep = 6

    reordered = shortest_path_first_order(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
    )

    assert _independent_cost(grid_map, agent_a, max_timestep) == 4
    assert _independent_cost(grid_map, agent_c, max_timestep) == 4
    assert tuple(agent.agent_id for agent in reordered.agents) == (2, 9, 8)


def test_shortest_path_first_does_not_mutate_original_scenario() -> None:
    grid_map = build_grid_map([[0, 0, 0]])
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=2),
        _agent(agent_id=1, start_row=0, start_col=0, goal_row=0, goal_col=1),
    )
    original_order = scenario.agents

    shortest_path_first_order(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
    )

    assert scenario.agents == original_order


def test_shortest_path_first_contains_each_original_agent_once() -> None:
    agents = (
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=2),
        _agent(agent_id=1, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=2, start_row=0, start_col=0, goal_row=0, goal_col=3),
    )
    scenario = _scenario(*agents)
    grid_map = build_grid_map([[0, 0, 0, 0]])

    reordered = shortest_path_first_order(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=6,
    )

    assert len(reordered.agents) == len(agents)
    for agent in agents:
        assert agent in reordered.agents


def test_shortest_path_first_is_deterministic() -> None:
    grid_map = build_grid_map([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=4),
        _agent(agent_id=1, start_row=1, start_col=0, goal_row=1, goal_col=2),
        _agent(agent_id=2, start_row=0, start_col=0, goal_row=1, goal_col=4),
    )

    first = shortest_path_first_order(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=8,
    )
    second = shortest_path_first_order(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=8,
    )

    assert first.agents == second.agents


def test_shortest_path_first_raises_when_independent_path_missing() -> None:
    grid_map = build_grid_map([[0, 0, 0]])
    scenario = _scenario(
        _agent(agent_id=4, start_row=0, start_col=0, goal_row=0, goal_col=2),
    )

    with pytest.raises(ValueError, match="agent_id=4"):
        shortest_path_first_order(
            grid_map=grid_map,
            scenario=scenario,
            max_timestep=1,
        )


def test_shortest_path_first_works_with_plan_prioritized() -> None:
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
    reordered = shortest_path_first_order(
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
