from pathfinding.src.algorithms.mapf.conflicts import detect_conflicts
from pathfinding.src.algorithms.mapf.models import MAPFAgent, MAPFScenario
from pathfinding.src.algorithms.mapf.priority_ordering import random_priority_order
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
