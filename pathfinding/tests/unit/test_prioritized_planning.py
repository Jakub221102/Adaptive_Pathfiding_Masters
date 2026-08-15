import pytest

from pathfinding.src.algorithms.mapf.conflicts import detect_conflicts
from pathfinding.src.algorithms.mapf.models import (
    EdgeConflict,
    MAPFAgent,
    MAPFScenario,
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


def _assert_successful_result(
    result,
    scenario: MAPFScenario,
) -> None:
    assert result.success is True
    assert len(result.paths) == len(scenario.agents)
    assert detect_conflicts(result.paths) == ()

    for path, agent in zip(result.paths, scenario.agents, strict=True):
        assert path.agent_id == agent.agent_id
        assert path.states[0].row == agent.start.row
        assert path.states[0].col == agent.start.col
        assert path.states[-1].row == agent.goal.row
        assert path.states[-1].col == agent.goal.col


def test_single_agent_succeeds() -> None:
    grid_map = build_grid_map([[0, 0, 0]])
    scenario = _scenario(_agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=2))

    result = plan_prioritized(grid_map=grid_map, scenario=scenario, max_timestep=5)

    _assert_successful_result(result, scenario)
    assert result.paths[0].agent_id == 0


def test_two_independent_agents_succeed() -> None:
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

    result = plan_prioritized(grid_map=grid_map, scenario=scenario, max_timestep=6)

    _assert_successful_result(result, scenario)


def test_result_paths_follow_scenario_order_not_sorted_ids() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=7, start_row=0, start_col=0, goal_row=0, goal_col=2),
        _agent(agent_id=2, start_row=1, start_col=0, goal_row=1, goal_col=2),
        _agent(agent_id=5, start_row=2, start_col=0, goal_row=2, goal_col=2),
    )

    result = plan_prioritized(grid_map=grid_map, scenario=scenario, max_timestep=6)

    _assert_successful_result(result, scenario)
    assert tuple(path.agent_id for path in result.paths) == (7, 2, 5)


def test_coordination_makes_lower_priority_agent_wait() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=1, start_col=1, goal_row=0, goal_col=0),
        _agent(agent_id=1, start_row=0, start_col=0, goal_row=0, goal_col=2),
    )
    max_timestep = 5

    result = plan_prioritized(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
    )

    _assert_successful_result(result, scenario)

    lower_priority_agent = scenario.agents[1]
    unconstrained_path = find_path(
        grid_map=grid_map,
        agent=lower_priority_agent,
        max_timestep=max_timestep,
    )
    assert unconstrained_path is not None
    assert detect_conflicts((result.paths[0], unconstrained_path)) != ()
    assert result.paths[1] != unconstrained_path


def test_swap_prevention_allows_wait_or_detour() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=2),
        _agent(agent_id=1, start_row=0, start_col=2, goal_row=0, goal_col=0),
    )

    result = plan_prioritized(grid_map=grid_map, scenario=scenario, max_timestep=6)

    _assert_successful_result(result, scenario)
    assert not any(isinstance(conflict, EdgeConflict) for conflict in detect_conflicts(result.paths))


def test_stay_at_goal_blocks_lower_priority_goal_entry_and_fails() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=2),
        _agent(agent_id=1, start_row=1, start_col=0, goal_row=0, goal_col=2),
    )

    result = plan_prioritized(grid_map=grid_map, scenario=scenario, max_timestep=5)

    assert result.success is False
    assert result.paths == ()


def test_impossible_lower_priority_agent_returns_full_failure() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=2),
        _agent(agent_id=1, start_row=1, start_col=0, goal_row=0, goal_col=2),
    )

    result = plan_prioritized(grid_map=grid_map, scenario=scenario, max_timestep=5)

    assert result.success is False
    assert result.paths == ()


def test_failure_does_not_return_partial_paths() -> None:
    grid_map = build_grid_map([[0, 0, 0]])
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=2),
        _agent(agent_id=1, start_row=0, start_col=0, goal_row=0, goal_col=1),
    )

    result = plan_prioritized(grid_map=grid_map, scenario=scenario, max_timestep=5)

    assert result.success is False
    assert result.paths == ()


def test_sufficient_horizon_succeeds() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=4),
        _agent(agent_id=1, start_row=1, start_col=4, goal_row=1, goal_col=0),
    )

    result = plan_prioritized(grid_map=grid_map, scenario=scenario, max_timestep=6)

    _assert_successful_result(result, scenario)


def test_insufficient_horizon_fails() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=4),
        _agent(agent_id=1, start_row=1, start_col=4, goal_row=1, goal_col=0),
    )

    result = plan_prioritized(grid_map=grid_map, scenario=scenario, max_timestep=3)

    assert result.success is False
    assert result.paths == ()


def test_negative_max_timestep_raises_value_error() -> None:
    grid_map = build_grid_map([[0, 0]])
    scenario = _scenario(_agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1))

    with pytest.raises(ValueError, match="max_timestep must be non-negative"):
        plan_prioritized(grid_map=grid_map, scenario=scenario, max_timestep=-1)


def test_same_start_position_fails_for_lower_priority_agent() -> None:
    grid_map = build_grid_map([[0, 0, 0]])
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=2),
        _agent(agent_id=1, start_row=0, start_col=0, goal_row=0, goal_col=1),
    )

    result = plan_prioritized(grid_map=grid_map, scenario=scenario, max_timestep=5)

    assert result.success is False
    assert result.paths == ()


def test_priority_order_can_change_solvability() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    agent_a = _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=2)
    agent_b = _agent(agent_id=1, start_row=0, start_col=1, goal_row=0, goal_col=0)
    max_timestep = 3

    success_first = plan_prioritized(
        grid_map=grid_map,
        scenario=_scenario(agent_a, agent_b),
        max_timestep=max_timestep,
    )
    success_second = plan_prioritized(
        grid_map=grid_map,
        scenario=_scenario(agent_b, agent_a),
        max_timestep=max_timestep,
    )

    assert success_first.success is True
    assert success_second.success is False
    _assert_successful_result(success_first, _scenario(agent_a, agent_b))
    assert success_second.paths == ()
