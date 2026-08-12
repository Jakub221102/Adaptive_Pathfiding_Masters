import pytest

from pathfinding.src.algorithms.mapf.models import (
    AgentPath,
    MAPFAgent,
    MAPFResult,
    MAPFScenario,
    TimedState,
)
from pathfinding.src.core.models import Position


def _make_agent(agent_id: int, start_row: int, start_col: int, goal_row: int, goal_col: int) -> MAPFAgent:
    return MAPFAgent(
        agent_id=agent_id,
        start=Position(row=start_row, col=start_col),
        goal=Position(row=goal_row, col=goal_col),
    )


def test_mapf_agent_stores_id_start_and_goal() -> None:
    start = Position(row=1, col=2)
    goal = Position(row=3, col=4)
    agent = MAPFAgent(agent_id=7, start=start, goal=goal)

    assert agent.agent_id == 7
    assert agent.start == start
    assert agent.goal == goal


def test_equal_timed_states_compare_equal() -> None:
    first = TimedState(row=0, col=1, timestep=3)
    second = TimedState(row=0, col=1, timestep=3)

    assert first == second


def test_equal_timed_states_have_equal_hashes() -> None:
    first = TimedState(row=0, col=1, timestep=3)
    second = TimedState(row=0, col=1, timestep=3)

    assert hash(first) == hash(second)


def test_timed_states_are_hashable() -> None:
    state = TimedState(row=2, col=5, timestep=1)

    states = {state, TimedState(row=2, col=5, timestep=1)}
    lookup = {state: "occupied"}

    assert len(states) == 1
    assert lookup[state] == "occupied"


def test_different_timesteps_produce_different_timed_states() -> None:
    earlier = TimedState(row=0, col=0, timestep=0)
    later = TimedState(row=0, col=0, timestep=1)

    assert earlier != later
    assert len({earlier, later}) == 2


def test_from_position_copies_row_and_col() -> None:
    position = Position(row=4, col=7)
    state = TimedState.from_position(position=position, timestep=2)

    assert state.row == 4
    assert state.col == 7
    assert state.timestep == 2


def test_mutating_position_does_not_affect_timed_state() -> None:
    position = Position(row=1, col=2)
    state = TimedState.from_position(position=position, timestep=0)
    states = {state}

    position.row = 9
    position.col = 8

    assert state.row == 1
    assert state.col == 2
    assert state in states
    assert states == {TimedState(row=1, col=2, timestep=0)}


@pytest.mark.parametrize(
    ("row", "col", "timestep", "expected_message"),
    [
        (-1, 0, 0, "row must be non-negative"),
        (0, -1, 0, "col must be non-negative"),
        (0, 0, -1, "timestep must be non-negative"),
    ],
)
def test_negative_coordinates_or_timestep_are_rejected(
        row: int,
        col: int,
        timestep: int,
        expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        TimedState(row=row, col=col, timestep=timestep)


def test_mapf_scenario_accepts_multiple_unique_agents() -> None:
    agents = (
        _make_agent(agent_id=0, start_row=0, start_col=0, goal_row=1, goal_col=1),
        _make_agent(agent_id=1, start_row=2, start_col=2, goal_row=3, goal_col=3),
    )

    scenario = MAPFScenario(agents=agents)

    assert len(scenario.agents) == 2
    assert scenario.agents[0].agent_id == 0
    assert scenario.agents[1].agent_id == 1


def test_mapf_scenario_rejects_empty_agents() -> None:
    with pytest.raises(ValueError, match="scenario must contain at least one agent"):
        MAPFScenario(agents=())


def test_mapf_scenario_rejects_duplicate_agent_id() -> None:
    agents = (
        _make_agent(agent_id=0, start_row=0, start_col=0, goal_row=1, goal_col=1),
        _make_agent(agent_id=0, start_row=2, start_col=2, goal_row=3, goal_col=3),
    )

    with pytest.raises(ValueError, match="agent_id must be unique"):
        MAPFScenario(agents=agents)


def test_mapf_scenario_preserves_agent_order() -> None:
    second = _make_agent(agent_id=2, start_row=0, start_col=0, goal_row=1, goal_col=1)
    first = _make_agent(agent_id=1, start_row=2, start_col=2, goal_row=3, goal_col=3)
    scenario = MAPFScenario(agents=(second, first))

    assert [agent.agent_id for agent in scenario.agents] == [2, 1]


def test_agent_path_accepts_consecutive_timed_path() -> None:
    path = AgentPath(
        agent_id=0,
        states=(
            TimedState(row=0, col=0, timestep=0),
            TimedState(row=0, col=1, timestep=1),
            TimedState(row=1, col=1, timestep=2),
        ),
    )

    assert path.agent_id == 0
    assert len(path.states) == 3


def test_agent_path_accepts_wait_step() -> None:
    path = AgentPath(
        agent_id=0,
        states=(
            TimedState(row=2, col=3, timestep=4),
            TimedState(row=2, col=3, timestep=5),
            TimedState(row=2, col=4, timestep=6),
        ),
    )

    assert path.states[0].row == path.states[1].row == 2
    assert path.states[0].col == path.states[1].col == 3


def test_agent_path_accepts_non_zero_start_timestep() -> None:
    path = AgentPath(
        agent_id=1,
        states=(
            TimedState(row=1, col=1, timestep=3),
            TimedState(row=1, col=2, timestep=4),
        ),
    )

    assert path.states[0].timestep == 3


def test_agent_path_rejects_empty_states() -> None:
    with pytest.raises(ValueError, match="states must not be empty"):
        AgentPath(agent_id=0, states=())


def test_agent_path_rejects_timestep_gap() -> None:
    with pytest.raises(ValueError, match="timesteps must be strictly consecutive"):
        AgentPath(
            agent_id=0,
            states=(
                TimedState(row=2, col=3, timestep=4),
                TimedState(row=2, col=4, timestep=6),
            ),
        )


def test_agent_path_rejects_duplicate_timestep() -> None:
    with pytest.raises(ValueError, match="timesteps must be strictly consecutive"):
        AgentPath(
            agent_id=0,
            states=(
                TimedState(row=2, col=3, timestep=4),
                TimedState(row=2, col=4, timestep=4),
            ),
        )


def test_mapf_result_stores_successful_paths() -> None:
    paths = (
        AgentPath(
            agent_id=0,
            states=(
                TimedState(row=0, col=0, timestep=0),
                TimedState(row=0, col=1, timestep=1),
            ),
        ),
        AgentPath(
            agent_id=1,
            states=(
                TimedState(row=2, col=2, timestep=0),
                TimedState(row=2, col=3, timestep=1),
            ),
        ),
    )
    result = MAPFResult(success=True, paths=paths)

    assert result.success is True
    assert len(result.paths) == 2
    assert result.paths[0].agent_id == 0
    assert result.paths[1].agent_id == 1


def test_mapf_result_allows_failed_result_with_no_paths() -> None:
    result = MAPFResult(success=False, paths=())

    assert result.success is False
    assert result.paths == ()
