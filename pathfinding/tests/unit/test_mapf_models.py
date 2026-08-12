import pytest

from pathfinding.src.algorithms.mapf.models import MAPFAgent, TimedState
from pathfinding.src.core.models import Position


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
