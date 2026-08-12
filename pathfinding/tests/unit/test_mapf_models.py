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
    position = Position(row=0, col=1)
    first = TimedState(position=position, timestep=3)
    second = TimedState(position=Position(row=0, col=1), timestep=3)

    assert first == second


def test_timed_states_are_hashable() -> None:
    position = Position(row=2, col=5)
    state = TimedState(position=position, timestep=1)

    states = {state, TimedState(position=Position(row=2, col=5), timestep=1)}
    lookup = {state: "occupied"}

    assert len(states) == 1
    assert lookup[state] == "occupied"


def test_different_timesteps_produce_different_timed_states() -> None:
    position = Position(row=0, col=0)
    earlier = TimedState(position=position, timestep=0)
    later = TimedState(position=position, timestep=1)

    assert earlier != later
    assert len({earlier, later}) == 2


def test_negative_timestep_is_rejected() -> None:
    with pytest.raises(ValueError, match="timestep must be non-negative"):
        TimedState(position=Position(row=0, col=0), timestep=-1)
