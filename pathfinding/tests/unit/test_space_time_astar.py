import pytest

from pathfinding.src.algorithms.mapf.models import AgentPath, MAPFAgent, TimedState
from pathfinding.src.algorithms.mapf.space_time_astar import _successors, find_path
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


def test_adjacent_start_and_goal_returns_two_states() -> None:
    grid_map = build_grid_map([[0, 0, 0]])
    agent = _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1)

    path = find_path(grid_map=grid_map, agent=agent, max_timestep=5)

    assert path is not None
    assert path.states == (
        TimedState(row=0, col=0, timestep=0),
        TimedState(row=0, col=1, timestep=1),
    )


def test_multi_step_straight_path_has_consecutive_timesteps() -> None:
    grid_map = build_grid_map([[0, 0, 0, 0]])
    agent = _agent(agent_id=1, start_row=0, start_col=0, goal_row=0, goal_col=3)

    path = find_path(grid_map=grid_map, agent=agent, max_timestep=10)

    assert path is not None
    assert [state.timestep for state in path.states] == [0, 1, 2, 3]
    assert path.states[0].row == path.states[0].col == 0
    assert path.states[-1].col == 3


def test_returned_agent_id_matches_input_agent() -> None:
    grid_map = build_grid_map([[0, 0]])
    agent = _agent(agent_id=42, start_row=0, start_col=0, goal_row=0, goal_col=1)

    path = find_path(grid_map=grid_map, agent=agent, max_timestep=3)

    assert path is not None
    assert path.agent_id == 42


def test_start_equals_goal_returns_single_state_at_timestep_zero() -> None:
    grid_map = build_grid_map([[0]])
    agent = _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=0)

    path = find_path(grid_map=grid_map, agent=agent, max_timestep=0)

    assert path == AgentPath(
        agent_id=0,
        states=(TimedState(row=0, col=0, timestep=0),),
    )


def test_solver_routes_around_blocked_cell() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
    )
    agent = _agent(agent_id=0, start_row=0, start_col=0, goal_row=2, goal_col=0)

    path = find_path(grid_map=grid_map, agent=agent, max_timestep=10)

    assert path is not None
    occupied_cells = {(state.row, state.col) for state in path.states}
    assert (1, 1) not in occupied_cells
    assert path.states[-1] == TimedState(row=2, col=0, timestep=path.states[-1].timestep)


def test_impossible_map_returns_none() -> None:
    grid_map = build_grid_map([[0, 1, 0]])
    agent = _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=2)

    path = find_path(grid_map=grid_map, agent=agent, max_timestep=10)

    assert path is None


def test_generated_states_never_occupy_blocked_or_out_of_bounds_cells() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
    )
    agent = _agent(agent_id=0, start_row=0, start_col=0, goal_row=2, goal_col=2)

    path = find_path(grid_map=grid_map, agent=agent, max_timestep=10)

    assert path is not None
    for state in path.states:
        position = Position(row=state.row, col=state.col)
        assert grid_map.in_bounds(position)
        assert grid_map.is_walkable(position)


def test_diagonal_movement_is_not_used() -> None:
    grid_map = build_grid_map(
        [
            [0, 0],
            [0, 0],
        ]
    )
    agent = _agent(agent_id=0, start_row=0, start_col=0, goal_row=1, goal_col=1)

    path = find_path(grid_map=grid_map, agent=agent, max_timestep=10)

    assert path is not None
    assert len(path.states) == 3
    assert path.states[0] == TimedState(row=0, col=0, timestep=0)
    assert path.states[-1] == TimedState(row=1, col=1, timestep=2)


def test_all_spatial_moves_are_cardinal_or_wait() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
    )
    agent = _agent(agent_id=0, start_row=0, start_col=0, goal_row=2, goal_col=2)

    path = find_path(grid_map=grid_map, agent=agent, max_timestep=10)

    assert path is not None
    for previous, current in zip(path.states[:-1], path.states[1:], strict=True):
        row_diff = abs(current.row - previous.row)
        col_diff = abs(current.col - previous.col)
        assert row_diff + col_diff <= 1


def test_successor_model_includes_wait() -> None:
    grid_map = build_grid_map([[0]])
    current = TimedState(row=0, col=0, timestep=0)

    successors = _successors(grid_map=grid_map, state=current, max_timestep=3)

    assert TimedState(row=0, col=0, timestep=1) in successors


def test_sufficient_max_timestep_finds_path() -> None:
    grid_map = build_grid_map([[0, 0, 0, 0]])
    agent = _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=3)

    path = find_path(grid_map=grid_map, agent=agent, max_timestep=3)

    assert path is not None


def test_too_small_max_timestep_returns_none() -> None:
    grid_map = build_grid_map([[0, 0, 0, 0]])
    agent = _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=3)

    path = find_path(grid_map=grid_map, agent=agent, max_timestep=2)

    assert path is None


def test_max_timestep_zero_succeeds_only_when_start_equals_goal() -> None:
    grid_map = build_grid_map([[0, 0]])
    same_cell_agent = _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=0)
    different_goal_agent = _agent(agent_id=1, start_row=0, start_col=0, goal_row=0, goal_col=1)

    same_cell_path = find_path(grid_map=grid_map, agent=same_cell_agent, max_timestep=0)
    different_goal_path = find_path(grid_map=grid_map, agent=different_goal_agent, max_timestep=0)

    assert same_cell_path == AgentPath(
        agent_id=0,
        states=(TimedState(row=0, col=0, timestep=0),),
    )
    assert different_goal_path is None


def test_negative_max_timestep_raises_value_error() -> None:
    grid_map = build_grid_map([[0]])
    agent = _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=0)

    with pytest.raises(ValueError, match="max_timestep must be non-negative"):
        find_path(grid_map=grid_map, agent=agent, max_timestep=-1)
