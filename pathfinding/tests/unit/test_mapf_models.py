import pytest

from pathfinding.src.algorithms.mapf.models import (
    AgentPath,
    EdgeConflict,
    EdgeConstraint,
    MAPFAgent,
    MAPFResult,
    MAPFScenario,
    TimedState,
    VertexConflict,
    VertexConstraint,
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


def test_vertex_conflict_is_created_with_valid_values() -> None:
    conflict = VertexConflict(agent1_id=0, agent2_id=1, row=2, col=3, timestep=4)

    assert conflict.agent1_id == 0
    assert conflict.agent2_id == 1
    assert conflict.row == 2
    assert conflict.col == 3
    assert conflict.timestep == 4


def test_vertex_conflict_rejects_same_agent() -> None:
    with pytest.raises(ValueError, match="agent1_id and agent2_id must differ"):
        VertexConflict(agent1_id=0, agent2_id=0, row=0, col=0, timestep=0)


@pytest.mark.parametrize(
    ("row", "col", "timestep", "expected_message"),
    [
        (-1, 0, 0, "row must be non-negative"),
        (0, -1, 0, "col must be non-negative"),
        (0, 0, -1, "timestep must be non-negative"),
    ],
)
def test_vertex_conflict_rejects_negative_values(
    row: int,
    col: int,
    timestep: int,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        VertexConflict(agent1_id=0, agent2_id=1, row=row, col=col, timestep=timestep)


def test_equal_vertex_conflicts_compare_equal_and_hash_equally() -> None:
    first = VertexConflict(agent1_id=0, agent2_id=1, row=2, col=3, timestep=4)
    second = VertexConflict(agent1_id=0, agent2_id=1, row=2, col=3, timestep=4)

    assert first == second
    assert hash(first) == hash(second)


def test_vertex_conflict_can_be_used_in_a_set() -> None:
    conflict = VertexConflict(agent1_id=0, agent2_id=1, row=2, col=3, timestep=4)

    conflicts = {conflict, VertexConflict(agent1_id=0, agent2_id=1, row=2, col=3, timestep=4)}

    assert len(conflicts) == 1
    assert conflict in conflicts


def test_edge_conflict_represents_swap_style_conflict() -> None:
    conflict = EdgeConflict(
        agent1_id=0,
        agent2_id=1,
        agent1_from_row=2,
        agent1_from_col=3,
        agent1_to_row=2,
        agent1_to_col=4,
        agent2_from_row=2,
        agent2_from_col=4,
        agent2_to_row=2,
        agent2_to_col=3,
        timestep=5,
    )

    assert conflict.agent1_id == 0
    assert conflict.agent2_id == 1
    assert conflict.timestep == 5


def test_edge_conflict_rejects_same_agent() -> None:
    with pytest.raises(ValueError, match="agent1_id and agent2_id must differ"):
        EdgeConflict(
            agent1_id=0,
            agent2_id=0,
            agent1_from_row=0,
            agent1_from_col=0,
            agent1_to_row=0,
            agent1_to_col=1,
            agent2_from_row=0,
            agent2_from_col=1,
            agent2_to_row=0,
            agent2_to_col=0,
            timestep=1,
        )


def test_edge_conflict_rejects_timestep_zero() -> None:
    with pytest.raises(ValueError, match="timestep must be positive"):
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
            timestep=0,
        )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("agent1_from_row", -1),
        ("agent1_from_col", -1),
        ("agent1_to_row", -1),
        ("agent1_to_col", -1),
        ("agent2_from_row", -1),
        ("agent2_from_col", -1),
        ("agent2_to_row", -1),
        ("agent2_to_col", -1),
    ],
)
def test_edge_conflict_rejects_negative_coordinates(field_name: str, field_value: int) -> None:
    values = {
        "agent1_id": 0,
        "agent2_id": 1,
        "agent1_from_row": 0,
        "agent1_from_col": 0,
        "agent1_to_row": 0,
        "agent1_to_col": 1,
        "agent2_from_row": 0,
        "agent2_from_col": 1,
        "agent2_to_row": 0,
        "agent2_to_col": 0,
        "timestep": 1,
    }
    values[field_name] = field_value

    with pytest.raises(ValueError, match=f"{field_name} must be non-negative"):
        EdgeConflict(**values)


def test_equal_edge_conflicts_compare_equal_and_hash_equally() -> None:
    first = EdgeConflict(
        agent1_id=0,
        agent2_id=1,
        agent1_from_row=2,
        agent1_from_col=3,
        agent1_to_row=2,
        agent1_to_col=4,
        agent2_from_row=2,
        agent2_from_col=4,
        agent2_to_row=2,
        agent2_to_col=3,
        timestep=5,
    )
    second = EdgeConflict(
        agent1_id=0,
        agent2_id=1,
        agent1_from_row=2,
        agent1_from_col=3,
        agent1_to_row=2,
        agent1_to_col=4,
        agent2_from_row=2,
        agent2_from_col=4,
        agent2_to_row=2,
        agent2_to_col=3,
        timestep=5,
    )

    assert first == second
    assert hash(first) == hash(second)


def test_vertex_constraint_is_created_with_valid_values() -> None:
    constraint = VertexConstraint(agent_id=0, row=2, col=3, timestep=4)

    assert constraint.agent_id == 0
    assert constraint.row == 2
    assert constraint.col == 3
    assert constraint.timestep == 4


def test_equal_vertex_constraints_compare_equal_and_hash_equally() -> None:
    first = VertexConstraint(agent_id=0, row=2, col=3, timestep=4)
    second = VertexConstraint(agent_id=0, row=2, col=3, timestep=4)

    assert first == second
    assert hash(first) == hash(second)


def test_different_timestep_produces_different_vertex_constraint() -> None:
    earlier = VertexConstraint(agent_id=0, row=2, col=3, timestep=4)
    later = VertexConstraint(agent_id=0, row=2, col=3, timestep=5)

    assert earlier != later
    assert len({earlier, later}) == 2


@pytest.mark.parametrize(
    ("row", "col", "timestep", "expected_message"),
    [
        (-1, 0, 0, "row must be non-negative"),
        (0, -1, 0, "col must be non-negative"),
        (0, 0, -1, "timestep must be non-negative"),
    ],
)
def test_vertex_constraint_rejects_negative_values(
    row: int,
    col: int,
    timestep: int,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        VertexConstraint(agent_id=0, row=row, col=col, timestep=timestep)


def test_vertex_constraint_can_be_used_in_set_and_dict() -> None:
    constraint = VertexConstraint(agent_id=0, row=2, col=3, timestep=4)

    constraints = {constraint, VertexConstraint(agent_id=0, row=2, col=3, timestep=4)}
    lookup = {constraint: "blocked"}

    assert len(constraints) == 1
    assert lookup[constraint] == "blocked"


def test_edge_constraint_is_created_with_valid_values() -> None:
    constraint = EdgeConstraint(
        agent_id=0,
        from_row=2,
        from_col=3,
        to_row=2,
        to_col=4,
        timestep=5,
    )

    assert constraint.agent_id == 0
    assert constraint.from_row == 2
    assert constraint.from_col == 3
    assert constraint.to_row == 2
    assert constraint.to_col == 4
    assert constraint.timestep == 5


def test_equal_edge_constraints_compare_equal_and_hash_equally() -> None:
    first = EdgeConstraint(
        agent_id=0,
        from_row=2,
        from_col=3,
        to_row=2,
        to_col=4,
        timestep=5,
    )
    second = EdgeConstraint(
        agent_id=0,
        from_row=2,
        from_col=3,
        to_row=2,
        to_col=4,
        timestep=5,
    )

    assert first == second
    assert hash(first) == hash(second)


def test_reversed_edge_constraint_is_not_equal() -> None:
    forward = EdgeConstraint(
        agent_id=0,
        from_row=2,
        from_col=3,
        to_row=2,
        to_col=4,
        timestep=5,
    )
    reverse = EdgeConstraint(
        agent_id=0,
        from_row=2,
        from_col=4,
        to_row=2,
        to_col=3,
        timestep=5,
    )

    assert forward != reverse
    assert len({forward, reverse}) == 2


def test_edge_constraint_rejects_timestep_zero() -> None:
    with pytest.raises(ValueError, match="timestep must be positive"):
        EdgeConstraint(
            agent_id=0,
            from_row=0,
            from_col=0,
            to_row=0,
            to_col=1,
            timestep=0,
        )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("from_row", -1),
        ("from_col", -1),
        ("to_row", -1),
        ("to_col", -1),
    ],
)
def test_edge_constraint_rejects_negative_coordinates(field_name: str, field_value: int) -> None:
    values = {
        "agent_id": 0,
        "from_row": 0,
        "from_col": 0,
        "to_row": 0,
        "to_col": 1,
        "timestep": 1,
    }
    values[field_name] = field_value

    with pytest.raises(ValueError, match=f"{field_name} must be non-negative"):
        EdgeConstraint(**values)


def test_edge_constraint_can_be_used_in_set_and_dict() -> None:
    constraint = EdgeConstraint(
        agent_id=0,
        from_row=2,
        from_col=3,
        to_row=2,
        to_col=4,
        timestep=5,
    )

    constraints = {
        constraint,
        EdgeConstraint(
            agent_id=0,
            from_row=2,
            from_col=3,
            to_row=2,
            to_col=4,
            timestep=5,
        ),
    }
    lookup = {constraint: "forbidden"}

    assert len(constraints) == 1
    assert lookup[constraint] == "forbidden"
