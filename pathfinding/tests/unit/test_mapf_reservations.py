import pytest

from pathfinding.src.algorithms.mapf.conflicts import detect_conflicts
from pathfinding.src.algorithms.mapf.models import (
    AgentPath,
    Constraint,
    EdgeConstraint,
    MAPFAgent,
    TimedState,
    VertexConstraint,
)
from pathfinding.src.algorithms.mapf.reservations import build_reservation_constraints
from pathfinding.src.algorithms.mapf.space_time_astar import find_path
from pathfinding.src.core.models import Position
from pathfinding.tests.helpers import build_grid_map


def _path(agent_id: int, coordinates: list[tuple[int, int, int]]) -> AgentPath:
    return AgentPath(
        agent_id=agent_id,
        states=tuple(
            TimedState(row=row, col=col, timestep=timestep)
            for row, col, timestep in coordinates
        ),
    )


def _assert_path_respects_constraints(
    path: AgentPath,
    agent_id: int,
    constraints: tuple[Constraint, ...],
) -> None:
    vertex_constraints = {
        (constraint.row, constraint.col, constraint.timestep)
        for constraint in constraints
        if isinstance(constraint, VertexConstraint) and constraint.agent_id == agent_id
    }
    edge_constraints = {
        (
            constraint.from_row,
            constraint.from_col,
            constraint.to_row,
            constraint.to_col,
            constraint.timestep,
        )
        for constraint in constraints
        if isinstance(constraint, EdgeConstraint) and constraint.agent_id == agent_id
    }

    for state in path.states:
        assert (state.row, state.col, state.timestep) not in vertex_constraints

    for previous, current in zip(path.states[:-1], path.states[1:], strict=True):
        transition = (
            previous.row,
            previous.col,
            current.row,
            current.col,
            current.timestep,
        )
        assert transition not in edge_constraints


def test_explicit_states_generate_matching_vertex_constraints() -> None:
    planned = _path(0, [(1, 1, 0), (1, 2, 1), (1, 3, 2)])

    constraints = build_reservation_constraints(
        planned_paths=(planned,),
        target_agent_id=5,
        max_timestep=5,
    )

    vertex_constraints = [
        constraint
        for constraint in constraints
        if isinstance(constraint, VertexConstraint)
    ]

    assert vertex_constraints[:3] == [
        VertexConstraint(agent_id=5, row=1, col=1, timestep=0),
        VertexConstraint(agent_id=5, row=1, col=2, timestep=1),
        VertexConstraint(agent_id=5, row=1, col=3, timestep=2),
    ]


def test_generated_constraints_use_target_agent_id() -> None:
    planned = _path(7, [(0, 0, 0), (0, 1, 1)])

    constraints = build_reservation_constraints(
        planned_paths=(planned,),
        target_agent_id=3,
        max_timestep=2,
    )

    assert all(constraint.agent_id == 3 for constraint in constraints)


def test_non_zero_start_timestep_is_preserved() -> None:
    planned = _path(0, [(2, 2, 3), (2, 3, 4)])

    constraints = build_reservation_constraints(
        planned_paths=(planned,),
        target_agent_id=1,
        max_timestep=6,
    )

    assert VertexConstraint(agent_id=1, row=2, col=2, timestep=3) in constraints
    assert VertexConstraint(agent_id=1, row=2, col=3, timestep=4) in constraints


def test_explicit_states_beyond_max_timestep_generate_no_reservation() -> None:
    planned = _path(0, [(0, 0, 0), (0, 1, 1), (0, 2, 2), (0, 3, 3)])

    constraints = build_reservation_constraints(
        planned_paths=(planned,),
        target_agent_id=1,
        max_timestep=1,
    )

    vertex_timesteps = {
        constraint.timestep
        for constraint in constraints
        if isinstance(constraint, VertexConstraint)
    }
    assert vertex_timesteps == {0, 1}


def test_movement_generates_reverse_edge_constraint() -> None:
    planned = _path(0, [(2, 3, 4), (2, 4, 5)])

    constraints = build_reservation_constraints(
        planned_paths=(planned,),
        target_agent_id=1,
        max_timestep=5,
    )

    assert EdgeConstraint(
        agent_id=1,
        from_row=2,
        from_col=4,
        to_row=2,
        to_col=3,
        timestep=5,
    ) in constraints


def test_edge_constraint_uses_arrival_timestep() -> None:
    planned = _path(0, [(4, 5, 3), (4, 6, 4)])

    constraints = build_reservation_constraints(
        planned_paths=(planned,),
        target_agent_id=2,
        max_timestep=4,
    )

    edge_constraints = [
        constraint for constraint in constraints if isinstance(constraint, EdgeConstraint)
    ]

    assert edge_constraints == [
        EdgeConstraint(
            agent_id=2,
            from_row=4,
            from_col=6,
            to_row=4,
            to_col=5,
            timestep=4,
        )
    ]


def test_reverse_direction_is_not_normalized() -> None:
    planned = _path(0, [(1, 1, 0), (1, 2, 1)])

    constraints = build_reservation_constraints(
        planned_paths=(planned,),
        target_agent_id=1,
        max_timestep=2,
    )

    assert EdgeConstraint(
        agent_id=1,
        from_row=1,
        from_col=2,
        to_row=1,
        to_col=1,
        timestep=1,
    ) in constraints
    assert EdgeConstraint(
        agent_id=1,
        from_row=1,
        from_col=1,
        to_row=1,
        to_col=2,
        timestep=1,
    ) not in constraints


def test_wait_transition_does_not_generate_edge_constraint() -> None:
    planned = _path(0, [(2, 2, 0), (2, 2, 1), (2, 3, 2)])

    constraints = build_reservation_constraints(
        planned_paths=(planned,),
        target_agent_id=1,
        max_timestep=3,
    )

    edge_constraints = [
        constraint for constraint in constraints if isinstance(constraint, EdgeConstraint)
    ]

    assert len(edge_constraints) == 1
    assert edge_constraints[0] == EdgeConstraint(
        agent_id=1,
        from_row=2,
        from_col=3,
        to_row=2,
        to_col=2,
        timestep=2,
    )


def test_final_position_is_reserved_through_max_timestep() -> None:
    planned = _path(0, [(4, 7, 2), (4, 8, 3), (4, 9, 4), (4, 9, 5)])

    constraints = build_reservation_constraints(
        planned_paths=(planned,),
        target_agent_id=1,
        max_timestep=9,
    )

    assert [
        VertexConstraint(agent_id=1, row=4, col=9, timestep=timestep)
        for timestep in range(6, 10)
    ] == [
        constraint
        for constraint in constraints
        if isinstance(constraint, VertexConstraint)
        and constraint.row == 4
        and constraint.col == 9
        and constraint.timestep >= 6
    ]


def test_logical_goal_extension_does_not_mutate_original_agent_path() -> None:
    planned = _path(0, [(1, 1, 0), (1, 2, 1)])
    original_states = planned.states

    build_reservation_constraints(
        planned_paths=(planned,),
        target_agent_id=2,
        max_timestep=4,
    )

    assert planned.states == original_states
    assert len(planned.states) == 2


def test_no_reservations_are_generated_beyond_horizon() -> None:
    planned = _path(0, [(0, 0, 0), (0, 1, 1)])

    constraints = build_reservation_constraints(
        planned_paths=(planned,),
        target_agent_id=1,
        max_timestep=3,
    )

    assert all(constraint.timestep <= 3 for constraint in constraints)


def test_path_ending_at_max_timestep_has_no_extra_goal_padding() -> None:
    planned = _path(0, [(3, 3, 0), (3, 4, 1), (3, 4, 2)])

    constraints = build_reservation_constraints(
        planned_paths=(planned,),
        target_agent_id=1,
        max_timestep=2,
    )

    goal_padding = [
        constraint
        for constraint in constraints
        if isinstance(constraint, VertexConstraint)
        and constraint.row == 3
        and constraint.col == 4
        and constraint.timestep > 2
    ]

    assert goal_padding == []


def test_constraints_are_generated_for_multiple_planned_paths() -> None:
    first = _path(0, [(0, 0, 0), (0, 1, 1)])
    second = _path(1, [(1, 0, 0), (1, 1, 1)])

    constraints = build_reservation_constraints(
        planned_paths=(first, second),
        target_agent_id=2,
        max_timestep=2,
    )

    assert VertexConstraint(agent_id=2, row=0, col=0, timestep=0) in constraints
    assert VertexConstraint(agent_id=2, row=1, col=1, timestep=1) in constraints


def test_output_ordering_follows_input_path_and_timestep_order() -> None:
    first = _path(0, [(0, 0, 0), (0, 1, 1)])
    second = _path(1, [(1, 0, 0), (1, 1, 1)])

    constraints = build_reservation_constraints(
        planned_paths=(first, second),
        target_agent_id=2,
        max_timestep=2,
    )

    assert constraints.index(VertexConstraint(agent_id=2, row=0, col=0, timestep=0)) < constraints.index(
        VertexConstraint(agent_id=2, row=1, col=0, timestep=0)
    )
    assert constraints.index(VertexConstraint(agent_id=2, row=0, col=1, timestep=1)) < constraints.index(
        VertexConstraint(agent_id=2, row=1, col=1, timestep=1)
    )


def test_duplicate_planned_agent_ids_are_rejected() -> None:
    first = _path(0, [(0, 0, 0)])
    second = _path(0, [(1, 1, 0)])

    with pytest.raises(ValueError, match="agent_id must be unique among supplied paths"):
        build_reservation_constraints(
            planned_paths=(first, second),
            target_agent_id=1,
            max_timestep=2,
        )


def test_planned_path_for_target_agent_is_rejected() -> None:
    planned = _path(2, [(0, 0, 0)])

    with pytest.raises(ValueError, match="target agent must not appear in planned paths"):
        build_reservation_constraints(
            planned_paths=(planned,),
            target_agent_id=2,
            max_timestep=2,
        )


def test_empty_planned_paths_returns_empty_tuple() -> None:
    assert build_reservation_constraints((), target_agent_id=0, max_timestep=3) == ()


def test_negative_max_timestep_raises_value_error() -> None:
    planned = _path(0, [(0, 0, 0)])

    with pytest.raises(ValueError, match="max_timestep must be non-negative"):
        build_reservation_constraints(
            planned_paths=(planned,),
            target_agent_id=1,
            max_timestep=-1,
        )


def test_one_state_path_generates_vertex_and_goal_padding_only() -> None:
    planned = _path(0, [(5, 5, 1)])

    constraints = build_reservation_constraints(
        planned_paths=(planned,),
        target_agent_id=3,
        max_timestep=3,
    )

    assert constraints == (
        VertexConstraint(agent_id=3, row=5, col=5, timestep=1),
        VertexConstraint(agent_id=3, row=5, col=5, timestep=2),
        VertexConstraint(agent_id=3, row=5, col=5, timestep=3),
    )


def test_generated_reservations_force_target_agent_to_wait() -> None:
    grid_map = build_grid_map([[0, 0, 0]])
    higher_priority = _path(0, [(0, 1, 1)])
    target_agent = MAPFAgent(
        agent_id=1,
        start=Position(row=0, col=0),
        goal=Position(row=0, col=2),
    )
    search_horizon = 4

    reservations = build_reservation_constraints(
        planned_paths=(higher_priority,),
        target_agent_id=target_agent.agent_id,
        max_timestep=1,
    )

    unconstrained_path = find_path(
        grid_map=grid_map,
        agent=target_agent,
        max_timestep=search_horizon,
    )
    constrained_path = find_path(
        grid_map=grid_map,
        agent=target_agent,
        max_timestep=search_horizon,
        constraints=reservations,
    )

    assert unconstrained_path is not None
    assert unconstrained_path.states == (
        TimedState(row=0, col=0, timestep=0),
        TimedState(row=0, col=1, timestep=1),
        TimedState(row=0, col=2, timestep=2),
    )
    assert detect_conflicts((higher_priority, unconstrained_path)) != ()

    assert constrained_path is not None
    assert constrained_path.states == (
        TimedState(row=0, col=0, timestep=0),
        TimedState(row=0, col=0, timestep=1),
        TimedState(row=0, col=1, timestep=2),
        TimedState(row=0, col=2, timestep=3),
    )
    _assert_path_respects_constraints(
        constrained_path,
        agent_id=target_agent.agent_id,
        constraints=reservations,
    )
