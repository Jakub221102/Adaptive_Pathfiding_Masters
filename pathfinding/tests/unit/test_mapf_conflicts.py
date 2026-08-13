import pytest

from pathfinding.src.algorithms.mapf.conflicts import detect_conflicts
from pathfinding.src.algorithms.mapf.models import (
    AgentPath,
    EdgeConflict,
    TimedState,
    VertexConflict,
)


def _path(agent_id: int, coordinates: list[tuple[int, int, int]]) -> AgentPath:
    return AgentPath(
        agent_id=agent_id,
        states=tuple(TimedState(row=row, col=col, timestep=timestep) for row, col, timestep in coordinates),
    )


def test_empty_collection_returns_no_conflicts() -> None:
    assert detect_conflicts(()) == ()


def test_one_path_returns_no_conflicts() -> None:
    path = _path(0, [(0, 0, 0), (0, 1, 1)])

    assert detect_conflicts((path,)) == ()


def test_two_non_conflicting_paths_return_no_conflicts() -> None:
    path_a = _path(0, [(0, 0, 0), (0, 1, 1), (0, 2, 2)])
    path_b = _path(1, [(2, 0, 0), (2, 1, 1), (2, 2, 2)])

    assert detect_conflicts((path_a, path_b)) == ()


def test_parallel_same_direction_movement_is_not_edge_conflict() -> None:
    path_a = _path(0, [(0, 0, 0), (0, 1, 1), (0, 2, 2)])
    path_b = _path(1, [(2, 0, 0), (2, 1, 1), (2, 2, 2)])

    conflicts = detect_conflicts((path_a, path_b))

    assert not any(isinstance(conflict, EdgeConflict) for conflict in conflicts)


def test_same_cell_at_same_timestep_produces_vertex_conflict() -> None:
    path_a = _path(0, [(4, 5, 0), (4, 5, 1), (4, 5, 2)])
    path_b = _path(1, [(4, 6, 0), (4, 6, 1), (4, 5, 2)])

    conflicts = detect_conflicts((path_a, path_b))

    assert conflicts == (
        VertexConflict(agent1_id=0, agent2_id=1, row=4, col=5, timestep=2),
    )


def test_same_coordinate_at_different_timesteps_is_not_conflict() -> None:
    path_a = _path(0, [(1, 1, 0), (1, 2, 1), (1, 3, 2)])
    path_b = _path(1, [(2, 2, 0), (2, 1, 1), (1, 2, 2)])

    assert detect_conflicts((path_a, path_b)) == ()


def test_wait_can_participate_in_vertex_conflict() -> None:
    path_a = _path(0, [(2, 2, 0), (2, 2, 1), (2, 2, 2)])
    path_b = _path(1, [(2, 3, 0), (2, 2, 1), (2, 1, 2)])

    conflicts = detect_conflicts((path_a, path_b))

    assert VertexConflict(agent1_id=0, agent2_id=1, row=2, col=2, timestep=1) in conflicts


def test_stay_at_goal_detects_vertex_conflict_after_path_ends() -> None:
    path_a = _path(0, [(1, 1, 0), (1, 2, 1)])
    path_b = _path(1, [(3, 3, 0), (3, 2, 1), (2, 2, 2), (1, 2, 3)])

    conflicts = detect_conflicts((path_a, path_b))

    assert conflicts == (
        VertexConflict(agent1_id=0, agent2_id=1, row=1, col=2, timestep=3),
    )


def test_direct_swap_produces_edge_conflict() -> None:
    path_a = _path(0, [(4, 5, 3), (4, 6, 4)])
    path_b = _path(1, [(4, 6, 3), (4, 5, 4)])

    conflicts = detect_conflicts((path_a, path_b))

    assert conflicts == (
        EdgeConflict(
            agent1_id=0,
            agent2_id=1,
            agent1_from_row=4,
            agent1_from_col=5,
            agent1_to_row=4,
            agent1_to_col=6,
            agent2_from_row=4,
            agent2_from_col=6,
            agent2_to_row=4,
            agent2_to_col=5,
            timestep=4,
        ),
    )


def test_edge_conflict_uses_arrival_timestep() -> None:
    path_a = _path(0, [(4, 5, 3), (4, 6, 4)])
    path_b = _path(1, [(4, 6, 3), (4, 5, 4)])

    conflicts = detect_conflicts((path_a, path_b))

    assert conflicts == (
        EdgeConflict(
            agent1_id=0,
            agent2_id=1,
            agent1_from_row=4,
            agent1_from_col=5,
            agent1_to_row=4,
            agent1_to_col=6,
            agent2_from_row=4,
            agent2_from_col=6,
            agent2_to_row=4,
            agent2_to_col=5,
            timestep=4,
        ),
    )


def test_edge_conflict_preserves_movement_geometry() -> None:
    path_a = _path(7, [(1, 1, 0), (1, 2, 1)])
    path_b = _path(2, [(1, 2, 0), (1, 1, 1)])

    conflicts = detect_conflicts((path_a, path_b))

    assert conflicts == (
        EdgeConflict(
            agent1_id=7,
            agent2_id=2,
            agent1_from_row=1,
            agent1_from_col=1,
            agent1_to_row=1,
            agent1_to_col=2,
            agent2_from_row=1,
            agent2_from_col=2,
            agent2_to_row=1,
            agent2_to_col=1,
            timestep=1,
        ),
    )


def test_same_direction_movement_does_not_produce_edge_conflict() -> None:
    path_a = _path(0, [(0, 0, 0), (0, 1, 1)])
    path_b = _path(1, [(1, 0, 0), (1, 1, 1)])

    conflicts = detect_conflicts((path_a, path_b))

    assert not any(isinstance(conflict, EdgeConflict) for conflict in conflicts)


def test_wait_versus_movement_is_not_edge_conflict() -> None:
    path_a = _path(0, [(0, 0, 0), (0, 0, 1)])
    path_b = _path(1, [(0, 1, 0), (0, 0, 1)])

    conflicts = detect_conflicts((path_a, path_b))

    assert not any(isinstance(conflict, EdgeConflict) for conflict in conflicts)
    assert conflicts == (
        VertexConflict(agent1_id=0, agent2_id=1, row=0, col=0, timestep=1),
    )


def test_paths_of_different_lengths_are_handled() -> None:
    path_a = _path(0, [(1, 1, 0), (1, 2, 1)])
    path_b = _path(1, [(3, 3, 0), (3, 2, 1), (2, 2, 2), (1, 2, 3), (0, 2, 4)])

    conflicts = detect_conflicts((path_a, path_b))

    assert VertexConflict(agent1_id=0, agent2_id=1, row=1, col=2, timestep=3) in conflicts
    assert all(conflict.timestep <= 4 for conflict in conflicts)


def test_path_is_not_backward_padded_before_first_timestep() -> None:
    path_a = _path(0, [(5, 6, 3), (5, 7, 4)])
    path_b = _path(1, [(5, 5, 0), (5, 5, 1), (5, 5, 2)])

    conflicts = detect_conflicts((path_a, path_b))

    assert conflicts == ()


def test_stay_at_goal_padding_does_not_mutate_agent_path() -> None:
    path_a = _path(0, [(1, 1, 0), (1, 2, 1)])
    path_b = _path(1, [(3, 3, 0), (3, 2, 1), (2, 2, 2), (1, 2, 3)])
    original_a_states = path_a.states
    original_b_states = path_b.states

    detect_conflicts((path_a, path_b))

    assert path_a.states == original_a_states
    assert path_b.states == original_b_states
    assert len(path_a.states) == 2
    assert len(path_b.states) == 4


def test_duplicate_agent_id_paths_are_rejected() -> None:
    path_a = _path(0, [(0, 0, 0), (0, 1, 1)])
    path_b = _path(0, [(1, 1, 0), (1, 2, 1)])

    with pytest.raises(ValueError, match="agent_id must be unique among supplied paths"):
        detect_conflicts((path_a, path_b))


def test_multiple_conflicts_are_returned_in_timestep_first_order() -> None:
    path_a = _path(7, [(0, 0, 0), (0, 1, 1), (0, 0, 2), (0, 1, 3)])
    path_b = _path(2, [(0, 1, 0), (0, 0, 1), (0, 1, 2), (0, 0, 3)])
    path_c = _path(9, [(2, 2, 0), (2, 2, 1), (2, 2, 2)])

    conflicts = detect_conflicts((path_a, path_b, path_c))

    timesteps = [conflict.timestep for conflict in conflicts]
    assert timesteps == sorted(timesteps)


def test_later_pair_conflict_does_not_precede_earlier_timestep_conflict() -> None:
    path_a = _path(1, [(0, 0, 0), (0, 1, 1), (0, 2, 2), (1, 1, 3), (1, 1, 4), (1, 1, 5)])
    path_b = _path(2, [(2, 1, 0), (2, 1, 1), (2, 1, 2), (2, 1, 3), (2, 1, 4), (1, 1, 5)])
    path_c = _path(3, [(0, 3, 0), (0, 2, 1), (0, 2, 2)])

    conflicts = detect_conflicts((path_a, path_b, path_c))

    assert conflicts[0] == VertexConflict(agent1_id=1, agent2_id=3, row=0, col=2, timestep=2)
    assert conflicts[1] == VertexConflict(agent1_id=1, agent2_id=2, row=1, col=1, timestep=5)


def test_same_timestep_conflicts_follow_input_pair_order() -> None:
    path_a = _path(1, [(0, 0, 0), (1, 1, 1), (2, 2, 2), (5, 5, 3)])
    path_b = _path(2, [(0, 1, 0), (1, 2, 1), (2, 3, 2), (5, 5, 3)])
    path_c = _path(3, [(0, 2, 0), (1, 3, 1), (2, 4, 2), (5, 5, 3)])

    conflicts = detect_conflicts((path_a, path_b, path_c))

    assert conflicts == (
        VertexConflict(agent1_id=1, agent2_id=2, row=5, col=5, timestep=3),
        VertexConflict(agent1_id=1, agent2_id=3, row=5, col=5, timestep=3),
        VertexConflict(agent1_id=2, agent2_id=3, row=5, col=5, timestep=3),
    )


def test_agent_pair_orientation_follows_input_path_order() -> None:
    path_first = _path(2, [(0, 0, 0), (0, 1, 1)])
    path_second = _path(7, [(0, 1, 0), (0, 0, 1)])

    conflicts = detect_conflicts((path_first, path_second))

    assert conflicts[0].agent1_id == 2
    assert conflicts[0].agent2_id == 7


def test_converging_movement_is_vertex_conflict_not_edge_conflict() -> None:
    path_a = _path(0, [(0, 0, 0), (0, 1, 1)])
    path_b = _path(1, [(0, 2, 0), (0, 1, 1)])

    conflicts = detect_conflicts((path_a, path_b))

    assert conflicts == (
        VertexConflict(agent1_id=0, agent2_id=1, row=0, col=1, timestep=1),
    )
