import pytest

from pathfinding.src.algorithms.mapf.cbs_splitting import split_conflict
from pathfinding.src.algorithms.mapf.models import (
    EdgeConflict,
    EdgeConstraint,
    VertexConflict,
    VertexConstraint,
)


def _vertex_conflict(
    agent1_id: int = 4,
    agent2_id: int = 9,
    row: int = 2,
    col: int = 3,
    timestep: int = 7,
) -> VertexConflict:
    return VertexConflict(
        agent1_id=agent1_id,
        agent2_id=agent2_id,
        row=row,
        col=col,
        timestep=timestep,
    )


def _edge_conflict(
    agent1_id: int = 3,
    agent2_id: int = 8,
    timestep: int = 5,
) -> EdgeConflict:
    return EdgeConflict(
        agent1_id=agent1_id,
        agent2_id=agent2_id,
        agent1_from_row=1,
        agent1_from_col=2,
        agent1_to_row=1,
        agent1_to_col=3,
        agent2_from_row=1,
        agent2_from_col=3,
        agent2_to_row=1,
        agent2_to_col=2,
        timestep=timestep,
    )


def test_vertex_split_assigns_constraints_to_correct_agents() -> None:
    first, second = split_conflict(_vertex_conflict(agent1_id=4, agent2_id=9))

    assert first.agent_id == 4
    assert second.agent_id == 9


def test_vertex_split_preserves_position() -> None:
    first, second = split_conflict(_vertex_conflict(row=5, col=6))

    assert first.row == 5
    assert first.col == 6
    assert second.row == 5
    assert second.col == 6


def test_vertex_split_preserves_timestep() -> None:
    first, second = split_conflict(_vertex_conflict(timestep=11))

    assert first.timestep == 11
    assert second.timestep == 11


def test_vertex_split_returns_vertex_constraints() -> None:
    first, second = split_conflict(_vertex_conflict())

    assert isinstance(first, VertexConstraint)
    assert isinstance(second, VertexConstraint)


def test_vertex_split_preserves_agent_ordering() -> None:
    first, second = split_conflict(_vertex_conflict(agent1_id=9, agent2_id=2))

    assert first.agent_id == 9
    assert second.agent_id == 2


def test_edge_split_returns_edge_constraints() -> None:
    first, second = split_conflict(_edge_conflict())

    assert isinstance(first, EdgeConstraint)
    assert isinstance(second, EdgeConstraint)


def test_edge_split_preserves_agent1_movement() -> None:
    first, _ = split_conflict(_edge_conflict(agent1_id=3, timestep=5))

    assert first == EdgeConstraint(
        agent_id=3,
        from_row=1,
        from_col=2,
        to_row=1,
        to_col=3,
        timestep=5,
    )


def test_edge_split_preserves_agent2_movement() -> None:
    _, second = split_conflict(_edge_conflict(agent2_id=8, timestep=5))

    assert second == EdgeConstraint(
        agent_id=8,
        from_row=1,
        from_col=3,
        to_row=1,
        to_col=2,
        timestep=5,
    )


def test_edge_split_does_not_reverse_edges() -> None:
    first, second = split_conflict(_edge_conflict())

    assert not (
        first.from_row == 1
        and first.from_col == 3
        and first.to_row == 1
        and first.to_col == 2
    )
    assert not (
        second.from_row == 1
        and second.from_col == 2
        and second.to_row == 1
        and second.to_col == 3
    )


def test_edge_split_preserves_arrival_timestep() -> None:
    conflict = _edge_conflict(timestep=5)
    first, second = split_conflict(conflict)

    assert conflict.timestep == 5
    assert first.timestep == 5
    assert second.timestep == 5


def test_edge_split_preserves_agent_ordering() -> None:
    first, second = split_conflict(_edge_conflict(agent1_id=9, agent2_id=2))

    assert first.agent_id == 9
    assert second.agent_id == 2


def test_split_conflict_rejects_unsupported_type() -> None:
    with pytest.raises(TypeError, match="unsupported conflict type"):
        split_conflict(object())  # type: ignore[arg-type]
