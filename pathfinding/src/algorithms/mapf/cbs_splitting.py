from __future__ import annotations

from pathfinding.src.algorithms.mapf.models import (
    Conflict,
    Constraint,
    EdgeConflict,
    EdgeConstraint,
    VertexConflict,
    VertexConstraint,
)


def split_conflict(conflict: Conflict) -> tuple[Constraint, Constraint]:
    if isinstance(conflict, VertexConflict):
        return (
            VertexConstraint(
                agent_id=conflict.agent1_id,
                row=conflict.row,
                col=conflict.col,
                timestep=conflict.timestep,
            ),
            VertexConstraint(
                agent_id=conflict.agent2_id,
                row=conflict.row,
                col=conflict.col,
                timestep=conflict.timestep,
            ),
        )

    if isinstance(conflict, EdgeConflict):
        return (
            EdgeConstraint(
                agent_id=conflict.agent1_id,
                from_row=conflict.agent1_from_row,
                from_col=conflict.agent1_from_col,
                to_row=conflict.agent1_to_row,
                to_col=conflict.agent1_to_col,
                timestep=conflict.timestep,
            ),
            EdgeConstraint(
                agent_id=conflict.agent2_id,
                from_row=conflict.agent2_from_row,
                from_col=conflict.agent2_from_col,
                to_row=conflict.agent2_to_row,
                to_col=conflict.agent2_to_col,
                timestep=conflict.timestep,
            ),
        )

    raise TypeError(f"unsupported conflict type: {type(conflict)!r}")
