from __future__ import annotations

from collections.abc import Sequence

from pathfinding.src.algorithms.mapf.models import (
    AgentPath,
    Constraint,
    EdgeConstraint,
    VertexConstraint,
)


def _append_unique(
    constraints: list[Constraint],
    seen: set[Constraint],
    constraint: Constraint,
) -> None:
    if constraint in seen:
        return

    seen.add(constraint)
    constraints.append(constraint)


def build_reservation_constraints(
    planned_paths: Sequence[AgentPath],
    target_agent_id: int,
    max_timestep: int,
) -> tuple[Constraint, ...]:
    if max_timestep < 0:
        raise ValueError("max_timestep must be non-negative")

    path_list = list(planned_paths)
    if not path_list:
        return ()

    agent_ids = [path.agent_id for path in path_list]
    if len(set(agent_ids)) != len(agent_ids):
        raise ValueError("agent_id must be unique among supplied paths")

    if target_agent_id in agent_ids:
        raise ValueError("target agent must not appear in planned paths")

    constraints: list[Constraint] = []
    seen: set[Constraint] = set()

    for path in path_list:
        for index, state in enumerate(path.states):
            if state.timestep > max_timestep:
                continue

            _append_unique(
                constraints,
                seen,
                VertexConstraint(
                    agent_id=target_agent_id,
                    row=state.row,
                    col=state.col,
                    timestep=state.timestep,
                ),
            )

            if index == 0:
                continue

            previous = path.states[index - 1]
            if previous.row == state.row and previous.col == state.col:
                continue

            _append_unique(
                constraints,
                seen,
                EdgeConstraint(
                    agent_id=target_agent_id,
                    from_row=state.row,
                    from_col=state.col,
                    to_row=previous.row,
                    to_col=previous.col,
                    timestep=state.timestep,
                ),
            )

        final_state = path.states[-1]
        for timestep in range(final_state.timestep + 1, max_timestep + 1):
            _append_unique(
                constraints,
                seen,
                VertexConstraint(
                    agent_id=target_agent_id,
                    row=final_state.row,
                    col=final_state.col,
                    timestep=timestep,
                ),
            )

    return tuple(constraints)
