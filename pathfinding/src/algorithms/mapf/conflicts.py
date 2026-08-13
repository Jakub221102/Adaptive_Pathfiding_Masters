from __future__ import annotations

from collections.abc import Sequence

from pathfinding.src.algorithms.mapf.models import (
    AgentPath,
    Conflict,
    EdgeConflict,
    VertexConflict,
)


def _position_at(path: AgentPath, timestep: int) -> tuple[int, int] | None:
    first_timestep = path.states[0].timestep
    if timestep < first_timestep:
        return None

    final_state = path.states[-1]
    if timestep >= final_state.timestep:
        return final_state.row, final_state.col

    index = timestep - first_timestep
    state = path.states[index]
    return state.row, state.col


def detect_conflicts(paths: Sequence[AgentPath]) -> tuple[Conflict, ...]:
    path_list = list(paths)

    if len(path_list) <= 1:
        return ()

    agent_ids = [path.agent_id for path in path_list]
    if len(set(agent_ids)) != len(agent_ids):
        raise ValueError("agent_id must be unique among supplied paths")

    horizon = max(path.states[-1].timestep for path in path_list)
    conflicts: list[Conflict] = []

    for timestep in range(horizon + 1):
        for first_index in range(len(path_list)):
            for second_index in range(first_index + 1, len(path_list)):
                first_path = path_list[first_index]
                second_path = path_list[second_index]
                agent1_id = first_path.agent_id
                agent2_id = second_path.agent_id

                first_position = _position_at(first_path, timestep)
                second_position = _position_at(second_path, timestep)

                if (
                    first_position is not None
                    and second_position is not None
                    and first_position == second_position
                ):
                    conflicts.append(
                        VertexConflict(
                            agent1_id=agent1_id,
                            agent2_id=agent2_id,
                            row=first_position[0],
                            col=first_position[1],
                            timestep=timestep,
                        )
                    )

                if timestep > 0:
                    first_from = _position_at(first_path, timestep - 1)
                    first_to = _position_at(first_path, timestep)
                    second_from = _position_at(second_path, timestep - 1)
                    second_to = _position_at(second_path, timestep)

                    if (
                        first_from is not None
                        and first_to is not None
                        and second_from is not None
                        and second_to is not None
                        and first_from == second_to
                        and first_to == second_from
                        and first_from != first_to
                    ):
                        conflicts.append(
                            EdgeConflict(
                                agent1_id=agent1_id,
                                agent2_id=agent2_id,
                                agent1_from_row=first_from[0],
                                agent1_from_col=first_from[1],
                                agent1_to_row=first_to[0],
                                agent1_to_col=first_to[1],
                                agent2_from_row=second_from[0],
                                agent2_from_col=second_from[1],
                                agent2_to_row=second_to[0],
                                agent2_to_col=second_to[1],
                                timestep=timestep,
                            )
                        )

    return tuple(conflicts)
