from __future__ import annotations

from collections.abc import Sequence

from pathfinding.src.algorithms.mapf.models import AgentPath


def count_wait_steps(path: AgentPath) -> int:
    wait_steps = 0

    for index in range(1, len(path.states)):
        previous = path.states[index - 1]
        current = path.states[index]
        if previous.row == current.row and previous.col == current.col:
            wait_steps += 1

    return wait_steps


def count_total_wait_steps(paths: Sequence[AgentPath]) -> int:
    return sum(count_wait_steps(path) for path in paths)


def count_agents_with_wait(paths: Sequence[AgentPath]) -> int:
    return sum(1 for path in paths if count_wait_steps(path) > 0)


def paths_have_identical_states(
    left: AgentPath,
    right: AgentPath,
) -> bool:
    if len(left.states) != len(right.states):
        return False

    for left_state, right_state in zip(left.states, right.states, strict=True):
        if (
            left_state.row != right_state.row
            or left_state.col != right_state.col
            or left_state.timestep != right_state.timestep
        ):
            return False

    return True


def count_agents_with_changed_path(
    independent_paths: Sequence[AgentPath],
    coordinated_paths: Sequence[AgentPath],
) -> int:
    independent_by_agent = {path.agent_id: path for path in independent_paths}
    changed_agents = 0

    for coordinated_path in coordinated_paths:
        independent_path = independent_by_agent[coordinated_path.agent_id]
        if not paths_have_identical_states(independent_path, coordinated_path):
            changed_agents += 1

    return changed_agents
