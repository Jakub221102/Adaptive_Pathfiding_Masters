from __future__ import annotations

from collections.abc import Sequence

from pathfinding.src.algorithms.mapf.models import AgentPath


def _path_cost(path: AgentPath) -> int:
    return len(path.states) - 1


def sum_of_costs(paths: Sequence[AgentPath]) -> int:
    return sum(_path_cost(path) for path in paths)


def makespan(paths: Sequence[AgentPath]) -> int:
    if not paths:
        return 0

    return max(_path_cost(path) for path in paths)
