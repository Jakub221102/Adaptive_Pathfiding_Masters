from __future__ import annotations

from dataclasses import dataclass

from pathfinding.src.algorithms.mapf.cbs_splitting import split_conflict
from pathfinding.src.algorithms.mapf.conflicts import detect_conflicts
from pathfinding.src.algorithms.mapf.metrics import sum_of_costs
from pathfinding.src.algorithms.mapf.models import (
    AgentPath,
    Conflict,
    Constraint,
    MAPFAgent,
    MAPFScenario,
)
from pathfinding.src.algorithms.mapf.space_time_astar import find_path
from pathfinding.src.core.models import GridMap


@dataclass(frozen=True, slots=True)
class CBSNode:
    constraints: tuple[Constraint, ...]
    paths: tuple[AgentPath, ...]
    cost: int
    conflicts: tuple[Conflict, ...]


def _find_agent(scenario: MAPFScenario, agent_id: int) -> MAPFAgent:
    for agent in scenario.agents:
        if agent.agent_id == agent_id:
            return agent

    raise ValueError(f"agent_id {agent_id} not found in scenario")


def _replace_agent_path(
    paths: tuple[AgentPath, ...],
    agent_id: int,
    new_path: AgentPath,
) -> tuple[AgentPath, ...]:
    return tuple(
        new_path if path.agent_id == agent_id else path
        for path in paths
    )


def _try_expand_branch(
    grid_map: GridMap,
    scenario: MAPFScenario,
    node: CBSNode,
    new_constraint: Constraint,
    max_timestep: int,
) -> CBSNode | None:
    child_constraints = node.constraints + (new_constraint,)
    affected_agent = _find_agent(scenario, new_constraint.agent_id)
    new_path = find_path(
        grid_map=grid_map,
        agent=affected_agent,
        max_timestep=max_timestep,
        constraints=child_constraints,
    )
    if new_path is None:
        return None

    child_paths = _replace_agent_path(
        node.paths,
        affected_agent.agent_id,
        new_path,
    )
    return CBSNode(
        constraints=child_constraints,
        paths=child_paths,
        cost=sum_of_costs(child_paths),
        conflicts=detect_conflicts(child_paths),
    )


def build_cbs_root(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
) -> CBSNode | None:
    if max_timestep < 0:
        raise ValueError("max_timestep must be non-negative")

    paths: list[AgentPath] = []

    for agent in scenario.agents:
        path = find_path(
            grid_map=grid_map,
            agent=agent,
            max_timestep=max_timestep,
            constraints=(),
        )
        if path is None:
            return None

        paths.append(path)

    path_tuple = tuple(paths)
    return CBSNode(
        constraints=(),
        paths=path_tuple,
        cost=sum_of_costs(path_tuple),
        conflicts=detect_conflicts(path_tuple),
    )


def expand_cbs_node(
    grid_map: GridMap,
    scenario: MAPFScenario,
    node: CBSNode,
    max_timestep: int,
) -> tuple[CBSNode, ...]:
    if max_timestep < 0:
        raise ValueError("max_timestep must be non-negative")

    if not node.conflicts:
        return ()

    conflict = node.conflicts[0]
    branch_constraints = split_conflict(conflict)

    children: list[CBSNode] = []
    for new_constraint in branch_constraints:
        child = _try_expand_branch(
            grid_map=grid_map,
            scenario=scenario,
            node=node,
            new_constraint=new_constraint,
            max_timestep=max_timestep,
        )
        if child is not None:
            children.append(child)

    return tuple(children)
