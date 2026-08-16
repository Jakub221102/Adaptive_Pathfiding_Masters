from __future__ import annotations

from enum import Enum

from pathfinding.src.algorithms.mapf.cbs import CBSNode
from pathfinding.src.algorithms.mapf.cbs_splitting import split_conflict
from pathfinding.src.algorithms.mapf.models import (
    AgentPath,
    Conflict,
    Constraint,
    MAPFAgent,
    MAPFScenario,
)
from pathfinding.src.algorithms.mapf.space_time_astar import find_path
from pathfinding.src.core.models import GridMap


class ConflictCardinality(Enum):
    CARDINAL = "cardinal"
    SEMI_CARDINAL = "semi_cardinal"
    NON_CARDINAL = "non_cardinal"


def _path_cost(path: AgentPath) -> int:
    return len(path.states) - 1


def _find_agent(scenario: MAPFScenario, agent_id: int) -> MAPFAgent:
    for agent in scenario.agents:
        if agent.agent_id == agent_id:
            return agent

    raise ValueError(f"agent_id {agent_id} not found in scenario")


def _find_agent_path(node: CBSNode, agent_id: int) -> AgentPath:
    for path in node.paths:
        if path.agent_id == agent_id:
            return path

    raise ValueError(f"agent_id {agent_id} not found in node paths")


def _branch_forces_cost_increase(
    grid_map: GridMap,
    scenario: MAPFScenario,
    node: CBSNode,
    new_constraint: Constraint,
    max_timestep: int,
) -> bool:
    affected_agent = _find_agent(scenario, new_constraint.agent_id)
    parent_path = _find_agent_path(node, affected_agent.agent_id)
    old_cost = _path_cost(parent_path)

    branch_constraints = node.constraints + (new_constraint,)
    replanned_path = find_path(
        grid_map=grid_map,
        agent=affected_agent,
        max_timestep=max_timestep,
        constraints=branch_constraints,
    )
    if replanned_path is None:
        return True

    return _path_cost(replanned_path) > old_cost


def classify_conflict(
    grid_map: GridMap,
    scenario: MAPFScenario,
    node: CBSNode,
    conflict: Conflict,
    max_timestep: int,
) -> ConflictCardinality:
    if max_timestep < 0:
        raise ValueError("max_timestep must be non-negative")

    _find_agent(scenario, conflict.agent1_id)
    _find_agent(scenario, conflict.agent2_id)
    _find_agent_path(node, conflict.agent1_id)
    _find_agent_path(node, conflict.agent2_id)

    first_constraint, second_constraint = split_conflict(conflict)

    first_increases = _branch_forces_cost_increase(
        grid_map=grid_map,
        scenario=scenario,
        node=node,
        new_constraint=first_constraint,
        max_timestep=max_timestep,
    )
    second_increases = _branch_forces_cost_increase(
        grid_map=grid_map,
        scenario=scenario,
        node=node,
        new_constraint=second_constraint,
        max_timestep=max_timestep,
    )

    if first_increases and second_increases:
        return ConflictCardinality.CARDINAL
    if first_increases or second_increases:
        return ConflictCardinality.SEMI_CARDINAL
    return ConflictCardinality.NON_CARDINAL
