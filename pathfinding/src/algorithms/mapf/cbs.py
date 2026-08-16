from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass
from itertools import count

from pathfinding.src.algorithms.mapf.cbs_splitting import split_conflict
from pathfinding.src.algorithms.mapf.conflicts import detect_conflicts
from pathfinding.src.algorithms.mapf.metrics import sum_of_costs
from pathfinding.src.algorithms.mapf.models import (
    AgentPath,
    Conflict,
    Constraint,
    MAPFAgent,
    MAPFResult,
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


@dataclass(frozen=True, slots=True)
class CBSStats:
    expanded_ct_nodes: int
    generated_ct_nodes: int
    low_level_replans: int
    max_open_size: int
    unique_constraint_signatures: int
    duplicate_constraint_signatures: int
    unique_path_signatures: int
    duplicate_path_signatures: int
    generated_cost_distribution: tuple[tuple[int, int], ...]
    expanded_cost_distribution: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class CBSRunResult:
    result: MAPFResult | None
    stats: CBSStats
    termination_reason: str


def _constraint_signature(constraints: tuple[Constraint, ...]) -> frozenset[Constraint]:
    return frozenset(constraints)


def _path_signature(path: AgentPath) -> tuple[int, tuple[tuple[int, int, int], ...]]:
    return (
        path.agent_id,
        tuple((state.row, state.col, state.timestep) for state in path.states),
    )


def _path_set_signature(paths: tuple[AgentPath, ...]) -> tuple[tuple[int, tuple[tuple[int, int, int], ...]], ...]:
    return tuple(_path_signature(path) for path in paths)


def _cost_distribution(counter: Counter[int]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(counter.items()))


@dataclass
class _CBSRunTracker:
    expanded_ct_nodes: int = 0
    generated_ct_nodes: int = 0
    low_level_replans: int = 0
    max_open_size: int = 0
    _seen_constraint_signatures: set[frozenset[Constraint]] | None = None
    _seen_path_signatures: set[tuple[tuple[int, tuple[tuple[int, int, int], ...]], ...]] | None = None
    duplicate_constraint_signatures: int = 0
    duplicate_path_signatures: int = 0
    generated_cost_counts: Counter[int] | None = None
    expanded_cost_counts: Counter[int] | None = None

    def __post_init__(self) -> None:
        if self._seen_constraint_signatures is None:
            self._seen_constraint_signatures = set()
        if self._seen_path_signatures is None:
            self._seen_path_signatures = set()
        if self.generated_cost_counts is None:
            self.generated_cost_counts = Counter()
        if self.expanded_cost_counts is None:
            self.expanded_cost_counts = Counter()

    def record_generated_node(self, node: CBSNode) -> None:
        constraint_signature = _constraint_signature(node.constraints)
        path_signature = _path_set_signature(node.paths)

        if constraint_signature in self._seen_constraint_signatures:
            self.duplicate_constraint_signatures += 1
        else:
            self._seen_constraint_signatures.add(constraint_signature)

        if path_signature in self._seen_path_signatures:
            self.duplicate_path_signatures += 1
        else:
            self._seen_path_signatures.add(path_signature)

        self.generated_cost_counts[node.cost] += 1

    def record_expansion(self, node: CBSNode) -> None:
        self.expanded_cost_counts[node.cost] += 1

    def to_stats(self) -> CBSStats:
        return CBSStats(
            expanded_ct_nodes=self.expanded_ct_nodes,
            generated_ct_nodes=self.generated_ct_nodes,
            low_level_replans=self.low_level_replans,
            max_open_size=self.max_open_size,
            unique_constraint_signatures=len(self._seen_constraint_signatures),
            duplicate_constraint_signatures=self.duplicate_constraint_signatures,
            unique_path_signatures=len(self._seen_path_signatures),
            duplicate_path_signatures=self.duplicate_path_signatures,
            generated_cost_distribution=_cost_distribution(self.generated_cost_counts),
            expanded_cost_distribution=_cost_distribution(self.expanded_cost_counts),
        )


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


def _empty_stats() -> CBSStats:
    return CBSStats(
        expanded_ct_nodes=0,
        generated_ct_nodes=0,
        low_level_replans=0,
        max_open_size=0,
        unique_constraint_signatures=0,
        duplicate_constraint_signatures=0,
        unique_path_signatures=0,
        duplicate_path_signatures=0,
        generated_cost_distribution=(),
        expanded_cost_distribution=(),
    )


def _solve_cbs_internal(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
    *,
    max_expanded_nodes: int | None,
) -> CBSRunResult:
    if max_timestep < 0:
        raise ValueError("max_timestep must be non-negative")
    if max_expanded_nodes is not None and max_expanded_nodes < 0:
        raise ValueError("max_expanded_nodes must be non-negative")

    root = build_cbs_root(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
    )
    if root is None:
        return CBSRunResult(
            result=MAPFResult(success=False, paths=()),
            stats=_empty_stats(),
            termination_reason="failure",
        )

    tracker = _CBSRunTracker(
        generated_ct_nodes=1,
        max_open_size=1,
    )
    tracker.record_generated_node(root)

    open_heap: list[tuple[int, int, CBSNode]] = []
    insertion_counter = count()
    heapq.heappush(open_heap, (root.cost, next(insertion_counter), root))

    while open_heap:
        _, _, node = heapq.heappop(open_heap)

        if not node.conflicts:
            return CBSRunResult(
                result=MAPFResult(success=True, paths=node.paths),
                stats=tracker.to_stats(),
                termination_reason="success",
            )

        if (
            max_expanded_nodes is not None
            and tracker.expanded_ct_nodes >= max_expanded_nodes
        ):
            return CBSRunResult(
                result=None,
                stats=tracker.to_stats(),
                termination_reason="expansion_limit",
            )

        tracker.expanded_ct_nodes += 1
        tracker.low_level_replans += 2
        tracker.record_expansion(node)

        children = expand_cbs_node(
            grid_map=grid_map,
            scenario=scenario,
            node=node,
            max_timestep=max_timestep,
        )
        for child in children:
            tracker.generated_ct_nodes += 1
            tracker.record_generated_node(child)
            heapq.heappush(
                open_heap,
                (child.cost, next(insertion_counter), child),
            )
            if len(open_heap) > tracker.max_open_size:
                tracker.max_open_size = len(open_heap)

    return CBSRunResult(
        result=MAPFResult(success=False, paths=()),
        stats=tracker.to_stats(),
        termination_reason="failure",
    )


def solve_cbs(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
) -> MAPFResult:
    run = _solve_cbs_internal(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
        max_expanded_nodes=None,
    )
    assert run.result is not None
    return run.result


def solve_cbs_with_stats(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
    *,
    max_expanded_nodes: int | None = None,
) -> CBSRunResult:
    return _solve_cbs_internal(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
        max_expanded_nodes=max_expanded_nodes,
    )
