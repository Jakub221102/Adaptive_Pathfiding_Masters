import pytest

from pathfinding.src.algorithms.mapf.cbs import build_cbs_root, expand_cbs_node
from pathfinding.src.algorithms.mapf.cbs_conflict_classification import (
    ConflictCardinality,
    classify_conflict,
)
from pathfinding.src.algorithms.mapf.cbs_splitting import split_conflict
from pathfinding.src.algorithms.mapf.models import (
    EdgeConflict,
    MAPFAgent,
    MAPFScenario,
    VertexConflict,
    VertexConstraint,
)
from pathfinding.src.algorithms.mapf.space_time_astar import find_path
from pathfinding.src.core.models import Position
from pathfinding.tests.helpers import build_grid_map


def _agent(
    agent_id: int,
    start_row: int,
    start_col: int,
    goal_row: int,
    goal_col: int,
) -> MAPFAgent:
    return MAPFAgent(
        agent_id=agent_id,
        start=Position(row=start_row, col=start_col),
        goal=Position(row=goal_row, col=goal_col),
    )


def _scenario(*agents: MAPFAgent) -> MAPFScenario:
    return MAPFScenario(agents=agents)


def _path_cost(path) -> int:
    return len(path.states) - 1


def _replanned_cost(
    grid_map,
    scenario,
    node,
    constraint,
    max_timestep: int,
) -> int | None:
    agent = next(agent for agent in scenario.agents if agent.agent_id == constraint.agent_id)
    path = find_path(
        grid_map=grid_map,
        agent=agent,
        max_timestep=max_timestep,
        constraints=node.constraints + (constraint,),
    )
    if path is None:
        return None
    return _path_cost(path)


def test_non_cardinal_vertex_conflict_keeps_both_agent_costs() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=2, goal_col=2),
        _agent(agent_id=1, start_row=2, start_col=0, goal_row=0, goal_col=2),
    )
    max_timestep = 8
    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=max_timestep)
    assert root is not None
    conflict = root.conflicts[0]
    assert isinstance(conflict, VertexConflict)

    cardinality = classify_conflict(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        conflict=conflict,
        max_timestep=max_timestep,
    )
    first_constraint, second_constraint = split_conflict(conflict)

    first_old = _path_cost(next(path for path in root.paths if path.agent_id == conflict.agent1_id))
    second_old = _path_cost(next(path for path in root.paths if path.agent_id == conflict.agent2_id))
    first_new = _replanned_cost(grid_map, scenario, root, first_constraint, max_timestep)
    second_new = _replanned_cost(grid_map, scenario, root, second_constraint, max_timestep)

    assert cardinality == ConflictCardinality.NON_CARDINAL
    assert first_new == first_old
    assert second_new == second_old


def test_non_cardinal_edge_conflict_keeps_both_agent_costs() -> None:
    grid_map = build_grid_map(
        [
            [0, 0],
            [0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=1, goal_col=1),
        _agent(agent_id=1, start_row=1, start_col=0, goal_row=0, goal_col=1),
    )
    max_timestep = 5
    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=max_timestep)
    assert root is not None
    conflict = root.conflicts[0]
    assert isinstance(conflict, EdgeConflict)

    assert classify_conflict(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        conflict=conflict,
        max_timestep=max_timestep,
    ) == ConflictCardinality.NON_CARDINAL


def test_semi_cardinal_vertex_conflict() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=1, start_col=1, goal_row=0, goal_col=0),
        _agent(agent_id=1, start_row=0, start_col=0, goal_row=0, goal_col=2),
    )
    max_timestep = 5
    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=max_timestep)
    assert root is not None
    conflict = root.conflicts[0]

    cardinality = classify_conflict(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        conflict=conflict,
        max_timestep=max_timestep,
    )
    first_constraint, second_constraint = split_conflict(conflict)
    first_old = _path_cost(next(path for path in root.paths if path.agent_id == conflict.agent1_id))
    second_old = _path_cost(next(path for path in root.paths if path.agent_id == conflict.agent2_id))
    first_new = _replanned_cost(grid_map, scenario, root, first_constraint, max_timestep)
    second_new = _replanned_cost(grid_map, scenario, root, second_constraint, max_timestep)

    assert cardinality == ConflictCardinality.SEMI_CARDINAL
    assert first_new == first_old
    assert second_new == second_old + 1


def test_semi_cardinal_with_non_index_agent_ids() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=7, start_row=1, start_col=1, goal_row=0, goal_col=0),
        _agent(agent_id=2, start_row=0, start_col=0, goal_row=0, goal_col=2),
    )
    max_timestep = 5
    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=max_timestep)
    assert root is not None

    assert classify_conflict(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        conflict=root.conflicts[0],
        max_timestep=max_timestep,
    ) == ConflictCardinality.SEMI_CARDINAL


def test_cardinal_edge_conflict_forces_both_agents_to_increase_cost() -> None:
    grid_map = build_grid_map(
        [
            [0, 0],
            [0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=1, start_row=0, start_col=1, goal_row=0, goal_col=0),
    )
    max_timestep = 5
    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=max_timestep)
    assert root is not None
    conflict = root.conflicts[0]

    cardinality = classify_conflict(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        conflict=conflict,
        max_timestep=max_timestep,
    )
    first_constraint, second_constraint = split_conflict(conflict)
    first_old = _path_cost(next(path for path in root.paths if path.agent_id == conflict.agent1_id))
    second_old = _path_cost(next(path for path in root.paths if path.agent_id == conflict.agent2_id))

    assert cardinality == ConflictCardinality.CARDINAL
    assert _replanned_cost(grid_map, scenario, root, first_constraint, max_timestep) == first_old + 1
    assert _replanned_cost(grid_map, scenario, root, second_constraint, max_timestep) == second_old + 1


def test_infeasible_branch_counts_as_cost_increasing() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=1, start_col=1, goal_row=0, goal_col=0),
        _agent(agent_id=1, start_row=0, start_col=0, goal_row=0, goal_col=2),
    )
    max_timestep = 2
    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=max_timestep)
    assert root is not None
    conflict = root.conflicts[0]
    first_constraint, second_constraint = split_conflict(conflict)

    assert _replanned_cost(grid_map, scenario, root, first_constraint, max_timestep) == 2
    assert _replanned_cost(grid_map, scenario, root, second_constraint, max_timestep) is None
    assert classify_conflict(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        conflict=conflict,
        max_timestep=max_timestep,
    ) == ConflictCardinality.SEMI_CARDINAL


def test_both_infeasible_branches_are_cardinal() -> None:
    grid_map = build_grid_map([[0, 0]])
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=1, start_row=0, start_col=1, goal_row=0, goal_col=0),
    )
    max_timestep = 1
    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=max_timestep)
    assert root is not None

    assert classify_conflict(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        conflict=root.conflicts[0],
        max_timestep=max_timestep,
    ) == ConflictCardinality.CARDINAL


def test_existing_parent_constraints_are_used_during_classification() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=1, start_col=1, goal_row=0, goal_col=0),
        _agent(agent_id=1, start_row=0, start_col=0, goal_row=0, goal_col=2),
    )
    max_timestep = 5
    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=max_timestep)
    assert root is not None
    children = expand_cbs_node(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        max_timestep=max_timestep,
    )
    child = children[1]
    conflict = child.conflicts[0]
    first_constraint, second_constraint = split_conflict(conflict)

    with_parent = classify_conflict(
        grid_map=grid_map,
        scenario=scenario,
        node=child,
        conflict=conflict,
        max_timestep=max_timestep,
    )
    without_parent = classify_conflict(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        conflict=conflict,
        max_timestep=max_timestep,
    )

    assert with_parent == ConflictCardinality.SEMI_CARDINAL
    assert without_parent == ConflictCardinality.NON_CARDINAL
    assert child.constraints != root.constraints
    assert _replanned_cost(grid_map, scenario, child, second_constraint, max_timestep) == 4
    assert _replanned_cost(grid_map, scenario, root, second_constraint, max_timestep) == 2


def test_parent_node_remains_unchanged() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=1, start_col=1, goal_row=0, goal_col=0),
        _agent(agent_id=1, start_row=0, start_col=0, goal_row=0, goal_col=2),
    )
    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=5)
    assert root is not None

    original_constraints = root.constraints
    original_paths = root.paths
    original_cost = root.cost
    original_conflicts = root.conflicts

    classify_conflict(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        conflict=root.conflicts[0],
        max_timestep=5,
    )

    assert root.constraints == original_constraints
    assert root.paths == original_paths
    assert root.cost == original_cost
    assert root.conflicts == original_conflicts


def test_classification_is_deterministic() -> None:
    grid_map = build_grid_map(
        [
            [0, 0],
            [0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=1, start_row=0, start_col=1, goal_row=0, goal_col=0),
    )
    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=5)
    assert root is not None
    conflict = root.conflicts[0]

    first = classify_conflict(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        conflict=conflict,
        max_timestep=5,
    )
    second = classify_conflict(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        conflict=conflict,
        max_timestep=5,
    )

    assert first == second


def test_missing_agent_in_scenario_raises_value_error() -> None:
    grid_map = build_grid_map([[0, 0, 0], [0, 0, 0]])
    scenario = _scenario(
        _agent(agent_id=0, start_row=1, start_col=1, goal_row=0, goal_col=0),
    )
    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=5)
    assert root is not None
    conflict = VertexConflict(
        agent1_id=0,
        agent2_id=99,
        row=0,
        col=1,
        timestep=1,
    )

    with pytest.raises(ValueError, match="agent_id 99 not found in scenario"):
        classify_conflict(
            grid_map=grid_map,
            scenario=scenario,
            node=root,
            conflict=conflict,
            max_timestep=5,
        )


def test_negative_horizon_raises_value_error() -> None:
    grid_map = build_grid_map([[0, 0, 0], [0, 0, 0]])
    scenario = _scenario(
        _agent(agent_id=0, start_row=1, start_col=1, goal_row=0, goal_col=0),
        _agent(agent_id=1, start_row=0, start_col=0, goal_row=0, goal_col=2),
    )
    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=5)
    assert root is not None

    with pytest.raises(ValueError, match="max_timestep must be non-negative"):
        classify_conflict(
            grid_map=grid_map,
            scenario=scenario,
            node=root,
            conflict=root.conflicts[0],
            max_timestep=-1,
        )
