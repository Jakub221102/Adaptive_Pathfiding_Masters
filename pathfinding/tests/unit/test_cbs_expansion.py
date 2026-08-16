import pytest

from pathfinding.src.algorithms.mapf.cbs import (
    _find_agent,
    build_cbs_root,
    expand_cbs_node,
)
from pathfinding.src.algorithms.mapf.conflicts import detect_conflicts
from pathfinding.src.algorithms.mapf.metrics import sum_of_costs
from pathfinding.src.algorithms.mapf.models import (
    EdgeConflict,
    EdgeConstraint,
    MAPFAgent,
    MAPFScenario,
    VertexConflict,
    VertexConstraint,
)
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


def _vertex_conflict_root():
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
    assert len(root.conflicts) > 0
    assert isinstance(root.conflicts[0], VertexConflict)
    return grid_map, scenario, root


def test_vertex_conflict_expansion_produces_two_children() -> None:
    grid_map, scenario, root = _vertex_conflict_root()

    children = expand_cbs_node(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        max_timestep=5,
    )

    assert len(children) == 2
    assert len(children[0].constraints) == len(root.constraints) + 1
    assert len(children[1].constraints) == len(root.constraints) + 1


def test_vertex_split_ownership_follows_conflict_order() -> None:
    grid_map, scenario, root = _vertex_conflict_root()
    conflict = root.conflicts[0]

    children = expand_cbs_node(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        max_timestep=5,
    )

    assert children[0].constraints[-1].agent_id == conflict.agent1_id
    assert children[1].constraints[-1].agent_id == conflict.agent2_id
    assert isinstance(children[0].constraints[-1], VertexConstraint)
    assert isinstance(children[1].constraints[-1], VertexConstraint)


def test_only_affected_agent_path_is_replaced() -> None:
    grid_map, scenario, root = _vertex_conflict_root()
    conflict = root.conflicts[0]

    children = expand_cbs_node(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        max_timestep=5,
    )

    agent1_index = next(
        index for index, path in enumerate(root.paths) if path.agent_id == conflict.agent1_id
    )
    agent2_index = next(
        index for index, path in enumerate(root.paths) if path.agent_id == conflict.agent2_id
    )

    child_for_agent1, child_for_agent2 = children

    assert child_for_agent1.paths[agent1_index] != root.paths[agent1_index]
    assert child_for_agent1.paths[agent2_index] is root.paths[agent2_index]

    assert child_for_agent2.paths[agent2_index] != root.paths[agent2_index]
    assert child_for_agent2.paths[agent1_index] is root.paths[agent1_index]


def test_parent_node_remains_unchanged_after_expansion() -> None:
    grid_map, scenario, root = _vertex_conflict_root()

    original_constraints = root.constraints
    original_paths = root.paths
    original_cost = root.cost
    original_conflicts = root.conflicts

    expand_cbs_node(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        max_timestep=5,
    )

    assert root.constraints == original_constraints
    assert root.paths == original_paths
    assert root.cost == original_cost
    assert root.conflicts == original_conflicts


def test_child_constraints_inherit_parent_constraints() -> None:
    grid_map, scenario, root = _vertex_conflict_root()

    first_children = expand_cbs_node(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        max_timestep=5,
    )
    parent_with_constraint = first_children[1]
    parent_constraint = parent_with_constraint.constraints[0]

    second_children = expand_cbs_node(
        grid_map=grid_map,
        scenario=scenario,
        node=parent_with_constraint,
        max_timestep=5,
    )

    assert len(second_children) > 0
    for child in second_children:
        assert parent_constraint in child.constraints
        assert len(child.constraints) == len(parent_with_constraint.constraints) + 1


def test_replanning_uses_agent_id_not_tuple_index() -> None:
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
    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=5)
    assert root is not None

    children = expand_cbs_node(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        max_timestep=5,
    )

    assert len(children) == 2
    assert children[0].constraints[-1].agent_id == root.conflicts[0].agent1_id == 7
    assert children[1].constraints[-1].agent_id == root.conflicts[0].agent2_id == 2
    assert children[0].paths[0].agent_id == 7
    assert children[1].paths[1].agent_id == 2


def test_child_path_order_matches_scenario() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=7, start_row=0, start_col=0, goal_row=0, goal_col=2),
        _agent(agent_id=2, start_row=1, start_col=1, goal_row=0, goal_col=0),
        _agent(agent_id=5, start_row=2, start_col=0, goal_row=2, goal_col=2),
    )
    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=8)
    assert root is not None
    assert root.conflicts

    children = expand_cbs_node(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        max_timestep=8,
    )

    for child in children:
        assert tuple(path.agent_id for path in child.paths) == (7, 2, 5)


def test_child_cost_equals_sum_of_costs() -> None:
    grid_map, scenario, root = _vertex_conflict_root()

    children = expand_cbs_node(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        max_timestep=5,
    )

    for child in children:
        assert child.cost == sum_of_costs(child.paths)


def test_child_conflicts_are_fully_recomputed() -> None:
    grid_map, scenario, root = _vertex_conflict_root()

    children = expand_cbs_node(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        max_timestep=5,
    )

    for child in children:
        assert child.conflicts == detect_conflicts(child.paths)


def test_edge_conflict_expansion_uses_own_movement_constraints() -> None:
    grid_map = build_grid_map([[0, 0]])
    scenario = _scenario(
        _agent(agent_id=3, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=8, start_row=0, start_col=1, goal_row=0, goal_col=0),
    )
    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=3)
    assert root is not None
    conflict = root.conflicts[0]
    assert isinstance(conflict, EdgeConflict)

    children = expand_cbs_node(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        max_timestep=3,
    )

    assert len(children) == 2
    first_constraint = children[0].constraints[-1]
    second_constraint = children[1].constraints[-1]
    assert isinstance(first_constraint, EdgeConstraint)
    assert isinstance(second_constraint, EdgeConstraint)

    assert first_constraint == EdgeConstraint(
        agent_id=3,
        from_row=0,
        from_col=0,
        to_row=0,
        to_col=1,
        timestep=1,
    )
    assert second_constraint == EdgeConstraint(
        agent_id=8,
        from_row=0,
        from_col=1,
        to_row=0,
        to_col=0,
        timestep=1,
    )

    forbidden_first = (
        first_constraint.from_row,
        first_constraint.from_col,
        first_constraint.to_row,
        first_constraint.to_col,
        first_constraint.timestep,
    )
    forbidden_second = (
        second_constraint.from_row,
        second_constraint.from_col,
        second_constraint.to_row,
        second_constraint.to_col,
        second_constraint.timestep,
    )

    for child, forbidden in zip(children, (forbidden_first, forbidden_second), strict=True):
        agent_id = child.constraints[-1].agent_id
        replanned_path = next(path for path in child.paths if path.agent_id == agent_id)
        for index in range(len(replanned_path.states) - 1):
            current = replanned_path.states[index]
            successor = replanned_path.states[index + 1]
            transition = (
                current.row,
                current.col,
                successor.row,
                successor.col,
                successor.timestep,
            )
            assert transition != forbidden


def test_one_infeasible_branch_returns_single_child() -> None:
    grid_map, scenario, root = _vertex_conflict_root()
    conflict = root.conflicts[0]

    children = expand_cbs_node(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        max_timestep=2,
    )

    assert len(children) == 1
    assert children[0].constraints[-1].agent_id == conflict.agent1_id


def test_both_infeasible_branches_return_empty_tuple() -> None:
    grid_map = build_grid_map([[0, 0]])
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=1, start_row=0, start_col=1, goal_row=0, goal_col=0),
    )
    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=1)
    assert root is not None
    assert root.conflicts

    children = expand_cbs_node(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        max_timestep=1,
    )

    assert children == ()


def test_conflict_free_node_returns_empty_tuple() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=3),
        _agent(agent_id=1, start_row=1, start_col=0, goal_row=1, goal_col=3),
    )
    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=6)
    assert root is not None
    assert root.conflicts == ()

    children = expand_cbs_node(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        max_timestep=6,
    )

    assert children == ()


def test_negative_max_timestep_raises_value_error() -> None:
    grid_map, scenario, root = _vertex_conflict_root()

    with pytest.raises(ValueError, match="max_timestep must be non-negative"):
        expand_cbs_node(
            grid_map=grid_map,
            scenario=scenario,
            node=root,
            max_timestep=-1,
        )


def test_find_agent_raises_for_missing_agent_id() -> None:
    scenario = _scenario(_agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1))

    with pytest.raises(ValueError, match="agent_id 99 not found in scenario"):
        _find_agent(scenario, 99)


def test_end_to_end_vertex_expansion_properties() -> None:
    grid_map, scenario, root = _vertex_conflict_root()
    selected_conflict = root.conflicts[0]

    children = expand_cbs_node(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        max_timestep=5,
    )

    assert len(children) == 2
    for child in children:
        assert len(child.constraints) == 1
        assert child.cost == sum_of_costs(child.paths)
        assert child.conflicts == detect_conflicts(child.paths)
        assert selected_conflict not in child.conflicts

    assert children[0].conflicts == ()
    assert len(children[1].conflicts) >= 0
