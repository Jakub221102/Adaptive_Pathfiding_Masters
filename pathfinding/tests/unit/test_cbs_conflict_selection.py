from unittest.mock import patch

import pytest

from pathfinding.src.algorithms.mapf.cbs import (
    CBSNode,
    build_cbs_root,
    expand_cbs_node,
    expand_cbs_node_for_conflict,
    solve_cbs,
    solve_cbs_cardinal_first,
    solve_cbs_cardinal_first_with_stats,
    solve_cbs_with_stats,
)
from pathfinding.src.algorithms.mapf.cbs_conflict_classification import ConflictCardinality
from pathfinding.src.algorithms.mapf.cbs_conflict_selection import (
    CBSConflictSelection,
    select_conflict_cardinal_first,
)
from pathfinding.src.algorithms.mapf.cbs_splitting import split_conflict
from pathfinding.src.algorithms.mapf.conflicts import detect_conflicts
from pathfinding.src.algorithms.mapf.metrics import sum_of_costs
from pathfinding.src.algorithms.mapf.models import (
    EdgeConflict,
    MAPFAgent,
    MAPFScenario,
    VertexConflict,
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


def _conflict(agent1_id: int, agent2_id: int, timestep: int) -> VertexConflict:
    return VertexConflict(
        agent1_id=agent1_id,
        agent2_id=agent2_id,
        row=0,
        col=0,
        timestep=timestep,
    )


def _node_with_conflicts(*conflicts: VertexConflict) -> CBSNode:
    return CBSNode(
        constraints=(),
        paths=(),
        cost=0,
        conflicts=conflicts,
    )


def _swap_root():
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
    return grid_map, scenario, root


@patch("pathfinding.src.algorithms.mapf.cbs_conflict_selection.classify_conflict")
def test_first_cardinal_conflict_stops_immediately(mock_classify) -> None:
    conflicts = (
        _conflict(agent1_id=0, agent2_id=1, timestep=1),
        _conflict(agent1_id=0, agent2_id=1, timestep=2),
        _conflict(agent1_id=0, agent2_id=1, timestep=3),
    )
    mock_classify.return_value = ConflictCardinality.CARDINAL
    grid_map = build_grid_map([[0, 0], [0, 0]])
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=1, start_row=0, start_col=1, goal_row=0, goal_col=0),
    )

    selection = select_conflict_cardinal_first(
        grid_map=grid_map,
        scenario=scenario,
        node=_node_with_conflicts(*conflicts),
        max_timestep=5,
    )

    assert selection == CBSConflictSelection(
        conflict=conflicts[0],
        cardinality=ConflictCardinality.CARDINAL,
        classified_conflicts=1,
        classification_low_level_searches=2,
    )
    assert mock_classify.call_count == 1


@patch("pathfinding.src.algorithms.mapf.cbs_conflict_selection.classify_conflict")
def test_non_cardinal_then_cardinal_selects_first_cardinal(mock_classify) -> None:
    conflicts = (
        _conflict(agent1_id=0, agent2_id=1, timestep=1),
        _conflict(agent1_id=0, agent2_id=1, timestep=2),
    )
    mock_classify.side_effect = [
        ConflictCardinality.NON_CARDINAL,
        ConflictCardinality.CARDINAL,
    ]
    grid_map = build_grid_map([[0, 0], [0, 0]])
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=1, start_row=0, start_col=1, goal_row=0, goal_col=0),
    )

    selection = select_conflict_cardinal_first(
        grid_map=grid_map,
        scenario=scenario,
        node=_node_with_conflicts(*conflicts),
        max_timestep=5,
    )

    assert selection is not None
    assert selection.conflict == conflicts[1]
    assert selection.cardinality == ConflictCardinality.CARDINAL
    assert selection.classified_conflicts == 2
    assert selection.classification_low_level_searches == 4


@patch("pathfinding.src.algorithms.mapf.cbs_conflict_selection.classify_conflict")
def test_semi_then_cardinal_selects_cardinal_not_earlier_semi(mock_classify) -> None:
    conflicts = (
        _conflict(agent1_id=0, agent2_id=1, timestep=1),
        _conflict(agent1_id=0, agent2_id=1, timestep=2),
        _conflict(agent1_id=0, agent2_id=1, timestep=3),
        _conflict(agent1_id=0, agent2_id=1, timestep=4),
    )
    mock_classify.side_effect = [
        ConflictCardinality.NON_CARDINAL,
        ConflictCardinality.SEMI_CARDINAL,
        ConflictCardinality.NON_CARDINAL,
        ConflictCardinality.CARDINAL,
    ]
    grid_map = build_grid_map([[0, 0], [0, 0]])
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=1, start_row=0, start_col=1, goal_row=0, goal_col=0),
    )

    selection = select_conflict_cardinal_first(
        grid_map=grid_map,
        scenario=scenario,
        node=_node_with_conflicts(*conflicts),
        max_timestep=5,
    )

    assert selection is not None
    assert selection.conflict == conflicts[3]
    assert selection.classified_conflicts == 4


@patch("pathfinding.src.algorithms.mapf.cbs_conflict_selection.classify_conflict")
def test_multiple_semi_conflicts_select_first_semi(mock_classify) -> None:
    conflicts = (
        _conflict(agent1_id=0, agent2_id=1, timestep=1),
        _conflict(agent1_id=0, agent2_id=1, timestep=2),
        _conflict(agent1_id=0, agent2_id=1, timestep=3),
    )
    mock_classify.side_effect = [
        ConflictCardinality.NON_CARDINAL,
        ConflictCardinality.SEMI_CARDINAL,
        ConflictCardinality.NON_CARDINAL,
    ]
    grid_map = build_grid_map([[0, 0], [0, 0]])
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=1, start_row=0, start_col=1, goal_row=0, goal_col=0),
    )

    selection = select_conflict_cardinal_first(
        grid_map=grid_map,
        scenario=scenario,
        node=_node_with_conflicts(*conflicts),
        max_timestep=5,
    )

    assert selection is not None
    assert selection.conflict == conflicts[1]
    assert selection.cardinality == ConflictCardinality.SEMI_CARDINAL
    assert selection.classified_conflicts == 3


@patch("pathfinding.src.algorithms.mapf.cbs_conflict_selection.classify_conflict")
def test_all_non_cardinal_conflicts_select_first_after_full_scan(mock_classify) -> None:
    conflicts = (
        _conflict(agent1_id=0, agent2_id=1, timestep=1),
        _conflict(agent1_id=0, agent2_id=1, timestep=2),
        _conflict(agent1_id=0, agent2_id=1, timestep=3),
    )
    mock_classify.return_value = ConflictCardinality.NON_CARDINAL
    grid_map = build_grid_map([[0, 0], [0, 0]])
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=1, start_row=0, start_col=1, goal_row=0, goal_col=0),
    )

    selection = select_conflict_cardinal_first(
        grid_map=grid_map,
        scenario=scenario,
        node=_node_with_conflicts(*conflicts),
        max_timestep=5,
    )

    assert selection is not None
    assert selection.conflict == conflicts[0]
    assert selection.cardinality == ConflictCardinality.NON_CARDINAL
    assert selection.classified_conflicts == 3
    assert mock_classify.call_count == 3


def test_conflict_free_node_returns_none() -> None:
    grid_map = build_grid_map([[0, 0], [0, 0]])
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=1, start_row=0, start_col=1, goal_row=0, goal_col=0),
    )

    assert select_conflict_cardinal_first(
        grid_map=grid_map,
        scenario=scenario,
        node=CBSNode(constraints=(), paths=(), cost=0, conflicts=()),
        max_timestep=5,
    ) is None


def test_expand_cbs_node_for_conflict_splits_explicit_conflict() -> None:
    grid_map, scenario, root = _swap_root()
    children = expand_cbs_node(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        max_timestep=5,
    )
    assert len(children) == 2
    second_conflict = children[0].conflicts[0]
    assert second_conflict != root.conflicts[0]

    node = CBSNode(
        constraints=root.constraints,
        paths=root.paths,
        cost=root.cost,
        conflicts=(root.conflicts[0], second_conflict),
    )

    explicit_children = expand_cbs_node_for_conflict(
        grid_map=grid_map,
        scenario=scenario,
        node=node,
        conflict=second_conflict,
        max_timestep=5,
    )

    assert explicit_children[0].constraints[-1] == split_conflict(second_conflict)[0]
    assert explicit_children != expand_cbs_node(
        grid_map=grid_map,
        scenario=scenario,
        node=node,
        max_timestep=5,
    )


def test_expand_cbs_node_matches_explicit_first_conflict_expansion() -> None:
    grid_map, scenario, root = _swap_root()

    old_style = expand_cbs_node(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        max_timestep=5,
    )
    explicit = expand_cbs_node_for_conflict(
        grid_map=grid_map,
        scenario=scenario,
        node=root,
        conflict=root.conflicts[0],
        max_timestep=5,
    )

    assert old_style == explicit


def test_basic_cbs_stats_have_zero_classification_overhead() -> None:
    grid_map, scenario, _ = _swap_root()

    run = solve_cbs_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
    )

    assert run.stats.classified_conflicts == 0
    assert run.stats.classification_low_level_searches == 0


def test_basic_cbs_solver_unchanged() -> None:
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

    assert solve_cbs(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
    ) == solve_cbs_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
    ).result


def test_cardinal_first_solver_finds_conflict_free_solution() -> None:
    grid_map, scenario, _ = _swap_root()

    result = solve_cbs_cardinal_first(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
    )

    assert result.success is True
    assert detect_conflicts(result.paths) == ()
    assert tuple(path.agent_id for path in result.paths) == (0, 1)


def test_cardinal_first_matches_basic_optimal_soc_on_known_case() -> None:
    grid_map, scenario, _ = _swap_root()

    basic = solve_cbs(grid_map=grid_map, scenario=scenario, max_timestep=5)
    cardinal = solve_cbs_cardinal_first(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
    )

    assert sum_of_costs(basic.paths) == 4
    assert sum_of_costs(cardinal.paths) == sum_of_costs(basic.paths)


def test_cardinal_first_solver_is_deterministic() -> None:
    grid_map, scenario, _ = _swap_root()

    first = solve_cbs_cardinal_first_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
    )
    second = solve_cbs_cardinal_first_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
    )

    assert first.result == second.result
    assert first.stats == second.stats
    assert first.termination_reason == second.termination_reason


def test_cardinal_first_classification_search_invariant() -> None:
    grid_map, scenario, _ = _swap_root()

    run = solve_cbs_cardinal_first_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
    )

    assert run.stats.classification_low_level_searches == 2 * run.stats.classified_conflicts
    assert run.stats.classified_conflicts > 0


def test_cardinal_first_cutoff_zero_has_no_classification_overhead() -> None:
    grid_map, scenario, _ = _swap_root()

    run = solve_cbs_cardinal_first_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
        max_expanded_nodes=0,
    )

    assert run.termination_reason == "expansion_limit"
    assert run.stats.expanded_ct_nodes == 0
    assert run.stats.classified_conflicts == 0
    assert run.stats.classification_low_level_searches == 0
