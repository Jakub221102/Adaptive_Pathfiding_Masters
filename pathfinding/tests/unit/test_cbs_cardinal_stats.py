import pytest

from pathfinding.src.algorithms.mapf.cbs import (
    solve_cbs_cardinal_first,
    solve_cbs_cardinal_first_with_stats,
    solve_cbs_with_stats,
)
from pathfinding.src.algorithms.mapf.cbs_conflict_classification import (
    classify_conflict,
)
from pathfinding.src.algorithms.mapf.cbs import build_cbs_root
from pathfinding.src.algorithms.mapf.models import MAPFAgent, MAPFScenario
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


def _swap_root_scenario() -> tuple:
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
    return grid_map, scenario


def _non_cardinal_root_scenario() -> tuple:
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
    return grid_map, scenario


def _selected_total(stats) -> int:
    return (
        stats.selected_cardinal_conflicts
        + stats.selected_semi_cardinal_conflicts
        + stats.selected_non_cardinal_conflicts
    )


def test_basic_cbs_selected_cardinality_counters_are_zero() -> None:
    grid_map, scenario = _swap_root_scenario()

    run = solve_cbs_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
        max_expanded_nodes=1,
    )

    assert run.stats.selected_cardinal_conflicts == 0
    assert run.stats.selected_semi_cardinal_conflicts == 0
    assert run.stats.selected_non_cardinal_conflicts == 0


def test_cardinal_selected_total_equals_expanded_nodes() -> None:
    grid_map, scenario = _swap_root_scenario()

    run = solve_cbs_cardinal_first_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
        max_expanded_nodes=1,
    )

    assert _selected_total(run.stats) == run.stats.expanded_ct_nodes


def test_known_cardinal_selection_increments_counter() -> None:
    grid_map, scenario = _swap_root_scenario()
    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=5)
    assert root is not None

    from pathfinding.src.algorithms.mapf.cbs_conflict_classification import (
        ConflictCardinality,
    )

    assert (
        classify_conflict(
            grid_map=grid_map,
            scenario=scenario,
            node=root,
            conflict=root.conflicts[0],
            max_timestep=5,
        )
        == ConflictCardinality.CARDINAL
    )

    run = solve_cbs_cardinal_first_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
        max_expanded_nodes=1,
    )

    assert run.stats.expanded_ct_nodes == 1
    assert run.stats.selected_cardinal_conflicts == 1
    assert run.stats.selected_semi_cardinal_conflicts == 0
    assert run.stats.selected_non_cardinal_conflicts == 0


def test_known_non_cardinal_selection_with_expansion_budget_one() -> None:
    grid_map, scenario = _non_cardinal_root_scenario()
    root = build_cbs_root(grid_map=grid_map, scenario=scenario, max_timestep=8)
    assert root is not None

    from pathfinding.src.algorithms.mapf.cbs_conflict_classification import (
        ConflictCardinality,
    )

    assert (
        classify_conflict(
            grid_map=grid_map,
            scenario=scenario,
            node=root,
            conflict=root.conflicts[0],
            max_timestep=8,
        )
        == ConflictCardinality.NON_CARDINAL
    )

    run = solve_cbs_cardinal_first_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=8,
        max_expanded_nodes=1,
    )

    assert run.stats.expanded_ct_nodes == 1
    assert run.stats.selected_non_cardinal_conflicts == 1
    assert run.stats.selected_cardinal_conflicts == 0
    assert run.stats.selected_semi_cardinal_conflicts == 0


def test_cutoff_zero_selected_cardinality_counters_are_zero() -> None:
    grid_map, scenario = _swap_root_scenario()

    run = solve_cbs_cardinal_first_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
        max_expanded_nodes=0,
    )

    assert run.stats.selected_cardinal_conflicts == 0
    assert run.stats.selected_semi_cardinal_conflicts == 0
    assert run.stats.selected_non_cardinal_conflicts == 0


def test_selected_cardinality_counters_are_deterministic() -> None:
    grid_map, scenario = _swap_root_scenario()

    first = solve_cbs_cardinal_first_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
        max_expanded_nodes=3,
    )
    second = solve_cbs_cardinal_first_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
        max_expanded_nodes=3,
    )

    assert first.stats.selected_cardinal_conflicts == second.stats.selected_cardinal_conflicts
    assert (
        first.stats.selected_semi_cardinal_conflicts
        == second.stats.selected_semi_cardinal_conflicts
    )
    assert (
        first.stats.selected_non_cardinal_conflicts
        == second.stats.selected_non_cardinal_conflicts
    )


def test_cardinal_solver_result_unchanged_after_instrumentation() -> None:
    grid_map, scenario = _swap_root_scenario()

    first = solve_cbs_cardinal_first(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
    )
    second = solve_cbs_cardinal_first(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
    )

    assert first == second


def test_existing_classification_invariant_preserved() -> None:
    grid_map, scenario = _swap_root_scenario()

    run = solve_cbs_cardinal_first_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
    )

    assert (
        run.stats.classification_low_level_searches
        == 2 * run.stats.classified_conflicts
    )


def test_existing_cost_distribution_invariants_preserved() -> None:
    grid_map, scenario = _swap_root_scenario()

    run = solve_cbs_cardinal_first_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
        max_expanded_nodes=2,
    )

    assert (
        sum(count for _, count in run.stats.generated_cost_distribution)
        == run.stats.generated_ct_nodes
    )
    assert (
        sum(count for _, count in run.stats.expanded_cost_distribution)
        == run.stats.expanded_ct_nodes
    )


def test_classified_conflicts_are_at_least_selected_conflicts() -> None:
    grid_map, scenario = _swap_root_scenario()

    run = solve_cbs_cardinal_first_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
        max_expanded_nodes=3,
    )

    selected_total = _selected_total(run.stats)
    assert run.stats.classified_conflicts >= selected_total
    assert selected_total == run.stats.expanded_ct_nodes
