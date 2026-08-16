import pytest

from pathfinding.src.algorithms.mapf.cbs import CBSStats, solve_cbs, solve_cbs_with_stats
from pathfinding.src.algorithms.mapf.conflicts import detect_conflicts
from pathfinding.src.algorithms.mapf.models import MAPFAgent, MAPFResult, MAPFScenario
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


def _conflict_free_scenario() -> tuple:
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
    return grid_map, scenario


def _vertex_conflict_scenario() -> tuple:
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
    return grid_map, scenario


def _multi_expansion_scenario() -> tuple:
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


def test_conflict_free_root_stats() -> None:
    grid_map, scenario = _conflict_free_scenario()

    run = solve_cbs_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=6,
    )

    assert run.termination_reason == "success"
    assert run.stats == CBSStats(
        expanded_ct_nodes=0,
        generated_ct_nodes=1,
        low_level_replans=0,
        max_open_size=1,
    )


def test_root_failure_stats() -> None:
    grid_map = build_grid_map([[0, 1]])
    scenario = _scenario(
        _agent(agent_id=0, start_row=0, start_col=0, goal_row=0, goal_col=1),
        _agent(agent_id=1, start_row=0, start_col=0, goal_row=0, goal_col=1),
    )

    run = solve_cbs_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
    )

    assert run.result == MAPFResult(success=False, paths=())
    assert run.termination_reason == "failure"
    assert run.stats == CBSStats(
        expanded_ct_nodes=0,
        generated_ct_nodes=0,
        low_level_replans=0,
        max_open_size=0,
    )


def test_single_expansion_stats_match_standard_splitting() -> None:
    grid_map, scenario = _vertex_conflict_scenario()

    run = solve_cbs_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
    )

    assert run.termination_reason == "success"
    assert run.stats.expanded_ct_nodes == 1
    assert run.stats.low_level_replans == 2 * run.stats.expanded_ct_nodes


def test_generated_node_count_on_single_expansion_case() -> None:
    grid_map, scenario = _vertex_conflict_scenario()

    run = solve_cbs_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
    )

    assert run.stats.generated_ct_nodes == 3


def test_max_open_size_on_single_expansion_case() -> None:
    grid_map, scenario = _vertex_conflict_scenario()

    run = solve_cbs_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
    )

    assert run.stats.max_open_size == 2


def test_expansion_limit_zero_on_conflicting_root() -> None:
    grid_map, scenario = _vertex_conflict_scenario()

    run = solve_cbs_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
        max_expanded_nodes=0,
    )

    assert run.result is None
    assert run.termination_reason == "expansion_limit"
    assert run.stats == CBSStats(
        expanded_ct_nodes=0,
        generated_ct_nodes=1,
        low_level_replans=0,
        max_open_size=1,
    )


def test_expansion_limit_zero_on_conflict_free_root() -> None:
    grid_map, scenario = _conflict_free_scenario()

    run = solve_cbs_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=6,
        max_expanded_nodes=0,
    )

    assert run.result is not None
    assert run.result.success is True
    assert run.termination_reason == "success"
    assert detect_conflicts(run.result.paths) == ()


def test_small_expansion_cutoff_on_multi_expansion_scenario() -> None:
    grid_map, scenario = _multi_expansion_scenario()

    run = solve_cbs_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
        max_expanded_nodes=1,
    )

    assert run.result is None
    assert run.termination_reason == "expansion_limit"
    assert run.stats.expanded_ct_nodes == 1


def test_sufficient_expansion_limit_finds_solution() -> None:
    grid_map, scenario = _multi_expansion_scenario()

    run = solve_cbs_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
        max_expanded_nodes=100,
    )

    assert run.result is not None
    assert run.result.success is True
    assert run.termination_reason == "success"
    assert detect_conflicts(run.result.paths) == ()


def test_negative_expansion_limit_raises_value_error() -> None:
    grid_map, scenario = _conflict_free_scenario()

    with pytest.raises(ValueError, match="max_expanded_nodes must be non-negative"):
        solve_cbs_with_stats(
            grid_map=grid_map,
            scenario=scenario,
            max_timestep=6,
            max_expanded_nodes=-1,
        )


def test_ordinary_solve_cbs_matches_unlimited_stats_solver() -> None:
    grid_map, scenario = _vertex_conflict_scenario()

    normal_result = solve_cbs(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
    )
    stats_run = solve_cbs_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
        max_expanded_nodes=None,
    )

    assert stats_run.result == normal_result
    assert stats_run.termination_reason == "success"


def test_structural_stats_are_deterministic() -> None:
    grid_map, scenario = _multi_expansion_scenario()

    first = solve_cbs_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
    )
    second = solve_cbs_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=5,
    )

    assert first.stats == second.stats
    assert first.termination_reason == second.termination_reason
    assert first.result == second.result
