"""MAPF-9.2 integration validation: single-instance SPF → CGLPS → UBLS flow."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from pathfinding.src.algorithms.mapf.models import (
    AgentPath,
    MAPFAgent,
    MAPFResult,
    MAPFScenario,
    TimedState,
)
from pathfinding.src.algorithms.mapf.prioritized_planning import (
    PrioritizedPlanningRunResult,
    PrioritizedPlanningStats,
)
from pathfinding.src.experiments.mapf_bounded_local_priority_search import (
    MAPF9_CGLPS_BUDGET,
    MAPF9CandidateEvaluation,
    MAPF9Method,
    MAPF9MethodResult,
    MAPF9MethodTimings,
    MAPF9PreprocessingTimings,
    MAPF9SharedInstance,
    evaluate_mapf9_instance,
    evaluate_ubls,
    generate_cglps_candidate_specs,
    generate_ubls_candidate_specs,
    mapf9_cglps_logical_total_ms,
    mapf9_spf_logical_total_ms,
    mapf9_ubls_logical_total_ms,
    prepare_mapf9_instance,
    transpose_order,
)
from pathfinding.src.experiments.mapf_benchmark_execution import TERMINATION_FAILURE
from pathfinding.tests.helpers import build_grid_map

from pathfinding.src.core.models import Position


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


def _path(agent_id: int, coordinates: list[tuple[int, int, int]]) -> AgentPath:
    return AgentPath(
        agent_id=agent_id,
        states=tuple(
            TimedState(row=row, col=col, timestep=timestep)
            for row, col, timestep in coordinates
        ),
    )


def _pp_run(
    *,
    success: bool,
    soc: int = 10,
    makespan: int = 5,
    agent_count: int = 2,
) -> PrioritizedPlanningRunResult:
    if success:
        paths = tuple(
            _path(
                agent_id,
                [(agent_id, 0, 0), (agent_id, 1, 1)],
            )
            for agent_id in range(agent_count)
        )
        return PrioritizedPlanningRunResult(
            result=MAPFResult(success=True, paths=paths),
            stats=PrioritizedPlanningStats(
                low_level_searches=agent_count,
                agents_planned=agent_count,
            ),
            termination_reason="success",
        )
    return PrioritizedPlanningRunResult(
        result=MAPFResult(success=False, paths=()),
        stats=PrioritizedPlanningStats(low_level_searches=1, agents_planned=0),
        termination_reason=TERMINATION_FAILURE,
    )


def _synthetic_conflict_scenario() -> tuple:
    """5×5 open grid with crossing agents; independent-path conflicts expected."""
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        name="mapf9_integration_smoke",
    )
    scenario = _scenario(
        _agent(0, 0, 2, 4, 2),
        _agent(1, 2, 0, 2, 4),
        _agent(2, 0, 0, 0, 4),
    )
    return grid_map, scenario


# --- integration orchestration / shared SPF ---


def test_integration_spf_baseline_pp_called_exactly_once() -> None:
    grid_map, scenario = _synthetic_conflict_scenario()
    pp_runner = MagicMock(side_effect=lambda **_kwargs: _pp_run(success=True, agent_count=3))

    result = evaluate_mapf9_instance(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=30,
        instance_id="integration_spf_once",
        pp_runner=pp_runner,
    )

    a_i = result.actual_a_i
    assert pp_runner.call_count == 1 + a_i + a_i
    assert result.physical_pp_eval_count == 1 + a_i + a_i
    assert result.cglps.additional_pp_eval_count == a_i
    assert result.ubls.additional_pp_eval_count == a_i
    assert result.spf.baseline_spf_evaluation is result.cglps.baseline_spf_evaluation
    assert result.spf.baseline_spf_evaluation is result.ubls.baseline_spf_evaluation


def test_integration_matched_ai_invariant() -> None:
    grid_map, scenario = _synthetic_conflict_scenario()
    result = evaluate_mapf9_instance(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=30,
        instance_id="integration_ai_match",
    )
    assert result.actual_a_i == result.cglps.additional_pp_eval_count
    assert result.actual_a_i == result.ubls.additional_pp_eval_count
    assert result.physical_pp_eval_count == 1 + 2 * result.actual_a_i


# --- UBLS isolation ---


def test_integration_ubls_isolated_from_cglps_outcomes() -> None:
    grid_map, scenario = _synthetic_conflict_scenario()
    shared = prepare_mapf9_instance(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=30,
    )
    assert isinstance(shared, MAPF9SharedInstance)

    baseline_reference = evaluate_mapf9_instance(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=30,
        instance_id="ubls_isolation_ref",
    )
    a_i = baseline_reference.actual_a_i
    expected_ubls_specs = generate_ubls_candidate_specs(
        shared,
        instance_id="ubls_isolation_ref",
        additional_count=a_i,
    )

    fake_cglps = replace(
        baseline_reference.cglps,
        success=not baseline_reference.cglps.success,
        soc=(baseline_reference.cglps.soc or 0) + 999,
        makespan=(baseline_reference.cglps.makespan or 0) + 999,
        selected_candidate_id=99,
    )

    ubls_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def capture_ubls(*args, **kwargs):
        ubls_calls.append((args, kwargs))
        return evaluate_ubls(*args, **kwargs)

    with patch(
        "pathfinding.src.experiments.mapf_bounded_local_priority_search.evaluate_cglps",
        return_value=fake_cglps,
    ):
        with patch(
            "pathfinding.src.experiments.mapf_bounded_local_priority_search.evaluate_ubls",
            side_effect=capture_ubls,
        ):
            result = evaluate_mapf9_instance(
                grid_map=grid_map,
                scenario=scenario,
                max_timestep=30,
                instance_id="ubls_isolation_ref",
            )

    assert len(ubls_calls) == 1
    ubls_args, ubls_kwargs = ubls_calls[0]
    assert len(ubls_args) == 2
    assert ubls_kwargs.keys() == {"instance_id", "additional_count", "pp_runner"}
    assert ubls_kwargs["additional_count"] == a_i
    assert "cglps" not in ubls_kwargs
    assert not any(
        isinstance(value, MAPF9MethodResult) and value.method == MAPF9Method.CGLPS
        for value in (*ubls_args, *ubls_kwargs.values())
    )
    assert result.ubls.additional_candidate_specs == expected_ubls_specs


# --- timing accounting ---


def test_integration_preprocessing_times_non_negative() -> None:
    grid_map, scenario = _synthetic_conflict_scenario()
    result = evaluate_mapf9_instance(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=30,
        instance_id="timing_non_negative",
    )
    assert result.shared is not None
    prep = result.shared.preprocessing_timings
    assert prep.independent_path_time_ms >= 0.0
    assert prep.conflict_detection_time_ms >= 0.0
    assert prep.spf_ordering_time_ms >= 0.0


def test_integration_logical_runtime_reconstruction_with_fixture_timings() -> None:
    preprocessing = MAPF9PreprocessingTimings(
        independent_path_time_ms=100.0,
        conflict_detection_time_ms=20.0,
        spf_ordering_time_ms=5.0,
    )
    spf_timings = MAPF9MethodTimings(baseline_spf_pp_time_ms=50.0)
    cglps_timings = MAPF9MethodTimings(
        baseline_spf_pp_time_ms=50.0,
        cglps_candidate_ranking_time_ms=3.0,
        cglps_additional_pp_time_ms=12.0,
    )
    ubls_timings = MAPF9MethodTimings(
        baseline_spf_pp_time_ms=50.0,
        ubls_candidate_generation_time_ms=2.0,
        ubls_additional_pp_time_ms=12.0,
    )

    spf_total = mapf9_spf_logical_total_ms(preprocessing, spf_timings)
    cglps_total = mapf9_cglps_logical_total_ms(preprocessing, cglps_timings)
    ubls_total = mapf9_ubls_logical_total_ms(preprocessing, ubls_timings)

    assert spf_total == 155.0
    assert cglps_total == spf_total + 20.0 + 3.0 + 12.0
    assert ubls_total == spf_total + 2.0 + 12.0
    assert preprocessing.conflict_detection_time_ms not in (
        spf_total,
        ubls_total - spf_total,
    )


def test_integration_measured_timings_not_double_counted() -> None:
    grid_map, scenario = _synthetic_conflict_scenario()
    result = evaluate_mapf9_instance(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=30,
        instance_id="timing_no_double_count",
    )
    assert result.shared is not None
    prep = result.shared.preprocessing_timings
    assert result.cglps.method_timings is not None
    assert result.ubls.method_timings is not None

    spf_total = mapf9_spf_logical_total_ms(prep, result.spf.method_timings)
    cglps_total = mapf9_cglps_logical_total_ms(prep, result.cglps.method_timings)
    ubls_total = mapf9_ubls_logical_total_ms(prep, result.ubls.method_timings)

    assert cglps_total >= spf_total
    assert ubls_total >= spf_total
    assert result.cglps.method_timings.cglps_candidate_ranking_time_ms >= 0.0
    assert prep.conflict_detection_time_ms >= 0.0


# --- candidate invariants ---


def test_integration_cglps_candidate_invariants() -> None:
    grid_map, scenario = _synthetic_conflict_scenario()
    shared = prepare_mapf9_instance(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=30,
    )
    assert isinstance(shared, MAPF9SharedInstance)
    assert len(shared.conflict_edges) >= 1

    specs = generate_cglps_candidate_specs(shared)
    assert len(specs) <= MAPF9_CGLPS_BUDGET

    baseline_order = shared.spf_order
    for spec in specs:
        assert spec.agent_order == transpose_order(
            baseline_order,
            spec.source_agent_a_id,
            spec.source_agent_b_id,
        )
        assert spec.canonical_original_index_pair in shared.conflict_edges

    ranked_pairs = [
        spec.canonical_original_index_pair
        for spec in specs
        if spec.canonical_original_index_pair is not None
    ]
    for left_index in range(len(ranked_pairs) - 1):
        left_pair = ranked_pairs[left_index]
        right_pair = ranked_pairs[left_index + 1]
        assert left_pair is not None and right_pair is not None
        left_count = shared.pair_event_counts.get(left_pair, 0)
        right_count = shared.pair_event_counts.get(right_pair, 0)
        assert (-left_count, left_pair[0], left_pair[1]) <= (
            -right_count,
            right_pair[0],
            right_pair[1],
        )


def test_integration_ubls_candidate_invariants() -> None:
    grid_map, scenario = _synthetic_conflict_scenario()
    result = evaluate_mapf9_instance(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=30,
        instance_id="ubls_candidate_invariants",
    )
    assert result.shared is not None
    a_i = result.actual_a_i

    baseline_eval = result.ubls.candidate_evaluations[0]
    assert baseline_eval.candidate_id == 0
    assert baseline_eval.agent_order == result.shared.spf_order
    assert len(result.ubls.additional_candidate_specs) == a_i

    for spec in result.ubls.additional_candidate_specs:
        assert spec.agent_order == transpose_order(
            result.shared.spf_order,
            spec.source_agent_a_id,
            spec.source_agent_b_id,
        )
        assert spec.pair_conflict_event_count is None

    repeat = generate_ubls_candidate_specs(
        result.shared,
        instance_id="ubls_candidate_invariants",
        additional_count=a_i,
    )
    assert repeat == result.ubls.additional_candidate_specs


# --- selection invariants ---


def test_integration_selection_not_worse_than_spf_when_spf_succeeds() -> None:
    grid_map, scenario = _synthetic_conflict_scenario()
    result = evaluate_mapf9_instance(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=30,
        instance_id="selection_invariants",
    )
    if not result.spf.success:
        pytest.skip("SPF did not succeed on synthetic scenario")

    spf_soc = result.spf.soc
    spf_makespan = result.spf.makespan
    assert spf_soc is not None and spf_makespan is not None

    for method_result in (result.cglps, result.ubls):
        if not method_result.success:
            continue
        assert method_result.soc is not None
        assert method_result.makespan is not None
        assert method_result.soc <= spf_soc
        if method_result.soc == spf_soc:
            assert method_result.makespan <= spf_makespan


# --- failure integration ---


def test_integration_ordering_failure_zero_pp_calls() -> None:
    grid_map = build_grid_map([[0, 1], [0, 0]])
    scenario = _scenario(_agent(0, 0, 0, 1, 1))
    pp_runner = MagicMock()

    result = evaluate_mapf9_instance(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=1,
        instance_id="ordering_failure",
        pp_runner=pp_runner,
    )

    assert result.ordering_failure is not None
    assert result.physical_pp_eval_count == 0
    assert result.actual_a_i == 0
    pp_runner.assert_not_called()
    for method_result in (result.spf, result.cglps, result.ubls):
        assert method_result.success is False
        assert method_result.soc is None
        assert method_result.makespan is None


def test_integration_spf_failure_cglps_recovery() -> None:
    grid_map, scenario = _synthetic_conflict_scenario()
    call_index = {"count": 0}

    def pp_runner(**_kwargs):
        call_index["count"] += 1
        if call_index["count"] == 1:
            return _pp_run(success=False)
        return _pp_run(success=True, soc=8, makespan=4, agent_count=3)

    result = evaluate_mapf9_instance(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=30,
        instance_id="cglps_recovery",
        pp_runner=pp_runner,
    )

    assert result.spf.success is False
    if result.cglps.success:
        assert result.cglps.selected_candidate_id != 0


def test_integration_spf_failure_ubls_recovery() -> None:
    grid_map, scenario = _synthetic_conflict_scenario()
    prepared = prepare_mapf9_instance(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=30,
    )
    assert isinstance(prepared, MAPF9SharedInstance)
    a_i = len(generate_cglps_candidate_specs(prepared))
    if a_i == 0:
        pytest.skip("synthetic scenario produced A_i=0")

    call_index = {"count": 0}

    def pp_runner(**_kwargs):
        call_index["count"] += 1
        current = call_index["count"]
        if current == 1:
            return _pp_run(success=False)
        if current <= 1 + a_i:
            return _pp_run(success=False)
        if current == 2 + a_i:
            return _pp_run(success=True, soc=9, makespan=4, agent_count=3)
        return _pp_run(success=False)

    result = evaluate_mapf9_instance(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=30,
        instance_id="ubls_recovery",
        pp_runner=pp_runner,
    )

    assert result.spf.success is False
    assert result.ubls.success is True
    assert result.ubls.selected_candidate_id != 0


def test_integration_all_candidates_fail_no_fabricated_metrics() -> None:
    grid_map, scenario = _synthetic_conflict_scenario()
    pp_runner = MagicMock(side_effect=lambda **_kwargs: _pp_run(success=False))

    result = evaluate_mapf9_instance(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=30,
        instance_id="all_fail",
        pp_runner=pp_runner,
    )

    for method_result in (result.spf, result.cglps, result.ubls):
        assert method_result.success is False
        assert method_result.soc is None
        assert method_result.makespan is None


# --- synthetic real-API smoke ---


def test_synthetic_real_api_smoke() -> None:
    grid_map, scenario = _synthetic_conflict_scenario()
    result = evaluate_mapf9_instance(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=30,
        instance_id="mapf9_synthetic_smoke",
    )

    assert result.shared is not None
    shared = result.shared
    assert shared.grid_map.height == 5
    assert shared.grid_map.width == 5
    assert len(scenario.agents) == 3
    assert len(shared.inputs.conflicts) >= 1
    assert result.actual_a_i >= 1

    smoke_report = {
        "map_dimensions": (shared.grid_map.height, shared.grid_map.width),
        "agent_count": len(scenario.agents),
        "spf_order": shared.spf_order,
        "independent_conflict_count": len(shared.inputs.conflicts),
        "pair_event_counts": dict(shared.pair_event_counts),
        "a_i": result.actual_a_i,
        "cglps_candidate_orders": [
            evaluation.agent_order
            for evaluation in result.cglps.candidate_evaluations
        ],
        "ubls_candidate_orders": [
            evaluation.agent_order
            for evaluation in result.ubls.candidate_evaluations
        ],
        "spf_success": result.spf.success,
        "cglps_success": result.cglps.success,
        "ubls_success": result.ubls.success,
        "spf_soc": result.spf.soc,
        "cglps_soc": result.cglps.soc,
        "ubls_soc": result.ubls.soc,
        "spf_makespan": result.spf.makespan,
        "cglps_makespan": result.cglps.makespan,
        "ubls_makespan": result.ubls.makespan,
        "physical_pp_eval_count": result.physical_pp_eval_count,
    }

    for method_key in ("spf", "cglps", "ubls"):
        method_result: MAPF9MethodResult = getattr(result, method_key)
        if method_result.success:
            assert method_result.conflict_count == 0
            smoke_report[f"{method_key}_final_conflict_count"] = method_result.conflict_count

    assert smoke_report["physical_pp_eval_count"] == 1 + 2 * smoke_report["a_i"]
    assert smoke_report["cglps_candidate_orders"][0] == smoke_report["spf_order"]
    assert smoke_report["ubls_candidate_orders"][0] == smoke_report["spf_order"]
