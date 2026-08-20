from __future__ import annotations

import json
from pathlib import Path

import pytest

from pathfinding.src.algorithms.mapf.models import MAPFAgent, MAPFScenario
from pathfinding.src.core.models import Position
from pathfinding.src.experiments.mapf_benchmark_execution import (
    MAIN_BENCHMARK_CSV,
    MAIN_BENCHMARK_JSONL,
    MAPFBenchmarkAlgorithm,
    TERMINATION_EXPANSION_LIMIT,
    TERMINATION_FAILURE,
    TERMINATION_SUCCESS,
)
from pathfinding.src.experiments.mapf_cbs_budget_calibration import (
    DEFAULT_CALIBRATION_RESULTS_DIR,
    MAPFCBSSearchMetrics,
    DEFAULT_EXPANSION_BUDGETS,
    CalibrationCheckpointStore,
    MAPFCBSCalibrationPlanEntry,
    MAPFCBSCalibrationRecord,
    budgets_to_execute_for_pair,
    build_calibration_run_plan,
    calibration_pair_key,
    execute_cbs_calibration_run_from_loaded,
    is_calibration_pair_complete,
    load_calibration_records,
    run_progressive_calibration,
)
from pathfinding.src.experiments.mapf_benchmark_instances import (
    MAPFBenchmarkInstance,
    MAPFInteractionLevel,
)
from pathfinding.tests.helpers import build_grid_map


def _instance(instance_id: str) -> MAPFBenchmarkInstance:
    return MAPFBenchmarkInstance(
        instance_id=instance_id,
        agent_count=2,
        interaction_level=MAPFInteractionLevel.LOW,
        scenario_indices=(0, 1),
        independent_conflict_count=0,
        conflicting_agent_pair_count=0,
        independent_soc=4,
        independent_makespan=2,
        vertex_conflict_count=0,
        edge_conflict_count=0,
    )


def _metrics(**overrides: int) -> MAPFCBSSearchMetrics:
    defaults = dict(
        expanded_ct_nodes=0,
        generated_ct_nodes=0,
        low_level_replans=0,
        max_open_size=0,
        unique_constraint_signatures=0,
        duplicate_constraint_signatures=0,
        unique_path_signatures=0,
        duplicate_path_signatures=0,
        classified_conflicts=0,
        classification_low_level_searches=0,
        selected_cardinal_conflicts=0,
        selected_semi_cardinal_conflicts=0,
        selected_non_cardinal_conflicts=0,
        combined_low_level_searches=0,
    )
    defaults.update(overrides)
    return MAPFCBSSearchMetrics(**defaults)


def _calibration_record(
    *,
    instance_id: str = "bench_n02_low_000",
    algorithm: MAPFBenchmarkAlgorithm = MAPFBenchmarkAlgorithm.CBS_BASIC,
    budget: int,
    termination_reason: str,
    success: bool = False,
) -> MAPFCBSCalibrationRecord:
    return MAPFCBSCalibrationRecord(
        instance_id=instance_id,
        algorithm=algorithm,
        max_expanded_nodes=budget,
        agent_count=2,
        interaction_level="low",
        success=success,
        termination_reason=termination_reason,
        execution_time_ms=float(budget),
        soc=10 if success else None,
        makespan=5 if success else None,
        independent_soc=4,
        independent_makespan=2,
        independent_conflict_count=0,
        cbs_search_metrics=_metrics(
            expanded_ct_nodes=budget,
            low_level_replans=budget,
            combined_low_level_searches=budget,
        ),
    )


def test_progressive_escalation_runs_until_success() -> None:
    budgets = (10, 25, 50, 100, 250)
    executed: list[int] = []

    def fake_executor(entry: MAPFCBSCalibrationPlanEntry) -> MAPFCBSCalibrationRecord:
        executed.append(entry.max_expanded_nodes)
        if entry.max_expanded_nodes < 50:
            return _calibration_record(
                budget=entry.max_expanded_nodes,
                termination_reason=TERMINATION_EXPANSION_LIMIT,
            )
        return _calibration_record(
            budget=entry.max_expanded_nodes,
            termination_reason=TERMINATION_SUCCESS,
            success=True,
        )

    run_progressive_calibration(
        (_instance("bench_n02_low_000"),),
        algorithms=(MAPFBenchmarkAlgorithm.CBS_BASIC,),
        budgets=budgets,
        executor=fake_executor,
        resume=False,
    )

    assert executed == [10, 25, 50]


def test_normal_failure_stops_escalation() -> None:
    budgets = (10, 25, 50, 100, 250)
    executed: list[int] = []

    def fake_executor(entry: MAPFCBSCalibrationPlanEntry) -> MAPFCBSCalibrationRecord:
        executed.append(entry.max_expanded_nodes)
        return _calibration_record(
            budget=entry.max_expanded_nodes,
            termination_reason=TERMINATION_FAILURE,
        )

    run_progressive_calibration(
        (_instance("bench_n02_low_000"),),
        algorithms=(MAPFBenchmarkAlgorithm.CBS_BASIC,),
        budgets=budgets,
        executor=fake_executor,
        resume=False,
    )

    assert executed == [10]


def test_success_stops_escalation() -> None:
    budgets = (10, 25, 50, 100, 250)
    executed: list[int] = []

    def fake_executor(entry: MAPFCBSCalibrationPlanEntry) -> MAPFCBSCalibrationRecord:
        executed.append(entry.max_expanded_nodes)
        return _calibration_record(
            budget=entry.max_expanded_nodes,
            termination_reason=TERMINATION_SUCCESS,
            success=True,
        )

    run_progressive_calibration(
        (_instance("bench_n02_low_000"),),
        algorithms=(MAPFBenchmarkAlgorithm.CBS_BASIC,),
        budgets=budgets,
        executor=fake_executor,
        resume=False,
    )

    assert executed == [10]


def test_full_limit_ladder_executes_all_budgets() -> None:
    budgets = (10, 25, 50, 100, 250)
    executed: list[int] = []

    def fake_executor(entry: MAPFCBSCalibrationPlanEntry) -> MAPFCBSCalibrationRecord:
        executed.append(entry.max_expanded_nodes)
        return _calibration_record(
            budget=entry.max_expanded_nodes,
            termination_reason=TERMINATION_EXPANSION_LIMIT,
        )

    run_progressive_calibration(
        (_instance("bench_n02_low_000"),),
        algorithms=(MAPFBenchmarkAlgorithm.CBS_BASIC,),
        budgets=budgets,
        executor=fake_executor,
        resume=False,
    )

    assert executed == [10, 25, 50, 100, 250]


def test_resume_mid_ladder_skips_completed_budgets(tmp_path: Path) -> None:
    budgets = (10, 25, 50, 100, 250)
    store = CalibrationCheckpointStore(
        results_dir=tmp_path,
        csv_path=tmp_path / "calibration_results.csv",
        jsonl_path=tmp_path / "calibration_results.jsonl",
    )
    store.append_record(
        _calibration_record(budget=10, termination_reason=TERMINATION_EXPANSION_LIMIT)
    )
    store.append_record(
        _calibration_record(budget=25, termination_reason=TERMINATION_EXPANSION_LIMIT)
    )

    executed: list[int] = []

    def fake_executor(entry: MAPFCBSCalibrationPlanEntry) -> MAPFCBSCalibrationRecord:
        executed.append(entry.max_expanded_nodes)
        return _calibration_record(
            budget=entry.max_expanded_nodes,
            termination_reason=TERMINATION_SUCCESS,
            success=True,
        )

    run_progressive_calibration(
        (_instance("bench_n02_low_000"),),
        algorithms=(MAPFBenchmarkAlgorithm.CBS_BASIC,),
        budgets=budgets,
        executor=fake_executor,
        checkpoint=store,
        resume=True,
    )

    assert executed == [50]


def test_resume_after_success_skips_remaining_budgets(tmp_path: Path) -> None:
    budgets = (10, 25, 50, 100, 250)
    store = CalibrationCheckpointStore(
        results_dir=tmp_path,
        csv_path=tmp_path / "calibration_results.csv",
        jsonl_path=tmp_path / "calibration_results.jsonl",
    )
    store.append_record(
        _calibration_record(budget=10, termination_reason=TERMINATION_EXPANSION_LIMIT)
    )
    store.append_record(
        _calibration_record(
            budget=25,
            termination_reason=TERMINATION_SUCCESS,
            success=True,
        )
    )

    executed: list[int] = []

    def fake_executor(entry: MAPFCBSCalibrationPlanEntry) -> MAPFCBSCalibrationRecord:
        executed.append(entry.max_expanded_nodes)
        return _calibration_record(
            budget=entry.max_expanded_nodes,
            termination_reason=TERMINATION_SUCCESS,
            success=True,
        )

    run_progressive_calibration(
        (_instance("bench_n02_low_000"),),
        algorithms=(MAPFBenchmarkAlgorithm.CBS_BASIC,),
        budgets=budgets,
        executor=fake_executor,
        checkpoint=store,
        resume=True,
    )

    assert executed == []


def test_calibration_serialization_roundtrip(tmp_path: Path) -> None:
    record = _calibration_record(
        budget=50,
        termination_reason=TERMINATION_SUCCESS,
        success=True,
    )
    store = CalibrationCheckpointStore(
        results_dir=tmp_path,
        csv_path=tmp_path / "calibration_results.csv",
        jsonl_path=tmp_path / "calibration_results.jsonl",
    )
    store.append_record(record)

    loaded = load_calibration_records(tmp_path / "calibration_results.jsonl")
    restored = next(iter(loaded.values()))
    assert restored.instance_id == record.instance_id
    assert restored.algorithm == record.algorithm
    assert restored.max_expanded_nodes == 50
    assert restored.termination_reason == TERMINATION_SUCCESS
    assert restored.execution_time_ms == 50.0
    assert restored.cbs_search_metrics.expanded_ct_nodes == 50

    line = (tmp_path / "calibration_results.jsonl").read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    assert payload["independent_soc"] == 4
    assert payload["cbs_combined_low_level_searches"] == 50


def test_cardinal_first_metrics_on_tiny_synthetic_run() -> None:
    from pathfinding.src.core.models import Scenario

    grid_map = build_grid_map([[0, 0, 0, 0, 0] for _ in range(5)], name="bench.map")
    scenarios = [
        Scenario(
            map_name="bench.map",
            width=5,
            height=5,
            start=Position(row=0, col=0),
            goal=Position(row=0, col=4),
            optimal_length=25.0,
        ),
        Scenario(
            map_name="bench.map",
            width=5,
            height=5,
            start=Position(row=1, col=0),
            goal=Position(row=1, col=4),
            optimal_length=25.0,
        ),
    ]
    instance = _instance("bench_n02_low_000")

    record = execute_cbs_calibration_run_from_loaded(
        grid_map=grid_map,
        scenarios=scenarios,
        instance=instance,
        algorithm=MAPFBenchmarkAlgorithm.CBS_CARDINAL_FIRST,
        max_timestep=8,
        max_expanded_nodes=250,
    )

    assert record.cbs_search_metrics.classified_conflicts >= 0
    assert record.cbs_search_metrics.classification_low_level_searches >= 0
    assert (
        record.cbs_search_metrics.combined_low_level_searches
        == record.cbs_search_metrics.low_level_replans
        + record.cbs_search_metrics.classification_low_level_searches
    )


def test_main_benchmark_paths_are_isolated_from_calibration_paths() -> None:
    assert str(DEFAULT_CALIBRATION_RESULTS_DIR) != str(MAIN_BENCHMARK_JSONL.parent)
    assert "mapf_cbs_calibration" in str(DEFAULT_CALIBRATION_RESULTS_DIR)
    assert "mapf_benchmark_execution" in str(MAIN_BENCHMARK_JSONL)
    assert MAIN_BENCHMARK_CSV.name == "results.csv"


def test_build_calibration_run_plan_only_schedules_first_budget_initially() -> None:
    plan = build_calibration_run_plan(
        (_instance("bench_n02_low_000"),),
        algorithms=(MAPFBenchmarkAlgorithm.CBS_BASIC,),
        budgets=(10, 25, 50),
    )
    assert len(plan) == 1
    assert plan[0].max_expanded_nodes == 10


def test_budgets_to_execute_for_pair_after_expansion_limit() -> None:
    existing = {
        10: _calibration_record(budget=10, termination_reason=TERMINATION_EXPANSION_LIMIT),
    }
    assert budgets_to_execute_for_pair(budgets=(10, 25, 50), existing_records=existing) == (
        25,
    )
    assert not is_calibration_pair_complete(
        budgets=(10, 25, 50),
        existing_records=existing,
    )


def test_pair_complete_after_success() -> None:
    existing = {
        10: _calibration_record(
            budget=10,
            termination_reason=TERMINATION_SUCCESS,
            success=True,
        ),
    }
    assert is_calibration_pair_complete(
        budgets=(10, 25, 50),
        existing_records=existing,
    )
    assert budgets_to_execute_for_pair(budgets=(10, 25, 50), existing_records=existing) == ()
