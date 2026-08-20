from __future__ import annotations

import json
from pathlib import Path

from pathfinding.src.experiments.mapf_benchmark_analysis import (
    COMMON_BUDGET_MS,
    AnalysisConfig,
    build_cardinal_overhead_summary,
    build_paired_quality_comparison,
    build_pp_vs_basic_quality_summary,
    build_runtime_success_summary,
    count_directional_soc_better,
    run_benchmark_analysis,
    validate_dataset,
    within_common_budget,
)
from pathfinding.src.experiments.mapf_benchmark_execution import (
    MAPFBenchmarkAlgorithm,
    MAPFBenchmarkRunRecord,
    MAPFCBSSearchMetrics,
    MAPFPPSearchMetrics,
    TERMINATION_ERROR,
    TERMINATION_EXPANSION_LIMIT,
    TERMINATION_FAILURE,
    TERMINATION_SUCCESS,
    TERMINATION_TIME_LIMIT,
    _run_record_to_json,
)
from pathfinding.src.experiments.mapf_benchmark_instances import (
    MAPFBenchmarkInstance,
    MAPFBenchmarkManifest,
    MAPFInteractionLevel,
    save_benchmark_manifest,
)


def _instance(
    instance_id: str,
    agent_count: int,
    level: MAPFInteractionLevel,
    *,
    conflict_count: int = 0,
) -> MAPFBenchmarkInstance:
    return MAPFBenchmarkInstance(
        instance_id=instance_id,
        agent_count=agent_count,
        interaction_level=level,
        scenario_indices=(0, 1),
        independent_conflict_count=conflict_count,
        conflicting_agent_pair_count=0,
        independent_soc=100,
        independent_makespan=20,
        vertex_conflict_count=0,
        edge_conflict_count=0,
    )


def _manifest(*instances: MAPFBenchmarkInstance) -> MAPFBenchmarkManifest:
    return MAPFBenchmarkManifest(
        map_name="bench.map",
        scenario_name="bench.map.scen",
        seed=2026,
        max_timestep=8,
        min_reference_length=1.0,
        instances=instances,
    )


def _pp_record(
    instance_id: str,
    *,
    agent_count: int = 5,
    interaction_level: str = "low",
    success: bool = True,
    termination_reason: str = TERMINATION_SUCCESS,
    execution_time_ms: float = 1000.0,
    soc: int | None = 100,
    makespan: int | None = 20,
) -> MAPFBenchmarkRunRecord:
    return MAPFBenchmarkRunRecord(
        instance_id=instance_id,
        algorithm=MAPFBenchmarkAlgorithm.FIXED_PRIORITY_PP,
        agent_count=agent_count,
        interaction_level=interaction_level,
        success=success,
        termination_reason=termination_reason,
        execution_time_ms=execution_time_ms,
        soc=soc if success else None,
        makespan=makespan if success else None,
        remaining_conflicts=0 if success else None,
        independent_soc=100,
        independent_makespan=20,
        independent_conflict_count=0,
        pp_search_metrics=MAPFPPSearchMetrics(low_level_searches=5, agents_planned=5),
    )


def _cbs_record(
    instance_id: str,
    algorithm: MAPFBenchmarkAlgorithm,
    *,
    agent_count: int = 5,
    interaction_level: str = "low",
    success: bool = True,
    termination_reason: str = TERMINATION_SUCCESS,
    execution_time_ms: float = 2000.0,
    soc: int | None = 90,
    makespan: int | None = 18,
    expanded_ct_nodes: int = 10,
    combined_low_level_searches: int = 12,
    classification_low_level_searches: int = 0,
    classified_conflicts: int = 0,
) -> MAPFBenchmarkRunRecord:
    return MAPFBenchmarkRunRecord(
        instance_id=instance_id,
        algorithm=algorithm,
        agent_count=agent_count,
        interaction_level=interaction_level,
        success=success,
        termination_reason=termination_reason,
        execution_time_ms=execution_time_ms,
        soc=soc if success else None,
        makespan=makespan if success else None,
        remaining_conflicts=0 if success else None,
        independent_soc=100,
        independent_makespan=20,
        independent_conflict_count=0,
        cbs_search_metrics=MAPFCBSSearchMetrics(
            expanded_ct_nodes=expanded_ct_nodes,
            generated_ct_nodes=expanded_ct_nodes + 1,
            low_level_replans=combined_low_level_searches - classification_low_level_searches,
            max_open_size=3,
            unique_constraint_signatures=1,
            duplicate_constraint_signatures=0,
            unique_path_signatures=1,
            duplicate_path_signatures=0,
            classified_conflicts=classified_conflicts,
            classification_low_level_searches=classification_low_level_searches,
            selected_cardinal_conflicts=0,
            selected_semi_cardinal_conflicts=0,
            selected_non_cardinal_conflicts=0,
            combined_low_level_searches=combined_low_level_searches,
        ),
    )


def _write_jsonl(path: Path, records: list[MAPFBenchmarkRunRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(_run_record_to_json(record), sort_keys=True) + "\n")


def _complete_matrix_records() -> list[MAPFBenchmarkRunRecord]:
    instances = (
        _instance("bench_n05_low_000", 5, MAPFInteractionLevel.LOW),
        _instance("bench_n05_medium_000", 5, MAPFInteractionLevel.MEDIUM, conflict_count=1),
        _instance("bench_n10_high_000", 10, MAPFInteractionLevel.HIGH, conflict_count=4),
    )
    records: list[MAPFBenchmarkRunRecord] = []
    for instance in instances:
        records.append(_pp_record(instance.instance_id, agent_count=instance.agent_count))
        records.append(
            _cbs_record(
                instance.instance_id,
                MAPFBenchmarkAlgorithm.CBS_BASIC,
                agent_count=instance.agent_count,
            )
        )
        records.append(
            _cbs_record(
                instance.instance_id,
                MAPFBenchmarkAlgorithm.CBS_CARDINAL_FIRST,
                agent_count=instance.agent_count,
                expanded_ct_nodes=8,
                combined_low_level_searches=20,
                classification_low_level_searches=8,
                classified_conflicts=4,
            )
        )
    return records


def _expected_counts_for_matrix() -> dict[str, dict[str, int]]:
    return {
        MAPFBenchmarkAlgorithm.FIXED_PRIORITY_PP.value: {TERMINATION_SUCCESS: 3},
        MAPFBenchmarkAlgorithm.CBS_BASIC.value: {TERMINATION_SUCCESS: 3},
        MAPFBenchmarkAlgorithm.CBS_CARDINAL_FIRST.value: {TERMINATION_SUCCESS: 3},
        "__overall__": {
            TERMINATION_SUCCESS: 9,
            TERMINATION_EXPANSION_LIMIT: 0,
            TERMINATION_TIME_LIMIT: 0,
            TERMINATION_FAILURE: 0,
            TERMINATION_ERROR: 0,
        },
    }


def test_complete_matrix_validation_passes(tmp_path: Path) -> None:
    instances = (
        _instance("bench_n05_low_000", 5, MAPFInteractionLevel.LOW),
        _instance("bench_n05_medium_000", 5, MAPFInteractionLevel.MEDIUM, conflict_count=1),
        _instance("bench_n10_high_000", 10, MAPFInteractionLevel.HIGH, conflict_count=4),
    )
    manifest = _manifest(*instances)
    records = _complete_matrix_records()

    report = validate_dataset(
        records,
        manifest,
        expected_record_count=9,
        expected_instance_count=3,
        expected_termination_counts=_expected_counts_for_matrix(),
    )
    assert report.passed


def test_duplicate_run_detection_fails(tmp_path: Path) -> None:
    manifest = _manifest(_instance("bench_n05_low_000", 5, MAPFInteractionLevel.LOW))
    records = [
        _pp_record("bench_n05_low_000"),
        _pp_record("bench_n05_low_000"),
        _cbs_record("bench_n05_low_000", MAPFBenchmarkAlgorithm.CBS_BASIC),
        _cbs_record("bench_n05_low_000", MAPFBenchmarkAlgorithm.CBS_CARDINAL_FIRST),
    ]
    report = validate_dataset(
        records,
        manifest,
        expected_record_count=4,
        expected_instance_count=1,
        expected_termination_counts=_expected_counts_for_matrix(),
    )
    assert not report.passed
    assert any("duplicate_keys" in message for message in report.messages)


def test_missing_run_detection_fails() -> None:
    manifest = _manifest(_instance("bench_n05_low_000", 5, MAPFInteractionLevel.LOW))
    records = [
        _pp_record("bench_n05_low_000"),
        _cbs_record("bench_n05_low_000", MAPFBenchmarkAlgorithm.CBS_BASIC),
    ]
    report = validate_dataset(
        records,
        manifest,
        expected_record_count=2,
        expected_instance_count=1,
        expected_termination_counts=_expected_counts_for_matrix(),
    )
    assert not report.passed
    assert any("missing_algorithm_combinations" in message for message in report.messages)


def test_invalid_termination_reason_fails() -> None:
    manifest = _manifest(_instance("bench_n05_low_000", 5, MAPFInteractionLevel.LOW))
    bad_record = MAPFBenchmarkRunRecord(
        instance_id="bench_n05_low_000",
        algorithm=MAPFBenchmarkAlgorithm.FIXED_PRIORITY_PP,
        agent_count=5,
        interaction_level="low",
        success=False,
        termination_reason="mystery_stop",
        execution_time_ms=1000.0,
        soc=None,
        makespan=None,
        remaining_conflicts=None,
        independent_soc=100,
        independent_makespan=20,
        independent_conflict_count=0,
        pp_search_metrics=MAPFPPSearchMetrics(low_level_searches=5, agents_planned=5),
    )

    report = validate_dataset(
        [bad_record],
        manifest,
        expected_record_count=1,
        expected_instance_count=1,
        expected_termination_counts=_expected_counts_for_matrix(),
    )
    assert not report.passed
    assert any("invalid_termination_reasons" in message for message in report.messages)


def test_timeout_stays_distinct_from_failure() -> None:
    manifest = _manifest(_instance("bench_n05_low_000", 5, MAPFInteractionLevel.LOW))
    records = [
        MAPFBenchmarkRunRecord(
            instance_id="bench_n05_low_000",
            algorithm=MAPFBenchmarkAlgorithm.FIXED_PRIORITY_PP,
            agent_count=5,
            interaction_level="low",
            success=False,
            termination_reason=TERMINATION_FAILURE,
            execution_time_ms=1000.0,
            soc=None,
            makespan=None,
            remaining_conflicts=None,
            independent_soc=100,
            independent_makespan=20,
            independent_conflict_count=0,
            pp_search_metrics=MAPFPPSearchMetrics(low_level_searches=5, agents_planned=5),
        ),
        _cbs_record(
            "bench_n05_low_000",
            MAPFBenchmarkAlgorithm.CBS_BASIC,
            success=False,
            termination_reason=TERMINATION_TIME_LIMIT,
            execution_time_ms=180_000.0,
            soc=None,
            makespan=None,
        ),
        _cbs_record(
            "bench_n05_low_000",
            MAPFBenchmarkAlgorithm.CBS_CARDINAL_FIRST,
            success=False,
            termination_reason=TERMINATION_FAILURE,
            execution_time_ms=5000.0,
            soc=None,
            makespan=None,
        ),
    ]

    expected = {
        MAPFBenchmarkAlgorithm.FIXED_PRIORITY_PP.value: {TERMINATION_FAILURE: 1},
        MAPFBenchmarkAlgorithm.CBS_BASIC.value: {TERMINATION_TIME_LIMIT: 1},
        MAPFBenchmarkAlgorithm.CBS_CARDINAL_FIRST.value: {TERMINATION_FAILURE: 1},
        "__overall__": {
            TERMINATION_SUCCESS: 0,
            TERMINATION_EXPANSION_LIMIT: 0,
            TERMINATION_TIME_LIMIT: 1,
            TERMINATION_FAILURE: 2,
            TERMINATION_ERROR: 0,
        },
    }
    report = validate_dataset(
        records,
        manifest,
        expected_record_count=3,
        expected_instance_count=1,
        expected_termination_counts=expected,
    )
    assert report.passed
    assert any("time_limit=1" in message for message in report.messages)
    assert any("failure=2" in message for message in report.messages)


def test_successful_runtime_filtering_excludes_timeouts() -> None:
    records = [
        _pp_record("a", success=True, execution_time_ms=10_000.0),
        _pp_record("b", success=False, termination_reason=TERMINATION_TIME_LIMIT, execution_time_ms=180_000.0),
        _pp_record("c", success=True, execution_time_ms=20_000.0),
    ]
    summary = build_runtime_success_summary(records)
    pp_row = next(row for row in summary if row["algorithm"] == "fixed_priority_pp")
    assert pp_row["n"] == 2
    assert pp_row["mean_ms"] == 15_000.0


def test_paired_comparison_excludes_non_common_success_cases() -> None:
    records = [
        _pp_record("common", soc=110, makespan=22),
        _cbs_record("common", MAPFBenchmarkAlgorithm.CBS_BASIC, soc=100, makespan=20),
        _pp_record("pp_only", soc=120, makespan=24),
        _cbs_record(
            "cbs_only",
            MAPFBenchmarkAlgorithm.CBS_BASIC,
            success=False,
            termination_reason=TERMINATION_TIME_LIMIT,
            soc=None,
            makespan=None,
        ),
    ]
    paired = build_paired_quality_comparison(records)
    pp_basic = [row for row in paired if row["comparison"] == "pp_minus_basic"]
    assert len(pp_basic) == 1
    assert pp_basic[0]["instance_id"] == "common"
    assert pp_basic[0]["pp_minus_basic_soc"] == 10


def test_cardinal_overhead_metrics() -> None:
    records = [
        _cbs_record(
            "inst",
            MAPFBenchmarkAlgorithm.CBS_BASIC,
            execution_time_ms=1000.0,
            expanded_ct_nodes=20,
            combined_low_level_searches=30,
        ),
        _cbs_record(
            "inst",
            MAPFBenchmarkAlgorithm.CBS_CARDINAL_FIRST,
            execution_time_ms=1500.0,
            expanded_ct_nodes=10,
            combined_low_level_searches=45,
            classification_low_level_searches=15,
            classified_conflicts=5,
        ),
    ]
    instance_rows, combined_rows = build_cardinal_overhead_summary(records)
    assert len(instance_rows) == 1
    row = instance_rows[0]
    assert row["runtime_ratio_cardinal_over_basic"] == 1.5
    assert row["ct_expansion_reduction_basic_minus_cardinal"] == 10
    assert row["additional_low_level_searches_cardinal_minus_basic"] == 15
    assert len(combined_rows) == 2


def test_pp_179s_is_within_post_hoc_budget() -> None:
    record = _pp_record("fast", success=True, execution_time_ms=179_000.0)
    assert within_common_budget(record, COMMON_BUDGET_MS)


def test_pp_181s_success_outside_post_hoc_budget() -> None:
    record = _pp_record("slow", success=True, execution_time_ms=181_000.0)
    assert record.success
    assert record.termination_reason == TERMINATION_SUCCESS
    assert not within_common_budget(record, COMMON_BUDGET_MS)


def test_none_metrics_on_timeout_remain_none() -> None:
    record = MAPFBenchmarkRunRecord(
        instance_id="timeout",
        algorithm=MAPFBenchmarkAlgorithm.CBS_BASIC,
        agent_count=5,
        interaction_level="low",
        success=False,
        termination_reason=TERMINATION_TIME_LIMIT,
        execution_time_ms=180_000.0,
        soc=None,
        makespan=None,
        remaining_conflicts=None,
        independent_soc=100,
        independent_makespan=20,
        independent_conflict_count=0,
        cbs_search_metrics=None,
    )
    assert record.soc is None
    assert record.makespan is None
    assert record.cbs_search_metrics is None


def test_validation_failure_writes_only_validation_file(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    instances = (_instance("bench_n05_low_000", 5, MAPFInteractionLevel.LOW),)
    save_benchmark_manifest(_manifest(*instances), manifest_path)

    jsonl_path = tmp_path / "results.jsonl"
    _write_jsonl(jsonl_path, [_pp_record("bench_n05_low_000")])

    output_dir = tmp_path / "analysis"
    result = run_benchmark_analysis(
        AnalysisConfig(
            results_jsonl_path=jsonl_path,
            manifest_path=manifest_path,
            output_dir=output_dir,
            expected_record_count=3,
            expected_instance_count=1,
        )
    )
    assert not result.validation.passed
    assert result.tables_written == ("dataset_validation.txt",)
    assert (output_dir / "dataset_validation.txt").is_file()
    assert not (output_dir / "termination_summary.csv").exists()


def test_count_directional_soc_better_positive_means_right_better() -> None:
    counts = count_directional_soc_better([5])
    assert counts.right_better == 1
    assert counts.left_better == 0
    assert counts.equal == 0


def test_count_directional_soc_better_negative_means_left_better() -> None:
    counts = count_directional_soc_better([-5])
    assert counts.left_better == 1
    assert counts.right_better == 0
    assert counts.equal == 0


def test_count_directional_soc_better_zero_means_equal() -> None:
    counts = count_directional_soc_better([0])
    assert counts.equal == 1
    assert counts.left_better == 0
    assert counts.right_better == 0


def test_pp_vs_basic_quality_summary_directional_counts() -> None:
    paired_rows = [
        {
            "comparison": "pp_minus_basic",
            "instance_id": "a",
            "pp_minus_basic_soc": 10,
            "pp_minus_basic_makespan": 0,
        },
        {
            "comparison": "pp_minus_basic",
            "instance_id": "b",
            "pp_minus_basic_soc": 0,
            "pp_minus_basic_makespan": 0,
        },
        {
            "comparison": "pp_minus_basic",
            "instance_id": "c",
            "pp_minus_basic_soc": -3,
            "pp_minus_basic_makespan": 0,
        },
    ]
    summary = build_pp_vs_basic_quality_summary(paired_rows)[0]
    assert summary["right_better_soc_count"] == 1
    assert summary["equal_soc_count"] == 1
    assert summary["left_better_soc_count"] == 1
