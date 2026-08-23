from __future__ import annotations

import json
from pathlib import Path

import pytest

from pathfinding.src.experiments.mapf_benchmark_execution import TERMINATION_SUCCESS
from pathfinding.src.experiments.mapf_random_k20_analysis import (
    K10_ORDERING_INDICES,
    K20_ORDERING_INDICES,
    SUPPLEMENTAL_ORDERING_INDICES,
    RandomK20AnalysisConfig,
    build_distinct_soc_value_rows,
    build_k10_vs_k20_comparison_rows,
    compute_global_sample_summary,
    compute_instance_sample_metrics,
    run_random_k20_analysis,
    validate_random_k20_datasets,
)
from pathfinding.src.experiments.mapf_random_priority_execution import (
    DEFAULT_RANDOM_PRIORITY_BASE_SEED,
    RandomPriorityRunRecord,
    RandomPriorityRunKey,
)
from pathfinding.tests.unit.test_mapf_random_priority_execution import _instance, _tiny_manifest


def _record(
    *,
    instance_id: str,
    ordering_index: int,
    agent_count: int = 5,
    interaction_level: str = "low",
    soc: int = 100,
    makespan: int = 20,
    agent_order: tuple[int, ...] | None = None,
) -> RandomPriorityRunRecord:
    if agent_order is None:
        agent_order = tuple(range(agent_count))
    return RandomPriorityRunRecord(
        instance_id=instance_id,
        agent_count=agent_count,
        interaction_level=interaction_level,
        ordering_index=ordering_index,
        ordering_seed=1000 + ordering_index,
        agent_order=agent_order,
        base_seed=DEFAULT_RANDOM_PRIORITY_BASE_SEED,
        success=True,
        termination_reason=TERMINATION_SUCCESS,
        execution_time_ms=1.0,
        soc=soc,
        makespan=makespan,
        conflict_count=0,
        pp_search_metrics=None,
    )


def _build_valid_partitions(
    instance_id: str = "bench_n05_low_000",
    *,
    agent_count: int = 5,
    k10_socs: list[int] | None = None,
    supplemental_socs: list[int] | None = None,
) -> tuple[
    dict[RandomPriorityRunKey, RandomPriorityRunRecord],
    dict[RandomPriorityRunKey, RandomPriorityRunRecord],
]:
    if k10_socs is None:
        k10_socs = [100] * len(K10_ORDERING_INDICES)
    if supplemental_socs is None:
        supplemental_socs = [100] * len(SUPPLEMENTAL_ORDERING_INDICES)

    k10: dict[RandomPriorityRunKey, RandomPriorityRunRecord] = {}
    supplemental: dict[RandomPriorityRunKey, RandomPriorityRunRecord] = {}

    for index, soc in zip(K10_ORDERING_INDICES, k10_socs, strict=True):
        record = _record(
            instance_id=instance_id,
            ordering_index=index,
            agent_count=agent_count,
            soc=soc,
            agent_order=tuple(
                (index + agent_id) % agent_count for agent_id in range(agent_count)
            ),
        )
        k10[RandomPriorityRunKey(instance_id, index)] = record

    for offset, (index, soc) in enumerate(
        zip(SUPPLEMENTAL_ORDERING_INDICES, supplemental_socs, strict=True)
    ):
        record = _record(
            instance_id=instance_id,
            ordering_index=index,
            agent_count=agent_count,
            soc=soc,
            agent_order=tuple(
                (offset + agent_id) % agent_count for agent_id in range(agent_count)
            ),
        )
        supplemental[RandomPriorityRunKey(instance_id, index)] = record

    return k10, supplemental


def test_correct_partition_by_ordering_index() -> None:
    k10, supplemental = _build_valid_partitions()
    assert {key.ordering_index for key in k10} == set(K10_ORDERING_INDICES)
    assert {key.ordering_index for key in supplemental} == set(
        SUPPLEMENTAL_ORDERING_INDICES
    )
    combined = dict(k10)
    combined.update(supplemental)
    assert {key.ordering_index for key in combined} == set(K20_ORDERING_INDICES)


def test_correct_per_instance_soc_range() -> None:
    k10, supplemental = _build_valid_partitions(
        k10_socs=[100, 100, 110, 100, 100, 100, 100, 100, 100, 100],
        supplemental_socs=[100] * len(SUPPLEMENTAL_ORDERING_INDICES),
    )
    metrics = compute_instance_sample_metrics(
        instance_id="bench_n05_low_000",
        agent_count=5,
        interaction_level="low",
        records=tuple(k10.values()),
    )
    assert metrics.soc_min == 100
    assert metrics.soc_max == 110
    assert metrics.soc_range == 10
    assert metrics.unique_soc_count == 2


def test_correct_sensitivity_classification() -> None:
    insensitive = compute_instance_sample_metrics(
        instance_id="bench_n05_low_000",
        agent_count=5,
        interaction_level="low",
        records=(_record(instance_id="bench_n05_low_000", ordering_index=0),),
    )
    sensitive = compute_instance_sample_metrics(
        instance_id="bench_n05_low_000",
        agent_count=5,
        interaction_level="low",
        records=(
            _record(instance_id="bench_n05_low_000", ordering_index=0, soc=100),
            _record(instance_id="bench_n05_low_000", ordering_index=1, soc=105),
        ),
    )
    assert not insensitive.soc_sensitive
    assert not insensitive.makespan_sensitive
    assert sensitive.soc_sensitive
    assert not sensitive.makespan_sensitive


def test_correct_distinct_soc_comparison() -> None:
    k10, supplemental = _build_valid_partitions(
        k10_socs=[100, 100, 110, 100, 100, 100, 100, 100, 100, 100],
        supplemental_socs=[105, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    )
    combined = dict(k10)
    combined.update(supplemental)
    rows = build_k10_vs_k20_comparison_rows(
        k10_by_instance={"bench_n05_low_000": tuple(k10.values())},
        k20_by_instance={"bench_n05_low_000": tuple(combined.values())},
    )
    assert len(rows) == 1
    assert json.loads(rows[0]["new_soc_values_found_in_10_19"]) == [105]
    assert rows[0]["K10_soc_range"] == 10
    assert rows[0]["K20_soc_range"] == 10
    assert rows[0]["new_internal_soc_without_range_increase"] is True


def test_duplicate_key_detection_fails_validation() -> None:
    manifest = _tiny_manifest(_instance("bench_n05_low_000", 5))
    k10, supplemental = _build_valid_partitions()
    duplicate_key = next(iter(k10))
    supplemental[duplicate_key] = k10[duplicate_key]

    report = validate_random_k20_datasets(
        manifest=manifest,
        k10_records=k10,
        supplemental_records=supplemental,
    )
    assert not report.passed
    assert any("disjoint" in message for message in report.messages)


def test_missing_ordering_index_detection() -> None:
    manifest = _tiny_manifest(_instance("bench_n05_low_000", 5))
    k10, supplemental = _build_valid_partitions()
    del k10[RandomPriorityRunKey("bench_n05_low_000", 9)]

    report = validate_random_k20_datasets(
        manifest=manifest,
        k10_records=k10,
        supplemental_records=supplemental,
    )
    assert not report.passed
    assert any("ordering indices exactly" in message for message in report.messages)


def test_mismatch_between_historical_and_supplemental_instance_sets() -> None:
    manifest = _tiny_manifest(_instance("bench_n05_low_000", 5))
    k10, supplemental = _build_valid_partitions(instance_id="bench_n05_low_000")
    extra = _record(instance_id="bench_n05_low_001", ordering_index=10, soc=100)
    supplemental[RandomPriorityRunKey("bench_n05_low_001", 10)] = extra

    report = validate_random_k20_datasets(
        manifest=manifest,
        k10_records=k10,
        supplemental_records=supplemental,
    )
    assert not report.passed
    assert any(
        "unexpected instance IDs" in message or "missing instance IDs" in message
        for message in report.messages
    )


def test_correct_global_aggregation() -> None:
    metrics = {
        "a": compute_instance_sample_metrics(
            instance_id="a",
            agent_count=5,
            interaction_level="low",
            records=(
                _record(instance_id="a", ordering_index=0, soc=100),
                _record(instance_id="a", ordering_index=1, soc=110),
            ),
        ),
        "b": compute_instance_sample_metrics(
            instance_id="b",
            agent_count=5,
            interaction_level="low",
            records=(_record(instance_id="b", ordering_index=0, soc=200),),
        ),
    }
    summary = compute_global_sample_summary(metrics)
    assert summary["soc_sensitive_count"] == 1
    assert summary["makespan_sensitive_count"] == 0
    assert summary["max_soc_range"] == 10
    assert summary["mean_soc_range"] == 5.0
    assert summary["median_soc_range"] == 5.0


def test_distinct_soc_rows_for_sensitive_instance() -> None:
    k10, supplemental = _build_valid_partitions(
        k10_socs=[100, 110, 100, 100, 100, 100, 100, 100, 100, 100],
        supplemental_socs=[105, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    )
    combined = dict(k10)
    combined.update(supplemental)
    comparison = build_k10_vs_k20_comparison_rows(
        k10_by_instance={"bench_n05_low_000": tuple(k10.values())},
        k20_by_instance={"bench_n05_low_000": tuple(combined.values())},
    )
    rows = build_distinct_soc_value_rows(
        k10_by_instance={"bench_n05_low_000": tuple(k10.values())},
        k20_by_instance={"bench_n05_low_000": tuple(combined.values())},
        comparison_rows=comparison,
    )
    assert len(rows) == 1
    assert json.loads(rows[0]["k10_distinct_socs_0_9"]) == [100, 110]
    assert json.loads(rows[0]["supplemental_distinct_socs_10_19"]) == [100, 105]
    assert json.loads(rows[0]["k20_distinct_socs_0_19"]) == [100, 105, 110]


@pytest.mark.skipif(
    not Path("pathfinding/results/mapf_random_priority_pilot/results_details.jsonl").is_file()
    or not Path("pathfinding/results/mapf_random_k20_supplemental/results_details.jsonl").is_file(),
    reason="Full Random K=20 result files not present",
)
def test_full_dataset_analysis_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    result = run_random_k20_analysis(
        RandomK20AnalysisConfig(
            historical_k10_jsonl=repo_root
            / "pathfinding/results/mapf_random_priority_pilot/results_details.jsonl",
            supplemental_jsonl=repo_root
            / "pathfinding/results/mapf_random_k20_supplemental/results_details.jsonl",
            manifest_path=repo_root
            / "pathfinding/results/mapf_benchmarks/AR0204SR_manifest.json",
            output_dir=tmp_path / "analysis",
        )
    )
    assert result.validation.passed
    assert result.validation.combined_record_count == 540
    assert (tmp_path / "analysis" / "analysis_summary.md").is_file()
