from __future__ import annotations

import json
from pathlib import Path

import pytest

from pathfinding.src.experiments.mapf8_cross_map_analysis import (
    EXPECTED_COMBINED_INSTANCES,
    EXPECTED_COMBINED_RECORDS,
    EXPECTED_INSTANCES_PER_MAP,
    EXPECTED_RECORDS_PER_MAP,
    MAPF8_STRATEGIES,
    build_agent_count_sensitivity_summary,
    build_catalogue_summary_rows,
    build_instance_level_summary_from_records,
    build_interaction_sensitivity_summary,
    build_pairwise_summary_for_catalogue,
    combined_spf_cd_order_counts,
    compute_four_strategy_makespan_range,
    compute_four_strategy_soc_range,
    compute_pairwise_soc_diff,
    default_mapf8_config,
    is_makespan_sensitive,
    is_soc_sensitive,
    load_cross_map_results,
    order_diff_without_soc_diff,
    run_mapf8_cross_map_analysis,
    spf_cd_same_order,
    validate_mapf8_combined_dataset,
    validate_mapf8_map_dataset,
)
from pathfinding.src.experiments.mapf_benchmark_instances import load_benchmark_manifest
from pathfinding.src.experiments.mapf_cross_map_priority_execution import (
    CrossMapPriorityRunRecord,
    CrossMapPriorityStrategy,
    FROZEN_MANIFEST_SHA256,
    MAPF8_CROSS_MAP_SEEDS,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sample_instance_row(
    *,
    spf_soc: int = 100,
    cdf_h_soc: int = 100,
    cdf_l_soc: int = 100,
    spf_cd_soc: int = 100,
) -> dict[str, object]:
    return {
        "spf_soc": spf_soc,
        "cdf_h_soc": cdf_h_soc,
        "cdf_l_soc": cdf_l_soc,
        "spf_cd_soc": spf_cd_soc,
        "spf_makespan": 20,
        "cdf_h_makespan": 20,
        "cdf_l_makespan": 20,
        "spf_cd_makespan": 20,
    }


def test_four_strategy_soc_range_calculation() -> None:
    row = _sample_instance_row(spf_soc=100, cdf_h_soc=110, cdf_l_soc=105, spf_cd_soc=100)
    assert compute_four_strategy_soc_range(row) == 10


def test_four_strategy_makespan_range() -> None:
    row = {
        "spf_makespan": 20,
        "cdf_h_makespan": 22,
        "cdf_l_makespan": 20,
        "spf_cd_makespan": 21,
    }
    assert compute_four_strategy_makespan_range(row) == 2


def test_sensitivity_flags() -> None:
    equal = _sample_instance_row()
    diff = _sample_instance_row(cdf_l_soc=95)
    assert not is_soc_sensitive(equal)
    assert is_soc_sensitive(diff)
    assert not is_makespan_sensitive(equal)


def test_pairwise_left_right_sign_convention() -> None:
    assert compute_pairwise_soc_diff(90, 100) < 0
    assert compute_pairwise_soc_diff(100, 100) == 0
    assert compute_pairwise_soc_diff(110, 100) > 0


def test_spf_cd_order_equality() -> None:
    assert spf_cd_same_order((0, 1, 2), (0, 1, 2)) is True
    assert spf_cd_same_order((0, 1, 2), (1, 0, 2)) is False
    assert spf_cd_same_order(None, (0, 1, 2)) is None


def test_order_difference_without_soc_difference() -> None:
    assert order_diff_without_soc_diff(same_order=False, soc_equal=True) is True
    assert order_diff_without_soc_diff(same_order=False, soc_equal=False) is False
    assert order_diff_without_soc_diff(same_order=True, soc_equal=True) is False


def test_sensitive_counts_by_interaction() -> None:
    rows = [
        {
            "interaction_level": "high",
            "soc_range": 5,
            "soc_sensitive": True,
            "makespan_sensitive": False,
            "agent_count": 5,
        },
        {
            "interaction_level": "low",
            "soc_range": 0,
            "soc_sensitive": False,
            "makespan_sensitive": False,
            "agent_count": 5,
        },
    ]
    summary = build_interaction_sensitivity_summary(catalogue="test", instance_rows=rows)
    high = next(row for row in summary if row["interaction_level"] == "high")
    assert high["soc_sensitive_count"] == 1
    assert high["instance_count"] == 1


def test_sensitive_counts_by_agent_count() -> None:
    rows = [
        {
            "agent_count": 10,
            "soc_range": 3,
            "soc_sensitive": True,
            "makespan_sensitive": False,
            "interaction_level": "medium",
        },
        {
            "agent_count": 5,
            "soc_range": 0,
            "soc_sensitive": False,
            "makespan_sensitive": False,
            "interaction_level": "low",
        },
    ]
    summary = build_agent_count_sensitivity_summary(catalogue="test", instance_rows=rows)
    n10 = next(row for row in summary if row["agent_count"] == 10)
    assert n10["soc_sensitive_count"] == 1


def test_zero_range_instances_remain_in_denominators() -> None:
    rows = [
        {
            "soc_range": 0,
            "makespan_range": 0,
            "soc_sensitive": False,
            "makespan_sensitive": False,
            "mapf7_three_strategy_soc_sensitive": False,
        },
        {
            "soc_range": 4,
            "makespan_range": 0,
            "soc_sensitive": True,
            "makespan_sensitive": False,
            "mapf7_three_strategy_soc_sensitive": True,
        },
    ]
    summary = build_catalogue_summary_rows(catalogue="test", instance_rows=rows)
    assert summary["instance_count"] == 2
    assert summary["soc_sensitive_count"] == 1
    assert summary["soc_sensitive_pct"] == "50.0%"


def test_pooled_cross_map_aggregation() -> None:
    config = default_mapf8_config(_repo_root())
    result = run_mapf8_cross_map_analysis(config)
    assert result.validation.passed
    assert len(result.pooled_instance_rows) == EXPECTED_COMBINED_INSTANCES
    assert len(result.ar0400_instance_rows) == EXPECTED_INSTANCES_PER_MAP
    assert len(result.ar0307_instance_rows) == EXPECTED_INSTANCES_PER_MAP


def test_real_dataset_record_counts() -> None:
    config = default_mapf8_config(_repo_root())
    ar0400 = load_cross_map_results(config.ar0400_csv)
    ar0307 = load_cross_map_results(config.ar0307_csv)
    assert len(ar0400) == EXPECTED_RECORDS_PER_MAP
    assert len(ar0307) == EXPECTED_RECORDS_PER_MAP
    assert len(ar0400) + len(ar0307) == EXPECTED_COMBINED_RECORDS


def test_real_dataset_validation_passes() -> None:
    config = default_mapf8_config(_repo_root())
    ar0400 = load_cross_map_results(config.ar0400_csv)
    ar0307 = load_cross_map_results(config.ar0307_csv)
    ar0400_manifest = load_benchmark_manifest(config.ar0400_manifest)
    ar0307_manifest = load_benchmark_manifest(config.ar0307_manifest)

    ar0400_report = validate_mapf8_map_dataset(
        map_name="AR0400SR",
        records=ar0400,
        manifest=ar0400_manifest,
        expected_manifest_sha256=FROZEN_MANIFEST_SHA256["AR0400SR"],
        expected_seed=MAPF8_CROSS_MAP_SEEDS["AR0400SR"],
    )
    ar0307_report = validate_mapf8_map_dataset(
        map_name="AR0307SR",
        records=ar0307,
        manifest=ar0307_manifest,
        expected_manifest_sha256=FROZEN_MANIFEST_SHA256["AR0307SR"],
        expected_seed=MAPF8_CROSS_MAP_SEEDS["AR0307SR"],
    )
    combined_report = validate_mapf8_combined_dataset(
        ar0400_records=ar0400,
        ar0307_records=ar0307,
    )
    assert ar0400_report.passed
    assert ar0307_report.passed
    assert combined_report.passed


def test_duplicate_run_key_detection() -> None:
    record = CrossMapPriorityRunRecord(
        map_name="AR0400SR",
        manifest_sha256=FROZEN_MANIFEST_SHA256["AR0400SR"],
        instance_id="AR0400SR_n05_low_000",
        agent_count=5,
        interaction_level="low",
        strategy=CrossMapPriorityStrategy.SPF,
        agent_order=(0, 1, 2, 3, 4),
        original_index_order=(0, 1, 2, 3, 4),
        agent_degrees=None,
        independent_conflict_count=0,
        independent_conflict_pair_count=0,
        ordering_low_level_searches=5,
        incident_conflict_counts=None,
        success=True,
        termination_reason="success",
        ordering_time_ms=1.0,
        pp_time_ms=2.0,
        total_time_ms=3.0,
        soc=100,
        makespan=20,
        conflict_count=0,
        catalogue_seed=2028,
    )
    manifest = load_benchmark_manifest(
        _repo_root() / "pathfinding/results/mapf_benchmarks/AR0400SR_manifest.json"
    )
    report = validate_mapf8_map_dataset(
        map_name="AR0400SR",
        records=(record, record),
        manifest=manifest,
        expected_manifest_sha256=FROZEN_MANIFEST_SHA256["AR0400SR"],
        expected_seed=MAPF8_CROSS_MAP_SEEDS["AR0400SR"],
    )
    assert not report.passed


def test_missing_strategy_detection() -> None:
    record = CrossMapPriorityRunRecord(
        map_name="AR0400SR",
        manifest_sha256="abc",
        instance_id="AR0400SR_n05_low_000",
        agent_count=5,
        interaction_level="low",
        strategy=CrossMapPriorityStrategy.SPF,
        agent_order=(0, 1, 2, 3, 4),
        original_index_order=(0, 1, 2, 3, 4),
        agent_degrees=None,
        independent_conflict_count=0,
        independent_conflict_pair_count=0,
        ordering_low_level_searches=5,
        incident_conflict_counts=None,
        success=True,
        termination_reason="success",
        ordering_time_ms=1.0,
        pp_time_ms=2.0,
        total_time_ms=3.0,
        soc=100,
        makespan=20,
        conflict_count=0,
        catalogue_seed=2028,
    )
    with pytest.raises(ValueError, match="missing strategies"):
        build_instance_level_summary_from_records(map_name="AR0400SR", records=[record])


def test_wrong_manifest_hash_detection() -> None:
    config = default_mapf8_config(_repo_root())
    ar0400 = load_cross_map_results(config.ar0400_csv)
    manifest = load_benchmark_manifest(config.ar0400_manifest)
    report = validate_mapf8_map_dataset(
        map_name="AR0400SR",
        records=ar0400,
        manifest=manifest,
        expected_manifest_sha256="0" * 64,
        expected_seed=MAPF8_CROSS_MAP_SEEDS["AR0400SR"],
    )
    assert not report.passed


def test_historical_catalogue_label_separation() -> None:
    config = default_mapf8_config(_repo_root())
    result = run_mapf8_cross_map_analysis(config)
    historical_path = result.output_dir / "historical_comparison.csv"
    assert historical_path.is_file()
    text = historical_path.read_text(encoding="utf-8")
    assert "AR0204SR_primary" in text
    assert "AR0204SR_held_out" in text
    assert "AR0400SR" in text
    assert "AR0307SR" in text


def test_pairwise_summary_uses_left_minus_right() -> None:
    rows = [
        {
            "instance_id": "a",
            "agent_count": 5,
            "interaction_level": "low",
            "spf_soc": 100,
            "cdf_h_soc": 90,
            "cdf_l_soc": 100,
            "spf_cd_soc": 100,
        }
    ]
    summary = build_pairwise_summary_for_catalogue(catalogue="test", instance_rows=rows)
    cdf_h_vs_spf = next(row for row in summary if row["comparison"] == "cdf_h_vs_spf")
    assert cdf_h_vs_spf["left_better_count"] == 1
    assert cdf_h_vs_spf["equal_count"] == 0
    assert cdf_h_vs_spf["right_better_count"] == 0


def test_strategy_set_exactly_four() -> None:
    config = default_mapf8_config(_repo_root())
    records = load_cross_map_results(config.ar0400_csv)
    observed = {record.strategy.value for record in records}
    assert observed == set(MAPF8_STRATEGIES)


def test_thesis_table1_has_no_duplicate_catalogue_rows() -> None:
    config = default_mapf8_config(_repo_root())
    result = run_mapf8_cross_map_analysis(config)
    thesis_text = (result.output_dir / "thesis_tables.md").read_text(encoding="utf-8")
    table1_section = thesis_text.split("## Table 2")[0]
    data_rows = [
        line
        for line in table1_section.splitlines()
        if line.startswith("|") and not line.startswith("|---") and "Catalogue" not in line
    ]
    catalogue_names = [line.split("|")[1].strip() for line in data_rows]
    assert len(catalogue_names) == len(set(catalogue_names))
    assert catalogue_names.count("AR0400SR") == 1
    assert catalogue_names.count("AR0307SR") == 1
    assert "AR0204SR primary" in catalogue_names
    assert "AR0204SR held-out" in catalogue_names


def test_combined_spf_cd_denominator_is_108() -> None:
    config = default_mapf8_config(_repo_root())
    result = run_mapf8_cross_map_analysis(config)
    historical = (
        result.output_dir / "historical_comparison.csv"
    ).read_text(encoding="utf-8")
    spf_cd_text = (result.output_dir / "spf_cd_order_summary.csv").read_text(
        encoding="utf-8"
    )
    pooled_aggregate_line = next(
        line for line in spf_cd_text.splitlines() if ",aggregate,54," in line
    )
    pooled_same = int(pooled_aggregate_line.split(",")[3])
    primary_identical = int(
        next(
            line.split(",")[-2]
            for line in historical.splitlines()
            if line.startswith("AR0204SR_primary")
        )
    )
    heldout_identical = int(
        next(
            line.split(",")[-2]
            for line in historical.splitlines()
            if line.startswith("AR0204SR_held_out")
        )
    )
    counts = combined_spf_cd_order_counts(
        historical_rows=[
            {
                "catalogue": "AR0204SR_primary",
                "spf_cd_identical_ordering": primary_identical,
            },
            {
                "catalogue": "AR0204SR_held_out",
                "spf_cd_identical_ordering": heldout_identical,
            },
        ],
        pooled_spf_cd={"same_order_count": pooled_same, "instance_count": 54},
    )
    assert counts["combined_total"] == 108
    assert counts["combined_same"] == 108
    thesis_text = (result.output_dir / "thesis_tables.md").read_text(encoding="utf-8")
    assert "108/108" in thesis_text


def test_thesis_table2_header_has_no_broken_markdown_separator() -> None:
    config = default_mapf8_config(_repo_root())
    result = run_mapf8_cross_map_analysis(config)
    thesis_text = (result.output_dir / "thesis_tables.md").read_text(encoding="utf-8")
    header_line = next(
        line for line in thesis_text.splitlines() if line.startswith("| Comparison |")
    )
    assert "Max abs(diff)" in header_line
    assert "|abs diff|" not in header_line


def test_rq86_wording_remains_non_causal() -> None:
    config = default_mapf8_config(_repo_root())
    result = run_mapf8_cross_map_analysis(config)
    rq_text = (result.output_dir / "research_questions.md").read_text(encoding="utf-8")
    rq86 = rq_text.split("## RQ8.6")[1].split("##")[0]
    assert "Descriptive differences were observed" in rq86
    assert "does not support causal or predictive claims" in rq86
    assert "predicts sensitivity" not in rq86.lower()
    assert "causes sensitivity" not in rq86.lower()
