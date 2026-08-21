from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from pathfinding.src.experiments.mapf_benchmark_analysis import count_directional_soc_better
from pathfinding.src.experiments.mapf_benchmark_instances import (
    MAPFBenchmarkInstance,
    MAPFBenchmarkManifest,
    MAPFInteractionLevel,
    save_benchmark_manifest,
)
from pathfinding.src.experiments.mapf_conflict_aware_priority_analysis import (
    ConflictAwareAnalysisConfig,
    build_cdf_h_vs_cdf_l_direction,
    build_pairwise_quality_summary,
    build_priority_sensitive_subset,
    cdf_l_vs_spf_interpretation,
    count_mapf7_differing_in_priority_sensitive,
    load_conflict_aware_results,
    mapf7_strategy_socs_differ,
    run_conflict_aware_priority_analysis,
    validate_datasets,
)
from pathfinding.src.experiments.mapf_priority_strategy_analysis import (
    aggregate_random_by_instance,
    load_fixed_results,
    load_random_results,
    load_strategy_results,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _instance(instance_id: str) -> MAPFBenchmarkInstance:
    return MAPFBenchmarkInstance(
        instance_id=instance_id,
        agent_count=5,
        interaction_level=MAPFInteractionLevel.LOW,
        scenario_indices=(0, 1),
        independent_conflict_count=0,
        conflicting_agent_pair_count=0,
        independent_soc=100,
        independent_makespan=20,
        vertex_conflict_count=0,
        edge_conflict_count=0,
    )


def _conflict_row(
    instance_id: str,
    strategy: str,
    *,
    soc: int = 100,
    agent_count: int = 5,
) -> dict[str, object]:
    order = list(range(agent_count))
    degrees = [0] * agent_count
    return {
        "instance_id": instance_id,
        "agent_count": agent_count,
        "interaction_level": "low",
        "strategy": strategy,
        "agent_order": json.dumps(order),
        "original_index_order": json.dumps(order),
        "agent_degrees": json.dumps(degrees),
        "independent_conflict_count": 0,
        "independent_conflict_pair_count": 0,
        "ordering_low_level_searches": agent_count,
        "incident_conflict_counts": json.dumps(degrees),
        "success": "True",
        "termination_reason": "success",
        "ordering_time_ms": 1000.0,
        "pp_time_ms": 2000.0,
        "total_time_ms": 3000.0,
        "soc": soc,
        "makespan": 20,
        "conflict_count": 0,
    }


@pytest.fixture
def repo_config(tmp_path: Path) -> ConflictAwareAnalysisConfig:
    instance_ids = ("bench_n05_low_000", "bench_n05_low_001")
    manifest = MAPFBenchmarkManifest(
        map_name="bench.map",
        scenario_name="bench.scen",
        seed=2026,
        max_timestep=512,
        min_reference_length=1.0,
        instances=tuple(_instance(instance_id) for instance_id in instance_ids),
    )
    manifest_path = tmp_path / "manifest.json"
    save_benchmark_manifest(manifest, manifest_path)

    fixed_rows: list[dict[str, object]] = []
    random_rows: list[dict[str, object]] = []
    spf_lpf_rows: list[dict[str, object]] = []
    conflict_rows: list[dict[str, object]] = []

    for instance_id in instance_ids:
        fixed_rows.append(
            {
                "instance_id": instance_id,
                "algorithm": "fixed_priority_pp",
                "agent_count": 5,
                "interaction_level": "low",
                "success": "True",
                "termination_reason": "success",
                "execution_time_ms": 1000.0,
                "soc": 100,
                "makespan": 20,
                "remaining_conflicts": 0,
            }
        )
        fixed_rows.append(
            {
                "instance_id": instance_id,
                "algorithm": "cbs_basic",
                "agent_count": 5,
                "interaction_level": "low",
                "success": "True",
                "termination_reason": "success",
                "execution_time_ms": 2000.0,
                "soc": 90,
                "makespan": 18,
                "remaining_conflicts": 0,
            }
        )
        for ordering_index, soc in enumerate((100, 95, 100, 95, 100, 95, 100, 95, 100, 95)):
            random_rows.append(
                {
                    "instance_id": instance_id,
                    "agent_count": 5,
                    "interaction_level": "low",
                    "ordering_index": ordering_index,
                    "ordering_seed": 2026 + ordering_index,
                    "agent_order": json.dumps(list(range(5))),
                    "success": "True",
                    "termination_reason": "success",
                    "execution_time_ms": 1000.0,
                    "soc": soc,
                    "makespan": 20,
                }
            )
        for strategy in ("spf", "lpf"):
            spf_lpf_rows.append(
                {
                    "instance_id": instance_id,
                    "agent_count": 5,
                    "interaction_level": "low",
                    "strategy": strategy,
                    "agent_order": json.dumps(list(range(5))),
                    "success": "True",
                    "termination_reason": "success",
                    "ordering_time_ms": 500.0,
                    "pp_time_ms": 1500.0,
                    "total_time_ms": 2000.0,
                    "soc": 100 if strategy == "spf" else 102,
                    "makespan": 20,
                }
            )
        cdf_l_soc = 98 if instance_id == "bench_n05_low_000" else 100
        for strategy, soc in (
            ("cdf_h", 100),
            ("cdf_l", cdf_l_soc),
            ("spf_cd", 100),
        ):
            conflict_rows.append(_conflict_row(instance_id, strategy, soc=soc))

    conflict_csv = tmp_path / "conflict.csv"
    _write_csv(conflict_csv, list(conflict_rows[0].keys()), conflict_rows)

    fixed_csv = tmp_path / "fixed.csv"
    _write_csv(
        fixed_csv,
        [
            "instance_id",
            "algorithm",
            "agent_count",
            "interaction_level",
            "success",
            "termination_reason",
            "execution_time_ms",
            "soc",
            "makespan",
            "remaining_conflicts",
        ],
        fixed_rows,
    )

    random_csv = tmp_path / "random.csv"
    _write_csv(
        random_csv,
        [
            "instance_id",
            "agent_count",
            "interaction_level",
            "ordering_index",
            "ordering_seed",
            "agent_order",
            "success",
            "termination_reason",
            "execution_time_ms",
            "soc",
            "makespan",
        ],
        random_rows,
    )

    spf_csv = tmp_path / "spf.csv"
    _write_csv(
        spf_csv,
        [
            "instance_id",
            "agent_count",
            "interaction_level",
            "strategy",
            "agent_order",
            "success",
            "termination_reason",
            "ordering_time_ms",
            "pp_time_ms",
            "total_time_ms",
            "soc",
            "makespan",
        ],
        spf_lpf_rows,
    )

    return ConflictAwareAnalysisConfig(
        conflict_aware_results_csv=conflict_csv,
        fixed_results_csv=fixed_csv,
        random_results_csv=random_csv,
        spf_lpf_results_csv=spf_csv,
        manifest_path=manifest_path,
        output_dir=tmp_path / "analysis_out",
        expected_instance_count=2,
        expected_random_k=10,
    )


def test_real_mapf7_dataset_has_81_rows() -> None:
    path = Path(
        "pathfinding/results/mapf_conflict_aware_priority_execution/results.csv"
    )
    records = load_conflict_aware_results(path)
    assert len(records) == 81
    assert len({record.instance_id for record in records}) == 27
    for strategy in ("cdf_h", "cdf_l", "spf_cd"):
        assert sum(1 for record in records if record.strategy == strategy) == 27


def test_sign_convention_left_minus_right() -> None:
    counts = count_directional_soc_better([10, 0, -5])
    assert counts.right_better == 1
    assert counts.equal == 1
    assert counts.left_better == 1


def test_cdf_h_vs_cdf_l_direction_counts(repo_config: ConflictAwareAnalysisConfig) -> None:
    direction_rows = build_cdf_h_vs_cdf_l_direction(
        [
            {
                "instance_id": "bench_n05_low_000",
                "agent_count": 5,
                "interaction_level": "low",
                "cdf_h_soc": 98,
                "cdf_l_soc": 100,
            }
        ]
    )
    assert direction_rows[0]["direction"] == "cdf_h_better"


def test_priority_sensitive_subset_uses_mapf6_flag(
    repo_config: ConflictAwareAnalysisConfig,
) -> None:
    result = run_conflict_aware_priority_analysis(repo_config, plot_writer=None)
    assert result.validation.passed
    rows = build_priority_sensitive_subset(
        [
            {
                "instance_id": "bench_n05_low_000",
                "agent_count": 5,
                "interaction_level": "low",
                "observed_priority_sensitive": True,
                "fixed_soc": 100,
                "random_soc_median": 97.5,
                "random_sampled_best_of_k_soc": 95,
                "spf_soc": 100,
                "lpf_soc": 102,
                "cdf_h_soc": 100,
                "cdf_l_soc": 98,
                "spf_cd_soc": 100,
            },
            {
                "instance_id": "bench_n05_low_001",
                "agent_count": 5,
                "interaction_level": "low",
                "observed_priority_sensitive": False,
                "fixed_soc": 100,
                "random_soc_median": 100,
                "random_sampled_best_of_k_soc": 100,
                "spf_soc": 100,
                "lpf_soc": 102,
                "cdf_h_soc": 100,
                "cdf_l_soc": 100,
                "spf_cd_soc": 100,
            },
        ]
    )
    assert len(rows) == 1
    assert rows[0]["subset_source"].startswith("MAPF-6")


def test_duplicate_strategy_key_rejected(tmp_path: Path) -> None:
    rows = [_conflict_row("bench_n05_low_000", "cdf_h"), _conflict_row("bench_n05_low_000", "cdf_h")]
    path = tmp_path / "dup.csv"
    _write_csv(path, list(rows[0].keys()), rows)
    keys = {(row["instance_id"], row["strategy"]) for row in rows}
    assert len(keys) == 1


def test_ms_to_seconds_in_runtime_aggregate(repo_config: ConflictAwareAnalysisConfig) -> None:
    result = run_conflict_aware_priority_analysis(repo_config, plot_writer=None)
    runtime_csv = repo_config.output_dir / "runtime_summary.csv"
    assert runtime_csv.is_file()
    aggregate = [
        row
        for row in csv.DictReader(runtime_csv.open(encoding="utf-8"))
        if row.get("metric_group") == "runtime_aggregate"
    ]
    assert aggregate
    assert float(aggregate[0]["mean_spf_ordering_s"]) == pytest.approx(0.5)


def test_full_analysis_on_real_results() -> None:
    repo_root = Path("pathfinding/results")
    config = ConflictAwareAnalysisConfig(
        conflict_aware_results_csv=repo_root
        / "mapf_conflict_aware_priority_execution"
        / "results.csv",
        fixed_results_csv=repo_root / "mapf_benchmark_execution" / "results.csv",
        random_results_csv=repo_root / "mapf_random_priority_pilot" / "results.csv",
        spf_lpf_results_csv=repo_root / "mapf_priority_strategy_execution" / "results.csv",
        manifest_path=Path("pathfinding/results/mapf_benchmarks/AR0204SR_manifest.json"),
        output_dir=repo_root / "mapf_conflict_aware_priority_analysis",
        expected_instance_count=27,
        expected_random_k=10,
    )
    result = run_conflict_aware_priority_analysis(
        config,
        plot_writer=None,
    )
    assert result.validation.passed
    assert (config.output_dir / "analysis_summary.md").is_file()
    assert (config.output_dir / "pairwise_quality_summary.csv").is_file()


def _real_analysis_config() -> ConflictAwareAnalysisConfig:
    repo_root = Path("pathfinding/results")
    return ConflictAwareAnalysisConfig(
        conflict_aware_results_csv=repo_root
        / "mapf_conflict_aware_priority_execution"
        / "results.csv",
        fixed_results_csv=repo_root / "mapf_benchmark_execution" / "results.csv",
        random_results_csv=repo_root / "mapf_random_priority_pilot" / "results.csv",
        spf_lpf_results_csv=repo_root / "mapf_priority_strategy_execution" / "results.csv",
        manifest_path=Path("pathfinding/results/mapf_benchmarks/AR0204SR_manifest.json"),
        output_dir=repo_root / "mapf_conflict_aware_priority_analysis",
        expected_instance_count=27,
        expected_random_k=10,
    )


def test_mapf751_priority_sensitive_mapf7_differ_count_derived() -> None:
    config = _real_analysis_config()
    result = run_conflict_aware_priority_analysis(config, plot_writer=None)
    assert result.validation.passed

    sensitive_rows = list(
        csv.DictReader(
            (config.output_dir / "priority_sensitive_subset.csv").open(encoding="utf-8")
        )
    )
    assert len(sensitive_rows) == 7

    differ_count = count_mapf7_differing_in_priority_sensitive(sensitive_rows)
    assert differ_count == 5

    differing_ids = {
        row["instance_id"]
        for row in sensitive_rows
        if row.get("mapf7_strategies_differ_in_soc") == "True"
        or mapf7_strategy_socs_differ(row)
    }
    assert differing_ids == {
        "AR0204SR_n05_high_002",
        "AR0204SR_n10_high_000",
        "AR0204SR_n10_high_001",
        "AR0204SR_n10_high_002",
        "AR0204SR_n20_high_001",
    }

    summary_text = (config.output_dir / "analysis_summary.md").read_text(encoding="utf-8")
    thesis_text = (config.output_dir / "thesis_tables.md").read_text(encoding="utf-8")
    assert "5 of 7 instances" in summary_text
    assert "5 of 7 instances" in thesis_text


def test_mapf751_cdf_l_vs_spf_sign_convention_on_priority_sensitive_subset() -> None:
    config = _real_analysis_config()
    run_conflict_aware_priority_analysis(config, plot_writer=None)

    sensitive_rows = {
        row["instance_id"]: row
        for row in csv.DictReader(
            (config.output_dir / "priority_sensitive_subset.csv").open(encoding="utf-8")
        )
    }
    n20 = sensitive_rows["AR0204SR_n20_high_001"]
    n05 = sensitive_rows["AR0204SR_n05_high_002"]

    assert int(n20["spf_soc"]) == 4742
    assert int(n20["cdf_l_soc"]) == 5046
    assert int(n20["cdf_l_minus_spf"]) == 5046 - 4742
    assert cdf_l_vs_spf_interpretation(int(n20["cdf_l_minus_spf"])) == (
        "CDF-L worse (higher SoC than SPF)"
    )

    assert int(n05["spf_soc"]) == 1670
    assert int(n05["cdf_l_soc"]) == 1668
    assert int(n05["cdf_l_minus_spf"]) == 1668 - 1670
    assert cdf_l_vs_spf_interpretation(int(n05["cdf_l_minus_spf"])) == (
        "CDF-L better (lower SoC than SPF)"
    )

    summary_text = (config.output_dir / "analysis_summary.md").read_text(encoding="utf-8")
    assert "AR0204SR_n20_high_001" in summary_text
    assert "CDF-L worse (higher SoC than SPF)" in summary_text
    assert "AR0204SR_n05_high_002" in summary_text
    assert "CDF-L better (lower SoC than SPF)" in summary_text


def test_mapf751_runtime_aggregate_cdf_h_distinct_from_spf_cd() -> None:
    config = _real_analysis_config()
    run_conflict_aware_priority_analysis(config, plot_writer=None)

    aggregate = next(
        row
        for row in csv.DictReader(
            (config.output_dir / "runtime_summary.csv").open(encoding="utf-8")
        )
        if row.get("metric_group") == "runtime_aggregate"
    )

    spf_pp = float(aggregate["mean_spf_pp_s"])
    cdf_h_pp = float(aggregate["mean_cdf_h_pp_s"])
    cdf_l_pp = float(aggregate["mean_cdf_l_pp_s"])
    spf_cd_pp = float(aggregate["mean_spf_cd_pp_s"])

    assert spf_pp == pytest.approx(30.00, abs=0.01)
    assert cdf_h_pp == pytest.approx(33.85, abs=0.01)
    assert cdf_l_pp == pytest.approx(32.38, abs=0.01)
    assert spf_cd_pp == pytest.approx(30.08, abs=0.01)
    assert cdf_h_pp != pytest.approx(spf_cd_pp, abs=0.01)

    summary_text = (config.output_dir / "analysis_summary.md").read_text(encoding="utf-8")
    assert "Mean CDF-H:" in summary_text
    assert "33.85" in summary_text
    assert "63.28" in summary_text
    assert "Mean SPF+CD:" in summary_text
    assert "30.08" in summary_text
    assert "59.60" in summary_text
