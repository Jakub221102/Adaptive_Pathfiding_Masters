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
    load_conflict_aware_results,
    mapf7_strategy_socs_differ,
)
from pathfinding.src.experiments.mapf7_final_analysis import (
    Mapf7FinalAnalysisConfig,
    build_heldout_instance_level_summary,
    build_spf_cd_ablation_aggregate,
    build_spf_cd_ablation_rows,
    count_mapf7_soc_sensitive_instances,
    load_heldout_spf_results,
    run_mapf7_final_analysis,
    validate_final_analysis_inputs,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _conflict_row(
    instance_id: str,
    strategy: str,
    *,
    soc: int = 100,
    agent_count: int = 5,
    interaction: str = "low",
) -> dict[str, object]:
    order = list(range(agent_count))
    degrees = [0] * agent_count
    return {
        "instance_id": instance_id,
        "agent_count": agent_count,
        "interaction_level": interaction,
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
        "catalogue_role": "held_out_validation",
        "catalogue_seed": 2027,
    }


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


def _real_config(output_dir: Path | None = None) -> Mapf7FinalAnalysisConfig:
    repo_root = Path("pathfinding/results")
    return Mapf7FinalAnalysisConfig(
        primary_conflict_csv=repo_root
        / "mapf_conflict_aware_priority_execution"
        / "results.csv",
        primary_manifest_path=Path(
            "pathfinding/results/mapf_benchmarks/AR0204SR_manifest.json"
        ),
        primary_spf_lpf_csv=repo_root
        / "mapf_priority_strategy_execution"
        / "results.csv",
        fixed_results_csv=repo_root / "mapf_benchmark_execution" / "results.csv",
        random_results_csv=repo_root / "mapf_random_priority_pilot" / "results.csv",
        heldout_conflict_csv=repo_root
        / "mapf_conflict_aware_priority_heldout_execution"
        / "results.csv",
        heldout_spf_csv=repo_root / "mapf_spf_heldout_execution" / "results.csv",
        heldout_manifest_path=Path(
            "pathfinding/results/mapf_benchmarks/AR0204SR_heldout_manifest.json"
        ),
        output_dir=output_dir or repo_root / "mapf7_final_analysis",
        expected_instance_count=27,
    )


def test_primary_mapf7_record_count() -> None:
    records = load_conflict_aware_results(
        Path("pathfinding/results/mapf_conflict_aware_priority_execution/results.csv")
    )
    assert len(records) == 81


def test_heldout_mapf7_record_count() -> None:
    records = load_conflict_aware_results(
        Path(
            "pathfinding/results/mapf_conflict_aware_priority_heldout_execution/results.csv"
        )
    )
    assert len(records) == 81


def test_heldout_spf_record_count() -> None:
    records = load_heldout_spf_results(
        Path("pathfinding/results/mapf_spf_heldout_execution/results.csv")
    )
    assert len(records) == 27
    assert all(record.strategy == "spf" for record in records)


def test_heldout_instance_alignment() -> None:
    mapf7 = load_conflict_aware_results(
        Path(
            "pathfinding/results/mapf_conflict_aware_priority_heldout_execution/results.csv"
        )
    )
    spf = load_heldout_spf_results(
        Path("pathfinding/results/mapf_spf_heldout_execution/results.csv")
    )
    mapf7_ids = {record.instance_id for record in mapf7}
    spf_ids = {record.instance_id for record in spf}
    assert mapf7_ids == spf_ids
    assert len(mapf7_ids) == 27


def test_sign_convention_left_minus_right() -> None:
    counts = count_directional_soc_better([10, 0, -5])
    assert counts.right_better == 1
    assert counts.equal == 1
    assert counts.left_better == 1


def test_validation_passes_on_real_data() -> None:
    report = validate_final_analysis_inputs(_real_config())
    assert report.passed, report.messages


def test_primary_and_heldout_never_instance_paired() -> None:
    config = _real_config()
    result = run_mapf7_final_analysis(config, plot_writer=None)
    assert result.validation.passed
    primary_ids = {row["instance_id"] for row in result.primary_instance_rows}
    heldout_ids = {row["instance_id"] for row in result.heldout_instance_rows}
    assert primary_ids.isdisjoint(heldout_ids)


def test_sensitive_instance_count_derived_not_hardcoded(tmp_path: Path) -> None:
    config = _real_config(output_dir=tmp_path / "out")
    result = run_mapf7_final_analysis(config, plot_writer=None)
    assert result.validation.passed

    derived_primary = count_mapf7_soc_sensitive_instances(result.primary_instance_rows)
    derived_heldout = count_mapf7_soc_sensitive_instances(result.heldout_instance_rows)

    sensitive_csv = list(
        csv.DictReader((config.output_dir / "sensitive_instances.csv").open(encoding="utf-8"))
    )
    primary_csv = [row for row in sensitive_csv if row["catalogue"] == "primary"]
    heldout_csv = [row for row in sensitive_csv if row["catalogue"] == "held_out"]

    assert len(primary_csv) == derived_primary
    assert len(heldout_csv) == derived_heldout
    assert derived_primary == 5
    assert derived_heldout == 3


def test_spf_cd_order_equality_derived(tmp_path: Path) -> None:
    config = _real_config(output_dir=tmp_path / "out2")
    result = run_mapf7_final_analysis(config, plot_writer=None)
    assert result.validation.passed

    ablation_rows = list(
        csv.DictReader((config.output_dir / "spf_cd_ablation.csv").open(encoding="utf-8"))
    )
    primary_detail = [
        row for row in ablation_rows if row.get("catalogue") == "primary" and row.get("instance_id")
    ]
    identical_order = sum(
        1 for row in primary_detail if row.get("same_priority_order") == "True"
    )
    aggregate = next(
        row for row in ablation_rows if row.get("catalogue") == "primary" and not row.get("instance_id")
    )
    assert int(aggregate["identical_ordering"]) == identical_order
    assert int(aggregate["identical_ordering"]) == 27


def test_runtime_aggregation_uses_correct_strategies(tmp_path: Path) -> None:
    config = _real_config(output_dir=tmp_path / "out3")
    result = run_mapf7_final_analysis(config, plot_writer=None)
    assert result.validation.passed

    runtime_rows = list(
        csv.DictReader((config.output_dir / "runtime_summary.csv").open(encoding="utf-8"))
    )
    heldout_agg = next(
        row
        for row in runtime_rows
        if row.get("metric_group") == "runtime_aggregate"
        and row.get("catalogue") == "held_out"
    )
    assert float(heldout_agg["mean_spf_ordering_ms"]) > 0
    assert float(heldout_agg["mean_cdf_h_ordering_ms"]) > 0
    assert float(heldout_agg["mean_cdf_l_ordering_ms"]) > 0
    assert float(heldout_agg["mean_spf_cd_ordering_ms"]) > 0


def test_sensitive_instance_table_has_all_strategies(tmp_path: Path) -> None:
    config = _real_config(output_dir=tmp_path / "out4")
    run_mapf7_final_analysis(config, plot_writer=None)

    for row in csv.DictReader(
        (config.output_dir / "sensitive_instances.csv").open(encoding="utf-8")
    ):
        assert row["cdf_h_soc"] != ""
        assert row["cdf_l_soc"] != ""
        assert row["spf_cd_soc"] != ""
        assert row["spf_soc"] != ""


def test_primary_pairwise_reproduces_cdf_h_vs_spf() -> None:
    config = _real_config()
    result = run_mapf7_final_analysis(config, plot_writer=None)
    assert result.validation.passed

    primary_pairwise = list(
        csv.DictReader(
            (config.output_dir / "primary_pairwise_summary.csv").open(encoding="utf-8")
        )
    )
    cdf_h_vs_spf = next(row for row in primary_pairwise if row["comparison"] == "cdf_h_vs_spf")
    assert int(cdf_h_vs_spf["equal_count"]) == 23
    assert int(cdf_h_vs_spf["right_better_count"]) == 4
    assert int(cdf_h_vs_spf["left_better_count"]) == 0


def test_full_final_analysis_writes_outputs(tmp_path: Path) -> None:
    config = _real_config(output_dir=tmp_path / "full")
    result = run_mapf7_final_analysis(config, plot_writer=None)
    assert result.validation.passed
    assert (config.output_dir / "final_analysis_summary.md").is_file()
    assert (config.output_dir / "generalization_summary.csv").is_file()
    assert (config.output_dir / "sensitive_instances.csv").is_file()


def test_mapf7_strategy_socs_differ_unit() -> None:
    assert mapf7_strategy_socs_differ(
        {"cdf_h_soc": 100, "cdf_l_soc": 100, "spf_cd_soc": 100}
    ) is False
    assert mapf7_strategy_socs_differ(
        {"cdf_h_soc": 100, "cdf_l_soc": 98, "spf_cd_soc": 100}
    ) is True


def test_heldout_cdf_h_vs_cdf_l_direction_reversal(tmp_path: Path) -> None:
    config = _real_config(output_dir=tmp_path / "out5")
    run_mapf7_final_analysis(config, plot_writer=None)

    generalization = list(
        csv.DictReader(
            (config.output_dir / "generalization_summary.csv").open(encoding="utf-8")
        )
    )
    cdf_row = next(row for row in generalization if row["comparison"] == "cdf_h_vs_cdf_l")
    assert int(cdf_row["primary_right_better"]) == 5
    assert int(cdf_row["heldout_left_better"]) == 3
    assert "direction reversed" in cdf_row["generalization_interpretation"]

    summary_text = (config.output_dir / "final_analysis_summary.md").read_text(encoding="utf-8")
    assert "reversed between catalogues" in summary_text
    assert "Primary catalogue: CDF-H better=0, equal=22, CDF-L better=5" in summary_text
    assert "Held-out catalogue: CDF-H better=3, equal=24, CDF-L better=0" in summary_text


def test_spf_cd_ablation_summary_wording_and_values(tmp_path: Path) -> None:
    config = _real_config(output_dir=tmp_path / "out6")
    run_mapf7_final_analysis(config, plot_writer=None)

    generalization = list(
        csv.DictReader(
            (config.output_dir / "generalization_summary.csv").open(encoding="utf-8")
        )
    )
    spf_cd_row = next(
        row for row in generalization if row["comparison"] == "spf_cd_ablation_identical_soc"
    )
    assert spf_cd_row["primary_mean_soc_diff"] == "27/27"
    assert spf_cd_row["heldout_mean_soc_diff"] == "27/27"
    assert spf_cd_row["combined_identical_soc"] == "54/54"
    assert spf_cd_row["combined_identical_ordering"] == "54/54"
    assert spf_cd_row["combined_differing_ordering"] == "0/54"

    summary_text = (config.output_dir / "final_analysis_summary.md").read_text(encoding="utf-8")
    assert "primary identical SoC=27/27" in summary_text
    assert "held-out identical SoC=27/27" in summary_text
    assert "combined identical SoC=54/54" in summary_text
    assert "When orders differ" not in summary_text
    assert "never changed the resulting order" in summary_text


def test_fig05_runtime_values_unchanged(tmp_path: Path) -> None:
    config = _real_config(output_dir=tmp_path / "out7")
    run_mapf7_final_analysis(config, plot_writer=None)

    heldout_agg = next(
        row
        for row in csv.DictReader(
            (config.output_dir / "runtime_summary.csv").open(encoding="utf-8")
        )
        if row.get("metric_group") == "runtime_aggregate"
        and row.get("catalogue") == "held_out"
    )
    assert float(heldout_agg["mean_spf_ordering_ms"]) == pytest.approx(33812.61944445712)
    assert float(heldout_agg["mean_cdf_h_ordering_ms"]) == pytest.approx(35600.92710372043)
    assert float(heldout_agg["mean_cdf_l_ordering_ms"]) == pytest.approx(35727.98911851607)
    assert float(heldout_agg["mean_spf_cd_ordering_ms"]) == pytest.approx(35804.304248144574)
    assert float(heldout_agg["mean_spf_pp_ms"]) == pytest.approx(34216.83939999504)
    assert float(heldout_agg["mean_cdf_h_pp_ms"]) == pytest.approx(36388.91617407057)

    instance_ids = ("HO_n05_low_000", "HO_n05_low_001")
    manifest = MAPFBenchmarkManifest(
        map_name="bench.map",
        scenario_name="bench.scen",
        seed=2027,
        max_timestep=512,
        min_reference_length=1.0,
        catalogue_role="held_out_validation",
        instances=tuple(_instance(iid) for iid in instance_ids),
    )
    manifest_path = tmp_path / "heldout_manifest.json"
    save_benchmark_manifest(manifest, manifest_path)

    conflict_rows: list[dict[str, object]] = []
    spf_rows: list[dict[str, object]] = []
    for instance_id in instance_ids:
        for strategy in ("cdf_h", "cdf_l", "spf_cd"):
            conflict_rows.append(_conflict_row(instance_id, strategy))
        spf_rows.append(
            {
                "instance_id": instance_id,
                "agent_count": 5,
                "interaction_level": "low",
                "strategy": "spf",
                "agent_order": json.dumps([1, 0, 2, 3, 4]),
                "success": "True",
                "termination_reason": "success",
                "ordering_time_ms": 500.0,
                "pp_time_ms": 1500.0,
                "total_time_ms": 2000.0,
                "soc": 100,
                "makespan": 20,
            }
        )

    conflict_csv = tmp_path / "heldout_conflict.csv"
    _write_csv(conflict_csv, list(conflict_rows[0].keys()), conflict_rows)
    spf_csv = tmp_path / "heldout_spf.csv"
    _write_csv(spf_csv, list(spf_rows[0].keys()), spf_rows)

    conflict_records = load_conflict_aware_results(conflict_csv)
    spf_records = load_heldout_spf_results(spf_csv)
    instance_rows = build_heldout_instance_level_summary(
        manifest=manifest,
        conflict_records=conflict_records,
        spf_records=spf_records,
    )
    ablation = build_spf_cd_ablation_rows(
        instance_rows,
        catalogue="held_out",
        conflict_records=conflict_records,
        spf_records=spf_records,
    )
    aggregate = build_spf_cd_ablation_aggregate(ablation, catalogue="held_out")
    assert aggregate["differing_ordering"] == 2
    assert aggregate["identical_soc"] == 2
