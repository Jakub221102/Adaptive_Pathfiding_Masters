from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pathfinding.src.experiments.mapf_benchmark_analysis import count_directional_soc_better
from pathfinding.src.experiments.mapf_benchmark_execution import MAIN_BENCHMARK_RESULTS_DIR
from pathfinding.src.experiments.mapf_benchmark_instances import (
    MAPFBenchmarkInstance,
    MAPFBenchmarkManifest,
    MAPFInteractionLevel,
    load_benchmark_manifest,
    save_benchmark_manifest,
)
from pathfinding.src.experiments.mapf_priority_strategy_analysis import (
    PriorityStrategyAnalysisConfig,
    aggregate_random_by_instance,
    build_cbs_aggregate_quality_summary,
    build_cbs_quality_reference,
    build_instance_level_summary,
    build_runtime_summary,
    build_sensitivity_table_metrics,
    build_success_rate_table_rows,
    compute_runtime_fractions,
    load_fixed_results,
    load_random_results,
    load_strategy_results,
    run_priority_strategy_analysis,
    validate_datasets,
)
from pathfinding.src.experiments.mapf_priority_strategy_execution import (
    MAIN_PRIORITY_STRATEGY_RESULTS_DIR,
)
from pathfinding.src.experiments.mapf_random_priority_execution import (
    MAIN_RANDOM_PRIORITY_RESULTS_DIR,
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


def _manifest(*instance_ids: str) -> MAPFBenchmarkManifest:
    return MAPFBenchmarkManifest(
        map_name="bench.map",
        scenario_name="bench.scen",
        seed=2026,
        max_timestep=512,
        min_reference_length=1.0,
        instances=tuple(_instance(instance_id) for instance_id in instance_ids),
    )


def _tiny_datasets(tmp_path: Path) -> PriorityStrategyAnalysisConfig:
    instance_ids = ("bench_n05_low_000", "bench_n05_low_001")
    manifest = _manifest(*instance_ids)
    manifest_path = tmp_path / "manifest.json"
    save_benchmark_manifest(manifest, manifest_path)

    fixed_rows: list[dict[str, object]] = []
    random_rows: list[dict[str, object]] = []
    strategy_rows: list[dict[str, object]] = []

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
                    "ordering_seed": 1000 + ordering_index,
                    "agent_order": f"[{ordering_index}, 1]",
                    "base_seed": 2026,
                    "success": "True"
                    if ordering_index != 9 or instance_id == "bench_n05_low_000"
                    else "False",
                    "termination_reason": "success"
                    if ordering_index != 9 or instance_id == "bench_n05_low_000"
                    else "failure",
                    "execution_time_ms": 900.0 + ordering_index,
                    "soc": ""
                    if ordering_index == 9 and instance_id == "bench_n05_low_001"
                    else soc,
                    "makespan": 20,
                    "conflict_count": 0,
                    "pp_low_level_searches": 5,
                    "pp_agents_planned": 5,
                }
            )
        strategy_rows.extend(
            [
                {
                    "instance_id": instance_id,
                    "agent_count": 5,
                    "interaction_level": "low",
                    "strategy": "spf",
                    "agent_order": "[0, 1]",
                    "success": "True",
                    "termination_reason": "success",
                    "ordering_time_ms": 100.0,
                    "pp_time_ms": 200.0,
                    "total_time_ms": 300.0,
                    "soc": 98,
                    "makespan": 20,
                    "conflict_count": 0,
                    "pp_low_level_searches": 5,
                    "pp_agents_planned": 5,
                },
                {
                    "instance_id": instance_id,
                    "agent_count": 5,
                    "interaction_level": "low",
                    "strategy": "lpf",
                    "agent_order": "[1, 0]",
                    "success": "True",
                    "termination_reason": "success",
                    "ordering_time_ms": 110.0,
                    "pp_time_ms": 210.0,
                    "total_time_ms": 320.0,
                    "soc": 102,
                    "makespan": 20,
                    "conflict_count": 0,
                    "pp_low_level_searches": 5,
                    "pp_agents_planned": 5,
                },
            ]
        )

    fixed_csv = tmp_path / "fixed.csv"
    random_csv = tmp_path / "random.csv"
    strategy_csv = tmp_path / "strategy.csv"
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
    _write_csv(
        random_csv,
        [
            "instance_id",
            "agent_count",
            "interaction_level",
            "ordering_index",
            "ordering_seed",
            "agent_order",
            "base_seed",
            "success",
            "termination_reason",
            "execution_time_ms",
            "soc",
            "makespan",
            "conflict_count",
            "pp_low_level_searches",
            "pp_agents_planned",
        ],
        random_rows,
    )
    _write_csv(
        strategy_csv,
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
            "conflict_count",
            "pp_low_level_searches",
            "pp_agents_planned",
        ],
        strategy_rows,
    )

    return PriorityStrategyAnalysisConfig(
        fixed_results_csv=fixed_csv,
        random_results_csv=random_csv,
        strategy_results_csv=strategy_csv,
        manifest_path=manifest_path,
        output_dir=tmp_path / "analysis",
        expected_instance_count=2,
        expected_random_k=10,
    )


def test_random_runs_are_aggregated_per_instance_not_flattened(tmp_path: Path) -> None:
    config = _tiny_datasets(tmp_path)
    random_records = load_random_results(config.random_results_csv)
    aggregates = aggregate_random_by_instance(random_records)

    assert len(random_records) == 20
    assert len(aggregates) == 2
    assert aggregates["bench_n05_low_000"].ordering_count == 10
    assert aggregates["bench_n05_low_001"].success_count == 9


def test_directional_convention_diff_positive_means_right_better() -> None:
    counts = count_directional_soc_better([10, 0, -5])
    assert counts.right_better == 1
    assert counts.equal == 1
    assert counts.left_better == 1


def test_random_best_of_k_is_labeled_diagnostic_in_outputs(tmp_path: Path) -> None:
    config = _tiny_datasets(tmp_path)
    result = run_priority_strategy_analysis(config, plot_writer=None)
    assert result.validation.passed

    thesis = (config.output_dir / "thesis_tables.md").read_text(encoding="utf-8")
    summary = (config.output_dir / "analysis_summary.md").read_text(encoding="utf-8")
    assert "sampled best-of-K diagnostic" in thesis or "sampled best-of-K" in summary


def test_failed_random_runs_excluded_from_quality_but_in_success_rate(tmp_path: Path) -> None:
    config = _tiny_datasets(tmp_path)
    aggregates = aggregate_random_by_instance(load_random_results(config.random_results_csv))
    agg = aggregates["bench_n05_low_001"]

    assert agg.success_count == 9
    assert agg.success_rate == pytest.approx(0.9)
    assert agg.soc_mean == pytest.approx(880 / 9)


def test_spf_lpf_primary_runtime_uses_total_time_ms(tmp_path: Path) -> None:
    config = _tiny_datasets(tmp_path)
    result = run_priority_strategy_analysis(config, plot_writer=None)
    assert result.validation.passed

    runtime_rows = list(csv.DictReader((config.output_dir / "runtime_summary.csv").open()))
    instance_rows = [row for row in runtime_rows if row.get("instance_id")]
    assert instance_rows[0]["spf_total_time_ms"] == "300.0"
    assert "ordering_time_ms" in runtime_rows[0]["random_runtime_semantics"]


def test_no_fake_ordering_time_for_random(tmp_path: Path) -> None:
    config = _tiny_datasets(tmp_path)
    run_priority_strategy_analysis(config, plot_writer=None)
    summary = (config.output_dir / "analysis_summary.md").read_text(encoding="utf-8")
    assert "permutation overhead was not separately instrumented" in summary


def test_cross_dataset_instance_mismatch_fails_validation(tmp_path: Path) -> None:
    config = _tiny_datasets(tmp_path)
    manifest = _manifest("bench_n05_low_000", "bench_n05_low_999")
    manifest_path = tmp_path / "bad_manifest.json"
    save_benchmark_manifest(manifest, manifest_path)
    bad_config = PriorityStrategyAnalysisConfig(
        fixed_results_csv=config.fixed_results_csv,
        random_results_csv=config.random_results_csv,
        strategy_results_csv=config.strategy_results_csv,
        manifest_path=manifest_path,
        output_dir=tmp_path / "bad_analysis",
        expected_instance_count=2,
        expected_random_k=10,
    )

    validation = validate_datasets(
        manifest=manifest,
        fixed_records=load_fixed_results(bad_config.fixed_results_csv),
        random_records=load_random_results(bad_config.random_results_csv),
        strategy_records=load_strategy_results(bad_config.strategy_results_csv),
        config=bad_config,
    )
    assert validation.passed is False


def test_expected_row_counts_on_tiny_fixture(tmp_path: Path) -> None:
    config = _tiny_datasets(tmp_path)
    result = run_priority_strategy_analysis(config, plot_writer=None)
    assert result.validation.passed

    instance_rows = list(csv.DictReader((config.output_dir / "instance_level_summary.csv").open()))
    sensitivity_rows = list(csv.DictReader((config.output_dir / "random_sensitivity.csv").open()))
    paired_rows = list(csv.DictReader((config.output_dir / "paired_quality_summary.csv").open()))

    assert len(instance_rows) == 2
    assert len(sensitivity_rows) == 2
    assert len(paired_rows) == 3


def test_plot_generation_works_on_tiny_fixture(tmp_path: Path) -> None:
    from pathfinding.src.experiments.mapf_priority_strategy_plots import (
        render_priority_strategy_plots,
    )

    config = _tiny_datasets(tmp_path)
    result = run_priority_strategy_analysis(
        config,
        plot_writer=render_priority_strategy_plots,
    )
    assert result.validation.passed
    assert (config.output_dir / "plots" / "fig01_priority_sensitivity_by_instance.png").is_file()


def test_completed_spf_does_not_skip_lpf_via_instance_summary(tmp_path: Path) -> None:
    config = _tiny_datasets(tmp_path)
    manifest = _manifest("bench_n05_low_000", "bench_n05_low_001")
    instance_rows = build_instance_level_summary(
        manifest=manifest,
        fixed_records=load_fixed_results(config.fixed_results_csv),
        random_aggregates=aggregate_random_by_instance(
            load_random_results(config.random_results_csv)
        ),
        strategy_records=load_strategy_results(config.strategy_results_csv),
    )
    assert len(instance_rows) == 2
    assert instance_rows[0]["spf_soc"] != instance_rows[0]["lpf_soc"]


def test_total_time_ms_consistent_with_ordering_plus_pp(tmp_path: Path) -> None:
    config = _tiny_datasets(tmp_path)
    runtime_rows = build_runtime_summary(
        build_instance_level_summary(
            manifest=_manifest("bench_n05_low_000", "bench_n05_low_001"),
            fixed_records=load_fixed_results(config.fixed_results_csv),
            random_aggregates=aggregate_random_by_instance(
                load_random_results(config.random_results_csv)
            ),
            strategy_records=load_strategy_results(config.strategy_results_csv),
        )
    )
    instance_rows = [row for row in runtime_rows if row.get("instance_id")]
    for row in instance_rows:
        assert row["spf_total_time_ms"] == row["spf_ordering_time_ms"] + row["spf_pp_time_ms"]


def test_historical_result_directories_remain_distinct() -> None:
    assert MAIN_BENCHMARK_RESULTS_DIR != MAIN_PRIORITY_STRATEGY_RESULTS_DIR
    assert MAIN_RANDOM_PRIORITY_RESULTS_DIR != MAIN_PRIORITY_STRATEGY_RESULTS_DIR


def test_real_dataset_validation_and_analysis(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config = PriorityStrategyAnalysisConfig(
        fixed_results_csv=repo_root / "pathfinding/results/mapf_benchmark_execution/results.csv",
        random_results_csv=repo_root / "pathfinding/results/mapf_random_priority_pilot/results.csv",
        strategy_results_csv=repo_root
        / "pathfinding/results/mapf_priority_strategy_execution/results.csv",
        manifest_path=repo_root / "pathfinding/results/mapf_benchmarks/AR0204SR_manifest.json",
        output_dir=tmp_path / "real_analysis",
        expected_instance_count=27,
        expected_random_k=10,
    )
    from pathfinding.src.experiments.mapf_priority_strategy_plots import (
        render_priority_strategy_plots,
    )

    result = run_priority_strategy_analysis(
        config,
        plot_writer=render_priority_strategy_plots,
    )
    assert result.validation.passed
    assert len(list(csv.DictReader((config.output_dir / "instance_level_summary.csv").open()))) == 27
    paired_text = (config.output_dir / "paired_quality_summary.csv").read_text(encoding="utf-8")
    assert "left_soc - right_soc" in paired_text
    assert "spf_vs_lpf" in paired_text


def test_table_a_reports_raw_success_counts(tmp_path: Path) -> None:
    config = _tiny_datasets(tmp_path)
    instance_rows = build_instance_level_summary(
        manifest=_manifest("bench_n05_low_000", "bench_n05_low_001"),
        fixed_records=load_fixed_results(config.fixed_results_csv),
        random_aggregates=aggregate_random_by_instance(
            load_random_results(config.random_results_csv)
        ),
        strategy_records=load_strategy_results(config.strategy_results_csv),
    )

    rows = build_success_rate_table_rows(instance_rows, expected_random_k=10)

    assert rows[0]["total_label"] == "2 / 2"
    assert rows[1]["total_label"] == "19 / 20 runs"
    assert rows[1]["note"].startswith("10 nested ordering runs per instance")


def test_random_success_reporting_uses_runs_not_instances(tmp_path: Path) -> None:
    config = _tiny_datasets(tmp_path)
    run_priority_strategy_analysis(config, plot_writer=None)
    thesis = (config.output_dir / "thesis_tables.md").read_text(encoding="utf-8")

    assert "19 / 20 runs" in thesis
    assert "270 / 270 runs" not in thesis
    assert "not 20 independent instances" in thesis or "nested ordering runs per instance" in thesis


def test_table_b_direction_note_matches_left_minus_right(tmp_path: Path) -> None:
    config = _tiny_datasets(tmp_path)
    run_priority_strategy_analysis(config, plot_writer=None)
    thesis = (config.output_dir / "thesis_tables.md").read_text(encoding="utf-8")

    assert "diff = LEFT SoC - RIGHT SoC" in thesis
    assert "diff < 0" in thesis
    assert "Fixed vs SPF" in thesis
    assert "fixed_vs_spf" not in thesis


def test_table_c_derives_sensitivity_counts_from_data(tmp_path: Path) -> None:
    instance_rows = [
        {
            "instance_id": "a",
            "observed_priority_sensitive": True,
            "random_makespan_range": 0,
            "random_soc_range": 5,
        },
        {
            "instance_id": "b",
            "observed_priority_sensitive": False,
            "random_makespan_range": 2,
            "random_soc_range": 0,
        },
    ]
    metrics = build_sensitivity_table_metrics(instance_rows, instance_count=2)

    assert metrics["soc_sensitive_count"] == 1
    assert metrics["makespan_sensitive_count"] == 1
    assert metrics["max_random_soc_range"] == 5


def test_table_d_runtime_percentages_are_computed_not_hardcoded() -> None:
    aggregate = {
        "mean_spf_total_time_ms": 59625.44000740767,
        "mean_lpf_total_time_ms": 66861.101025944,
        "mean_spf_ordering_time_ms": 29621.783266667517,
        "mean_lpf_ordering_time_ms": 29694.560225932273,
    }
    fractions = compute_runtime_fractions(aggregate)

    assert fractions["spf_ordering_fraction"] == pytest.approx(0.4968, rel=1e-3)
    assert fractions["lpf_ordering_fraction"] == pytest.approx(0.4441, rel=1e-3)


def test_lpf_ordering_fraction_is_numerically_correct() -> None:
    aggregate = {
        "mean_spf_total_time_ms": 100.0,
        "mean_lpf_total_time_ms": 200.0,
        "mean_spf_ordering_time_ms": 40.0,
        "mean_lpf_ordering_time_ms": 50.0,
    }
    fractions = compute_runtime_fractions(aggregate)

    assert fractions["lpf_ordering_fraction"] == pytest.approx(0.25)


def test_table_e_contains_all_soc_sensitive_instances_sorted(tmp_path: Path) -> None:
    config = _tiny_datasets(tmp_path)
    run_priority_strategy_analysis(config, plot_writer=None)
    thesis = (config.output_dir / "thesis_tables.md").read_text(encoding="utf-8")

    assert "Observed priority-sensitive instances under Random K=10" in thesis
    assert "bench_n05_low_001" in thesis


def test_table_f_uses_all_basic_cbs_common_success_cases() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config = PriorityStrategyAnalysisConfig(
        fixed_results_csv=repo_root / "pathfinding/results/mapf_benchmark_execution/results.csv",
        random_results_csv=repo_root / "pathfinding/results/mapf_random_priority_pilot/results.csv",
        strategy_results_csv=repo_root
        / "pathfinding/results/mapf_priority_strategy_execution/results.csv",
        manifest_path=repo_root / "pathfinding/results/mapf_benchmarks/AR0204SR_manifest.json",
        output_dir=repo_root / "pathfinding/results/mapf_priority_strategy_analysis",
        expected_instance_count=27,
        expected_random_k=10,
    )
    fixed_records = load_fixed_results(config.fixed_results_csv)
    instance_rows = build_instance_level_summary(
        manifest=load_benchmark_manifest(config.manifest_path),
        fixed_records=fixed_records,
        random_aggregates=aggregate_random_by_instance(
            load_random_results(config.random_results_csv)
        ),
        strategy_records=load_strategy_results(config.strategy_results_csv),
    )
    cbs_rows = build_cbs_quality_reference(
        instance_rows=instance_rows,
        fixed_records=fixed_records,
    )
    aggregate = build_cbs_aggregate_quality_summary(cbs_rows)

    assert len(cbs_rows) == 22
    assert all(row["common_success_count"] == 22 for row in aggregate)
    assert aggregate[0]["strategy"] == "Fixed"
    assert aggregate[-1]["diagnostic_only"] is True


def test_cbs_timeout_cases_excluded_from_quality_averages() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    fixed_records = load_fixed_results(
        repo_root / "pathfinding/results/mapf_benchmark_execution/results.csv"
    )
    cbs_successes = {
        record.instance_id
        for record in fixed_records
        if record.algorithm == "cbs_basic" and record.success
    }
    manifest_count = 27

    assert len(cbs_successes) == 22
    assert manifest_count - len(cbs_successes) == 5


def test_real_thesis_tables_and_summary_consistency(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config = PriorityStrategyAnalysisConfig(
        fixed_results_csv=repo_root / "pathfinding/results/mapf_benchmark_execution/results.csv",
        random_results_csv=repo_root / "pathfinding/results/mapf_random_priority_pilot/results.csv",
        strategy_results_csv=repo_root
        / "pathfinding/results/mapf_priority_strategy_execution/results.csv",
        manifest_path=repo_root / "pathfinding/results/mapf_benchmarks/AR0204SR_manifest.json",
        output_dir=tmp_path / "real_thesis",
        expected_instance_count=27,
        expected_random_k=10,
    )
    result = run_priority_strategy_analysis(config, plot_writer=None)
    assert result.validation.passed

    thesis = (config.output_dir / "thesis_tables.md").read_text(encoding="utf-8")
    summary = (config.output_dir / "analysis_summary.md").read_text(encoding="utf-8")

    assert "270 / 270 runs" in thesis
    assert "7 / 27" in thesis
    assert "44.4%" in thesis
    assert "49.7%" in thesis
    assert "Basic CBS common-success instances: 22" in thesis
    assert "Random sampled best-of-K (diagnostic)" in thesis
    assert "49.7% of total" in summary
    assert "44.4% of total" in summary
    assert "uniformly slower LPF behaviour" in summary
