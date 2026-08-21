from __future__ import annotations

from pathlib import Path

import pytest

from pathfinding.src.experiments.mapf_priority_strategy_analysis import (
    PriorityStrategyAnalysisConfig,
    run_priority_strategy_analysis,
)
from pathfinding.src.experiments.mapf_priority_strategy_plots import (
    compute_mean_runtime_seconds,
    render_priority_strategy_plots,
    select_spf_lpf_differing_instances,
    select_top_priority_sensitive_instances,
)


def _instance_row(
    instance_id: str,
    *,
    fixed_soc: int,
    spf_soc: int,
    lpf_soc: int,
    spf_ordering_ms: float = 100.0,
    spf_pp_ms: float = 200.0,
    lpf_ordering_ms: float = 110.0,
    lpf_pp_ms: float = 210.0,
) -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "agent_count": 5,
        "interaction_level": "high",
        "fixed_soc": fixed_soc,
        "spf_soc": spf_soc,
        "lpf_soc": lpf_soc,
        "random_soc_median": fixed_soc,
        "spf_ordering_time_ms": spf_ordering_ms,
        "spf_pp_time_ms": spf_pp_ms,
        "lpf_ordering_time_ms": lpf_ordering_ms,
        "lpf_pp_time_ms": lpf_pp_ms,
    }


def test_figure2_filters_out_equal_spf_lpf_instances() -> None:
    rows = (
        _instance_row("AR0204SR_n05_low_000", fixed_soc=100, spf_soc=100, lpf_soc=100),
        _instance_row("AR0204SR_n10_high_000", fixed_soc=200, spf_soc=180, lpf_soc=200),
    )

    differing = select_spf_lpf_differing_instances(rows)

    assert len(differing) == 1
    assert differing[0].short_label == "n10_high_000"


def test_figure2_keeps_spf_minus_lpf_sign_convention() -> None:
    rows = (
        _instance_row("AR0204SR_n10_high_000", fixed_soc=200, spf_soc=180, lpf_soc=200),
        _instance_row("AR0204SR_n10_high_001", fixed_soc=300, spf_soc=320, lpf_soc=300),
    )

    differing = {row.short_label: row for row in select_spf_lpf_differing_instances(rows)}

    assert differing["n10_high_000"].soc_diff == -20
    assert differing["n10_high_001"].soc_diff == 20


def test_figure4_selects_top_four_by_random_soc_range() -> None:
    sensitivity_rows = [
        {
            "instance_id": "AR0204SR_n05_high_002",
            "observed_priority_sensitive": True,
            "random_soc_range": 2,
        },
        {
            "instance_id": "AR0204SR_n20_high_001",
            "observed_priority_sensitive": True,
            "random_soc_range": 719,
        },
        {
            "instance_id": "AR0204SR_n10_high_001",
            "observed_priority_sensitive": True,
            "random_soc_range": 221,
        },
        {
            "instance_id": "AR0204SR_n10_medium_000",
            "observed_priority_sensitive": True,
            "random_soc_range": 138,
        },
        {
            "instance_id": "AR0204SR_n10_high_000",
            "observed_priority_sensitive": True,
            "random_soc_range": 79,
        },
        {
            "instance_id": "AR0204SR_n20_medium_000",
            "observed_priority_sensitive": True,
            "random_soc_range": 2,
        },
    ]

    selected = select_top_priority_sensitive_instances(sensitivity_rows, limit=4)

    assert [row["instance_id"] for row in selected] == [
        "AR0204SR_n20_high_001",
        "AR0204SR_n10_high_001",
        "AR0204SR_n10_medium_000",
        "AR0204SR_n10_high_000",
    ]


def test_figure4_selection_is_data_driven_not_hardcoded() -> None:
    sensitivity_rows = [
        {
            "instance_id": "AR0204SR_custom_a",
            "observed_priority_sensitive": True,
            "random_soc_range": 50,
        },
        {
            "instance_id": "AR0204SR_custom_b",
            "observed_priority_sensitive": True,
            "random_soc_range": 100,
        },
        {
            "instance_id": "AR0204SR_custom_c",
            "observed_priority_sensitive": False,
            "random_soc_range": 999,
        },
    ]

    selected = select_top_priority_sensitive_instances(sensitivity_rows, limit=2)

    assert [row["instance_id"] for row in selected] == [
        "AR0204SR_custom_b",
        "AR0204SR_custom_a",
    ]


def test_runtime_plot_uses_total_equals_ordering_plus_pp() -> None:
    rows = [
        _instance_row(
            "AR0204SR_n05_low_000",
            fixed_soc=100,
            spf_soc=100,
            lpf_soc=100,
            spf_ordering_ms=1000.0,
            spf_pp_ms=2000.0,
            lpf_ordering_ms=3000.0,
            lpf_pp_ms=4000.0,
        ),
        _instance_row(
            "AR0204SR_n05_low_001",
            fixed_soc=100,
            spf_soc=100,
            lpf_soc=100,
            spf_ordering_ms=3000.0,
            spf_pp_ms=4000.0,
            lpf_ordering_ms=1000.0,
            lpf_pp_ms=2000.0,
        ),
    ]

    runtime = compute_mean_runtime_seconds(rows)

    assert runtime.spf_total_s == pytest.approx(5.0)
    assert runtime.lpf_total_s == pytest.approx(5.0)
    assert runtime.spf_total_s == pytest.approx(runtime.spf_ordering_s + runtime.spf_pp_s)
    assert runtime.lpf_total_s == pytest.approx(runtime.lpf_ordering_s + runtime.lpf_pp_s)


def test_png_and_pdf_outputs_are_generated(tmp_path: Path) -> None:
    instance_rows = [
        _instance_row("AR0204SR_n05_low_000", fixed_soc=100, spf_soc=98, lpf_soc=102),
    ]
    sensitivity_rows = [
        {
            "instance_id": "AR0204SR_n05_low_000",
            "observed_priority_sensitive": True,
            "random_soc_range": 5,
            "random_soc_min": 95,
            "random_soc_max": 100,
            "random_soc_median": 98,
        }
    ]

    written = render_priority_strategy_plots(tmp_path, instance_rows, sensitivity_rows)

    assert "fig01_priority_sensitivity_by_instance.png" in written
    assert "fig02_spf_vs_lpf_soc_difference.pdf" in written
    assert (tmp_path / "plots" / "fig03_spf_lpf_runtime_cost.png").is_file()
    assert (tmp_path / "plots" / "fig04_priority_sensitive_case_study.pdf").is_file()


def test_real_analysis_regenerates_polished_plots_without_changing_csvs(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    analysis_dir = repo_root / "pathfinding/results/mapf_priority_strategy_analysis"
    instance_csv = analysis_dir / "instance_level_summary.csv"
    original_instance_csv = instance_csv.read_text(encoding="utf-8")

    config = PriorityStrategyAnalysisConfig(
        fixed_results_csv=repo_root / "pathfinding/results/mapf_benchmark_execution/results.csv",
        random_results_csv=repo_root / "pathfinding/results/mapf_random_priority_pilot/results.csv",
        strategy_results_csv=repo_root
        / "pathfinding/results/mapf_priority_strategy_execution/results.csv",
        manifest_path=repo_root / "pathfinding/results/mapf_benchmarks/AR0204SR_manifest.json",
        output_dir=tmp_path / "regenerated_analysis",
        expected_instance_count=27,
        expected_random_k=10,
    )

    result = run_priority_strategy_analysis(
        config,
        plot_writer=render_priority_strategy_plots,
    )

    assert result.validation.passed
    assert (config.output_dir / "plots" / "fig02_spf_vs_lpf_soc_difference.png").is_file()
    assert instance_csv.read_text(encoding="utf-8") == original_instance_csv
