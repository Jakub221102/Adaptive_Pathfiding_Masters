from __future__ import annotations

from pathlib import Path

import pytest

from pathfinding.src.experiments.thesis_mapf_figures import (
    THESIS_FIGURE_BASENAMES,
    default_thesis_mapf_figures_config,
    load_ch6_priority_sensitivity_data,
    load_ch7_order_vs_soc_data,
    load_ch8_matched_budget_data,
    load_ch8_search_cost_data,
    run_thesis_mapf_figures,
    verify_thesis_mapf_figure_sources,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_frozen_source_files_exist() -> None:
    config = default_thesis_mapf_figures_config(_repo_root())
    required = (
        config.mapf6_analysis_dir / "random_sensitivity.csv",
        config.mapf6_analysis_dir / "instance_level_summary.csv",
        config.mapf8_analysis_dir / "spf_cd_order_summary.csv",
        config.mapf8_analysis_dir / "pairwise_summary.csv",
        config.mapf9_primary_analysis_dir / "soc_pairwise_summary.csv",
        config.mapf9_primary_analysis_dir / "lexicographic_pairwise_summary.csv",
        config.mapf9_primary_analysis_dir / "cost_benefit_summary.csv",
    )
    for path in required:
        assert path.is_file(), path


def test_ch6_frozen_counts_match_evidence() -> None:
    config = default_thesis_mapf_figures_config(_repo_root())
    sensitive_rows, _ = load_ch6_priority_sensitivity_data(config)
    assert len(sensitive_rows) == 7
    assert int(sensitive_rows[0]["random_soc_range"]) == 719


def test_ch7_frozen_counts_match_evidence() -> None:
    config = default_thesis_mapf_figures_config(_repo_root())
    points = load_ch7_order_vs_soc_data(config)
    by_strategy = {str(point["strategy"]): point for point in points}
    assert by_strategy["CDF-H"]["order_changed"] == 36
    assert by_strategy["CDF-H"]["soc_changed"] == 2
    assert by_strategy["CDF-L"]["order_changed"] == 34
    assert by_strategy["CDF-L"]["soc_changed"] == 3


def test_ch8_matched_budget_counts_match_evidence() -> None:
    config = default_thesis_mapf_figures_config(_repo_root())
    points = load_ch8_matched_budget_data(config)
    by_criterion = {str(point["criterion"]): point for point in points}
    assert (
        by_criterion["SoC"]["cglps_better"],
        by_criterion["SoC"]["equal"],
        by_criterion["SoC"]["ubls_better"],
    ) == (1, 53, 0)
    assert (
        by_criterion["SoC → makespan"]["cglps_better"],
        by_criterion["SoC → makespan"]["equal"],
        by_criterion["SoC → makespan"]["ubls_better"],
    ) == (4, 50, 0)


def test_ch8_search_cost_ratios_match_evidence() -> None:
    config = default_thesis_mapf_figures_config(_repo_root())
    points = load_ch8_search_cost_data(config)
    by_method = {str(point["method"]): point for point in points}
    assert float(by_method["SPF"]["logical_runtime_ratio_vs_spf"]) == 1.0
    assert abs(float(by_method["CGLPS"]["logical_runtime_ratio_vs_spf"]) - 1.945) < 0.01
    assert abs(float(by_method["UBLS"]["logical_runtime_ratio_vs_spf"]) - 1.934) < 0.01


def test_verify_sources_passes_on_repository_data() -> None:
    config = default_thesis_mapf_figures_config(_repo_root())
    verify_thesis_mapf_figure_sources(config)


def test_run_writes_deterministic_outputs(tmp_path: Path) -> None:
    repo_root = _repo_root()
    base_config = default_thesis_mapf_figures_config(repo_root)
    config = type(base_config)(
        repo_root=repo_root,
        mapf6_analysis_dir=base_config.mapf6_analysis_dir,
        mapf8_analysis_dir=base_config.mapf8_analysis_dir,
        mapf9_primary_analysis_dir=base_config.mapf9_primary_analysis_dir,
        output_dir=tmp_path / "thesis_img",
    )
    result = run_thesis_mapf_figures(config)
    assert result.files_written == tuple(
        f"{basename}.{ext}"
        for basename in THESIS_FIGURE_BASENAMES
        for ext in ("png", "pdf")
    )
    for basename in THESIS_FIGURE_BASENAMES:
        png = config.output_dir / f"{basename}.png"
        pdf = config.output_dir / f"{basename}.pdf"
        assert png.is_file()
        assert pdf.is_file()
        assert png.stat().st_size > 0
        assert pdf.stat().st_size > 0
