"""Tests for MAPF-9.7 final thesis artifact generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from pathfinding.src.experiments.mapf9_final_analysis import (
    EXPECTED_CGLPS_LEX_INSTANCES,
    FROZEN_PRESENTATION_CHECKS,
    MANDATORY_FIGURE_BASENAMES,
    build_cglps_lex_improved_table_rows,
    build_fig01_quality_points,
    build_fig02_matched_budget_points,
    build_fig03_cost_points,
    build_thesis_tables_md,
    default_mapf9_final_config,
    load_mapf9_final_presentation_data,
    run_mapf9_final_analysis,
    verify_final_presentation_consistency,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _primary_available() -> bool:
    root = _repo_root() / "pathfinding/results/mapf9_primary_analysis"
    return (root / "instance_level_summary.csv").is_file()


@pytest.mark.skipif(not _primary_available(), reason="MAPF-9.6 analysis missing")
def test_exactly_three_mandatory_figures() -> None:
    assert len(MANDATORY_FIGURE_BASENAMES) == 3


@pytest.mark.skipif(not _primary_available(), reason="MAPF-9.6 analysis missing")
def test_primary_instance_count_is_54() -> None:
    config = default_mapf9_final_config(_repo_root())
    data = load_mapf9_final_presentation_data(config.primary_analysis_dir)
    assert len(data.instance_level) == 54


@pytest.mark.skipif(not _primary_available(), reason="MAPF-9.6 analysis missing")
def test_final_consistency_passes() -> None:
    config = default_mapf9_final_config(_repo_root())
    data = load_mapf9_final_presentation_data(config.primary_analysis_dir)
    assert verify_final_presentation_consistency(data) == []


@pytest.mark.skipif(not _primary_available(), reason="MAPF-9.6 analysis missing")
def test_table2_values_match_analysis() -> None:
    config = default_mapf9_final_config(_repo_root())
    data = load_mapf9_final_presentation_data(config.primary_analysis_dir)
    fig1 = build_fig01_quality_points(data)
    cglps = next(point for point in fig1 if point["method"] == "CGLPS")
    ubls = next(point for point in fig1 if point["method"] == "UBLS")
    assert cglps == {"method": "CGLPS", "soc_improved": 2, "makespan_only": 3}
    assert ubls == {"method": "UBLS", "soc_improved": 1, "makespan_only": 0}


@pytest.mark.skipif(not _primary_available(), reason="MAPF-9.6 analysis missing")
def test_table3_cglps_vs_ubls_values() -> None:
    config = default_mapf9_final_config(_repo_root())
    data = load_mapf9_final_presentation_data(config.primary_analysis_dir)
    points = build_fig02_matched_budget_points(data)
    soc = next(point for point in points if point["objective"] == "SoC-only")
    lex = next(point for point in points if point["objective"] == "Lexicographic")
    assert (soc["cglps_better"], soc["equal"], soc["ubls_better"]) == (1, 53, 0)
    assert (lex["cglps_better"], lex["equal"], lex["ubls_better"]) == (4, 50, 0)


@pytest.mark.skipif(not _primary_available(), reason="MAPF-9.6 analysis missing")
def test_table4_has_five_distinct_cglps_instances() -> None:
    config = default_mapf9_final_config(_repo_root())
    data = load_mapf9_final_presentation_data(config.primary_analysis_dir)
    rows = build_cglps_lex_improved_table_rows(data)
    assert len(rows) == 5
    assert {row["instance_id"] for row in rows} == EXPECTED_CGLPS_LEX_INSTANCES


@pytest.mark.skipif(not _primary_available(), reason="MAPF-9.6 analysis missing")
def test_table4_excludes_shared_ubls_only_row() -> None:
    config = default_mapf9_final_config(_repo_root())
    data = load_mapf9_final_presentation_data(config.primary_analysis_dir)
    rows = build_cglps_lex_improved_table_rows(data)
    assert sum(1 for row in rows if row["instance_id"] == "AR0400SR_n05_high_000") == 1
    assert all(row["instance_id"] in EXPECTED_CGLPS_LEX_INSTANCES for row in rows)


@pytest.mark.skipif(not _primary_available(), reason="MAPF-9.6 analysis missing")
def test_cost_table_runtime_ratios_match_source() -> None:
    config = default_mapf9_final_config(_repo_root())
    data = load_mapf9_final_presentation_data(config.primary_analysis_dir)
    points = build_fig03_cost_points(data)
    cglps = next(point for point in points if point["method"] == "CGLPS")
    ubls = next(point for point in points if point["method"] == "UBLS")
    spf = next(point for point in points if point["method"] == "SPF")
    assert spf["logical_runtime_ratio_vs_spf"] == pytest.approx(1.0)
    assert cglps["logical_runtime_ratio_vs_spf"] == pytest.approx(1.9446244958983179)
    assert ubls["logical_runtime_ratio_vs_spf"] == pytest.approx(1.9337868764869446)


@pytest.mark.skipif(not _primary_available(), reason="MAPF-9.6 analysis missing")
def test_figure1_values_match_source() -> None:
    config = default_mapf9_final_config(_repo_root())
    data = load_mapf9_final_presentation_data(config.primary_analysis_dir)
    points = build_fig01_quality_points(data)
    assert points[0]["soc_improved"] == FROZEN_PRESENTATION_CHECKS["cglps_soc_improved"]
    assert points[0]["makespan_only"] == FROZEN_PRESENTATION_CHECKS["cglps_makespan_only"]
    assert points[1]["soc_improved"] == FROZEN_PRESENTATION_CHECKS["ubls_soc_improved"]


@pytest.mark.skipif(not _primary_available(), reason="MAPF-9.6 analysis missing")
def test_figure2_values_match_source() -> None:
    config = default_mapf9_final_config(_repo_root())
    data = load_mapf9_final_presentation_data(config.primary_analysis_dir)
    points = build_fig02_matched_budget_points(data)
    assert points[0]["cglps_better"] == FROZEN_PRESENTATION_CHECKS["cglps_vs_ubls_soc"][0]
    assert points[1]["cglps_better"] == FROZEN_PRESENTATION_CHECKS["cglps_vs_ubls_lex"][0]


@pytest.mark.skipif(not _primary_available(), reason="MAPF-9.6 analysis missing")
def test_figure3_runtime_ratios_match_source() -> None:
    config = default_mapf9_final_config(_repo_root())
    data = load_mapf9_final_presentation_data(config.primary_analysis_dir)
    points = build_fig03_cost_points(data)
    assert points[1]["logical_runtime_ratio_vs_spf"] == pytest.approx(1.9446244958983179, rel=1e-6)


@pytest.mark.skipif(not _primary_available(), reason="MAPF-9.6 analysis missing")
def test_polish_subsection_has_no_incorrect_ar0307_soc_claim() -> None:
    config = default_mapf9_final_config(_repo_root())
    run_mapf9_final_analysis(config)
    text = (config.output_dir / "thesis_subsection_pl.md").read_text(encoding="utf-8")
    assert "2/27" not in text or "AR0307SR **1/27**" in text
    assert "AR0307SR **1/27**" in text
    assert "udowodniono" not in text.lower()


@pytest.mark.skipif(not _primary_available(), reason="MAPF-9.6 analysis missing")
def test_no_162_instance_combined_denominator() -> None:
    config = default_mapf9_final_config(_repo_root())
    run_mapf9_final_analysis(config)
    for name in (
        "final_mapf9_interpretation.md",
        "thesis_subsection_pl.md",
        "thesis_tables.md",
    ):
        text = (config.output_dir / name).read_text(encoding="utf-8")
        assert "162" not in text


@pytest.mark.skipif(not _primary_available(), reason="MAPF-9.6 analysis missing")
def test_no_statistical_significance_claim() -> None:
    config = default_mapf9_final_config(_repo_root())
    run_mapf9_final_analysis(config)
    for name in ("final_mapf9_interpretation.md", "thesis_tables.md", "figure_captions.md"):
        text = (config.output_dir / name).read_text(encoding="utf-8").lower()
        assert "statistically significant" not in text
        assert "p < " not in text
        assert "p-value" not in text


@pytest.mark.skipif(not _primary_available(), reason="MAPF-9.6 analysis missing")
def test_close_declares_mapf9_frozen() -> None:
    config = default_mapf9_final_config(_repo_root())
    run_mapf9_final_analysis(config)
    text = (config.output_dir / "mapf9_close.md").read_text(encoding="utf-8")
    assert "MAPF-9 CLOSED" in text
    assert "frozen" in text.lower()


@pytest.mark.skipif(not _primary_available(), reason="MAPF-9.6 analysis missing")
def test_run_generates_all_final_artifacts() -> None:
    config = default_mapf9_final_config(_repo_root())
    result = run_mapf9_final_analysis(config)
    expected_md = (
        "thesis_tables.md",
        "figure_captions.md",
        "final_mapf9_interpretation.md",
        "thesis_subsection_pl.md",
        "mapf9_close.md",
    )
    for name in expected_md:
        assert (config.output_dir / name).is_file(), name
    for basename in MANDATORY_FIGURE_BASENAMES:
        assert (config.output_dir / "figures" / f"{basename}.png").is_file(), basename
        assert (config.output_dir / "figures" / f"{basename}.pdf").is_file(), basename
    assert len(result.figures_written) == 6


@pytest.mark.skipif(not _primary_available(), reason="MAPF-9.6 analysis missing")
def test_thesis_tables_contain_table_headers() -> None:
    config = default_mapf9_final_config(_repo_root())
    data = load_mapf9_final_presentation_data(config.primary_analysis_dir)
    thesis = build_thesis_tables_md(data)
    for header in (
        "## Table 1",
        "## Table 2",
        "## Table 3",
        "## Table 4",
        "## Table 5",
    ):
        assert header in thesis
