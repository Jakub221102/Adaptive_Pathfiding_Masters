from __future__ import annotations

from pathlib import Path

import pytest

from pathfinding.src.experiments.mapf8_final_analysis import (
    CROSS_MAP_INSTANCE_COUNT,
    EXPECTED_SENSITIVE_INSTANCE_COUNT,
    FIG3_SEMANTIC_COLORS,
    FIG4_ANNOTATION_VALUES,
    FROZEN_CATALOGUE_SUMMARY,
    FROZEN_ORDERING_TABLE,
    MANDATORY_FIGURE_BASENAMES,
    build_fig01_catalogue_sensitivity_points,
    build_fig02_soc_range_points,
    build_fig03_cdf_vs_spf_points,
    build_fig04_order_vs_soc_points,
    build_figure_captions_md,
    build_thesis_tables_md,
    default_mapf8_final_config,
    load_all_sensitive_instances,
    load_historical_comparison,
    load_pooled_spf_cd_aggregate,
    run_mapf8_final_analysis,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_exactly_four_mandatory_figure_definitions() -> None:
    assert len(MANDATORY_FIGURE_BASENAMES) == 4
    assert MANDATORY_FIGURE_BASENAMES == (
        "fig01_soc_sensitivity_across_catalogues",
        "fig02_soc_on_all_sensitive_instances",
        "fig03_cdf_vs_spf_across_catalogues",
        "fig04_order_change_vs_soc_change",
    )


def test_figure1_includes_all_four_catalogues() -> None:
    points = build_fig01_catalogue_sensitivity_points()
    catalogues = {point["catalogue"] for point in points}
    assert len(points) == 4
    assert catalogues == {
        "AR0204SR primary",
        "AR0204SR held-out",
        "AR0400SR",
        "AR0307SR",
    }


def test_figure2_has_twelve_sensitive_instances() -> None:
    config = default_mapf8_final_config(_repo_root())
    rows = load_all_sensitive_instances(
        analysis_dir=config.analysis_dir,
        mapf7_analysis_dir=config.mapf7_analysis_dir,
    )
    assert len(rows) == EXPECTED_SENSITIVE_INSTANCE_COUNT == 12


def test_figure2_labels_are_unique_and_unambiguous() -> None:
    config = default_mapf8_final_config(_repo_root())
    rows = load_all_sensitive_instances(
        analysis_dir=config.analysis_dir,
        mapf7_analysis_dir=config.mapf7_analysis_dir,
    )
    points = build_fig02_soc_range_points(rows)
    labels = [point["figure_label"] for point in points]
    assert len(labels) == 12
    assert len(set(labels)) == 12
    assert all("AR0" != label for label in labels)
    assert not any(label == "AR0" or label.startswith("AR0\n") for label in labels)
    assert all(" · " in label for label in labels)
    prefixes = {label.split(" · ", 1)[0] for label in labels}
    assert prefixes == {"AR0204-P", "AR0204-H", "AR0400", "AR0307"}


def test_figure2_sorted_by_soc_range_descending() -> None:
    config = default_mapf8_final_config(_repo_root())
    rows = load_all_sensitive_instances(
        analysis_dir=config.analysis_dir,
        mapf7_analysis_dir=config.mapf7_analysis_dir,
    )
    points = build_fig02_soc_range_points(rows)
    ranges = [point["soc_range"] for point in points]
    assert ranges == sorted(ranges, reverse=True)
    assert ranges[0] == 415
    assert ranges[-1] == 1


def test_figure3_semantic_colors_are_shared() -> None:
    assert FIG3_SEMANTIC_COLORS["cdf_better"] == "#4C72B0"
    assert FIG3_SEMANTIC_COLORS["equal"] == "#B0B0B0"
    assert FIG3_SEMANTIC_COLORS["spf_better"] != FIG3_SEMANTIC_COLORS["cdf_better"]


def test_figure4_annotation_values_exact() -> None:
    assert FIG4_ANNOTATION_VALUES == (
        ("CDF-H", "36/54"),
        ("CDF-H", "2/54"),
        ("CDF-L", "34/54"),
        ("CDF-L", "3/54"),
    )


def test_figure2_captions_describe_soc_range() -> None:
    captions = build_figure_captions_md()
    fig2 = captions.split("## Figure 3")[0].split("## Figure 2")[1]
    assert "SoC range" in fig2 or "zakresu SoC" in fig2
    assert "four-strategy" in fig2.lower() or "czterech strategii" in fig2


def test_table_d_present_with_twelve_rows() -> None:
    config = default_mapf8_final_config(_repo_root())
    rows = load_all_sensitive_instances(
        analysis_dir=config.analysis_dir,
        mapf7_analysis_dir=config.mapf7_analysis_dir,
    )
    thesis = build_thesis_tables_md(rows)
    assert "## Table D" in thesis
    table_d_rows = [
        line
        for line in thesis.split("## Table D")[1].splitlines()
        if line.startswith("|") and not line.startswith("|---") and "Catalogue" not in line
    ]
    assert len(table_d_rows) == 12


def test_thesis_subsection_wording_is_sample_bounded() -> None:
    config = default_mapf8_final_config(_repo_root())
    run_mapf8_final_analysis(config)
    text = (config.output_dir / "thesis_subsection_pl.md").read_text(encoding="utf-8")
    assert "we wszystkich czterech badanych katalogach" in text
    assert "predyktor wrażliwości" not in text
    assert "nie wystarczał do jednoznacznego rozróżnienia" in text


def test_figure4_uses_cross_map_denominator_54() -> None:
    config = default_mapf8_final_config(_repo_root())
    pooled = load_pooled_spf_cd_aggregate(config.analysis_dir)
    points = build_fig04_order_vs_soc_points(pooled)
    assert all(point["denominator"] == CROSS_MAP_INSTANCE_COUNT for point in points)
    assert points[0]["order_changed"] == 36
    assert points[0]["soc_changed"] == 2
    assert points[1]["order_changed"] == 34
    assert points[1]["soc_changed"] == 3


def test_spf_cd_combined_count_is_108() -> None:
    assert FROZEN_ORDERING_TABLE["spf_cd_combined_same"] == "108/108"
    thesis = build_thesis_tables_md()
    assert "108/108" in thesis


def test_catalogue_table_has_no_duplicated_rows() -> None:
    thesis = build_thesis_tables_md()
    table_a = thesis.split("## Table B")[0]
    data_rows = [
        line
        for line in table_a.splitlines()
        if line.startswith("|") and not line.startswith("|---") and "Catalogue" not in line
    ]
    catalogues = [line.split("|")[1].strip() for line in data_rows]
    independent = [
        name
        for name in catalogues
        if "pooled" not in name.lower() and name != "Summary"
    ]
    assert len(independent) == len(set(independent)) == 4
    assert independent.count("AR0400SR") == 1
    assert independent.count("AR0307SR") == 1


def test_sign_convention_preserved_in_fig3_points() -> None:
    config = default_mapf8_final_config(_repo_root())
    historical = load_historical_comparison(config.analysis_dir)
    points = build_fig03_cdf_vs_spf_points(historical)
    ar0400_cdf_h = next(
        point
        for point in points
        if point["catalogue"] == "AR0400SR" and point["strategy"] == "CDF-H"
    )
    assert ar0400_cdf_h["cdf_better"] == 0
    assert ar0400_cdf_h["equal"] == 27
    assert ar0400_cdf_h["spf_better"] == 0


def test_thesis_table2_header_has_no_broken_separator() -> None:
    thesis = build_thesis_tables_md()
    header = next(line for line in thesis.splitlines() if line.startswith("| Comparison |"))
    assert "Max abs(diff)" in header
    assert "|abs diff|" not in header


def test_rq86_wording_not_in_final_interpretation_as_causal() -> None:
    config = default_mapf8_final_config(_repo_root())
    text = (config.output_dir / "final_mapf8_interpretation.md")
    if not text.is_file():
        run_mapf8_final_analysis(config)
    content = text.read_text(encoding="utf-8")
    assert "descriptive association only" in content.lower() or "descriptive" in content
    assert "causal or predictive" in content


def test_output_filenames_deterministic_and_png_pdf_generated() -> None:
    config = default_mapf8_final_config(_repo_root())
    result = run_mapf8_final_analysis(config)
    figures_dir = result.output_dir / "figures"
    for basename in MANDATORY_FIGURE_BASENAMES:
        assert (figures_dir / f"{basename}.png").is_file()
        assert (figures_dir / f"{basename}.pdf").is_file()
    assert len(result.figures_written) == len(MANDATORY_FIGURE_BASENAMES) * 2


def test_mapf7_final_outputs_not_overwritten() -> None:
    config = default_mapf8_final_config(_repo_root())
    mapf7_sensitive = config.mapf7_analysis_dir / "sensitive_instances.csv"
    assert mapf7_sensitive.is_file()
    before = mapf7_sensitive.read_text(encoding="utf-8")
    run_mapf8_final_analysis(config)
    after = mapf7_sensitive.read_text(encoding="utf-8")
    assert before == after


def test_frozen_catalogue_counts_match_spec() -> None:
    primary = next(row for row in FROZEN_CATALOGUE_SUMMARY if row["catalogue"] == "AR0204SR primary")
    assert primary["soc_sensitive"] == "5/27 (18.5%)"
    assert primary["max_soc_range"] == 415
    cross_maps = [row for row in FROZEN_CATALOGUE_SUMMARY if row["catalogue"].startswith("AR04") or row["catalogue"].startswith("AR03")]
    assert len(cross_maps) == 2
