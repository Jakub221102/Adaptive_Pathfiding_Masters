from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pathfinding.src.experiments.mapf7_final_analysis import run_mapf7_final_analysis
from pathfinding.src.experiments.mapf7_final_analysis_plots import (
    CATALOGUE_HELD_OUT,
    CATALOGUE_PRIMARY,
    build_fig01_cdf_h_minus_spf_points,
    build_fig04_scatter_points,
    count_sensitive_by_interaction,
    instance_full_soc_range,
    render_mapf7_final_analysis_plots,
)
from pathfinding.tests.unit.test_mapf7_final_analysis import _real_config


def _run_with_plots(tmp_path: Path):
    config = _real_config(output_dir=tmp_path / "plot_out")
    result = run_mapf7_final_analysis(
        config,
        plot_writer=render_mapf7_final_analysis_plots,
    )
    assert result.validation.passed
    return config, result


def test_fig04_uses_instance_level_rows_not_aggregate_groups(tmp_path: Path) -> None:
    _, result = _run_with_plots(tmp_path)

    points = build_fig04_scatter_points(
        result.primary_instance_rows,
        result.heldout_instance_rows,
    )
    assert len(points) == 54
    assert {point["catalogue"] for point in points} == {CATALOGUE_PRIMARY, CATALOGUE_HELD_OUT}
    assert sum(1 for point in points if point["catalogue"] == CATALOGUE_PRIMARY) == 27
    assert sum(1 for point in points if point["catalogue"] == CATALOGUE_HELD_OUT) == 27

    for point in points:
        assert "instance_id" in point
        assert "max_degree" in point
        assert "soc_range" in point


def test_fig04_y_values_match_exact_instance_soc_ranges(tmp_path: Path) -> None:
    _, result = _run_with_plots(tmp_path)

    points = build_fig04_scatter_points(
        result.primary_instance_rows,
        result.heldout_instance_rows,
    )
    for row in result.primary_instance_rows:
        expected = instance_full_soc_range(dict(row))
        point = next(point for point in points if point["instance_id"] == row["instance_id"])
        assert point["soc_range"] == expected

    n20 = next(
        row for row in result.primary_instance_rows if row["instance_id"] == "AR0204SR_n20_high_001"
    )
    assert instance_full_soc_range(dict(n20)) == 415


def test_fig04_x_values_match_exact_max_degrees(tmp_path: Path) -> None:
    _, result = _run_with_plots(tmp_path)

    points = build_fig04_scatter_points(
        result.primary_instance_rows,
        result.heldout_instance_rows,
    )
    lookup = {row["instance_id"]: row for row in result.primary_instance_rows}
    lookup.update({row["instance_id"]: row for row in result.heldout_instance_rows})

    for point in points:
        row = lookup[point["instance_id"]]
        assert point["max_degree"] == float(row["max_degree"])


def test_fig01_uses_exact_cdf_h_minus_spf_values(tmp_path: Path) -> None:
    _, result = _run_with_plots(tmp_path)

    primary_points = build_fig01_cdf_h_minus_spf_points(
        result.primary_instance_rows,
        catalogue=CATALOGUE_PRIMARY,
    )
    assert len(primary_points) == 27
    assert sum(1 for point in primary_points if point["soc_diff"] == 0) == 23
    assert sum(1 for point in primary_points if point["soc_diff"] > 0) == 4

    heldout_points = build_fig01_cdf_h_minus_spf_points(
        result.heldout_instance_rows,
        catalogue=CATALOGUE_HELD_OUT,
    )
    assert len(heldout_points) == 27
    assert sum(1 for point in heldout_points if point["soc_diff"] == 0) == 26
    assert sum(1 for point in heldout_points if point["soc_diff"] == -2) == 1


def test_fig06_sensitivity_counts_unchanged(tmp_path: Path) -> None:
    config, _ = _run_with_plots(tmp_path)

    sensitive_rows = list(
        csv.DictReader((config.output_dir / "sensitive_instances.csv").open(encoding="utf-8"))
    )
    primary = count_sensitive_by_interaction(sensitive_rows, catalogue=CATALOGUE_PRIMARY)
    heldout = count_sensitive_by_interaction(sensitive_rows, catalogue=CATALOGUE_HELD_OUT)

    assert primary == {"low": 0, "medium": 0, "high": 5}
    assert heldout == {"low": 0, "medium": 2, "high": 1}


def test_plot_regeneration_does_not_change_analysis_tables(tmp_path: Path) -> None:
    config_before = _real_config(output_dir=tmp_path / "before")
    run_mapf7_final_analysis(config_before, plot_writer=None)

    config_after = _real_config(output_dir=tmp_path / "after")
    run_mapf7_final_analysis(config_after, plot_writer=render_mapf7_final_analysis_plots)

    for filename in (
        "primary_pairwise_summary.csv",
        "heldout_pairwise_summary.csv",
        "generalization_summary.csv",
        "sensitive_instances.csv",
        "runtime_summary.csv",
        "spf_cd_ablation.csv",
    ):
        before_text = (config_before.output_dir / filename).read_text(encoding="utf-8")
        after_text = (config_after.output_dir / filename).read_text(encoding="utf-8")
        assert before_text == after_text, filename

    assert (config_after.output_dir / "figures" / "fig04_conflict_structure_vs_priority_effect.png").is_file()


def test_fig05_runtime_values_unchanged_after_plot_polish(tmp_path: Path) -> None:
    config, _ = _run_with_plots(tmp_path)

    heldout_agg = next(
        row
        for row in csv.DictReader(
            (config.output_dir / "runtime_summary.csv").open(encoding="utf-8")
        )
        if row.get("metric_group") == "runtime_aggregate"
        and row.get("catalogue") == "held_out"
    )
    assert float(heldout_agg["mean_spf_ordering_ms"]) == pytest.approx(33812.61944445712)
    assert float(heldout_agg["mean_cdf_h_pp_ms"]) == pytest.approx(36388.91617407057)
