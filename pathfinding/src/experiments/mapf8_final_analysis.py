"""MAPF-8.6 final thesis figures, tables, and cross-map interpretation (presentation only)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pathfinding.src.experiments.mapf8_cross_map_analysis import MAPF8_ANALYSIS_DIR
from pathfinding.src.experiments.mapf_conflict_aware_priority_plots import (
    STRATEGY_COLORS,
    _apply_plot_style,
    _save_figure,
)
from pathfinding.src.experiments.mapf_priority_strategy_analysis import (
    _format_float,
    _read_csv_rows,
    _write_text,
)

MAPF8_FINAL_ANALYSIS_DIR = Path("pathfinding/results/mapf8_final_analysis")
MAPF7_FINAL_ANALYSIS_DIR = Path("pathfinding/results/mapf7_final_analysis")

MANDATORY_FIGURE_BASENAMES: tuple[str, ...] = (
    "fig01_soc_sensitivity_across_catalogues",
    "fig02_soc_on_all_sensitive_instances",
    "fig03_cdf_vs_spf_across_catalogues",
    "fig04_order_change_vs_soc_change",
)

CATALOGUE_DISPLAY_ORDER: tuple[tuple[str, str], ...] = (
    ("AR0204SR primary", "AR0204SR_primary"),
    ("AR0204SR held-out", "AR0204SR_held_out"),
    ("AR0400SR", "AR0400SR"),
    ("AR0307SR", "AR0307SR"),
)

CATALOGUE_SHORT_LABELS: dict[str, str] = {
    "AR0204SR primary": "AR0204-P",
    "AR0204SR held-out": "AR0204-H",
    "AR0400SR": "AR0400",
    "AR0307SR": "AR0307",
}

CATALOGUE_BAR_COLORS: dict[str, str] = {
    "AR0204SR primary": "#4C72B0",
    "AR0204SR held-out": "#C44E52",
    "AR0400SR": "#55A868",
    "AR0307SR": "#8172B2",
}

FIG3_SEMANTIC_COLORS: dict[str, str] = {
    "cdf_better": "#4C72B0",
    "equal": "#B0B0B0",
    "spf_better": STRATEGY_COLORS["spf"],
}

FIG4_ANNOTATION_VALUES: tuple[tuple[str, str], ...] = (
    ("CDF-H", "36/54"),
    ("CDF-H", "2/54"),
    ("CDF-L", "34/54"),
    ("CDF-L", "3/54"),
)

FROZEN_CATALOGUE_SUMMARY: tuple[dict[str, Any], ...] = (
    {
        "catalogue": "AR0204SR primary",
        "instances": 27,
        "soc_sensitive": "5/27 (18.5%)",
        "makespan_sensitive": "1/27 (3.7%)",
        "max_soc_range": 415,
        "mean_soc_range": "—",
    },
    {
        "catalogue": "AR0204SR held-out",
        "instances": 27,
        "soc_sensitive": "3/27 (11.1%)",
        "makespan_sensitive": "0/27 (0.0%)",
        "max_soc_range": 264,
        "mean_soc_range": "—",
    },
    {
        "catalogue": "AR0400SR",
        "instances": 27,
        "soc_sensitive": "2/27 (7.4%)",
        "makespan_sensitive": "1/27 (3.7%)",
        "max_soc_range": 396,
        "mean_soc_range": "14.70",
    },
    {
        "catalogue": "AR0307SR",
        "instances": 27,
        "soc_sensitive": "2/27 (7.4%)",
        "makespan_sensitive": "1/27 (3.7%)",
        "max_soc_range": 22,
        "mean_soc_range": "0.85",
    },
)

FROZEN_POOLED_CROSS_MAP_SUMMARY: dict[str, Any] = {
    "catalogue": "pooled cross-map (AR0400SR + AR0307SR aggregate)",
    "instances": 54,
    "soc_sensitive": "4/54 (7.4%)",
    "makespan_sensitive": "2/54 (3.7%)",
    "max_soc_range": 396,
    "mean_soc_range": "7.78",
}

FROZEN_PAIRWISE_CROSS_MAP: tuple[dict[str, Any], ...] = (
    {
        "comparison": "CDF-H vs SPF",
        "left_better": 0,
        "equal": 52,
        "right_better": 2,
        "mean_diff": 0.26,
        "median_diff": 0.0,
        "max_abs_diff": 13,
    },
    {
        "comparison": "CDF-L vs SPF",
        "left_better": 1,
        "equal": 51,
        "right_better": 2,
        "mean_diff": 7.72,
        "median_diff": 0.0,
        "max_abs_diff": 396,
    },
    {
        "comparison": "CDF-H vs CDF-L",
        "left_better": 2,
        "equal": 50,
        "right_better": 2,
        "mean_diff": -7.46,
        "median_diff": 0.0,
        "max_abs_diff": 396,
    },
    {
        "comparison": "SPF+CD vs SPF",
        "left_better": 0,
        "equal": 54,
        "right_better": 0,
        "mean_diff": 0.0,
        "median_diff": 0.0,
        "max_abs_diff": 0,
    },
)

FROZEN_ORDERING_TABLE: dict[str, Any] = {
    "spf_cd_cross_map_same": "54/54",
    "spf_cd_historical_same": "54/54",
    "spf_cd_combined_same": "108/108",
    "cdf_h_different_order": "36/54",
    "cdf_h_soc_different": "2/54",
    "cdf_l_different_order": "34/54",
    "cdf_l_soc_different": "3/54",
}

EXPECTED_SENSITIVE_INSTANCE_COUNT = 12
CROSS_MAP_INSTANCE_COUNT = 54


@dataclass(frozen=True, slots=True)
class Mapf8FinalAnalysisConfig:
    analysis_dir: Path
    mapf7_analysis_dir: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class Mapf8FinalAnalysisResult:
    output_dir: Path
    figures_written: tuple[str, ...]
    markdown_written: tuple[str, ...]
    sensitive_instance_count: int


def default_mapf8_final_config(repo_root: Path) -> Mapf8FinalAnalysisConfig:
    return Mapf8FinalAnalysisConfig(
        analysis_dir=repo_root / MAPF8_ANALYSIS_DIR,
        mapf7_analysis_dir=repo_root / MAPF7_FINAL_ANALYSIS_DIR,
        output_dir=repo_root / MAPF8_FINAL_ANALYSIS_DIR,
    )


def _load_csv(path: Path) -> list[dict[str, str]]:
    return _read_csv_rows(path)


def load_all_sensitive_instances(
    *,
    analysis_dir: Path,
    mapf7_analysis_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for row in _load_csv(mapf7_analysis_dir / "sensitive_instances.csv"):
        catalogue_key = str(row["catalogue"])
        if catalogue_key == "primary":
            display = "AR0204SR primary"
        elif catalogue_key == "held_out":
            display = "AR0204SR held-out"
        else:
            display = catalogue_key
        rows.append(
            {
                "catalogue": display,
                "instance_id": row["instance_id"],
                "agent_count": int(row["agent_count"]),
                "interaction_level": row["interaction_level"],
                "spf_soc": int(row["spf_soc"]),
                "cdf_h_soc": int(row["cdf_h_soc"]),
                "cdf_l_soc": int(row["cdf_l_soc"]),
                "spf_cd_soc": int(row["spf_cd_soc"]),
                "soc_range": int(row["soc_range"]),
            }
        )

    for row in _load_csv(analysis_dir / "sensitive_instances.csv"):
        rows.append(
            {
                "catalogue": row["map_name"],
                "instance_id": row["instance_id"],
                "agent_count": int(row["agent_count"]),
                "interaction_level": row["interaction_level"],
                "spf_soc": int(row["spf_soc"]),
                "cdf_h_soc": int(row["cdf_h_soc"]),
                "cdf_l_soc": int(row["cdf_l_soc"]),
                "spf_cd_soc": int(row["spf_cd_soc"]),
                "soc_range": int(row["soc_range"]),
            }
        )

    catalogue_rank = {label: index for index, (label, _) in enumerate(CATALOGUE_DISPLAY_ORDER)}
    rows.sort(
        key=lambda item: (
            catalogue_rank.get(str(item["catalogue"]), 99),
            -int(item["soc_range"]),
            str(item["instance_id"]),
        )
    )
    return rows


def load_historical_comparison(analysis_dir: Path) -> list[dict[str, Any]]:
    rows = _load_csv(analysis_dir / "historical_comparison.csv")
    return [
        {
            "catalogue_key": row["catalogue"],
            "catalogue": next(
                label
                for label, key in CATALOGUE_DISPLAY_ORDER
                if key == row["catalogue"]
            ),
            "cdf_h_left_better": int(row["cdf_h_vs_spf_left_better"]),
            "cdf_h_equal": int(row["cdf_h_vs_spf_equal"]),
            "cdf_h_right_better": int(row["cdf_h_vs_spf_right_better"]),
            "cdf_h_mean_diff": float(row["cdf_h_vs_spf_mean_diff"]),
            "cdf_l_left_better": int(row["cdf_l_vs_spf_left_better"]),
            "cdf_l_equal": int(row["cdf_l_vs_spf_equal"]),
            "cdf_l_right_better": int(row["cdf_l_vs_spf_right_better"]),
            "cdf_l_mean_diff": float(row["cdf_l_vs_spf_mean_diff"]),
        }
        for row in rows
    ]


def load_pooled_spf_cd_aggregate(analysis_dir: Path) -> dict[str, Any]:
    for row in _load_csv(analysis_dir / "spf_cd_order_summary.csv"):
        if row.get("catalogue") == "pooled_cross_map" and row.get("scope") == "aggregate":
            return {
                "instance_count": int(row["instance_count"]),
                "cdf_h_different_order_count": int(row["cdf_h_different_order_count"]),
                "cdf_l_different_order_count": int(row["cdf_l_different_order_count"]),
                "cdf_h_order_diff_no_soc_diff_count": int(
                    row["cdf_h_order_diff_no_soc_diff_count"]
                ),
                "cdf_l_order_diff_no_soc_diff_count": int(
                    row["cdf_l_order_diff_no_soc_diff_count"]
                ),
                "same_order_count": int(row["same_order_count"]),
            }
    raise ValueError("pooled_cross_map aggregate row not found in spf_cd_order_summary.csv")


def build_fig02_soc_range_points(
    sensitive_instances: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in sensitive_instances]
    rows.sort(key=lambda item: (-int(item["soc_range"]), str(item["instance_id"])))
    for row in rows:
        row["figure_label"] = sensitive_instance_figure_label(row)
    return rows


def sensitive_instance_figure_label(row: dict[str, Any]) -> str:
    catalogue = CATALOGUE_SHORT_LABELS[str(row["catalogue"])]
    instance_id = str(row["instance_id"])
    for prefix in ("AR0204SR_HO_", "AR0204SR_", "AR0400SR_", "AR0307SR_"):
        if instance_id.startswith(prefix):
            short_id = instance_id[len(prefix) :]
            return f"{catalogue} · {short_id}"
    return f"{catalogue} · {instance_id}"


def build_fig01_catalogue_sensitivity_points() -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in FROZEN_CATALOGUE_SUMMARY:
        soc_text = str(row["soc_sensitive"])
        count_part = soc_text.split(" ")[0]
        sensitive_count, total = count_part.split("/")
        pct = float(sensitive_count) / float(total) * 100.0
        points.append(
            {
                "catalogue": row["catalogue"],
                "soc_sensitive_count": int(sensitive_count),
                "instance_count": int(total),
                "soc_sensitive_pct": pct,
                "annotation": count_part,
            }
        )
    return points


def build_fig03_cdf_vs_spf_points(
    historical_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    order = [key for _, key in CATALOGUE_DISPLAY_ORDER]
    lookup = {row["catalogue_key"]: row for row in historical_rows}
    for key in order:
        row = lookup[key]
        for strategy in ("cdf_h", "cdf_l"):
            label = "CDF-H" if strategy == "cdf_h" else "CDF-L"
            points.append(
                {
                    "catalogue": row["catalogue"],
                    "strategy": label,
                    "cdf_better": row[f"{strategy}_left_better"],
                    "equal": row[f"{strategy}_equal"],
                    "spf_better": row[f"{strategy}_right_better"],
                    "mean_diff_cdf_minus_spf": row[f"{strategy}_mean_diff"],
                }
            )
    return points


def build_fig04_order_vs_soc_points(pooled_spf_cd: dict[str, Any]) -> list[dict[str, Any]]:
    total = int(pooled_spf_cd["instance_count"])
    cdf_h_order = int(pooled_spf_cd["cdf_h_different_order_count"])
    cdf_l_order = int(pooled_spf_cd["cdf_l_different_order_count"])
    cdf_h_soc = total - int(
        next(row["equal"] for row in FROZEN_PAIRWISE_CROSS_MAP if row["comparison"] == "CDF-H vs SPF")
    )
    cdf_l_soc = total - int(
        next(row["equal"] for row in FROZEN_PAIRWISE_CROSS_MAP if row["comparison"] == "CDF-L vs SPF")
    )
    return [
        {
            "strategy": "CDF-H",
            "order_changed": cdf_h_order,
            "soc_changed": cdf_h_soc,
            "denominator": total,
        },
        {
            "strategy": "CDF-L",
            "order_changed": cdf_l_order,
            "soc_changed": cdf_l_soc,
            "denominator": total,
        },
    ]


def render_mapf8_final_figures(
    *,
    output_dir: Path,
    sensitive_instances: Sequence[dict[str, Any]],
    historical_rows: Sequence[dict[str, Any]],
    pooled_spf_cd: dict[str, Any],
) -> tuple[str, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    _apply_plot_style()
    plots_dir = output_dir / "figures"
    written: list[str] = []

    # Figure 1
    fig1_points = build_fig01_catalogue_sensitivity_points()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(fig1_points))
    pcts = [point["soc_sensitive_pct"] for point in fig1_points]
    colors = [CATALOGUE_BAR_COLORS[point["catalogue"]] for point in fig1_points]
    bars = ax.bar(x, pcts, color=colors, width=0.62, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([point["catalogue"] for point in fig1_points], rotation=15, ha="right")
    ax.set_ylabel("SoC-sensitive instances (%)")
    ax.set_title("Four-strategy SoC sensitivity across catalogues")
    ax.set_ylim(0, max(pcts) * 1.25 if pcts else 20)
    for bar, point in zip(bars, fig1_points, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            point["annotation"],
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    _save_figure(fig, plots_dir, MANDATORY_FIGURE_BASENAMES[0])
    plt.close(fig)
    written.extend([f"{MANDATORY_FIGURE_BASENAMES[0]}.png", f"{MANDATORY_FIGURE_BASENAMES[0]}.pdf"])

    # Figure 2 — horizontal SoC range bars (magnitude-oriented)
    fig2_points = build_fig02_soc_range_points(sensitive_instances)
    fig_height = max(5.5, len(fig2_points) * 0.42 + 1.5)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    y = np.arange(len(fig2_points))
    ranges = [int(point["soc_range"]) for point in fig2_points]
    colors = [CATALOGUE_BAR_COLORS[str(point["catalogue"])] for point in fig2_points]
    ax.barh(y, ranges, color=colors, height=0.62, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([point["figure_label"] for point in fig2_points], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Four-strategy SoC range")
    ax.set_title(
        f"SoC range on all sensitive instances across four catalogues (n={len(fig2_points)})"
    )
    x_pad = max(ranges) * 0.015 if ranges else 1
    for idx, value in enumerate(ranges):
        ax.text(value + x_pad, idx, str(value), va="center", ha="left", fontsize=8)
    ax.set_xlim(0, max(ranges) * 1.12 if ranges else 1)
    fig.tight_layout()
    _save_figure(fig, plots_dir, MANDATORY_FIGURE_BASENAMES[1])
    plt.close(fig)
    written.extend([f"{MANDATORY_FIGURE_BASENAMES[1]}.png", f"{MANDATORY_FIGURE_BASENAMES[1]}.pdf"])

    # Figure 3 — wins/equal/loss counts with shared semantic colors
    fig3_points = build_fig03_cdf_vs_spf_points(historical_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    for ax, strategy in zip(axes, ("CDF-H", "CDF-L"), strict=True):
        subset = [point for point in fig3_points if point["strategy"] == strategy]
        x = np.arange(len(subset))
        cdf_better = [point["cdf_better"] for point in subset]
        equal = [point["equal"] for point in subset]
        spf_better = [point["spf_better"] for point in subset]
        bar_cdf = ax.bar(
            x,
            cdf_better,
            label="CDF better",
            color=FIG3_SEMANTIC_COLORS["cdf_better"],
            width=0.62,
        )
        bar_equal = ax.bar(
            x,
            equal,
            bottom=cdf_better,
            label="Equal",
            color=FIG3_SEMANTIC_COLORS["equal"],
            width=0.62,
        )
        bar_spf = ax.bar(
            x,
            spf_better,
            bottom=[a + b for a, b in zip(cdf_better, equal, strict=True)],
            label="SPF better",
            color=FIG3_SEMANTIC_COLORS["spf_better"],
            width=0.62,
        )
        if not legend_handles:
            legend_handles = [bar_cdf, bar_equal, bar_spf]
            legend_labels = ["CDF better", "Equal", "SPF better"]
        ax.set_xticks(x)
        ax.set_xticklabels([point["catalogue"] for point in subset], rotation=15, ha="right")
        ax.set_ylabel("Instance count (of 27)")
        ax.set_title(f"{strategy} vs SPF")
        ax.set_ylim(0, 30)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
        fontsize=9,
    )
    fig.suptitle(
        "CDF vs SPF direction by catalogue (stacked counts; diff = CDF − SPF)",
        fontsize=11,
        y=1.08,
    )
    fig.tight_layout()
    _save_figure(fig, plots_dir, MANDATORY_FIGURE_BASENAMES[2])
    plt.close(fig)
    written.extend([f"{MANDATORY_FIGURE_BASENAMES[2]}.png", f"{MANDATORY_FIGURE_BASENAMES[2]}.pdf"])

    # Figure 4
    fig4_points = build_fig04_order_vs_soc_points(pooled_spf_cd)
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    x = np.arange(len(fig4_points))
    width = 0.34
    order_vals = [point["order_changed"] for point in fig4_points]
    soc_vals = [point["soc_changed"] for point in fig4_points]
    ax.bar(
        x - width / 2,
        order_vals,
        width=width,
        label="Order changed vs SPF",
        color="#4C72B0",
    )
    ax.bar(
        x + width / 2,
        soc_vals,
        width=width,
        label="SoC changed vs SPF",
        color="#C44E52",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([point["strategy"] for point in fig4_points])
    ax.set_ylabel(f"Instance count (of {CROSS_MAP_INSTANCE_COUNT})")
    ax.set_title("Order changes vs SoC changes on cross-map instances")
    ax.set_ylim(0, 42)
    for idx, point in enumerate(fig4_points):
        ax.text(
            idx - width / 2,
            order_vals[idx] + 1.0,
            f"{point['order_changed']}/54",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.text(
            idx + width / 2,
            soc_vals[idx] + 1.0,
            f"{point['soc_changed']}/54",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=2,
        frameon=True,
        fontsize=9,
    )
    fig.subplots_adjust(top=0.78)
    fig.tight_layout()
    _save_figure(fig, plots_dir, MANDATORY_FIGURE_BASENAMES[3])
    plt.close(fig)
    written.extend([f"{MANDATORY_FIGURE_BASENAMES[3]}.png", f"{MANDATORY_FIGURE_BASENAMES[3]}.pdf"])

    return tuple(written)


def build_thesis_tables_md(
    sensitive_instances: Sequence[dict[str, Any]] | None = None,
) -> str:
    lines = [
        "# MAPF-8 Final Thesis Tables",
        "",
        "## Table A — Catalogue-level summary (four-strategy SoC range)",
        "",
        "| Catalogue | Instances | SoC-sensitive | Makespan-sensitive | Max SoC range | Mean SoC range |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in FROZEN_CATALOGUE_SUMMARY:
        lines.append(
            f"| {row['catalogue']} | {row['instances']} | {row['soc_sensitive']} | "
            f"{row['makespan_sensitive']} | {row['max_soc_range']} | {row['mean_soc_range']} |"
        )
    pooled = FROZEN_POOLED_CROSS_MAP_SUMMARY
    lines.extend(
        [
            "",
            "### Aggregate (not an independent catalogue)",
            "",
            "| Summary | Instances | SoC-sensitive | Makespan-sensitive | Max SoC range | Mean SoC range |",
            "|---|---:|---:|---:|---:|---:|",
            f"| {pooled['catalogue']} | {pooled['instances']} | {pooled['soc_sensitive']} | "
            f"{pooled['makespan_sensitive']} | {pooled['max_soc_range']} | {pooled['mean_soc_range']} |",
            "",
            "## Table B — Pairwise cross-map comparisons (diff = LEFT − RIGHT)",
            "",
            "| Comparison | LEFT better | Equal | RIGHT better | Mean diff | Median diff | Max abs(diff) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in FROZEN_PAIRWISE_CROSS_MAP:
        lines.append(
            f"| {row['comparison']} | {row['left_better']} | {row['equal']} | {row['right_better']} | "
            f"{_format_float(row['mean_diff'])} | {_format_float(row['median_diff'])} | "
            f"{row['max_abs_diff']} |"
        )
    order = FROZEN_ORDERING_TABLE
    lines.extend(
        [
            "",
            "## Table C — Ordering behaviour",
            "",
            "| Comparison | Result |",
            "|---|---|",
            f"| SPF+CD vs SPF (cross-map) | same order = {order['spf_cd_cross_map_same']} |",
            f"| SPF+CD vs SPF (historical AR0204SR) | same order = {order['spf_cd_historical_same']} |",
            f"| SPF+CD vs SPF (combined four catalogues) | same order = {order['spf_cd_combined_same']} |",
            f"| CDF-H vs SPF (cross-map) | different order = {order['cdf_h_different_order']}; "
            f"SoC different = {order['cdf_h_soc_different']} |",
            f"| CDF-L vs SPF (cross-map) | different order = {order['cdf_l_different_order']}; "
            f"SoC different = {order['cdf_l_soc_different']} |",
            "",
        ]
    )
    if sensitive_instances:
        lines.extend(
            [
                "## Table D — Sensitive-instance strategy SoC values (supporting detail)",
                "",
                "Supporting detail for Figure 2; primary Figure 2 displays four-strategy SoC range.",
                "",
                "| Catalogue | instance_id | SPF | CDF-H | CDF-L | SPF+CD | SoC range |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in build_fig02_soc_range_points(sensitive_instances):
            lines.append(
                f"| {row['catalogue']} | {row['instance_id']} | {row['spf_soc']} | "
                f"{row['cdf_h_soc']} | {row['cdf_l_soc']} | {row['spf_cd_soc']} | "
                f"{row['soc_range']} |"
            )
        lines.append("")
    return "\n".join(lines)


def build_figure_captions_md() -> str:
    return """# MAPF-8 Final Figure Captions

## Figure 1 — `fig01_soc_sensitivity_across_catalogues`

**English caption:** Four-strategy SoC sensitivity rate for the four independent 27-instance catalogues (AR0204SR primary, AR0204SR held-out, AR0400SR, AR0307SR). Bars show the percentage of instances with non-zero SoC range across SPF, CDF-H, CDF-L, and SPF+CD; exact counts are annotated.

**Polish caption:** Odsetek instancji wrażliwych na SoC w definicji czterech strategii priorytetu (SPF, CDF-H, CDF-L, SPF+CD) dla czterech niezależnych katalogów po 27 instancji. Słupki przedstawiają udział procentowy; wartości dokładne podano w adnotacjach.

**Interpretacja (PL):** Rysunek pokazuje, że wrażliwość jakościowa na wybór strategii priorytetu pozostaje rzadka we wszystkich czterech katalogach (od 7,4% do 18,5%). Nie wskazuje to na uniwersalną dominację jednej strategii, lecz na rzadkie, zależne od instancji efekty. Porównanie obejmuje wyłącznie cztery właściwe katalogi; pulę cross-map traktuje się osobno.

---

## Figure 2 — `fig02_soc_on_all_sensitive_instances`

**English caption:** Horizontal bar chart of four-strategy SoC range (max − min across SPF, CDF-H, CDF-L, SPF+CD) for all twelve sensitive instances across the four catalogues (5 + 3 + 2 + 2). Instances are sorted by range descending; labels use short catalogue codes (AR0204-P, AR0204-H, AR0400, AR0307) and instance identifiers.

**Polish caption:** Poziomy wykres słupkowy zakresu SoC w definicji czterech strategii (max − min dla SPF, CDF-H, CDF-L, SPF+CD) dla wszystkich dwunastu instancji wrażliwych w czterech katalogach (5 + 3 + 2 + 2). Instancje posortowano malejąco według zakresu; etykiety używają skrótów katalogów (AR0204-P, AR0204-H, AR0400, AR0307) oraz identyfikatorów instancji.

**Interpretacja (PL):** Rysunek pokazuje wielkość wrażliwości jakościowej: efekty mogą być bardzo duże (415, 396, 264) albo minimalne (1–2). Szczegółowe wartości SoC poszczególnych strategii podano w Tabeli D. Nie wolno z tego wnioskować o stałej przewadze strategii — wielkość efektu jest silnie zależna od instancji w obrębie rodziny bg512.

---

## Figure 3 — `fig03_cdf_vs_spf_across_catalogues`

**English caption:** Stacked instance counts for CDF-H vs SPF and CDF-L vs SPF by catalogue, using diff = CDF − SPF (negative = CDF better, zero = equal, positive = SPF better). Each catalogue contributes 27 instances.

**Polish caption:** Skumulowana liczba instancji dla CDF-H vs SPF oraz CDF-L vs SPF w podziale na katalogi; konwencja: diff = CDF − SPF (ujemna = CDF lepszy, zero = równy, dodatnia = SPF lepszy). Każdy katalog obejmuje 27 instancji.

**Interpretacja (PL):** We wszystkich katalogach dominuje klasa „equal”, a kierunek przewagi CDF nie jest stabilny między mapami. Pojedyncze odstępstwa mogą silnie wpływać na średnie, dlatego rysunek oparto na liczbach instancji, a nie wyłącznie na średnich różnicach SoC. Nie jest to dowód skuteczności reguły konfliktowej w sensie przyczynowym.

---

## Figure 4 — `fig04_order_change_vs_soc_change`

**English caption:** Cross-map instances only (n = 54): comparison of priority order changes vs SoC changes relative to SPF for CDF-H and CDF-L. Order changes are frequent; final SoC changes are rare.

**Polish caption:** Wyłącznie instancje cross-map (n = 54): porównanie zmian kolejności priorytetów i zmian SoC względem SPF dla CDF-H i CDF-L. Zmiany kolejności są częste; zmiany końcowego SoC — rzadkie.

**Interpretacja (PL):** Rysunek pokazuje kluczowy wynik MAPF-8: modyfikacja kolejności priorytetów jest częsta (36/54 dla CDF-H, 34/54 dla CDF-L), podczas gdy zmiana jakości rozwiązania (SoC) występuje tylko w 2–3 przypadkach na 54. Oznacza to, że zmiana kolejności nie implikuje automatycznie zmiany kosztu końcowego. Nie wolno stąd wnioskować o przyczynowości ani o równoważności strategii.
"""


def build_final_mapf8_interpretation_md() -> str:
    return """# MAPF-8 Final Interpretation

## 1. Main result

Priority sensitivity remained sparse across all four tested catalogues. Across all **108** MAPF instances, **12/108** were SoC-sensitive under the four-strategy definition, while **3/108** were makespan-sensitive. Sensitivity therefore affects a small minority of instances, but can be large when it occurs.

## 2. Cross-map validation

On the two new bg512 maps, sensitivity was **2/27 (7.4%)** for both AR0400SR and AR0307SR. Together with historical AR0204SR results (**5/27** primary, **3/27** held-out), this supports **persistence of sparse sensitivity across topologies within the bg512 family**. It does **not** support universal MovingAI generalization.

## 3. Magnitude

Sensitivity magnitude is strongly instance-dependent. Max observed SoC ranges include **415** (AR0204SR primary), **264** (held-out), **396** (AR0400SR), and **22** (AR0307SR). Equal catalogue-level sensitivity rates can therefore mask very different effect sizes.

## 4. SPF baseline

SPF remained a stable practical baseline in the tested catalogues: most pairwise comparisons against CDF variants are equal on SoC. This should not be read as “SPF is universally optimal”, only that it frequently matched the best observed cost in this sample.

## 5. Conflict-degree ordering

CDF-H and CDF-L did not show a consistent cross-map SoC advantage. On cross-map instances, priority order often differed from SPF (**36/54** and **34/54**), while SoC differed only **2/54** and **3/54** respectively. Ordering changes are common; quality changes are sparse.

## 6. SPF+CD

No different SPF+CD ordering was observed in **108/108** tested instances (54 historical + 54 cross-map). This is **observational only** and does not establish mathematical equivalence between SPF and SPF+CD.

## 7. Conflict structure

On pooled cross-map data, SoC-sensitive instances (n = 4) showed higher mean conflict-event, pair, and degree summaries than insensitive instances (n = 50). This is a **descriptive association only**; MAPF-8 does not support causal or predictive claims from conflict degree alone.

## 8. Limitation

All maps belong to the MovingAI **bg512 arena family**. MAPF-8 demonstrates **topology-level cross-map validation within bg512**, not universal MovingAI validation or cross-domain validation.
"""


def build_thesis_subsection_pl_md() -> str:
    return """# MAPF-8 — Walidacja cross-map w rodzinie bg512 (wersja robocza)

## 1. Cel walidacji cross-map

Celem MAPF-8 było sprawdzenie, czy wnioski z wcześniejszych eksperymentów priorytetowych na mapie AR0204SR utrzymują się na innych topologiach z tej samej rodziny scenariuszy MovingAI bg512. Testowano dwie nowe mapy — AR0400SR i AR0307SR — z zamrożonymi katalogami po 27 instancji MAPF każda. Dla każdej instancji wykonano cztery strategie ustalania kolejności priorytetów: SPF, CDF-H, CDF-L oraz SPF+CD. Walidacja dotyczyła wyłącznie poziomu topologii w obrębie bg512, a nie pełnej generalizacji na wszystkie mapy MovingAI.

## 2. Metodologia

Analiza opierała się na zamrożonych wynikach wykonania MAPF-8.4 oraz formalnej analizie MAPF-8.5. Wrażliwość primary definiowano jako niezerowy zakres SoC między wszystkimi czterema strategiami; analogicznie badano wrażliwość makespan. Porównania parami stosowały konwencję diff = LEFT − RIGHT. Dla SPF+CD porównywano kolejność priorytetów ze SPF; wynik traktowano obserwacyjnie, bez twierdzenia o równoważności formalnej. Historyczne katalogi AR0204SR (primary i held-out) włączono do interpretacji łącznej (108 instancji), zachowując rozdzielenie katalogów.

## 3. Wyniki ogólne

W czterech katalogach łącznie zebrano 108 instancji MAPF. Przy definicji czterech strategii wrażliwych na SoC było 12 instancji (11,1%), a wrażliwych na makespan — 3 (2,8%). Wrażliwość pozostawała rzadka we wszystkich czterech badanych katalogach. Jednocześnie maksymalne obserwowane zakresy SoC sięgały 415, 264, 396 i 22 w poszczególnych katalogach, co pokazuje silną zależność wielkości efektu od instancji.

## 4. Sensitivity

Na mapach cross-map odsetek wrażliwości SoC wyniósł 2/27 (7,4%) dla AR0400SR i AR0307SR — wartość zgodna z rzadkością obserwowaną historycznie (5/27 primary, 3/27 held-out). Wszystkie cztery wrażliwe instancje cross-map należały do warstwy interakcji HIGH; nie wolno jednak z tego wnioskować, że HIGH „powoduje” wrażliwość — jest to opis statystyczny na małej próbie.

## 5. CDF-H i CDF-L

Analiza parami wskazuje, że większość instancji ma równy SoC między CDF a SPF; kierunek ewentualnej przewagi nie jest stabilny między katalogami. Na 54 instancjach cross-map CDF-H zmieniał kolejność względem SPF w 36 przypadkach, a CDF-L w 34, podczas gdy różnica SoC wystąpiła odpowiednio tylko w 2 i 3 przypadkach. Zmiana kolejności priorytetów jest więc częstsza niż zmiana jakości końcowego rozwiązania.

## 6. SPF+CD

We wszystkich 108 analizowanych instancjach (54 historycznych AR0204SR oraz 54 cross-map) nie zaobserwowano innego porządku priorytetów SPF+CD względem SPF. Obserwacja ta nie stanowi dowodu formalnej równoważności obu strategii — w badanej próbie tie-break stopnia konfliktu nie zmienił kolejności.

## 7. Struktura konfliktowa

Instancje wrażliwe w puli cross-map charakteryzowały się wyższymi średnimi liczbami zdarzeń konfliktowych, par konfliktowych i maksymalnego stopnia w grafie konfliktów niż instancje niewrażliwe. MAPF-8 traktuje to wyłącznie jako związek opisowy; sam stopień konfliktu nie wystarczał do jednoznacznego rozróżnienia instancji wrażliwych i niewrażliwych w badanej próbie.

## 8. Ograniczenia

Wszystkie mapy należą do rodziny bg512. MAPF-8 dostarcza dowodu walidacji cross-map/topologii w tym ograniczonym zbiorze, a nie walidacji uniwersalnej dla całego zbioru MovingAI ani dla innych domen. Liczby wrażliwych instancji są małe, więc wnioski należy formułować konserwatywnie.

## 9. Wniosek końcowy

MAPF-8 potwierdza, że wrażliwość priorytetów na jakość SoC pozostaje rzadka także na nowych topologiach bg512, przy silnie instancyjnej wielkości efektu. SPF zachowuje rolę praktycznego baseline'u, a CDF-H/CDF-L nie wykazują spójnej przewagi cross-map. Częste zmiany kolejności CDF nie przekładają się proporcjonalnie na zmiany SoC. Wyniki wspierają ostrożną interpretację wyników MAPF-7 w szerszym, lecz nadal lokalnym kontekście topologii bg512.
"""


def run_mapf8_final_analysis(config: Mapf8FinalAnalysisConfig) -> Mapf8FinalAnalysisResult:
    sensitive_instances = load_all_sensitive_instances(
        analysis_dir=config.analysis_dir,
        mapf7_analysis_dir=config.mapf7_analysis_dir,
    )
    if len(sensitive_instances) != EXPECTED_SENSITIVE_INSTANCE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SENSITIVE_INSTANCE_COUNT} sensitive instances, "
            f"found {len(sensitive_instances)}"
        )

    historical_rows = load_historical_comparison(config.analysis_dir)
    pooled_spf_cd = load_pooled_spf_cd_aggregate(config.analysis_dir)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    figures_written = render_mapf8_final_figures(
        output_dir=config.output_dir,
        sensitive_instances=sensitive_instances,
        historical_rows=historical_rows,
        pooled_spf_cd=pooled_spf_cd,
    )

    markdown_files = {
        "thesis_tables.md": build_thesis_tables_md(sensitive_instances),
        "figure_captions.md": build_figure_captions_md(),
        "final_mapf8_interpretation.md": build_final_mapf8_interpretation_md(),
        "thesis_subsection_pl.md": build_thesis_subsection_pl_md(),
    }
    for name, content in markdown_files.items():
        _write_text(config.output_dir / name, content)

    return Mapf8FinalAnalysisResult(
        output_dir=config.output_dir,
        figures_written=figures_written,
        markdown_written=tuple(markdown_files),
        sensitive_instance_count=len(sensitive_instances),
    )
