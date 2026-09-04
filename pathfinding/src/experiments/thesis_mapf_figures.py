"""Thesis-ready MAPF figure exporter (presentation only, frozen CSV inputs).

Reads existing MAPF-6 / MAPF-8 / MAPF-9 analysis artifacts without modifying them.
Outputs Polish-labelled PDF + PNG assets under docs/thesis/img/.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --- Source provenance (frozen analysis artifacts; do not overwrite) ---
#
# Figure 1 (Ch.6): mapf_priority_strategy_analysis/random_sensitivity.csv
#                   mapf_priority_strategy_analysis/instance_level_summary.csv
#   Metric: Random K=10 observed min/max/median SoC; SPF/LPF deterministic SoC.
#
# Figure 2 (Ch.7): mapf8_cross_map_analysis/spf_cd_order_summary.csv (pooled_cross_map)
#                   mapf8_cross_map_analysis/pairwise_summary.csv (pooled SoC equality)
#   Metric: order change vs SPF; SoC change vs SPF on n=54 cross-map instances.
#
# Figure 3 (Ch.8): mapf9_primary_analysis/soc_pairwise_summary.csv (pooled CGLPS_vs_UBLS_soc)
#                   mapf9_primary_analysis/lexicographic_pairwise_summary.csv (pooled)
#   Metric: matched-budget directional instance counts.
#
# Figure 4 (Ch.8): mapf9_primary_analysis/cost_benefit_summary.csv (pooled rows)
#   Metric: ratio of summed logical runtimes vs SPF (not per-instance mean ratios).

THESIS_FIGURE_BASENAMES: tuple[str, ...] = (
    "thesis_ch6_priority_sensitivity",
    "thesis_ch7_order_vs_soc",
    "thesis_ch8_cglps_vs_ubls",
    "thesis_ch8_search_cost",
)

EXPECTED_SOC_SENSITIVE_COUNT = 7
EXPECTED_MAX_SOC_RANGE = 719
CROSS_MAP_INSTANCE_COUNT = 54


@dataclass(frozen=True, slots=True)
class ThesisMapfFiguresConfig:
    repo_root: Path
    mapf6_analysis_dir: Path
    mapf8_analysis_dir: Path
    mapf9_primary_analysis_dir: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class ThesisMapfFiguresResult:
    output_dir: Path
    files_written: tuple[str, ...]


def default_thesis_mapf_figures_config(repo_root: Path) -> ThesisMapfFiguresConfig:
    results = repo_root / "pathfinding" / "results"
    return ThesisMapfFiguresConfig(
        repo_root=repo_root,
        mapf6_analysis_dir=results / "mapf_priority_strategy_analysis",
        mapf8_analysis_dir=results / "mapf8_cross_map_analysis",
        mapf9_primary_analysis_dir=results / "mapf9_primary_analysis",
        output_dir=repo_root / "docs" / "thesis" / "img",
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _require_row(
    rows: Sequence[dict[str, str]],
    *,
    predicate: Any,
    description: str,
) -> dict[str, str]:
    matches = [row for row in rows if predicate(row)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one row for {description}, found {len(matches)}")
    return matches[0]


def _short_instance_label(instance_id: str) -> str:
    return instance_id.removeprefix("AR0204SR_")


def _human_instance_label(instance_id: str) -> str:
    """Compact thesis label, e.g. n20_high_001 -> n=20, HIGH, 001."""
    short = _short_instance_label(instance_id)
    parts = short.split("_")
    if len(parts) == 3 and parts[0].startswith("n") and parts[0][1:].isdigit():
        return f"n={parts[0][1:]}, {parts[1].upper()}, {parts[2]}"
    return short


def load_ch6_priority_sensitivity_data(
    config: ThesisMapfFiguresConfig,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    sensitivity_path = config.mapf6_analysis_dir / "random_sensitivity.csv"
    instance_path = config.mapf6_analysis_dir / "instance_level_summary.csv"
    sensitivity_rows = _read_csv_rows(sensitivity_path)
    instance_rows = _read_csv_rows(instance_path)

    sensitive = [row for row in sensitivity_rows if row["observed_priority_sensitive"] == "True"]
    if len(sensitive) != EXPECTED_SOC_SENSITIVE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SOC_SENSITIVE_COUNT} SoC-sensitive instances, got {len(sensitive)}"
        )

    max_range = max(int(row["random_soc_range"]) for row in sensitivity_rows)
    if max_range != EXPECTED_MAX_SOC_RANGE:
        raise ValueError(f"Expected max observed SoC range {EXPECTED_MAX_SOC_RANGE}, got {max_range}")

    sensitive.sort(key=lambda row: (-int(row["random_soc_range"]), row["instance_id"]))
    instance_lookup = {row["instance_id"]: row for row in instance_rows}
    return sensitive, [instance_lookup[row["instance_id"]] for row in sensitive]


def load_ch7_order_vs_soc_data(config: ThesisMapfFiguresConfig) -> list[dict[str, int | str]]:
    order_rows = _read_csv_rows(config.mapf8_analysis_dir / "spf_cd_order_summary.csv")
    pairwise_rows = _read_csv_rows(config.mapf8_analysis_dir / "pairwise_summary.csv")

    pooled_order = _require_row(
        order_rows,
        predicate=lambda row: row["catalogue"] == "pooled_cross_map" and row["scope"] == "aggregate",
        description="pooled_cross_map aggregate in spf_cd_order_summary.csv",
    )
    cdf_h_pairwise = _require_row(
        pairwise_rows,
        predicate=lambda row: row["catalogue"] == "pooled_cross_map"
        and row["comparison"] == "cdf_h_vs_spf",
        description="pooled cdf_h_vs_spf in pairwise_summary.csv",
    )
    cdf_l_pairwise = _require_row(
        pairwise_rows,
        predicate=lambda row: row["catalogue"] == "pooled_cross_map"
        and row["comparison"] == "cdf_l_vs_spf",
        description="pooled cdf_l_vs_spf in pairwise_summary.csv",
    )

    total = int(pooled_order["instance_count"])
    if total != CROSS_MAP_INSTANCE_COUNT:
        raise ValueError(f"Expected {CROSS_MAP_INSTANCE_COUNT} cross-map instances, got {total}")

    points: list[dict[str, int | str]] = []
    for strategy, order_key, pairwise in (
        ("CDF-H", "cdf_h_different_order_count", cdf_h_pairwise),
        ("CDF-L", "cdf_l_different_order_count", cdf_l_pairwise),
    ):
        order_changed = int(pooled_order[order_key])
        equal_soc = int(pairwise["equal_count"])
        soc_changed = total - equal_soc
        points.append(
            {
                "strategy": strategy,
                "order_changed": order_changed,
                "soc_changed": soc_changed,
                "total": total,
            }
        )

    cdf_h = points[0]
    cdf_l = points[1]
    assert cdf_h["order_changed"] == 36, f"CDF-H order changed: {cdf_h['order_changed']}"
    assert cdf_h["soc_changed"] == 2, f"CDF-H SoC changed: {cdf_h['soc_changed']}"
    assert cdf_l["order_changed"] == 34, f"CDF-L order changed: {cdf_l['order_changed']}"
    assert cdf_l["soc_changed"] == 3, f"CDF-L SoC changed: {cdf_l['soc_changed']}"

    order_no_soc_h = int(pooled_order["cdf_h_order_diff_no_soc_diff_count"])
    order_no_soc_l = int(pooled_order["cdf_l_order_diff_no_soc_diff_count"])
    assert order_no_soc_h == 34, f"CDF-H order without SoC change: {order_no_soc_h}"
    assert order_no_soc_l == 31, f"CDF-L order without SoC change: {order_no_soc_l}"

    return points


def load_ch8_matched_budget_data(config: ThesisMapfFiguresConfig) -> list[dict[str, int | str]]:
    soc_rows = _read_csv_rows(config.mapf9_primary_analysis_dir / "soc_pairwise_summary.csv")
    lex_rows = _read_csv_rows(
        config.mapf9_primary_analysis_dir / "lexicographic_pairwise_summary.csv"
    )

    soc = _require_row(
        soc_rows,
        predicate=lambda row: row["scope"] == "pooled" and row["comparison"] == "CGLPS_vs_UBLS_soc",
        description="pooled CGLPS_vs_UBLS_soc",
    )
    lex = _require_row(
        lex_rows,
        predicate=lambda row: row["scope"] == "pooled" and row["comparison"] == "CGLPS_vs_UBLS",
        description="pooled CGLPS_vs_UBLS lexicographic",
    )

    soc_counts = (
        int(soc["cglps_better_count"]),
        int(soc["equal_count"]),
        int(soc["ubls_better_count"]),
    )
    lex_counts = (
        int(lex["left_better_count"]),
        int(lex["equal_count"]),
        int(lex["right_better_count"]),
    )
    assert soc_counts == (1, 53, 0), f"SoC matched-budget counts: {soc_counts}"
    assert lex_counts == (4, 50, 0), f"Lex matched-budget counts: {lex_counts}"

    return [
        {
            "criterion": "SoC",
            "cglps_better": soc_counts[0],
            "equal": soc_counts[1],
            "ubls_better": soc_counts[2],
        },
        {
            "criterion": "SoC → makespan",
            "cglps_better": lex_counts[0],
            "equal": lex_counts[1],
            "ubls_better": lex_counts[2],
        },
    ]


def load_ch8_search_cost_data(config: ThesisMapfFiguresConfig) -> list[dict[str, float | int | str]]:
    rows = _read_csv_rows(config.mapf9_primary_analysis_dir / "cost_benefit_summary.csv")
    points: list[dict[str, float | int | str]] = []
    for method in ("SPF", "CGLPS", "UBLS"):
        row = _require_row(
            rows,
            predicate=lambda item, m=method: item["scope"] == "pooled" and item["method"] == m,
            description=f"pooled {method} in cost_benefit_summary.csv",
        )
        points.append(
            {
                "method": method,
                "logical_pp_candidates": int(row["logical_pp_candidates"]),
                "logical_runtime_ratio_vs_spf": float(row["logical_runtime_ratio_vs_spf"]),
            }
        )

    spf, cglps, ubls = points
    assert float(spf["logical_runtime_ratio_vs_spf"]) == 1.0
    assert abs(float(cglps["logical_runtime_ratio_vs_spf"]) - 1.945) < 0.01
    assert abs(float(ubls["logical_runtime_ratio_vs_spf"]) - 1.934) < 0.01
    assert int(spf["logical_pp_candidates"]) == 54
    assert int(cglps["logical_pp_candidates"]) == 129
    assert int(ubls["logical_pp_candidates"]) == 129

    return points


def _apply_thesis_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": ":",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_thesis_figure(fig: Any, output_dir: Path, basename: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{basename}.png"
    pdf_path = output_dir / f"{basename}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path


def render_ch6_priority_sensitivity(
    output_dir: Path,
    sensitivity_rows: Sequence[dict[str, str]],
    instance_rows: Sequence[dict[str, str]],
) -> tuple[str, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    _apply_thesis_style()
    n = len(sensitivity_rows)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    y = np.arange(n)
    labels = [_human_instance_label(row["instance_id"]) for row in sensitivity_rows]

    for index, (sens_row, inst_row) in enumerate(zip(sensitivity_rows, instance_rows, strict=True)):
        x_min = int(sens_row["random_soc_min"])
        x_max = int(sens_row["random_soc_max"])
        x_med = float(sens_row["random_soc_median"])
        ax.hlines(index, x_min, x_max, color="#666666", linewidth=2.8, alpha=0.85, zorder=1)
        ax.scatter(x_med, index, color="#444444", marker="|", s=120, linewidths=2.0, zorder=2)
        ax.scatter(
            int(inst_row["spf_soc"]),
            index,
            marker="s",
            facecolors="white",
            edgecolors="#333333",
            s=42,
            linewidths=1.1,
            zorder=3,
        )
        ax.scatter(
            int(inst_row["lpf_soc"]),
            index,
            marker="^",
            facecolors="white",
            edgecolors="#777777",
            s=48,
            linewidths=1.1,
            zorder=3,
        )

    ax.scatter([], [], color="#666666", linewidth=2.8, label="Losowe K=10: obserw. min–max")
    ax.scatter([], [], color="#444444", marker="|", label="Losowe K=10: mediana")
    ax.scatter(
        [],
        [],
        marker="s",
        facecolors="white",
        edgecolors="#333333",
        label="SPF",
    )
    ax.scatter(
        [],
        [],
        marker="^",
        facecolors="white",
        edgecolors="#777777",
        label="LPF",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("SoC")
    ax.set_ylabel("Instancja wrażliwa (K=10)")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True,
        framealpha=0.92,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    fig.subplots_adjust(top=0.86)
    _save_thesis_figure(fig, output_dir, THESIS_FIGURE_BASENAMES[0])
    plt.close(fig)
    return (f"{THESIS_FIGURE_BASENAMES[0]}.png", f"{THESIS_FIGURE_BASENAMES[0]}.pdf")


def render_ch7_order_vs_soc(
    output_dir: Path,
    points: Sequence[dict[str, int | str]],
) -> tuple[str, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    _apply_thesis_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    x = np.arange(len(points))
    width = 0.36
    order_vals = [int(point["order_changed"]) for point in points]
    soc_vals = [int(point["soc_changed"]) for point in points]
    total = int(points[0]["total"])

    order_bars = ax.bar(
        x - width / 2,
        order_vals,
        width=width,
        label="Zmiana kolejności względem SPF",
        color="#dddddd",
        edgecolor="#333333",
        linewidth=0.9,
        hatch="",
        zorder=2,
    )
    soc_bars = ax.bar(
        x + width / 2,
        soc_vals,
        width=width,
        label="Zmiana SoC względem SPF",
        color="#888888",
        edgecolor="#333333",
        linewidth=0.9,
        hatch="///",
        zorder=2,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(point["strategy"]) for point in points])
    ax.set_ylabel(f"Liczba instancji (z {total})")
    ax.set_xlabel("Strategia priorytetu")
    ax.set_ylim(0, total + 4)
    ax.set_yticks(range(0, total + 1, 6))

    for idx, (order_val, soc_val) in enumerate(zip(order_vals, soc_vals, strict=True)):
        ax.text(
            x[idx] - width / 2,
            order_val + 0.8,
            f"{order_val}/{total}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.text(
            x[idx] + width / 2,
            soc_val + 0.8,
            f"{soc_val}/{total}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    fig.tight_layout()
    _save_thesis_figure(fig, output_dir, THESIS_FIGURE_BASENAMES[1])
    plt.close(fig)
    return (f"{THESIS_FIGURE_BASENAMES[1]}.png", f"{THESIS_FIGURE_BASENAMES[1]}.pdf")


def render_ch8_cglps_vs_ubls(
    output_dir: Path,
    points: Sequence[dict[str, int | str]],
) -> tuple[str, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import numpy as np

    _apply_thesis_style()
    bar_label_effects = [pe.withStroke(linewidth=2.8, foreground="white")]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    y = np.arange(len(points))
    left = np.array([int(point["cglps_better"]) for point in points], dtype=float)
    mid = np.array([int(point["equal"]) for point in points], dtype=float)
    right = np.array([int(point["ubls_better"]) for point in points], dtype=float)

    ax.barh(
        y,
        left,
        color="#cccccc",
        edgecolor="#333333",
        linewidth=0.8,
        height=0.52,
        label="CGLPS lepszy",
    )
    ax.barh(
        y,
        mid,
        left=left,
        color="#eeeeee",
        edgecolor="#333333",
        linewidth=0.8,
        height=0.52,
        hatch="...",
        label="Remis",
    )
    ax.barh(
        y,
        right,
        left=left + mid,
        color="#999999",
        edgecolor="#333333",
        linewidth=0.8,
        height=0.52,
        hatch="xxx",
        label="UBLS lepszy",
    )

    ax.set_yticks(y)
    ax.set_yticklabels([str(point["criterion"]) for point in points])
    ax.invert_yaxis()
    ax.set_xlabel("Liczba instancji (n = 54, dopasowany budżet)")
    ax.set_xlim(0, 54)
    ax.set_xticks(range(0, 55, 6))

    for idx, point in enumerate(points):
        if point["cglps_better"]:
            ax.text(
                int(point["cglps_better"]) / 2,
                idx,
                str(point["cglps_better"]),
                ha="center",
                va="center",
                fontsize=8,
                fontweight="semibold",
                color="black",
                path_effects=bar_label_effects,
            )
        ax.text(
            int(point["cglps_better"]) + int(point["equal"]) / 2,
            idx,
            str(point["equal"]),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="semibold",
            color="black",
            path_effects=bar_label_effects,
        )

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
        framealpha=0.92,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    fig.subplots_adjust(top=0.82)
    _save_thesis_figure(fig, output_dir, THESIS_FIGURE_BASENAMES[2])
    plt.close(fig)
    return (f"{THESIS_FIGURE_BASENAMES[2]}.png", f"{THESIS_FIGURE_BASENAMES[2]}.pdf")


def render_ch8_search_cost(
    output_dir: Path,
    points: Sequence[dict[str, float | int | str]],
) -> tuple[str, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_thesis_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    methods = [str(point["method"]) for point in points]
    ratios = [float(point["logical_runtime_ratio_vs_spf"]) for point in points]
    hatches = ["", "///", "xxx"]
    edgecolors = ["#333333", "#333333", "#333333"]
    facecolors = ["#eeeeee", "#cccccc", "#aaaaaa"]

    bars = ax.bar(
        methods,
        ratios,
        width=0.55,
        color=facecolors,
        edgecolor=edgecolors,
        linewidth=0.9,
        hatch=hatches,
    )
    ax.axhline(1.0, color="#666666", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_ylabel("Logiczny czas wykonania wzgl. SPF")
    ax.set_xlabel("Metoda")
    ax.set_ylim(0, max(ratios) * 1.15)

    for bar, point in zip(bars, points, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{float(point['logical_runtime_ratio_vs_spf']):.3f}×",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    candidate_text = (
        "Logiczne kandydaty PP: "
        f"SPF {points[0]['logical_pp_candidates']}; "
        f"CGLPS {points[1]['logical_pp_candidates']}; "
        f"UBLS {points[2]['logical_pp_candidates']}"
    )
    ax.text(
        0.5,
        -0.18,
        candidate_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        color="#444444",
    )
    fig.subplots_adjust(bottom=0.22)
    _save_thesis_figure(fig, output_dir, THESIS_FIGURE_BASENAMES[3])
    plt.close(fig)
    return (f"{THESIS_FIGURE_BASENAMES[3]}.png", f"{THESIS_FIGURE_BASENAMES[3]}.pdf")


def verify_thesis_mapf_figure_sources(config: ThesisMapfFiguresConfig) -> None:
    """Load frozen data and run assertions without writing figures."""
    load_ch6_priority_sensitivity_data(config)
    load_ch7_order_vs_soc_data(config)
    load_ch8_matched_budget_data(config)
    load_ch8_search_cost_data(config)


def run_thesis_mapf_figures(config: ThesisMapfFiguresConfig) -> ThesisMapfFiguresResult:
    sensitive_rows, instance_rows = load_ch6_priority_sensitivity_data(config)
    ch7_points = load_ch7_order_vs_soc_data(config)
    ch8_budget_points = load_ch8_matched_budget_data(config)
    ch8_cost_points = load_ch8_search_cost_data(config)

    written: list[str] = []
    written.extend(
        render_ch6_priority_sensitivity(config.output_dir, sensitive_rows, instance_rows)
    )
    written.extend(render_ch7_order_vs_soc(config.output_dir, ch7_points))
    written.extend(render_ch8_cglps_vs_ubls(config.output_dir, ch8_budget_points))
    written.extend(render_ch8_search_cost(config.output_dir, ch8_cost_points))

    return ThesisMapfFiguresResult(output_dir=config.output_dir, files_written=tuple(written))
