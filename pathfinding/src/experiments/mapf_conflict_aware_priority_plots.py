from __future__ import annotations

import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

STRATEGY_COLORS: dict[str, str] = {
    "spf": "#55A868",
    "cdf_h": "#4C72B0",
    "cdf_l": "#8172B2",
    "spf_cd": "#CCB974",
}


def _numeric(value: Any) -> float | int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    return float(value)


def _numeric_int(value: Any) -> int | None:
    parsed = _numeric(value)
    if parsed is None:
        return None
    return int(parsed)


def short_instance_label(instance_id: str) -> str:
    return instance_id.removeprefix("AR0204SR_")


def _apply_plot_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_figure(fig: Any, plots_dir: Path, basename: str) -> tuple[Path, Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    png_path = plots_dir / f"{basename}.png"
    pdf_path = plots_dir / f"{basename}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path


def _mapf7_differing_instances(
    instance_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    differing: list[dict[str, Any]] = []
    for row in instance_rows:
        socs = {
            _numeric_int(row.get("cdf_h_soc")),
            _numeric_int(row.get("cdf_l_soc")),
            _numeric_int(row.get("spf_cd_soc")),
        }
        socs.discard(None)
        if len(socs) > 1:
            differing.append(row)
    differing.sort(
        key=lambda item: (
            max(
                _numeric_int(item.get("cdf_h_soc")) or 0,
                _numeric_int(item.get("cdf_l_soc")) or 0,
                _numeric_int(item.get("spf_cd_soc")) or 0,
            )
            - min(
                _numeric_int(item.get("cdf_h_soc")) or 0,
                _numeric_int(item.get("cdf_l_soc")) or 0,
                _numeric_int(item.get("spf_cd_soc")) or 0,
            ),
            str(item["instance_id"]),
        ),
        reverse=True,
    )
    return differing


def render_conflict_aware_priority_plots(
    output_dir: Path,
    instance_rows: Sequence[dict[str, Any]],
    *,
    cdf_direction_rows: Sequence[dict[str, Any]],
    conflict_diagnostic_rows: Sequence[dict[str, Any]],
    runtime_aggregate: dict[str, Any],
) -> tuple[str, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    _apply_plot_style()
    plots_dir = output_dir / "plots"
    written: list[str] = []

    differing = _mapf7_differing_instances(instance_rows)

    # Figure 1 — MAPF-7 SoC on differing instances
    fig, ax = plt.subplots(figsize=(max(8, len(differing) * 0.55), 5))
    x = np.arange(len(differing))
    width = 0.18
    series = (
        (-1.5, "spf_soc", "SPF", STRATEGY_COLORS["spf"]),
        (-0.5, "cdf_h_soc", "CDF-H", STRATEGY_COLORS["cdf_h"]),
        (0.5, "cdf_l_soc", "CDF-L", STRATEGY_COLORS["cdf_l"]),
        (1.5, "spf_cd_soc", "SPF+CD", STRATEGY_COLORS["spf_cd"]),
    )
    for offset, key, label, color in series:
        values = [_numeric_int(row.get(key)) or 0 for row in differing]
        ax.bar(x + offset * width, values, width=width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [short_instance_label(str(row["instance_id"])) for row in differing],
        rotation=45,
        ha="right",
    )
    ax.set_ylabel("SoC")
    ax.set_title(
        "MAPF-7 strategy SoC on instances with differing MAPF-7 outcomes\n"
        f"({len(differing)} of {len(instance_rows)} instances; SPF shown as reference)"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    _save_figure(fig, plots_dir, "fig01_mapf7_soc_on_differing_instances")
    plt.close(fig)
    written.extend(
        [
            "fig01_mapf7_soc_on_differing_instances.png",
            "fig01_mapf7_soc_on_differing_instances.pdf",
        ]
    )

    # Figure 2 — CDF-H vs CDF-L paired difference
    cdf_diff_rows = [
        row
        for row in cdf_direction_rows
        if _numeric_int(row.get("cdf_h_soc")) != _numeric_int(row.get("cdf_l_soc"))
    ]
    fig_height = max(3.5, 0.45 * len(cdf_diff_rows) + 1.5)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    y = np.arange(len(cdf_diff_rows))
    diffs = [_numeric_int(row.get("soc_diff_cdf_h_minus_cdf_l")) or 0 for row in cdf_diff_rows]
    colors = [
        STRATEGY_COLORS["cdf_h"] if value < 0 else STRATEGY_COLORS["cdf_l"] if value > 0 else "#999999"
        for value in diffs
    ]
    ax.barh(y, diffs, color=colors, height=0.65)
    ax.axvline(0.0, color="black", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [short_instance_label(str(row["instance_id"])) for row in cdf_diff_rows]
    )
    ax.set_xlabel("CDF-H SoC - CDF-L SoC")
    ax.set_title(
        "CDF-H vs CDF-L direction test on differing instances\n"
        "Negative = CDF-H better; Positive = CDF-L better"
    )
    fig.tight_layout()
    _save_figure(fig, plots_dir, "fig02_cdf_h_vs_cdf_l_soc_difference")
    plt.close(fig)
    written.extend(
        [
            "fig02_cdf_h_vs_cdf_l_soc_difference.png",
            "fig02_cdf_h_vs_cdf_l_soc_difference.pdf",
        ]
    )

    # Figure 3 — conflict structure vs MAPF-7 SoC range
    equal_rows = [row for row in conflict_diagnostic_rows if row.get("mapf7_soc_all_equal")]
    differ_rows = [row for row in conflict_diagnostic_rows if not row.get("mapf7_soc_all_equal")]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    groups = ["All MAPF-7\nSoC equal", "MAPF-7 SoC\ndiffers"]
    max_deg_equal = [
        _numeric(row.get("max_degree")) for row in equal_rows if _numeric(row.get("max_degree")) is not None
    ]
    max_deg_differ = [
        _numeric(row.get("max_degree")) for row in differ_rows if _numeric(row.get("max_degree")) is not None
    ]
    pair_equal = [
        _numeric(row.get("independent_conflict_pair_count"))
        for row in equal_rows
        if _numeric(row.get("independent_conflict_pair_count")) is not None
    ]
    pair_differ = [
        _numeric(row.get("independent_conflict_pair_count"))
        for row in differ_rows
        if _numeric(row.get("independent_conflict_pair_count")) is not None
    ]
    x = np.array([0, 1])
    width = 0.35
    ax.bar(
        x - width / 2,
        [
            statistics.mean(max_deg_equal) if max_deg_equal else 0,
            statistics.mean(max_deg_differ) if max_deg_differ else 0,
        ],
        width=width,
        label="Mean max degree",
        color="#4C72B0",
    )
    ax.bar(
        x + width / 2,
        [
            statistics.mean(pair_equal) if pair_equal else 0,
            statistics.mean(pair_differ) if pair_differ else 0,
        ],
        width=width,
        label="Mean conflict pairs",
        color="#C44E52",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Mean graph statistic")
    ax.set_title(
        "Conflict-graph structure vs MAPF-7 priority sensitivity\n"
        "(descriptive; not causal; n equal="
        f"{len(equal_rows)}, differ={len(differ_rows)})"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    _save_figure(fig, plots_dir, "fig03_conflict_structure_vs_priority_effect")
    plt.close(fig)
    written.extend(
        [
            "fig03_conflict_structure_vs_priority_effect.png",
            "fig03_conflict_structure_vs_priority_effect.pdf",
        ]
    )

    # Figure 4 — runtime composition
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["SPF", "CDF-H", "CDF-L", "SPF+CD"]
    ordering_keys = (
        "mean_spf_ordering_s",
        "mean_cdf_h_ordering_s",
        "mean_cdf_l_ordering_s",
        "mean_spf_cd_ordering_s",
    )
    pp_keys = (
        "mean_spf_pp_s",
        "mean_cdf_h_pp_s",
        "mean_cdf_l_pp_s",
        "mean_spf_cd_pp_s",
    )
    x = np.arange(len(labels))
    width = 0.55
    ordering = [float(runtime_aggregate.get(key, 0) or 0) for key in ordering_keys]
    pp = [float(runtime_aggregate.get(key, 0) or 0) for key in pp_keys]
    ax.bar(x, ordering, width, label="Ordering time", color="#B0B0B0")
    ax.bar(
        x,
        pp,
        width,
        bottom=ordering,
        label="PP time",
        color=[STRATEGY_COLORS["spf"], STRATEGY_COLORS["cdf_h"], STRATEGY_COLORS["cdf_l"], STRATEGY_COLORS["spf_cd"]],
    )
    totals = [o + p for o, p in zip(ordering, pp, strict=True)]
    for index, total in enumerate(totals):
        ax.text(index, total + max(totals) * 0.02, f"{total:.1f} s", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean runtime (s)")
    ax.set_title(
        "Mean runtime composition\n"
        "(SPF from MAPF-6; MAPF-7 from separate run; total = ordering + PP)"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save_figure(fig, plots_dir, "fig04_runtime_composition")
    plt.close(fig)
    written.extend(
        [
            "fig04_runtime_composition.png",
            "fig04_runtime_composition.pdf",
        ]
    )

    return tuple(written)
