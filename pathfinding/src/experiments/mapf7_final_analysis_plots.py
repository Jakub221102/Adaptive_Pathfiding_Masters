from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pathfinding.src.experiments.mapf_conflict_aware_priority_plots import (
    STRATEGY_COLORS,
    _apply_plot_style,
    _numeric,
    _numeric_int,
    _save_figure,
    short_instance_label,
)

CATALOGUE_PRIMARY = "primary"
CATALOGUE_HELD_OUT = "held_out"

FIGURE_DPI = 300
PRIMARY_MARKER = "o"
HELD_OUT_MARKER = "^"
PRIMARY_COLOR = "#4C72B0"
HELD_OUT_COLOR = "#C44E52"
ORDERING_COLOR = "#B0B0B0"
PP_COLOR = "#4C72B0"


def instance_full_soc_range(row: dict[str, Any]) -> int:
    """SoC range across CDF-H, CDF-L, SPF+CD, and SPF for one instance."""
    socs = [
        _numeric_int(row.get("cdf_h_soc")),
        _numeric_int(row.get("cdf_l_soc")),
        _numeric_int(row.get("spf_cd_soc")),
        _numeric_int(row.get("spf_soc")),
    ]
    values = [value for value in socs if value is not None]
    if not values:
        return 0
    return max(values) - min(values)


def build_fig01_cdf_h_minus_spf_points(
    instance_rows: Sequence[dict[str, Any]],
    *,
    catalogue: str,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in instance_rows:
        cdf_h = _numeric_int(row.get("cdf_h_soc"))
        spf = _numeric_int(row.get("spf_soc"))
        if cdf_h is None or spf is None:
            continue
        points.append(
            {
                "catalogue": catalogue,
                "instance_id": row["instance_id"],
                "soc_diff": cdf_h - spf,
            }
        )
    return points


def build_fig04_scatter_points(
    primary_instance_rows: Sequence[dict[str, Any]],
    heldout_instance_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for catalogue, rows in (
        (CATALOGUE_PRIMARY, primary_instance_rows),
        (CATALOGUE_HELD_OUT, heldout_instance_rows),
    ):
        for row in rows:
            max_degree = _numeric(row.get("max_degree"))
            points.append(
                {
                    "catalogue": catalogue,
                    "instance_id": row["instance_id"],
                    "max_degree": float(max_degree) if max_degree is not None else 0.0,
                    "soc_range": instance_full_soc_range(row),
                }
            )
    return points


def count_sensitive_by_interaction(
    sensitive_rows: Sequence[dict[str, Any]],
    *,
    catalogue: str,
) -> dict[str, int]:
    counts = {"low": 0, "medium": 0, "high": 0}
    for row in sensitive_rows:
        if row.get("catalogue") != catalogue:
            continue
        level = str(row.get("interaction_level", "")).lower()
        if level in counts:
            counts[level] += 1
    return counts


def render_mapf7_final_analysis_plots(
    output_dir: Path,
    *,
    primary_instance_rows: Sequence[dict[str, Any]],
    heldout_instance_rows: Sequence[dict[str, Any]],
    primary_pairwise: Sequence[dict[str, Any]],
    heldout_pairwise: Sequence[dict[str, Any]],
    sensitive_rows: Sequence[dict[str, Any]],
    conflict_structure_rows: Sequence[dict[str, Any]],
    runtime_rows: Sequence[dict[str, Any]],
    primary_instance_count: int,
    heldout_instance_count: int,
) -> tuple[str, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    _apply_plot_style()
    plots_dir = output_dir / "figures"
    written: list[str] = []

    # Figure 1 — instance-level CDF-H − SPF strip plot (separate catalogues)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=False)
    panel_specs = (
        (axes[0], primary_instance_rows, CATALOGUE_PRIMARY, "Primary catalogue", primary_instance_count),
        (axes[1], heldout_instance_rows, CATALOGUE_HELD_OUT, "Held-out catalogue", heldout_instance_count),
    )
    for ax, rows, catalogue, title, instance_count in panel_specs:
        points = build_fig01_cdf_h_minus_spf_points(rows, catalogue=catalogue)
        diffs = [point["soc_diff"] for point in points]
        y_positions = np.arange(len(points))
        ax.scatter(
            diffs,
            y_positions,
            s=36,
            color=PRIMARY_COLOR if catalogue == CATALOGUE_PRIMARY else HELD_OUT_COLOR,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        ax.axvline(0.0, color="black", linewidth=0.9, zorder=1)
        ax.set_xlabel("CDF-H SoC − SPF SoC")
        ax.set_ylabel("Instance index (within catalogue)")
        ax.set_title(f"{title}\n(n={instance_count}; negative = CDF-H better)")
        ax.set_yticks([])
        ax.margins(x=0.08)
    fig.suptitle(
        "CDF-H vs SPF SoC difference by catalogue\n"
        "(instance-level; separate catalogues; no cross-pairing)",
        fontsize=11,
    )
    fig.tight_layout()
    _save_figure(fig, plots_dir, "fig01_cdf_h_minus_spf_by_catalogue")
    plt.close(fig)
    written.extend(
        [
            "fig01_cdf_h_minus_spf_by_catalogue.png",
            "fig01_cdf_h_minus_spf_by_catalogue.pdf",
        ]
    )

    # Figure 2 — CDF-H − CDF-L direction: primary vs held-out (unchanged design)
    primary_cdf_diffs = [
        _numeric_int(row.get("cdf_h_soc")) - _numeric_int(row.get("cdf_l_soc"))
        for row in primary_instance_rows
        if _numeric_int(row.get("cdf_h_soc")) is not None
        and _numeric_int(row.get("cdf_l_soc")) is not None
    ]
    heldout_cdf_diffs = [
        _numeric_int(row.get("cdf_h_soc")) - _numeric_int(row.get("cdf_l_soc"))
        for row in heldout_instance_rows
        if _numeric_int(row.get("cdf_h_soc")) is not None
        and _numeric_int(row.get("cdf_l_soc")) is not None
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, diffs, title in (
        (axes[0], primary_cdf_diffs, "Primary"),
        (axes[1], heldout_cdf_diffs, "Held-out"),
    ):
        if diffs:
            colors = [
                STRATEGY_COLORS["cdf_h"] if value < 0 else STRATEGY_COLORS["cdf_l"] if value > 0 else "#999999"
                for value in diffs
            ]
            ax.bar(range(len(diffs)), diffs, color=colors, width=0.7)
            ax.axhline(0.0, color="black", linewidth=0.9)
        ax.set_xlabel("Instance index (within catalogue)")
        ax.set_ylabel("CDF-H SoC − CDF-L SoC")
        ax.set_title(f"{title}: direction test\n(negative = CDF-H better)")
    fig.suptitle("CDF-H vs CDF-L SoC difference: primary vs held-out", fontsize=11)
    fig.tight_layout()
    _save_figure(fig, plots_dir, "fig02_cdf_h_minus_cdf_l_by_catalogue")
    plt.close(fig)
    written.extend(
        [
            "fig02_cdf_h_minus_cdf_l_by_catalogue.png",
            "fig02_cdf_h_minus_cdf_l_by_catalogue.pdf",
        ]
    )

    # Figure 3 — SoC-sensitive instances only (unchanged design)
    sensitive_primary = [row for row in sensitive_rows if row["catalogue"] == CATALOGUE_PRIMARY]
    sensitive_heldout = [row for row in sensitive_rows if row["catalogue"] == CATALOGUE_HELD_OUT]
    all_sensitive = sensitive_primary + sensitive_heldout
    fig, ax = plt.subplots(figsize=(max(8, len(all_sensitive) * 0.65), 5.5))
    x = np.arange(len(all_sensitive))
    width = 0.18
    series = (
        (-1.5, "spf_soc", "SPF", STRATEGY_COLORS["spf"]),
        (-0.5, "cdf_h_soc", "CDF-H", STRATEGY_COLORS["cdf_h"]),
        (0.5, "cdf_l_soc", "CDF-L", STRATEGY_COLORS["cdf_l"]),
        (1.5, "spf_cd_soc", "SPF+CD", STRATEGY_COLORS["spf_cd"]),
    )
    for offset, key, label, color in series:
        values = [_numeric_int(row.get(key)) or 0 for row in all_sensitive]
        ax.bar(x + offset * width, values, width=width, label=label, color=color)
    ax.set_xticks(x)
    labels = []
    for row in all_sensitive:
        prefix = "P" if row["catalogue"] == CATALOGUE_PRIMARY else "H"
        labels.append(f"{prefix}:{short_instance_label(str(row['instance_id']))}")
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("SoC")
    ax.set_title(
        "SoC on MAPF-7-sensitive instances only\n"
        f"(P=primary n={len(sensitive_primary)}, H=held-out n={len(sensitive_heldout)})"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    _save_figure(fig, plots_dir, "fig03_soc_on_sensitive_instances")
    plt.close(fig)
    written.extend(
        [
            "fig03_soc_on_sensitive_instances.png",
            "fig03_soc_on_sensitive_instances.pdf",
        ]
    )

    # Figure 4 — instance-level max degree vs full SoC range
    scatter_points = build_fig04_scatter_points(primary_instance_rows, heldout_instance_rows)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for catalogue, marker, color, label in (
        (CATALOGUE_PRIMARY, PRIMARY_MARKER, PRIMARY_COLOR, "Primary"),
        (CATALOGUE_HELD_OUT, HELD_OUT_MARKER, HELD_OUT_COLOR, "Held-out"),
    ):
        subset = [point for point in scatter_points if point["catalogue"] == catalogue]
        x_values = [point["max_degree"] for point in subset]
        y_values = [point["soc_range"] for point in subset]
        ax.scatter(
            x_values,
            y_values,
            marker=marker,
            s=52,
            color=color,
            alpha=0.72,
            edgecolors="white",
            linewidths=0.6,
            label=label,
            zorder=3,
        )
    ax.set_xlabel("Max conflict degree")
    ax.set_ylabel("SoC range (CDF-H / CDF-L / SPF+CD / SPF)")
    ax.set_title(
        "Conflict structure vs priority sensitivity\n"
        "Instance-level descriptive relationship; not causal"
    )
    ax.legend(loc="upper left", framealpha=0.95)
    ax.set_ylim(bottom=-5)
    fig.tight_layout()
    _save_figure(fig, plots_dir, "fig04_conflict_structure_vs_priority_effect")
    plt.close(fig)
    written.extend(
        [
            "fig04_conflict_structure_vs_priority_effect.png",
            "fig04_conflict_structure_vs_priority_effect.pdf",
        ]
    )

    # Figure 5 — Held-out runtime composition (legend moved clear of data)
    heldout_runtime_agg = next(
        (
            row
            for row in runtime_rows
            if row.get("metric_group") == "runtime_aggregate"
            and row.get("catalogue") == CATALOGUE_HELD_OUT
        ),
        {},
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    strategy_labels = ["SPF", "CDF-H", "CDF-L", "SPF+CD"]
    prefixes = ("spf", "cdf_h", "cdf_l", "spf_cd")
    x = np.arange(len(strategy_labels))
    width = 0.55
    ordering = [
        float(heldout_runtime_agg.get(f"mean_{prefix}_ordering_ms", 0)) / 1000.0
        for prefix in prefixes
    ]
    pp = [
        float(heldout_runtime_agg.get(f"mean_{prefix}_pp_ms", 0)) / 1000.0
        for prefix in prefixes
    ]
    ax.bar(x, ordering, width, label="Ordering time", color=ORDERING_COLOR)
    ax.bar(
        x,
        pp,
        width,
        bottom=ordering,
        label="PP time",
        color=PP_COLOR,
    )
    totals = [o + p for o, p in zip(ordering, pp, strict=True)]
    ymax = max(totals) * 1.18
    ax.set_ylim(0, ymax)
    label_offset = max(totals) * 0.025
    for index, total in enumerate(totals):
        ax.text(
            index,
            total + label_offset,
            f"{total:.1f} s",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(strategy_labels)
    ax.set_ylabel("Mean runtime (s)")
    ax.set_title("Held-out runtime composition\n(total = ordering + PP from independent runs)")
    ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.99), framealpha=0.95)
    fig.tight_layout()
    _save_figure(fig, plots_dir, "fig05_heldout_runtime_composition")
    plt.close(fig)
    written.extend(
        [
            "fig05_heldout_runtime_composition.png",
            "fig05_heldout_runtime_composition.pdf",
        ]
    )

    # Figure 6 — Sensitivity by interaction stratum
    levels = ("low", "medium", "high")
    x = np.arange(len(levels))
    width = 0.35
    primary_counts = count_sensitive_by_interaction(sensitive_rows, catalogue=CATALOGUE_PRIMARY)
    heldout_counts = count_sensitive_by_interaction(sensitive_rows, catalogue=CATALOGUE_HELD_OUT)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.bar(
        x - width / 2,
        [primary_counts[level] for level in levels],
        width=width,
        label="Primary sensitive",
        color=PRIMARY_COLOR,
    )
    ax.bar(
        x + width / 2,
        [heldout_counts[level] for level in levels],
        width=width,
        label="Held-out sensitive",
        color=HELD_OUT_COLOR,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([level.upper() for level in levels])
    ax.set_ylabel("Sensitive instance count")
    ax.set_title("SoC-sensitive instances by interaction stratum")
    ax.legend(loc="best")
    fig.text(
        0.5,
        0.01,
        "Interaction strata are descriptive and not independent evidence.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(bottom=0.14)
    _save_figure(fig, plots_dir, "fig06_sensitivity_by_interaction_stratum")
    plt.close(fig)
    written.extend(
        [
            "fig06_sensitivity_by_interaction_stratum.png",
            "fig06_sensitivity_by_interaction_stratum.pdf",
        ]
    )

    return tuple(written)
