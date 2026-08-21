from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STRATEGY_COLORS: dict[str, str] = {
    "fixed": "#4C72B0",
    "random": "#777777",
    "spf": "#55A868",
    "lpf": "#C44E52",
}


@dataclass(frozen=True, slots=True)
class SpfLpfDifferenceRow:
    instance_id: str
    short_label: str
    spf_soc: int
    lpf_soc: int
    soc_diff: int


@dataclass(frozen=True, slots=True)
class MeanRuntimeSeconds:
    spf_ordering_s: float
    spf_pp_s: float
    lpf_ordering_s: float
    lpf_pp_s: float

    @property
    def spf_total_s(self) -> float:
        return self.spf_ordering_s + self.spf_pp_s

    @property
    def lpf_total_s(self) -> float:
        return self.lpf_ordering_s + self.lpf_pp_s


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


def select_spf_lpf_differing_instances(
    instance_rows: Sequence[dict[str, Any]],
) -> tuple[SpfLpfDifferenceRow, ...]:
    differing: list[SpfLpfDifferenceRow] = []
    for row in instance_rows:
        spf_soc = _numeric_int(row.get("spf_soc"))
        lpf_soc = _numeric_int(row.get("lpf_soc"))
        if spf_soc is None or lpf_soc is None or spf_soc == lpf_soc:
            continue
        instance_id = str(row["instance_id"])
        differing.append(
            SpfLpfDifferenceRow(
                instance_id=instance_id,
                short_label=short_instance_label(instance_id),
                spf_soc=spf_soc,
                lpf_soc=lpf_soc,
                soc_diff=spf_soc - lpf_soc,
            )
        )

    differing.sort(
        key=lambda item: (
            abs(item.soc_diff),
            item.short_label,
        ),
        reverse=True,
    )
    return tuple(differing)


def select_top_priority_sensitive_instances(
    sensitivity_rows: Sequence[dict[str, Any]],
    *,
    limit: int = 4,
) -> tuple[dict[str, Any], ...]:
    sensitive_rows = [
        row
        for row in sensitivity_rows
        if bool(row.get("observed_priority_sensitive"))
    ]
    sensitive_rows.sort(
        key=lambda row: (
            _numeric(row.get("random_soc_range")) or 0,
            str(row.get("instance_id", "")),
        ),
        reverse=True,
    )
    return tuple(sensitive_rows[:limit])


def compute_mean_runtime_seconds(
    instance_rows: Sequence[dict[str, Any]],
) -> MeanRuntimeSeconds:
    import statistics

    spf_order = [
        float(_numeric(row["spf_ordering_time_ms"]))
        for row in instance_rows
        if _numeric(row.get("spf_ordering_time_ms")) is not None
    ]
    spf_pp = [
        float(_numeric(row["spf_pp_time_ms"]))
        for row in instance_rows
        if _numeric(row.get("spf_pp_time_ms")) is not None
    ]
    lpf_order = [
        float(_numeric(row["lpf_ordering_time_ms"]))
        for row in instance_rows
        if _numeric(row.get("lpf_ordering_time_ms")) is not None
    ]
    lpf_pp = [
        float(_numeric(row["lpf_pp_time_ms"]))
        for row in instance_rows
        if _numeric(row.get("lpf_pp_time_ms")) is not None
    ]

    return MeanRuntimeSeconds(
        spf_ordering_s=statistics.mean(spf_order) / 1000.0,
        spf_pp_s=statistics.mean(spf_pp) / 1000.0,
        lpf_ordering_s=statistics.mean(lpf_order) / 1000.0,
        lpf_pp_s=statistics.mean(lpf_pp) / 1000.0,
    )


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


def render_priority_strategy_plots(
    output_dir: Path,
    instance_rows: Sequence[dict[str, Any]],
    sensitivity_rows: Sequence[dict[str, Any]],
) -> tuple[str, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    _apply_plot_style()
    plots_dir = output_dir / "plots"
    written: list[str] = []

    sensitive_rows = [
        row for row in sensitivity_rows if row.get("observed_priority_sensitive")
    ]
    sensitive_rows.sort(
        key=lambda row: (
            _numeric(row.get("random_soc_range")) or 0,
            str(row.get("instance_id", "")),
        ),
        reverse=True,
    )
    focus_rows = sensitive_rows if sensitive_rows else list(instance_rows)
    lookup = {row["instance_id"]: row for row in instance_rows}

    # Figure 1 — categorical markers only, no connecting lines
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(focus_rows))
    fixed_values: list[int] = []
    spf_values: list[int] = []
    lpf_values: list[int] = []
    labels: list[str] = []

    for index, row in enumerate(focus_rows):
        instance = lookup[row["instance_id"]]
        labels.append(short_instance_label(str(row["instance_id"])))
        y_min = _numeric_int(instance.get("random_soc_min"))
        y_max = _numeric_int(instance.get("random_soc_max"))
        y_med = _numeric_int(instance.get("random_soc_median"))
        if y_min is not None and y_max is not None:
            ax.vlines(index, y_min, y_max, color="#888888", linewidth=3, alpha=0.8)
        if y_med is not None:
            ax.scatter(
                index,
                y_med,
                color=STRATEGY_COLORS["random"],
                marker="_",
                s=220,
                linewidths=2,
                zorder=3,
            )
        fixed_values.append(_numeric_int(instance["fixed_soc"]) or 0)
        spf_values.append(_numeric_int(instance["spf_soc"]) or 0)
        lpf_values.append(_numeric_int(instance["lpf_soc"]) or 0)

    ax.scatter([], [], color="#888888", label="Random K=10 sampled min-max")
    ax.scatter(
        [],
        [],
        color=STRATEGY_COLORS["random"],
        marker="_",
        label="Random K=10 median",
    )
    ax.scatter(
        x,
        fixed_values,
        marker="o",
        label="Fixed",
        color=STRATEGY_COLORS["fixed"],
        s=55,
        zorder=4,
    )
    ax.scatter(
        x,
        spf_values,
        marker="s",
        label="SPF",
        color=STRATEGY_COLORS["spf"],
        s=55,
        zorder=4,
    )
    ax.scatter(
        x,
        lpf_values,
        marker="^",
        label="LPF",
        color=STRATEGY_COLORS["lpf"],
        s=60,
        zorder=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("SoC")
    ax.set_title(
        "Priority sensitivity by instance\n"
        "(sampled Random min-max range, not confidence intervals)"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    _save_figure(fig, plots_dir, "fig01_priority_sensitivity_by_instance")
    plt.close(fig)
    written.extend(
        [
            "fig01_priority_sensitivity_by_instance.png",
            "fig01_priority_sensitivity_by_instance.pdf",
        ]
    )

    # Figure 2 — horizontal bars for differing instances only
    differing_rows = select_spf_lpf_differing_instances(instance_rows)
    fig_height = max(4.0, 0.45 * len(differing_rows) + 1.5)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    y = np.arange(len(differing_rows))
    diffs = [row.soc_diff for row in differing_rows]
    colors = [
        STRATEGY_COLORS["spf"] if value < 0 else STRATEGY_COLORS["lpf"] if value > 0 else "#999999"
        for value in diffs
    ]
    ax.barh(y, diffs, color=colors, height=0.65)
    ax.axvline(0.0, color="black", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([row.short_label for row in differing_rows])
    ax.set_xlabel("SPF SoC - LPF SoC")
    ax.set_title(
        "SPF vs LPF SoC difference on differing instances\n"
        "Negative = SPF better; Positive = LPF better"
    )
    fig.tight_layout()
    _save_figure(fig, plots_dir, "fig02_spf_vs_lpf_soc_difference")
    plt.close(fig)
    written.extend(
        [
            "fig02_spf_vs_lpf_soc_difference.png",
            "fig02_spf_vs_lpf_soc_difference.pdf",
        ]
    )

    # Figure 3 — stacked SPF/LPF totals in seconds
    runtime = compute_mean_runtime_seconds(instance_rows)
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.array([0, 1])
    width = 0.55
    ordering = [runtime.spf_ordering_s, runtime.lpf_ordering_s]
    pp = [runtime.spf_pp_s, runtime.lpf_pp_s]
    ax.bar(
        x,
        ordering,
        width,
        label="Ordering time",
        color=["#8CBF8C", "#D98C8C"],
    )
    ax.bar(
        x,
        pp,
        width,
        bottom=ordering,
        label="PP time",
        color=[STRATEGY_COLORS["spf"], STRATEGY_COLORS["lpf"]],
    )
    totals = [runtime.spf_total_s, runtime.lpf_total_s]
    for index, total in enumerate(totals):
        ax.text(
            index,
            total + max(totals) * 0.02,
            f"{total:.1f} s",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["SPF", "LPF"])
    ax.set_ylabel("Mean runtime (s)")
    ax.set_title(
        "Mean SPF/LPF runtime cost\n"
        "(total = ordering + PP; ordering includes independent Space-Time A* searches)"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save_figure(fig, plots_dir, "fig03_spf_lpf_runtime_cost")
    plt.close(fig)
    written.extend(
        [
            "fig03_spf_lpf_runtime_cost.png",
            "fig03_spf_lpf_runtime_cost.pdf",
        ]
    )

    # Figure 4 — top 4 priority-sensitive instances by Random SoC range
    case_rows = select_top_priority_sensitive_instances(sensitivity_rows, limit=4)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(case_rows))
    width = 0.18
    for offset, key, label, color in (
        (-1.5, "fixed_soc", "Fixed", STRATEGY_COLORS["fixed"]),
        (-0.5, "random_soc_median", "Random median", STRATEGY_COLORS["random"]),
        (0.5, "spf_soc", "SPF", STRATEGY_COLORS["spf"]),
        (1.5, "lpf_soc", "LPF", STRATEGY_COLORS["lpf"]),
    ):
        values = [_numeric_int(lookup[row["instance_id"]][key]) or 0 for row in case_rows]
        ax.bar(x + offset * width, values, width=width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{short_instance_label(str(row['instance_id']))}\n"
            f"(range {_numeric_int(row.get('random_soc_range'))})"
            for row in case_rows
        ],
        rotation=0,
        ha="center",
    )
    ax.set_ylabel("SoC")
    ax.set_title(
        "Highest priority-sensitivity instances: strategy quality comparison\n"
        "(top 4 by Random K=10 SoC range among observed priority-sensitive cases)"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    _save_figure(fig, plots_dir, "fig04_priority_sensitive_case_study")
    plt.close(fig)
    written.extend(
        [
            "fig04_priority_sensitive_case_study.png",
            "fig04_priority_sensitive_case_study.pdf",
        ]
    )

    return tuple(written)
