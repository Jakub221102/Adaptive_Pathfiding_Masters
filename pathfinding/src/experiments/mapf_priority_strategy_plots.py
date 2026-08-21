from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any


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
        row for row in sensitivity_rows if row["observed_priority_sensitive"]
    ]
    sensitive_rows.sort(
        key=lambda row: (row["random_soc_range"], row["instance_id"]),
        reverse=True,
    )
    focus_rows = sensitive_rows[:12] if sensitive_rows else list(instance_rows[:12])
    labels = [row["instance_id"].replace("AR0204SR_", "") for row in focus_rows]
    lookup = {row["instance_id"]: row for row in instance_rows}

    # Figure 1
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(focus_rows))
    for index, row in enumerate(focus_rows):
        instance = lookup[row["instance_id"]]
        y_min = instance["random_soc_min"]
        y_max = instance["random_soc_max"]
        y_med = instance["random_soc_median"]
        if y_min is None or y_max is None or y_med is None:
            continue
        ax.vlines(index, y_min, y_max, color="#888888", linewidth=3, alpha=0.8)
        ax.scatter(index, y_med, color="#444444", marker="_", s=200, zorder=3)
    ax.scatter([], [], color="#888888", label="Random K=10 sampled min-max")
    ax.scatter([], [], color="#444444", marker="_", label="Random K=10 median")
    ax.plot(x, [lookup[row["instance_id"]]["fixed_soc"] for row in focus_rows], "o-", label="Fixed", color="#4C72B0")
    ax.plot(x, [lookup[row["instance_id"]]["spf_soc"] for row in focus_rows], "s-", label="SPF", color="#55A868")
    ax.plot(x, [lookup[row["instance_id"]]["lpf_soc"] for row in focus_rows], "^-", label="LPF", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("SoC")
    ax.set_title("Priority sensitivity by instance (sampled Random range, not confidence intervals)")
    ax.legend(loc="best")
    fig.tight_layout()
    _save_figure(fig, plots_dir, "fig01_priority_sensitivity_by_instance")
    plt.close(fig)
    written.extend(["fig01_priority_sensitivity_by_instance.png", "fig01_priority_sensitivity_by_instance.pdf"])

    # Figure 2
    fig, ax = plt.subplots(figsize=(10, 5))
    all_rows = list(instance_rows)
    x = np.arange(len(all_rows))
    diffs = [row["spf_soc"] - row["lpf_soc"] for row in all_rows]
    colors = ["#55A868" if value < 0 else "#C44E52" if value > 0 else "#999999" for value in diffs]
    ax.bar(x, diffs, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("SPF SoC - LPF SoC")
    ax.set_xlabel("Instance index (deterministic catalogue order)")
    ax.set_title("Paired SPF vs LPF SoC difference (negative => SPF lower SoC)")
    fig.tight_layout()
    _save_figure(fig, plots_dir, "fig02_spf_vs_lpf_soc_difference")
    plt.close(fig)
    written.extend(["fig02_spf_vs_lpf_soc_difference.png", "fig02_spf_vs_lpf_soc_difference.pdf"])

    # Figure 3
    fig, ax = plt.subplots(figsize=(8, 5))
    spf_order = [row["spf_ordering_time_ms"] for row in instance_rows]
    spf_pp = [row["spf_pp_time_ms"] for row in instance_rows]
    lpf_order = [row["lpf_ordering_time_ms"] for row in instance_rows]
    lpf_pp = [row["lpf_pp_time_ms"] for row in instance_rows]
    labels_bar = ["SPF ordering", "SPF PP", "LPF ordering", "LPF PP"]
    means = [
        float(np.mean(spf_order)),
        float(np.mean(spf_pp)),
        float(np.mean(lpf_order)),
        float(np.mean(lpf_pp)),
    ]
    ax.bar(labels_bar, means, color=["#8CBF8C", "#55A868", "#D98C8C", "#C44E52"])
    ax.set_ylabel("Mean runtime (ms)")
    ax.set_title("Mean SPF/LPF runtime split (ordering includes independent A*)")
    fig.tight_layout()
    _save_figure(fig, plots_dir, "fig03_spf_lpf_runtime_cost")
    plt.close(fig)
    written.extend(["fig03_spf_lpf_runtime_cost.png", "fig03_spf_lpf_runtime_cost.pdf"])

    # Figure 4
    case_rows = sensitive_rows[:6] if sensitive_rows else list(instance_rows[:6])
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(case_rows))
    width = 0.18
    for offset, key, label, color in (
        (-1.5, "fixed_soc", "Fixed", "#4C72B0"),
        (-0.5, "random_soc_median", "Random median", "#777777"),
        (0.5, "spf_soc", "SPF", "#55A868"),
        (1.5, "lpf_soc", "LPF", "#C44E52"),
    ):
        values = [lookup[row["instance_id"]][key] for row in case_rows]
        ax.bar(x + offset * width, values, width=width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [row["instance_id"].replace("AR0204SR_", "") for row in case_rows],
        rotation=30,
        ha="right",
    )
    ax.set_ylabel("SoC")
    ax.set_title("Selected priority-sensitive instances: quality comparison")
    ax.legend(loc="best")
    fig.tight_layout()
    _save_figure(fig, plots_dir, "fig04_priority_sensitive_case_study")
    plt.close(fig)
    written.extend(["fig04_priority_sensitive_case_study.png", "fig04_priority_sensitive_case_study.pdf"])

    return tuple(written)
