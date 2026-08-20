from __future__ import annotations

import csv
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pathfinding.src.experiments.mapf_benchmark_analysis import (
    AGENT_COUNT_ORDER,
    COMMON_BUDGET_MS,
    INTERACTION_ORDER,
    count_directional_soc_better,
)
from pathfinding.src.experiments.mapf_benchmark_execution import (
    DEFAULT_BENCHMARK_ALGORITHMS,
    TERMINATION_EXPANSION_LIMIT,
    TERMINATION_SUCCESS,
    TERMINATION_TIME_LIMIT,
)

ALGORITHM_ORDER: tuple[str, ...] = tuple(
    algorithm.value for algorithm in DEFAULT_BENCHMARK_ALGORITHMS
)

ALGORITHM_DISPLAY: dict[str, str] = {
    "fixed_priority_pp": "Fixed-Priority PP",
    "cbs_basic": "Basic CBS",
    "cbs_cardinal_first": "Cardinal-First CBS",
}

ALGORITHM_COLORS: dict[str, str] = {
    "fixed_priority_pp": "#4C72B0",
    "cbs_basic": "#55A868",
    "cbs_cardinal_first": "#C44E52",
}

INTERACTION_DISPLAY: dict[str, str] = {
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
}

TERMINATION_DISPLAY: dict[str, str] = {
    TERMINATION_SUCCESS: "success",
    TERMINATION_TIME_LIMIT: "time_limit",
    TERMINATION_EXPANSION_LIMIT: "expansion_limit",
    "failure": "failure",
    "error": "error",
}

TERMINATION_MARKERS: dict[str, str] = {
    TERMINATION_SUCCESS: "o",
    TERMINATION_TIME_LIMIT: "x",
    TERMINATION_EXPANSION_LIMIT: "s",
    "failure": "v",
    "error": "D",
}

FIGURE_BASENAMES: tuple[str, ...] = (
    "fig01_success_rate_scalability",
    "fig02_runtime_termination_outcomes",
    "fig03_pp_vs_basic_soc_gap",
    "fig04_cardinal_overhead_tradeoff",
    "fig05_high_interaction_outcome_matrix",
)

CBS_BUDGET_S = COMMON_BUDGET_MS / 1000.0


@dataclass(frozen=True, slots=True)
class PlotInputs:
    analysis_dir: Path

    @property
    def success_rate_by_group(self) -> Path:
        return self.analysis_dir / "success_rate_by_group.csv"

    @property
    def runtime_success_summary(self) -> Path:
        return self.analysis_dir / "runtime_success_summary.csv"

    @property
    def paired_quality_comparison(self) -> Path:
        return self.analysis_dir / "paired_quality_comparison.csv"

    @property
    def cardinal_overhead_summary(self) -> Path:
        return self.analysis_dir / "cardinal_overhead_summary.csv"

    @property
    def common_180s_budget_summary(self) -> Path:
        return self.analysis_dir / "common_180s_budget_summary.csv"

    @property
    def high_interaction_outcomes(self) -> Path:
        return self.analysis_dir / "high_interaction_outcomes.csv"

    @property
    def instance_level_analysis(self) -> Path:
        return self.analysis_dir / "instance_level_analysis.csv"

    @property
    def termination_summary(self) -> Path:
        return self.analysis_dir / "termination_summary.csv"

    @property
    def pp_vs_basic_quality_summary(self) -> Path:
        return self.analysis_dir / "pp_vs_basic_quality_summary.csv"

    @property
    def pp_vs_cardinal_quality_summary(self) -> Path:
        return self.analysis_dir / "pp_vs_cardinal_quality_summary.csv"

    @property
    def basic_vs_cardinal_quality_summary(self) -> Path:
        return self.analysis_dir / "basic_vs_cardinal_quality_summary.csv"


@dataclass(frozen=True, slots=True)
class PlotOutputs:
    plots_dir: Path
    thesis_tables_path: Path


@dataclass(frozen=True, slots=True)
class SuccessRateCell:
    algorithm: str
    agent_count: int
    interaction_level: str
    total_runs: int
    success_count: int
    success_rate: float


@dataclass(frozen=True, slots=True)
class RuntimeOutcomePoint:
    instance_id: str
    agent_count: int
    interaction_level: str
    independent_conflict_count: int
    algorithm: str
    termination_reason: str
    execution_time_s: float
    within_180s: bool | None
    is_pp_slow_success: bool


@dataclass(frozen=True, slots=True)
class PairedSocGapPoint:
    instance_id: str
    agent_count: int
    interaction_level: str
    independent_conflict_count: int
    pp_soc: int
    basic_soc: int
    soc_gap: float


@dataclass(frozen=True, slots=True)
class CardinalOverheadPoint:
    instance_id: str
    agent_count: int
    interaction_level: str
    ct_reduction: float
    runtime_ratio: float
    additional_low_level_searches: float


@dataclass(frozen=True, slots=True)
class HighOutcomeCell:
    instance_id: str
    agent_count: int
    independent_conflict_count: int
    algorithm: str
    termination_reason: str
    execution_time_s: float | None


@dataclass(frozen=True, slots=True)
class PlotBundle:
    success_rate_cells: tuple[SuccessRateCell, ...]
    runtime_outcome_points: tuple[RuntimeOutcomePoint, ...]
    paired_soc_gaps: tuple[PairedSocGapPoint, ...]
    cardinal_overhead_points: tuple[CardinalOverheadPoint, ...]
    high_outcome_cells: tuple[HighOutcomeCell, ...]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _parse_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    return value.strip().lower() == "true"


def load_success_rate_cells(rows: Sequence[dict[str, str]]) -> tuple[SuccessRateCell, ...]:
    cells: list[SuccessRateCell] = []
    for row in rows:
        cells.append(
            SuccessRateCell(
                algorithm=row["algorithm"],
                agent_count=int(row["agent_count"]),
                interaction_level=row["interaction_level"].lower(),
                total_runs=int(row["total_runs"]),
                success_count=int(row["success_count"]),
                success_rate=float(row["success_rate"]),
            )
        )
    cells.sort(
        key=lambda cell: (
            INTERACTION_ORDER.index(cell.interaction_level),
            AGENT_COUNT_ORDER.index(cell.agent_count),
            ALGORITHM_ORDER.index(cell.algorithm),
        )
    )
    return tuple(cells)


def load_runtime_outcome_points(
    rows: Sequence[dict[str, str]],
) -> tuple[RuntimeOutcomePoint, ...]:
    points: list[RuntimeOutcomePoint] = []
    for row in rows:
        interaction = row["interaction_level"].lower()
        instance_id = row["instance_id"]
        agent_count = int(row["agent_count"])
        conflict_count = int(row["independent_conflict_count"])

        for algorithm in ALGORITHM_ORDER:
            prefix = algorithm
            termination = row[f"{prefix}_termination_reason"]
            execution_ms = _parse_float(row.get(f"{prefix}_execution_time_ms"))
            if execution_ms is None:
                continue

            within_180s = _parse_bool(row.get(f"{prefix}_within_180s"))
            is_pp_slow_success = (
                algorithm == "fixed_priority_pp"
                and termination == TERMINATION_SUCCESS
                and within_180s is False
            )
            points.append(
                RuntimeOutcomePoint(
                    instance_id=instance_id,
                    agent_count=agent_count,
                    interaction_level=interaction,
                    independent_conflict_count=conflict_count,
                    algorithm=algorithm,
                    termination_reason=termination,
                    execution_time_s=execution_ms / 1000.0,
                    within_180s=within_180s,
                    is_pp_slow_success=is_pp_slow_success,
                )
            )

    points.sort(
        key=lambda point: (
            INTERACTION_ORDER.index(point.interaction_level),
            AGENT_COUNT_ORDER.index(point.agent_count),
            point.instance_id,
            ALGORITHM_ORDER.index(point.algorithm),
        )
    )
    return tuple(points)


def load_paired_soc_gaps(rows: Sequence[dict[str, str]]) -> tuple[PairedSocGapPoint, ...]:
    gaps: list[PairedSocGapPoint] = []
    for row in rows:
        if row["comparison"] != "pp_minus_basic":
            continue
        gap_value = _parse_float(row.get("pp_minus_basic_soc"))
        if gap_value is None:
            continue
        gaps.append(
            PairedSocGapPoint(
                instance_id=row["instance_id"],
                agent_count=int(row["agent_count"]),
                interaction_level=row["interaction_level"].lower(),
                independent_conflict_count=int(row["independent_conflict_count"]),
                pp_soc=int(row["left_soc"]),
                basic_soc=int(row["right_soc"]),
                soc_gap=gap_value,
            )
        )

    gaps.sort(
        key=lambda point: (
            point.soc_gap,
            AGENT_COUNT_ORDER.index(point.agent_count),
            INTERACTION_ORDER.index(point.interaction_level),
            point.instance_id,
        )
    )
    return tuple(gaps)


def load_cardinal_overhead_points(
    rows: Sequence[dict[str, str]],
) -> tuple[CardinalOverheadPoint, ...]:
    points: list[CardinalOverheadPoint] = []
    for row in rows:
        instance_id = row.get("instance_id", "")
        if not instance_id:
            continue

        ct_reduction = _parse_float(row.get("ct_expansion_reduction_basic_minus_cardinal"))
        runtime_ratio = _parse_float(row.get("runtime_ratio_cardinal_over_basic"))
        additional_ll = _parse_float(
            row.get("additional_low_level_searches_cardinal_minus_basic")
        )
        if ct_reduction is None or runtime_ratio is None or additional_ll is None:
            continue

        points.append(
            CardinalOverheadPoint(
                instance_id=instance_id,
                agent_count=int(row["agent_count"]),
                interaction_level=row["interaction_level"].lower(),
                ct_reduction=ct_reduction,
                runtime_ratio=runtime_ratio,
                additional_low_level_searches=additional_ll,
            )
        )

    points.sort(
        key=lambda point: (
            AGENT_COUNT_ORDER.index(point.agent_count),
            INTERACTION_ORDER.index(point.interaction_level),
            point.instance_id,
        )
    )
    return tuple(points)


def load_high_outcome_cells(rows: Sequence[dict[str, str]]) -> tuple[HighOutcomeCell, ...]:
    cells: list[HighOutcomeCell] = []
    for row in rows:
        execution_ms = _parse_float(row.get("execution_time_ms"))
        cells.append(
            HighOutcomeCell(
                instance_id=row["instance_id"],
                agent_count=int(row["agent_count"]),
                independent_conflict_count=int(row["independent_conflict_count"]),
                algorithm=row["algorithm"],
                termination_reason=row["termination_reason"],
                execution_time_s=None if execution_ms is None else execution_ms / 1000.0,
            )
        )

    cells.sort(
        key=lambda cell: (
            AGENT_COUNT_ORDER.index(cell.agent_count),
            cell.instance_id,
            ALGORITHM_ORDER.index(cell.algorithm),
        )
    )
    return tuple(cells)


def runtime_success_group_has_observations(
    rows: Sequence[dict[str, str]],
    *,
    algorithm: str,
    agent_count: int,
    interaction_level: str,
) -> bool | None:
    """Return True if successful observations exist, False if group absent, None if n=0 row exists."""
    for row in rows:
        if (
            row["algorithm"] == algorithm
            and int(row["agent_count"]) == agent_count
            and row["interaction_level"].lower() == interaction_level.lower()
        ):
            n_value = _parse_int(row.get("n"))
            if n_value == 0:
                return False
            return True
    return None


def load_plot_bundle(inputs: PlotInputs) -> PlotBundle:
    return PlotBundle(
        success_rate_cells=load_success_rate_cells(
            _read_csv_rows(inputs.success_rate_by_group)
        ),
        runtime_outcome_points=load_runtime_outcome_points(
            _read_csv_rows(inputs.instance_level_analysis)
        ),
        paired_soc_gaps=load_paired_soc_gaps(
            _read_csv_rows(inputs.paired_quality_comparison)
        ),
        cardinal_overhead_points=load_cardinal_overhead_points(
            _read_csv_rows(inputs.cardinal_overhead_summary)
        ),
        high_outcome_cells=load_high_outcome_cells(
            _read_csv_rows(inputs.high_interaction_outcomes)
        ),
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
            "xtick.labelsize": 9,
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


def render_success_rate_scalability(
    cells: Sequence[SuccessRateCell],
    plots_dir: Path,
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    _apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    bar_width = 0.24
    x_positions = np.arange(len(AGENT_COUNT_ORDER))

    for axis, interaction in zip(axes, INTERACTION_ORDER, strict=True):
        for algo_index, algorithm in enumerate(ALGORITHM_ORDER):
            offsets = x_positions + (algo_index - 1) * bar_width
            heights: list[float] = []
            labels: list[str] = []
            for agent_count in AGENT_COUNT_ORDER:
                match = next(
                    (
                        cell
                        for cell in cells
                        if cell.algorithm == algorithm
                        and cell.agent_count == agent_count
                        and cell.interaction_level == interaction
                    ),
                    None,
                )
                if match is None:
                    heights.append(0.0)
                    labels.append("0/0")
                else:
                    heights.append(match.success_rate)
                    labels.append(f"{match.success_count}/{match.total_runs}")

            bars = axis.bar(
                offsets,
                heights,
                width=bar_width,
                color=ALGORITHM_COLORS[algorithm],
                label=ALGORITHM_DISPLAY[algorithm],
                edgecolor="black",
                linewidth=0.4,
            )
            for bar, label in zip(bars, labels, strict=True):
                label_y = bar.get_height() + 0.04
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_y,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    clip_on=False,
                )

        axis.set_title(INTERACTION_DISPLAY[interaction])
        axis.set_xticks(x_positions)
        axis.set_xticklabels([str(count) for count in AGENT_COUNT_ORDER])
        axis.set_xlabel("Agent count")
        axis.set_ylim(0, 1.18)

    axes[0].set_ylabel("Success rate")
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=ALGORITHM_COLORS[algorithm])
        for algorithm in ALGORITHM_ORDER
    ]
    fig.legend(
        legend_handles,
        [ALGORITHM_DISPLAY[algorithm] for algorithm in ALGORITHM_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        frameon=True,
    )
    fig.suptitle("MAPF benchmark success rate by interaction stratum", y=1.16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save_figure(fig, plots_dir, FIGURE_BASENAMES[0])


def render_runtime_termination_outcomes(
    points: Sequence[RuntimeOutcomePoint],
    plots_dir: Path,
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    _apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)

    for axis, interaction in zip(axes, INTERACTION_ORDER, strict=True):
        panel_points = [
            point for point in points if point.interaction_level == interaction
        ]
        instance_ids = sorted({point.instance_id for point in panel_points})
        x_index = {instance_id: index for index, instance_id in enumerate(instance_ids)}

        for algorithm in ALGORITHM_ORDER:
            algo_points = [point for point in panel_points if point.algorithm == algorithm]
            for point in algo_points:
                x_value = x_index[point.instance_id] + (
                    ALGORITHM_ORDER.index(algorithm) - 1
                ) * 0.22
                marker = TERMINATION_MARKERS.get(point.termination_reason, "o")
                if marker == "x":
                    axis.scatter(
                        x_value,
                        point.execution_time_s,
                        marker=marker,
                        s=48,
                        color=ALGORITHM_COLORS[algorithm],
                        alpha=0.9,
                    )
                else:
                    axis.scatter(
                        x_value,
                        point.execution_time_s,
                        marker=marker,
                        s=48,
                        color=ALGORITHM_COLORS[algorithm],
                        edgecolors=(
                            "black"
                            if point.is_pp_slow_success
                            else ALGORITHM_COLORS[algorithm]
                        ),
                        linewidths=1.4 if point.is_pp_slow_success else 0.4,
                        alpha=0.9,
                    )

        axis.axhline(
            CBS_BUDGET_S,
            color="#666666",
            linestyle="--",
            linewidth=1.0,
            label="180 s CBS budget" if interaction == INTERACTION_ORDER[0] else None,
        )
        axis.set_yscale("log")
        axis.set_title(INTERACTION_DISPLAY[interaction])
        axis.set_xticks(list(x_index.values()))
        axis.set_xticklabels(
            [
                f"{instance_id.split('_')[-1]}\n(n={next(p.agent_count for p in panel_points if p.instance_id == instance_id)})"
                for instance_id in instance_ids
            ],
            rotation=0,
            fontsize=8,
        )
        axis.set_xlabel("Benchmark instance")

    axes[0].set_ylabel("Execution time [s]")

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#888888",
            markeredgecolor="black",
            markersize=7,
            label="success",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="#888888",
            linestyle="None",
            markersize=7,
            label="time_limit",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#888888",
            markeredgecolor="black",
            markersize=7,
            label="expansion_limit",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#888888",
            markeredgecolor="black",
            markeredgewidth=1.4,
            markersize=7,
            label="PP success >180 s",
        ),
        Line2D([0], [0], color="#666666", linestyle="--", label="180 s CBS budget"),
    ]
    for algorithm in ALGORITHM_ORDER:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=ALGORITHM_COLORS[algorithm],
                markeredgecolor=ALGORITHM_COLORS[algorithm],
                markersize=7,
                label=ALGORITHM_DISPLAY[algorithm],
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=True,
        fontsize=7,
    )
    fig.suptitle("Instance-level runtime and termination outcomes", y=1.02)
    fig.tight_layout(rect=(0, 0.08, 1, 0.98))
    return _save_figure(fig, plots_dir, FIGURE_BASENAMES[1])


def render_pp_vs_basic_soc_gap(
    gaps: Sequence[PairedSocGapPoint],
    plots_dir: Path,
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    _apply_plot_style()
    fig, axis = plt.subplots(figsize=(8, 5))

    y_positions = list(range(len(gaps)))
    axis.axvline(0.0, color="#333333", linewidth=1.0, linestyle="-")
    axis.hlines(
        y_positions,
        xmin=0.0,
        xmax=[gap.soc_gap for gap in gaps],
        color="#888888",
        linewidth=1.0,
        alpha=0.7,
    )
    axis.scatter(
        [gap.soc_gap for gap in gaps],
        y_positions,
        color="#4C72B0",
        s=36,
        zorder=3,
    )

    labels = [
        f"{gap.instance_id.replace('AR0204SR_', '')} "
        f"({gap.agent_count}, {INTERACTION_DISPLAY[gap.interaction_level]})"
        for gap in gaps
    ]
    axis.set_yticks(y_positions)
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_xlabel("PP SoC − Basic CBS SoC (positive = Basic lower SoC)")
    axis.set_ylabel("Common-success instance")
    axis.set_title("PP vs Basic CBS solution-quality gap (paired common-success cases)")
    fig.tight_layout()
    return _save_figure(fig, plots_dir, FIGURE_BASENAMES[2])


def render_cardinal_overhead_tradeoff(
    points: Sequence[CardinalOverheadPoint],
    plots_dir: Path,
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    _apply_plot_style()
    fig, axis = plt.subplots(figsize=(7.5, 6.0))

    sizes = [
        30 + max(point.additional_low_level_searches, 0) * 4
        for point in points
    ]
    axis.scatter(
        [point.ct_reduction for point in points],
        [point.runtime_ratio for point in points],
        s=sizes,
        color="#C44E52",
        alpha=0.75,
        edgecolors="black",
        linewidths=0.4,
    )
    axis.axhline(1.0, color="#666666", linestyle="--", linewidth=1.0)
    axis.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Basic CT expansions − Cardinal CT expansions")
    axis.set_ylabel("Cardinal runtime / Basic runtime")
    axis.set_title(
        "Cardinal-First overhead trade-off (common-success CBS pairs)",
        pad=12,
    )

    if points:
        max_ct = max(points, key=lambda point: point.ct_reduction)
        max_ratio = max(points, key=lambda point: point.runtime_ratio)
        axis.annotate(
            f"{max_ct.instance_id.replace('AR0204SR_', '')}\nmax CT reduction",
            (max_ct.ct_reduction, max_ct.runtime_ratio),
            textcoords="offset points",
            xytext=(8, -12),
            fontsize=8,
            ha="left",
        )
        axis.annotate(
            f"{max_ratio.instance_id.replace('AR0204SR_', '')}\nmax runtime ratio",
            (max_ratio.ct_reduction, max_ratio.runtime_ratio),
            textcoords="offset points",
            xytext=(-48, -18),
            fontsize=8,
            ha="left",
        )

    size_legend_handles = [
        plt.scatter([], [], s=30, color="#C44E52", edgecolors="black", linewidths=0.4),
        plt.scatter([], [], s=54, color="#C44E52", edgecolors="black", linewidths=0.4),
    ]
    axis.legend(
        size_legend_handles,
        ["baseline size", "+6 additional combined LL searches"],
        title="Marker size",
        loc="upper right",
        frameon=True,
        fontsize=8,
        title_fontsize=8,
    )

    fig.tight_layout()
    return _save_figure(fig, plots_dir, FIGURE_BASENAMES[3])


def _outcome_label(cell: HighOutcomeCell) -> str:
    if cell.termination_reason == TERMINATION_SUCCESS and cell.execution_time_s is not None:
        return f"success\n{cell.execution_time_s:.1f}s"
    return TERMINATION_DISPLAY.get(cell.termination_reason, cell.termination_reason)


def render_high_interaction_outcome_matrix(
    cells: Sequence[HighOutcomeCell],
    plots_dir: Path,
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    _apply_plot_style()
    instance_ids = sorted({cell.instance_id for cell in cells})
    matrix = np.empty((len(instance_ids), len(ALGORITHM_ORDER)), dtype=object)

    conflict_by_instance = {
        cell.instance_id: cell.independent_conflict_count for cell in cells
    }
    agent_by_instance = {cell.instance_id: cell.agent_count for cell in cells}

    for row_index, instance_id in enumerate(instance_ids):
        for col_index, algorithm in enumerate(ALGORITHM_ORDER):
            match = next(
                (
                    cell
                    for cell in cells
                    if cell.instance_id == instance_id and cell.algorithm == algorithm
                ),
                None,
            )
            matrix[row_index, col_index] = "" if match is None else _outcome_label(match)

    fig, axis = plt.subplots(figsize=(8.5, 6.0))
    axis.axis("off")
    row_labels = [
        f"{instance_id.replace('AR0204SR_', '')} "
        f"(n={agent_by_instance[instance_id]}, c={conflict_by_instance[instance_id]})"
        for instance_id in instance_ids
    ]
    column_labels = [ALGORITHM_DISPLAY[algorithm] for algorithm in ALGORITHM_ORDER]
    table = axis.table(
        cellText=matrix,
        rowLabels=row_labels,
        colLabels=column_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)
    axis.set_title("HIGH interaction stratum outcome matrix", pad=16)
    fig.tight_layout(pad=1.2)
    return _save_figure(fig, plots_dir, FIGURE_BASENAMES[4])


def _format_rate(count: int, total: int) -> str:
    if total == 0:
        return "—"
    return f"{count}/{total} ({count / total:.1%})"


def _load_quality_summary_row(path: Path) -> dict[str, str]:
    rows = _read_csv_rows(path)
    if not rows:
        raise ValueError(f"Missing quality summary data: {path}")
    return rows[0]


def build_thesis_tables_markdown(inputs: PlotInputs) -> str:
    termination_rows = _read_csv_rows(inputs.termination_summary)
    budget_rows = _read_csv_rows(inputs.common_180s_budget_summary)
    overhead_rows = _read_csv_rows(inputs.cardinal_overhead_summary)
    instance_overhead_rows = load_cardinal_overhead_points(overhead_rows)

    termination_counts: dict[str, dict[str, int]] = {
        algorithm: {} for algorithm in ALGORITHM_ORDER
    }
    for row in termination_rows:
        termination_counts[row["algorithm"]][row["termination_reason"]] = int(row["count"])

    budget_by_algorithm = {row["algorithm"]: row for row in budget_rows}

    pp_basic = _load_quality_summary_row(inputs.pp_vs_basic_quality_summary)
    pp_cardinal = _load_quality_summary_row(inputs.pp_vs_cardinal_quality_summary)
    basic_cardinal = _load_quality_summary_row(inputs.basic_vs_cardinal_quality_summary)

    aggregate_row = next(
        (row for row in overhead_rows if row.get("summary_level") == "common_success_instances"),
        None,
    )

    runtime_ratios = [point.runtime_ratio for point in instance_overhead_rows]
    ct_reductions = [point.ct_reduction for point in instance_overhead_rows]
    additional_ll = [point.additional_low_level_searches for point in instance_overhead_rows]
    cardinal_faster = sum(1 for ratio in runtime_ratios if ratio < 1.0)
    basic_faster = sum(1 for ratio in runtime_ratios if ratio > 1.0)
    ct_reduced_cases = sum(1 for value in ct_reductions if value > 0)

    lines = [
        "# MAPF Benchmark Thesis Tables (MAPF-5C.2)",
        "",
        "## Table A — Overall algorithm performance",
        "",
        "| Algorithm | Observed success | Common-180s success | Timeouts | Expansion limits |",
        "| --- | --- | --- | --- | --- |",
    ]

    for algorithm in ALGORITHM_ORDER:
        budget = budget_by_algorithm[algorithm]
        term = termination_counts[algorithm]
        observed = f"{budget['observed_success_count']}/{budget['total_runs']} ({float(budget['observed_success_rate']):.1%})"
        if algorithm == "fixed_priority_pp":
            within = (
                f"{budget['within_common_budget_success_count']}/"
                f"{budget['total_runs']} ({float(budget['within_common_budget_success_rate']):.1%}) "
                "**POST-HOC**"
            )
        else:
            within = (
                f"{budget['within_common_budget_success_count']}/"
                f"{budget['total_runs']} ({float(budget['within_common_budget_success_rate']):.1%})"
            )
        lines.append(
            "| "
            + " | ".join(
                [
                    ALGORITHM_DISPLAY[algorithm],
                    observed,
                    within,
                    str(term.get(TERMINATION_TIME_LIMIT, 0)),
                    str(term.get(TERMINATION_EXPANSION_LIMIT, 0)),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "Notes: PP common-180s success is a secondary post-hoc comparison; original PP termination labels are unchanged.",
            "",
            "## Table B — Paired solution quality",
            "",
            "SoC difference = left-algorithm SoC − right-algorithm SoC; positive values mean the right-hand algorithm achieved lower SoC.",
            "",
            "| Comparison | Common-success n | Mean SoC diff | Median SoC diff | Max |SoC diff| | Left better | Equal | Right better |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    comparison_specs = (
        ("PP vs Basic CBS", pp_basic, "pp_minus_basic_soc"),
        ("PP vs Cardinal-First CBS", pp_cardinal, "pp_minus_cardinal_soc"),
        ("Basic CBS vs Cardinal-First CBS", basic_cardinal, "basic_minus_cardinal_soc"),
    )
    paired_rows = _read_csv_rows(inputs.paired_quality_comparison)
    for label, summary, diff_column in comparison_specs:
        paired_values = [
            float(row[diff_column])
            for row in paired_rows
            if row["comparison"] == diff_column.replace("_soc", "")
            and row.get(diff_column, "") != ""
        ]
        max_abs_gap = max((abs(value) for value in paired_values), default=0.0)
        directional_counts = count_directional_soc_better(paired_values)
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(len(paired_values)),
                    f"{float(summary['mean_soc_diff']):.2f}",
                    f"{float(summary['median_soc_diff']):.2f}",
                    f"{max_abs_gap:.0f}",
                    str(directional_counts.left_better),
                    str(directional_counts.equal),
                    str(directional_counts.right_better),
                ]
            )
            + " |"
        )

    mean_runtime_ratio = (
        float(aggregate_row["mean_runtime_ratio"])
        if aggregate_row and aggregate_row.get("mean_runtime_ratio")
        else statistics.mean(runtime_ratios)
    )
    median_runtime_ratio = (
        float(aggregate_row["median_runtime_ratio"])
        if aggregate_row and aggregate_row.get("median_runtime_ratio")
        else statistics.median(runtime_ratios)
    )
    mean_ct_reduction = (
        float(aggregate_row["mean_ct_reduction"])
        if aggregate_row and aggregate_row.get("mean_ct_reduction")
        else statistics.mean(ct_reductions)
    )
    mean_additional_ll = (
        float(aggregate_row["mean_additional_low_level_searches"])
        if aggregate_row and aggregate_row.get("mean_additional_low_level_searches")
        else statistics.mean(additional_ll)
    )

    lines.extend(
        [
            "",
            "## Table C — Basic vs Cardinal computational trade-off",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Common-success CBS pairs (n) | {len(instance_overhead_rows)} |",
            f"| Mean runtime ratio (Cardinal / Basic) | {mean_runtime_ratio:.3f} |",
            f"| Median runtime ratio (Cardinal / Basic) | {median_runtime_ratio:.3f} |",
            f"| Cardinal faster (count) | {cardinal_faster} |",
            f"| Basic faster (count) | {basic_faster} |",
            f"| Cases with reduced CT expansions | {ct_reduced_cases} |",
            f"| Mean CT expansion reduction (Basic − Cardinal) | {mean_ct_reduction:.2f} |",
            f"| Mean additional combined low-level searches (Cardinal − Basic) | {mean_additional_ll:.2f} |",
            "",
            "Interpretation: CT expansion reduction alone does not imply runtime improvement; low-level classification/replan overhead must be considered jointly.",
        ]
    )
    return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class PlotGenerationResult:
    plots_dir: Path
    thesis_tables_path: Path
    figure_paths: tuple[Path, ...]


def generate_benchmark_plots(
    inputs: PlotInputs,
    outputs: PlotOutputs,
) -> PlotGenerationResult:
    bundle = load_plot_bundle(inputs)
    figure_paths: list[Path] = []

    for renderer, data in (
        (render_success_rate_scalability, bundle.success_rate_cells),
        (render_runtime_termination_outcomes, bundle.runtime_outcome_points),
        (render_pp_vs_basic_soc_gap, bundle.paired_soc_gaps),
        (render_cardinal_overhead_tradeoff, bundle.cardinal_overhead_points),
        (render_high_interaction_outcome_matrix, bundle.high_outcome_cells),
    ):
        png_path, pdf_path = renderer(data, outputs.plots_dir)
        figure_paths.extend((png_path, pdf_path))

    thesis_markdown = build_thesis_tables_markdown(inputs)
    outputs.thesis_tables_path.parent.mkdir(parents=True, exist_ok=True)
    outputs.thesis_tables_path.write_text(thesis_markdown, encoding="utf-8")

    return PlotGenerationResult(
        plots_dir=outputs.plots_dir,
        thesis_tables_path=outputs.thesis_tables_path,
        figure_paths=tuple(figure_paths),
    )
