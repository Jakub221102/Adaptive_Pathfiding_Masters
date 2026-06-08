from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_PATH = Path(
    "../../Results/dynamic_algorithms/astar_vs_dstar_dynamic_seed_42.csv"
)
PLOTS_DIR = Path("../../Results/plots/dynamic_algorithms")

COLORS = {
    "A*": "#4C72B0",
    "D* Lite": "#55A868",
}

METRICS = [
    ("avg_total_execution_time_ms", "Średni czas wykonania", "Czas [ms]"),
    ("median_total_execution_time_ms", "Mediana czasu wykonania", "Czas [ms]"),
    ("avg_total_path_cost", "Średni koszt trasy", "Koszt trasy"),
    ("avg_replanning_count", "Średnia liczba replanningów", "Replanningi"),
    ("avg_waiting_steps", "Średnia liczba kroków oczekiwania", "Waiting steps"),
    ("avg_collision_count", "Średnia liczba kolizji", "Kolizje"),
]


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(RESULTS_PATH)
    summary = build_summary(data)

    print("\nSummary:")
    print(summary.to_string(index=False))

    for metric_column, title, ylabel in METRICS:
        plot_bar_metric(
            summary=summary,
            y_column=metric_column,
            title=title,
            ylabel=ylabel,
            output_path=PLOTS_DIR / f"astar_vs_dstar_{metric_column}.png",
        )

    plot_median_execution_time(
        summary=summary,
        output_path=PLOTS_DIR / "algorithms_vs_median_execution_time.png",
    )
    plot_execution_time_boxplot(
        data=data,
        output_path=PLOTS_DIR / "algorithms_vs_execution_time_boxplot.png",
    )


def build_summary(data: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        data.groupby("algorithm", as_index=False)
        .agg(
            scenarios=("scenario_index", "count"),
            goal_reached=("final_goal_reached", "sum"),
            avg_total_execution_time_ms=("total_execution_time_ms", "mean"),
            median_total_execution_time_ms=("total_execution_time_ms", "median"),
            p95_total_execution_time_ms=(
                "total_execution_time_ms",
                lambda series: series.quantile(0.95),
            ),
            max_total_execution_time_ms=("total_execution_time_ms", "max"),
            avg_wall_clock_s=("wall_clock_s", "mean"),
            median_wall_clock_s=("wall_clock_s", "median"),
            avg_total_path_cost=("total_path_cost", "mean"),
            avg_replanning_count=("replanning_count", "mean"),
            avg_waiting_steps=("waiting_steps", "mean"),
            avg_collision_count=("collision_count", "mean"),
            avg_travelled_steps=("travelled_steps", "mean"),
        )
    )
    grouped["goal_reached_rate"] = grouped["goal_reached"] / grouped["scenarios"]
    return grouped


def plot_bar_metric(
        summary: pd.DataFrame,
        y_column: str,
        title: str,
        ylabel: str,
        output_path: Path,
) -> None:
    algorithms = summary["algorithm"].tolist()
    values = summary[y_column].tolist()
    colors = [COLORS.get(algorithm, "#888888") for algorithm in algorithms]

    plt.figure(figsize=(8, 5))
    plt.bar(algorithms, values, color=colors)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_median_execution_time(summary: pd.DataFrame, output_path: Path) -> None:
    plot_bar_metric(
        summary=summary,
        y_column="median_total_execution_time_ms",
        title="Mediana czasu wykonania algorytmów",
        ylabel="Czas [ms]",
        output_path=output_path,
    )


def plot_execution_time_boxplot(data: pd.DataFrame, output_path: Path) -> None:
    algorithms = sorted(data["algorithm"].unique())
    values_by_algorithm = [
        data.loc[data["algorithm"] == algorithm, "total_execution_time_ms"].tolist()
        for algorithm in algorithms
    ]
    colors = [COLORS.get(algorithm, "#888888") for algorithm in algorithms]

    plt.figure(figsize=(8, 5))
    boxplot = plt.boxplot(
        values_by_algorithm,
        tick_labels=algorithms,
        patch_artist=True,
    )
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.title("Rozkład czasu wykonania algorytmów")
    plt.ylabel("Czas [ms]")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
