from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_PATH = Path(
    "../../Results/static_algorithms/astar_hpa_jps_cluster_16_scenarios_100.csv"
)
PLOTS_DIR = Path("../../Results/plots/static_algorithms/astar_hpa_jps/summary")

COLORS = {
    "A*": "#4C72B0",
    "HPA*": "#55A868",
    "JPS": "#C44E52",
}


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(RESULTS_PATH)
    summary = build_summary(data)

    print("\nSummary:")
    print(summary.to_string(index=False))

    plot_bar_metric(
        summary=summary,
        y_column="avg_execution_time_ms",
        title="Średni czas wykonania algorytmów",
        ylabel="Czas wykonania [ms]",
        output_path=PLOTS_DIR / "algorithms_vs_execution_time.png",
    )

    plot_bar_metric(
        summary=summary,
        y_column="avg_path_cost",
        title="Średni koszt ścieżki",
        ylabel="Koszt ścieżki",
        output_path=PLOTS_DIR / "algorithms_vs_path_cost.png",
    )

    plot_bar_metric(
        summary=summary,
        y_column="avg_cost_error",
        title="Średni błąd kosztu względem MovingAI",
        ylabel="Błąd kosztu",
        output_path=PLOTS_DIR / "algorithms_vs_cost_error.png",
    )

    plot_bar_metric(
        summary=summary,
        y_column="avg_visited_nodes",
        title="Średnia liczba odwiedzonych węzłów",
        ylabel="Liczba odwiedzonych węzłów",
        output_path=PLOTS_DIR / "algorithms_vs_visited_nodes.png",
    )

    plot_jps_scanned_vs_visited(
        data=data,
        output_path=PLOTS_DIR / "jps_visited_vs_scanned_nodes.png",
    )

    plot_astar_vs_hpa_cumulative_time(
        summary=summary,
        max_queries=1000,
        output_path=PLOTS_DIR / "astar_vs_hpa_cumulative_time.png",
    )


def build_summary(
        data: pd.DataFrame,
) -> pd.DataFrame:
    found_data = data[data["found"] == True].copy()

    found_data["cost_error"] = (
            found_data["path_cost"] - found_data["optimal_length"]
    )

    grouped = (
        found_data
        .groupby("algorithm")
        .agg(
            scenario_count=("scenario_index", "count"),
            avg_execution_time_ms=("execution_time_ms", "mean"),
            avg_path_length=("path_length", "mean"),
            avg_path_cost=("path_cost", "mean"),
            avg_cost_error=("cost_error", "mean"),
            avg_visited_nodes=("visited_nodes", "mean"),
            avg_scanned_nodes=("scanned_nodes", "mean"),
            preprocessing_time_ms=("preprocessing_time_ms", "first"),
            avg_query_time_ms=("query_time_ms", "mean"),
        )
        .reset_index()
    )

    order = {
        "A*": 0,
        "HPA*": 1,
        "JPS": 2,
    }

    grouped["order"] = grouped["algorithm"].map(order)
    return grouped.sort_values("order").drop(columns=["order"])


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_bar_metric(
        summary: pd.DataFrame,
        y_column: str,
        title: str,
        ylabel: str,
        output_path: Path,
) -> None:
    apply_paper_style()

    algorithms = summary["algorithm"].tolist()
    values = summary[y_column].fillna(0.0).tolist()
    colors = [COLORS.get(algorithm, "#777777") for algorithm in algorithms]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        algorithms,
        values,
        color=colors,
        edgecolor="#222222",
        linewidth=0.8,
    )

    ax.set_title(title, pad=12)
    ax.set_xlabel("Algorytm")
    ax.set_ylabel(ylabel)

    ax.bar_label(
        bars,
        labels=[f"{value:.2f}" for value in values],
        padding=4,
        fontsize=10,
    )

    ax.margins(y=0.15)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_jps_scanned_vs_visited(
        data: pd.DataFrame,
        output_path: Path,
) -> None:
    apply_paper_style()

    jps_data = data[
        (data["algorithm"] == "JPS")
        & (data["found"] == True)
        ].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        jps_data["visited_nodes"],
        jps_data["scanned_nodes"],
        alpha=0.75,
        s=48,
        color=COLORS["JPS"],
        edgecolor="#222222",
        linewidth=0.5,
    )

    ax.set_title("JPS: odwiedzone węzły a skanowane komórki", pad=12)
    ax.set_xlabel("Odwiedzone jump pointy")
    ax.set_ylabel("Skanowane komórki")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_astar_vs_hpa_cumulative_time(
        summary: pd.DataFrame,
        max_queries: int,
        output_path: Path,
) -> None:
    apply_paper_style()

    astar = summary[summary["algorithm"] == "A*"].iloc[0]
    hpa = summary[summary["algorithm"] == "HPA*"].iloc[0]

    astar_query_time = astar["avg_execution_time_ms"]
    hpa_query_time = hpa["avg_query_time_ms"]
    hpa_preprocessing_time = hpa["preprocessing_time_ms"]

    queries = list(range(1, max_queries + 1))

    astar_total = [
        query_count * astar_query_time
        for query_count in queries
    ]

    hpa_total = [
        hpa_preprocessing_time + query_count * hpa_query_time
        for query_count in queries
    ]

    break_even = calculate_break_even(
        preprocessing_time=hpa_preprocessing_time,
        astar_query_time=astar_query_time,
        hpa_query_time=hpa_query_time,
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        queries,
        astar_total,
        label="A*",
        color=COLORS["A*"],
        linewidth=2.2,
    )

    ax.plot(
        queries,
        hpa_total,
        label="HPA*",
        color=COLORS["HPA*"],
        linewidth=2.2,
    )

    if break_even is not None and break_even <= max_queries:
        ax.axvline(
            x=break_even,
            color="#222222",
            linestyle=":",
            linewidth=1.5,
        )

        ax.text(
            break_even,
            max(astar_total[-1], hpa_total[-1]) * 0.08,
            f"break-even ≈ {break_even:.0f}",
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=10,
        )

    ax.set_title("Całkowity koszt A* i HPA* względem liczby zapytań", pad=12)
    ax.set_xlabel("Liczba zapytań")
    ax.set_ylabel("Całkowity czas [ms]")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def calculate_break_even(
        preprocessing_time: float,
        astar_query_time: float,
        hpa_query_time: float,
) -> float | None:
    gain_per_query = astar_query_time - hpa_query_time

    if gain_per_query <= 0:
        return None

    return preprocessing_time / gain_per_query


if __name__ == "__main__":
    main()
