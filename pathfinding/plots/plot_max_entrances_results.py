from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("../../Results/static_algorithms/max_entrances")
PLOTS_DIR = Path("../../Results/plots/static_algorithms/astar_hpa_jps/max_entrances")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = load_max_entrances_summary()

    print("\nSummary:")
    print(summary.to_string(index=False))

    plot_metric(
        summary=summary,
        x_column="max_entrances",
        y_column="found_rate",
        title="Wpływ liczby wejść na skuteczność HPA*",
        xlabel="Maksymalna liczba wejść między parą klastrów",
        ylabel="Odsetek znalezionych ścieżek",
        output_path=PLOTS_DIR / "max_entrances_vs_found_rate.png",
    )

    plot_metric(
        summary=summary,
        x_column="max_entrances",
        y_column="avg_query_time_ms",
        title="Wpływ liczby wejść na średni czas zapytania HPA*",
        xlabel="Maksymalna liczba wejść między parą klastrów",
        ylabel="Średni czas zapytania [ms]",
        output_path=PLOTS_DIR / "max_entrances_vs_query_time.png",
    )

    plot_metric(
        summary=summary,
        x_column="max_entrances",
        y_column="preprocessing_time_ms",
        title="Wpływ liczby wejść na czas preprocessingu HPA*",
        xlabel="Maksymalna liczba wejść między parą klastrów",
        ylabel="Czas preprocessingu [ms]",
        output_path=PLOTS_DIR / "max_entrances_vs_preprocessing_time.png",
    )

    plot_metric(
        summary=summary,
        x_column="max_entrances",
        y_column="avg_abstract_nodes_visited",
        title="Wpływ liczby wejść na liczbę odwiedzonych węzłów abstrakcyjnych",
        xlabel="Maksymalna liczba wejść między parą klastrów",
        ylabel="Średnia liczba odwiedzonych węzłów abstrakcyjnych",
        output_path=PLOTS_DIR / "max_entrances_vs_abstract_nodes_visited.png",
    )

    plot_metric(
        summary=summary,
        x_column="max_entrances",
        y_column="total_time_ms",
        title="Wpływ liczby wejść na całkowity koszt HPA*",
        xlabel="Maksymalna liczba wejść między parą klastrów",
        ylabel="Całkowity koszt [ms]",
        output_path=PLOTS_DIR / "max_entrances_vs_total_cost.png",
    )


def load_max_entrances_summary() -> pd.DataFrame:
    rows: list[dict] = []

    for csv_path in sorted(RESULTS_DIR.glob("hpa_cluster_*_entrances_*.csv")):
        max_entrances = extract_max_entrances(csv_path)

        data = pd.read_csv(csv_path)
        found_data = data[data["found"] == True]

        rows.append(
            {
                "max_entrances": max_entrances,
                "scenario_count": len(data),
                "found_count": len(found_data),
                "found_rate": len(found_data) / len(data),
                "avg_execution_time_ms": found_data["execution_time_ms"].mean(),
                "avg_query_time_ms": found_data["query_time_ms"].mean(),
                "avg_path_length": found_data["path_length"].mean(),
                "avg_abstract_nodes_visited": found_data[
                    "abstract_nodes_visited"
                ].mean(),
                "preprocessing_time_ms": data["preprocessing_time_ms"]
                .dropna()
                .iloc[0],
            }
        )

    if not rows:
        raise ValueError(f"No max entrances CSV files found in {RESULTS_DIR}")

    summary = pd.DataFrame(rows).sort_values("max_entrances")
    benchmark_scenarios = summary["scenario_count"].iloc[0]

    summary["total_time_ms"] = (
            summary["preprocessing_time_ms"]
            + benchmark_scenarios * summary["avg_query_time_ms"]
    )

    return summary


def extract_max_entrances(
        csv_path: Path,
) -> int:
    return int(csv_path.stem.split("_")[-1])


def plot_metric(
        summary: pd.DataFrame,
        x_column: str,
        y_column: str,
        title: str,
        xlabel: str,
        ylabel: str,
        output_path: Path,
) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(summary[x_column], summary[y_column], marker="o")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
