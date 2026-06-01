import csv
from pathlib import Path

from code.src.core.benchmark_models import BenchmarkResult


def export_benchmark_results(
        results: list[BenchmarkResult],
        output_path: Path,
) -> None:
    if not results:
        return

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].model_dump().keys()),
        )

        writer.writeheader()

        for result in results:
            writer.writerow(result.model_dump())
