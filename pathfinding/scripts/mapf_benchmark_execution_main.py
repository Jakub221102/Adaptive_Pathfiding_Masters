"""Execute MAPF-5B benchmark algorithms on the stored MAPF-5A catalogue.

Open in PyCharm and press Run. No command-line arguments required.
Edit the MAPF-5B BENCHMARK EXECUTION CONFIGURATION block below to change settings.
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathfinding.src.experiments.mapf_benchmark_execution import (
    DEFAULT_BENCHMARK_ALGORITHMS,
    MAPFBenchmarkAlgorithm,
    MAPFBenchmarkRunRecord,
    execute_benchmark_manifest_file,
    export_execution_report_csv,
    save_execution_report,
    validate_execution_report,
)

# ============================================================
# MAPF-5B BENCHMARK EXECUTION CONFIGURATION
# ============================================================

MAP = _REPO_ROOT / "Data" / "bg512-map" / "AR0204SR.map"
SCEN = _REPO_ROOT / "Data" / "bg512-scen" / "AR0204SR.map.scen"
MANIFEST = (
    _REPO_ROOT / "pathfinding" / "results" / "mapf_benchmarks" / "AR0204SR_manifest.json"
)

ALGORITHMS: tuple[MAPFBenchmarkAlgorithm, ...] = DEFAULT_BENCHMARK_ALGORITHMS

OUTPUT_JSON = (
    _REPO_ROOT
    / "pathfinding"
    / "results"
    / "mapf_benchmarks"
    / "AR0204SR_execution_results.json"
)
OUTPUT_CSV = (
    _REPO_ROOT
    / "pathfinding"
    / "results"
    / "mapf_benchmarks"
    / "AR0204SR_execution_results.csv"
)
LOG_FILE = _REPO_ROOT / "pathfinding" / "results" / "mapf_benchmark_execution.log"

EXPECTED_INSTANCE_COUNT = 27
EXPECTED_RUN_COUNT = EXPECTED_INSTANCE_COUNT * len(ALGORITHMS)

# ============================================================
# END CONFIGURATION
# ============================================================


class ElapsedTimer:
    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def stamp(self) -> str:
        elapsed = self.elapsed()
        minutes = int(elapsed // 60)
        seconds = elapsed - minutes * 60
        return f"[{minutes:02d}:{seconds:06.3f}]"


def _setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("mapf_benchmark_execution")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.flush = sys.stdout.flush  # type: ignore[method-assign]
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    original_emit = file_handler.emit

    def flushing_emit(record: logging.LogRecord) -> None:
        original_emit(record)
        file_handler.flush()

    file_handler.emit = flushing_emit  # type: ignore[method-assign]
    logger.addHandler(file_handler)

    return logger


def _log(logger: logging.Logger, timer: ElapsedTimer, message: str) -> None:
    logger.info(f"{timer.stamp()} {message}")


def _format_run_record(record: MAPFBenchmarkRunRecord) -> list[str]:
    lines = [
        f"  instance: {record.instance_id}",
        f"  algorithm: {record.algorithm.value}",
        f"  success: {record.success}",
        f"  termination: {record.termination_reason}",
        f"  runtime_s: {record.runtime_s:.4f}",
    ]
    if record.success:
        lines.extend(
            [
                f"  soc: {record.soc}",
                f"  makespan: {record.makespan}",
                f"  independent_soc: {record.independent_soc}",
                f"  independent_makespan: {record.independent_makespan}",
            ]
        )
    if record.pp_search_metrics is not None:
        lines.append(
            "  pp_low_level_searches: "
            f"{record.pp_search_metrics.low_level_searches}"
        )
    if record.cbs_search_metrics is not None:
        lines.extend(
            [
                "  cbs_expanded_ct_nodes: "
                f"{record.cbs_search_metrics.expanded_ct_nodes}",
                "  cbs_generated_ct_nodes: "
                f"{record.cbs_search_metrics.generated_ct_nodes}",
                "  cbs_combined_low_level_searches: "
                f"{record.cbs_search_metrics.combined_low_level_searches}",
            ]
        )
    return lines


def _print_summary(
    logger: logging.Logger,
    *,
    report,
    output_json: Path,
    output_csv: Path,
) -> None:
    success_count = sum(1 for record in report.runs if record.success)
    failure_count = len(report.runs) - success_count

    logger.info("=" * 72)
    logger.info("MAPF-5B BENCHMARK EXECUTION SUMMARY")
    logger.info("=" * 72)
    logger.info(f"Manifest: {report.manifest_path}")
    logger.info(f"Map: {report.map_name}")
    logger.info(f"Scenario: {report.scenario_name}")
    logger.info(f"Max timestep: {report.max_timestep}")
    logger.info(f"Algorithms: {', '.join(algorithm.value for algorithm in report.algorithms)}")
    logger.info(f"Total runs: {len(report.runs)}")
    logger.info(f"Successful runs: {success_count}")
    logger.info(f"Failed runs: {failure_count}")
    logger.info(f"Total runtime: {report.total_runtime_s:.3f} s")
    logger.info(f"JSON results: {output_json}")
    logger.info(f"CSV results: {output_csv}")
    logger.info("")

    by_algorithm: dict[str, list[MAPFBenchmarkRunRecord]] = {}
    for record in report.runs:
        by_algorithm.setdefault(record.algorithm.value, []).append(record)

    logger.info("Algorithm | Success | Fail | Avg runtime (s) | Avg SoC | Avg makespan")
    for algorithm in report.algorithms:
        records = by_algorithm[algorithm.value]
        successes = [record for record in records if record.success]
        failures = len(records) - len(successes)
        avg_runtime = sum(record.runtime_s for record in records) / len(records)
        avg_soc = (
            sum(record.soc for record in successes) / len(successes)
            if successes
            else float("nan")
        )
        avg_makespan = (
            sum(record.makespan for record in successes) / len(successes)
            if successes
            else float("nan")
        )
        logger.info(
            f"{algorithm.value:22} | "
            f"{len(successes):7d} | "
            f"{failures:4d} | "
            f"{avg_runtime:15.4f} | "
            f"{avg_soc:7.1f} | "
            f"{avg_makespan:12.1f}"
        )


def run_benchmark_execution() -> None:
    timer = ElapsedTimer()
    logger = _setup_logging(LOG_FILE)

    _log(logger, timer, "MAPF-5B benchmark execution starting")
    _log(logger, timer, f"Manifest: {MANIFEST}")
    _log(logger, timer, f"Map: {MAP}")
    _log(logger, timer, f"Scenario: {SCEN}")
    _log(
        logger,
        timer,
        "Algorithms: " + ", ".join(algorithm.value for algorithm in ALGORITHMS),
    )

    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Benchmark manifest not found: {MANIFEST}")

    def progress_callback(
        record: MAPFBenchmarkRunRecord,
        completed: int,
        total: int,
    ) -> None:
        _log(
            logger,
            timer,
            f"Completed run {completed}/{total}: "
            f"{record.instance_id} / {record.algorithm.value} "
            f"success={record.success} runtime_s={record.runtime_s:.4f}",
        )
        for line in _format_run_record(record):
            logger.info(line)
        logger.info("")

    report = execute_benchmark_manifest_file(
        manifest_path=MANIFEST,
        map_path=MAP,
        scen_path=SCEN,
        algorithms=ALGORITHMS,
        progress_callback=progress_callback,
    )

    if len(report.runs) != EXPECTED_RUN_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_RUN_COUNT} runs "
            f"({EXPECTED_INSTANCE_COUNT} instances × {len(ALGORITHMS)} algorithms), "
            f"got {len(report.runs)}"
        )

    validate_execution_report(report)
    save_execution_report(report, OUTPUT_JSON)
    export_execution_report_csv(report, OUTPUT_CSV)

    _print_summary(
        logger,
        report=report,
        output_json=OUTPUT_JSON,
        output_csv=OUTPUT_CSV,
    )
    _log(logger, timer, "MAPF-5B benchmark execution complete")


def main() -> None:
    try:
        run_benchmark_execution()
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
