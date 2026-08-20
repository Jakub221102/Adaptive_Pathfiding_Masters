"""MAPF-5B.1 CBS budget calibration on representative benchmark instances.

Open in PyCharm and press Run. No command-line arguments required.
Edit the MAPF-5B.1 CBS CALIBRATION CONFIGURATION block below to change settings.
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
    MAPFBenchmarkAlgorithm,
    MAIN_BENCHMARK_JSONL,
    TERMINATION_ERROR,
    TERMINATION_EXPANSION_LIMIT,
    TERMINATION_SUCCESS,
    load_checkpoint_records as load_main_benchmark_records,
)
from pathfinding.src.experiments.mapf_cbs_budget_calibration import (
    DEFAULT_CALIBRATION_ALGORITHMS,
    DEFAULT_CALIBRATION_INSTANCE_IDS,
    DEFAULT_EXPANSION_BUDGETS,
    CalibrationCheckpointStore,
    MAPFCBSCalibrationPlanEntry,
    MAPFCBSCalibrationRecord,
    build_pair_summaries,
    calibration_pair_key,
    execute_cbs_calibration_run_from_loaded,
    group_calibration_records_by_pair,
    is_calibration_pair_complete,
    resolve_calibration_instances,
    run_progressive_calibration,
    sort_calibration_instances,
)
from pathfinding.src.experiments.mapf_benchmark_instances import load_benchmark_manifest
from pathfinding.src.loaders.map_loader import load_moving_ai_map
from pathfinding.src.loaders.scen_loader import load_moving_ai_scenarios

# ============================================================
# MAPF-5B.1 CBS CALIBRATION CONFIGURATION
# ============================================================

MANIFEST_PATH = (
    _REPO_ROOT / "pathfinding" / "results" / "mapf_benchmarks" / "AR0204SR_manifest.json"
)
MAP_PATH = _REPO_ROOT / "Data" / "bg512-map" / "AR0204SR.map"
SCEN_PATH = _REPO_ROOT / "Data" / "bg512-scen" / "AR0204SR.map.scen"

MAX_TIMESTEP = 512

ALGORITHMS: tuple[MAPFBenchmarkAlgorithm, ...] = DEFAULT_CALIBRATION_ALGORITHMS

CALIBRATION_INSTANCE_IDS: tuple[str, ...] = DEFAULT_CALIBRATION_INSTANCE_IDS

EXPANSION_BUDGETS: tuple[int, ...] = DEFAULT_EXPANSION_BUDGETS

RESUME = True
RESET_RESULTS = False
STOP_ON_ERROR = False

RESULTS_DIR = _REPO_ROOT / "pathfinding" / "results" / "mapf_cbs_calibration"
RESULTS_CSV = RESULTS_DIR / "calibration_results.csv"
RESULTS_JSONL = RESULTS_DIR / "calibration_results.jsonl"
LOG_FILE = RESULTS_DIR / "calibration.log"

MAIN_BENCHMARK_JSONL = _REPO_ROOT / MAIN_BENCHMARK_JSONL

# ============================================================
# END CONFIGURATION
# ============================================================

ALGORITHM_LABELS: dict[MAPFBenchmarkAlgorithm, str] = {
    MAPFBenchmarkAlgorithm.CBS_BASIC: "Basic CBS",
    MAPFBenchmarkAlgorithm.CBS_CARDINAL_FIRST: "Cardinal-First CBS",
}


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


def _setup_logging(log_file: Path, *, append: bool) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("mapf_cbs_budget_calibration")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.flush = sys.stdout.flush  # type: ignore[method-assign]
    logger.addHandler(stream_handler)

    file_mode = "a" if append else "w"
    file_handler = logging.FileHandler(log_file, mode=file_mode, encoding="utf-8")
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


def _log_blank(logger: logging.Logger) -> None:
    logger.info("")


def _print_run_banner(
    logger: logging.Logger,
    entry: MAPFCBSCalibrationPlanEntry,
    run_number: int,
) -> None:
    _log_blank(logger)
    logger.info("=" * 60)
    logger.info("CALIBRATION RUN")
    logger.info(f"Instance: {entry.instance.instance_id}")
    logger.info(f"Agents: {entry.instance.agent_count}")
    logger.info(f"Interaction: {entry.instance.interaction_level.name}")
    logger.info(f"Algorithm: {ALGORITHM_LABELS[entry.algorithm]}")
    logger.info(f"Expansion budget: {entry.max_expanded_nodes}")
    logger.info("=" * 60)
    _log_blank(logger)


def _print_run_result(logger: logging.Logger, record: MAPFCBSCalibrationRecord) -> None:
    metrics = record.cbs_search_metrics
    logger.info(f"termination_reason: {record.termination_reason}")
    logger.info(f"success: {record.success}")
    logger.info(f"execution_time_ms: {record.execution_time_ms:.2f}")
    logger.info(f"expanded_ct_nodes: {metrics.expanded_ct_nodes}")
    logger.info(f"generated_ct_nodes: {metrics.generated_ct_nodes}")
    logger.info(f"low_level_replans: {metrics.low_level_replans}")
    logger.info(f"max_open_size: {metrics.max_open_size}")
    if record.success:
        logger.info(f"SoC: {record.soc}")
        logger.info(f"makespan: {record.makespan}")
    if record.algorithm == MAPFBenchmarkAlgorithm.CBS_CARDINAL_FIRST:
        logger.info(f"classified_conflicts: {metrics.classified_conflicts}")
        logger.info(
            "classification_low_level_searches: "
            f"{metrics.classification_low_level_searches}"
        )
        logger.info(
            f"selected_cardinal_conflicts: {metrics.selected_cardinal_conflicts}"
        )
        logger.info(
            f"selected_semi_cardinal_conflicts: {metrics.selected_semi_cardinal_conflicts}"
        )
        logger.info(
            f"selected_non_cardinal_conflicts: {metrics.selected_non_cardinal_conflicts}"
        )
    logger.info(f"combined_low_level_searches: {metrics.combined_low_level_searches}")
    _log_blank(logger)
    logger.info("CHECKPOINT SAVED")
    _log_blank(logger)


def _count_pair_outcomes(records, instances, algorithms, budgets):
    grouped = group_calibration_records_by_pair(records)
    successful_pairs = 0
    expansion_limited_pairs = 0

    for instance in instances:
        for algorithm in algorithms:
            pair = calibration_pair_key(instance.instance_id, algorithm)
            pair_records = grouped.get(pair, {})
            if not pair_records:
                continue
            if not is_calibration_pair_complete(
                budgets=budgets,
                existing_records=pair_records,
            ):
                continue
            summaries = build_pair_summaries(
                (instance,),
                algorithms=(algorithm,),
                budgets=budgets,
                records=records,
            )
            if not summaries:
                continue
            if summaries[0].outcome == TERMINATION_SUCCESS:
                successful_pairs += 1
            elif summaries[0].outcome == TERMINATION_EXPANSION_LIMIT:
                expansion_limited_pairs += 1

    errors = sum(
        1 for record in records.values() if record.termination_reason == TERMINATION_ERROR
    )
    return successful_pairs, expansion_limited_pairs, errors


def _optional_pp_reference(logger: logging.Logger, instance_id: str) -> None:
    if not MAIN_BENCHMARK_JSONL.is_file():
        return

    main_records = load_main_benchmark_records(MAIN_BENCHMARK_JSONL)
    for record in main_records.values():
        if (
            record.instance_id == instance_id
            and record.algorithm == MAPFBenchmarkAlgorithm.FIXED_PRIORITY_PP
        ):
            logger.info(
                f"PP reference (main benchmark): execution_time_ms="
                f"{record.execution_time_ms:.2f}"
            )
            return


def _print_startup_summary(
    logger: logging.Logger,
    *,
    instance_count: int,
    pair_count: int,
    completed_runs: int,
    algorithms,
    budgets,
) -> None:
    _log_blank(logger)
    logger.info("MAPF-5B.1 CBS BUDGET CALIBRATION")
    _log_blank(logger)
    logger.info(f"Manifest: {MANIFEST_PATH}")
    logger.info(f"Representative instances: {instance_count}")
    logger.info(f"Algorithm-instance pairs: {pair_count}")
    logger.info(f"Budget ladder: {', '.join(str(b) for b in budgets)}")
    logger.info(f"Already completed calibration runs: {completed_runs}")
    logger.info(f"Resume: {RESUME}")
    logger.info(f"Reset results: {RESET_RESULTS}")
    _log_blank(logger)
    logger.info("Algorithms:")
    for algorithm in algorithms:
        logger.info(f"  - {ALGORITHM_LABELS[algorithm]}")
    _log_blank(logger)
    logger.info("Execution order:")
    logger.info("  agent_count 5 → 10 → 20")
    logger.info("  interaction LOW → MEDIUM → HIGH")
    logger.info("  Basic CBS, then Cardinal-First CBS per instance")
    logger.info("  budget ladder ascending with progressive escalation")
    _log_blank(logger)
    logger.info(f"Results CSV: {RESULTS_CSV}")
    logger.info(f"Results JSONL: {RESULTS_JSONL}")
    logger.info(f"Log: {LOG_FILE}")
    _log_blank(logger)


def _print_progress(
    logger: logging.Logger,
    timer: ElapsedTimer,
    *,
    completed_runs: int,
    records,
    instances,
    algorithms,
) -> None:
    successful_pairs, expansion_limited_pairs, errors = _count_pair_outcomes(
        records,
        instances,
        algorithms,
        EXPANSION_BUDGETS,
    )
    logger.info(
        f"Completed calibration runs: {completed_runs} | "
        f"Successful algorithm-instance pairs: {successful_pairs} | "
        f"Still expansion-limited pairs: {expansion_limited_pairs} | "
        f"Errors: {errors} | "
        f"Elapsed total: {timer.elapsed():.1f} s"
    )
    _log_blank(logger)


def _print_pair_summaries(
    logger: logging.Logger,
    summaries,
) -> None:
    _log_blank(logger)
    logger.info("=" * 72)
    logger.info("CALIBRATION PER-PAIR SUMMARY")
    logger.info("=" * 72)
    for summary in summaries:
        label = ALGORITHM_LABELS.get(summary.algorithm, summary.algorithm.value)
        logger.info(f"{summary.instance_id} | {label}")
        logger.info(f"  result: {summary.outcome}")
        if summary.smallest_successful_budget is not None:
            logger.info(
                f"  smallest successful budget: {summary.smallest_successful_budget}"
            )
            logger.info(
                f"  runtime at successful budget: "
                f"{summary.runtime_at_outcome_ms:.2f} ms"
            )
        else:
            logger.info(f"  largest tested budget: {summary.largest_tested_budget}")
            if summary.runtime_at_outcome_ms is not None:
                logger.info(
                    f"  runtime at budget {summary.largest_tested_budget}: "
                    f"{summary.runtime_at_outcome_ms:.2f} ms"
                )
        _log_blank(logger)


def _print_aggregate_table(
    logger: logging.Logger,
    summaries,
) -> None:
    _log_blank(logger)
    logger.info("=" * 72)
    logger.info("CALIBRATION AGGREGATE TABLE")
    logger.info("=" * 72)
    logger.info(
        "Agents | Level | Algorithm | Highest budget tested | Outcome | "
        "Runtime (ms) | Expansions | LL searches"
    )
    for summary in summaries:
        algorithm_label = "basic" if summary.algorithm == MAPFBenchmarkAlgorithm.CBS_BASIC else "cardinal"
        logger.info(
            f"{summary.agent_count:6} | "
            f"{summary.interaction_level:5} | "
            f"{algorithm_label:9} | "
            f"{summary.largest_tested_budget:21} | "
            f"{summary.outcome:7} | "
            f"{summary.runtime_at_outcome_ms or 0:11.1f} | "
            f"{summary.expanded_ct_nodes_at_outcome or 0:10} | "
            f"{summary.combined_low_level_searches_at_outcome or 0:11}"
        )
    _log_blank(logger)


def run_cbs_budget_calibration() -> None:
    timer = ElapsedTimer()

    if RESET_RESULTS and RESUME:
        raise ValueError("RESET_RESULTS=True cannot be combined with RESUME=True")

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Benchmark manifest not found: {MANIFEST_PATH}")

    checkpoint = CalibrationCheckpointStore(
        results_dir=RESULTS_DIR,
        csv_path=RESULTS_CSV,
        jsonl_path=RESULTS_JSONL,
        reset_results=RESET_RESULTS,
    )

    append_log = RESUME and LOG_FILE.is_file() and not RESET_RESULTS
    logger = _setup_logging(LOG_FILE, append=append_log)

    manifest = load_benchmark_manifest(MANIFEST_PATH)
    max_timestep = MAX_TIMESTEP if MAX_TIMESTEP is not None else manifest.max_timestep

    instances = sort_calibration_instances(
        resolve_calibration_instances(manifest, CALIBRATION_INSTANCE_IDS),
        CALIBRATION_INSTANCE_IDS,
    )

    pair_count = len(instances) * len(ALGORITHMS)
    _print_startup_summary(
        logger,
        instance_count=len(instances),
        pair_count=pair_count,
        completed_runs=len(checkpoint.records),
        algorithms=ALGORITHMS,
        budgets=EXPANSION_BUDGETS,
    )

    grid_map = load_moving_ai_map(MAP_PATH)
    scenarios = load_moving_ai_scenarios(SCEN_PATH)

    def executor(entry: MAPFCBSCalibrationPlanEntry) -> MAPFCBSCalibrationRecord:
        return execute_cbs_calibration_run_from_loaded(
            grid_map=grid_map,
            scenarios=scenarios,
            instance=entry.instance,
            algorithm=entry.algorithm,
            max_timestep=max_timestep,
            max_expanded_nodes=entry.max_expanded_nodes,
        )

    session_start_count = len(checkpoint.records)

    def on_before_run(entry: MAPFCBSCalibrationPlanEntry, run_number: int) -> None:
        _print_run_banner(logger, entry, run_number)
        _optional_pp_reference(logger, entry.instance.instance_id)
        _log(logger, timer, "Calibration run started")

    def on_record(record: MAPFCBSCalibrationRecord, _run_number: int) -> None:
        _print_run_result(logger, record)
        _print_progress(
            logger,
            timer,
            completed_runs=len(checkpoint.records),
            records=checkpoint.records,
            instances=instances,
            algorithms=ALGORITHMS,
        )

    try:
        run_progressive_calibration(
            instances,
            algorithms=ALGORITHMS,
            budgets=EXPANSION_BUDGETS,
            executor=executor,
            checkpoint=checkpoint,
            resume=RESUME,
            on_before_run=on_before_run,
            on_record=on_record,
            stop_on_error=STOP_ON_ERROR,
        )
    except KeyboardInterrupt:
        _log_blank(logger)
        logger.info("=" * 72)
        logger.info("CALIBRATION INTERRUPTED BY USER")
        logger.info("=" * 72)
        logger.info(f"Completed calibration runs: {len(checkpoint.records)}")
        logger.info(f"Runs added this session: {len(checkpoint.records) - session_start_count}")
        logger.info(f"Checkpoint preserved in: {RESULTS_JSONL}")
        logger.info(f"Elapsed before interrupt: {timer.elapsed():.1f} s")
        logger.info("Press Run again with RESUME=True to continue.")
        _log_blank(logger)
        raise SystemExit(130)

    summaries = build_pair_summaries(
        instances,
        algorithms=ALGORITHMS,
        budgets=EXPANSION_BUDGETS,
        records=checkpoint.records,
    )
    _print_pair_summaries(logger, summaries)
    _print_aggregate_table(logger, summaries)
    _log(logger, timer, "MAPF-5B.1 CBS budget calibration complete")


def main() -> None:
    try:
        run_cbs_budget_calibration()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
