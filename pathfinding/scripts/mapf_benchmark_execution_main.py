"""Execute MAPF-5B benchmark algorithms on the stored MAPF-5A catalogue.

Open in PyCharm and press Run. No command-line arguments required.
Edit the MAPF-5B LOCAL BENCHMARK CONFIGURATION block below to change settings.
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from collections.abc import Sequence
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathfinding.src.experiments.mapf_benchmark_execution import (
    DEFAULT_BENCHMARK_ALGORITHMS,
    TERMINATION_ERROR,
    TERMINATION_EXPANSION_LIMIT,
    TERMINATION_FAILURE,
    TERMINATION_SUCCESS,
    TERMINATION_TIME_LIMIT,
    BenchmarkCheckpointStore,
    MAPFBenchmarkAlgorithm,
    MAPFBenchmarkCBSLimits,
    MAPFBenchmarkRunKey,
    MAPFBenchmarkRunPlanEntry,
    MAPFBenchmarkRunRecord,
    build_benchmark_run_plan,
    create_error_benchmark_record,
    execute_benchmark_run,
    load_benchmark_manifest,
    reconstruct_mapf_scenario_from_instance,
    run_key,
    validate_benchmark_run_record,
)
from pathfinding.src.experiments.mapf_cbs_process_runner import terminate_process
from pathfinding.src.loaders.map_loader import load_moving_ai_map
from pathfinding.src.loaders.scen_loader import load_moving_ai_scenarios

# ============================================================
# MAPF-5B LOCAL BENCHMARK CONFIGURATION
# ============================================================

MANIFEST_PATH = (
    _REPO_ROOT / "pathfinding" / "results" / "mapf_benchmarks" / "AR0204SR_manifest.json"
)
MAP_PATH = _REPO_ROOT / "Data" / "bg512-map" / "AR0204SR.map"
SCEN_PATH = _REPO_ROOT / "Data" / "bg512-scen" / "AR0204SR.map.scen"

ALGORITHMS: tuple[MAPFBenchmarkAlgorithm, ...] = DEFAULT_BENCHMARK_ALGORITHMS

AGENT_COUNTS: tuple[int, ...] | None = (5, 10, 20)
INTERACTION_LEVELS: tuple[str, ...] | None = ("low", "medium", "high")
RUN_INDEX_FILTER: tuple[int, ...] | None = None

MAX_TIMESTEP: int | None = None

RESUME = True
RESET_RESULTS = False
SAVE_AFTER_EACH_RUN = True
STOP_ON_ERROR = False

# CBS execution guards
# 180 s: 10-agent MEDIUM successes were ~147–151 s; 120 s is too tight.
# Difficult 10-agent HIGH cases exceeded 160 s at only 25 CT expansions.
BASIC_CBS_MAX_EXPANDED_NODES: int | None = 250
BASIC_CBS_MAX_RUNTIME_SECONDS: float | None = 180.0

CARDINAL_CBS_MAX_EXPANDED_NODES: int | None = 250
CARDINAL_CBS_MAX_RUNTIME_SECONDS: float | None = 180.0

RESULTS_DIR = _REPO_ROOT / "pathfinding" / "results" / "mapf_benchmark_execution"
RESULTS_CSV = RESULTS_DIR / "results.csv"
RESULTS_JSONL = RESULTS_DIR / "results_details.jsonl"
LOG_FILE = RESULTS_DIR / "run.log"

# ============================================================
# END CONFIGURATION
# ============================================================

ALGORITHM_LABELS: dict[MAPFBenchmarkAlgorithm, str] = {
    MAPFBenchmarkAlgorithm.FIXED_PRIORITY_PP: "Fixed-Priority PP",
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

    logger = logging.getLogger("mapf_benchmark_execution")
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


def _format_limit(value: int | None) -> str:
    if value is None:
        return "unlimited"
    return str(value)


def _format_runtime_limit(value: float | None) -> str:
    if value is None:
        return "unlimited"
    return f"{value:.1f} s"


def _count_by_termination(
    records: Sequence[MAPFBenchmarkRunRecord],
) -> dict[str, int]:
    counts = {
        TERMINATION_SUCCESS: 0,
        TERMINATION_FAILURE: 0,
        TERMINATION_EXPANSION_LIMIT: 0,
        TERMINATION_TIME_LIMIT: 0,
        TERMINATION_ERROR: 0,
    }
    for record in records:
        counts[record.termination_reason] = counts.get(record.termination_reason, 0) + 1
    return counts


def _print_run_banner(
    logger: logging.Logger,
    *,
    run_number: int,
    total_runs: int,
    entry: MAPFBenchmarkRunPlanEntry,
) -> None:
    _log_blank(logger)
    logger.info("=" * 60)
    logger.info(f"RUN {run_number} / {total_runs}")
    logger.info(f"Instance: {entry.instance.instance_id}")
    logger.info(f"Agents: {entry.instance.agent_count}")
    logger.info(f"Interaction: {entry.instance.interaction_level.name}")
    logger.info(f"Algorithm: {ALGORITHM_LABELS[entry.algorithm]}")
    if entry.algorithm == MAPFBenchmarkAlgorithm.CBS_BASIC:
        logger.info(f"Max CT expansions: {_format_limit(BASIC_CBS_MAX_EXPANDED_NODES)}")
        logger.info(
            f"Wall-clock limit: {_format_runtime_limit(BASIC_CBS_MAX_RUNTIME_SECONDS)}"
        )
    elif entry.algorithm == MAPFBenchmarkAlgorithm.CBS_CARDINAL_FIRST:
        logger.info(
            f"Max CT expansions: {_format_limit(CARDINAL_CBS_MAX_EXPANDED_NODES)}"
        )
        logger.info(
            "Wall-clock limit: "
            f"{_format_runtime_limit(CARDINAL_CBS_MAX_RUNTIME_SECONDS)}"
        )
    logger.info("=" * 60)
    _log_blank(logger)


def _print_run_result(
    logger: logging.Logger,
    record: MAPFBenchmarkRunRecord,
) -> None:
    logger.info("RESULT")
    logger.info(f"success: {record.success}")
    logger.info(f"termination_reason: {record.termination_reason}")
    logger.info(f"execution_time_ms: {record.execution_time_ms:.2f}")
    if record.success:
        logger.info(f"SoC: {record.soc}")
        logger.info(f"makespan: {record.makespan}")
        logger.info(f"remaining_conflicts: {record.remaining_conflicts}")
    if record.pp_search_metrics is not None:
        logger.info(
            "pp_low_level_searches: "
            f"{record.pp_search_metrics.low_level_searches}"
        )
        logger.info(
            "pp_agents_planned: "
            f"{record.pp_search_metrics.agents_planned}"
        )
    if record.cbs_search_metrics is not None:
        logger.info(
            "expanded CT nodes: "
            f"{record.cbs_search_metrics.expanded_ct_nodes}"
        )
        logger.info(
            "generated CT nodes: "
            f"{record.cbs_search_metrics.generated_ct_nodes}"
        )
        logger.info(
            "low-level replans: "
            f"{record.cbs_search_metrics.low_level_replans}"
        )
        logger.info(
            "max OPEN size: "
            f"{record.cbs_search_metrics.max_open_size}"
        )
        logger.info(
            "combined low-level searches: "
            f"{record.cbs_search_metrics.combined_low_level_searches}"
        )
        if record.algorithm == MAPFBenchmarkAlgorithm.CBS_CARDINAL_FIRST:
            logger.info(
                "classified_conflicts: "
                f"{record.cbs_search_metrics.classified_conflicts}"
            )
            logger.info(
                "classification low-level searches: "
                f"{record.cbs_search_metrics.classification_low_level_searches}"
            )
    if record.error_message is not None:
        logger.info(f"error_message: {record.error_message}")
    if record.termination_reason == TERMINATION_TIME_LIMIT:
        logger.info("CBS worker terminated after wall-clock limit.")
    _log_blank(logger)


def _print_global_progress(
    logger: logging.Logger,
    timer: ElapsedTimer,
    *,
    completed: int,
    total_runs: int,
    records: Sequence[MAPFBenchmarkRunRecord],
) -> None:
    counts = _count_by_termination(records)
    logger.info(
        f"Completed: {completed} / {total_runs} | "
        f"Succeeded: {counts[TERMINATION_SUCCESS]} | "
        f"Failed: {counts[TERMINATION_FAILURE]} | "
        f"Expansion-limited: {counts[TERMINATION_EXPANSION_LIMIT]} | "
        f"Time-limited: {counts[TERMINATION_TIME_LIMIT]} | "
        f"Errors: {counts[TERMINATION_ERROR]} | "
        f"Elapsed total: {timer.elapsed():.1f} s"
    )
    _log_blank(logger)


def _print_startup_summary(
    logger: logging.Logger,
    *,
    manifest_path: Path,
    selected_instance_count: int,
    total_runs: int,
    already_completed: int,
    remaining: int,
) -> None:
    _log_blank(logger)
    logger.info("MAPF-5B BENCHMARK EXECUTION")
    _log_blank(logger)
    logger.info(f"Manifest:\n  {manifest_path}")
    _log_blank(logger)
    logger.info("Algorithms:")
    for algorithm in ALGORITHMS:
        logger.info(f"  - {ALGORITHM_LABELS[algorithm]}")
    _log_blank(logger)
    logger.info(f"Selected instances: {selected_instance_count}")
    logger.info(f"Total possible runs: {total_runs}")
    logger.info(f"Already completed: {already_completed}")
    logger.info(f"Remaining: {remaining}")
    logger.info(f"Resume: {RESUME}")
    logger.info(f"Reset results: {RESET_RESULTS}")
    logger.info(f"Save after each run: {SAVE_AFTER_EACH_RUN}")
    logger.info(f"Stop on error: {STOP_ON_ERROR}")
    _log_blank(logger)
    logger.info("CBS execution guards:")
    logger.info("  Basic:")
    logger.info(f"    max CT expansions: {_format_limit(BASIC_CBS_MAX_EXPANDED_NODES)}")
    logger.info(
        f"    wall-clock limit: {_format_runtime_limit(BASIC_CBS_MAX_RUNTIME_SECONDS)}"
    )
    logger.info("  Cardinal-First:")
    logger.info(
        f"    max CT expansions: {_format_limit(CARDINAL_CBS_MAX_EXPANDED_NODES)}"
    )
    logger.info(
        "    wall-clock limit: "
        f"{_format_runtime_limit(CARDINAL_CBS_MAX_RUNTIME_SECONDS)}"
    )
    _log_blank(logger)
    logger.info("Execution order:")
    logger.info("  Phase 1: Fixed-Priority PP on all selected instances")
    logger.info("  Phase 2: Basic CBS on all selected instances")
    logger.info("  Phase 3: Cardinal-First CBS on all selected instances")
    logger.info("  Within each phase: agent_count 5→10→20, LOW→MEDIUM→HIGH")
    _log_blank(logger)
    logger.info(f"Results CSV: {RESULTS_CSV}")
    logger.info(f"Results JSONL: {RESULTS_JSONL}")
    logger.info(f"Log: {LOG_FILE}")
    _log_blank(logger)


def _print_completion_summary(
    logger: logging.Logger,
    timer: ElapsedTimer,
    records: Sequence[MAPFBenchmarkRunRecord],
) -> None:
    counts = _count_by_termination(records)
    _log_blank(logger)
    logger.info("=" * 72)
    logger.info("MAPF-5B BENCHMARK EXECUTION COMPLETE")
    logger.info("=" * 72)
    logger.info(f"Total runs: {len(records)}")
    logger.info(f"Completed: {len(records)}")
    logger.info(f"Success: {counts[TERMINATION_SUCCESS]}")
    logger.info(f"Failure: {counts[TERMINATION_FAILURE]}")
    logger.info(f"Expansion limit: {counts[TERMINATION_EXPANSION_LIMIT]}")
    logger.info(f"Time limit: {counts[TERMINATION_TIME_LIMIT]}")
    logger.info(f"Errors: {counts[TERMINATION_ERROR]}")
    _log_blank(logger)

    by_algorithm: dict[MAPFBenchmarkAlgorithm, list[MAPFBenchmarkRunRecord]] = {}
    for record in records:
        by_algorithm.setdefault(record.algorithm, []).append(record)

    logger.info("By algorithm:")
    for algorithm in ALGORITHMS:
        algorithm_records = by_algorithm.get(algorithm, [])
        algorithm_counts = _count_by_termination(algorithm_records)
        logger.info(
            f"  {ALGORITHM_LABELS[algorithm]}: "
            f"success={algorithm_counts[TERMINATION_SUCCESS]}, "
            f"failure={algorithm_counts[TERMINATION_FAILURE]}, "
            f"expansion_limit={algorithm_counts[TERMINATION_EXPANSION_LIMIT]}, "
            f"time_limit={algorithm_counts[TERMINATION_TIME_LIMIT]}, "
            f"errors={algorithm_counts[TERMINATION_ERROR]}"
        )

    _log_blank(logger)
    logger.info(f"Total elapsed: {timer.elapsed():.1f} s")
    _log_blank(logger)


def _print_interrupt_summary(
    logger: logging.Logger,
    timer: ElapsedTimer,
    *,
    completed: int,
    total_runs: int,
    records: Sequence[MAPFBenchmarkRunRecord],
) -> None:
    remaining = total_runs - completed
    _log_blank(logger)
    logger.info("=" * 72)
    logger.info("BENCHMARK INTERRUPTED BY USER")
    logger.info("=" * 72)
    logger.info(f"Completed before interrupt: {completed} / {total_runs}")
    logger.info(f"Remaining: {remaining}")
    logger.info(f"Checkpointed results preserved in: {RESULTS_JSONL}")
    logger.info(f"Flat CSV preserved in: {RESULTS_CSV}")
    logger.info(f"Elapsed before interrupt: {timer.elapsed():.1f} s")
    logger.info("Press Run again with RESUME=True to continue.")
    _log_blank(logger)


def run_benchmark_execution() -> None:
    timer = ElapsedTimer()

    if RESET_RESULTS and RESUME:
        raise ValueError("RESET_RESULTS=True cannot be combined with RESUME=True")

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Benchmark manifest not found: {MANIFEST_PATH}")

    checkpoint = BenchmarkCheckpointStore(
        results_dir=RESULTS_DIR,
        csv_path=RESULTS_CSV,
        jsonl_path=RESULTS_JSONL,
        reset_results=RESET_RESULTS,
    )

    append_log = RESUME and LOG_FILE.is_file() and not RESET_RESULTS
    logger = _setup_logging(LOG_FILE, append=append_log)

    manifest = load_benchmark_manifest(MANIFEST_PATH)
    max_timestep = MAX_TIMESTEP if MAX_TIMESTEP is not None else manifest.max_timestep

    plan = build_benchmark_run_plan(
        manifest,
        algorithms=ALGORITHMS,
        agent_counts=AGENT_COUNTS,
        interaction_levels=INTERACTION_LEVELS,
        run_index_filter=RUN_INDEX_FILTER,
    )
    total_runs = len(plan)
    completed_keys = checkpoint.completed_keys() if RESUME else set()

    already_completed = sum(
        1
        for entry in plan
        if MAPFBenchmarkRunKey(entry.instance.instance_id, entry.algorithm)
        in completed_keys
    )
    remaining = total_runs - already_completed

    selected_instance_ids = {entry.instance.instance_id for entry in plan}
    _print_startup_summary(
        logger,
        manifest_path=MANIFEST_PATH,
        selected_instance_count=len(selected_instance_ids),
        total_runs=total_runs,
        already_completed=already_completed,
        remaining=remaining,
    )

    grid_map = load_moving_ai_map(MAP_PATH)
    scenarios = load_moving_ai_scenarios(SCEN_PATH)
    if grid_map.name != manifest.map_name:
        raise ValueError(
            f"map name mismatch: manifest expects {manifest.map_name!r}, "
            f"loaded {grid_map.name!r}"
        )

    cbs_limits = MAPFBenchmarkCBSLimits(
        basic_max_expanded_nodes=BASIC_CBS_MAX_EXPANDED_NODES,
        cardinal_first_max_expanded_nodes=CARDINAL_CBS_MAX_EXPANDED_NODES,
        basic_max_runtime_seconds=BASIC_CBS_MAX_RUNTIME_SECONDS,
        cardinal_first_max_runtime_seconds=CARDINAL_CBS_MAX_RUNTIME_SECONDS,
    )

    completed = 0
    session_records: list[MAPFBenchmarkRunRecord] = list(checkpoint.records.values())
    active_worker: dict[str, object] = {"process": None}

    try:
        for run_number, entry in enumerate(plan, start=1):
            key = MAPFBenchmarkRunKey(entry.instance.instance_id, entry.algorithm)
            if RESUME and key in completed_keys:
                logger.info(
                    f"SKIP — already completed: {entry.instance.instance_id} / "
                    f"{entry.algorithm.value}"
                )
                continue

            _print_run_banner(
                logger,
                run_number=run_number,
                total_runs=total_runs,
                entry=entry,
            )
            _log(logger, timer, "Run started")

            scenario = reconstruct_mapf_scenario_from_instance(
                scenarios=scenarios,
                grid_map=grid_map,
                instance=entry.instance,
            )

            run_start = time.perf_counter()
            try:
                record = execute_benchmark_run(
                    grid_map=grid_map,
                    scenario=scenario,
                    instance=entry.instance,
                    algorithm=entry.algorithm,
                    max_timestep=max_timestep,
                    cbs_limits=cbs_limits,
                    active_worker=active_worker,
                )
            except Exception as error:
                execution_time_ms = (time.perf_counter() - run_start) * 1000.0
                logger.info("UNEXPECTED ERROR")
                logger.info(traceback.format_exc())
                record = create_error_benchmark_record(
                    instance=entry.instance,
                    algorithm=entry.algorithm,
                    execution_time_ms=execution_time_ms,
                    error_message=f"{type(error).__name__}: {error}",
                )
                if STOP_ON_ERROR:
                    if SAVE_AFTER_EACH_RUN:
                        validate_benchmark_run_record(record)
                        checkpoint.append_record(record)
                        session_records.append(record)
                        logger.info("CHECKPOINT SAVED")
                    raise

            validate_benchmark_run_record(record)
            _print_run_result(logger, record)

            if SAVE_AFTER_EACH_RUN:
                checkpoint.append_record(record)
                completed_keys.add(run_key(record))
                logger.info("CHECKPOINT SAVED")

            session_records.append(record)
            completed += 1
            _print_global_progress(
                logger,
                timer,
                completed=already_completed + completed,
                total_runs=total_runs,
                records=session_records,
            )

    except KeyboardInterrupt:
        worker = active_worker.get("process")
        if worker is not None:
            from multiprocessing import Process

            if isinstance(worker, Process):
                terminate_process(worker)
        _print_interrupt_summary(
            logger,
            timer,
            completed=already_completed + completed,
            total_runs=total_runs,
            records=session_records,
        )
        raise SystemExit(130)

    if already_completed + completed < total_runs:
        logger.info(
            f"Stopped early: {already_completed + completed} / {total_runs} runs present"
        )
        return

    _print_completion_summary(logger, timer, session_records)


def main() -> None:
    try:
        run_benchmark_execution()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
