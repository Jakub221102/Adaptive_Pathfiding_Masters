"""Execute plain SPF baseline on the FROZEN held-out validation catalogue.

Open in PyCharm and press Run. No command-line arguments required.
Edit the MAPF-7.8a HELD-OUT SPF CONFIGURATION block below to change settings.

This runner reuses the existing MAPF-6 SPF execution path. It does NOT modify
the held-out manifest, primary MAPF-6 results, or MAPF-7 held-out CDF results.
"""

from __future__ import annotations

import hashlib
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
    TERMINATION_FAILURE,
    TERMINATION_SUCCESS,
)
from pathfinding.src.experiments.mapf_benchmark_instances import (
    CATALOGUE_ROLE_HELD_OUT,
    DEFAULT_HELD_OUT_SEED,
    load_benchmark_manifest,
)
from pathfinding.src.experiments.mapf_conflict_aware_priority_execution import (
    HELD_OUT_CONFLICT_AWARE_JSONL,
    PRIMARY_MANIFEST_PATH,
    validate_held_out_execution_manifest,
)
from pathfinding.src.experiments.mapf_priority_strategy_execution import (
    HELD_OUT_SPF_CSV,
    HELD_OUT_SPF_JSONL,
    HELD_OUT_SPF_LOG,
    HELD_OUT_SPF_RESULTS_DIR,
    TERMINATION_ORDERING_FAILURE,
    PriorityStrategy,
    PriorityStrategyCheckpointStore,
    PriorityStrategyRunKey,
    PriorityStrategyRunPlanEntry,
    PriorityStrategyRunRecord,
    build_priority_strategy_run_plan,
    execute_priority_strategy_plan_entry,
    run_key,
    validate_priority_strategy_run_record,
)
from pathfinding.src.loaders.map_loader import load_moving_ai_map
from pathfinding.src.loaders.scen_loader import load_moving_ai_scenarios

# ============================================================
# MAPF-7.8a HELD-OUT SPF CONFIGURATION
# ============================================================

MANIFEST_PATH = (
    _REPO_ROOT / "pathfinding" / "results" / "mapf_benchmarks" / "AR0204SR_heldout_manifest.json"
)
MAP_PATH = _REPO_ROOT / "Data" / "bg512-map" / "AR0204SR.map"
SCEN_PATH = _REPO_ROOT / "Data" / "bg512-scen" / "AR0204SR.map.scen"

# Set SMOKE_MODE=True for infrastructure validation only (1 instance × SPF).
SMOKE_MODE = False

STRATEGIES: tuple[PriorityStrategy, ...] = (PriorityStrategy.SPF,)
MAX_TIMESTEP: int | None = None

AGENT_COUNTS: tuple[int, ...] | None = (5, 10, 20)
INTERACTION_LEVELS: tuple[str, ...] | None = ("low", "medium", "high")
INSTANCE_IDS: tuple[str, ...] | None = None

SMOKE_INSTANCE_IDS: tuple[str, ...] = ("AR0204SR_HO_n05_low_000",)

RESUME = False
RESET_RESULTS = True
SAVE_AFTER_EACH_RUN = True
STOP_ON_ERROR = False

RESULTS_DIR = _REPO_ROOT / HELD_OUT_SPF_RESULTS_DIR
RESULTS_CSV = _REPO_ROOT / HELD_OUT_SPF_CSV
RESULTS_JSONL = _REPO_ROOT / HELD_OUT_SPF_JSONL
LOG_FILE = _REPO_ROOT / HELD_OUT_SPF_LOG

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


def _effective_instance_ids() -> tuple[str, ...] | None:
    if SMOKE_MODE:
        return SMOKE_INSTANCE_IDS
    return INSTANCE_IDS


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _setup_logging(log_file: Path, *, append: bool) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("mapf_spf_heldout_execution")
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


def _count_by_outcome(
    records: Sequence[PriorityStrategyRunRecord],
) -> dict[str, int]:
    counts = {
        TERMINATION_SUCCESS: 0,
        TERMINATION_FAILURE: 0,
        TERMINATION_ORDERING_FAILURE: 0,
    }
    for record in records:
        counts[record.termination_reason] = (
            counts.get(record.termination_reason, 0) + 1
        )
    return counts


def _print_startup_summary(
    logger: logging.Logger,
    *,
    manifest_path: Path,
    catalogue_seed: int,
    max_timestep: int,
    selected_instance_count: int,
    total_runs: int,
    already_completed: int,
    remaining: int,
    smoke_mode: bool,
) -> None:
    _log_blank(logger)
    logger.info("MAPF-7 HELD-OUT SPF BASELINE")
    _log_blank(logger)
    logger.info(f"Manifest: {manifest_path.name}")
    logger.info(f"  {manifest_path}")
    logger.info(f"Catalogue seed: {catalogue_seed}")
    logger.info(f"Catalogue role: {CATALOGUE_ROLE_HELD_OUT}")
    _log_blank(logger)
    logger.info(f"Smoke mode: {smoke_mode}")
    logger.info(f"Max timestep: {max_timestep}")
    logger.info("Strategy: SPF (Shortest-Path-First)")
    logger.info(f"Selected instances: {selected_instance_count}")
    logger.info(f"Total planned runs: {total_runs}")
    logger.info(f"Already completed: {already_completed}")
    logger.info(f"Remaining: {remaining}")
    logger.info(f"Resume: {RESUME}")
    logger.info(f"Reset results: {RESET_RESULTS}")
    logger.info(f"Save after each run: {SAVE_AFTER_EACH_RUN}")
    logger.info(f"Stop on error: {STOP_ON_ERROR}")
    _log_blank(logger)
    logger.info("Execution order:")
    logger.info("  For each held-out instance (agent_count 5→10→20, LOW→MEDIUM→HIGH):")
    logger.info("    SPF")
    _log_blank(logger)
    logger.info(f"Output directory: {RESULTS_DIR}")
    logger.info(f"Results CSV: {RESULTS_CSV}")
    logger.info(f"Results JSONL: {RESULTS_JSONL}")
    logger.info(f"Log: {LOG_FILE}")
    _log_blank(logger)


def _print_run_banner(
    logger: logging.Logger,
    *,
    run_number: int,
    total_runs: int,
    entry: PriorityStrategyRunPlanEntry,
) -> None:
    _log_blank(logger)
    logger.info("=" * 60)
    logger.info(f"RUN {run_number} / {total_runs}")
    logger.info(f"Held-out instance: {entry.instance.instance_id}")
    logger.info(f"Agents: {entry.instance.agent_count}")
    logger.info(f"Interaction: {entry.instance.interaction_level.name}")
    logger.info("Strategy: SPF (Shortest-Path-First)")
    logger.info("=" * 60)
    _log_blank(logger)


def _print_run_result(
    logger: logging.Logger,
    record: PriorityStrategyRunRecord,
) -> None:
    logger.info("RESULT")
    logger.info(f"success: {record.success}")
    logger.info(f"termination_reason: {record.termination_reason}")
    logger.info(f"ordering_time_ms: {record.ordering_time_ms:.2f}")
    if record.pp_time_ms is not None:
        logger.info(f"pp_time_ms: {record.pp_time_ms:.2f}")
    else:
        logger.info("pp_time_ms: null")
    logger.info(f"total_time_ms: {record.total_time_ms:.2f}")
    if record.agent_order is not None:
        logger.info(f"agent_order: {list(record.agent_order)}")
    if record.success:
        logger.info(f"SoC: {record.soc}")
        logger.info(f"makespan: {record.makespan}")
        logger.info(f"conflict_count: {record.conflict_count}")
    if record.pp_search_metrics is not None:
        logger.info(
            "pp_low_level_searches: "
            f"{record.pp_search_metrics.low_level_searches}"
        )
        logger.info(
            "pp_agents_planned: "
            f"{record.pp_search_metrics.agents_planned}"
        )
    if record.error_message is not None:
        logger.info(f"error_message: {record.error_message}")
    _log_blank(logger)


def _print_global_progress(
    logger: logging.Logger,
    timer: ElapsedTimer,
    *,
    completed: int,
    total_runs: int,
    records: Sequence[PriorityStrategyRunRecord],
) -> None:
    counts = _count_by_outcome(records)
    logger.info(
        f"Completed: {completed} / {total_runs} | "
        f"Succeeded: {counts[TERMINATION_SUCCESS]} | "
        f"Failed: {counts[TERMINATION_FAILURE]} | "
        f"Ordering failed: {counts[TERMINATION_ORDERING_FAILURE]} | "
        f"Elapsed total: {timer.elapsed():.1f} s"
    )
    _log_blank(logger)


def _print_completion_summary(
    logger: logging.Logger,
    timer: ElapsedTimer,
    records: Sequence[PriorityStrategyRunRecord],
) -> None:
    counts = _count_by_outcome(records)
    _log_blank(logger)
    logger.info("=" * 72)
    logger.info("MAPF-7 HELD-OUT SPF BASELINE EXECUTION COMPLETE")
    logger.info("=" * 72)
    logger.info(f"Total runs: {len(records)}")
    logger.info(f"Success: {counts[TERMINATION_SUCCESS]}")
    logger.info(f"Failure: {counts[TERMINATION_FAILURE]}")
    logger.info(f"Ordering failure: {counts[TERMINATION_ORDERING_FAILURE]}")
    _log_blank(logger)
    logger.info(f"Total elapsed: {timer.elapsed():.1f} s")
    _log_blank(logger)


def _print_interrupt_summary(
    logger: logging.Logger,
    timer: ElapsedTimer,
    *,
    completed: int,
    total_runs: int,
) -> None:
    remaining = total_runs - completed
    _log_blank(logger)
    logger.info("=" * 72)
    logger.info("HELD-OUT SPF BENCHMARK INTERRUPTED BY USER")
    logger.info("=" * 72)
    logger.info(f"Completed before interrupt: {completed} / {total_runs}")
    logger.info(f"Remaining: {remaining}")
    logger.info(f"Checkpointed results preserved in: {RESULTS_JSONL}")
    logger.info(f"Flat CSV preserved in: {RESULTS_CSV}")
    logger.info(f"Elapsed before interrupt: {timer.elapsed():.1f} s")
    logger.info("Press Run again with RESUME=True, RESET_RESULTS=False to continue.")
    _log_blank(logger)


def run_held_out_spf_benchmark() -> None:
    timer = ElapsedTimer()

    if RESET_RESULTS and RESUME:
        raise ValueError("RESET_RESULTS=True cannot be combined with RESUME=True")

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Held-out manifest not found: {MANIFEST_PATH}")

    primary_manifest_path = _REPO_ROOT / PRIMARY_MANIFEST_PATH
    primary_manifest_digest_before: str | None = None
    if primary_manifest_path.is_file():
        primary_manifest_digest_before = _file_digest(primary_manifest_path)

    held_out_manifest_digest_before = _file_digest(MANIFEST_PATH)

    frozen_paths = [
        _REPO_ROOT / "pathfinding/results/mapf_priority_strategy_execution/results_details.jsonl",
        _REPO_ROOT / HELD_OUT_CONFLICT_AWARE_JSONL,
        _REPO_ROOT / "pathfinding/results/mapf_conflict_aware_priority_execution/results_details.jsonl",
    ]
    frozen_digests_before = {
        path: _file_digest(path) for path in frozen_paths if path.is_file()
    }

    instance_ids = _effective_instance_ids()

    checkpoint = PriorityStrategyCheckpointStore(
        results_dir=RESULTS_DIR,
        csv_path=RESULTS_CSV,
        jsonl_path=RESULTS_JSONL,
        reset_results=RESET_RESULTS,
    )

    append_log = RESUME and LOG_FILE.is_file() and not RESET_RESULTS
    logger = _setup_logging(LOG_FILE, append=append_log)

    manifest = load_benchmark_manifest(MANIFEST_PATH)
    primary_manifest = (
        load_benchmark_manifest(primary_manifest_path)
        if primary_manifest_path.is_file()
        else None
    )
    validate_held_out_execution_manifest(
        manifest,
        primary_manifest=primary_manifest,
        expected_seed=DEFAULT_HELD_OUT_SEED,
        agent_counts=AGENT_COUNTS or (5, 10, 20),
        instances_per_level=3,
    )

    max_timestep = MAX_TIMESTEP if MAX_TIMESTEP is not None else manifest.max_timestep

    plan = build_priority_strategy_run_plan(
        manifest,
        strategies=STRATEGIES,
        agent_counts=AGENT_COUNTS,
        interaction_levels=INTERACTION_LEVELS,
        instance_ids=instance_ids,
    )
    total_runs = len(plan)
    completed_keys = checkpoint.completed_keys() if RESUME else set()

    already_completed = sum(
        1
        for entry in plan
        if PriorityStrategyRunKey(entry.instance.instance_id, entry.strategy)
        in completed_keys
    )
    remaining = total_runs - already_completed

    selected_instance_ids = {entry.instance.instance_id for entry in plan}
    _print_startup_summary(
        logger,
        manifest_path=MANIFEST_PATH,
        catalogue_seed=manifest.seed,
        max_timestep=max_timestep,
        selected_instance_count=len(selected_instance_ids),
        total_runs=total_runs,
        already_completed=already_completed,
        remaining=remaining,
        smoke_mode=SMOKE_MODE,
    )

    grid_map = load_moving_ai_map(MAP_PATH)
    scenarios = load_moving_ai_scenarios(SCEN_PATH)
    if grid_map.name != manifest.map_name:
        raise ValueError(
            f"map name mismatch: manifest expects {manifest.map_name!r}, "
            f"loaded {grid_map.name!r}"
        )

    completed = 0
    session_records: list[PriorityStrategyRunRecord] = list(
        checkpoint.records.values()
    )

    try:
        for run_number, entry in enumerate(plan, start=1):
            key = PriorityStrategyRunKey(
                entry.instance.instance_id,
                entry.strategy,
            )
            if RESUME and key in completed_keys:
                logger.info(
                    f"SKIP — already completed: {entry.instance.instance_id} / "
                    f"{entry.strategy.value}"
                )
                continue

            _print_run_banner(
                logger,
                run_number=run_number,
                total_runs=total_runs,
                entry=entry,
            )
            _log(logger, timer, "Run started")

            try:
                record = execute_priority_strategy_plan_entry(
                    grid_map=grid_map,
                    scenarios=scenarios,
                    entry=entry,
                    max_timestep=max_timestep,
                    catalogue_role=manifest.catalogue_role,
                    catalogue_seed=manifest.seed,
                )
            except Exception as error:
                logger.info("UNEXPECTED ERROR")
                logger.info(traceback.format_exc())
                raise RuntimeError(
                    f"Unexpected error for {entry.instance.instance_id} / "
                    f"{entry.strategy.value}: {error}"
                ) from error

            validate_priority_strategy_run_record(record)
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

            if STOP_ON_ERROR and not record.success:
                logger.info("Stopping because STOP_ON_ERROR=True and run failed.")
                break

    except KeyboardInterrupt:
        _print_interrupt_summary(
            logger,
            timer,
            completed=already_completed + completed,
            total_runs=total_runs,
        )
        raise SystemExit(130)

    if primary_manifest_digest_before is not None:
        if _file_digest(primary_manifest_path) != primary_manifest_digest_before:
            raise RuntimeError("Primary manifest was modified during held-out SPF execution")

    if _file_digest(MANIFEST_PATH) != held_out_manifest_digest_before:
        raise RuntimeError("Held-out manifest was modified during SPF execution")

    for path, digest_before in frozen_digests_before.items():
        if not path.is_file():
            raise RuntimeError(f"Frozen results file disappeared during execution: {path}")
        if _file_digest(path) != digest_before:
            raise RuntimeError(f"Frozen results were modified during execution: {path}")

    if already_completed + completed < total_runs:
        logger.info(
            f"Stopped early: {already_completed + completed} / {total_runs} runs present"
        )
        return

    _print_completion_summary(logger, timer, session_records)


def main() -> None:
    try:
        run_held_out_spf_benchmark()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
