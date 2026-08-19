"""Local diagnostic runner for MAPF-5A.3 benchmark generation.

Open in PyCharm and press Run. No command-line arguments required.
Edit the DEBUG CONFIGURATION block below to change experiment settings.
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathfinding.src.experiments import mapf_benchmark_instances as benchmark_module
from pathfinding.src.experiments.mapf_benchmark_instances import (
    MAPFBenchmarkInstance,
    MAPFBenchmarkSourcePool,
    MAPFInteractionLevel,
    build_precomputed_path_lookup,
    eligible_scenario_indices,
    generate_benchmark_instances_from_precomputed,
    precompute_independent_paths,
)
from pathfinding.src.loaders.map_loader import load_moving_ai_map
from pathfinding.src.loaders.scen_loader import load_moving_ai_scenarios

# ============================================================
# DEBUG CONFIGURATION
# ============================================================

MAP = _REPO_ROOT / "Data" / "bg512-map" / "AR0204SR.map"
SCEN = _REPO_ROOT / "Data" / "bg512-scen" / "AR0204SR.map.scen"

AGENT_COUNT = 5
INSTANCES_PER_LEVEL = 1
SEED = 2026
MAX_TIMESTEP = 512
MIN_REFERENCE_LENGTH = 20.0
MAX_ATTEMPTS = 1000

PRECOMPUTE_PROGRESS_EVERY = 100
SAMPLING_PROGRESS_EVERY = 50
SLOW_CANDIDATE_THRESHOLD_S = 1.0

LOG_FILE = _REPO_ROOT / "pathfinding" / "results" / "mapf_generation_debug.log"

# ============================================================
# END DEBUG CONFIGURATION
# ============================================================


@dataclass
class ConflictHistogram:
    counts: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def record(self, conflict_count: int) -> None:
        self.counts[conflict_count] += 1

    def bucket_lines(self) -> list[str]:
        buckets = {
            "0": 0,
            "1": 0,
            "2": 0,
            "3-5": 0,
            "6-10": 0,
            "11+": 0,
        }
        for count, frequency in self.counts.items():
            if count == 0:
                buckets["0"] += frequency
            elif count == 1:
                buckets["1"] += frequency
            elif count == 2:
                buckets["2"] += frequency
            elif 3 <= count <= 5:
                buckets["3-5"] += frequency
            elif 6 <= count <= 10:
                buckets["6-10"] += frequency
            else:
                buckets["11+"] += frequency
        return [f"  {label}: {value}" for label, value in buckets.items()]


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

    logger = logging.getLogger("mapf_benchmark_generation_debug")
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


def _format_level_counts(
    found: dict[MAPFInteractionLevel, int],
    target: int,
) -> list[str]:
    return [
        f"  {level.name:6s}: {found[level]:3d} / {target}"
        for level in MAPFInteractionLevel
    ]


def _print_histogram(logger: logging.Logger, histogram: ConflictHistogram) -> None:
    logger.info("Observed candidate independent-conflict histogram:")
    for line in histogram.bucket_lines():
        logger.info(line)


def run_debug_experiment() -> None:
    log_file = LOG_FILE.resolve()
    logger = _setup_logging(log_file)
    timer = ElapsedTimer()

    _log(logger, timer, "Starting MAPF benchmark debug run")
    logger.info(f"Log file: {log_file}")
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  MAP: {MAP}")
    logger.info(f"  SCEN: {SCEN}")
    logger.info(f"  AGENT_COUNT: {AGENT_COUNT}")
    logger.info(f"  INSTANCES_PER_LEVEL: {INSTANCES_PER_LEVEL}")
    logger.info(f"  SEED: {SEED}")
    logger.info(f"  MAX_TIMESTEP: {MAX_TIMESTEP}")
    logger.info(f"  MIN_REFERENCE_LENGTH: {MIN_REFERENCE_LENGTH:g}")
    logger.info(f"  MAX_ATTEMPTS: {MAX_ATTEMPTS}")
    logger.info(f"  PRECOMPUTE_PROGRESS_EVERY: {PRECOMPUTE_PROGRESS_EVERY}")
    logger.info(f"  SAMPLING_PROGRESS_EVERY: {SAMPLING_PROGRESS_EVERY}")
    logger.info("")

    histogram = ConflictHistogram()
    precompute_feasible = 0
    precompute_failed = 0
    precompute_start = time.perf_counter()
    current_attempt = 0
    sampling_start = 0.0
    accepted_instances: list[MAPFBenchmarkInstance] = []

    original_find_path = benchmark_module.find_path
    original_evaluate = benchmark_module._evaluate_candidate

    def counting_find_path(*args, **kwargs):
        nonlocal precompute_feasible, precompute_failed
        result = original_find_path(*args, **kwargs)
        if result is None:
            precompute_failed += 1
        else:
            precompute_feasible += 1
        return result

    def tracking_evaluate(*args, **kwargs):
        eval_start = time.perf_counter()
        result = original_evaluate(*args, **kwargs)
        eval_elapsed = time.perf_counter() - eval_start
        if result is not None:
            histogram.record(result.independent_conflict_count)
            if eval_elapsed > SLOW_CANDIDATE_THRESHOLD_S:
                logger.info("")
                logger.info("SLOW CANDIDATE")
                logger.info(f"  attempt: {current_attempt}")
                logger.info(f"  evaluation time: {eval_elapsed:.3f} s")
                logger.info("")
        return result

    # STEP 1 — Load map
    _log(logger, timer, "STEP 1 — Load map")
    grid_map = load_moving_ai_map(MAP)
    _log(logger, timer, f"Map loaded: {grid_map.name}")

    # STEP 2 — Load scenarios
    _log(logger, timer, "STEP 2 — Load scenarios")
    scenarios = load_moving_ai_scenarios(SCEN)
    _log(logger, timer, f"Scenarios loaded: {len(scenarios)}")

    # STEP 3 — Determine eligible source indices
    _log(logger, timer, "STEP 3 — Determine eligible source indices")
    eligible_indices = eligible_scenario_indices(
        scenarios=scenarios,
        grid_map=grid_map,
        min_reference_length=MIN_REFERENCE_LENGTH,
    )
    logger.info(f"Total MovingAI scenarios: {len(scenarios)}")
    logger.info(f"Min reference length: {MIN_REFERENCE_LENGTH:g}")
    logger.info(f"Eligible source scenarios: {len(eligible_indices)}")
    logger.info(f"Max timestep: {MAX_TIMESTEP}")
    _log(logger, timer, "Eligible pool determined")

    # STEP 4 — Precompute independent source paths
    _log(logger, timer, "STEP 4 — Precompute independent source paths")
    precompute_start = time.perf_counter()
    precompute_feasible = 0
    precompute_failed = 0

    def precompute_progress(completed: int, total: int) -> None:
        elapsed = time.perf_counter() - precompute_start
        logger.info(f"PRECOMPUTE {completed} / {total}")
        logger.info(f"  elapsed: {elapsed:.1f} s")
        logger.info(f"  feasible: {precompute_feasible}")
        logger.info(f"  failed: {precompute_failed}")

    benchmark_module.find_path = counting_find_path
    try:
        precompute_result = precompute_independent_paths(
            grid_map=grid_map,
            scenarios=scenarios,
            scenario_indices=eligible_indices,
            max_timestep=MAX_TIMESTEP,
            progress_every=PRECOMPUTE_PROGRESS_EVERY,
            progress_callback=precompute_progress,
        )
    finally:
        benchmark_module.find_path = original_find_path

    precompute_elapsed = time.perf_counter() - precompute_start
    logger.info("")
    logger.info("PRECOMPUTE COMPLETE")
    logger.info(f"  eligible: {len(eligible_indices)}")
    logger.info(f"  feasible: {len(precompute_result.paths)}")
    logger.info(f"  failed: {len(precompute_result.failed_scenario_indices)}")
    logger.info(f"  elapsed: {precompute_elapsed:.1f} s")
    if eligible_indices:
        logger.info(
            f"  paths/sec: {len(eligible_indices) / precompute_elapsed:.1f}"
        )
    _log(logger, timer, "Precompute finished")

    # STEP 5 — Build lookup
    _log(logger, timer, "STEP 5 — Build lookup")
    precomputed_lookup = build_precomputed_path_lookup(precompute_result.paths)
    feasible_indices = tuple(
        index for index in eligible_indices if index in precomputed_lookup
    )
    source_pool = MAPFBenchmarkSourcePool(
        eligible_indices=eligible_indices,
        precompute_result=precompute_result,
        precomputed_lookup=precomputed_lookup,
        feasible_indices=feasible_indices,
    )
    _log(
        logger,
        timer,
        f"Lookup built for {len(precomputed_lookup)} feasible scenarios",
    )

    # STEP 6 — Generate benchmark candidates using lookup only
    _log(logger, timer, "STEP 6 — Generate benchmark candidates using lookup only")
    logger.info("Starting lookup-only candidate sampling.")
    logger.info("No Space-Time A* calls should occur during this phase.")
    logger.info("")

    sampling_start = time.perf_counter()

    def sampling_progress(
        attempts: int,
        max_attempts: int,
        needed: dict[MAPFInteractionLevel, int],
        found: dict[MAPFInteractionLevel, int],
    ) -> None:
        nonlocal current_attempt
        current_attempt = attempts
        sampling_elapsed = time.perf_counter() - sampling_start
        logger.info("=" * 60)
        logger.info("SAMPLING PROGRESS")
        logger.info("=" * 60)
        logger.info(f"Attempt: {attempts} / {max_attempts}")
        logger.info(f"Elapsed sampling time: {sampling_elapsed:.3f} s")
        logger.info("")
        logger.info("Accepted:")
        for line in _format_level_counts(found, INSTANCES_PER_LEVEL):
            logger.info(line)
        logger.info("")
        _print_histogram(logger, histogram)
        logger.info("")

    def accepted_instance(instance: MAPFBenchmarkInstance) -> None:
        accepted_instances.append(instance)
        sampling_elapsed = time.perf_counter() - sampling_start
        logger.info("ACCEPTED")
        logger.info(f"  ID: {instance.instance_id}")
        logger.info(f"  Interaction: {instance.interaction_level.name}")
        logger.info(f"  Scenario indices: {instance.scenario_indices}")
        logger.info(f"  Independent conflicts: {instance.independent_conflict_count}")
        logger.info(
            f"  Conflicting agent pairs: {instance.conflicting_agent_pair_count}"
        )
        logger.info(f"  Independent SoC: {instance.independent_soc}")
        logger.info(f"  Independent makespan: {instance.independent_makespan}")
        logger.info(f"  Attempt: {current_attempt}")
        logger.info(f"  Sampling elapsed: {sampling_elapsed:.3f} s")
        logger.info("")

    def guard_find_path(*_args, **_kwargs):
        raise AssertionError("find_path must not be called during lookup-only sampling")

    benchmark_module.find_path = guard_find_path
    benchmark_module._evaluate_candidate = tracking_evaluate
    generation_error: ValueError | None = None
    result = None
    try:
        result = generate_benchmark_instances_from_precomputed(
            grid_map=grid_map,
            scenarios=scenarios,
            agent_count=AGENT_COUNT,
            instances_per_level=INSTANCES_PER_LEVEL,
            max_timestep=MAX_TIMESTEP,
            seed=SEED,
            max_attempts=MAX_ATTEMPTS,
            source_pool=source_pool,
            sampling_progress_every=SAMPLING_PROGRESS_EVERY,
            sampling_progress_callback=sampling_progress,
            accepted_instance_callback=accepted_instance,
        )
    except ValueError as error:
        generation_error = error
    finally:
        benchmark_module.find_path = original_find_path
        benchmark_module._evaluate_candidate = original_evaluate

    sampling_elapsed = time.perf_counter() - sampling_start
    total_elapsed = timer.elapsed()

    # STEP 7 — Print final summary
    _log(logger, timer, "STEP 7 — Print final summary")
    logger.info("")

    if generation_error is not None:
        found = {level: 0 for level in MAPFInteractionLevel}
        for instance in accepted_instances:
            found[instance.interaction_level] += 1

        logger.info("GENERATION INCOMPLETE")
        logger.info("")
        logger.info(f"Attempts: {current_attempt or MAX_ATTEMPTS}")
        logger.info("")
        logger.info("Found:")
        for line in _format_level_counts(found, INSTANCES_PER_LEVEL):
            logger.info(line)
        logger.info("")
        _print_histogram(logger, histogram)
        logger.info("")
        logger.info("Interpretation:")
        missing_levels = [
            level.name
            for level in MAPFInteractionLevel
            if found[level] < INSTANCES_PER_LEVEL
        ]
        if missing_levels:
            logger.info(
                f"{', '.join(missing_levels)} was not obtained within the "
                "configured attempt budget."
            )
        logger.info("")
        logger.info(f"Underlying error: {generation_error}")
        logger.info(f"Sampling time: {sampling_elapsed:.2f} s")
        logger.info(f"Total time including precompute: {total_elapsed:.2f} s")
        return

    assert result is not None
    logger.info("GENERATION SUCCESS")
    logger.info("")
    logger.info(f"Agents: {AGENT_COUNT}")
    logger.info(f"Attempts: {result.attempts}")
    logger.info(f"Sampling time: {sampling_elapsed:.2f} s")
    logger.info(f"Total time including precompute: {total_elapsed:.2f} s")
    logger.info("")

    by_level: dict[MAPFInteractionLevel, list[MAPFBenchmarkInstance]] = {
        level: [] for level in MAPFInteractionLevel
    }
    for instance in result.instances:
        by_level[instance.interaction_level].append(instance)

    for level in MAPFInteractionLevel:
        logger.info(f"{level.name}:")
        for instance in by_level[level]:
            logger.info(f"  instance id: {instance.instance_id}")
            logger.info(f"  conflicts: {instance.independent_conflict_count}")
            logger.info(f"  SoC: {instance.independent_soc}")
            logger.info(f"  makespan: {instance.independent_makespan}")
        logger.info("")

    _print_histogram(logger, histogram)


def main() -> None:
    log_file = LOG_FILE.resolve()
    logger = _setup_logging(log_file)
    timer = ElapsedTimer()

    try:
        run_debug_experiment()
    except Exception:
        logger.exception(f"{timer.stamp()} Unexpected error during debug run:")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
