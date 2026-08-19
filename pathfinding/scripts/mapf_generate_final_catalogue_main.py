"""Generate and validate the final MAPF-5A benchmark catalogue.

Open in PyCharm and press Run. No command-line arguments required.
Edit the FINAL MAPF-5A CATALOGUE CONFIGURATION block below to change settings.
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from collections import defaultdict
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathfinding.src.experiments import mapf_benchmark_instances as benchmark_module
from pathfinding.src.experiments import mapf_independent_static_path as static_module
from pathfinding.src.experiments.mapf_benchmark_instances import (
    INTERACTION_HIGH_MIN_CONFLICTS,
    INTERACTION_LOW_MAX_CONFLICTS,
    INTERACTION_MEDIUM_MAX_CONFLICTS,
    INTERACTION_MEDIUM_MIN_CONFLICTS,
    MAPFBenchmarkInstance,
    MAPFBenchmarkManifest,
    MAPFBenchmarkSourcePool,
    MAPFInteractionLevel,
    StaticPrecomputeProgress,
    _candidate_signature,
    evaluate_benchmark_candidate,
    generate_benchmark_instances_from_precomputed,
    generate_benchmark_manifest,
    load_benchmark_manifest,
    prepare_benchmark_source_pool,
    reconstruct_mapf_scenario_from_instance,
    save_benchmark_manifest,
)
from pathfinding.src.core.models import GridMap, Scenario
from pathfinding.src.loaders.map_loader import load_moving_ai_map
from pathfinding.src.loaders.scen_loader import load_moving_ai_scenarios

# ============================================================
# FINAL MAPF-5A CATALOGUE CONFIGURATION
# ============================================================

MAP = _REPO_ROOT / "Data" / "bg512-map" / "AR0204SR.map"
SCEN = _REPO_ROOT / "Data" / "bg512-scen" / "AR0204SR.map.scen"

SEED = 2026
MAX_TIMESTEP = 512
MIN_REFERENCE_LENGTH = 20.0

VALIDATION_AGENT_COUNTS = (10, 20)
VALIDATION_INSTANCES_PER_LEVEL = 1
VALIDATION_MAX_ATTEMPTS = 1000

FINAL_AGENT_COUNTS = (5, 10, 20)
FINAL_INSTANCES_PER_LEVEL = 3
FINAL_MAX_ATTEMPTS = 5000

PRECOMPUTE_PROGRESS_EVERY = 100

OUTPUT_MANIFEST = (
    _REPO_ROOT / "pathfinding" / "results" / "mapf_benchmarks" / "AR0204SR_manifest.json"
)
LOG_FILE = _REPO_ROOT / "pathfinding" / "results" / "mapf_final_catalogue_generation.log"

EXPECTED_ELIGIBLE = 2160
EXPECTED_REACHABLE = 2160
EXPECTED_FEASIBLE = 998
EXPECTED_OVER_HORIZON = 1162
EXPECTED_NO_PATH = 0
EXPECTED_COMPONENTS = 1

# ============================================================
# END CONFIGURATION
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


@dataclass
class ValidationRunResult:
    agent_count: int
    success: bool
    attempts: int
    sampling_elapsed_s: float
    accepted: list[tuple[MAPFBenchmarkInstance, int]]
    histogram: ConflictHistogram


@dataclass
class SamplingSummary:
    agent_count: int
    attempts: int
    sampling_elapsed_s: float
    conflict_counts_by_level: dict[MAPFInteractionLevel, list[int]]


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

    logger = logging.getLogger("mapf_final_catalogue_generation")
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


@contextmanager
def _lookup_only_guards():
    original_find_path = benchmark_module.find_path
    original_static_path = static_module.find_independent_static_path
    original_bounded_static_path = static_module.find_independent_static_path_bounded
    original_reachability_index = static_module.build_static_reachability_index

    def guard_find_path(*_args, **_kwargs):
        raise AssertionError("find_path must not be called during lookup-only sampling")

    def guard_static_path(*_args, **_kwargs):
        raise AssertionError(
            "find_independent_static_path must not be called during lookup-only sampling"
        )

    def guard_bounded_static_path(*_args, **_kwargs):
        raise AssertionError(
            "find_independent_static_path_bounded must not be called during lookup-only sampling"
        )

    def guard_reachability_index(*_args, **_kwargs):
        raise AssertionError(
            "build_static_reachability_index must not be called during lookup-only sampling"
        )

    benchmark_module.find_path = guard_find_path
    static_module.find_independent_static_path = guard_static_path
    static_module.find_independent_static_path_bounded = guard_bounded_static_path
    static_module.build_static_reachability_index = guard_reachability_index
    try:
        yield
    finally:
        benchmark_module.find_path = original_find_path
        static_module.find_independent_static_path = original_static_path
        static_module.find_independent_static_path_bounded = original_bounded_static_path
        static_module.build_static_reachability_index = original_reachability_index


def _format_instance(instance: MAPFBenchmarkInstance, attempt: int | None = None) -> list[str]:
    lines = [
        f"  ID: {instance.instance_id}",
        f"  Interaction: {instance.interaction_level.name}",
        f"  Scenario indices: {instance.scenario_indices}",
        f"  Independent conflicts: {instance.independent_conflict_count}",
        f"  Conflicting agent pairs: {instance.conflicting_agent_pair_count}",
        f"  Independent SoC: {instance.independent_soc}",
        f"  Independent makespan: {instance.independent_makespan}",
    ]
    if attempt is not None:
        lines.append(f"  Attempt: {attempt}")
    return lines


def _log_source_pool_summary(
    logger: logging.Logger,
    source_pool: MAPFBenchmarkSourcePool,
    *,
    precompute_elapsed_s: float,
) -> None:
    precompute_result = source_pool.precompute_result
    logger.info(f"  eligible MovingAI sources: {len(source_pool.eligible_indices)}")
    logger.info(
        f"  sources spatially reachable: {precompute_result.spatially_reachable_count}"
    )
    logger.info(f"  within horizon: {len(source_pool.feasible_indices)}")
    logger.info(f"  over horizon: {precompute_result.over_horizon_count}")
    logger.info(f"  no spatial path: {precompute_result.no_spatial_path_count}")
    logger.info(f"  feasible source pool: {len(source_pool.feasible_indices)}")
    logger.info(
        f"  reachability preprocessing time: "
        f"{precompute_result.reachability_preprocess_s:.3f} s"
    )
    logger.info(
        f"  bounded path preparation time: "
        f"{precompute_result.bounded_preprocess_s:.1f} s"
    )
    logger.info(f"  total source preparation: {precompute_elapsed_s:.1f} s")


def _validate_source_pool_counts(source_pool: MAPFBenchmarkSourcePool) -> None:
    precompute_result = source_pool.precompute_result
    checks = {
        "eligible": (len(source_pool.eligible_indices), EXPECTED_ELIGIBLE),
        "reachable": (precompute_result.spatially_reachable_count, EXPECTED_REACHABLE),
        "feasible": (len(source_pool.feasible_indices), EXPECTED_FEASIBLE),
        "over_horizon": (precompute_result.over_horizon_count, EXPECTED_OVER_HORIZON),
        "no_spatial_path": (precompute_result.no_spatial_path_count, EXPECTED_NO_PATH),
        "components": (
            precompute_result.reachability_component_count,
            EXPECTED_COMPONENTS,
        ),
    }
    mismatches = [
        f"{name}: expected {expected}, got {actual}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if mismatches:
        raise RuntimeError(
            "Source pool counts differ from verified MAPF-5A baseline:\n"
            + "\n".join(f"  - {line}" for line in mismatches)
        )


def _run_validation_generation(
    *,
    grid_map: GridMap,
    scenarios: Sequence[Scenario],
    source_pool: MAPFBenchmarkSourcePool,
    agent_count: int,
    logger: logging.Logger,
) -> ValidationRunResult:
    histogram = ConflictHistogram()
    accepted: list[tuple[MAPFBenchmarkInstance, int]] = []
    sampling_start = time.perf_counter()

    def accepted_callback(instance: MAPFBenchmarkInstance, attempt: int) -> None:
        accepted.append((instance, attempt))
        histogram.record(instance.independent_conflict_count)
        logger.info("ACCEPTED")
        for line in _format_instance(instance, attempt):
            logger.info(line)
        logger.info("")

    generation_error: ValueError | None = None
    result = None
    with _lookup_only_guards():
        try:
            result = generate_benchmark_instances_from_precomputed(
                grid_map=grid_map,
                scenarios=scenarios,
                agent_count=agent_count,
                instances_per_level=VALIDATION_INSTANCES_PER_LEVEL,
                max_timestep=MAX_TIMESTEP,
                seed=SEED,
                max_attempts=VALIDATION_MAX_ATTEMPTS,
                source_pool=source_pool,
                accepted_instance_callback=accepted_callback,
            )
        except ValueError as error:
            generation_error = error

    sampling_elapsed = time.perf_counter() - sampling_start
    success = (
        generation_error is None
        and result is not None
        and len(result.instances) == len(MAPFInteractionLevel) * VALIDATION_INSTANCES_PER_LEVEL
    )
    attempts = result.attempts if result is not None else VALIDATION_MAX_ATTEMPTS

    logger.info(f"Validation agent_count={agent_count} success={success}")
    logger.info(f"  attempts: {attempts}")
    logger.info(f"  sampling elapsed: {sampling_elapsed:.3f} s")
    logger.info("Observed candidate independent-conflict histogram:")
    for line in histogram.bucket_lines():
        logger.info(line)
    logger.info("")

    if generation_error is not None:
        logger.info(f"Underlying error: {generation_error}")

    return ValidationRunResult(
        agent_count=agent_count,
        success=success,
        attempts=attempts,
        sampling_elapsed_s=sampling_elapsed,
        accepted=accepted,
        histogram=histogram,
    )


def _interaction_matches_level(
    instance: MAPFBenchmarkInstance,
) -> bool:
    count = instance.independent_conflict_count
    level = instance.interaction_level
    if level == MAPFInteractionLevel.LOW:
        return count == INTERACTION_LOW_MAX_CONFLICTS
    if level == MAPFInteractionLevel.MEDIUM:
        return (
            INTERACTION_MEDIUM_MIN_CONFLICTS
            <= count
            <= INTERACTION_MEDIUM_MAX_CONFLICTS
        )
    if level == MAPFInteractionLevel.HIGH:
        return count >= INTERACTION_HIGH_MIN_CONFLICTS
    return False


def _validate_manifest_content(
    manifest: MAPFBenchmarkManifest,
    *,
    grid_map: GridMap,
    scenarios: Sequence[Scenario],
    source_pool: MAPFBenchmarkSourcePool,
    agent_counts: Sequence[int],
    instances_per_level: int,
) -> None:
    expected_total = len(agent_counts) * len(MAPFInteractionLevel) * instances_per_level
    if len(manifest.instances) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} instances, found {len(manifest.instances)}"
        )

    instance_ids = [instance.instance_id for instance in manifest.instances]
    if len(set(instance_ids)) != len(instance_ids):
        raise RuntimeError("Duplicate instance IDs found in manifest")

    feasible_set = set(source_pool.feasible_indices)
    signatures_by_agent_count: dict[int, set[tuple[int, ...]]] = {
        agent_count: set() for agent_count in agent_counts
    }
    counts_by_agent_level: dict[tuple[int, MAPFInteractionLevel], list[MAPFBenchmarkInstance]] = {
        (agent_count, level): []
        for agent_count in agent_counts
        for level in MAPFInteractionLevel
    }

    for instance in manifest.instances:
        if instance.agent_count not in agent_counts:
            raise RuntimeError(
                f"Unexpected agent_count in manifest: {instance.agent_count}"
            )
        if len(instance.scenario_indices) != instance.agent_count:
            raise RuntimeError(
                f"Instance {instance.instance_id} has "
                f"{len(instance.scenario_indices)} scenario indices "
                f"for agent_count={instance.agent_count}"
            )
        if len(set(instance.scenario_indices)) != len(instance.scenario_indices):
            raise RuntimeError(
                f"Duplicate scenario indices in instance {instance.instance_id}"
            )
        if not _interaction_matches_level(instance):
            raise RuntimeError(
                f"Instance {instance.instance_id} interaction label "
                f"{instance.interaction_level.name} does not match "
                f"conflict_count={instance.independent_conflict_count}"
            )

        for scenario_index in instance.scenario_indices:
            if scenario_index not in feasible_set:
                raise RuntimeError(
                    f"Instance {instance.instance_id} uses infeasible "
                    f"scenario_index={scenario_index}"
                )

        signature = _candidate_signature(instance.scenario_indices)
        seen = signatures_by_agent_count[instance.agent_count]
        if signature in seen:
            raise RuntimeError(
                f"Duplicate unordered candidate set for agent_count="
                f"{instance.agent_count}: {signature}"
            )
        seen.add(signature)

        counts_by_agent_level[(instance.agent_count, instance.interaction_level)].append(
            instance
        )

        recomputed = evaluate_benchmark_candidate(
            grid_map=grid_map,
            scenarios=scenarios,
            scenario_indices=instance.scenario_indices,
            max_timestep=manifest.max_timestep,
            precomputed_lookup=source_pool.precomputed_lookup,
        )
        if recomputed is None:
            raise RuntimeError(
                f"Failed to recompute metadata for instance {instance.instance_id}"
            )

        if (
            recomputed.independent_conflict_count != instance.independent_conflict_count
            or recomputed.vertex_conflict_count != instance.vertex_conflict_count
            or recomputed.edge_conflict_count != instance.edge_conflict_count
            or recomputed.conflicting_agent_pair_count
            != instance.conflicting_agent_pair_count
            or recomputed.independent_soc != instance.independent_soc
            or recomputed.independent_makespan != instance.independent_makespan
            or recomputed.interaction_level != instance.interaction_level
        ):
            raise RuntimeError(
                f"Metadata mismatch for instance {instance.instance_id}"
            )

        mapf_scenario = reconstruct_mapf_scenario_from_instance(
            scenarios=scenarios,
            grid_map=grid_map,
            instance=instance,
        )
        expected_agent_ids = tuple(range(instance.agent_count))
        actual_agent_ids = tuple(agent.agent_id for agent in mapf_scenario.agents)
        if actual_agent_ids != expected_agent_ids:
            raise RuntimeError(
                f"Agent ID order mismatch for instance {instance.instance_id}: "
                f"{actual_agent_ids} != {expected_agent_ids}"
            )

    for agent_count in agent_counts:
        agent_instances = [
            instance
            for instance in manifest.instances
            if instance.agent_count == agent_count
        ]
        expected_per_agent = len(MAPFInteractionLevel) * instances_per_level
        if len(agent_instances) != expected_per_agent:
            raise RuntimeError(
                f"Expected {expected_per_agent} instances for agent_count={agent_count}, "
                f"found {len(agent_instances)}"
            )

        for level in MAPFInteractionLevel:
            level_instances = counts_by_agent_level[(agent_count, level)]
            if len(level_instances) != instances_per_level:
                raise RuntimeError(
                    f"Expected {instances_per_level} {level.name} instances for "
                    f"agent_count={agent_count}, found {len(level_instances)}"
                )


def _build_sampling_summaries(
    manifest: MAPFBenchmarkManifest,
    attempts_by_agent_count: dict[int, int],
    sampling_elapsed_by_agent_count: dict[int, float],
) -> list[SamplingSummary]:
    summaries: list[SamplingSummary] = []
    for agent_count in FINAL_AGENT_COUNTS:
        conflict_counts_by_level: dict[MAPFInteractionLevel, list[int]] = {
            level: [] for level in MAPFInteractionLevel
        }
        for instance in manifest.instances:
            if instance.agent_count != agent_count:
                continue
            conflict_counts_by_level[instance.interaction_level].append(
                instance.independent_conflict_count
            )
        summaries.append(
            SamplingSummary(
                agent_count=agent_count,
                attempts=attempts_by_agent_count[agent_count],
                sampling_elapsed_s=sampling_elapsed_by_agent_count[agent_count],
                conflict_counts_by_level=conflict_counts_by_level,
            )
        )
    return summaries


def _print_final_summary(
    logger: logging.Logger,
    manifest: MAPFBenchmarkManifest,
    sampling_summaries: list[SamplingSummary],
    source_pool: MAPFBenchmarkSourcePool,
    *,
    output_path: Path,
) -> None:
    logger.info("=" * 72)
    logger.info("FINAL MAPF-5A CATALOGUE SUMMARY")
    logger.info("=" * 72)
    logger.info(f"Manifest saved to: {output_path}")
    logger.info(f"Total instances: {len(manifest.instances)}")
    logger.info("")
    logger.info("Agent count | Level  | Instances | Attempts needed | Conflict counts")
    for summary in sampling_summaries:
        for level in MAPFInteractionLevel:
            counts = summary.conflict_counts_by_level[level]
            logger.info(
                f"{summary.agent_count:11d} | {level.name:6s} | "
                f"{len(counts):9d} | {summary.attempts:15d} | {counts}"
            )
    logger.info("")
    total_sampling = sum(summary.sampling_elapsed_s for summary in sampling_summaries)
    logger.info("Sampling runtime summary:")
    for summary in sampling_summaries:
        logger.info(
            f"  n={summary.agent_count}: attempts={summary.attempts}, "
            f"sampling elapsed={summary.sampling_elapsed_s:.3f} s"
        )
    logger.info(f"  total catalogue sampling time: {total_sampling:.3f} s")
    logger.info("")
    logger.info("Final instance list:")
    for instance in manifest.instances:
        logger.info(f"- {instance.instance_id}")
        logger.info(f"    agent_count: {instance.agent_count}")
        logger.info(f"    interaction_level: {instance.interaction_level.value}")
        logger.info(f"    scenario_indices: {instance.scenario_indices}")
        logger.info(
            f"    independent_conflict_count: {instance.independent_conflict_count}"
        )
        logger.info(
            f"    conflicting_agent_pair_count: "
            f"{instance.conflicting_agent_pair_count}"
        )
        logger.info(f"    independent_soc: {instance.independent_soc}")
        logger.info(f"    independent_makespan: {instance.independent_makespan}")
    logger.info("")
    logger.info("Source preparation summary:")
    _log_source_pool_summary(
        logger,
        source_pool,
        precompute_elapsed_s=source_pool.precompute_result.reachability_preprocess_s
        + source_pool.precompute_result.bounded_preprocess_s,
    )


def run_final_catalogue_generation() -> None:
    log_file = LOG_FILE.resolve()
    logger = _setup_logging(log_file)
    timer = ElapsedTimer()

    _log(logger, timer, "Starting MAPF-5A final catalogue generation")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Output manifest: {OUTPUT_MANIFEST.resolve()}")
    logger.info("")

    # STEP 1 — Load map/scenarios
    _log(logger, timer, "STEP 1 — Load map/scenarios")
    grid_map = load_moving_ai_map(MAP)
    scenarios = load_moving_ai_scenarios(SCEN)
    logger.info(f"Map loaded: {grid_map.name}")
    logger.info(f"Scenarios loaded: {len(scenarios)}")
    logger.info("")

    # STEP 2 — Build source pool once
    _log(logger, timer, "STEP 2 — Build source pool")
    precompute_start = time.perf_counter()

    def precompute_progress(progress: StaticPrecomputeProgress) -> None:
        elapsed = time.perf_counter() - precompute_start
        logger.info(
            f"BOUNDED STATIC PRECOMPUTE {progress.completed} / {progress.total}"
        )
        logger.info(f"  elapsed: {elapsed:.1f} s")
        logger.info(f"  feasible: {progress.feasible}")
        logger.info(f"  no spatial path: {progress.no_spatial_path}")
        logger.info(f"  over horizon: {progress.over_horizon}")

    original_find_path = benchmark_module.find_path

    def guard_sta_find_path(*_args, **_kwargs):
        raise AssertionError(
            "Space-Time A* must not be used for benchmark source precompute"
        )

    benchmark_module.find_path = guard_sta_find_path
    try:
        source_pool = prepare_benchmark_source_pool(
            grid_map=grid_map,
            scenarios=scenarios,
            min_reference_length=MIN_REFERENCE_LENGTH,
            max_timestep=MAX_TIMESTEP,
            precompute_progress_every=PRECOMPUTE_PROGRESS_EVERY,
            precompute_progress_callback=precompute_progress,
        )
    finally:
        benchmark_module.find_path = original_find_path

    precompute_elapsed = time.perf_counter() - precompute_start
    precompute_result = source_pool.precompute_result
    logger.info("")
    logger.info("REACHABILITY PREPROCESSING COMPLETE")
    logger.info(f"  components: {precompute_result.reachability_component_count}")
    logger.info(f"  walkable cells: {precompute_result.reachability_walkable_cells}")
    logger.info(f"  elapsed: {precompute_result.reachability_preprocess_s:.3f} s")
    logger.info("")
    logger.info("BOUNDED STATIC PRECOMPUTE COMPLETE")
    _log_source_pool_summary(logger, source_pool, precompute_elapsed_s=precompute_elapsed)
    logger.info("")

    _validate_source_pool_counts(source_pool)
    logger.info("Source pool counts match verified MAPF-5A baseline.")
    logger.info("")

    validation_results: list[ValidationRunResult] = []

    # STEP 3 — Validate 10-agent generation
    _log(logger, timer, "STEP 3 — Validate 10-agent generation")
    validation_results.append(
        _run_validation_generation(
            grid_map=grid_map,
            scenarios=scenarios,
            source_pool=source_pool,
            agent_count=10,
            logger=logger,
        )
    )
    if not validation_results[-1].success:
        logger.error("10-agent validation failed. Stopping before final catalogue generation.")
        raise SystemExit(1)

    # STEP 4 — Validate 20-agent generation
    _log(logger, timer, "STEP 4 — Validate 20-agent generation")
    validation_results.append(
        _run_validation_generation(
            grid_map=grid_map,
            scenarios=scenarios,
            source_pool=source_pool,
            agent_count=20,
            logger=logger,
        )
    )
    if not validation_results[-1].success:
        logger.error("20-agent validation failed. Stopping before final catalogue generation.")
        raise SystemExit(1)

    # STEP 5 — Generate final 27-instance catalogue
    _log(logger, timer, "STEP 5 — Generate final 27-instance catalogue")
    sampling_elapsed_by_agent_count: dict[int, float] = {}
    generation_error: ValueError | None = None
    generation_result = None
    sampling_starts: dict[int, float] = {}

    def generation_start_tracked(agent_count: int) -> None:
        sampling_starts[agent_count] = time.perf_counter()

    def accepted_callback(instance: MAPFBenchmarkInstance, attempt: int) -> None:
        logger.info("FINAL ACCEPTED")
        for line in _format_instance(instance, attempt):
            logger.info(line)
        logger.info("")

    with _lookup_only_guards():
        try:
            generation_result = generate_benchmark_manifest(
                grid_map=grid_map,
                scenarios=scenarios,
                scenario_name=SCEN.name,
                agent_counts=FINAL_AGENT_COUNTS,
                instances_per_level=FINAL_INSTANCES_PER_LEVEL,
                max_timestep=MAX_TIMESTEP,
                seed=SEED,
                min_reference_length=MIN_REFERENCE_LENGTH,
                max_attempts=FINAL_MAX_ATTEMPTS,
                source_pool=source_pool,
                accepted_instance_callback=accepted_callback,
                generation_start_callback=generation_start_tracked,
            )
            for agent_count in FINAL_AGENT_COUNTS:
                sampling_elapsed_by_agent_count[agent_count] = (
                    time.perf_counter() - sampling_starts[agent_count]
                )
        except ValueError as error:
            generation_error = error

    if generation_error is not None or generation_result is None:
        logger.error("")
        logger.error("FINAL CATALOGUE GENERATION INCOMPLETE")
        logger.error(f"Underlying error: {generation_error}")
        raise SystemExit(1)

    manifest = generation_result.manifest
    attempts_by_agent_count = generation_result.attempts_by_agent_count

    expected_total = len(FINAL_AGENT_COUNTS) * len(MAPFInteractionLevel) * FINAL_INSTANCES_PER_LEVEL
    if len(manifest.instances) != expected_total:
        logger.error("")
        logger.error("FINAL CATALOGUE GENERATION INCOMPLETE")
        logger.error(
            f"Expected {expected_total} instances, found {len(manifest.instances)}"
        )
        raise SystemExit(1)

    # STEP 6 — Validate manifest
    _log(logger, timer, "STEP 6 — Validate manifest")
    _validate_manifest_content(
        manifest,
        grid_map=grid_map,
        scenarios=scenarios,
        source_pool=source_pool,
        agent_counts=FINAL_AGENT_COUNTS,
        instances_per_level=FINAL_INSTANCES_PER_LEVEL,
    )
    logger.info("Manifest content validation passed.")
    logger.info("")

    # STEP 7 — Determinism check (same source pool, no re-precompute)
    _log(logger, timer, "STEP 7 — Determinism check")
    with _lookup_only_guards():
        repeat_result = generate_benchmark_manifest(
            grid_map=grid_map,
            scenarios=scenarios,
            scenario_name=SCEN.name,
            agent_counts=FINAL_AGENT_COUNTS,
            instances_per_level=FINAL_INSTANCES_PER_LEVEL,
            max_timestep=MAX_TIMESTEP,
            seed=SEED,
            min_reference_length=MIN_REFERENCE_LENGTH,
            max_attempts=FINAL_MAX_ATTEMPTS,
            source_pool=source_pool,
        )
    if repeat_result.manifest.instances != manifest.instances:
        raise RuntimeError("Deterministic regeneration mismatch")
    logger.info("Determinism check passed.")
    logger.info("")

    # STEP 8 — Save JSON
    _log(logger, timer, "STEP 8 — Save JSON")
    output_path = OUTPUT_MANIFEST.resolve()
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    save_benchmark_manifest(manifest, tmp_path)
    logger.info(f"Temporary manifest written to: {tmp_path}")
    logger.info("")

    # STEP 9 — Reload JSON and verify roundtrip
    _log(logger, timer, "STEP 9 — Reload JSON and verify roundtrip")
    loaded_manifest = load_benchmark_manifest(tmp_path)
    if loaded_manifest != manifest:
        raise RuntimeError("JSON roundtrip mismatch")
    logger.info("JSON roundtrip validation passed.")
    logger.info("")

    tmp_path.replace(output_path)
    logger.info(f"Final manifest atomically saved to: {output_path}")
    logger.info("")

    # STEP 10 — Final MAPF-5A summary
    _log(logger, timer, "STEP 10 — Final MAPF-5A summary")
    sampling_summaries = _build_sampling_summaries(
        manifest,
        attempts_by_agent_count,
        sampling_elapsed_by_agent_count,
    )
    _print_final_summary(
        logger,
        manifest,
        sampling_summaries,
        source_pool,
        output_path=output_path,
    )
    logger.info("")
    logger.info("MAPF-5A CLOSED")


def main() -> None:
    log_file = LOG_FILE.resolve()
    logger = _setup_logging(log_file)
    timer = ElapsedTimer()

    try:
        run_final_catalogue_generation()
    except SystemExit:
        raise
    except Exception:
        logger.exception(f"{timer.stamp()} Unexpected error during final catalogue generation:")
        logger.error(traceback.format_exc())
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
