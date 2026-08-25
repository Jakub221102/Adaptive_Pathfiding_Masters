"""Map-agnostic deterministic MAPF benchmark catalogue generation.

Extracted from the AR0204SR production scripts so cross-map catalogues can reuse
the same methodology without duplicating generation logic or hardcoded pool counts.
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

from pathfinding.src.core.models import GridMap, Scenario
from pathfinding.src.experiments import mapf_benchmark_instances as benchmark_module
from pathfinding.src.experiments import mapf_independent_static_path as static_module
from pathfinding.src.experiments.mapf_benchmark_instances import (
    INTERACTION_HIGH_MIN_CONFLICTS,
    INTERACTION_LOW_MAX_CONFLICTS,
    INTERACTION_MEDIUM_MAX_CONFLICTS,
    INTERACTION_MEDIUM_MIN_CONFLICTS,
    MAPFBenchmarkInstance,
    MAPFBenchmarkManifest,
    MAPFBenchmarkManifestGenerationResult,
    MAPFBenchmarkSourcePool,
    MAPFInteractionLevel,
    StaticPrecomputeProgress,
    _candidate_signature,
    evaluate_benchmark_candidate,
    generate_benchmark_instances_from_precomputed,
    generate_benchmark_manifest,
    load_benchmark_manifest,
    save_benchmark_manifest,
)
from pathfinding.src.loaders.map_loader import load_moving_ai_map
from pathfinding.src.loaders.scen_loader import load_moving_ai_scenarios

MAPF8_CROSS_MAP_TARGETS: frozenset[str] = frozenset({"AR0400SR", "AR0307SR"})

MAPF8_CROSS_MAP_SEEDS: dict[str, int] = {
    "AR0400SR": 2028,
    "AR0307SR": 2029,
}

PROTECTED_PRODUCTION_MANIFEST_NAMES: frozenset[str] = frozenset(
    {
        "AR0204SR_manifest.json",
        "AR0204SR_heldout_manifest.json",
        "AR0400SR_manifest.json",
        "AR0307SR_manifest.json",
    }
)

DEFAULT_MAX_TIMESTEP = 512
DEFAULT_MIN_REFERENCE_LENGTH = 20.0
DEFAULT_MAX_ATTEMPTS = 5000
DEFAULT_AGENT_COUNTS: tuple[int, ...] = (5, 10, 20)
DEFAULT_INSTANCES_PER_LEVEL = 3
DEFAULT_PRECOMPUTE_PROGRESS_EVERY = 100
DEFAULT_VALIDATION_AGENT_COUNTS: tuple[int, ...] = (10, 20)
DEFAULT_VALIDATION_INSTANCES_PER_LEVEL = 1
DEFAULT_VALIDATION_MAX_ATTEMPTS = 1000


@dataclass(frozen=True, slots=True)
class MAPFBenchmarkCatalogueConfig:
    """Configuration for one deterministic primary catalogue generation run."""

    map_stem: str
    map_path: Path
    scen_path: Path
    output_manifest_path: Path
    log_file_path: Path
    seed: int
    max_timestep: int = DEFAULT_MAX_TIMESTEP
    min_reference_length: float = DEFAULT_MIN_REFERENCE_LENGTH
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    agent_counts: tuple[int, ...] = DEFAULT_AGENT_COUNTS
    instances_per_level: int = DEFAULT_INSTANCES_PER_LEVEL
    precompute_progress_every: int = DEFAULT_PRECOMPUTE_PROGRESS_EVERY
    validation_agent_counts: tuple[int, ...] = DEFAULT_VALIDATION_AGENT_COUNTS
    validation_instances_per_level: int = DEFAULT_VALIDATION_INSTANCES_PER_LEVEL
    validation_max_attempts: int = DEFAULT_VALIDATION_MAX_ATTEMPTS
    skip_validation: bool = False
    allow_production_manifest_write: bool = False
    expected_source_pool_counts: dict[str, int] | None = None


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


@dataclass(frozen=True, slots=True)
class MAPFBenchmarkCatalogueGenerationResult:
    manifest: MAPFBenchmarkManifest
    source_pool: MAPFBenchmarkSourcePool
    attempts_by_agent_count: dict[int, int]
    manifest_written: bool
    output_manifest_path: Path


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


def resolve_mapf8_cross_map_target(target_map: str) -> str:
    normalized = target_map.strip()
    if normalized == "AR0404SR":
        raise ValueError(
            "AR0404SR is not a valid MAPF-8 cross-map target. Use AR0400SR."
        )
    if normalized not in MAPF8_CROSS_MAP_TARGETS:
        raise ValueError(
            f"Unsupported MAPF-8 cross-map target: {target_map!r}. "
            f"Expected one of: {sorted(MAPF8_CROSS_MAP_TARGETS)}"
        )
    return normalized


def mapf8_cross_map_catalogue_config(
    target_map: str,
    repo_root: Path,
    *,
    allow_production_manifest_write: bool = False,
) -> MAPFBenchmarkCatalogueConfig:
    """Build frozen MAPF-8 cross-map catalogue configuration for one target map."""
    map_stem = resolve_mapf8_cross_map_target(target_map)
    seed = MAPF8_CROSS_MAP_SEEDS[map_stem]
    benchmarks_dir = repo_root / "pathfinding" / "results" / "mapf_benchmarks"
    return MAPFBenchmarkCatalogueConfig(
        map_stem=map_stem,
        map_path=repo_root / "Data" / "bg512-map" / f"{map_stem}.map",
        scen_path=repo_root / "Data" / "bg512-scen" / f"{map_stem}.map.scen",
        output_manifest_path=benchmarks_dir / f"{map_stem}_manifest.json",
        log_file_path=repo_root / "pathfinding" / "results" / f"mapf_cross_map_catalogue_{map_stem}.log",
        seed=seed,
        allow_production_manifest_write=allow_production_manifest_write,
    )


def assert_safe_manifest_output(
    output_manifest_path: Path,
    *,
    allow_production_manifest_write: bool,
) -> None:
    manifest_name = output_manifest_path.name
    if manifest_name in PROTECTED_PRODUCTION_MANIFEST_NAMES:
        if not allow_production_manifest_write:
            raise RuntimeError(
                f"Refusing to write protected production manifest {output_manifest_path}. "
                "Set allow_production_manifest_write=True explicitly for MAPF-8.2 production runs."
            )
        if output_manifest_path.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing protected manifest {output_manifest_path} "
                "without explicit confirmation. Delete or rename the existing file first."
            )


def setup_catalogue_logging(log_file: Path, logger_name: str) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
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


def assert_production_generation_ready(config: MAPFBenchmarkCatalogueConfig) -> None:
    """Fail fast before expensive work when production output is unsafe."""
    if config.map_stem == "AR0404SR":
        raise ValueError(
            "AR0404SR is not a valid MAPF-8 cross-map target. Use AR0400SR."
        )
    if config.allow_production_manifest_write:
        if not config.map_path.is_file():
            raise FileNotFoundError(f"Map file not found: {config.map_path}")
        if not config.scen_path.is_file():
            raise FileNotFoundError(f"Scenario file not found: {config.scen_path}")
        assert_safe_manifest_output(
            config.output_manifest_path.resolve(),
            allow_production_manifest_write=True,
        )


def log_catalogue_config_summary(
    logger: logging.Logger,
    config: MAPFBenchmarkCatalogueConfig,
) -> None:
    logger.info("=" * 72)
    logger.info("MAPF-8.2 PRODUCTION CROSS-MAP CATALOGUE GENERATION")
    logger.info("=" * 72)
    logger.info(f"  target map: {config.map_stem}")
    if config.map_stem == "AR0400SR":
        logger.info("  note: AR0400SR is the frozen MAPF-8 open-arena map (NOT AR0404SR)")
    logger.info(f"  map path: {config.map_path.resolve()}")
    logger.info(f"  scenario path: {config.scen_path.resolve()}")
    logger.info(f"  output manifest: {config.output_manifest_path.resolve()}")
    logger.info(f"  log file: {config.log_file_path.resolve()}")
    logger.info(f"  seed: {config.seed}")
    logger.info(f"  max_timestep: {config.max_timestep}")
    logger.info(f"  min_reference_length: {config.min_reference_length}")
    logger.info(f"  max_attempts: {config.max_attempts}")
    logger.info(f"  agent_counts: {config.agent_counts}")
    logger.info(f"  instances_per_level: {config.instances_per_level}")
    logger.info(
        "  interaction targets: LOW=0, MEDIUM=1..2, HIGH>=3 independent-path conflicts"
    )
    logger.info(
        f"  expected catalogue size: "
        f"{len(config.agent_counts) * 3 * config.instances_per_level} instances"
    )
    logger.info(
        f"  allow_production_manifest_write: {config.allow_production_manifest_write}"
    )
    logger.info("")


@contextmanager
def lookup_only_guards():
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


def format_instance_lines(
    instance: MAPFBenchmarkInstance,
    attempt: int | None = None,
) -> list[str]:
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


def log_source_pool_summary(
    logger: logging.Logger,
    source_pool: MAPFBenchmarkSourcePool,
    scenarios: Sequence[Scenario],
    *,
    precompute_elapsed_s: float,
) -> None:
    precompute_result = source_pool.precompute_result
    logger.info(f"  source scenario count: {len(scenarios)}")
    logger.info(f"  eligible MovingAI sources: {len(source_pool.eligible_indices)}")
    logger.info(
        f"  sources spatially reachable: {precompute_result.spatially_reachable_count}"
    )
    logger.info(f"  within horizon (feasible pool): {len(source_pool.feasible_indices)}")
    logger.info(f"  over horizon: {precompute_result.over_horizon_count}")
    logger.info(f"  no spatial path: {precompute_result.no_spatial_path_count}")
    logger.info(
        f"  reachability components: {precompute_result.reachability_component_count}"
    )
    logger.info(
        f"  reachability preprocessing time: "
        f"{precompute_result.reachability_preprocess_s:.3f} s"
    )
    logger.info(
        f"  bounded path preparation time: "
        f"{precompute_result.bounded_preprocess_s:.1f} s"
    )
    logger.info(f"  total source preparation: {precompute_elapsed_s:.1f} s")


def validate_optional_source_pool_counts(
    source_pool: MAPFBenchmarkSourcePool,
    expected_counts: dict[str, int],
) -> None:
    precompute_result = source_pool.precompute_result
    actual = {
        "eligible": len(source_pool.eligible_indices),
        "reachable": precompute_result.spatially_reachable_count,
        "feasible": len(source_pool.feasible_indices),
        "over_horizon": precompute_result.over_horizon_count,
        "no_spatial_path": precompute_result.no_spatial_path_count,
        "components": precompute_result.reachability_component_count,
    }
    mismatches = [
        f"{name}: expected {expected_counts[name]}, got {actual[name]}"
        for name in expected_counts
        if actual[name] != expected_counts[name]
    ]
    if mismatches:
        raise RuntimeError(
            "Source pool counts differ from expected baseline:\n"
            + "\n".join(f"  - {line}" for line in mismatches)
        )


def interaction_matches_level(instance: MAPFBenchmarkInstance) -> bool:
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


def validate_manifest_content(
    manifest: MAPFBenchmarkManifest,
    *,
    grid_map: GridMap,
    scenarios: Sequence[Scenario],
    source_pool: MAPFBenchmarkSourcePool,
    agent_counts: Sequence[int],
    instances_per_level: int,
    expected_map_stem: str | None = None,
) -> None:
    expected_total = len(agent_counts) * len(MAPFInteractionLevel) * instances_per_level
    if len(manifest.instances) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} instances, found {len(manifest.instances)}"
        )

    if expected_map_stem is not None:
        expected_prefix = f"{expected_map_stem}_"
        for instance in manifest.instances:
            if not instance.instance_id.startswith(expected_prefix):
                raise RuntimeError(
                    f"Instance ID {instance.instance_id!r} does not use map stem "
                    f"{expected_map_stem!r}"
                )
            if "_HO_" in instance.instance_id:
                raise RuntimeError(
                    f"Primary catalogue instance must not use _HO_ suffix: "
                    f"{instance.instance_id}"
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
        if not interaction_matches_level(instance):
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

    for agent_count in agent_counts:
        expected_per_agent = len(MAPFInteractionLevel) * instances_per_level
        agent_instances = [
            instance
            for instance in manifest.instances
            if instance.agent_count == agent_count
        ]
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


def build_sampling_summaries(
    manifest: MAPFBenchmarkManifest,
    attempts_by_agent_count: dict[int, int],
    sampling_elapsed_by_agent_count: dict[int, float],
    agent_counts: Sequence[int],
) -> list[SamplingSummary]:
    summaries: list[SamplingSummary] = []
    for agent_count in agent_counts:
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
                sampling_elapsed_s=sampling_elapsed_by_agent_count.get(agent_count, 0.0),
                conflict_counts_by_level=conflict_counts_by_level,
            )
        )
    return summaries


def _run_validation_generation(
    *,
    config: MAPFBenchmarkCatalogueConfig,
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
        logger.info("VALIDATION ACCEPTED")
        for line in format_instance_lines(instance, attempt):
            logger.info(line)
        logger.info("")

    generation_error: ValueError | None = None
    result = None
    with lookup_only_guards():
        try:
            result = generate_benchmark_instances_from_precomputed(
                grid_map=grid_map,
                scenarios=scenarios,
                agent_count=agent_count,
                instances_per_level=config.validation_instances_per_level,
                max_timestep=config.max_timestep,
                seed=config.seed,
                max_attempts=config.validation_max_attempts,
                source_pool=source_pool,
                accepted_instance_callback=accepted_callback,
            )
        except ValueError as error:
            generation_error = error

    sampling_elapsed = time.perf_counter() - sampling_start
    expected_count = (
        len(MAPFInteractionLevel) * config.validation_instances_per_level
    )
    success = (
        generation_error is None
        and result is not None
        and len(result.instances) == expected_count
    )
    attempts = result.attempts if result is not None else config.validation_max_attempts

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


def log_completion_summary(
    logger: logging.Logger,
    config: MAPFBenchmarkCatalogueConfig,
    manifest: MAPFBenchmarkManifest,
    *,
    manifest_written: bool,
    output_path: Path,
    attempts_by_agent_count: dict[int, int],
) -> None:
    expected_total = (
        len(config.agent_counts)
        * len(MAPFInteractionLevel)
        * config.instances_per_level
    )
    by_level = {level.value: 0 for level in MAPFInteractionLevel}
    by_agent = {agent_count: 0 for agent_count in config.agent_counts}
    for instance in manifest.instances:
        by_level[instance.interaction_level.value] += 1
        by_agent[instance.agent_count] += 1

    logger.info("=" * 72)
    logger.info("MAPF-8.2 PRODUCTION CATALOGUE COMPLETION SUMMARY")
    logger.info("=" * 72)
    logger.info(f"Map: {config.map_stem}")
    logger.info(f"Seed: {config.seed}")
    logger.info(f"Generated: {len(manifest.instances)} / {expected_total}")
    logger.info(
        f"LOW: {by_level['low']} | MEDIUM: {by_level['medium']} | HIGH: {by_level['high']}"
    )
    for agent_count in config.agent_counts:
        logger.info(
            f"n={agent_count}: {by_agent[agent_count]} "
            f"(attempts={attempts_by_agent_count.get(agent_count, 'n/a')})"
        )
    if manifest_written:
        logger.info(f"Manifest written: {output_path}")
    else:
        logger.info("Manifest written: (disabled — in-memory only)")
    logger.info("")
    logger.info("Selected instances:")
    for instance in manifest.instances:
        logger.info(
            f"  {instance.instance_id}: "
            f"conflicts={instance.independent_conflict_count}, "
            f"indices={list(instance.scenario_indices)}"
        )
    logger.info("")


def run_benchmark_catalogue_generation(
    config: MAPFBenchmarkCatalogueConfig,
    *,
    logger: logging.Logger | None = None,
) -> MAPFBenchmarkCatalogueGenerationResult:
    """Run deterministic catalogue generation for one map/scenario pair."""
    active_logger = logger or setup_catalogue_logging(
        config.log_file_path,
        logger_name=f"mapf_catalogue_generation_{config.map_stem}",
    )
    timer = ElapsedTimer()

    active_logger.info(f"{timer.stamp()} Starting catalogue generation")
    log_catalogue_config_summary(active_logger, config)
    assert_production_generation_ready(config)

    active_logger.info(f"{timer.stamp()} STEP 1 — Load map/scenarios")
    grid_map = load_moving_ai_map(config.map_path)
    scenarios = load_moving_ai_scenarios(config.scen_path)
    active_logger.info(f"Map loaded: {grid_map.name}")
    active_logger.info(f"Scenarios loaded: {len(scenarios)}")
    active_logger.info("")

    active_logger.info(f"{timer.stamp()} STEP 2 — Build source pool")
    precompute_start = time.perf_counter()

    def precompute_progress(progress: StaticPrecomputeProgress) -> None:
        elapsed = time.perf_counter() - precompute_start
        active_logger.info(
            f"BOUNDED STATIC PRECOMPUTE {progress.completed} / {progress.total}"
        )
        active_logger.info(f"  elapsed: {elapsed:.1f} s")
        active_logger.info(f"  feasible: {progress.feasible}")
        active_logger.info(f"  no spatial path: {progress.no_spatial_path}")
        active_logger.info(f"  over horizon: {progress.over_horizon}")

    original_find_path = benchmark_module.find_path

    def guard_sta_find_path(*_args, **_kwargs):
        raise AssertionError(
            "Space-Time A* must not be used for benchmark source precompute"
        )

    benchmark_module.find_path = guard_sta_find_path
    try:
        source_pool = benchmark_module.prepare_benchmark_source_pool(
            grid_map=grid_map,
            scenarios=scenarios,
            min_reference_length=config.min_reference_length,
            max_timestep=config.max_timestep,
            precompute_progress_every=config.precompute_progress_every,
            precompute_progress_callback=precompute_progress,
        )
    finally:
        benchmark_module.find_path = original_find_path

    precompute_elapsed = time.perf_counter() - precompute_start
    active_logger.info("")
    active_logger.info("SOURCE POOL PREPARATION COMPLETE")
    log_source_pool_summary(
        active_logger,
        source_pool,
        scenarios,
        precompute_elapsed_s=precompute_elapsed,
    )
    active_logger.info("")

    if config.expected_source_pool_counts is not None:
        validate_optional_source_pool_counts(
            source_pool,
            config.expected_source_pool_counts,
        )
        active_logger.info("Source pool counts match expected baseline.")
        active_logger.info("")

    if not config.skip_validation:
        for agent_count in config.validation_agent_counts:
            active_logger.info(
                f"{timer.stamp()} STEP — Validate {agent_count}-agent generation"
            )
            validation = _run_validation_generation(
                config=config,
                grid_map=grid_map,
                scenarios=scenarios,
                source_pool=source_pool,
                agent_count=agent_count,
                logger=active_logger,
            )
            if not validation.success:
                raise RuntimeError(
                    f"Validation generation failed for agent_count={agent_count}"
                )

    active_logger.info(f"{timer.stamp()} STEP — Generate catalogue manifest")
    sampling_elapsed_by_agent_count: dict[int, float] = {}
    generation_error: ValueError | None = None
    generation_result: MAPFBenchmarkManifestGenerationResult | None = None
    sampling_starts: dict[int, float] = {}

    def generation_start_tracked(agent_count: int) -> None:
        sampling_starts[agent_count] = time.perf_counter()

    def accepted_callback(instance: MAPFBenchmarkInstance, attempt: int) -> None:
        active_logger.info("CATALOGUE ACCEPTED")
        for line in format_instance_lines(instance, attempt):
            active_logger.info(line)
        active_logger.info("")

    with lookup_only_guards():
        try:
            generation_result = generate_benchmark_manifest(
                grid_map=grid_map,
                scenarios=scenarios,
                scenario_name=config.scen_path.name,
                agent_counts=config.agent_counts,
                instances_per_level=config.instances_per_level,
                max_timestep=config.max_timestep,
                seed=config.seed,
                min_reference_length=config.min_reference_length,
                max_attempts=config.max_attempts,
                source_pool=source_pool,
                accepted_instance_callback=accepted_callback,
                generation_start_callback=generation_start_tracked,
            )
            for agent_count in config.agent_counts:
                sampling_elapsed_by_agent_count[agent_count] = (
                    time.perf_counter() - sampling_starts[agent_count]
                )
        except ValueError as error:
            generation_error = error

    if generation_error is not None or generation_result is None:
        active_logger.error("")
        active_logger.error("CATALOGUE GENERATION FAILED")
        active_logger.error(f"Map: {config.map_stem}")
        active_logger.error(f"Seed: {config.seed}")
        active_logger.error(f"Underlying error: {generation_error}")
        active_logger.error(
            "No interaction thresholds, seeds, agent counts, or target instance "
            "counts were changed automatically."
        )
        raise RuntimeError(
            f"Catalogue generation incomplete for map={config.map_stem}, "
            f"seed={config.seed}: {generation_error}"
        ) from generation_error

    manifest = generation_result.manifest
    attempts_by_agent_count = generation_result.attempts_by_agent_count

    expected_total = (
        len(config.agent_counts)
        * len(MAPFInteractionLevel)
        * config.instances_per_level
    )
    if len(manifest.instances) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} instances, found {len(manifest.instances)}"
        )

    active_logger.info(f"{timer.stamp()} STEP — Validate manifest content")
    validate_manifest_content(
        manifest,
        grid_map=grid_map,
        scenarios=scenarios,
        source_pool=source_pool,
        agent_counts=config.agent_counts,
        instances_per_level=config.instances_per_level,
        expected_map_stem=config.map_stem,
    )
    active_logger.info("Manifest content validation passed.")
    active_logger.info("")

    active_logger.info(f"{timer.stamp()} STEP — Determinism check")
    with lookup_only_guards():
        repeat_result = generate_benchmark_manifest(
            grid_map=grid_map,
            scenarios=scenarios,
            scenario_name=config.scen_path.name,
            agent_counts=config.agent_counts,
            instances_per_level=config.instances_per_level,
            max_timestep=config.max_timestep,
            seed=config.seed,
            min_reference_length=config.min_reference_length,
            max_attempts=config.max_attempts,
            source_pool=source_pool,
        )
    if repeat_result.manifest.instances != manifest.instances:
        raise RuntimeError("Deterministic regeneration mismatch")
    active_logger.info("Determinism check passed.")
    active_logger.info("")

    manifest_written = False
    output_path = config.output_manifest_path.resolve()

    if config.allow_production_manifest_write:
        assert_safe_manifest_output(
            output_path,
            allow_production_manifest_write=True,
        )
        active_logger.info(f"{timer.stamp()} STEP — Save manifest")
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        save_benchmark_manifest(manifest, tmp_path)
        loaded_manifest = load_benchmark_manifest(tmp_path)
        if loaded_manifest != manifest:
            raise RuntimeError("JSON roundtrip mismatch")
        tmp_path.replace(output_path)
        manifest_written = True
        active_logger.info(f"Manifest atomically saved to: {output_path}")
    else:
        active_logger.info(
            "Production manifest write disabled (allow_production_manifest_write=False). "
            "Generation completed in memory only."
        )

    summaries = build_sampling_summaries(
        manifest,
        attempts_by_agent_count,
        sampling_elapsed_by_agent_count,
        config.agent_counts,
    )
    active_logger.info("=" * 72)
    active_logger.info("CATALOGUE GENERATION SUMMARY")
    active_logger.info("=" * 72)
    active_logger.info(f"Total instances: {len(manifest.instances)}")
    active_logger.info("")
    active_logger.info("Agent count | Level  | Instances | Attempts needed | Conflict counts")
    for summary in summaries:
        for level in MAPFInteractionLevel:
            counts = summary.conflict_counts_by_level[level]
            active_logger.info(
                f"{summary.agent_count:11d} | {level.name:6s} | "
                f"{len(counts):9d} | {summary.attempts:15d} | {counts}"
            )
    active_logger.info("")
    log_completion_summary(
        active_logger,
        config,
        manifest,
        manifest_written=manifest_written,
        output_path=output_path,
        attempts_by_agent_count=attempts_by_agent_count,
    )
    if manifest_written:
        from pathfinding.src.experiments.mapf_cross_map_catalogue_validation import (
            manifest_file_sha256,
        )

        active_logger.info(f"SHA-256: {manifest_file_sha256(output_path)}")
        active_logger.info(
            "Run mapf_validate_cross_map_catalogue_main.py to validate and freeze."
        )
        active_logger.info("")

    return MAPFBenchmarkCatalogueGenerationResult(
        manifest=manifest,
        source_pool=source_pool,
        attempts_by_agent_count=attempts_by_agent_count,
        manifest_written=manifest_written,
        output_manifest_path=output_path,
    )


def run_benchmark_catalogue_generation_main(
    config: MAPFBenchmarkCatalogueConfig,
) -> int:
    logger = setup_catalogue_logging(
        config.log_file_path,
        logger_name=f"mapf_catalogue_generation_{config.map_stem}",
    )
    timer = ElapsedTimer()
    try:
        run_benchmark_catalogue_generation(config, logger=logger)
        logger.info(f"{timer.stamp()} Catalogue generation complete")
        return 0
    except Exception:
        logger.exception(f"{timer.stamp()} Unexpected error during catalogue generation:")
        logger.error(traceback.format_exc())
        return 1
