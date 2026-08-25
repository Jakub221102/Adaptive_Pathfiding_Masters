"""Post-generation validation and freeze reporting for MAPF-8 cross-map catalogues."""

from __future__ import annotations

import hashlib
import logging
import sys
import traceback
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pathfinding.src.core.models import GridMap, Scenario
from pathfinding.src.experiments import mapf_benchmark_instances as benchmark_module
from pathfinding.src.experiments.mapf_benchmark_catalogue_generation import (
    DEFAULT_AGENT_COUNTS,
    DEFAULT_INSTANCES_PER_LEVEL,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_TIMESTEP,
    DEFAULT_MIN_REFERENCE_LENGTH,
    MAPF8_CROSS_MAP_SEEDS,
    MAPFBenchmarkCatalogueConfig,
    interaction_matches_level,
    mapf8_cross_map_catalogue_config,
    resolve_mapf8_cross_map_target,
    setup_catalogue_logging,
    validate_manifest_content,
)
from pathfinding.src.experiments.mapf_benchmark_instances import (
    MAPFBenchmarkManifest,
    MAPFInteractionLevel,
    _validate_candidate_indices,
    benchmark_scenario_set_signature,
    collect_scenario_set_signatures_by_agent_count,
    instance_agent_position_signature,
    load_benchmark_manifest,
    prepare_benchmark_source_pool,
)
from pathfinding.src.loaders.map_loader import load_moving_ai_map
from pathfinding.src.loaders.scen_loader import load_moving_ai_scenarios

FORBIDDEN_INSTANCE_ID_MARKERS: tuple[str, ...] = (
    "AR0204SR_",
    "AR0404SR_",
    "AR0204SR_HO_",
)


@dataclass(frozen=True, slots=True)
class CrossMapCatalogueValidationConfig:
    map_stem: str
    manifest_path: Path
    map_path: Path
    scen_path: Path
    report_path: Path
    log_file_path: Path
    expected_seed: int
    agent_counts: tuple[int, ...] = DEFAULT_AGENT_COUNTS
    instances_per_level: int = DEFAULT_INSTANCES_PER_LEVEL
    max_timestep: int = DEFAULT_MAX_TIMESTEP
    min_reference_length: float = DEFAULT_MIN_REFERENCE_LENGTH
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    recompute_source_pool: bool = True


@dataclass
class CrossMapCatalogueValidationResult:
    passed: bool
    map_stem: str
    manifest_path: Path
    sha256: str
    messages: list[str] = field(default_factory=list)
    source_pool_diagnostics: dict[str, int | float] = field(default_factory=dict)
    instance_count: int = 0
    counts_by_agent_count: dict[int, int] = field(default_factory=dict)
    counts_by_interaction_level: dict[str, int] = field(default_factory=dict)
    cell_counts: dict[tuple[int, str], int] = field(default_factory=dict)

    def add(self, message: str, *, passed: bool = True) -> None:
        self.messages.append(message)
        if not passed:
            self.passed = False


def mapf8_cross_map_validation_config(
    target_map: str,
    repo_root: Path,
) -> CrossMapCatalogueValidationConfig:
    map_stem = resolve_mapf8_cross_map_target(target_map)
    benchmarks_dir = repo_root / "pathfinding" / "results" / "mapf_benchmarks"
    return CrossMapCatalogueValidationConfig(
        map_stem=map_stem,
        manifest_path=benchmarks_dir / f"{map_stem}_manifest.json",
        map_path=repo_root / "Data" / "bg512-map" / f"{map_stem}.map",
        scen_path=repo_root / "Data" / "bg512-scen" / f"{map_stem}.map.scen",
        report_path=benchmarks_dir / f"{map_stem}_validation.md",
        log_file_path=repo_root
        / "pathfinding"
        / "results"
        / f"mapf_cross_map_catalogue_validation_{map_stem}.log",
        expected_seed=MAPF8_CROSS_MAP_SEEDS[map_stem],
    )


def manifest_file_sha256(manifest_path: Path) -> str:
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def validate_instance_id_prefixes(
    manifest: MAPFBenchmarkManifest,
    *,
    map_stem: str,
) -> None:
    expected_prefix = f"{map_stem}_"
    for instance in manifest.instances:
        if not instance.instance_id.startswith(expected_prefix):
            raise ValueError(
                f"Instance ID {instance.instance_id!r} does not begin with {expected_prefix!r}"
            )
        for forbidden in FORBIDDEN_INSTANCE_ID_MARKERS:
            if forbidden != expected_prefix and forbidden in instance.instance_id:
                raise ValueError(
                    f"Forbidden map identity marker {forbidden!r} in {instance.instance_id!r}"
                )
        if "_HO_" in instance.instance_id:
            raise ValueError(
                f"Primary catalogue instance must not use _HO_ suffix: {instance.instance_id}"
            )


def validate_manifest_identity(
    manifest: MAPFBenchmarkManifest,
    config: CrossMapCatalogueValidationConfig,
) -> None:
    if Path(manifest.map_name).stem != config.map_stem:
        raise ValueError(
            f"Expected map stem {config.map_stem!r}, found map_name={manifest.map_name!r}"
        )
    if manifest.seed != config.expected_seed:
        raise ValueError(
            f"Expected seed {config.expected_seed}, found seed={manifest.seed}"
        )
    if manifest.max_timestep != config.max_timestep:
        raise ValueError(
            f"Expected max_timestep={config.max_timestep}, "
            f"found {manifest.max_timestep}"
        )
    if manifest.min_reference_length != config.min_reference_length:
        raise ValueError(
            f"Expected min_reference_length={config.min_reference_length}, "
            f"found {manifest.min_reference_length}"
        )
    if manifest.scenario_name != config.scen_path.name:
        raise ValueError(
            f"Expected scenario_name={config.scen_path.name!r}, "
            f"found {manifest.scenario_name!r}"
        )


def validate_manifest_structure(
    manifest: MAPFBenchmarkManifest,
    *,
    agent_counts: Sequence[int],
    instances_per_level: int,
) -> dict[tuple[int, str], int]:
    expected_total = len(agent_counts) * len(MAPFInteractionLevel) * instances_per_level
    if len(manifest.instances) != expected_total:
        raise ValueError(
            f"Expected {expected_total} instances, found {len(manifest.instances)}"
        )

    cell_counts: Counter[tuple[int, str]] = Counter()
    counts_by_agent: Counter[int] = Counter()
    counts_by_level: Counter[str] = Counter()

    for instance in manifest.instances:
        if instance.agent_count not in agent_counts:
            raise ValueError(
                f"Unexpected agent_count in manifest: {instance.agent_count}"
            )
        if len(instance.scenario_indices) != instance.agent_count:
            raise ValueError(
                f"Instance {instance.instance_id} has "
                f"{len(instance.scenario_indices)} scenario indices "
                f"for agent_count={instance.agent_count}"
            )
        if not interaction_matches_level(instance):
            raise ValueError(
                f"Instance {instance.instance_id} interaction label "
                f"{instance.interaction_level.value} does not match "
                f"independent_conflict_count={instance.independent_conflict_count}"
            )
        cell_counts[(instance.agent_count, instance.interaction_level.value)] += 1
        counts_by_agent[instance.agent_count] += 1
        counts_by_level[instance.interaction_level.value] += 1

    for agent_count in agent_counts:
        expected_per_agent = len(MAPFInteractionLevel) * instances_per_level
        if counts_by_agent[agent_count] != expected_per_agent:
            raise ValueError(
                f"Expected {expected_per_agent} instances for agent_count={agent_count}, "
                f"found {counts_by_agent[agent_count]}"
            )
        for level in MAPFInteractionLevel:
            key = (agent_count, level.value)
            if cell_counts[key] != instances_per_level:
                raise ValueError(
                    f"Expected {instances_per_level} instances for "
                    f"agent_count={agent_count}, interaction={level.value}; "
                    f"found {cell_counts[key]}"
                )

    if len({instance.instance_id for instance in manifest.instances}) != len(
        manifest.instances
    ):
        raise ValueError("Duplicate instance IDs found in manifest")

    signatures_by_agent = collect_scenario_set_signatures_by_agent_count(
        manifest.instances
    )
    for agent_count, signatures in signatures_by_agent.items():
        instance_count_for_agent = sum(
            1 for instance in manifest.instances if instance.agent_count == agent_count
        )
        if len(signatures) != instance_count_for_agent:
            raise ValueError(
                f"Duplicate scenario-set signatures for agent_count={agent_count}"
            )

    return dict(cell_counts)


def validate_manifest_scenario_semantics(
    manifest: MAPFBenchmarkManifest,
    *,
    scenarios: Sequence[Scenario],
    grid_map: GridMap,
) -> None:
    seen_position_signatures: set[tuple[tuple[int, int, int, int], ...]] = set()
    for instance in manifest.instances:
        for scenario_index in instance.scenario_indices:
            if scenario_index < 0 or scenario_index >= len(scenarios):
                raise ValueError(
                    f"Instance {instance.instance_id} references invalid "
                    f"scenario_index={scenario_index}"
                )
        if not _validate_candidate_indices(
            instance.scenario_indices,
            scenarios,
            grid_map,
        ):
            raise ValueError(
                f"Invalid starts/goals for instance {instance.instance_id}"
            )
        position_signature = instance_agent_position_signature(
            scenarios,
            instance.scenario_indices,
        )
        if position_signature in seen_position_signatures:
            raise ValueError(
                f"Duplicate agent position signature for instance {instance.instance_id}"
            )
        seen_position_signatures.add(position_signature)
        if len(benchmark_scenario_set_signature(instance.scenario_indices)) != len(
            instance.scenario_indices
        ):
            raise ValueError(
                f"Duplicate scenario indices within instance {instance.instance_id}"
            )


def validate_cross_map_production_manifest(
    config: CrossMapCatalogueValidationConfig,
    *,
    manifest: MAPFBenchmarkManifest | None = None,
    logger: logging.Logger | None = None,
) -> CrossMapCatalogueValidationResult:
    manifest_path = config.manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    sha256 = manifest_file_sha256(manifest_path)
    loaded = manifest if manifest is not None else load_benchmark_manifest(manifest_path)

    result = CrossMapCatalogueValidationResult(
        passed=True,
        map_stem=config.map_stem,
        manifest_path=manifest_path,
        sha256=sha256,
        instance_count=len(loaded.instances),
    )

    def _check(label: str, fn) -> None:
        try:
            fn()
            result.add(f"PASS: {label}")
            if logger is not None:
                logger.info(f"PASS: {label}")
        except Exception as exc:
            result.add(f"FAIL: {label}: {exc}", passed=False)
            if logger is not None:
                logger.error(f"FAIL: {label}: {exc}")

    _check("manifest identity", lambda: validate_manifest_identity(loaded, config))
    _check(
        "instance ID prefixes",
        lambda: validate_instance_id_prefixes(loaded, map_stem=config.map_stem),
    )

    try:
        cell_counts = validate_manifest_structure(
            loaded,
            agent_counts=config.agent_counts,
            instances_per_level=config.instances_per_level,
        )
        result.cell_counts = cell_counts
        result.counts_by_agent_count = dict(
            Counter(instance.agent_count for instance in loaded.instances)
        )
        result.counts_by_interaction_level = dict(
            Counter(instance.interaction_level.value for instance in loaded.instances)
        )
        result.add("PASS: manifest structure (27-instance 3x3 design)")
        if logger is not None:
            logger.info("PASS: manifest structure (27-instance 3x3 design)")
    except Exception as exc:
        result.add(f"FAIL: manifest structure: {exc}", passed=False)
        if logger is not None:
            logger.error(f"FAIL: manifest structure: {exc}")

    grid_map = None
    scenarios = None
    if config.recompute_source_pool:
        grid_map = load_moving_ai_map(config.map_path)
        scenarios = load_moving_ai_scenarios(config.scen_path)
        _check(
            "scenario semantics",
            lambda: validate_manifest_scenario_semantics(
                loaded,
                scenarios=scenarios,
                grid_map=grid_map,
            ),
        )
    else:
        result.add("SKIP: scenario semantics (recompute_source_pool=False)")

    if config.recompute_source_pool and grid_map is not None and scenarios is not None:
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
                min_reference_length=config.min_reference_length,
                max_timestep=config.max_timestep,
            )
        finally:
            benchmark_module.find_path = original_find_path

        precompute_result = source_pool.precompute_result
        result.source_pool_diagnostics = {
            "source_scenario_count": len(scenarios),
            "eligible_scenario_count": len(source_pool.eligible_indices),
            "feasible_independent_path_count": len(source_pool.feasible_indices),
            "over_horizon_count": precompute_result.over_horizon_count,
            "no_spatial_path_count": precompute_result.no_spatial_path_count,
        }

        def _full_content_validation() -> None:
            validate_manifest_content(
                loaded,
                grid_map=grid_map,
                scenarios=scenarios,
                source_pool=source_pool,
                agent_counts=config.agent_counts,
                instances_per_level=config.instances_per_level,
                expected_map_stem=config.map_stem,
            )

        _check(
            "independent-path metadata recomputation",
            _full_content_validation,
        )
    elif config.recompute_source_pool:
        result.add(
            "FAIL: missing map/scenario data for full validation",
            passed=False,
        )
    else:
        result.add(
            "SKIP: independent-path metadata recomputation (recompute_source_pool=False)"
        )

    return result


def build_freeze_report_markdown(
    config: CrossMapCatalogueValidationConfig,
    result: CrossMapCatalogueValidationResult,
    *,
    generation_config: MAPFBenchmarkCatalogueConfig | None = None,
) -> str:
    status = "PASS" if result.passed else "FAIL"
    lines = [
        f"# MAPF-8 Cross-Map Catalogue Validation — {config.map_stem}",
        "",
        "## Identity",
        "",
        f"- Map: **{config.map_stem}**",
        f"- Expected seed: **{config.expected_seed}**",
        f"- Manifest path: `{result.manifest_path}`",
        f"- SHA-256: `{result.sha256}`",
        f"- Validation status: **{status}**",
        "",
        "## Generation configuration",
        "",
        f"- max_timestep: {config.max_timestep}",
        f"- min_reference_length: {config.min_reference_length}",
        f"- max_attempts: {config.max_attempts}",
        f"- agent_counts: {list(config.agent_counts)}",
        f"- instances_per_interaction_cell: {config.instances_per_level}",
        "- interaction strata: LOW=0, MEDIUM=1..2, HIGH>=3 independent-path conflicts",
        "",
        "## Catalogue structure",
        "",
        f"- instance_count: {result.instance_count}",
        f"- counts_by_agent_count: {dict(result.counts_by_agent_count)}",
        f"- counts_by_interaction_level: {dict(result.counts_by_interaction_level)}",
        f"- cell_counts (agent_count, interaction): {result.cell_counts}",
        "",
    ]

    if result.source_pool_diagnostics:
        lines.extend(["## Source-pool diagnostics", ""])
        for key, value in result.source_pool_diagnostics.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    if generation_config is not None:
        lines.extend(
            [
                "## Generation paths",
                "",
                f"- map_path: `{generation_config.map_path}`",
                f"- scen_path: `{generation_config.scen_path}`",
                f"- generation_log: `{generation_config.log_file_path}`",
                "",
            ]
        )

    lines.extend(["## Validation checks", ""])
    for message in result.messages:
        lines.append(f"- {message}")
    lines.extend(
        [
            "",
            "## Methodological note",
            "",
            "Catalogue construction used only deterministic scenario sampling and "
            "independent-path interaction classification. No SPF, CDF-H, CDF-L, "
            "SPF+CD, PP, or other strategy outcomes influenced instance selection.",
            "",
        ]
    )
    return "\n".join(lines)


def write_freeze_report(
    config: CrossMapCatalogueValidationConfig,
    result: CrossMapCatalogueValidationResult,
    *,
    generation_config: MAPFBenchmarkCatalogueConfig | None = None,
) -> Path:
    report_path = config.report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_freeze_report_markdown(
            config,
            result,
            generation_config=generation_config,
        ),
        encoding="utf-8",
    )
    return report_path


def log_validation_summary(
    logger: logging.Logger,
    config: CrossMapCatalogueValidationConfig,
    result: CrossMapCatalogueValidationResult,
    *,
    report_path: Path | None = None,
) -> None:
    logger.info("=" * 72)
    logger.info("MAPF-8.2 CROSS-MAP CATALOGUE VALIDATION SUMMARY")
    logger.info("=" * 72)
    logger.info(f"Map: {config.map_stem}")
    logger.info(f"Seed: {config.expected_seed}")
    logger.info(f"Manifest: {result.manifest_path}")
    logger.info(f"Generated: {result.instance_count} / 27")
    logger.info(
        "LOW: "
        f"{result.counts_by_interaction_level.get('low', 0)} | "
        "MEDIUM: "
        f"{result.counts_by_interaction_level.get('medium', 0)} | "
        "HIGH: "
        f"{result.counts_by_interaction_level.get('high', 0)}"
    )
    for agent_count in config.agent_counts:
        logger.info(f"n={agent_count}: {result.counts_by_agent_count.get(agent_count, 0)}")
    logger.info(f"Validation: {'PASS' if result.passed else 'FAIL'}")
    logger.info(f"SHA-256: {result.sha256}")
    if report_path is not None:
        logger.info(f"Freeze report: {report_path}")
    logger.info("")


def run_cross_map_catalogue_validation_main(
    config: CrossMapCatalogueValidationConfig,
    *,
    repo_root: Path,
    write_report: bool = True,
) -> int:
    logger = setup_catalogue_logging(
        config.log_file_path,
        logger_name=f"mapf_cross_map_validation_{config.map_stem}",
    )
    generation_config = mapf8_cross_map_catalogue_config(config.map_stem, repo_root)

    logger.info("=" * 72)
    logger.info("MAPF-8.2 CROSS-MAP CATALOGUE VALIDATION")
    logger.info("=" * 72)
    logger.info(f"  target map: {config.map_stem}")
    logger.info("  NOT AR0404SR — valid MAPF-8 targets are AR0400SR and AR0307SR only")
    logger.info(f"  manifest path: {config.manifest_path.resolve()}")
    logger.info(f"  expected seed: {config.expected_seed}")
    logger.info(f"  recompute_source_pool: {config.recompute_source_pool}")
    logger.info("")

    try:
        result = validate_cross_map_production_manifest(config, logger=logger)
        report_path = None
        if write_report:
            report_path = write_freeze_report(
                config,
                result,
                generation_config=generation_config,
            )
        log_validation_summary(logger, config, result, report_path=report_path)
        return 0 if result.passed else 1
    except Exception:
        logger.exception("Unexpected error during cross-map catalogue validation:")
        logger.error(traceback.format_exc())
        return 1
