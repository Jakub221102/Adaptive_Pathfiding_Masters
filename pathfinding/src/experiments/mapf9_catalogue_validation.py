"""Post-generation validation and freeze reporting for MAPF-9 catalogues."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pathfinding.src.experiments import mapf_benchmark_instances as benchmark_module
from pathfinding.src.experiments.mapf_benchmark_catalogue_generation import (
    DEFAULT_AGENT_COUNTS,
    DEFAULT_INSTANCES_PER_LEVEL,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_TIMESTEP,
    DEFAULT_MIN_REFERENCE_LENGTH,
    MAPFBenchmarkCatalogueConfig,
    mapf9_cross_map_catalogue_config,
    setup_catalogue_logging,
    validate_manifest_content,
)
from pathfinding.src.experiments.mapf_benchmark_instances import (
    CATALOGUE_ROLE_MAPF9,
    MAPF8_CROSS_MAP_REFERENCE_SEEDS,
    MAPF9_CATALOGUE_SEEDS,
    MAPFBenchmarkManifest,
    MAPFInteractionLevel,
    collect_all_scenario_set_signatures,
    load_benchmark_manifest,
    mapf8_mapf9_signature_overlap,
    validate_mapf8_mapf9_signature_disjointness,
)
from pathfinding.src.experiments.mapf_cross_map_catalogue_validation import (
    manifest_file_sha256,
    validate_instance_id_prefixes,
    validate_manifest_identity,
    validate_manifest_scenario_semantics,
    validate_manifest_structure,
)
from pathfinding.src.loaders.map_loader import load_moving_ai_map
from pathfinding.src.loaders.scen_loader import load_moving_ai_scenarios


@dataclass(frozen=True, slots=True)
class MAPF9CatalogueValidationConfig:
    map_stem: str
    manifest_path: Path
    mapf8_reference_manifest_path: Path
    map_path: Path
    scen_path: Path
    report_path: Path
    log_file_path: Path
    expected_seed: int
    expected_mapf8_seed: int
    agent_counts: tuple[int, ...] = DEFAULT_AGENT_COUNTS
    instances_per_level: int = DEFAULT_INSTANCES_PER_LEVEL
    max_timestep: int = DEFAULT_MAX_TIMESTEP
    min_reference_length: float = DEFAULT_MIN_REFERENCE_LENGTH
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    recompute_source_pool: bool = True


@dataclass
class MAPF9CatalogueValidationResult:
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
    mapf9_signature_count: int = 0
    mapf8_signature_count: int = 0
    signature_overlap_count: int = 0

    def add(self, message: str, *, passed: bool = True) -> None:
        self.messages.append(message)
        if not passed:
            self.passed = False


def mapf9_catalogue_validation_config(
    target_map: str,
    repo_root: Path,
) -> MAPF9CatalogueValidationConfig:
    generation_config = mapf9_cross_map_catalogue_config(target_map, repo_root)
    benchmarks_dir = repo_root / "pathfinding" / "results" / "mapf_benchmarks"
    map_stem = generation_config.map_stem
    return MAPF9CatalogueValidationConfig(
        map_stem=map_stem,
        manifest_path=generation_config.output_manifest_path,
        mapf8_reference_manifest_path=benchmarks_dir / f"{map_stem}_manifest.json",
        map_path=generation_config.map_path,
        scen_path=generation_config.scen_path,
        report_path=benchmarks_dir / f"{map_stem}_mapf9_validation.md",
        log_file_path=repo_root / "pathfinding" / "results" / f"mapf9_catalogue_validation_{map_stem}.log",
        expected_seed=MAPF9_CATALOGUE_SEEDS[map_stem],
        expected_mapf8_seed=MAPF8_CROSS_MAP_REFERENCE_SEEDS[map_stem],
    )


def validate_mapf9_manifest_identity(
    manifest: MAPFBenchmarkManifest,
    config: MAPF9CatalogueValidationConfig,
) -> None:
    if Path(manifest.map_name).stem != config.map_stem:
        raise ValueError(
            f"Expected map stem {config.map_stem!r}, found map_name={manifest.map_name!r}"
        )
    if manifest.seed != config.expected_seed:
        raise ValueError(
            f"Expected MAPF-9 seed {config.expected_seed}, found seed={manifest.seed}"
        )
    if manifest.catalogue_role != CATALOGUE_ROLE_MAPF9:
        raise ValueError(
            f"Expected catalogue_role={CATALOGUE_ROLE_MAPF9!r}, "
            f"found {manifest.catalogue_role!r}"
        )
    if manifest.max_timestep != config.max_timestep:
        raise ValueError(
            f"Expected max_timestep={config.max_timestep}, found {manifest.max_timestep}"
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


def validate_mapf9_production_manifest(
    config: MAPF9CatalogueValidationConfig,
    *,
    manifest: MAPFBenchmarkManifest | None = None,
    logger: logging.Logger | None = None,
) -> MAPF9CatalogueValidationResult:
    manifest_path = config.manifest_path.resolve()
    if manifest is None and not manifest_path.is_file():
        raise FileNotFoundError(f"MAPF-9 manifest not found: {manifest_path}")

    mapf8_reference = load_benchmark_manifest(config.mapf8_reference_manifest_path)
    if mapf8_reference.seed != config.expected_mapf8_seed:
        raise ValueError(
            f"Expected frozen MAPF-8 seed {config.expected_mapf8_seed}, "
            f"found {mapf8_reference.seed}"
        )

    loaded = manifest if manifest is not None else load_benchmark_manifest(manifest_path)
    if manifest_path.is_file():
        sha256 = manifest_file_sha256(manifest_path)
    else:
        from pathfinding.src.experiments.mapf_benchmark_instances import _manifest_to_json

        payload = json.dumps(_manifest_to_json(loaded), sort_keys=True, indent=2)
        sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    mapf9_signatures, mapf8_signatures, intersection = mapf8_mapf9_signature_overlap(
        loaded,
        mapf8_reference,
    )

    result = MAPF9CatalogueValidationResult(
        passed=True,
        map_stem=config.map_stem,
        manifest_path=manifest_path,
        sha256=sha256,
        instance_count=len(loaded.instances),
        mapf9_signature_count=len(mapf9_signatures),
        mapf8_signature_count=len(mapf8_signatures),
        signature_overlap_count=len(intersection),
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

    _check("MAPF-9 manifest identity", lambda: validate_mapf9_manifest_identity(loaded, config))
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

    _check(
        "MAPF-9 vs MAPF-8 scenario-set signature disjointness",
        lambda: validate_mapf8_mapf9_signature_disjointness(loaded, mapf8_reference),
    )

    if len(collect_all_scenario_set_signatures(loaded.instances)) != len(loaded.instances):
        result.add(
            "FAIL: within-manifest duplicate scenario-set signatures",
            passed=False,
        )
    else:
        result.add("PASS: within-manifest scenario-set signatures unique")

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
            source_pool = benchmark_module.prepare_benchmark_source_pool(
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


def build_mapf9_freeze_report_markdown(
    config: MAPF9CatalogueValidationConfig,
    result: MAPF9CatalogueValidationResult,
    *,
    generation_config: MAPFBenchmarkCatalogueConfig | None = None,
) -> str:
    status = "PASS" if result.passed else "FAIL"
    lines = [
        f"# MAPF-9 Catalogue Validation — {config.map_stem}",
        "",
        "## Identity",
        "",
        f"- Map: **{config.map_stem}**",
        f"- MAPF-9 seed: **{config.expected_seed}**",
        f"- Frozen MAPF-8 reference seed: **{config.expected_mapf8_seed}**",
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
        "## Freshness / disjointness",
        "",
        f"- MAPF-9 scenario-set signature count: {result.mapf9_signature_count}",
        f"- MAPF-8 reference scenario-set signature count: {result.mapf8_signature_count}",
        f"- signature intersection count: **{result.signature_overlap_count}**",
        f"- MAPF-8 reference manifest: `{config.mapf8_reference_manifest_path}`",
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
            "SPF+CD, PP, CGLPS, UBLS, or other strategy outcomes influenced instance "
            "selection.",
            "",
        ]
    )
    return "\n".join(lines)


def write_mapf9_freeze_report(
    config: MAPF9CatalogueValidationConfig,
    result: MAPF9CatalogueValidationResult,
    *,
    generation_config: MAPFBenchmarkCatalogueConfig | None = None,
) -> Path:
    report_path = config.report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_mapf9_freeze_report_markdown(
            config,
            result,
            generation_config=generation_config,
        ),
        encoding="utf-8",
    )
    return report_path


def run_mapf9_catalogue_validation_main(
    config: MAPF9CatalogueValidationConfig,
    *,
    repo_root: Path,
    write_report: bool = True,
) -> int:
    logger = setup_catalogue_logging(
        config.log_file_path,
        logger_name=f"mapf9_catalogue_validation_{config.map_stem}",
    )
    generation_config = mapf9_cross_map_catalogue_config(config.map_stem, repo_root)

    logger.info("=" * 72)
    logger.info("MAPF-9.3 CATALOGUE VALIDATION")
    logger.info("=" * 72)
    logger.info(f"  target map: {config.map_stem}")
    logger.info(f"  manifest path: {config.manifest_path.resolve()}")
    logger.info(f"  expected MAPF-9 seed: {config.expected_seed}")
    logger.info(
        f"  MAPF-8 reference manifest: {config.mapf8_reference_manifest_path.resolve()}"
    )
    logger.info(f"  recompute_source_pool: {config.recompute_source_pool}")
    logger.info("")

    try:
        result = validate_mapf9_production_manifest(config, logger=logger)
        report_path = None
        if write_report:
            report_path = write_mapf9_freeze_report(
                config,
                result,
                generation_config=generation_config,
            )
        logger.info("=" * 72)
        logger.info("MAPF-9.3 CATALOGUE VALIDATION SUMMARY")
        logger.info("=" * 72)
        logger.info(f"Map: {config.map_stem}")
        logger.info(f"Seed: {config.expected_seed}")
        logger.info(f"Manifest: {result.manifest_path}")
        logger.info(f"Generated: {result.instance_count} / 27")
        logger.info(
            f"Signature overlap with MAPF-8: {result.signature_overlap_count}"
        )
        logger.info(f"Validation: {'PASS' if result.passed else 'FAIL'}")
        logger.info(f"SHA-256: {result.sha256}")
        if report_path is not None:
            logger.info(f"Freeze report: {report_path}")
        logger.info("")
        return 0 if result.passed else 1
    except Exception:
        logger.exception("Unexpected error during MAPF-9 catalogue validation:")
        logger.error(traceback.format_exc())
        return 1
