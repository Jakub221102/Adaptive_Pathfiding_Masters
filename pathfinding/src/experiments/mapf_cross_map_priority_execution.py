"""MAPF-8 cross-map priority strategy execution harness.

Unifies SPF, CDF-H, CDF-L, and SPF+CD under one orchestrator for AR0400SR and
AR0307SR frozen catalogues. Does not modify solver or ordering implementations.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import time
import traceback
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pathfinding.src.algorithms.mapf.models import MAPFScenario
from pathfinding.src.core.models import GridMap, Scenario
from pathfinding.src.experiments.mapf_benchmark_catalogue_generation import (
    DEFAULT_AGENT_COUNTS,
    DEFAULT_INSTANCES_PER_LEVEL,
    MAPF8_CROSS_MAP_SEEDS,
    resolve_mapf8_cross_map_target,
)
from pathfinding.src.experiments.mapf_benchmark_execution import (
    TERMINATION_FAILURE,
    TERMINATION_SUCCESS,
)
from pathfinding.src.experiments.mapf_benchmark_instances import (
    MAPFBenchmarkInstance,
    MAPFBenchmarkManifest,
    MAPFInteractionLevel,
    load_benchmark_manifest,
    reconstruct_mapf_scenario_from_instance,
)
from pathfinding.src.experiments.mapf_conflict_aware_priority_execution import (
    TERMINATION_ORDERING_FAILURE as CA_TERMINATION_ORDERING_FAILURE,
    ConflictAwarePriorityRunRecord,
    ConflictAwareStrategy,
    execute_conflict_aware_priority_run,
    original_index_order,
    validate_conflict_aware_priority_run_record,
)
from pathfinding.src.experiments.mapf_cross_map_catalogue_validation import (
    manifest_file_sha256,
    validate_instance_id_prefixes,
    validate_manifest_identity,
    validate_manifest_structure,
)
from pathfinding.src.experiments.mapf_priority_strategy_execution import (
    TERMINATION_ORDERING_FAILURE as SPF_TERMINATION_ORDERING_FAILURE,
    PriorityStrategy,
    PriorityStrategyRunRecord,
    execute_priority_strategy_run,
    validate_priority_strategy_run_record,
)
from pathfinding.src.loaders.map_loader import load_moving_ai_map
from pathfinding.src.loaders.scen_loader import load_moving_ai_scenarios

MAPF8_CROSS_MAP_RESULTS_ROOT = Path("pathfinding/results/mapf8_cross_map_execution")

FROZEN_MANIFEST_SHA256: dict[str, str] = {
    "AR0400SR": "ab652d77ad99c251d7827684b7b2978a3bd5151609c5b557840eec8522a54259",
    "AR0307SR": "ff6bb64e15d8c507861966e27a00df0df1efcf7b070584b0313fc9448b4f2732",
}

FORBIDDEN_CROSS_MAP_EXECUTION_TARGETS: frozenset[str] = frozenset(
    {"AR0204SR", "AR0404SR"}
)

FORBIDDEN_HISTORICAL_RESULTS_DIRS: frozenset[Path] = frozenset(
    {
        Path("pathfinding/results/mapf_benchmark_execution"),
        Path("pathfinding/results/mapf_priority_strategy_execution"),
        Path("pathfinding/results/mapf_conflict_aware_priority_execution"),
        Path("pathfinding/results/mapf_conflict_aware_priority_heldout_execution"),
        Path("pathfinding/results/mapf_spf_heldout_execution"),
        Path("pathfinding/results/mapf_random_priority_pilot"),
        Path("pathfinding/results/mapf_random_k20_supplemental"),
    }
)

EXPECTED_INSTANCES_PER_MAP = (
    len(DEFAULT_AGENT_COUNTS)
    * len(MAPFInteractionLevel)
    * DEFAULT_INSTANCES_PER_LEVEL
)
EXPECTED_RUNS_PER_MAP = EXPECTED_INSTANCES_PER_MAP * 4


class CrossMapPriorityStrategy(str, Enum):
    SPF = "spf"
    CDF_H = "cdf_h"
    CDF_L = "cdf_l"
    SPF_CD = "spf_cd"


DEFAULT_CROSS_MAP_STRATEGIES: tuple[CrossMapPriorityStrategy, ...] = (
    CrossMapPriorityStrategy.SPF,
    CrossMapPriorityStrategy.CDF_H,
    CrossMapPriorityStrategy.CDF_L,
    CrossMapPriorityStrategy.SPF_CD,
)

STRATEGY_DISPLAY_LABELS: dict[CrossMapPriorityStrategy, str] = {
    CrossMapPriorityStrategy.SPF: "SPF",
    CrossMapPriorityStrategy.CDF_H: "CDF-H",
    CrossMapPriorityStrategy.CDF_L: "CDF-L",
    CrossMapPriorityStrategy.SPF_CD: "SPF+CD",
}


@dataclass(frozen=True, slots=True)
class CrossMapPriorityExecutionConfig:
    map_stem: str
    manifest_path: Path
    map_path: Path
    scen_path: Path
    results_dir: Path
    csv_path: Path
    jsonl_path: Path
    log_file_path: Path
    expected_manifest_sha256: str
    expected_seed: int
    resume: bool = False
    reset_results: bool = False
    save_after_each_run: bool = True
    stop_on_error: bool = False
    max_timestep: int | None = None
    instance_ids: Sequence[str] | None = None
    strategies: Sequence[CrossMapPriorityStrategy] | None = None


@dataclass(frozen=True, slots=True)
class CrossMapPriorityRunKey:
    map_name: str
    instance_id: str
    strategy: CrossMapPriorityStrategy


@dataclass(frozen=True, slots=True)
class CrossMapPriorityRunPlanEntry:
    instance: MAPFBenchmarkInstance
    strategy: CrossMapPriorityStrategy


@dataclass(frozen=True, slots=True)
class CrossMapPriorityRunRecord:
    map_name: str
    manifest_sha256: str
    instance_id: str
    agent_count: int
    interaction_level: str
    strategy: CrossMapPriorityStrategy
    agent_order: tuple[int, ...] | None
    original_index_order: tuple[int, ...] | None
    agent_degrees: tuple[int, ...] | None
    independent_conflict_count: int | None
    independent_conflict_pair_count: int | None
    ordering_low_level_searches: int | None
    incident_conflict_counts: tuple[int, ...] | None
    success: bool
    termination_reason: str
    ordering_time_ms: float
    pp_time_ms: float | None
    total_time_ms: float
    soc: int | None
    makespan: int | None
    conflict_count: int | None
    pp_low_level_searches: int | None = None
    pp_agents_planned: int | None = None
    error_message: str | None = None
    catalogue_seed: int | None = None


def resolve_mapf8_cross_map_execution_target(target_map: str) -> str:
    normalized = target_map.strip()
    if normalized in FORBIDDEN_CROSS_MAP_EXECUTION_TARGETS:
        raise ValueError(
            f"{normalized} is not a valid MAPF-8 cross-map execution target. "
            f"Use AR0400SR or AR0307SR."
        )
    return resolve_mapf8_cross_map_target(normalized)


def mapf8_cross_map_execution_config(
    target_map: str,
    repo_root: Path,
    *,
    resume: bool = False,
    reset_results: bool = False,
    save_after_each_run: bool = True,
    stop_on_error: bool = False,
    max_timestep: int | None = None,
    instance_ids: Sequence[str] | None = None,
    strategies: Sequence[CrossMapPriorityStrategy] | None = None,
) -> CrossMapPriorityExecutionConfig:
    map_stem = resolve_mapf8_cross_map_execution_target(target_map)
    benchmarks_dir = repo_root / "pathfinding" / "results" / "mapf_benchmarks"
    results_dir = repo_root / MAPF8_CROSS_MAP_RESULTS_ROOT / map_stem
    return CrossMapPriorityExecutionConfig(
        map_stem=map_stem,
        manifest_path=benchmarks_dir / f"{map_stem}_manifest.json",
        map_path=repo_root / "Data" / "bg512-map" / f"{map_stem}.map",
        scen_path=repo_root / "Data" / "bg512-scen" / f"{map_stem}.map.scen",
        results_dir=results_dir,
        csv_path=results_dir / "results.csv",
        jsonl_path=results_dir / "results_details.jsonl",
        log_file_path=results_dir / "run.log",
        expected_manifest_sha256=FROZEN_MANIFEST_SHA256[map_stem],
        expected_seed=MAPF8_CROSS_MAP_SEEDS[map_stem],
        resume=resume,
        reset_results=reset_results,
        save_after_each_run=save_after_each_run,
        stop_on_error=stop_on_error,
        max_timestep=max_timestep,
        instance_ids=instance_ids,
        strategies=strategies,
    )


def _cross_map_validation_config(config: CrossMapPriorityExecutionConfig):
    from pathfinding.src.experiments.mapf_cross_map_catalogue_validation import (
        CrossMapCatalogueValidationConfig,
    )

    return CrossMapCatalogueValidationConfig(
        map_stem=config.map_stem,
        manifest_path=config.manifest_path,
        map_path=config.map_path,
        scen_path=config.scen_path,
        report_path=config.manifest_path.with_name(
            f"{config.map_stem}_validation.md"
        ),
        log_file_path=config.log_file_path.with_name(
            f"mapf_cross_map_catalogue_validation_{config.map_stem}.log"
        ),
        expected_seed=config.expected_seed,
        recompute_source_pool=False,
    )


def assert_results_dir_isolated(results_dir: Path, repo_root: Path) -> None:
    resolved = results_dir.resolve()
    for forbidden in FORBIDDEN_HISTORICAL_RESULTS_DIRS:
        forbidden_resolved = (repo_root / forbidden).resolve()
        if resolved == forbidden_resolved or forbidden_resolved in resolved.parents:
            raise ValueError(
                f"Results directory {results_dir} must not target historical "
                f"MAPF-6/7 output: {forbidden}"
            )
    expected_root = (repo_root / MAPF8_CROSS_MAP_RESULTS_ROOT).resolve()
    if expected_root not in resolved.parents and resolved != expected_root:
        raise ValueError(
            f"Results directory {results_dir} must live under {MAPF8_CROSS_MAP_RESULTS_ROOT}"
        )


def verify_frozen_manifest(
    config: CrossMapPriorityExecutionConfig,
) -> tuple[MAPFBenchmarkManifest, str]:
    manifest_path = config.manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Frozen manifest not found: {manifest_path}")

    sha256 = manifest_file_sha256(manifest_path)
    if sha256 != config.expected_manifest_sha256:
        raise ValueError(
            f"Manifest SHA-256 mismatch for {config.map_stem}: "
            f"expected {config.expected_manifest_sha256}, found {sha256}"
        )

    manifest = load_benchmark_manifest(manifest_path)
    validation_config = _cross_map_validation_config(config)
    validate_manifest_identity(manifest, validation_config)
    validate_instance_id_prefixes(manifest, map_stem=config.map_stem)
    validate_manifest_structure(
        manifest,
        agent_counts=DEFAULT_AGENT_COUNTS,
        instances_per_level=DEFAULT_INSTANCES_PER_LEVEL,
    )
    return manifest, sha256


def build_cross_map_priority_run_plan(
    manifest: MAPFBenchmarkManifest,
    *,
    strategies: Sequence[CrossMapPriorityStrategy] = DEFAULT_CROSS_MAP_STRATEGIES,
    instance_ids: Sequence[str] | None = None,
) -> tuple[CrossMapPriorityRunPlanEntry, ...]:
    if not strategies:
        raise ValueError("strategies must not be empty")

    allowed_ids = set(instance_ids) if instance_ids is not None else None
    plan: list[CrossMapPriorityRunPlanEntry] = []

    for instance in manifest.instances:
        if allowed_ids is not None and instance.instance_id not in allowed_ids:
            continue
        for strategy in strategies:
            plan.append(
                CrossMapPriorityRunPlanEntry(instance=instance, strategy=strategy)
            )

    if not plan:
        raise ValueError("no benchmark instances matched the selection filters")

    return tuple(plan)


def run_key(record: CrossMapPriorityRunRecord) -> CrossMapPriorityRunKey:
    return CrossMapPriorityRunKey(
        map_name=record.map_name,
        instance_id=record.instance_id,
        strategy=record.strategy,
    )


def _conflict_aware_strategy(
    strategy: CrossMapPriorityStrategy,
) -> ConflictAwareStrategy:
    mapping = {
        CrossMapPriorityStrategy.CDF_H: ConflictAwareStrategy.CDF_H,
        CrossMapPriorityStrategy.CDF_L: ConflictAwareStrategy.CDF_L,
        CrossMapPriorityStrategy.SPF_CD: ConflictAwareStrategy.SPF_CD,
    }
    return mapping[strategy]


def execute_cross_map_priority_run(
    *,
    grid_map: GridMap,
    scenarios: Sequence[Scenario],
    instance: MAPFBenchmarkInstance,
    strategy: CrossMapPriorityStrategy,
    max_timestep: int,
    map_name: str,
    manifest_sha256: str,
    catalogue_seed: int | None = None,
) -> CrossMapPriorityRunRecord:
    scenario_bundle = reconstruct_mapf_scenario_from_instance(
        scenarios=scenarios,
        grid_map=grid_map,
        instance=instance,
    )
    original_scenario = scenario_bundle.scenario

    if strategy == CrossMapPriorityStrategy.SPF:
        spf_record = execute_priority_strategy_run(
            grid_map=grid_map,
            scenario=original_scenario,
            instance=instance,
            strategy=PriorityStrategy.SPF,
            max_timestep=max_timestep,
            catalogue_seed=catalogue_seed,
        )
        validate_priority_strategy_run_record(spf_record)
        index_order: tuple[int, ...] | None = None
        if spf_record.agent_order is not None:
            agent_by_id = {
                agent.agent_id: agent for agent in original_scenario.agents
            }
            reordered_agents = tuple(
                agent_by_id[agent_id] for agent_id in spf_record.agent_order
            )
            index_order = original_index_order(
                original_scenario,
                MAPFScenario(agents=reordered_agents),
            )
        return _from_spf_record(
            spf_record,
            map_name=map_name,
            manifest_sha256=manifest_sha256,
            original_index_order=index_order,
        )

    ca_strategy = _conflict_aware_strategy(strategy)
    ca_record = execute_conflict_aware_priority_run(
        grid_map=grid_map,
        scenario=original_scenario,
        instance=instance,
        strategy=ca_strategy,
        max_timestep=max_timestep,
        catalogue_seed=catalogue_seed,
    )
    validate_conflict_aware_priority_run_record(ca_record)
    return _from_conflict_aware_record(
        ca_record,
        map_name=map_name,
        manifest_sha256=manifest_sha256,
        strategy=strategy,
    )


def _from_spf_record(
    record: PriorityStrategyRunRecord,
    *,
    map_name: str,
    manifest_sha256: str,
    original_index_order: tuple[int, ...] | None,
) -> CrossMapPriorityRunRecord:
    pp_metrics = record.pp_search_metrics
    return CrossMapPriorityRunRecord(
        map_name=map_name,
        manifest_sha256=manifest_sha256,
        instance_id=record.instance_id,
        agent_count=record.agent_count,
        interaction_level=record.interaction_level,
        strategy=CrossMapPriorityStrategy.SPF,
        agent_order=record.agent_order,
        original_index_order=original_index_order,
        agent_degrees=None,
        independent_conflict_count=None,
        independent_conflict_pair_count=None,
        ordering_low_level_searches=None,
        incident_conflict_counts=None,
        success=record.success,
        termination_reason=record.termination_reason,
        ordering_time_ms=record.ordering_time_ms,
        pp_time_ms=record.pp_time_ms,
        total_time_ms=record.total_time_ms,
        soc=record.soc,
        makespan=record.makespan,
        conflict_count=record.conflict_count,
        pp_low_level_searches=(
            None if pp_metrics is None else pp_metrics.low_level_searches
        ),
        pp_agents_planned=(
            None if pp_metrics is None else pp_metrics.agents_planned
        ),
        error_message=record.error_message,
        catalogue_seed=record.catalogue_seed,
    )


def _from_conflict_aware_record(
    record: ConflictAwarePriorityRunRecord,
    *,
    map_name: str,
    manifest_sha256: str,
    strategy: CrossMapPriorityStrategy,
) -> CrossMapPriorityRunRecord:
    pp_metrics = record.pp_search_metrics
    return CrossMapPriorityRunRecord(
        map_name=map_name,
        manifest_sha256=manifest_sha256,
        instance_id=record.instance_id,
        agent_count=record.agent_count,
        interaction_level=record.interaction_level,
        strategy=strategy,
        agent_order=record.agent_order,
        original_index_order=record.original_index_order,
        agent_degrees=record.agent_degrees,
        independent_conflict_count=record.independent_conflict_count,
        independent_conflict_pair_count=record.independent_conflict_pair_count,
        ordering_low_level_searches=record.ordering_low_level_searches,
        incident_conflict_counts=record.incident_conflict_counts,
        success=record.success,
        termination_reason=record.termination_reason,
        ordering_time_ms=record.ordering_time_ms,
        pp_time_ms=record.pp_time_ms,
        total_time_ms=record.total_time_ms,
        soc=record.soc,
        makespan=record.makespan,
        conflict_count=record.conflict_count,
        pp_low_level_searches=(
            None if pp_metrics is None else pp_metrics.low_level_searches
        ),
        pp_agents_planned=(
            None if pp_metrics is None else pp_metrics.agents_planned
        ),
        error_message=record.error_message,
        catalogue_seed=record.catalogue_seed,
    )


def _int_tuple_to_json(values: tuple[int, ...] | None) -> list[int] | None:
    if values is None:
        return None
    return list(values)


def _run_record_to_json(record: CrossMapPriorityRunRecord) -> dict[str, object]:
    payload: dict[str, object] = {
        "map_name": record.map_name,
        "manifest_sha256": record.manifest_sha256,
        "instance_id": record.instance_id,
        "agent_count": record.agent_count,
        "interaction_level": record.interaction_level,
        "strategy": record.strategy.value,
        "agent_order": _int_tuple_to_json(record.agent_order),
        "original_index_order": _int_tuple_to_json(record.original_index_order),
        "agent_degrees": _int_tuple_to_json(record.agent_degrees),
        "independent_conflict_count": record.independent_conflict_count,
        "independent_conflict_pair_count": record.independent_conflict_pair_count,
        "ordering_low_level_searches": record.ordering_low_level_searches,
        "incident_conflict_counts": _int_tuple_to_json(record.incident_conflict_counts),
        "success": record.success,
        "termination_reason": record.termination_reason,
        "ordering_time_ms": record.ordering_time_ms,
        "pp_time_ms": record.pp_time_ms,
        "total_time_ms": record.total_time_ms,
        "soc": record.soc,
        "makespan": record.makespan,
        "conflict_count": record.conflict_count,
        "pp_low_level_searches": record.pp_low_level_searches,
        "pp_agents_planned": record.pp_agents_planned,
        "catalogue_seed": record.catalogue_seed,
    }
    if record.error_message is not None:
        payload["error_message"] = record.error_message
    return payload


def _parse_int_tuple(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("expected list for integer tuple field")
    return tuple(int(item) for item in value)


def _run_record_from_json(data: dict[str, object]) -> CrossMapPriorityRunRecord:
    return CrossMapPriorityRunRecord(
        map_name=str(data["map_name"]),
        manifest_sha256=str(data["manifest_sha256"]),
        instance_id=str(data["instance_id"]),
        agent_count=int(data["agent_count"]),
        interaction_level=str(data["interaction_level"]),
        strategy=CrossMapPriorityStrategy(str(data["strategy"])),
        agent_order=_parse_int_tuple(data.get("agent_order")),
        original_index_order=_parse_int_tuple(data.get("original_index_order")),
        agent_degrees=_parse_int_tuple(data.get("agent_degrees")),
        independent_conflict_count=(
            None
            if data.get("independent_conflict_count") is None
            else int(data["independent_conflict_count"])
        ),
        independent_conflict_pair_count=(
            None
            if data.get("independent_conflict_pair_count") is None
            else int(data["independent_conflict_pair_count"])
        ),
        ordering_low_level_searches=(
            None
            if data.get("ordering_low_level_searches") is None
            else int(data["ordering_low_level_searches"])
        ),
        incident_conflict_counts=_parse_int_tuple(data.get("incident_conflict_counts")),
        success=bool(data["success"]),
        termination_reason=str(data["termination_reason"]),
        ordering_time_ms=float(data["ordering_time_ms"]),
        pp_time_ms=(
            None if data.get("pp_time_ms") is None else float(data["pp_time_ms"])
        ),
        total_time_ms=float(data["total_time_ms"]),
        soc=None if data.get("soc") is None else int(data["soc"]),
        makespan=None if data.get("makespan") is None else int(data["makespan"]),
        conflict_count=(
            None if data.get("conflict_count") is None else int(data["conflict_count"])
        ),
        pp_low_level_searches=(
            None
            if data.get("pp_low_level_searches") is None
            else int(data["pp_low_level_searches"])
        ),
        pp_agents_planned=(
            None
            if data.get("pp_agents_planned") is None
            else int(data["pp_agents_planned"])
        ),
        error_message=(
            None if data.get("error_message") is None else str(data["error_message"])
        ),
        catalogue_seed=(
            None if data.get("catalogue_seed") is None else int(data["catalogue_seed"])
        ),
    )


def load_checkpoint_records(
    jsonl_path: Path,
) -> dict[CrossMapPriorityRunKey, CrossMapPriorityRunRecord]:
    if not jsonl_path.is_file():
        return {}

    records: dict[CrossMapPriorityRunKey, CrossMapPriorityRunRecord] = {}
    duplicate_keys: list[str] = []

    with jsonl_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            data = json.loads(stripped)
            if not isinstance(data, dict):
                raise ValueError(
                    f"Invalid JSONL record at {jsonl_path}:{line_number}"
                )

            record = _run_record_from_json(data)
            key = run_key(record)
            if key in records:
                duplicate_keys.append(
                    f"{key.map_name} / {key.instance_id} / {key.strategy.value} "
                    f"(line {line_number})"
                )
            records[key] = record

    if duplicate_keys:
        joined = "\n  - ".join(duplicate_keys)
        raise RuntimeError(
            "Duplicate cross-map priority run keys found in checkpoint file "
            f"{jsonl_path}:\n  - {joined}"
        )

    return records


def validate_checkpoint_records(
    records: dict[CrossMapPriorityRunKey, CrossMapPriorityRunRecord],
    *,
    map_name: str,
    manifest_sha256: str,
    allowed_instance_ids: set[str],
    allowed_strategies: set[CrossMapPriorityStrategy],
) -> None:
    for key, record in records.items():
        if key.map_name != map_name:
            raise ValueError(
                f"Checkpoint record map mismatch: expected {map_name!r}, "
                f"found {key.map_name!r} for {key.instance_id} / {key.strategy.value}"
            )
        if record.map_name != map_name:
            raise ValueError(
                f"Checkpoint record body map mismatch for {record.instance_id}"
            )
        if record.manifest_sha256 != manifest_sha256:
            raise ValueError(
                f"Checkpoint manifest hash mismatch for {record.instance_id} / "
                f"{record.strategy.value}"
            )
        if key.instance_id not in allowed_instance_ids:
            raise ValueError(
                f"Unknown checkpoint instance_id: {key.instance_id}"
            )
        if key.strategy not in allowed_strategies:
            raise ValueError(
                f"Unknown checkpoint strategy: {key.strategy.value}"
            )
        validate_cross_map_priority_run_record(record)


def validate_cross_map_priority_run_record(
    record: CrossMapPriorityRunRecord,
) -> None:
    allowed_terminations = {
        TERMINATION_SUCCESS,
        TERMINATION_FAILURE,
        SPF_TERMINATION_ORDERING_FAILURE,
        CA_TERMINATION_ORDERING_FAILURE,
    }
    if record.termination_reason not in allowed_terminations:
        raise RuntimeError(
            f"unexpected termination_reason: {record.termination_reason}"
        )

    if record.ordering_time_ms < 0 or record.total_time_ms < 0:
        raise RuntimeError("timing fields must be non-negative")
    if record.pp_time_ms is not None and record.pp_time_ms < 0:
        raise RuntimeError("pp_time_ms must be non-negative")

    is_ordering_failure = record.termination_reason in {
        SPF_TERMINATION_ORDERING_FAILURE,
        CA_TERMINATION_ORDERING_FAILURE,
    }
    if is_ordering_failure:
        if record.pp_time_ms is not None:
            raise RuntimeError("ordering failure must not include pp_time_ms")
        if record.total_time_ms != record.ordering_time_ms:
            raise RuntimeError(
                "ordering failure total_time_ms must equal ordering_time_ms"
            )
        return

    if record.pp_time_ms is None:
        raise RuntimeError("PP phase must include pp_time_ms")
    expected_total = record.ordering_time_ms + record.pp_time_ms
    if abs(record.total_time_ms - expected_total) > 1e-6:
        raise RuntimeError("total_time_ms must equal ordering_time_ms + pp_time_ms")

    if record.success:
        if record.conflict_count != 0:
            raise RuntimeError("successful run reports remaining conflicts")
        if record.soc is None or record.makespan is None:
            raise RuntimeError("successful run missing quality metrics")


def _flatten_run_record(record: CrossMapPriorityRunRecord) -> dict[str, object]:
    row: dict[str, object] = {
        "map_name": record.map_name,
        "manifest_sha256": record.manifest_sha256,
        "instance_id": record.instance_id,
        "agent_count": record.agent_count,
        "interaction_level": record.interaction_level,
        "strategy": record.strategy.value,
        "agent_order": (
            None if record.agent_order is None else json.dumps(list(record.agent_order))
        ),
        "original_index_order": (
            None
            if record.original_index_order is None
            else json.dumps(list(record.original_index_order))
        ),
        "agent_degrees": (
            None if record.agent_degrees is None else json.dumps(list(record.agent_degrees))
        ),
        "independent_conflict_count": record.independent_conflict_count,
        "independent_conflict_pair_count": record.independent_conflict_pair_count,
        "ordering_low_level_searches": record.ordering_low_level_searches,
        "incident_conflict_counts": (
            None
            if record.incident_conflict_counts is None
            else json.dumps(list(record.incident_conflict_counts))
        ),
        "success": record.success,
        "termination_reason": record.termination_reason,
        "ordering_time_ms": record.ordering_time_ms,
        "pp_time_ms": record.pp_time_ms,
        "total_time_ms": record.total_time_ms,
        "soc": record.soc,
        "makespan": record.makespan,
        "conflict_count": record.conflict_count,
        "pp_low_level_searches": record.pp_low_level_searches,
        "pp_agents_planned": record.pp_agents_planned,
        "error_message": record.error_message,
        "catalogue_seed": record.catalogue_seed,
    }
    return row


def rewrite_checkpoint_csv(
    records: Iterable[CrossMapPriorityRunRecord],
    csv_path: Path,
) -> None:
    rows = [_flatten_run_record(record) for record in records]
    if not rows:
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    os.replace(temp_path, csv_path)


class CrossMapPriorityCheckpointStore:
    """Append-only JSONL checkpoint store with atomic CSV rewrite."""

    def __init__(
        self,
        *,
        results_dir: Path,
        csv_path: Path,
        jsonl_path: Path,
        reset_results: bool = False,
    ) -> None:
        self.results_dir = results_dir
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path
        self.results_dir.mkdir(parents=True, exist_ok=True)

        if reset_results:
            self.csv_path.unlink(missing_ok=True)
            self.jsonl_path.unlink(missing_ok=True)

        self._records = load_checkpoint_records(self.jsonl_path)

    @property
    def records(self) -> dict[CrossMapPriorityRunKey, CrossMapPriorityRunRecord]:
        return dict(self._records)

    def completed_keys(self) -> set[CrossMapPriorityRunKey]:
        return set(self._records)

    def append_record(self, record: CrossMapPriorityRunRecord) -> None:
        key = run_key(record)
        if key in self._records:
            raise RuntimeError(
                f"Refusing to overwrite existing checkpoint for "
                f"{key.map_name} / {key.instance_id} / {key.strategy.value}"
            )

        line = json.dumps(_run_record_to_json(record), sort_keys=True)
        with self.jsonl_path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
            file.flush()
            os.fsync(file.fileno())

        self._records[key] = record
        rewrite_checkpoint_csv(self._ordered_records(), self.csv_path)

    def _ordered_records(self) -> tuple[CrossMapPriorityRunRecord, ...]:
        strategy_order = {
            CrossMapPriorityStrategy.SPF: 0,
            CrossMapPriorityStrategy.CDF_H: 1,
            CrossMapPriorityStrategy.CDF_L: 2,
            CrossMapPriorityStrategy.SPF_CD: 3,
        }
        return tuple(
            sorted(
                self._records.values(),
                key=lambda record: (
                    record.agent_count,
                    record.interaction_level,
                    record.instance_id,
                    strategy_order[record.strategy],
                ),
            )
        )


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


def setup_cross_map_logging(log_file: Path, *, append: bool) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("mapf_cross_map_priority_execution")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
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


def _count_by_outcome(records: Sequence[CrossMapPriorityRunRecord]) -> dict[str, int]:
    counts = {
        TERMINATION_SUCCESS: 0,
        TERMINATION_FAILURE: 0,
        SPF_TERMINATION_ORDERING_FAILURE: 0,
    }
    for record in records:
        counts[record.termination_reason] = counts.get(record.termination_reason, 0) + 1
    return counts


def _count_by_strategy(
    records: Sequence[CrossMapPriorityRunRecord],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        label = STRATEGY_DISPLAY_LABELS[record.strategy]
        counts[label] = counts.get(label, 0) + 1
    return counts


def _count_by_agent_count(
    records: Sequence[CrossMapPriorityRunRecord],
) -> dict[int, int]:
    counts: dict[int, int] = {}
    for record in records:
        counts[record.agent_count] = counts.get(record.agent_count, 0) + 1
    return counts


def _count_by_interaction(
    records: Sequence[CrossMapPriorityRunRecord],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.interaction_level] = (
            counts.get(record.interaction_level, 0) + 1
        )
    return counts


def run_cross_map_priority_benchmark(
    config: CrossMapPriorityExecutionConfig,
    *,
    repo_root: Path,
    logger: logging.Logger | None = None,
) -> int:
    if config.reset_results and config.resume:
        raise ValueError("reset_results=True cannot be combined with resume=True")

    assert_results_dir_isolated(config.results_dir, repo_root)

    manifest, manifest_sha256 = verify_frozen_manifest(config)
    map_name = config.map_stem

    append_log = config.resume and config.log_file_path.is_file() and not config.reset_results
    active_logger = logger or setup_cross_map_logging(
        config.log_file_path,
        append=append_log,
    )

    checkpoint = CrossMapPriorityCheckpointStore(
        results_dir=config.results_dir,
        csv_path=config.csv_path,
        jsonl_path=config.jsonl_path,
        reset_results=config.reset_results,
    )

    allowed_instance_ids = {instance.instance_id for instance in manifest.instances}
    strategies = config.strategies or DEFAULT_CROSS_MAP_STRATEGIES
    allowed_strategies = set(strategies)

    if checkpoint.records:
        validate_checkpoint_records(
            checkpoint.records,
            map_name=map_name,
            manifest_sha256=manifest_sha256,
            allowed_instance_ids=allowed_instance_ids,
            allowed_strategies=allowed_strategies,
        )

    max_timestep = (
        config.max_timestep if config.max_timestep is not None else manifest.max_timestep
    )

    plan = build_cross_map_priority_run_plan(
        manifest,
        strategies=config.strategies or DEFAULT_CROSS_MAP_STRATEGIES,
        instance_ids=config.instance_ids,
    )
    total_runs = len(plan)
    completed_keys = checkpoint.completed_keys() if config.resume else set()

    already_completed = sum(
        1
        for entry in plan
        if CrossMapPriorityRunKey(map_name, entry.instance.instance_id, entry.strategy)
        in completed_keys
    )
    remaining = total_runs - already_completed

    active_logger.info("")
    active_logger.info("MAPF-8.3 CROSS-MAP PRIORITY STRATEGY EXECUTION")
    active_logger.info("")
    active_logger.info(f"Target map: {map_name}")
    active_logger.info(f"Manifest path: {config.manifest_path.resolve()}")
    active_logger.info(f"Verified manifest SHA-256: {manifest_sha256}")
    active_logger.info(f"Map path: {config.map_path.resolve()}")
    active_logger.info(f"Scenario path: {config.scen_path.resolve()}")
    active_logger.info(f"Expected instances: {EXPECTED_INSTANCES_PER_MAP}")
    active_logger.info("Strategies (frozen order):")
    for strategy in DEFAULT_CROSS_MAP_STRATEGIES:
        active_logger.info(
            f"  - {STRATEGY_DISPLAY_LABELS[strategy]} ({strategy.value})"
        )
    active_logger.info(f"Expected runs: {EXPECTED_RUNS_PER_MAP}")
    active_logger.info(f"Planned runs (this session): {total_runs}")
    active_logger.info(f"Already completed: {already_completed}")
    active_logger.info(f"Remaining: {remaining}")
    active_logger.info(f"Max timestep: {max_timestep}")
    active_logger.info(f"Resume: {config.resume}")
    active_logger.info(f"Reset results: {config.reset_results}")
    active_logger.info("")
    active_logger.info("Execution order: manifest instance order, then SPF → CDF-H → CDF-L → SPF+CD")
    active_logger.info("")
    active_logger.info(f"Output directory: {config.results_dir.resolve()}")
    active_logger.info("")

    grid_map = load_moving_ai_map(config.map_path)
    scenarios = load_moving_ai_scenarios(config.scen_path)
    if grid_map.name != manifest.map_name:
        raise ValueError(
            f"map name mismatch: manifest expects {manifest.map_name!r}, "
            f"loaded {grid_map.name!r}"
        )

    timer = ElapsedTimer()
    completed = 0
    session_records: list[CrossMapPriorityRunRecord] = list(checkpoint.records.values())

    try:
        for run_index, entry in enumerate(plan, start=1):
            key = CrossMapPriorityRunKey(
                map_name,
                entry.instance.instance_id,
                entry.strategy,
            )
            if config.resume and key in completed_keys:
                active_logger.info(
                    f"SKIP — already completed: {entry.instance.instance_id} / "
                    f"{entry.strategy.value}"
                )
                continue

            strategy_label = STRATEGY_DISPLAY_LABELS[entry.strategy]
            active_logger.info(
                f"RUN {run_index}/{total_runs} | "
                f"{entry.instance.instance_id} | "
                f"n={entry.instance.agent_count} | "
                f"{entry.instance.interaction_level.value} | "
                f"{strategy_label}"
            )

            try:
                record = execute_cross_map_priority_run(
                    grid_map=grid_map,
                    scenarios=scenarios,
                    instance=entry.instance,
                    strategy=entry.strategy,
                    max_timestep=max_timestep,
                    map_name=map_name,
                    manifest_sha256=manifest_sha256,
                    catalogue_seed=manifest.seed,
                )
            except Exception as error:
                active_logger.error("UNEXPECTED ERROR")
                active_logger.error(traceback.format_exc())
                raise RuntimeError(
                    f"Unexpected error for {entry.instance.instance_id} / "
                    f"{entry.strategy.value}: {error}"
                ) from error

            validate_cross_map_priority_run_record(record)

            status = "SUCCESS" if record.success else "FAILURE"
            detail = (
                f"SoC={record.soc} makespan={record.makespan}"
                if record.success
                else f"reason={record.termination_reason}"
            )
            active_logger.info(
                f"  → {status} | {detail} | "
                f"ordering={record.ordering_time_ms:.1f}ms | "
                f"pp={record.pp_time_ms:.1f}ms | "
                f"total={record.total_time_ms:.1f}ms"
                if record.pp_time_ms is not None
                else f"  → {status} | {detail} | "
                f"ordering={record.ordering_time_ms:.1f}ms | total={record.total_time_ms:.1f}ms"
            )

            if config.save_after_each_run:
                checkpoint.append_record(record)
                completed_keys.add(run_key(record))
                active_logger.info("  CHECKPOINT SAVED")

            session_records.append(record)
            completed += 1

            if completed % 5 == 0 or completed == remaining:
                counts = _count_by_outcome(session_records)
                active_logger.info(
                    f"Progress: {already_completed + completed}/{total_runs} | "
                    f"elapsed={timer.elapsed():.1f}s | remaining="
                    f"{total_runs - already_completed - completed}"
                )
                active_logger.info(
                    f"  outcomes: success={counts[TERMINATION_SUCCESS]} "
                    f"failure={counts[TERMINATION_FAILURE]} "
                    f"ordering_failure={counts[SPF_TERMINATION_ORDERING_FAILURE]}"
                )

            if config.stop_on_error and not record.success:
                active_logger.info("Stopping because stop_on_error=True and run failed.")
                break

    except KeyboardInterrupt:
        active_logger.info("")
        active_logger.info("BENCHMARK INTERRUPTED BY USER")
        active_logger.info(
            f"Completed before interrupt: {already_completed + completed} / {total_runs}"
        )
        active_logger.info(f"Checkpoint preserved: {config.jsonl_path.resolve()}")
        active_logger.info("Resume with RESUME=True, RESET_RESULTS=False.")
        raise SystemExit(130)

    final_count = already_completed + completed
    if final_count < total_runs:
        active_logger.info(
            f"Stopped early: {final_count} / {total_runs} runs present"
        )
        return 0

    counts = _count_by_outcome(session_records)
    active_logger.info("")
    active_logger.info("=" * 72)
    active_logger.info("MAPF-8.3 CROSS-MAP EXECUTION COMPLETE")
    active_logger.info("=" * 72)
    active_logger.info(f"Planned: {EXPECTED_RUNS_PER_MAP}")
    active_logger.info(f"Completed: {final_count}")
    active_logger.info(f"Success: {counts[TERMINATION_SUCCESS]}")
    active_logger.info(f"Failure: {counts[TERMINATION_FAILURE]}")
    active_logger.info(
        f"Ordering failure: {counts[SPF_TERMINATION_ORDERING_FAILURE]}"
    )
    active_logger.info(f"By strategy: {_count_by_strategy(session_records)}")
    active_logger.info(f"By agent_count: {_count_by_agent_count(session_records)}")
    active_logger.info(f"By interaction: {_count_by_interaction(session_records)}")
    active_logger.info(f"Total elapsed: {timer.elapsed():.1f} s")
    active_logger.info("")

    if manifest_file_sha256(config.manifest_path) != manifest_sha256:
        raise RuntimeError("Manifest was modified during execution")

    return 0
