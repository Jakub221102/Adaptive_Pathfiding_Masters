"""MAPF-9.4 primary production benchmark harness for frozen MAPF-9 catalogues."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pathfinding.src.algorithms.mapf.models import MAPFScenario
from pathfinding.src.core.models import GridMap, Scenario
from pathfinding.src.experiments.mapf_benchmark_catalogue_generation import (
    DEFAULT_AGENT_COUNTS,
    DEFAULT_INSTANCES_PER_LEVEL,
    DEFAULT_MAX_TIMESTEP,
)
from pathfinding.src.experiments.mapf_benchmark_instances import (
    CATALOGUE_ROLE_MAPF9,
    MAPF9_CATALOGUE_GENERATION_VERSION,
    MAPF9_CATALOGUE_SEEDS,
    MAPFBenchmarkInstance,
    MAPFBenchmarkManifest,
    MAPFInteractionLevel,
    load_benchmark_manifest,
    reconstruct_mapf_scenario_from_instance,
)
from pathfinding.src.experiments.mapf_bounded_local_priority_search import (
    MAPF9_CGLPS_BUDGET,
    MAPF9Method,
    MAPF9MethodResult,
    MAPF9MethodTimings,
    MAPF9PreprocessingTimings,
    MAPF9CandidateEvaluation,
    MAPF9CandidateSpec,
    MAPF9IntegrationResult,
    TERMINATION_ORDERING_FAILURE,
    evaluate_mapf9_instance,
    mapf9_cglps_logical_total_ms,
    mapf9_spf_logical_total_ms,
    mapf9_ubls_logical_total_ms,
)
from pathfinding.src.experiments.mapf_cross_map_catalogue_validation import (
    manifest_file_sha256,
    validate_instance_id_prefixes,
    validate_manifest_structure,
)
from pathfinding.src.loaders.map_loader import load_moving_ai_map
from pathfinding.src.loaders.scen_loader import load_moving_ai_scenarios

MAPF9_PRIMARY_RESULTS_ROOT = Path("pathfinding/results/mapf9_primary_execution")

MAPF9_FROZEN_MANIFEST_SHA256: dict[str, str] = {
    "AR0400SR": "14964ee449b826c6280299f68a6cfc0cc42d34f3045d87070971027009dbb71c",
    "AR0307SR": "a8bec75ead3a9f18a6d4e99fa05aa5a2f4291ee289b47debf5533ec06dc18403",
}

MAPF9_ALLOWED_MAPS: frozenset[str] = frozenset({"AR0400SR", "AR0307SR"})

MAPF9_FORBIDDEN_TARGETS: frozenset[str] = frozenset({"AR0204SR", "AR0404SR"})

MAPF9_EXPECTED_INSTANCES_PER_MAP = (
    len(DEFAULT_AGENT_COUNTS)
    * len(MAPFInteractionLevel)
    * DEFAULT_INSTANCES_PER_LEVEL
)

MAPF9_METHODS: tuple[MAPF9Method, ...] = (
    MAPF9Method.SPF,
    MAPF9Method.CGLPS,
    MAPF9Method.UBLS,
)


class BudgetDiagnosticMismatchError(RuntimeError):
    """Raised when live A_i differs from manifest-derived expectation."""


@dataclass(frozen=True, slots=True)
class MAPF9InstanceKey:
    manifest_sha256: str
    instance_id: str


@dataclass(frozen=True, slots=True)
class MAPF9MethodKey:
    manifest_sha256: str
    instance_id: str
    method: MAPF9Method


@dataclass(frozen=True, slots=True)
class MAPF9PrimaryExecutionConfig:
    map_stem: str
    manifest_path: Path
    map_path: Path
    scen_path: Path
    results_dir: Path
    instance_csv_path: Path
    method_csv_path: Path
    candidate_jsonl_path: Path
    log_file_path: Path
    expected_manifest_sha256: str
    expected_seed: int
    resume: bool = False
    reset_results: bool = False
    plan_only: bool = False
    stop_on_budget_mismatch: bool = True


@dataclass(frozen=True, slots=True)
class MAPF9BudgetPlan:
    expected_a_i_by_instance_id: dict[str, int]
    expected_sum_a_i: int
    expected_physical_pp_eval_count: int


@dataclass(frozen=True, slots=True)
class MAPF9InstanceRecord:
    map_name: str
    catalogue_role: str
    catalogue_seed: int
    manifest_sha256: str
    instance_id: str
    agent_count: int
    interaction_level: str
    manifest_independent_conflict_count: int
    manifest_conflicting_agent_pair_count: int
    manifest_vertex_conflict_count: int
    manifest_edge_conflict_count: int
    manifest_independent_soc: int
    manifest_independent_makespan: int
    independent_conflict_count: int | None
    conflicting_agent_pair_count: int | None
    pair_event_counts_json: str | None
    max_conflict_degree: int | None
    spf_order_json: str | None
    expected_a_i_from_manifest: int
    actual_a_i: int
    physical_pp_eval_count: int
    budget_match_ok: bool
    preprocessing_success: bool
    termination_reason: str
    error_message: str | None
    independent_path_time_ms: float | None
    conflict_detection_time_ms: float | None
    spf_ordering_time_ms: float | None
    baseline_spf_pp_time_ms: float | None
    cglps_candidate_ranking_time_ms: float | None
    cglps_additional_pp_time_ms: float | None
    ubls_candidate_generation_time_ms: float | None
    ubls_additional_pp_time_ms: float | None
    physical_instance_wall_time_ms: float


@dataclass(frozen=True, slots=True)
class MAPF9MethodRecord:
    manifest_sha256: str
    map_name: str
    catalogue_role: str
    catalogue_seed: int
    instance_id: str
    method: str
    success: bool
    termination_reason: str
    selected_candidate_id: int
    selected_agent_order_json: str
    soc: int | None
    makespan: int | None
    final_conflict_count: int | None
    additional_pp_eval_count: int
    logical_pp_candidate_count: int
    logical_runtime_ms: float | None
    baseline_spf_pp_time_ms: float | None
    method_specific_runtime_ms: float | None


@dataclass(frozen=True, slots=True)
class MAPF9CandidateAuditRecord:
    map_name: str
    catalogue_role: str
    catalogue_seed: int
    manifest_sha256: str
    instance_id: str
    actual_a_i: int
    expected_a_i_from_manifest: int
    physical_pp_eval_count: int
    ubls_instance_seed_text: str
    cglps_selected_candidate_id: int
    ubls_selected_candidate_id: int
    spf_candidate: dict[str, object]
    cglps_additional_candidates: list[dict[str, object]]
    ubls_additional_candidates: list[dict[str, object]]
    cglps_candidate_evaluations: list[dict[str, object]]
    ubls_candidate_evaluations: list[dict[str, object]]


def resolve_mapf9_primary_target(target_map: str) -> str:
    normalized = target_map.strip()
    if normalized in MAPF9_FORBIDDEN_TARGETS:
        raise ValueError(
            f"{normalized} is not a valid MAPF-9 primary target. "
            f"Use AR0400SR or AR0307SR."
        )
    if normalized not in MAPF9_ALLOWED_MAPS:
        raise ValueError(
            f"Unsupported MAPF-9 primary target: {target_map!r}. "
            f"Expected one of: {sorted(MAPF9_ALLOWED_MAPS)}"
        )
    return normalized


def mapf9_primary_execution_config(
    target_map: str,
    repo_root: Path,
    *,
    resume: bool = False,
    reset_results: bool = False,
    plan_only: bool = False,
) -> MAPF9PrimaryExecutionConfig:
    map_stem = resolve_mapf9_primary_target(target_map)
    benchmarks_dir = repo_root / "pathfinding" / "results" / "mapf_benchmarks"
    results_dir = repo_root / MAPF9_PRIMARY_RESULTS_ROOT / map_stem
    return MAPF9PrimaryExecutionConfig(
        map_stem=map_stem,
        manifest_path=benchmarks_dir / f"{map_stem}_mapf9_manifest.json",
        map_path=repo_root / "Data" / "bg512-map" / f"{map_stem}.map",
        scen_path=repo_root / "Data" / "bg512-scen" / f"{map_stem}.map.scen",
        results_dir=results_dir,
        instance_csv_path=results_dir / "instance_results.csv",
        method_csv_path=results_dir / "method_results.csv",
        candidate_jsonl_path=results_dir / "candidate_details.jsonl",
        log_file_path=results_dir / "run.log",
        expected_manifest_sha256=MAPF9_FROZEN_MANIFEST_SHA256[map_stem],
        expected_seed=MAPF9_CATALOGUE_SEEDS[map_stem],
        resume=resume,
        reset_results=reset_results,
        plan_only=plan_only,
    )


def instance_key(record: MAPF9InstanceRecord | MAPF9CandidateAuditRecord) -> MAPF9InstanceKey:
    return MAPF9InstanceKey(record.manifest_sha256, record.instance_id)


def method_key(record: MAPF9MethodRecord) -> MAPF9MethodKey:
    return MAPF9MethodKey(
        record.manifest_sha256,
        record.instance_id,
        MAPF9Method(record.method),
    )


def expected_a_i_from_manifest_instance(instance: MAPFBenchmarkInstance) -> int:
    return min(MAPF9_CGLPS_BUDGET, instance.conflicting_agent_pair_count)


def build_budget_plan(manifest: MAPFBenchmarkManifest) -> MAPF9BudgetPlan:
    expected_by_id = {
        instance.instance_id: expected_a_i_from_manifest_instance(instance)
        for instance in manifest.instances
    }
    expected_sum = sum(expected_by_id.values())
    return MAPF9BudgetPlan(
        expected_a_i_by_instance_id=expected_by_id,
        expected_sum_a_i=expected_sum,
        expected_physical_pp_eval_count=(
            len(manifest.instances) + 2 * expected_sum
        ),
    )


def verify_frozen_mapf9_manifest(config: MAPF9PrimaryExecutionConfig) -> tuple[MAPFBenchmarkManifest, str]:
    manifest_path = config.manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MAPF-9 manifest not found: {manifest_path}")

    if manifest_path.name.endswith("_manifest.json") and not manifest_path.name.endswith(
        "_mapf9_manifest.json"
    ):
        raise ValueError(
            f"Refusing MAPF-8 or non-MAPF-9 manifest path: {manifest_path}"
        )

    sha256 = manifest_file_sha256(manifest_path)
    if sha256 != config.expected_manifest_sha256:
        raise ValueError(
            f"Manifest SHA-256 mismatch for {config.map_stem}: "
            f"expected {config.expected_manifest_sha256}, found {sha256}"
        )

    manifest = load_benchmark_manifest(manifest_path)
    if Path(manifest.map_name).stem != config.map_stem:
        raise ValueError(
            f"Expected map stem {config.map_stem!r}, found {manifest.map_name!r}"
        )
    if manifest.seed != config.expected_seed:
        raise ValueError(
            f"Expected seed {config.expected_seed}, found {manifest.seed}"
        )
    if manifest.catalogue_role != CATALOGUE_ROLE_MAPF9:
        raise ValueError(
            f"Expected catalogue_role={CATALOGUE_ROLE_MAPF9!r}, "
            f"found {manifest.catalogue_role!r}"
        )
    if manifest.generation_version != MAPF9_CATALOGUE_GENERATION_VERSION:
        raise ValueError(
            f"Expected generation_version={MAPF9_CATALOGUE_GENERATION_VERSION!r}, "
            f"found {manifest.generation_version!r}"
        )
    if manifest.max_timestep != DEFAULT_MAX_TIMESTEP:
        raise ValueError(
            f"Expected max_timestep={DEFAULT_MAX_TIMESTEP}, "
            f"found {manifest.max_timestep}"
        )
    if len(manifest.instances) != MAPF9_EXPECTED_INSTANCES_PER_MAP:
        raise ValueError(
            f"Expected {MAPF9_EXPECTED_INSTANCES_PER_MAP} instances, "
            f"found {len(manifest.instances)}"
        )

    validate_instance_id_prefixes(manifest, map_stem=config.map_stem)
    validate_manifest_structure(
        manifest,
        agent_counts=DEFAULT_AGENT_COUNTS,
        instances_per_level=DEFAULT_INSTANCES_PER_LEVEL,
    )
    return manifest, sha256


def assert_results_dir_isolated(results_dir: Path, repo_root: Path) -> None:
    resolved = results_dir.resolve()
    allowed_root = (repo_root / MAPF9_PRIMARY_RESULTS_ROOT).resolve()
    if allowed_root not in resolved.parents and resolved != allowed_root:
        raise ValueError(
            f"Results directory must be under {allowed_root}, got {resolved}"
        )


def _json_int_tuple(values: tuple[int, ...] | None) -> str:
    if values is None:
        return "[]"
    return json.dumps(list(values))


def _candidate_spec_to_json(spec: MAPF9CandidateSpec) -> dict[str, object]:
    return {
        "candidate_id": spec.candidate_id,
        "agent_order": list(spec.agent_order),
        "source_agent_a_id": spec.source_agent_a_id,
        "source_agent_b_id": spec.source_agent_b_id,
        "canonical_original_index_pair": (
            None
            if spec.canonical_original_index_pair is None
            else list(spec.canonical_original_index_pair)
        ),
        "pair_conflict_event_count": spec.pair_conflict_event_count,
    }


def _candidate_eval_to_json(evaluation: MAPF9CandidateEvaluation) -> dict[str, object]:
    payload: dict[str, object] = {
        "candidate_id": evaluation.candidate_id,
        "agent_order": list(evaluation.agent_order),
        "source_agent_a_id": evaluation.source_agent_a_id,
        "source_agent_b_id": evaluation.source_agent_b_id,
        "canonical_original_index_pair": (
            None
            if evaluation.canonical_original_index_pair is None
            else list(evaluation.canonical_original_index_pair)
        ),
        "pair_conflict_event_count": evaluation.pair_conflict_event_count,
        "success": evaluation.success,
        "termination_reason": evaluation.termination_reason,
        "soc": evaluation.soc,
        "makespan": evaluation.makespan,
        "conflict_count": evaluation.conflict_count,
        "pp_time_ms": evaluation.pp_time_ms,
    }
    if evaluation.pp_search_metrics is not None:
        payload["pp_search_metrics"] = {
            "low_level_searches": evaluation.pp_search_metrics.low_level_searches,
            "agents_planned": evaluation.pp_search_metrics.agents_planned,
        }
    if evaluation.error_message is not None:
        payload["error_message"] = evaluation.error_message
    return payload


def _method_logical_runtime_ms(
    method: MAPF9Method,
    preprocessing: MAPF9PreprocessingTimings | None,
    method_result: MAPF9MethodResult,
) -> float | None:
    if preprocessing is None or method_result.method_timings is None:
        return None
    timings = method_result.method_timings
    if method == MAPF9Method.SPF:
        return mapf9_spf_logical_total_ms(preprocessing, timings)
    if method == MAPF9Method.CGLPS:
        return mapf9_cglps_logical_total_ms(preprocessing, timings)
    return mapf9_ubls_logical_total_ms(preprocessing, timings)


def _method_specific_runtime_ms(
    method: MAPF9Method,
    method_result: MAPF9MethodResult,
) -> float | None:
    timings = method_result.method_timings
    if timings is None:
        return None
    if method == MAPF9Method.SPF:
        return timings.baseline_spf_pp_time_ms
    if method == MAPF9Method.CGLPS:
        return (
            timings.cglps_candidate_ranking_time_ms
            + timings.cglps_additional_pp_time_ms
        )
    return (
        timings.ubls_candidate_generation_time_ms
        + timings.ubls_additional_pp_time_ms
    )


def build_records_from_integration_result(
    *,
    manifest: MAPFBenchmarkManifest,
    manifest_sha256: str,
    benchmark_instance: MAPFBenchmarkInstance,
    integration: MAPF9IntegrationResult,
    expected_a_i: int,
    wall_time_ms: float,
) -> tuple[MAPF9InstanceRecord, tuple[MAPF9MethodRecord, ...], MAPF9CandidateAuditRecord]:
    shared = integration.shared
    preprocessing = (
        None if shared is None else shared.preprocessing_timings
    )
    budget_match_ok = integration.actual_a_i == expected_a_i

    if shared is not None:
        pair_counts_json = json.dumps(
            {f"{left},{right}": count for (left, right), count in shared.pair_event_counts.items()}
        )
        max_degree = max(shared.inputs.degrees) if shared.inputs.degrees else 0
        spf_order_json = _json_int_tuple(shared.spf_order)
        independent_conflict_count = len(shared.inputs.conflicts)
        conflicting_agent_pair_count = shared.inputs.conflict_pair_count
    else:
        pair_counts_json = None
        max_degree = None
        spf_order_json = None
        independent_conflict_count = None
        conflicting_agent_pair_count = None

    spf_timings = integration.spf.method_timings
    cglps_timings = integration.cglps.method_timings
    ubls_timings = integration.ubls.method_timings

    instance_record = MAPF9InstanceRecord(
        map_name=config_map_name(manifest),
        catalogue_role=manifest.catalogue_role,
        catalogue_seed=manifest.seed,
        manifest_sha256=manifest_sha256,
        instance_id=benchmark_instance.instance_id,
        agent_count=benchmark_instance.agent_count,
        interaction_level=benchmark_instance.interaction_level.value,
        manifest_independent_conflict_count=benchmark_instance.independent_conflict_count,
        manifest_conflicting_agent_pair_count=benchmark_instance.conflicting_agent_pair_count,
        manifest_vertex_conflict_count=benchmark_instance.vertex_conflict_count,
        manifest_edge_conflict_count=benchmark_instance.edge_conflict_count,
        manifest_independent_soc=benchmark_instance.independent_soc,
        manifest_independent_makespan=benchmark_instance.independent_makespan,
        independent_conflict_count=independent_conflict_count,
        conflicting_agent_pair_count=conflicting_agent_pair_count,
        pair_event_counts_json=pair_counts_json,
        max_conflict_degree=max_degree,
        spf_order_json=spf_order_json,
        expected_a_i_from_manifest=expected_a_i,
        actual_a_i=integration.actual_a_i,
        physical_pp_eval_count=integration.physical_pp_eval_count,
        budget_match_ok=budget_match_ok,
        preprocessing_success=integration.ordering_failure is None,
        termination_reason=(
            TERMINATION_ORDERING_FAILURE
            if integration.ordering_failure is not None
            else integration.spf.termination_reason
        ),
        error_message=(
            None
            if integration.ordering_failure is None
            else integration.ordering_failure.error_message
        ),
        independent_path_time_ms=(
            None if preprocessing is None else preprocessing.independent_path_time_ms
        ),
        conflict_detection_time_ms=(
            None if preprocessing is None else preprocessing.conflict_detection_time_ms
        ),
        spf_ordering_time_ms=(
            None if preprocessing is None else preprocessing.spf_ordering_time_ms
        ),
        baseline_spf_pp_time_ms=(
            None if spf_timings is None else spf_timings.baseline_spf_pp_time_ms
        ),
        cglps_candidate_ranking_time_ms=(
            None if cglps_timings is None else cglps_timings.cglps_candidate_ranking_time_ms
        ),
        cglps_additional_pp_time_ms=(
            None if cglps_timings is None else cglps_timings.cglps_additional_pp_time_ms
        ),
        ubls_candidate_generation_time_ms=(
            None if ubls_timings is None else ubls_timings.ubls_candidate_generation_time_ms
        ),
        ubls_additional_pp_time_ms=(
            None if ubls_timings is None else ubls_timings.ubls_additional_pp_time_ms
        ),
        physical_instance_wall_time_ms=wall_time_ms,
    )

    method_records: list[MAPF9MethodRecord] = []
    for method_result in (integration.spf, integration.cglps, integration.ubls):
        if method_result.success and method_result.conflict_count not in (0, None):
            raise RuntimeError(
                f"successful {method_result.method.value} selected result has "
                f"conflict_count={method_result.conflict_count}"
            )
        method_records.append(
            MAPF9MethodRecord(
                manifest_sha256=manifest_sha256,
                map_name=config_map_name(manifest),
                catalogue_role=manifest.catalogue_role,
                catalogue_seed=manifest.seed,
                instance_id=benchmark_instance.instance_id,
                method=method_result.method.value,
                success=method_result.success,
                termination_reason=method_result.termination_reason,
                selected_candidate_id=method_result.selected_candidate_id,
                selected_agent_order_json=_json_int_tuple(
                    method_result.selected_agent_order
                ),
                soc=method_result.soc,
                makespan=method_result.makespan,
                final_conflict_count=method_result.conflict_count,
                additional_pp_eval_count=method_result.additional_pp_eval_count,
                logical_pp_candidate_count=method_result.total_logical_pp_candidate_count,
                logical_runtime_ms=_method_logical_runtime_ms(
                    method_result.method,
                    preprocessing,
                    method_result,
                ),
                baseline_spf_pp_time_ms=(
                    None
                    if method_result.method_timings is None
                    else method_result.method_timings.baseline_spf_pp_time_ms
                ),
                method_specific_runtime_ms=_method_specific_runtime_ms(
                    method_result.method,
                    method_result,
                ),
            )
        )

    spf_eval = integration.spf.candidate_evaluations[0]
    audit = MAPF9CandidateAuditRecord(
        map_name=config_map_name(manifest),
        catalogue_role=manifest.catalogue_role,
        catalogue_seed=manifest.seed,
        manifest_sha256=manifest_sha256,
        instance_id=benchmark_instance.instance_id,
        actual_a_i=integration.actual_a_i,
        expected_a_i_from_manifest=expected_a_i,
        physical_pp_eval_count=integration.physical_pp_eval_count,
        ubls_instance_seed_text=f"9031:{benchmark_instance.instance_id}",
        cglps_selected_candidate_id=integration.cglps.selected_candidate_id,
        ubls_selected_candidate_id=integration.ubls.selected_candidate_id,
        spf_candidate=_candidate_eval_to_json(spf_eval),
        cglps_additional_candidates=[
            _candidate_spec_to_json(spec)
            for spec in integration.cglps.additional_candidate_specs
        ],
        ubls_additional_candidates=[
            _candidate_spec_to_json(spec)
            for spec in integration.ubls.additional_candidate_specs
        ],
        cglps_candidate_evaluations=[
            _candidate_eval_to_json(evaluation)
            for evaluation in integration.cglps.candidate_evaluations
        ],
        ubls_candidate_evaluations=[
            _candidate_eval_to_json(evaluation)
            for evaluation in integration.ubls.candidate_evaluations
        ],
    )
    return instance_record, tuple(method_records), audit


def config_map_name(manifest: MAPFBenchmarkManifest) -> str:
    return Path(manifest.map_name).stem


def _dataclass_to_row(record: object) -> dict[str, object]:
    from dataclasses import asdict

    return asdict(record)


def rewrite_instance_csv(records: Iterable[MAPF9InstanceRecord], csv_path: Path) -> None:
    rows = [_dataclass_to_row(record) for record in records]
    _write_csv(rows, csv_path)


def rewrite_method_csv(records: Iterable[MAPF9MethodRecord], csv_path: Path) -> None:
    rows = [_dataclass_to_row(record) for record in records]
    _write_csv(rows, csv_path)


def _write_csv(rows: list[dict[str, object]], csv_path: Path) -> None:
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


def _audit_to_json(audit: MAPF9CandidateAuditRecord) -> dict[str, object]:
    return _dataclass_to_row(audit)


def _audit_from_json(data: dict[str, object]) -> MAPF9CandidateAuditRecord:
    return MAPF9CandidateAuditRecord(
        map_name=str(data["map_name"]),
        catalogue_role=str(data["catalogue_role"]),
        catalogue_seed=int(data["catalogue_seed"]),
        manifest_sha256=str(data["manifest_sha256"]),
        instance_id=str(data["instance_id"]),
        actual_a_i=int(data["actual_a_i"]),
        expected_a_i_from_manifest=int(data["expected_a_i_from_manifest"]),
        physical_pp_eval_count=int(data["physical_pp_eval_count"]),
        ubls_instance_seed_text=str(data["ubls_instance_seed_text"]),
        cglps_selected_candidate_id=int(data["cglps_selected_candidate_id"]),
        ubls_selected_candidate_id=int(data["ubls_selected_candidate_id"]),
        spf_candidate=dict(data["spf_candidate"]),  # type: ignore[arg-type]
        cglps_additional_candidates=list(data["cglps_additional_candidates"]),  # type: ignore[arg-type]
        ubls_additional_candidates=list(data["ubls_additional_candidates"]),  # type: ignore[arg-type]
        cglps_candidate_evaluations=list(data["cglps_candidate_evaluations"]),  # type: ignore[arg-type]
        ubls_candidate_evaluations=list(data["ubls_candidate_evaluations"]),  # type: ignore[arg-type]
    )


def load_checkpoint_audits(
    jsonl_path: Path,
) -> dict[MAPF9InstanceKey, MAPF9CandidateAuditRecord]:
    if not jsonl_path.is_file():
        return {}

    audits: dict[MAPF9InstanceKey, MAPF9CandidateAuditRecord] = {}
    duplicates: list[str] = []
    with jsonl_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            data = json.loads(stripped)
            if not isinstance(data, dict):
                raise ValueError(f"Invalid JSONL at {jsonl_path}:{line_number}")
            audit = _audit_from_json(data)
            key = instance_key(audit)
            if key in audits:
                duplicates.append(f"{key.instance_id} (line {line_number})")
            audits[key] = audit
    if duplicates:
        raise RuntimeError(
            "Duplicate MAPF-9 checkpoint instance keys:\n  - "
            + "\n  - ".join(duplicates)
        )
    return audits


class MAPF9PrimaryCheckpointStore:
    def __init__(
        self,
        *,
        config: MAPF9PrimaryExecutionConfig,
        reset_results: bool = False,
    ) -> None:
        self.config = config
        self.results_dir = config.results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

        if reset_results:
            for path in (
                config.instance_csv_path,
                config.method_csv_path,
                config.candidate_jsonl_path,
            ):
                path.unlink(missing_ok=True)

        self._audits = load_checkpoint_audits(config.candidate_jsonl_path)
        self._instance_records: dict[MAPF9InstanceKey, MAPF9InstanceRecord] = {}
        self._method_records: dict[MAPF9MethodKey, MAPF9MethodRecord] = {}
        self._load_csv_state()

    @property
    def audits(self) -> dict[MAPF9InstanceKey, MAPF9CandidateAuditRecord]:
        return dict(self._audits)

    @property
    def method_records(self) -> dict[MAPF9MethodKey, MAPF9MethodRecord]:
        return dict(self._method_records)

    def completed_instance_keys(self) -> set[MAPF9InstanceKey]:
        return set(self._audits)

    def append_complete_instance(
        self,
        instance_record: MAPF9InstanceRecord,
        method_records: Sequence[MAPF9MethodRecord],
        audit: MAPF9CandidateAuditRecord,
    ) -> None:
        key = instance_key(audit)
        if key in self._audits:
            raise RuntimeError(
                f"Refusing to overwrite checkpoint for instance {key.instance_id}"
            )
        if len(method_records) != 3:
            raise RuntimeError("complete instance must include exactly 3 method rows")

        line = json.dumps(_audit_to_json(audit), sort_keys=True)
        with self.config.candidate_jsonl_path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
            file.flush()
            os.fsync(file.fileno())

        self._audits[key] = audit
        self._instance_records[key] = instance_record
        for method_record in method_records:
            self._method_records[method_key(method_record)] = method_record

        ordered_instances = self._ordered_instance_records()
        ordered_methods = self._ordered_method_records()
        rewrite_instance_csv(ordered_instances, self.config.instance_csv_path)
        rewrite_method_csv(ordered_methods, self.config.method_csv_path)

    def _ordered_instance_records(self) -> tuple[MAPF9InstanceRecord, ...]:
        return tuple(
            self._instance_records[key]
            for key in sorted(
                self._instance_records,
                key=lambda item: item.instance_id,
            )
        )

    def _ordered_method_records(self) -> tuple[MAPF9MethodRecord, ...]:
        method_order = {method.value: index for index, method in enumerate(MAPF9_METHODS)}
        return tuple(
            self._method_records[key]
            for key in sorted(
                self._method_records,
                key=lambda item: (item.instance_id, method_order[item.method.value]),
            )
        )

    def _load_csv_state(self) -> None:
        if self.config.instance_csv_path.is_file():
            with self.config.instance_csv_path.open(encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    record = MAPF9InstanceRecord(
                        map_name=str(row["map_name"]),
                        catalogue_role=str(row["catalogue_role"]),
                        catalogue_seed=int(row["catalogue_seed"]),
                        manifest_sha256=str(row["manifest_sha256"]),
                        instance_id=str(row["instance_id"]),
                        agent_count=int(row["agent_count"]),
                        interaction_level=str(row["interaction_level"]),
                        manifest_independent_conflict_count=int(
                            row["manifest_independent_conflict_count"]
                        ),
                        manifest_conflicting_agent_pair_count=int(
                            row["manifest_conflicting_agent_pair_count"]
                        ),
                        manifest_vertex_conflict_count=int(
                            row["manifest_vertex_conflict_count"]
                        ),
                        manifest_edge_conflict_count=int(
                            row["manifest_edge_conflict_count"]
                        ),
                        manifest_independent_soc=int(row["manifest_independent_soc"]),
                        manifest_independent_makespan=int(
                            row["manifest_independent_makespan"]
                        ),
                        independent_conflict_count=(
                            None
                            if row.get("independent_conflict_count") in (None, "")
                            else int(row["independent_conflict_count"])
                        ),
                        conflicting_agent_pair_count=(
                            None
                            if row.get("conflicting_agent_pair_count") in (None, "")
                            else int(row["conflicting_agent_pair_count"])
                        ),
                        pair_event_counts_json=row.get("pair_event_counts_json") or None,
                        max_conflict_degree=(
                            None
                            if row.get("max_conflict_degree") in (None, "")
                            else int(row["max_conflict_degree"])
                        ),
                        spf_order_json=row.get("spf_order_json") or None,
                        expected_a_i_from_manifest=int(row["expected_a_i_from_manifest"]),
                        actual_a_i=int(row["actual_a_i"]),
                        physical_pp_eval_count=int(row["physical_pp_eval_count"]),
                        budget_match_ok=row["budget_match_ok"] in {"True", "true", "1"},
                        preprocessing_success=row["preprocessing_success"] in {
                            "True",
                            "true",
                            "1",
                        },
                        termination_reason=str(row["termination_reason"]),
                        error_message=row.get("error_message") or None,
                        independent_path_time_ms=_optional_float(
                            row.get("independent_path_time_ms")
                        ),
                        conflict_detection_time_ms=_optional_float(
                            row.get("conflict_detection_time_ms")
                        ),
                        spf_ordering_time_ms=_optional_float(row.get("spf_ordering_time_ms")),
                        baseline_spf_pp_time_ms=_optional_float(
                            row.get("baseline_spf_pp_time_ms")
                        ),
                        cglps_candidate_ranking_time_ms=_optional_float(
                            row.get("cglps_candidate_ranking_time_ms")
                        ),
                        cglps_additional_pp_time_ms=_optional_float(
                            row.get("cglps_additional_pp_time_ms")
                        ),
                        ubls_candidate_generation_time_ms=_optional_float(
                            row.get("ubls_candidate_generation_time_ms")
                        ),
                        ubls_additional_pp_time_ms=_optional_float(
                            row.get("ubls_additional_pp_time_ms")
                        ),
                        physical_instance_wall_time_ms=float(
                            row["physical_instance_wall_time_ms"]
                        ),
                    )
                    self._instance_records[instance_key(record)] = record

        if self.config.method_csv_path.is_file():
            with self.config.method_csv_path.open(encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    record = MAPF9MethodRecord(
                        manifest_sha256=str(row["manifest_sha256"]),
                        map_name=str(row["map_name"]),
                        catalogue_role=str(row["catalogue_role"]),
                        catalogue_seed=int(row["catalogue_seed"]),
                        instance_id=str(row["instance_id"]),
                        method=str(row["method"]),
                        success=row["success"] in {"True", "true", "1"},
                        termination_reason=str(row["termination_reason"]),
                        selected_candidate_id=int(row["selected_candidate_id"]),
                        selected_agent_order_json=str(row["selected_agent_order_json"]),
                        soc=_optional_int(row.get("soc")),
                        makespan=_optional_int(row.get("makespan")),
                        final_conflict_count=_optional_int(row.get("final_conflict_count")),
                        additional_pp_eval_count=int(row["additional_pp_eval_count"]),
                        logical_pp_candidate_count=int(row["logical_pp_candidate_count"]),
                        logical_runtime_ms=_optional_float(row.get("logical_runtime_ms")),
                        baseline_spf_pp_time_ms=_optional_float(
                            row.get("baseline_spf_pp_time_ms")
                        ),
                        method_specific_runtime_ms=_optional_float(
                            row.get("method_specific_runtime_ms")
                        ),
                    )
                    self._method_records[method_key(record)] = record


def _optional_float(value: object | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_int(value: object | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def validate_checkpoint_state(
    checkpoint: MAPF9PrimaryCheckpointStore,
    *,
    manifest: MAPFBenchmarkManifest,
    manifest_sha256: str,
    budget_plan: MAPF9BudgetPlan,
) -> None:
    allowed_instance_ids = {instance.instance_id for instance in manifest.instances}
    for key, audit in checkpoint.audits.items():
        if key.manifest_sha256 != manifest_sha256:
            raise ValueError(
                f"Checkpoint manifest hash mismatch for {key.instance_id}"
            )
        if audit.manifest_sha256 != manifest_sha256:
            raise ValueError(
                f"Audit body manifest hash mismatch for {key.instance_id}"
            )
        if key.instance_id not in allowed_instance_ids:
            raise ValueError(f"Unknown checkpoint instance_id: {key.instance_id}")
        if audit.actual_a_i != audit.expected_a_i_from_manifest:
            if audit.physical_pp_eval_count > 0:
                raise ValueError(
                    f"Checkpoint budget mismatch for {key.instance_id}: "
                    f"actual_a_i={audit.actual_a_i}, "
                    f"expected={audit.expected_a_i_from_manifest}"
                )
        expected_a_i = budget_plan.expected_a_i_by_instance_id[key.instance_id]
        if audit.expected_a_i_from_manifest != expected_a_i:
            raise ValueError(
                f"Checkpoint expected A_i mismatch for {key.instance_id}"
            )
        if audit.physical_pp_eval_count != 1 + 2 * audit.actual_a_i and audit.actual_a_i > 0:
            if audit.physical_pp_eval_count != 0:
                raise ValueError(
                    f"Invalid physical PP count for {key.instance_id}: "
                    f"{audit.physical_pp_eval_count}"
                )
        if len(audit.cglps_candidate_evaluations) != (
            0 if audit.actual_a_i == 0 and audit.physical_pp_eval_count == 0
            else 1 + audit.actual_a_i
        ):
            raise ValueError(
                f"CGLPS candidate audit count mismatch for {key.instance_id}"
            )
        if len(audit.ubls_candidate_evaluations) != (
            0 if audit.actual_a_i == 0 and audit.physical_pp_eval_count == 0
            else 1 + audit.actual_a_i
        ):
            raise ValueError(
                f"UBLS candidate audit count mismatch for {key.instance_id}"
            )

    methods_by_instance: dict[str, list[MAPF9MethodRecord]] = {}
    for method_key_item, method_record in checkpoint.method_records.items():
        if method_key_item.manifest_sha256 != manifest_sha256:
            raise ValueError(
                f"Method row manifest hash mismatch for {method_key_item.instance_id}"
            )
        methods_by_instance.setdefault(method_key_item.instance_id, []).append(
            method_record
        )

    for key in checkpoint.audits:
        instance_methods = methods_by_instance.get(key.instance_id, [])
        if len(instance_methods) != 3:
            raise ValueError(
                f"Completed instance {key.instance_id} has "
                f"{len(instance_methods)} method rows, expected 3"
            )
        methods_by_name = {record.method: record for record in instance_methods}
        if set(methods_by_name) != {method.value for method in MAPF9_METHODS}:
            raise ValueError(
                f"Completed instance {key.instance_id} missing required methods"
            )
        cglps = methods_by_name[MAPF9Method.CGLPS.value]
        ubls = methods_by_name[MAPF9Method.UBLS.value]
        if cglps.additional_pp_eval_count != ubls.additional_pp_eval_count:
            raise ValueError(
                f"CGLPS/UBLS A_i mismatch for {key.instance_id}: "
                f"CGLPS={cglps.additional_pp_eval_count}, "
                f"UBLS={ubls.additional_pp_eval_count}"
            )
        for method_record in instance_methods:
            if method_record.success and method_record.final_conflict_count not in (
                0,
                None,
            ):
                raise ValueError(
                    f"Successful {method_record.method} for {key.instance_id} "
                    f"has final_conflict_count={method_record.final_conflict_count}"
                )


def _count_by_agent(manifest: MAPFBenchmarkManifest) -> dict[int, int]:
    counts: dict[int, int] = {}
    for instance in manifest.instances:
        counts[instance.agent_count] = counts.get(instance.agent_count, 0) + 1
    return counts


def _count_by_interaction(manifest: MAPFBenchmarkManifest) -> dict[str, int]:
    counts: dict[str, int] = {}
    for instance in manifest.instances:
        level = instance.interaction_level.value
        counts[level] = counts.get(level, 0) + 1
    return counts


def print_execution_plan(
    config: MAPF9PrimaryExecutionConfig,
    manifest: MAPFBenchmarkManifest,
    manifest_sha256: str,
    budget_plan: MAPF9BudgetPlan,
    *,
    completed_instance_ids: Sequence[str],
    logger: logging.Logger,
) -> None:
    remaining = [
        instance.instance_id
        for instance in manifest.instances
        if instance.instance_id not in set(completed_instance_ids)
    ]
    logger.info("")
    logger.info("MAPF-9.4 PRIMARY EXECUTION PLAN")
    logger.info("")
    logger.info(f"Target map: {config.map_stem}")
    logger.info(f"Manifest path: {config.manifest_path.resolve()}")
    logger.info(f"Verified manifest SHA-256: {manifest_sha256}")
    logger.info(f"Catalogue seed: {manifest.seed}")
    logger.info(f"Catalogue role: {manifest.catalogue_role}")
    logger.info(f"Generation version: {manifest.generation_version}")
    logger.info(f"Instances: {len(manifest.instances)}")
    logger.info(f"Counts by agent_count: {_count_by_agent(manifest)}")
    logger.info(f"Counts by interaction: {_count_by_interaction(manifest)}")
    logger.info(f"Frozen B: {MAPF9_CGLPS_BUDGET}")
    logger.info(f"Expected sum A_i: {budget_plan.expected_sum_a_i}")
    logger.info(
        f"Expected physical PP evaluations: "
        f"{budget_plan.expected_physical_pp_eval_count}"
    )
    logger.info(f"Already completed instances: {len(completed_instance_ids)}")
    logger.info(f"Remaining instances: {len(remaining)}")
    logger.info(f"Output directory: {config.results_dir.resolve()}")
    logger.info(f"Resume: {config.resume}")
    logger.info(f"Reset results: {config.reset_results}")
    logger.info(f"Plan only: {config.plan_only}")
    logger.info("")


def _soc_improved(spf_soc: int | None, method_soc: int | None) -> bool:
    if spf_soc is None or method_soc is None:
        return False
    return method_soc < spf_soc


def run_mapf9_primary_benchmark(
    config: MAPF9PrimaryExecutionConfig,
    *,
    repo_root: Path,
    logger: logging.Logger | None = None,
    evaluate_instance: Callable[..., MAPF9IntegrationResult] = evaluate_mapf9_instance,
) -> int:
    if config.reset_results and config.resume:
        raise ValueError("reset_results=True cannot be combined with resume=True")
    if config.plan_only and config.reset_results:
        raise ValueError("plan_only=True cannot be combined with reset_results=True")

    assert_results_dir_isolated(config.results_dir, repo_root)
    manifest, manifest_sha256 = verify_frozen_mapf9_manifest(config)
    budget_plan = build_budget_plan(manifest)

    active_logger = logger
    if active_logger is None:
        active_logger = logging.getLogger(
            f"mapf9_primary_execution_{config.map_stem}"
        )
        active_logger.setLevel(logging.INFO)
        active_logger.handlers.clear()
        active_logger.propagate = False
        formatter = logging.Formatter("%(message)s")
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        active_logger.addHandler(stream_handler)
        if not config.plan_only:
            config.log_file_path.parent.mkdir(parents=True, exist_ok=True)
            append_log = (
                config.resume
                and config.log_file_path.is_file()
                and not config.reset_results
            )
            log_mode = "a" if append_log else "w"
            file_handler = logging.FileHandler(
                config.log_file_path,
                mode=log_mode,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            active_logger.addHandler(file_handler)

    checkpoint = MAPF9PrimaryCheckpointStore(
        config=config,
        reset_results=config.reset_results,
    )
    if checkpoint.audits:
        validate_checkpoint_state(
            checkpoint,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            budget_plan=budget_plan,
        )

    completed_ids = [key.instance_id for key in checkpoint.completed_instance_keys()]
    print_execution_plan(
        config,
        manifest,
        manifest_sha256,
        budget_plan,
        completed_instance_ids=completed_ids,
        logger=active_logger,
    )

    if config.plan_only:
        active_logger.info("PLAN-ONLY mode: no planner/preprocessing execution.")
        return 0

    grid_map = load_moving_ai_map(config.map_path)
    scenarios = load_moving_ai_scenarios(config.scen_path)
    if grid_map.name != manifest.map_name:
        raise ValueError(
            f"map name mismatch: manifest expects {manifest.map_name!r}, "
            f"loaded {grid_map.name!r}"
        )

    session_start = time.perf_counter()
    instances_completed = 0
    physical_pp_total = sum(
        audit.physical_pp_eval_count for audit in checkpoint.audits.values()
    )
    cglps_improved = 0
    ubls_improved = 0
    spf_success = 0
    cglps_success = 0
    ubls_success = 0

    for index, benchmark_instance in enumerate(manifest.instances, start=1):
        key = MAPF9InstanceKey(manifest_sha256, benchmark_instance.instance_id)
        if config.resume and key in checkpoint.completed_instance_keys():
            active_logger.info(
                f"SKIP — already completed: {benchmark_instance.instance_id}"
            )
            continue

        expected_a_i = budget_plan.expected_a_i_by_instance_id[
            benchmark_instance.instance_id
        ]
        active_logger.info(
            f"INSTANCE {index}/{len(manifest.instances)} | "
            f"{benchmark_instance.instance_id} | "
            f"n={benchmark_instance.agent_count} | "
            f"{benchmark_instance.interaction_level.value.upper()}"
        )

        scenario_bundle = reconstruct_mapf_scenario_from_instance(
            scenarios=scenarios,
            grid_map=grid_map,
            instance=benchmark_instance,
        )
        instance_start = time.perf_counter()
        try:
            integration = evaluate_instance(
                grid_map=grid_map,
                scenario=scenario_bundle.scenario,
                max_timestep=manifest.max_timestep,
                instance_id=benchmark_instance.instance_id,
            )
        except Exception as error:
            active_logger.error("UNEXPECTED ERROR — instance not checkpointed")
            active_logger.error(traceback.format_exc())
            raise RuntimeError(
                f"Unexpected error for {benchmark_instance.instance_id}: {error}"
            ) from error

        wall_time_ms = (time.perf_counter() - instance_start) * 1000.0
        instance_record, method_records, audit = build_records_from_integration_result(
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            benchmark_instance=benchmark_instance,
            integration=integration,
            expected_a_i=expected_a_i,
            wall_time_ms=wall_time_ms,
        )

        if (
            integration.ordering_failure is None
            and config.stop_on_budget_mismatch
            and not instance_record.budget_match_ok
        ):
            active_logger.error(
                "BUDGET_DIAGNOSTIC_MISMATCH "
                f"{benchmark_instance.instance_id}: "
                f"expected_a_i={expected_a_i}, actual_a_i={integration.actual_a_i}"
            )
            raise BudgetDiagnosticMismatchError(
                f"A_i mismatch on {benchmark_instance.instance_id}: "
                f"expected {expected_a_i}, actual {integration.actual_a_i}"
            )

        if integration.ordering_failure is None:
            if integration.cglps.additional_pp_eval_count != integration.actual_a_i:
                raise RuntimeError("CGLPS additional_pp_eval_count != actual_a_i")
            if integration.ubls.additional_pp_eval_count != integration.actual_a_i:
                raise RuntimeError("UBLS additional_pp_eval_count != actual_a_i")
            if not (0 <= integration.actual_a_i <= MAPF9_CGLPS_BUDGET):
                raise RuntimeError(f"actual_a_i out of range: {integration.actual_a_i}")

        checkpoint.append_complete_instance(instance_record, method_records, audit)
        instances_completed += 1
        physical_pp_total += integration.physical_pp_eval_count

        if integration.spf.success:
            spf_success += 1
        if integration.cglps.success:
            cglps_success += 1
        if integration.ubls.success:
            ubls_success += 1
        if _soc_improved(integration.spf.soc, integration.cglps.soc):
            cglps_improved += 1
        if _soc_improved(integration.spf.soc, integration.ubls.soc):
            ubls_improved += 1

        active_logger.info(
            f"  conflicts={benchmark_instance.independent_conflict_count} "
            f"pairs={benchmark_instance.conflicting_agent_pair_count} | "
            f"A_i={integration.actual_a_i} | PP physical={integration.physical_pp_eval_count}"
        )
        for method_result in (integration.spf, integration.cglps, integration.ubls):
            label = method_result.method.value.upper()
            if method_result.success:
                active_logger.info(
                    f"  {label:5s} SUCCESS SoC={method_result.soc} "
                    f"makespan={method_result.makespan} "
                    f"selected_candidate={method_result.selected_candidate_id}"
                )
            else:
                active_logger.info(
                    f"  {label:5s} FAILURE reason={method_result.termination_reason}"
                )
        active_logger.info(
            f"  elapsed={wall_time_ms:.1f}ms | checkpoint saved"
        )

        total_completed = len(checkpoint.completed_instance_keys())
        if total_completed % 5 == 0 or total_completed == len(manifest.instances):
            elapsed = time.perf_counter() - session_start
            active_logger.info(
                f"Cumulative: {total_completed}/{len(manifest.instances)} instances | "
                f"SPF success={spf_success} CGLPS success={cglps_success} "
                f"UBLS success={ubls_success} | "
                f"CGLPS improved vs SPF={cglps_improved} | "
                f"UBLS improved vs SPF={ubls_improved} | "
                f"physical PP evaluations={physical_pp_total} | "
                f"elapsed={elapsed:.1f}s"
            )

    active_logger.info("")
    active_logger.info("MAPF-9.4 PRIMARY EXECUTION COMPLETE")
    active_logger.info(
        f"Completed instances this session: {instances_completed} | "
        f"total checkpointed: {len(checkpoint.completed_instance_keys())}"
    )
    return 0
