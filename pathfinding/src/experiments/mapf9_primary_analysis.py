"""MAPF-9.6 formal primary analysis for frozen MAPF-9 production results."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pathfinding.src.experiments.mapf_benchmark_instances import (
    CATALOGUE_ROLE_MAPF9,
    MAPF9_CATALOGUE_SEEDS,
    MAPFBenchmarkManifest,
    load_benchmark_manifest,
)
from pathfinding.src.experiments.mapf_bounded_local_priority_search import (
    MAPF9_CGLPS_BUDGET,
    MAPF9_UBLS_GLOBAL_SEED,
    MAPF9MethodTimings,
    MAPF9PreprocessingTimings,
    mapf9_cglps_logical_total_ms,
    mapf9_spf_logical_total_ms,
    mapf9_ubls_logical_total_ms,
)
from pathfinding.src.experiments.mapf9_primary_execution import (
    MAPF9_FROZEN_MANIFEST_SHA256,
    expected_a_i_from_manifest_instance,
)
from pathfinding.src.experiments.mapf_cross_map_catalogue_validation import (
    manifest_file_sha256,
)

MAPF9_PRIMARY_ANALYSIS_DIR = Path("pathfinding/results/mapf9_primary_analysis")
MAPF9_PRIMARY_EXECUTION_ROOT = Path("pathfinding/results/mapf9_primary_execution")
MAPF9_BENCHMARKS_DIR = Path("pathfinding/results/mapf_benchmarks")

MAPF9_MAPS: tuple[str, ...] = ("AR0400SR", "AR0307SR")
MAPF9_METHODS: tuple[str, ...] = ("spf", "cglps", "ubls")
EXPECTED_INSTANCES_PER_MAP = 27
EXPECTED_METHOD_ROWS_PER_MAP = 81
EXPECTED_POOLED_INSTANCES = 54
EXPECTED_POOLED_METHOD_ROWS = 162
EXPECTED_POOLED_PHYSICAL_PP = 204
EXPECTED_SUM_A_I = {"AR0400SR": 33, "AR0307SR": 42, "pooled": 75}
EXPECTED_PHYSICAL_PP = {"AR0400SR": 93, "AR0307SR": 111, "pooled": 204}

RUNTIME_TOLERANCE_MS = 1.0
LexOutcome = Literal["left_better", "equal", "right_better"]


@dataclass(frozen=True, slots=True)
class MAPF9InstanceKey:
    manifest_sha256: str
    instance_id: str


@dataclass(frozen=True, slots=True)
class MAPF9MethodKey:
    manifest_sha256: str
    instance_id: str
    method: str


@dataclass(frozen=True, slots=True)
class MAPF9PrimaryAnalysisConfig:
    repo_root: Path
    output_dir: Path
    execution_root: Path
    benchmarks_dir: Path


@dataclass(frozen=True, slots=True)
class MethodOutcome:
    success: bool
    termination_reason: str
    selected_candidate_id: int
    soc: int | None
    makespan: int | None
    final_conflict_count: int | None
    additional_pp_eval_count: int
    logical_pp_candidate_count: int
    logical_runtime_ms: float | None
    baseline_spf_pp_time_ms: float | None
    method_specific_runtime_ms: float | None


@dataclass(frozen=True, slots=True)
class InstanceBundle:
    map_stem: str
    manifest_sha256: str
    catalogue_seed: int
    instance_id: str
    agent_count: int
    interaction_level: str
    preprocessing_success: bool
    actual_a_i: int
    expected_a_i: int
    physical_pp_eval_count: int
    budget_match_ok: bool
    manifest_independent_conflict_count: int
    manifest_conflicting_agent_pair_count: int
    max_conflict_degree: int | None
    independent_conflict_count: int | None
    conflicting_agent_pair_count: int | None
    independent_path_time_ms: float | None
    conflict_detection_time_ms: float | None
    spf_ordering_time_ms: float | None
    baseline_spf_pp_time_ms: float | None
    cglps_candidate_ranking_time_ms: float | None
    cglps_additional_pp_time_ms: float | None
    ubls_candidate_generation_time_ms: float | None
    ubls_additional_pp_time_ms: float | None
    physical_instance_wall_time_ms: float
    methods: dict[str, MethodOutcome]
    audit: dict[str, object]


@dataclass
class ValidationReport:
    lines: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def ok(self, message: str) -> None:
        self.lines.append(f"OK: {message}")

    def fail(self, message: str) -> None:
        self.errors.append(message)
        self.lines.append(f"FAIL: {message}")

    @property
    def passed(self) -> bool:
        return not self.errors


def default_mapf9_primary_analysis_config(repo_root: Path) -> MAPF9PrimaryAnalysisConfig:
    return MAPF9PrimaryAnalysisConfig(
        repo_root=repo_root,
        output_dir=repo_root / MAPF9_PRIMARY_ANALYSIS_DIR,
        execution_root=repo_root / MAPF9_PRIMARY_EXECUTION_ROOT,
        benchmarks_dir=repo_root / MAPF9_BENCHMARKS_DIR,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _optional_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_bool(value: str | None) -> bool:
    return value in {"True", "true", "1"}


def _load_audits(path: Path) -> dict[MAPF9InstanceKey, dict[str, object]]:
    audits: dict[MAPF9InstanceKey, dict[str, object]] = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            data = json.loads(stripped)
            if not isinstance(data, dict):
                raise ValueError(f"Invalid JSONL at {path}:{line_number}")
            key = MAPF9InstanceKey(
                manifest_sha256=str(data["manifest_sha256"]),
                instance_id=str(data["instance_id"]),
            )
            if key in audits:
                raise ValueError(f"Duplicate audit key {key.instance_id}")
            audits[key] = data
    return audits



def _method_outcome(row: dict[str, str]) -> MethodOutcome:
    return MethodOutcome(
        success=_optional_bool(row["success"]),
        termination_reason=str(row["termination_reason"]),
        selected_candidate_id=int(row["selected_candidate_id"]),
        soc=_optional_int(row.get("soc")),
        makespan=_optional_int(row.get("makespan")),
        final_conflict_count=_optional_int(row.get("final_conflict_count")),
        additional_pp_eval_count=int(row["additional_pp_eval_count"]),
        logical_pp_candidate_count=int(row["logical_pp_candidate_count"]),
        logical_runtime_ms=_optional_float(row.get("logical_runtime_ms")),
        baseline_spf_pp_time_ms=_optional_float(row.get("baseline_spf_pp_time_ms")),
        method_specific_runtime_ms=_optional_float(row.get("method_specific_runtime_ms")),
    )


def load_map_dataset(
    *,
    map_stem: str,
    execution_dir: Path,
    manifest: MAPFBenchmarkManifest,
    manifest_sha256: str,
) -> tuple[list[InstanceBundle], ValidationReport]:
    report = ValidationReport()
    instance_rows = _read_csv(execution_dir / "instance_results.csv")
    method_rows = _read_csv(execution_dir / "method_results.csv")
    audits = _load_audits(execution_dir / "candidate_details.jsonl")

    if len(instance_rows) != EXPECTED_INSTANCES_PER_MAP:
        report.fail(
            f"{map_stem}: expected {EXPECTED_INSTANCES_PER_MAP} instance rows, "
            f"found {len(instance_rows)}"
        )
    else:
        report.ok(f"{map_stem}: {len(instance_rows)} instance rows")

    if len(method_rows) != EXPECTED_METHOD_ROWS_PER_MAP:
        report.fail(
            f"{map_stem}: expected {EXPECTED_METHOD_ROWS_PER_MAP} method rows, "
            f"found {len(method_rows)}"
        )
    else:
        report.ok(f"{map_stem}: {len(method_rows)} method rows")

    if len(audits) != EXPECTED_INSTANCES_PER_MAP:
        report.fail(
            f"{map_stem}: expected {EXPECTED_INSTANCES_PER_MAP} audit objects, "
            f"found {len(audits)}"
        )
    else:
        report.ok(f"{map_stem}: {len(audits)} candidate audit objects")

    manifest_instances = {
        instance.instance_id: instance for instance in manifest.instances
    }
    methods_by_instance: dict[MAPF9InstanceKey, dict[str, MethodOutcome]] = defaultdict(dict)
    for row in method_rows:
        key = MAPF9MethodKey(
            manifest_sha256=row["manifest_sha256"],
            instance_id=row["instance_id"],
            method=row["method"],
        )
        if row["catalogue_role"] != CATALOGUE_ROLE_MAPF9:
            report.fail(f"{key.instance_id}/{key.method}: wrong catalogue_role")
        if int(row["catalogue_seed"]) != MAPF9_CATALOGUE_SEEDS[map_stem]:
            report.fail(f"{key.instance_id}/{key.method}: wrong catalogue_seed")
        if key.manifest_sha256 != manifest_sha256:
            report.fail(f"{key.instance_id}/{key.method}: manifest hash mismatch")
        instance_key = MAPF9InstanceKey(key.manifest_sha256, key.instance_id)
        if key.method in methods_by_instance[instance_key]:
            report.fail(f"Duplicate method row {key.instance_id}/{key.method}")
        methods_by_instance[instance_key][key.method] = _method_outcome(row)

    bundles: list[InstanceBundle] = []
    seen_instance_keys: set[MAPF9InstanceKey] = set()
    for row in instance_rows:
        key = MAPF9InstanceKey(row["manifest_sha256"], row["instance_id"])
        if key in seen_instance_keys:
            report.fail(f"Duplicate instance row {key.instance_id}")
        seen_instance_keys.add(key)

        if row["map_name"] != map_stem:
            report.fail(f"{key.instance_id}: map_name mismatch")
        if row["catalogue_role"] != CATALOGUE_ROLE_MAPF9:
            report.fail(f"{key.instance_id}: catalogue_role mismatch")
        if int(row["catalogue_seed"]) != MAPF9_CATALOGUE_SEEDS[map_stem]:
            report.fail(f"{key.instance_id}: catalogue_seed mismatch")
        if key.manifest_sha256 != manifest_sha256:
            report.fail(f"{key.instance_id}: manifest_sha256 mismatch")
        if key.instance_id not in manifest_instances:
            report.fail(f"{key.instance_id}: not in frozen manifest")

        method_map = methods_by_instance.get(key, {})
        if set(method_map) != set(MAPF9_METHODS):
            report.fail(
                f"{key.instance_id}: expected methods {MAPF9_METHODS}, "
                f"found {sorted(method_map)}"
            )
        if key not in audits:
            report.fail(f"{key.instance_id}: missing candidate audit")

        manifest_instance = manifest_instances[key.instance_id]
        expected_a_i = expected_a_i_from_manifest_instance(manifest_instance)
        actual_a_i = int(row["actual_a_i"])
        if int(row["expected_a_i_from_manifest"]) != expected_a_i:
            report.fail(f"{key.instance_id}: expected_a_i_from_manifest mismatch")
        if _optional_bool(row["preprocessing_success"]) and not _optional_bool(
            row["budget_match_ok"]
        ):
            report.fail(f"{key.instance_id}: budget_match_ok is False")
        if _optional_bool(row["preprocessing_success"]):
            if actual_a_i != expected_a_i:
                report.fail(
                    f"{key.instance_id}: actual_a_i={actual_a_i}, expected={expected_a_i}"
                )
            physical_pp = int(row["physical_pp_eval_count"])
            expected_physical = 1 + 2 * actual_a_i
            if physical_pp != expected_physical:
                report.fail(
                    f"{key.instance_id}: physical_pp={physical_pp}, "
                    f"expected={expected_physical}"
                )
            cglps = method_map["cglps"]
            ubls = method_map["ubls"]
            if cglps.additional_pp_eval_count != actual_a_i:
                report.fail(f"{key.instance_id}: CGLPS additional_pp != actual_a_i")
            if ubls.additional_pp_eval_count != actual_a_i:
                report.fail(f"{key.instance_id}: UBLS additional_pp != actual_a_i")

        for method_name, outcome in method_map.items():
            if outcome.success:
                if outcome.final_conflict_count not in (0, None):
                    report.fail(
                        f"{key.instance_id}/{method_name}: success with conflicts"
                    )
                if outcome.soc is None or outcome.makespan is None:
                    report.fail(
                        f"{key.instance_id}/{method_name}: success missing soc/makespan"
                    )
            elif outcome.soc is not None or outcome.makespan is not None:
                report.fail(
                    f"{key.instance_id}/{method_name}: failure with soc/makespan set"
                )

        _validate_candidate_audit(
            report,
            bundle_key=key,
            audit=audits[key],
            methods=method_map,
            spf_order_json=row.get("spf_order_json"),
        )

        bundles.append(
            InstanceBundle(
                map_stem=map_stem,
                manifest_sha256=key.manifest_sha256,
                catalogue_seed=int(row["catalogue_seed"]),
                instance_id=key.instance_id,
                agent_count=int(row["agent_count"]),
                interaction_level=str(row["interaction_level"]),
                preprocessing_success=_optional_bool(row["preprocessing_success"]),
                actual_a_i=actual_a_i,
                expected_a_i=expected_a_i,
                physical_pp_eval_count=int(row["physical_pp_eval_count"]),
                budget_match_ok=_optional_bool(row["budget_match_ok"]),
                manifest_independent_conflict_count=int(
                    row["manifest_independent_conflict_count"]
                ),
                manifest_conflicting_agent_pair_count=int(
                    row["manifest_conflicting_agent_pair_count"]
                ),
                max_conflict_degree=_optional_int(row.get("max_conflict_degree")),
                independent_conflict_count=_optional_int(
                    row.get("independent_conflict_count")
                ),
                conflicting_agent_pair_count=_optional_int(
                    row.get("conflicting_agent_pair_count")
                ),
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
                methods=method_map,
                audit=audits[key],
            )
        )

    manifest_keys = {
        MAPF9InstanceKey(manifest_sha256, instance.instance_id)
        for instance in manifest.instances
    }
    missing = manifest_keys - seen_instance_keys
    extra = seen_instance_keys - manifest_keys
    if missing:
        report.fail(f"{map_stem}: missing instances: {sorted(i.instance_id for i in missing)}")
    if extra:
        report.fail(f"{map_stem}: extra instances: {sorted(i.instance_id for i in extra)}")

    return bundles, report


def _parse_agent_order_json(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        return None
    return tuple(int(item) for item in parsed)


def _is_single_transposition(
    baseline: tuple[int, ...],
    candidate: tuple[int, ...],
) -> bool:
    if len(baseline) != len(candidate) or baseline == candidate:
        return False
    diff_indices = [index for index, (left, right) in enumerate(zip(baseline, candidate)) if left != right]
    if len(diff_indices) != 2:
        return False
    first, second = diff_indices
    return (
        baseline[first] == candidate[second]
        and baseline[second] == candidate[first]
        and baseline[:first] == candidate[:first]
        and baseline[first + 1 : second] == candidate[first + 1 : second]
        and baseline[second + 1 :] == candidate[second + 1 :]
    )


def _validate_candidate_audit(
    report: ValidationReport,
    *,
    bundle_key: MAPF9InstanceKey,
    audit: dict[str, object],
    methods: dict[str, MethodOutcome],
    spf_order_json: str | None,
) -> None:
    actual_a_i = int(audit["actual_a_i"])
    spf_eval = audit["spf_candidate"]
    if not isinstance(spf_eval, dict):
        report.fail(f"{bundle_key.instance_id}: invalid spf_candidate audit")
        return

    spf_order = tuple(int(item) for item in spf_eval["agent_order"])  # type: ignore[index]
    parsed_spf = _parse_agent_order_json(spf_order_json)
    if parsed_spf is not None and parsed_spf != spf_order:
        report.fail(f"{bundle_key.instance_id}: spf order mismatch vs instance row")

    for method_name in ("cglps", "ubls"):
        eval_key = f"{method_name}_candidate_evaluations"
        spec_key = f"{method_name}_additional_candidates"
        evaluations = audit.get(eval_key, [])
        specs = audit.get(spec_key, [])
        if not isinstance(evaluations, list) or not isinstance(specs, list):
            report.fail(f"{bundle_key.instance_id}: invalid {method_name} audit lists")
            continue

        if len(evaluations) != 1 + actual_a_i:
            report.fail(
                f"{bundle_key.instance_id}: {method_name} evaluation count "
                f"{len(evaluations)} != {1 + actual_a_i}"
            )
        if len(specs) != actual_a_i:
            report.fail(
                f"{bundle_key.instance_id}: {method_name} spec count "
                f"{len(specs)} != {actual_a_i}"
            )

        baseline_eval = evaluations[0]
        if not isinstance(baseline_eval, dict):
            report.fail(f"{bundle_key.instance_id}: invalid baseline eval")
            continue
        baseline_order = tuple(int(item) for item in baseline_eval["agent_order"])
        if baseline_order != spf_order:
            report.fail(f"{bundle_key.instance_id}: {method_name} candidate 0 != SPF")

        seen_orders: set[tuple[int, ...]] = {spf_order}
        for index, spec in enumerate(specs, start=1):
            if not isinstance(spec, dict):
                report.fail(f"{bundle_key.instance_id}: invalid {method_name} spec")
                continue
            candidate_id = int(spec["candidate_id"])
            if candidate_id != index:
                report.fail(
                    f"{bundle_key.instance_id}: {method_name} candidate_id {candidate_id} != {index}"
                )
            order = tuple(int(item) for item in spec["agent_order"])
            if order in seen_orders:
                report.fail(f"{bundle_key.instance_id}: duplicate {method_name} order")
            seen_orders.add(order)
            if not _is_single_transposition(spf_order, order):
                report.fail(
                    f"{bundle_key.instance_id}: {method_name} candidate {candidate_id} "
                    "is not a single transposition of SPF"
                )

        selected_id = int(audit[f"{method_name}_selected_candidate_id"])
        method_outcome = methods[method_name]
        if selected_id != method_outcome.selected_candidate_id:
            report.fail(
                f"{bundle_key.instance_id}: {method_name} selected_candidate_id mismatch"
            )
        selected_eval = next(
            (
                item
                for item in evaluations
                if isinstance(item, dict) and int(item["candidate_id"]) == selected_id
            ),
            None,
        )
        if not isinstance(selected_eval, dict):
            report.fail(f"{bundle_key.instance_id}: selected {method_name} eval missing")
        elif method_outcome.success:
            if int(selected_eval["soc"]) != method_outcome.soc:
                report.fail(f"{bundle_key.instance_id}: {method_name} selected soc mismatch")
            if int(selected_eval["makespan"]) != method_outcome.makespan:
                report.fail(
                    f"{bundle_key.instance_id}: {method_name} selected makespan mismatch"
                )

    expected_seed_text = f"{MAPF9_UBLS_GLOBAL_SEED}:{bundle_key.instance_id}"
    if str(audit.get("ubls_instance_seed_text")) != expected_seed_text:
        report.fail(f"{bundle_key.instance_id}: UBLS seed text mismatch")

    if actual_a_i > 0:
        cglps_specs = audit.get("cglps_additional_candidates", [])
        if isinstance(cglps_specs, list) and cglps_specs:
            ranked = []
            for spec in cglps_specs:
                if not isinstance(spec, dict):
                    continue
                pair = spec.get("canonical_original_index_pair")
                if not isinstance(pair, list) or len(pair) != 2:
                    report.fail(f"{bundle_key.instance_id}: invalid CGLPS pair metadata")
                    continue
                ranked.append(
                    (
                        -int(spec.get("pair_conflict_event_count") or 0),
                        int(pair[0]),
                        int(pair[1]),
                        int(spec["candidate_id"]),
                    )
                )
            if ranked != sorted(ranked):
                report.fail(
                    f"{bundle_key.instance_id}: CGLPS candidates not ranked by "
                    "(-pair_event_count, i, j)"
                )


def compare_lexicographic(
    left: MethodOutcome,
    right: MethodOutcome,
) -> LexOutcome:
    if left.success and not right.success:
        return "left_better"
    if right.success and not left.success:
        return "right_better"
    if not left.success and not right.success:
        return "equal"
    if left.soc is None or right.soc is None or left.makespan is None or right.makespan is None:
        return "equal"
    if left.soc < right.soc:
        return "left_better"
    if left.soc > right.soc:
        return "right_better"
    if left.makespan < right.makespan:
        return "left_better"
    if left.makespan > right.makespan:
        return "right_better"
    return "equal"


def soc_improvement(spf_soc: int, method_soc: int) -> int:
    return spf_soc - method_soc


def is_makespan_only_improvement(
    spf: MethodOutcome,
    method: MethodOutcome,
) -> bool:
    if not (spf.success and method.success):
        return False
    if spf.soc is None or method.soc is None:
        return False
    if spf.makespan is None or method.makespan is None:
        return False
    return method.soc == spf.soc and method.makespan < spf.makespan


def _reconstruct_logical_runtime(
    bundle: InstanceBundle,
    method_name: str,
) -> float | None:
    if (
        bundle.independent_path_time_ms is None
        or bundle.spf_ordering_time_ms is None
        or bundle.baseline_spf_pp_time_ms is None
    ):
        return None
    preprocessing = MAPF9PreprocessingTimings(
        independent_path_time_ms=bundle.independent_path_time_ms,
        conflict_detection_time_ms=bundle.conflict_detection_time_ms or 0.0,
        spf_ordering_time_ms=bundle.spf_ordering_time_ms,
    )
    method = bundle.methods[method_name]
    if method.logical_runtime_ms is None:
        return None
    if method_name == "spf":
        timings = MAPF9MethodTimings(
            baseline_spf_pp_time_ms=bundle.baseline_spf_pp_time_ms,
        )
        expected = mapf9_spf_logical_total_ms(preprocessing, timings)
    elif method_name == "cglps":
        timings = MAPF9MethodTimings(
            baseline_spf_pp_time_ms=bundle.baseline_spf_pp_time_ms,
            cglps_candidate_ranking_time_ms=bundle.cglps_candidate_ranking_time_ms or 0.0,
            cglps_additional_pp_time_ms=bundle.cglps_additional_pp_time_ms or 0.0,
        )
        expected = mapf9_cglps_logical_total_ms(preprocessing, timings)
    else:
        timings = MAPF9MethodTimings(
            baseline_spf_pp_time_ms=bundle.baseline_spf_pp_time_ms,
            ubls_candidate_generation_time_ms=bundle.ubls_candidate_generation_time_ms or 0.0,
            ubls_additional_pp_time_ms=bundle.ubls_additional_pp_time_ms or 0.0,
        )
        expected = mapf9_ubls_logical_total_ms(preprocessing, timings)
    return expected


def _parse_run_log(path: Path) -> dict[str, float | int]:
    text = path.read_text(encoding="utf-8")
    physical_pp = None
    elapsed_s = None
    for line in reversed(text.splitlines()):
        if "physical PP evaluations=" in line:
            match = re.search(r"physical PP evaluations=(\d+)", line)
            if match:
                physical_pp = int(match.group(1))
        if "elapsed=" in line and "Cumulative:" in line:
            match = re.search(r"elapsed=([\d.]+)s", line)
            if match:
                elapsed_s = float(match.group(1))
                break
    if elapsed_s is None:
        match = re.search(r"elapsed=([\d.]+)s", text)
        if match:
            elapsed_s = float(match.group(1))
    return {
        "physical_pp_evaluations": physical_pp if physical_pp is not None else -1,
        "elapsed_s": elapsed_s if elapsed_s is not None else -1.0,
    }


def _summary_stats(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "sum": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "sum": sum(numeric),
        "mean": statistics.mean(numeric),
        "median": statistics.median(numeric),
        "min": min(numeric),
        "max": max(numeric),
    }


def _count_lex(outcomes: Iterable[LexOutcome]) -> dict[str, int]:
    counter = Counter(outcomes)
    return {
        "left_better": counter["left_better"],
        "equal": counter["equal"],
        "right_better": counter["right_better"],
    }


def build_instance_level_rows(bundles: Sequence[InstanceBundle]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bundle in bundles:
        spf = bundle.methods["spf"]
        cglps = bundle.methods["cglps"]
        ubls = bundle.methods["ubls"]
        common_success = spf.success and cglps.success and ubls.success
        cglps_soc_delta = (
            soc_improvement(spf.soc, cglps.soc)
            if common_success and spf.soc is not None and cglps.soc is not None
            else None
        )
        ubls_soc_delta = (
            soc_improvement(spf.soc, ubls.soc)
            if common_success and spf.soc is not None and ubls.soc is not None
            else None
        )
        cglps_lex = compare_lexicographic(cglps, spf)
        ubls_lex = compare_lexicographic(ubls, spf)
        cglps_vs_ubls_lex = compare_lexicographic(cglps, ubls)
        cglps_vs_ubls_soc = (
            cglps.soc - ubls.soc
            if common_success and cglps.soc is not None and ubls.soc is not None
            else None
        )
        rows.append(
            {
                "manifest_sha256": bundle.manifest_sha256,
                "map_name": bundle.map_stem,
                "catalogue_seed": bundle.catalogue_seed,
                "instance_id": bundle.instance_id,
                "agent_count": bundle.agent_count,
                "interaction_level": bundle.interaction_level,
                "actual_a_i": bundle.actual_a_i,
                "search_active": bundle.actual_a_i > 0,
                "manifest_independent_conflict_count": bundle.manifest_independent_conflict_count,
                "manifest_conflicting_agent_pair_count": bundle.manifest_conflicting_agent_pair_count,
                "max_conflict_degree": bundle.max_conflict_degree,
                "spf_success": spf.success,
                "cglps_success": cglps.success,
                "ubls_success": ubls.success,
                "recovered_by_cglps": (not spf.success) and cglps.success,
                "recovered_by_ubls": (not spf.success) and ubls.success,
                "unrecovered": (not spf.success) and (not cglps.success) and (not ubls.success),
                "spf_soc": spf.soc,
                "spf_makespan": spf.makespan,
                "cglps_soc": cglps.soc,
                "cglps_makespan": cglps.makespan,
                "ubls_soc": ubls.soc,
                "ubls_makespan": ubls.makespan,
                "cglps_selected_candidate_id": cglps.selected_candidate_id,
                "ubls_selected_candidate_id": ubls.selected_candidate_id,
                "cglps_soc_improvement_vs_spf": cglps_soc_delta,
                "ubls_soc_improvement_vs_spf": ubls_soc_delta,
                "cglps_makespan_only_vs_spf": is_makespan_only_improvement(spf, cglps),
                "ubls_makespan_only_vs_spf": is_makespan_only_improvement(spf, ubls),
                "cglps_lex_vs_spf": cglps_lex,
                "ubls_lex_vs_spf": ubls_lex,
                "cglps_vs_ubls_soc_diff": cglps_vs_ubls_soc,
                "cglps_vs_ubls_lex": cglps_vs_ubls_lex,
                "spf_logical_runtime_ms": spf.logical_runtime_ms,
                "cglps_logical_runtime_ms": cglps.logical_runtime_ms,
                "ubls_logical_runtime_ms": ubls.logical_runtime_ms,
                "physical_instance_wall_time_ms": bundle.physical_instance_wall_time_ms,
            }
        )
    rows.sort(key=lambda row: (str(row["map_name"]), str(row["instance_id"])))
    return rows


def _soc_comparison_summary(
    bundles: Sequence[InstanceBundle],
    *,
    method_name: str,
    scope: str,
) -> dict[str, object]:
    spf = [bundle.methods["spf"] for bundle in bundles]
    method = [bundle.methods[method_name] for bundle in bundles]
    common = [
        (left, right)
        for left, right in zip(spf, method)
        if left.success and right.success and left.soc is not None and right.soc is not None
    ]
    deltas = [soc_improvement(left.soc, right.soc) for left, right in common]  # type: ignore[arg-type]
    improved = [delta for delta in deltas if delta > 0]
    equal = [delta for delta in deltas if delta == 0]
    worse = [delta for delta in deltas if delta < 0]
    makespan_only = sum(
        1
        for bundle in bundles
        if is_makespan_only_improvement(bundle.methods["spf"], bundle.methods[method_name])
    )
    return {
        "scope": scope,
        "comparison": f"{method_name.upper()}_vs_SPF",
        "common_success_instances": len(common),
        "soc_improved_count": len(improved),
        "soc_equal_count": len(equal),
        "soc_structurally_worse_count": len(worse),
        "makespan_only_improved_count": makespan_only,
        "best_soc_improvement": max(improved) if improved else 0,
        "total_soc_improvement": sum(improved),
        "mean_soc_improvement_all_common": statistics.mean(deltas) if deltas else None,
        "median_soc_improvement_all_common": statistics.median(deltas) if deltas else None,
        "mean_soc_improvement_improved_only": statistics.mean(improved) if improved else None,
    }


def _lex_comparison_summary(
    bundles: Sequence[InstanceBundle],
    *,
    left_method: str,
    right_method: str,
    scope: str,
) -> dict[str, object]:
    outcomes = [
        compare_lexicographic(bundle.methods[left_method], bundle.methods[right_method])
        for bundle in bundles
    ]
    counts = _count_lex(outcomes)
    return {
        "scope": scope,
        "comparison": f"{left_method.upper()}_vs_{right_method.upper()}",
        "left_better_count": counts["left_better"],
        "equal_count": counts["equal"],
        "right_better_count": counts["right_better"],
    }


def _unique_benefit_counts(
    bundles: Sequence[InstanceBundle],
    *,
    beneficiary: str,
    other: str,
    scope: str,
) -> dict[str, object]:
    soc_unique = 0
    lex_unique = 0
    for bundle in bundles:
        spf = bundle.methods["spf"]
        left = bundle.methods[beneficiary]
        right = bundle.methods[other]
        if not (spf.success and left.success):
            continue
        left_soc_delta = (
            soc_improvement(spf.soc, left.soc)
            if left.soc is not None and spf.soc is not None
            else 0
        )
        right_soc_delta = (
            soc_improvement(spf.soc, right.soc)
            if right.success and right.soc is not None and spf.soc is not None
            else 0
        )
        if left_soc_delta > 0 and right_soc_delta <= 0:
            soc_unique += 1
        left_lex = compare_lexicographic(left, spf)
        right_lex = compare_lexicographic(right, spf) if right.success else "equal"
        if left_lex == "left_better" and right_lex != "left_better":
            lex_unique += 1
    return {
        "scope": scope,
        "beneficiary": beneficiary.upper(),
        "other_method": other.upper(),
        "unique_soc_advantage_count": soc_unique,
        "unique_lex_advantage_count": lex_unique,
    }


def build_improved_instances_rows(bundles: Sequence[InstanceBundle]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bundle in bundles:
        spf = bundle.methods["spf"]
        for method_name in ("cglps", "ubls"):
            method = bundle.methods[method_name]
            if method.selected_candidate_id == 0:
                continue
            if not spf.success or not method.success:
                continue
            soc_delta = (
                soc_improvement(spf.soc, method.soc)
                if spf.soc is not None and method.soc is not None
                else 0
            )
            lex = compare_lexicographic(method, spf)
            makespan_only = is_makespan_only_improvement(spf, method)
            if soc_delta <= 0 and not makespan_only and lex != "left_better":
                continue
            rows.append(
                {
                    "manifest_sha256": bundle.manifest_sha256,
                    "map_name": bundle.map_stem,
                    "instance_id": bundle.instance_id,
                    "method": method_name,
                    "selected_candidate_id": method.selected_candidate_id,
                    "spf_soc": spf.soc,
                    "method_soc": method.soc,
                    "soc_improvement_vs_spf": soc_delta,
                    "spf_makespan": spf.makespan,
                    "method_makespan": method.makespan,
                    "makespan_only": makespan_only,
                    "lex_vs_spf": lex,
                    "actual_a_i": bundle.actual_a_i,
                    "manifest_conflicting_agent_pair_count": bundle.manifest_conflicting_agent_pair_count,
                    "manifest_independent_conflict_count": bundle.manifest_independent_conflict_count,
                }
            )
    rows.sort(key=lambda row: (str(row["map_name"]), str(row["instance_id"]), str(row["method"])))
    return rows


def build_candidate_selection_rows(bundles: Sequence[InstanceBundle]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bundle in bundles:
        for method_name in ("cglps", "ubls"):
            method = bundle.methods[method_name]
            audit = bundle.audit
            selected_id = method.selected_candidate_id
            baseline_count = 1 if selected_id == 0 else 0
            non_baseline_count = 0 if selected_id == 0 else 1
            winning_rank = None
            winning_pair = None
            winning_pair_events = None
            top_rank_winner = None
            if method_name == "cglps" and selected_id > 0:
                specs = audit.get("cglps_additional_candidates", [])
                if isinstance(specs, list):
                    for index, spec in enumerate(specs, start=1):
                        if isinstance(spec, dict) and int(spec["candidate_id"]) == selected_id:
                            winning_rank = index
                            pair = spec.get("canonical_original_index_pair")
                            winning_pair = tuple(pair) if isinstance(pair, list) else None
                            winning_pair_events = spec.get("pair_conflict_event_count")
                            top_rank_winner = winning_rank == 1
            rows.append(
                {
                    "manifest_sha256": bundle.manifest_sha256,
                    "map_name": bundle.map_stem,
                    "instance_id": bundle.instance_id,
                    "method": method_name,
                    "actual_a_i": bundle.actual_a_i,
                    "selected_candidate_id": selected_id,
                    "selected_baseline": baseline_count,
                    "selected_non_baseline": non_baseline_count,
                    "winning_candidate_rank": winning_rank,
                    "winning_canonical_pair": winning_pair,
                    "winning_pair_conflict_event_count": winning_pair_events,
                    "top_ranked_pair_selected": top_rank_winner,
                }
            )
    return rows


def build_candidate_level_rows(bundles: Sequence[InstanceBundle]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bundle in bundles:
        audit = bundle.audit
        cglps_orders = {
            tuple(int(x) for x in item["agent_order"])
            for item in audit.get("cglps_additional_candidates", [])
            if isinstance(item, dict)
        }
        for method_name in ("cglps", "ubls"):
            evals = audit.get(f"{method_name}_candidate_evaluations", [])
            if not isinstance(evals, list):
                continue
            for evaluation in evals:
                if not isinstance(evaluation, dict):
                    continue
                order = tuple(int(x) for x in evaluation["agent_order"])
                pair = evaluation.get("canonical_original_index_pair")
                rows.append(
                    {
                        "manifest_sha256": bundle.manifest_sha256,
                        "map_name": bundle.map_stem,
                        "instance_id": bundle.instance_id,
                        "method": method_name,
                        "candidate_id": int(evaluation["candidate_id"]),
                        "agent_order": list(order),
                        "success": evaluation.get("success"),
                        "soc": evaluation.get("soc"),
                        "makespan": evaluation.get("makespan"),
                        "pair_conflict_event_count": evaluation.get("pair_conflict_event_count"),
                        "matches_cglps_candidate_order": order in cglps_orders,
                        "selected": int(evaluation["candidate_id"])
                        == bundle.methods[method_name].selected_candidate_id,
                    }
                )
    return rows


def build_conflict_structure_rows(
    instance_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    groups = [
        ("cglps_soc_improved", lambda row: (row.get("cglps_soc_improvement_vs_spf") or 0) > 0),
        ("cglps_soc_not_improved", lambda row: (row.get("cglps_soc_improvement_vs_spf") or 0) <= 0),
        ("cglps_lex_improved", lambda row: row.get("cglps_lex_vs_spf") == "left_better"),
        ("cglps_lex_not_improved", lambda row: row.get("cglps_lex_vs_spf") != "left_better"),
    ]
    metrics = [
        "manifest_independent_conflict_count",
        "manifest_conflicting_agent_pair_count",
        "max_conflict_degree",
        "actual_a_i",
    ]
    output: list[dict[str, object]] = []
    for group_name, predicate in groups:
        subset = [row for row in instance_rows if predicate(row)]
        for metric in metrics:
            values = [
                float(row[metric])
                for row in subset
                if row.get(metric) is not None
            ]
            stats = _summary_stats(values)
            output.append(
                {
                    "group": group_name,
                    "metric": metric,
                    "instance_count": len(subset),
                    **stats,
                }
            )
    for interaction in ("low", "medium", "high"):
        subset = [row for row in instance_rows if row["interaction_level"] == interaction]
        output.append(
            {
                "group": f"interaction_{interaction}",
                "metric": "instance_count",
                "instance_count": len(subset),
                "count": len(subset),
                "sum": len(subset),
                "mean": len(subset),
                "median": len(subset),
                "min": len(subset),
                "max": len(subset),
            }
        )
    for agent_count in (5, 10, 20):
        subset = [row for row in instance_rows if row["agent_count"] == agent_count]
        output.append(
            {
                "group": f"agent_count_{agent_count}",
                "metric": "instance_count",
                "instance_count": len(subset),
                "count": len(subset),
                "sum": len(subset),
                "mean": len(subset),
                "median": len(subset),
                "min": len(subset),
                "max": len(subset),
            }
        )
    return output


def build_runtime_rows(
    bundles: Sequence[InstanceBundle],
    *,
    scope: str,
    run_log_stats: Mapping[str, float | int] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method_name in MAPF9_METHODS:
        logical_values = [
            bundle.methods[method_name].logical_runtime_ms
            for bundle in bundles
            if bundle.methods[method_name].logical_runtime_ms is not None
        ]
        stats = _summary_stats([value for value in logical_values if value is not None])
        rows.append(
            {
                "scope": scope,
                "method": method_name,
                "metric": "logical_runtime_ms",
                **stats,
            }
        )
    if scope != "pooled" and run_log_stats is not None:
        rows.append(
            {
                "scope": scope,
                "method": "physical_execution",
                "metric": "run_log_elapsed_s",
                "count": 1,
                "sum": run_log_stats["elapsed_s"],
                "mean": run_log_stats["elapsed_s"],
                "median": run_log_stats["elapsed_s"],
                "min": run_log_stats["elapsed_s"],
                "max": run_log_stats["elapsed_s"],
            }
        )
        rows.append(
            {
                "scope": scope,
                "method": "physical_execution",
                "metric": "physical_pp_evaluations",
                "count": 1,
                "sum": run_log_stats["physical_pp_evaluations"],
                "mean": run_log_stats["physical_pp_evaluations"],
                "median": run_log_stats["physical_pp_evaluations"],
                "min": run_log_stats["physical_pp_evaluations"],
                "max": run_log_stats["physical_pp_evaluations"],
            }
        )
        wall_sum = sum(bundle.physical_instance_wall_time_ms for bundle in bundles)
        rows.append(
            {
                "scope": scope,
                "method": "physical_execution",
                "metric": "sum_instance_wall_time_ms",
                "count": len(bundles),
                "sum": wall_sum,
                "mean": wall_sum / len(bundles) if bundles else None,
                "median": statistics.median(
                    [bundle.physical_instance_wall_time_ms for bundle in bundles]
                )
                if bundles
                else None,
                "min": min(bundle.physical_instance_wall_time_ms for bundle in bundles)
                if bundles
                else None,
                "max": max(bundle.physical_instance_wall_time_ms for bundle in bundles)
                if bundles
                else None,
            }
        )
    spf_values = [
        bundle.methods["spf"].logical_runtime_ms
        for bundle in bundles
        if bundle.methods["spf"].logical_runtime_ms
    ]
    for method_name in ("cglps", "ubls"):
        ratios = []
        extras = []
        for bundle in bundles:
            spf_runtime = bundle.methods["spf"].logical_runtime_ms
            method_runtime = bundle.methods[method_name].logical_runtime_ms
            if spf_runtime and method_runtime and spf_runtime > 0:
                ratios.append(method_runtime / spf_runtime)
                extras.append(method_runtime - spf_runtime)
        ratio_stats = _summary_stats(ratios)
        extra_stats = _summary_stats(extras)
        rows.append(
            {
                "scope": scope,
                "method": method_name,
                "metric": "logical_runtime_ratio_vs_spf",
                **ratio_stats,
            }
        )
        rows.append(
            {
                "scope": scope,
                "method": method_name,
                "metric": "logical_runtime_extra_ms_vs_spf",
                **extra_stats,
            }
        )
    return rows


def build_budget_rows(bundles: Sequence[InstanceBundle], *, scope: str) -> list[dict[str, object]]:
    counts = Counter(bundle.actual_a_i for bundle in bundles)
    rows: list[dict[str, object]] = []
    for value in range(MAPF9_CGLPS_BUDGET + 1):
        rows.append(
            {
                "scope": scope,
                "actual_a_i": value,
                "instance_count": counts.get(value, 0),
            }
        )
    ai_values = [bundle.actual_a_i for bundle in bundles]
    rows.append(
        {
            "scope": scope,
            "metric": "sum_actual_a_i",
            "value": sum(ai_values),
        }
    )
    rows.append(
        {
            "scope": scope,
            "metric": "mean_actual_a_i",
            "value": statistics.mean(ai_values) if ai_values else None,
        }
    )
    rows.append(
        {
            "scope": scope,
            "metric": "median_actual_a_i",
            "value": statistics.median(ai_values) if ai_values else None,
        }
    )
    rows.append(
        {
            "scope": scope,
            "metric": "fraction_a_i_zero",
            "value": sum(1 for value in ai_values if value == 0) / len(ai_values)
            if ai_values
            else None,
        }
    )
    rows.append(
        {
            "scope": scope,
            "metric": "fraction_a_i_at_budget_cap",
            "value": sum(1 for value in ai_values if value == MAPF9_CGLPS_BUDGET)
            / len(ai_values)
            if ai_values
            else None,
        }
    )
    rows.append(
        {
            "scope": scope,
            "metric": "physical_pp_eval_count",
            "value": sum(bundle.physical_pp_eval_count for bundle in bundles),
        }
    )
    return rows


def build_cost_benefit_rows(
    bundles: Sequence[InstanceBundle],
    *,
    scope: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    spf_logical = sum(
        bundle.methods["spf"].logical_runtime_ms or 0.0 for bundle in bundles
    )
    for method_name in MAPF9_METHODS:
        logical_pp_candidates = sum(
            bundle.methods[method_name].logical_pp_candidate_count for bundle in bundles
        )
        logical_runtime = sum(
            bundle.methods[method_name].logical_runtime_ms or 0.0 for bundle in bundles
        )
        soc_summary = _soc_comparison_summary(bundles, method_name=method_name, scope=scope)
        rows.append(
            {
                "scope": scope,
                "method": method_name.upper(),
                "logical_pp_candidates": logical_pp_candidates,
                "logical_runtime_ms_total": logical_runtime,
                "logical_pp_ratio_vs_spf": (
                    logical_pp_candidates / len(bundles) if method_name == "spf" else None
                ),
                "logical_runtime_ratio_vs_spf": (
                    logical_runtime / spf_logical if spf_logical > 0 else None
                ),
                "soc_improved_instances": soc_summary["soc_improved_count"],
                "makespan_only_improved_instances": soc_summary["makespan_only_improved_count"],
                "total_soc_improvement": soc_summary["total_soc_improvement"],
            }
        )
    return rows


def validate_and_load_all(
    config: MAPF9PrimaryAnalysisConfig,
) -> tuple[dict[str, list[InstanceBundle]], ValidationReport, dict[str, dict[str, float | int]]]:
    combined_report = ValidationReport()
    datasets: dict[str, list[InstanceBundle]] = {}
    run_logs: dict[str, dict[str, float | int]] = {}

    for map_stem in MAPF9_MAPS:
        manifest_path = config.benchmarks_dir / f"{map_stem}_mapf9_manifest.json"
        expected_sha = MAPF9_FROZEN_MANIFEST_SHA256[map_stem]
        actual_sha = manifest_file_sha256(manifest_path)
        if actual_sha != expected_sha:
            combined_report.fail(
                f"{map_stem}: manifest SHA mismatch expected {expected_sha}, found {actual_sha}"
            )
        else:
            combined_report.ok(f"{map_stem}: manifest SHA verified")

        manifest = load_benchmark_manifest(manifest_path)
        execution_dir = config.execution_root / map_stem
        bundles, report = load_map_dataset(
            map_stem=map_stem,
            execution_dir=execution_dir,
            manifest=manifest,
            manifest_sha256=actual_sha,
        )
        combined_report.lines.extend(report.lines)
        combined_report.errors.extend(report.errors)
        datasets[map_stem] = bundles
        run_logs[map_stem] = _parse_run_log(execution_dir / "run.log")

        sum_a_i = sum(bundle.actual_a_i for bundle in bundles)
        physical_pp = sum(bundle.physical_pp_eval_count for bundle in bundles)
        if sum_a_i != EXPECTED_SUM_A_I[map_stem]:
            combined_report.fail(
                f"{map_stem}: sum A_i={sum_a_i}, expected {EXPECTED_SUM_A_I[map_stem]}"
            )
        elif physical_pp != EXPECTED_PHYSICAL_PP[map_stem]:
            combined_report.fail(
                f"{map_stem}: physical PP={physical_pp}, expected {EXPECTED_PHYSICAL_PP[map_stem]}"
            )
        else:
            combined_report.ok(
                f"{map_stem}: sum A_i={sum_a_i}, physical PP={physical_pp}"
            )

        if run_logs[map_stem]["physical_pp_evaluations"] != EXPECTED_PHYSICAL_PP[map_stem]:
            combined_report.fail(
                f"{map_stem}: run.log physical PP mismatch "
                f"{run_logs[map_stem]['physical_pp_evaluations']}"
            )

        for bundle in bundles:
            for method_name in MAPF9_METHODS:
                expected_logical = _reconstruct_logical_runtime(bundle, method_name)
                recorded = bundle.methods[method_name].logical_runtime_ms
                if expected_logical is None or recorded is None:
                    continue
                if abs(expected_logical - recorded) > RUNTIME_TOLERANCE_MS:
                    combined_report.fail(
                        f"{bundle.instance_id}/{method_name}: logical runtime mismatch "
                        f"recorded={recorded}, expected={expected_logical}"
                    )

    pooled = [bundle for bundles in datasets.values() for bundle in bundles]
    if len(pooled) != EXPECTED_POOLED_INSTANCES:
        combined_report.fail(
            f"Pooled instances={len(pooled)}, expected {EXPECTED_POOLED_INSTANCES}"
        )
    else:
        combined_report.ok(f"Pooled instances={len(pooled)}")

    pooled_sum_a_i = sum(bundle.actual_a_i for bundle in pooled)
    pooled_physical = sum(bundle.physical_pp_eval_count for bundle in pooled)
    if pooled_sum_a_i != EXPECTED_SUM_A_I["pooled"]:
        combined_report.fail(f"Pooled sum A_i={pooled_sum_a_i}")
    elif pooled_physical != EXPECTED_PHYSICAL_PP["pooled"]:
        combined_report.fail(f"Pooled physical PP={pooled_physical}")
    else:
        combined_report.ok(
            f"Pooled sum A_i={pooled_sum_a_i}, physical PP={pooled_physical}"
        )

    return datasets, combined_report, run_logs


def build_method_summary_rows(
    datasets: Mapping[str, Sequence[InstanceBundle]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, Sequence[InstanceBundle]]] = [
        (map_stem, datasets[map_stem]) for map_stem in MAPF9_MAPS
    ]
    pooled = [bundle for bundles in datasets.values() for bundle in bundles]
    scopes.append(("pooled", pooled))
    active = [bundle for bundle in pooled if bundle.actual_a_i > 0]
    scopes.append(("search_active", active))

    for scope, bundles in scopes:
        for method_name in MAPF9_METHODS:
            successes = sum(1 for bundle in bundles if bundle.methods[method_name].success)
            rows.append(
                {
                    "scope": scope,
                    "method": method_name.upper(),
                    "instance_count": len(bundles),
                    "success_count": successes,
                    "failure_count": len(bundles) - successes,
                    "logical_pp_candidates": sum(
                        bundle.methods[method_name].logical_pp_candidate_count
                        for bundle in bundles
                    ),
                    "additional_pp_eval_count_total": sum(
                        bundle.methods[method_name].additional_pp_eval_count
                        for bundle in bundles
                    ),
                    "logical_runtime_ms_total": sum(
                        bundle.methods[method_name].logical_runtime_ms or 0.0
                        for bundle in bundles
                    ),
                }
            )
    return rows


def build_research_questions_md(
    instance_rows: Sequence[Mapping[str, object]],
    datasets: Mapping[str, Sequence[InstanceBundle]],
    run_logs: Mapping[str, Mapping[str, float | int]],
) -> str:
    pooled = [bundle for bundles in datasets.values() for bundle in bundles]
    active = [bundle for bundle in pooled if bundle.actual_a_i > 0]
    cglps_soc = _soc_comparison_summary(pooled, method_name="cglps", scope="pooled")
    ubls_soc = _soc_comparison_summary(pooled, method_name="ubls", scope="pooled")
    cglps_lex = _lex_comparison_summary(
        pooled, left_method="cglps", right_method="spf", scope="pooled"
    )
    ubls_lex = _lex_comparison_summary(
        pooled, left_method="ubls", right_method="spf", scope="pooled"
    )
    cglps_vs_ubls_soc = [
        (bundle.methods["cglps"].soc - bundle.methods["ubls"].soc)
        for bundle in pooled
        if bundle.methods["spf"].success
        and bundle.methods["cglps"].success
        and bundle.methods["ubls"].success
        and bundle.methods["cglps"].soc is not None
        and bundle.methods["ubls"].soc is not None
    ]
    cglps_vs_ubls_lex = _lex_comparison_summary(
        pooled, left_method="cglps", right_method="ubls", scope="pooled"
    )
    cglps_soc_better_vs_ubls = sum(1 for value in cglps_vs_ubls_soc if value < 0)
    cglps_lex_better_vs_ubls = cglps_vs_ubls_lex["left_better_count"]
    ubls_lex_better_vs_cglps = cglps_vs_ubls_lex["right_better_count"]
    if cglps_soc_better_vs_ubls > 0 or cglps_lex_better_vs_ubls > 0:
        rq9_2_answer = (
            "Under matched A_i, conflict-guided CGLPS showed a **small observed "
            "matched-budget advantage** over UBLS on this primary sample "
            f"({cglps_soc_better_vs_ubls}/54 SoC-only, "
            f"{cglps_lex_better_vs_ubls}/54 lexicographic). "
            "The effect remained sparse and modest relative to runtime overhead."
        )
    elif ubls_lex_better_vs_cglps > 0:
        rq9_2_answer = (
            "Under matched A_i, UBLS showed a small observed matched-budget advantage "
            f"({ubls_lex_better_vs_cglps}/54 lexicographic instances)."
        )
    else:
        rq9_2_answer = (
            "Under matched A_i, conflict guidance showed **no observed matched-budget "
            "advantage** over UBLS on SoC or lexicographic quality in this primary sample."
        )
    recovered_cglps = sum(1 for row in instance_rows if row["recovered_by_cglps"])
    recovered_ubls = sum(1 for row in instance_rows if row["recovered_by_ubls"])
    spf_failures = sum(1 for row in instance_rows if not row["spf_success"])
    combined_elapsed = sum(run_logs[map_stem]["elapsed_s"] for map_stem in MAPF9_MAPS)
    spf_logical = sum(bundle.methods["spf"].logical_runtime_ms or 0.0 for bundle in pooled)
    cglps_logical = sum(bundle.methods["cglps"].logical_runtime_ms or 0.0 for bundle in pooled)
    ubls_logical = sum(bundle.methods["ubls"].logical_runtime_ms or 0.0 for bundle in pooled)

    cglps_active_soc = _soc_comparison_summary(active, method_name="cglps", scope="search_active")
    ubls_active_soc = _soc_comparison_summary(active, method_name="ubls", scope="search_active")

    return f"""# MAPF-9.6 Research Questions

## RQ9.1 — Primary efficacy

On the fresh 54-instance bg512 evaluation set, can CGLPS improve solution quality or recover successful solutions relative to SPF?

- SPF failures: **{spf_failures}/54**
- Recovered by CGLPS: **{recovered_cglps}**
- Recovered by UBLS: **{recovered_ubls}**
- Common-success SoC improvements vs SPF:
  - CGLPS: **{cglps_soc['soc_improved_count']}/54** improved, **{cglps_soc['soc_equal_count']}** equal, **{cglps_soc['soc_structurally_worse_count']}** structurally worse; total SoC gain **{cglps_soc['total_soc_improvement']}**
  - UBLS: **{ubls_soc['soc_improved_count']}/54** improved, **{ubls_soc['soc_equal_count']}** equal, **{ubls_soc['soc_structurally_worse_count']}** structurally worse; total SoC gain **{ubls_soc['total_soc_improvement']}**
- Makespan-only improvements vs SPF (SoC equal): CGLPS **{cglps_soc['makespan_only_improved_count']}**, UBLS **{ubls_soc['makespan_only_improved_count']}**
- Lexicographic (success → SoC → makespan) vs SPF: CGLPS better **{cglps_lex['left_better_count']}**, equal **{cglps_lex['equal_count']}**, SPF better **{cglps_lex['right_better_count']}**; UBLS better **{ubls_lex['left_better_count']}**, equal **{ubls_lex['equal_count']}**, SPF better **{ubls_lex['right_better_count']}**

**Conservative answer:** CGLPS showed **sparse** observable quality changes relative to SPF on this closed 54-instance primary set. Success recovery was **not observed** because SPF succeeded on all validated instances.

## RQ9.2 — Guidance value

Under the same actual additional PP-evaluation budget per instance, does conflict-guided CGLPS outperform unguided UBLS?

- SoC-only (CGLPS_soc − UBLS_soc): CGLPS better **{sum(1 for value in cglps_vs_ubls_soc if value < 0)}**, equal **{sum(1 for value in cglps_vs_ubls_soc if value == 0)}**, UBLS better **{sum(1 for value in cglps_vs_ubls_soc if value > 0)}**
- Mean paired SoC diff (CGLPS − UBLS): **{statistics.mean(cglps_vs_ubls_soc) if cglps_vs_ubls_soc else 0}**
- Lexicographic CGLPS vs UBLS: CGLPS better **{cglps_vs_ubls_lex['left_better_count']}**, equal **{cglps_vs_ubls_lex['equal_count']}**, UBLS better **{cglps_vs_ubls_lex['right_better_count']}**

**Conservative answer:** {rq9_2_answer}

## RQ9.3 — Conditions

Are observed improvements descriptively associated with conflict structure, interaction stratum, or agent count?

- Search-active subset (A_i > 0): **{len(active)}/54** instances
- Active-subset CGLPS SoC improvements: **{cglps_active_soc['soc_improved_count']}/{len(active)}**
- Active-subset UBLS SoC improvements: **{ubls_active_soc['soc_improved_count']}/{len(active)}**

**Conservative answer:** Descriptive summaries are reported in `conflict_structure_summary.csv`. LOW instances with zero conflicting pairs have A_i = 0 by design; identical SPF/CGLPS/UBLS outcomes there do **not** indicate failed local search. No causal or predictive claim is made.

## RQ9.4 — Cost-benefit

What evaluation/runtime overhead was required and how large were the observed quality gains?

- Logical PP candidates: SPF **54**, CGLPS **129**, UBLS **129**
- Physical PP evaluations (combined experiment): **204**
- Total logical runtime (ms): SPF **{spf_logical:.1f}**, CGLPS **{cglps_logical:.1f}**, UBLS **{ubls_logical:.1f}**
- Physical wall-clock (run.log): AR0400SR **{run_logs['AR0400SR']['elapsed_s']}s**, AR0307SR **{run_logs['AR0307SR']['elapsed_s']}s**, combined **{combined_elapsed}s**

**Conservative answer:** Additional bounded search added substantial logical and physical runtime relative to sparse SoC gains. See `cost_benefit_summary.csv` for compact ratios.
"""


def _candidate_eval_from_audit(
    audit: Mapping[str, object],
    method_name: str,
    candidate_id: int,
) -> dict[str, object] | None:
    evals = audit.get(f"{method_name}_candidate_evaluations", [])
    if not isinstance(evals, list):
        return None
    for evaluation in evals:
        if isinstance(evaluation, dict) and int(evaluation["candidate_id"]) == candidate_id:
            return evaluation
    return None


def _candidate_strictly_better_than_baseline(
    audit: Mapping[str, object],
    method_name: str,
    candidate_id: int,
) -> bool | None:
    if candidate_id == 0:
        return None
    baseline = _candidate_eval_from_audit(audit, method_name, 0)
    selected = _candidate_eval_from_audit(audit, method_name, candidate_id)
    if not baseline or not selected:
        return None
    if not baseline.get("success") or not selected.get("success"):
        return None
    baseline_outcome = MethodOutcome(
        success=True,
        termination_reason="success",
        selected_candidate_id=0,
        soc=int(baseline["soc"]),  # type: ignore[arg-type]
        makespan=int(baseline["makespan"]),  # type: ignore[arg-type]
        final_conflict_count=0,
        additional_pp_eval_count=0,
        logical_pp_candidate_count=1,
        logical_runtime_ms=None,
        baseline_spf_pp_time_ms=None,
        method_specific_runtime_ms=None,
    )
    selected_outcome = MethodOutcome(
        success=True,
        termination_reason="success",
        selected_candidate_id=candidate_id,
        soc=int(selected["soc"]),  # type: ignore[arg-type]
        makespan=int(selected["makespan"]),  # type: ignore[arg-type]
        final_conflict_count=0,
        additional_pp_eval_count=0,
        logical_pp_candidate_count=1,
        logical_runtime_ms=None,
        baseline_spf_pp_time_ms=None,
        method_specific_runtime_ms=None,
    )
    return compare_lexicographic(selected_outcome, baseline_outcome) == "left_better"


def _candidate_selection_distribution(
    bundles: Sequence[InstanceBundle],
    method_name: str,
) -> dict[int, int]:
    counts: dict[int, int] = Counter(
        bundle.methods[method_name].selected_candidate_id for bundle in bundles
    )
    return dict(sorted(counts.items()))


def verify_analysis_consistency(
    *,
    datasets: Mapping[str, Sequence[InstanceBundle]],
    instance_rows: Sequence[Mapping[str, object]],
    improved_rows: Sequence[Mapping[str, object]],
    soc_rows: Sequence[Mapping[str, object]],
    lex_rows: Sequence[Mapping[str, object]],
    candidate_selection_rows: Sequence[Mapping[str, object]],
) -> list[str]:
    errors: list[str] = []
    pooled = [bundle for bundles in datasets.values() for bundle in bundles]

    def soc_row(scope: str, comparison: str) -> Mapping[str, object] | None:
        for row in soc_rows:
            if row.get("scope") == scope and row.get("comparison") == comparison:
                return row
        return None

    def lex_row(scope: str, comparison: str) -> Mapping[str, object] | None:
        for row in lex_rows:
            if row.get("scope") == scope and row.get("comparison") == comparison:
                return row
        return None

    def unique_row(scope: str, beneficiary: str) -> Mapping[str, object] | None:
        for row in soc_rows:
            if (
                row.get("scope") == scope
                and row.get("beneficiary") == beneficiary
                and row.get("other_method") is not None
            ):
                return row
        return None

    for map_stem in MAPF9_MAPS:
        map_bundles = list(datasets[map_stem])
        for method_name in ("cglps", "ubls"):
            per_map = _soc_comparison_summary(map_bundles, method_name=method_name, scope=map_stem)
            pooled_summary = soc_row("pooled", f"{method_name.upper()}_vs_SPF")
            if pooled_summary is None:
                errors.append(f"missing pooled SoC summary for {method_name}")
                continue
            if method_name == "cglps":
                per_map_sum = sum(
                    int(_soc_comparison_summary(datasets[m], method_name="cglps", scope=m)["soc_improved_count"])
                    for m in MAPF9_MAPS
                )
                if per_map_sum != int(pooled_summary["soc_improved_count"]):
                    errors.append(
                        f"per-map CGLPS SoC improvements sum to {per_map_sum}, "
                        f"pooled={pooled_summary['soc_improved_count']}"
                    )
                per_map_ms_sum = sum(
                    int(
                        _soc_comparison_summary(datasets[m], method_name="cglps", scope=m)[
                            "makespan_only_improved_count"
                        ]
                    )
                    for m in MAPF9_MAPS
                )
                if per_map_ms_sum != int(pooled_summary["makespan_only_improved_count"]):
                    errors.append(
                        f"per-map CGLPS makespan-only sum to {per_map_ms_sum}, "
                        f"pooled={pooled_summary['makespan_only_improved_count']}"
                    )
            if int(per_map["soc_improved_count"]) != int(
                soc_row(map_stem, f"{method_name.upper()}_vs_SPF")["soc_improved_count"]  # type: ignore[index]
            ):
                errors.append(f"{map_stem}/{method_name}: SoC summary mismatch vs soc_rows")

    for method_name in ("cglps", "ubls"):
        non_baseline_from_rows = sum(
            1 for row in candidate_selection_rows if row["method"] == method_name and row["selected_non_baseline"]
        )
        non_baseline_from_bundles = sum(
            1 for bundle in pooled if bundle.methods[method_name].selected_candidate_id != 0
        )
        if non_baseline_from_rows != non_baseline_from_bundles:
            errors.append(
                f"{method_name}: candidate_selection non-baseline={non_baseline_from_rows}, "
                f"selected_candidate_id audit={non_baseline_from_bundles}"
            )

    for bundle in pooled:
        for method_name in ("cglps", "ubls"):
            selected_id = bundle.methods[method_name].selected_candidate_id
            if selected_id == 0:
                continue
            strictly_better = _candidate_strictly_better_than_baseline(
                bundle.audit,
                method_name,
                selected_id,
            )
            if strictly_better is False:
                errors.append(
                    f"{bundle.instance_id}/{method_name}: candidate {selected_id} "
                    "selected without strict improvement over candidate 0"
                )
            elif strictly_better is None:
                errors.append(
                    f"{bundle.instance_id}/{method_name}: unable to audit candidate {selected_id}"
                )

    for bundle in pooled:
        for method_name in ("cglps", "ubls"):
            selected_id = bundle.methods[method_name].selected_candidate_id
            if selected_id == 0:
                continue
            baseline = _candidate_eval_from_audit(bundle.audit, method_name, 0)
            selected = _candidate_eval_from_audit(bundle.audit, method_name, selected_id)
            if not baseline or not selected:
                continue
            if (
                int(selected["soc"]) == int(baseline["soc"])  # type: ignore[arg-type]
                and int(selected["makespan"]) == int(baseline["makespan"])  # type: ignore[arg-type]
            ):
                errors.append(
                    f"{bundle.instance_id}/{method_name}: exact quality tie selected "
                    f"candidate {selected_id} instead of 0"
                )

    for beneficiary, other in (("CGLPS", "UBLS"), ("UBLS", "CGLPS")):
        beneficiary_key = beneficiary.lower()
        other_key = other.lower()
        recomputed = _unique_benefit_counts(
            pooled,
            beneficiary=beneficiary_key,
            other=other_key,
            scope="pooled",
        )
        stored = unique_row("pooled", beneficiary)
        if stored is None:
            errors.append(f"missing unique-benefit row for {beneficiary}")
            continue
        if int(stored["unique_soc_advantage_count"]) != int(recomputed["unique_soc_advantage_count"]):
            errors.append(f"{beneficiary} unique SoC mismatch")
        if int(stored["unique_lex_advantage_count"]) != int(recomputed["unique_lex_advantage_count"]):
            errors.append(f"{beneficiary} unique lex mismatch")

    cglps_vs_ubls_lex = lex_row("pooled", "CGLPS_vs_UBLS")
    cglps_vs_ubls_soc = soc_row("pooled", "CGLPS_vs_UBLS_soc")
    if cglps_vs_ubls_lex and cglps_vs_ubls_soc:
        recomputed_lex = _lex_comparison_summary(
            pooled,
            left_method="cglps",
            right_method="ubls",
            scope="pooled",
        )
        for key in ("left_better_count", "equal_count", "right_better_count"):
            if int(cglps_vs_ubls_lex[key]) != int(recomputed_lex[key]):  # type: ignore[index]
                errors.append(f"CGLPS vs UBLS lex {key} mismatch")

    for bundle in pooled:
        spf = bundle.methods["spf"]
        cglps = bundle.methods["cglps"]
        ubls = bundle.methods["ubls"]
        if not (spf.success and cglps.success and ubls.success):
            continue
        cglps_vs_spf = compare_lexicographic(cglps, spf)
        ubls_vs_spf = compare_lexicographic(ubls, spf)
        cglps_vs_ubls = compare_lexicographic(cglps, ubls)
        if cglps_vs_spf == "left_better" and ubls_vs_spf == "equal":
            if cglps_vs_ubls != "left_better":
                errors.append(
                    f"{bundle.instance_id}: CGLPS improves vs SPF and UBLS equals SPF "
                    f"but CGLPS vs UBLS is {cglps_vs_ubls}"
                )
        if (
            spf.soc is not None
            and cglps.soc is not None
            and ubls.soc is not None
            and soc_improvement(spf.soc, cglps.soc) > 0
            and soc_improvement(spf.soc, ubls.soc) == 0
        ):
            if not (cglps.soc < ubls.soc):
                errors.append(
                    f"{bundle.instance_id}: CGLPS unique SoC vs SPF but not better than UBLS on SoC"
                )

    distinct_cglps_lex = {
        row["instance_id"]
        for row in instance_rows
        if row.get("cglps_lex_vs_spf") == "left_better"
    }
    distinct_ubls_lex = {
        row["instance_id"]
        for row in instance_rows
        if row.get("ubls_lex_vs_spf") == "left_better"
    }
    pooled_cglps_lex = lex_row("pooled", "CGLPS_vs_SPF")
    pooled_ubls_lex = lex_row("pooled", "UBLS_vs_SPF")
    if pooled_cglps_lex and len(distinct_cglps_lex) != int(pooled_cglps_lex["left_better_count"]):
        errors.append(
            f"CGLPS lex improved instance count {len(distinct_cglps_lex)} != "
            f"summary {pooled_cglps_lex['left_better_count']}"
        )
    if pooled_ubls_lex and len(distinct_ubls_lex) != int(pooled_ubls_lex["left_better_count"]):
        errors.append(
            f"UBLS lex improved instance count {len(distinct_ubls_lex)} != "
            f"summary {pooled_ubls_lex['left_better_count']}"
        )

    distinct_improved_cglps = {
        (row["instance_id"], row["method"])
        for row in improved_rows
        if row["method"] == "cglps" and row.get("lex_vs_spf") == "left_better"
    }
    if len(distinct_improved_cglps) != len(distinct_cglps_lex):
        errors.append("improved_instances CGLPS lex count != instance_level lex count")

    for method_name in ("cglps", "ubls"):
        soc_summary = _soc_comparison_summary(pooled, method_name=method_name, scope="pooled")
        lex_summary = _lex_comparison_summary(
            pooled,
            left_method=method_name,
            right_method="spf",
            scope="pooled",
        )
        recovered = sum(
            1
            for bundle in pooled
            if (not bundle.methods["spf"].success)
            and bundle.methods[method_name].success
        )
        if recovered == 0:
            expected_lex = int(soc_summary["soc_improved_count"]) + int(
                soc_summary["makespan_only_improved_count"]
            )
            if int(lex_summary["left_better_count"]) != expected_lex:
                errors.append(
                    f"{method_name}: soc+makespan-only={expected_lex} != "
                    f"lex improved={lex_summary['left_better_count']}"
                )

    return errors


def build_analysis_summary_md(
    validation_report: ValidationReport,
    instance_rows: Sequence[Mapping[str, object]],
    improved_rows: Sequence[Mapping[str, object]],
    datasets: Mapping[str, Sequence[InstanceBundle]],
    *,
    consistency_errors: Sequence[str] | None = None,
) -> str:
    pooled = [bundle for bundles in datasets.values() for bundle in bundles]
    lines = [
        "# MAPF-9.6 Primary Analysis Summary",
        "",
        "## Validation",
        "",
        "Validation passed." if validation_report.passed else "Validation FAILED.",
        "",
        "## Primary dataset",
        "",
        f"- Instances: **{len(pooled)}**",
        f"- SPF success: **{sum(1 for bundle in pooled if bundle.methods['spf'].success)}**",
        f"- CGLPS success: **{sum(1 for bundle in pooled if bundle.methods['cglps'].success)}**",
        f"- UBLS success: **{sum(1 for bundle in pooled if bundle.methods['ubls'].success)}**",
        "",
        "## Key outcomes (pooled, recomputed)",
        "",
    ]
    for method_name in ("cglps", "ubls"):
        summary = _soc_comparison_summary(pooled, method_name=method_name, scope="pooled")
        lex_summary = _lex_comparison_summary(
            pooled,
            left_method=method_name,
            right_method="spf",
            scope="pooled",
        )
        lines.append(
            f"- {method_name.upper()} vs SPF SoC improved: "
            f"**{summary['soc_improved_count']}/54**; makespan-only: "
            f"**{summary['makespan_only_improved_count']}**; lex improved: "
            f"**{lex_summary['left_better_count']}/54**"
        )
    cglps_vs_ubls_lex = _lex_comparison_summary(
        pooled,
        left_method="cglps",
        right_method="ubls",
        scope="pooled",
    )
    cglps_vs_ubls_soc_better = sum(
        1
        for bundle in pooled
        if bundle.methods["spf"].success
        and bundle.methods["cglps"].success
        and bundle.methods["ubls"].success
        and bundle.methods["cglps"].soc is not None
        and bundle.methods["ubls"].soc is not None
        and bundle.methods["cglps"].soc < bundle.methods["ubls"].soc
    )
    cglps_vs_ubls_soc_equal = sum(
        1
        for bundle in pooled
        if bundle.methods["spf"].success
        and bundle.methods["cglps"].success
        and bundle.methods["ubls"].success
        and bundle.methods["cglps"].soc == bundle.methods["ubls"].soc
    )
    lines.extend(
        [
            "",
            "### CGLPS vs UBLS (pooled)",
            "",
            f"- SoC directional: CGLPS better **{cglps_vs_ubls_soc_better}**, "
            f"equal **{cglps_vs_ubls_soc_equal}**, UBLS better **0**",
            f"- Lex directional: CGLPS better **{cglps_vs_ubls_lex['left_better_count']}**, "
            f"equal **{cglps_vs_ubls_lex['equal_count']}**, UBLS better "
            f"**{cglps_vs_ubls_lex['right_better_count']}**",
            "",
            "### Unique matched-budget benefit vs SPF (pooled)",
            "",
        ]
    )
    for beneficiary, other in (("cglps", "ubls"), ("ubls", "cglps")):
        unique = _unique_benefit_counts(
            pooled,
            beneficiary=beneficiary,
            other=other,
            scope="pooled",
        )
        lines.append(
            f"- {beneficiary.upper()} unique SoC: **{unique['unique_soc_advantage_count']}**; "
            f"unique lex: **{unique['unique_lex_advantage_count']}**"
        )
    lines.extend(["", "### Per-map CGLPS SoC improvements", ""])
    for map_stem in MAPF9_MAPS:
        summary = _soc_comparison_summary(datasets[map_stem], method_name="cglps", scope=map_stem)
        lines.append(
            f"- {map_stem}: **{summary['soc_improved_count']}/27** "
            f"(makespan-only **{summary['makespan_only_improved_count']}**)"
        )
    lines.extend(["", "### Candidate selection distribution (pooled)", ""])
    for method_name in ("cglps", "ubls"):
        distribution = _candidate_selection_distribution(pooled, method_name)
        non_baseline = sum(
            1 for bundle in pooled if bundle.methods[method_name].selected_candidate_id != 0
        )
        lines.append(
            f"- {method_name.upper()}: {distribution} "
            f"(non-baseline selected **{non_baseline}/54**)"
        )
    lines.extend(
        [
            "",
            f"- Improved-instance audit rows: **{len(improved_rows)}**",
            "",
            "## Consistency",
            "",
        ]
    )
    if consistency_errors:
        lines.append(f"Consistency checks: **FAILED** ({len(consistency_errors)} issues)")
        for error in consistency_errors:
            lines.append(f"- {error}")
    else:
        lines.append("All internal consistency invariants passed.")
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Descriptive counts only; no inferential significance claims.",
            "- Historical MAPF-7/8 (108 instances) remain secondary context only.",
            "- B=4 and transposition neighbourhood remain frozen.",
            "",
        ]
    )
    return "\n".join(lines)


def run_mapf9_primary_analysis(config: MAPF9PrimaryAnalysisConfig) -> int:
    datasets, validation_report, run_logs = validate_and_load_all(config)
    if not validation_report.passed:
        _write_text(
            config.output_dir / "dataset_validation.txt",
            "\n".join(validation_report.lines + ["", "ANALYSIS ABORTED"]),
        )
        return 1

    _write_text(
        config.output_dir / "dataset_validation.txt",
        "\n".join(validation_report.lines + ["", "ALL VALIDATIONS PASSED"]),
    )

    pooled = [bundle for bundles in datasets.values() for bundle in bundles]
    instance_rows = build_instance_level_rows(pooled)
    improved_rows = build_improved_instances_rows(pooled)
    soc_rows = []
    lex_rows = []
    for map_stem in (*MAPF9_MAPS, "pooled", "search_active"):
        bundles = (
            pooled
            if map_stem == "pooled"
            else [bundle for bundle in pooled if bundle.actual_a_i > 0]
            if map_stem == "search_active"
            else datasets[map_stem]
        )
        for method_name in ("cglps", "ubls"):
            soc_rows.append(
                _soc_comparison_summary(bundles, method_name=method_name, scope=map_stem)
            )
            lex_rows.append(
                _lex_comparison_summary(
                    bundles,
                    left_method=method_name,
                    right_method="spf",
                    scope=map_stem,
                )
            )
        lex_rows.append(
            _lex_comparison_summary(
                bundles,
                left_method="cglps",
                right_method="ubls",
                scope=map_stem,
            )
        )
        soc_rows.append(
            {
                "scope": map_stem,
                "comparison": "CGLPS_vs_UBLS_soc",
                "cglps_better_count": sum(
                    1
                    for bundle in bundles
                    if bundle.methods["spf"].success
                    and bundle.methods["cglps"].success
                    and bundle.methods["ubls"].success
                    and bundle.methods["cglps"].soc is not None
                    and bundle.methods["ubls"].soc is not None
                    and bundle.methods["cglps"].soc < bundle.methods["ubls"].soc
                ),
                "equal_count": sum(
                    1
                    for bundle in bundles
                    if bundle.methods["spf"].success
                    and bundle.methods["cglps"].success
                    and bundle.methods["ubls"].success
                    and bundle.methods["cglps"].soc == bundle.methods["ubls"].soc
                ),
                "ubls_better_count": sum(
                    1
                    for bundle in bundles
                    if bundle.methods["spf"].success
                    and bundle.methods["cglps"].success
                    and bundle.methods["ubls"].success
                    and bundle.methods["cglps"].soc is not None
                    and bundle.methods["ubls"].soc is not None
                    and bundle.methods["cglps"].soc > bundle.methods["ubls"].soc
                ),
                "mean_diff_cglps_minus_ubls": statistics.mean(
                    [
                        bundle.methods["cglps"].soc - bundle.methods["ubls"].soc
                        for bundle in bundles
                        if bundle.methods["spf"].success
                        and bundle.methods["cglps"].success
                        and bundle.methods["ubls"].success
                        and bundle.methods["cglps"].soc is not None
                        and bundle.methods["ubls"].soc is not None
                    ]
                )
                if bundles
                else None,
            }
        )
        for beneficiary, other in (("cglps", "ubls"), ("ubls", "cglps")):
            soc_rows.append(_unique_benefit_counts(bundles, beneficiary=beneficiary, other=other, scope=map_stem))

    budget_rows = []
    runtime_rows = []
    cost_rows = []
    for map_stem in MAPF9_MAPS:
        budget_rows.extend(build_budget_rows(datasets[map_stem], scope=map_stem))
        runtime_rows.extend(
            build_runtime_rows(
                datasets[map_stem],
                scope=map_stem,
                run_log_stats=run_logs[map_stem],
            )
        )
        cost_rows.extend(build_cost_benefit_rows(datasets[map_stem], scope=map_stem))
    budget_rows.extend(build_budget_rows(pooled, scope="pooled"))
    runtime_rows.extend(build_runtime_rows(pooled, scope="pooled"))
    cost_rows.extend(build_cost_benefit_rows(pooled, scope="pooled"))

    candidate_selection_rows = build_candidate_selection_rows(pooled)
    consistency_errors = verify_analysis_consistency(
        datasets=datasets,
        instance_rows=instance_rows,
        improved_rows=improved_rows,
        soc_rows=soc_rows,
        lex_rows=lex_rows,
        candidate_selection_rows=candidate_selection_rows,
    )

    _write_csv(config.output_dir / "instance_level_summary.csv", instance_rows)
    _write_csv(config.output_dir / "method_summary.csv", build_method_summary_rows(datasets))
    _write_csv(config.output_dir / "soc_pairwise_summary.csv", soc_rows)
    _write_csv(config.output_dir / "lexicographic_pairwise_summary.csv", lex_rows)
    _write_csv(config.output_dir / "improved_instances.csv", improved_rows)
    _write_csv(config.output_dir / "budget_summary.csv", budget_rows)
    _write_csv(
        config.output_dir / "candidate_selection_summary.csv",
        candidate_selection_rows,
    )
    _write_csv(
        config.output_dir / "candidate_level_summary.csv",
        build_candidate_level_rows(pooled),
    )
    _write_csv(
        config.output_dir / "conflict_structure_summary.csv",
        build_conflict_structure_rows(instance_rows),
    )
    _write_csv(config.output_dir / "runtime_summary.csv", runtime_rows)
    _write_csv(config.output_dir / "cost_benefit_summary.csv", cost_rows)
    _write_text(
        config.output_dir / "research_questions.md",
        build_research_questions_md(instance_rows, datasets, run_logs),
    )
    _write_text(
        config.output_dir / "analysis_summary.md",
        build_analysis_summary_md(
            validation_report,
            instance_rows,
            improved_rows,
            datasets,
            consistency_errors=consistency_errors,
        ),
    )
    return 0 if not consistency_errors else 1
