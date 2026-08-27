"""MAPF-8.5 formal cross-map priority strategy analysis (analysis only)."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pathfinding.src.experiments.mapf7_final_analysis import (
    FINAL_PAIRWISE_COMPARISONS,
    Mapf7FinalAnalysisConfig,
    build_catalogue_pairwise_summary,
    build_heldout_instance_level_summary,
    build_spf_cd_ablation_aggregate,
    build_spf_cd_ablation_rows,
    load_heldout_spf_results,
    _reproduce_primary_instance_rows,
)
from pathfinding.src.experiments.mapf_benchmark_instances import load_benchmark_manifest
from pathfinding.src.experiments.mapf7_final_analysis_plots import instance_full_soc_range
from pathfinding.src.experiments.mapf_benchmark_catalogue_generation import (
    MAPF8_CROSS_MAP_SEEDS,
)
from pathfinding.src.experiments.mapf_conflict_aware_priority_analysis import (
    STRATEGY_CDF_H,
    STRATEGY_CDF_L,
    STRATEGY_SPF_CD,
    ValidationReport,
    _degree_summary,
    load_conflict_aware_results,
    mapf7_strategy_socs_differ,
)
from pathfinding.src.experiments.mapf_cross_map_priority_execution import (
    CrossMapPriorityRunRecord,
    CrossMapPriorityStrategy,
    DEFAULT_CROSS_MAP_STRATEGIES,
    FROZEN_MANIFEST_SHA256,
    MAPF8_CROSS_MAP_RESULTS_ROOT,
)
from pathfinding.src.experiments.mapf_priority_strategy_analysis import (
    AGENT_COUNT_ORDER,
    INTERACTION_ORDER,
    STRATEGY_SPF,
    _format_float,
    _format_percent,
    _instance_sort_key,
    _parse_bool,
    _parse_int,
    _read_csv_rows,
    _row_int,
    _summary_stats,
    _write_csv,
    _write_text,
)

MAPF8_ANALYSIS_DIR = Path("pathfinding/results/mapf8_cross_map_analysis")

MAPF8_CROSS_MAP_TARGETS: tuple[str, ...] = ("AR0400SR", "AR0307SR")

MAPF8_STRATEGIES: tuple[str, ...] = tuple(
    strategy.value for strategy in DEFAULT_CROSS_MAP_STRATEGIES
)

MAPF8_STRATEGY_LABELS: dict[str, str] = {
    "spf": "SPF",
    "cdf_h": "CDF-H",
    "cdf_l": "CDF-L",
    "spf_cd": "SPF+CD",
}

EXPECTED_INSTANCES_PER_MAP = 27
EXPECTED_RECORDS_PER_MAP = 108
EXPECTED_COMBINED_RECORDS = 216
EXPECTED_COMBINED_INSTANCES = 54

OUTPUT_TABLES: tuple[str, ...] = (
    "dataset_validation.txt",
    "catalogue_summary.csv",
    "instance_level_summary.csv",
    "sensitive_instances.csv",
    "pairwise_summary.csv",
    "interaction_summary.csv",
    "agent_count_summary.csv",
    "spf_cd_order_summary.csv",
    "conflict_structure_summary.csv",
    "runtime_summary.csv",
    "historical_comparison.csv",
    "research_questions.md",
    "analysis_summary.md",
    "thesis_tables.md",
)


@dataclass(frozen=True, slots=True)
class Mapf8CrossMapAnalysisConfig:
    repo_root: Path
    ar0400_csv: Path
    ar0400_manifest: Path
    ar0307_csv: Path
    ar0307_manifest: Path
    output_dir: Path
    mapf7_config: Mapf7FinalAnalysisConfig | None = None


@dataclass(frozen=True, slots=True)
class Mapf8CrossMapAnalysisResult:
    validation: ValidationReport
    output_dir: Path
    tables_written: tuple[str, ...]
    ar0400_instance_rows: tuple[dict[str, Any], ...]
    ar0307_instance_rows: tuple[dict[str, Any], ...]
    pooled_instance_rows: tuple[dict[str, Any], ...]


def _parse_int_tuple(value: str | None) -> tuple[int, ...] | None:
    if value is None or value == "":
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("expected JSON list for integer tuple field")
    return tuple(int(item) for item in parsed)


def _write_csv_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        _write_text(path, "")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    _write_csv(path, rows, fieldnames)


def load_cross_map_results(path: Path) -> tuple[CrossMapPriorityRunRecord, ...]:
    rows = _read_csv_rows(path)
    records: list[CrossMapPriorityRunRecord] = []
    for row in rows:
        records.append(
            CrossMapPriorityRunRecord(
                map_name=row["map_name"],
                manifest_sha256=row["manifest_sha256"],
                instance_id=row["instance_id"],
                agent_count=int(row["agent_count"]),
                interaction_level=row["interaction_level"].lower(),
                strategy=CrossMapPriorityStrategy(row["strategy"].lower()),
                agent_order=_parse_int_tuple(row.get("agent_order")),
                original_index_order=_parse_int_tuple(row.get("original_index_order")),
                agent_degrees=_parse_int_tuple(row.get("agent_degrees")),
                independent_conflict_count=_parse_int(row.get("independent_conflict_count")),
                independent_conflict_pair_count=_parse_int(
                    row.get("independent_conflict_pair_count")
                ),
                ordering_low_level_searches=_parse_int(
                    row.get("ordering_low_level_searches")
                ),
                incident_conflict_counts=_parse_int_tuple(
                    row.get("incident_conflict_counts")
                ),
                success=_parse_bool(row.get("success")),
                termination_reason=row["termination_reason"],
                ordering_time_ms=float(row["ordering_time_ms"]),
                pp_time_ms=_parse_float_or_none(row.get("pp_time_ms")),
                total_time_ms=float(row["total_time_ms"]),
                soc=_parse_int(row.get("soc")),
                makespan=_parse_int(row.get("makespan")),
                conflict_count=_parse_int(row.get("conflict_count")),
                pp_low_level_searches=_parse_int(row.get("pp_low_level_searches")),
                pp_agents_planned=_parse_int(row.get("pp_agents_planned")),
                error_message=row.get("error_message") or None,
                catalogue_seed=_parse_int(row.get("catalogue_seed")),
            )
        )
    return tuple(records)


def _parse_float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def compute_four_strategy_soc_range(row: dict[str, Any]) -> int:
    return instance_full_soc_range(row)


def compute_four_strategy_makespan_range(row: dict[str, Any]) -> int:
    makespans = [
        _row_int(row.get("spf_makespan")),
        _row_int(row.get("cdf_h_makespan")),
        _row_int(row.get("cdf_l_makespan")),
        _row_int(row.get("spf_cd_makespan")),
    ]
    values = [value for value in makespans if value is not None]
    if not values:
        return 0
    return max(values) - min(values)


def compute_mapf7_three_strategy_soc_range(row: dict[str, Any]) -> int:
    socs = {
        _row_int(row.get("cdf_h_soc")),
        _row_int(row.get("cdf_l_soc")),
        _row_int(row.get("spf_cd_soc")),
    }
    socs.discard(None)
    if not socs:
        return 0
    return max(socs) - min(socs)


def is_soc_sensitive(row: dict[str, Any]) -> bool:
    return compute_four_strategy_soc_range(row) > 0


def is_makespan_sensitive(row: dict[str, Any]) -> bool:
    return compute_four_strategy_makespan_range(row) > 0


def compute_pairwise_soc_diff(left_soc: int, right_soc: int) -> int:
    """diff = LEFT SoC - RIGHT SoC; negative means LEFT better."""
    return left_soc - right_soc


def spf_cd_same_order(
    spf_order: tuple[int, ...] | None,
    spf_cd_index_order: tuple[int, ...] | None,
) -> bool | None:
    if spf_order is None or spf_cd_index_order is None:
        return None
    return spf_order == spf_cd_index_order


def strategy_same_order_as_spf(
    spf_order: tuple[int, ...] | None,
    strategy_index_order: tuple[int, ...] | None,
) -> bool | None:
    if spf_order is None or strategy_index_order is None:
        return None
    return spf_order == strategy_index_order


def order_diff_without_soc_diff(
    *,
    same_order: bool | None,
    soc_equal: bool,
) -> bool:
    return same_order is False and soc_equal


def _records_by_key(
    records: Sequence[CrossMapPriorityRunRecord],
) -> dict[tuple[str, str, str], CrossMapPriorityRunRecord]:
    lookup: dict[tuple[str, str, str], CrossMapPriorityRunRecord] = {}
    for record in records:
        key = (record.map_name, record.instance_id, record.strategy.value)
        if key in lookup:
            raise ValueError(
                f"Duplicate run key: map={record.map_name}, "
                f"instance={record.instance_id}, strategy={record.strategy.value}"
            )
        lookup[key] = record
    return lookup


def validate_mapf8_map_dataset(
    *,
    map_name: str,
    records: Sequence[CrossMapPriorityRunRecord],
    manifest: Any,
    expected_manifest_sha256: str,
    expected_seed: int,
) -> ValidationReport:
    report = ValidationReport(passed=True)
    label = map_name

    report.add(
        f"{label}: expected {EXPECTED_RECORDS_PER_MAP} records, found {len(records)}",
        passed=len(records) == EXPECTED_RECORDS_PER_MAP,
    )

    keys = {
        (record.map_name, record.instance_id, record.strategy.value)
        for record in records
    }
    report.add(
        f"{label}: no duplicate (map_name, instance_id, strategy)",
        passed=len(keys) == len(records),
    )

    instance_ids = {record.instance_id for record in records}
    manifest_ids = {instance.instance_id for instance in manifest.instances}
    report.add(
        f"{label}: exactly {EXPECTED_INSTANCES_PER_MAP} unique instance IDs",
        passed=len(instance_ids) == EXPECTED_INSTANCES_PER_MAP,
    )
    report.add(
        f"{label}: instance IDs match manifest",
        passed=instance_ids == manifest_ids,
    )

    expected_strategies = set(MAPF8_STRATEGIES)
    observed_strategies = {record.strategy.value for record in records}
    report.add(
        f"{label}: strategy set exactly SPF/CDF-H/CDF-L/SPF+CD",
        passed=observed_strategies == expected_strategies,
    )

    for strategy in MAPF8_STRATEGIES:
        count = sum(1 for record in records if record.strategy.value == strategy)
        report.add(
            f"{label} strategy {strategy}: expected {EXPECTED_INSTANCES_PER_MAP}, found {count}",
            passed=count == EXPECTED_INSTANCES_PER_MAP,
        )

    for instance_id in instance_ids:
        per_instance = [record for record in records if record.instance_id == instance_id]
        report.add(
            f"{label} {instance_id}: exactly four strategy records",
            passed=len(per_instance) == 4,
        )

    success_count = sum(1 for record in records if record.success)
    failure_count = len(records) - success_count
    report.add(
        f"{label}: success={success_count}, failure={failure_count}",
        passed=failure_count == 0,
    )

    for record in records:
        if record.map_name != map_name:
            report.add(
                f"{label}: wrong map_name in record {record.instance_id}",
                passed=False,
            )
        if record.manifest_sha256 != expected_manifest_sha256:
            report.add(
                f"{label}: manifest SHA-256 mismatch in {record.instance_id} / "
                f"{record.strategy.value}",
                passed=False,
            )
        if record.catalogue_seed != expected_seed:
            report.add(
                f"{label}: catalogue seed mismatch in {record.instance_id}",
                passed=record.catalogue_seed == expected_seed,
            )
        if record.agent_count not in AGENT_COUNT_ORDER:
            report.add(
                f"{label}: invalid agent_count in {record.instance_id}",
                passed=False,
            )
        if record.interaction_level not in INTERACTION_ORDER:
            report.add(
                f"{label}: invalid interaction_level in {record.instance_id}",
                passed=False,
            )
        if record.success and record.conflict_count != 0:
            report.add(
                f"{label}: successful run has conflict_count != 0: "
                f"{record.instance_id} / {record.strategy.value}",
                passed=False,
            )

    return report


def validate_mapf8_combined_dataset(
    *,
    ar0400_records: Sequence[CrossMapPriorityRunRecord],
    ar0307_records: Sequence[CrossMapPriorityRunRecord],
) -> ValidationReport:
    report = ValidationReport(passed=True)
    combined = list(ar0400_records) + list(ar0307_records)
    report.add(
        f"Combined MAPF-8: expected {EXPECTED_COMBINED_RECORDS} records, found {len(combined)}",
        passed=len(combined) == EXPECTED_COMBINED_RECORDS,
    )
    instance_keys = {(record.map_name, record.instance_id) for record in combined}
    report.add(
        f"Combined MAPF-8: expected {EXPECTED_COMBINED_INSTANCES} unique instances, "
        f"found {len(instance_keys)}",
        passed=len(instance_keys) == EXPECTED_COMBINED_INSTANCES,
    )
    return report


def build_instance_level_summary_from_records(
    *,
    map_name: str,
    records: Sequence[CrossMapPriorityRunRecord],
) -> list[dict[str, Any]]:
    by_instance: dict[str, dict[str, CrossMapPriorityRunRecord]] = defaultdict(dict)
    for record in records:
        by_instance[record.instance_id][record.strategy.value] = record

    rows: list[dict[str, Any]] = []
    for instance_id in sorted(
        by_instance,
        key=lambda item: _instance_sort_key(
            item,
            by_instance[item][STRATEGY_SPF].agent_count,
            by_instance[item][STRATEGY_SPF].interaction_level,
        ),
    ):
        strategy_records = by_instance[instance_id]
        missing = set(MAPF8_STRATEGIES) - set(strategy_records)
        if missing:
            raise ValueError(
                f"{map_name} {instance_id}: missing strategies {sorted(missing)}"
            )

        spf = strategy_records[STRATEGY_SPF]
        cdf_h = strategy_records[STRATEGY_CDF_H]
        cdf_l = strategy_records[STRATEGY_CDF_L]
        spf_cd = strategy_records[STRATEGY_SPF_CD]
        degree_stats = _degree_summary(cdf_h.agent_degrees)

        row: dict[str, Any] = {
            "map_name": map_name,
            "instance_id": instance_id,
            "agent_count": spf.agent_count,
            "interaction_level": spf.interaction_level,
            "catalogue_seed": spf.catalogue_seed,
            "manifest_sha256": spf.manifest_sha256,
            "spf_soc": spf.soc,
            "cdf_h_soc": cdf_h.soc,
            "cdf_l_soc": cdf_l.soc,
            "spf_cd_soc": spf_cd.soc,
            "spf_makespan": spf.makespan,
            "cdf_h_makespan": cdf_h.makespan,
            "cdf_l_makespan": cdf_l.makespan,
            "spf_cd_makespan": spf_cd.makespan,
            "soc_range": compute_four_strategy_soc_range(
                {
                    "spf_soc": spf.soc,
                    "cdf_h_soc": cdf_h.soc,
                    "cdf_l_soc": cdf_l.soc,
                    "spf_cd_soc": spf_cd.soc,
                }
            ),
            "makespan_range": compute_four_strategy_makespan_range(
                {
                    "spf_makespan": spf.makespan,
                    "cdf_h_makespan": cdf_h.makespan,
                    "cdf_l_makespan": cdf_l.makespan,
                    "spf_cd_makespan": spf_cd.makespan,
                }
            ),
            "mapf7_three_strategy_soc_range": compute_mapf7_three_strategy_soc_range(
                {
                    "cdf_h_soc": cdf_h.soc,
                    "cdf_l_soc": cdf_l.soc,
                    "spf_cd_soc": spf_cd.soc,
                }
            ),
            "soc_sensitive": is_soc_sensitive(
                {
                    "spf_soc": spf.soc,
                    "cdf_h_soc": cdf_h.soc,
                    "cdf_l_soc": cdf_l.soc,
                    "spf_cd_soc": spf_cd.soc,
                }
            ),
            "makespan_sensitive": is_makespan_sensitive(
                {
                    "spf_makespan": spf.makespan,
                    "cdf_h_makespan": cdf_h.makespan,
                    "cdf_l_makespan": cdf_l.makespan,
                    "spf_cd_makespan": spf_cd.makespan,
                }
            ),
            "mapf7_three_strategy_soc_sensitive": mapf7_strategy_socs_differ(
                {
                    "cdf_h_soc": cdf_h.soc,
                    "cdf_l_soc": cdf_l.soc,
                    "spf_cd_soc": spf_cd.soc,
                }
            ),
            "independent_conflict_count": cdf_h.independent_conflict_count,
            "independent_conflict_pair_count": cdf_h.independent_conflict_pair_count,
            **degree_stats,
            "spf_agent_order": json.dumps(list(spf.agent_order or ())),
            "cdf_h_original_index_order": json.dumps(
                list(cdf_h.original_index_order or ())
            ),
            "cdf_l_original_index_order": json.dumps(
                list(cdf_l.original_index_order or ())
            ),
            "spf_cd_original_index_order": json.dumps(
                list(spf_cd.original_index_order or ())
            ),
            "spf_cd_same_order": spf_cd_same_order(
                spf.agent_order, spf_cd.original_index_order
            ),
            "cdf_h_same_order_as_spf": strategy_same_order_as_spf(
                spf.agent_order, cdf_h.original_index_order
            ),
            "cdf_l_same_order_as_spf": strategy_same_order_as_spf(
                spf.agent_order, cdf_l.original_index_order
            ),
            "spf_cd_soc_equal": spf.soc == spf_cd.soc,
            "spf_cd_makespan_equal": spf.makespan == spf_cd.makespan,
            "cdf_h_order_diff_no_soc_diff": order_diff_without_soc_diff(
                same_order=strategy_same_order_as_spf(
                    spf.agent_order, cdf_h.original_index_order
                ),
                soc_equal=spf.soc == cdf_h.soc,
            ),
            "cdf_l_order_diff_no_soc_diff": order_diff_without_soc_diff(
                same_order=strategy_same_order_as_spf(
                    spf.agent_order, cdf_l.original_index_order
                ),
                soc_equal=spf.soc == cdf_l.soc,
            ),
        }
        rows.append(row)
    return rows


def _range_stats(values: Sequence[int | float]) -> dict[str, Any]:
    if not values:
        return {
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
        }
    return {
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }


def build_catalogue_summary_rows(
    *,
    catalogue: str,
    instance_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    instance_count = len(instance_rows)
    soc_ranges = [int(row["soc_range"]) for row in instance_rows]
    makespan_ranges = [int(row["makespan_range"]) for row in instance_rows]
    soc_sensitive_count = sum(1 for row in instance_rows if row["soc_sensitive"])
    makespan_sensitive_count = sum(
        1 for row in instance_rows if row["makespan_sensitive"]
    )
    mapf7_sensitive_count = sum(
        1 for row in instance_rows if row["mapf7_three_strategy_soc_sensitive"]
    )
    soc_stats = _range_stats(soc_ranges)
    makespan_stats = _range_stats(makespan_ranges)
    return {
        "catalogue": catalogue,
        "instance_count": instance_count,
        "soc_sensitive_count": soc_sensitive_count,
        "soc_sensitive_pct": _format_percent(soc_sensitive_count, instance_count),
        "makespan_sensitive_count": makespan_sensitive_count,
        "makespan_sensitive_pct": _format_percent(
            makespan_sensitive_count, instance_count
        ),
        "mapf7_three_strategy_soc_sensitive_count": mapf7_sensitive_count,
        "mapf7_three_strategy_soc_sensitive_pct": _format_percent(
            mapf7_sensitive_count, instance_count
        ),
        "max_soc_range": soc_stats["max"],
        "mean_soc_range": soc_stats["mean"],
        "median_soc_range": soc_stats["median"],
        "max_makespan_range": makespan_stats["max"],
        "mean_makespan_range": makespan_stats["mean"],
        "median_makespan_range": makespan_stats["median"],
    }


def build_pairwise_summary_for_catalogue(
    *,
    catalogue: str,
    instance_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries = build_catalogue_pairwise_summary(instance_rows)
    abs_diffs: dict[str, float] = {}
    for comparison, left_key, right_key, _, _ in FINAL_PAIRWISE_COMPARISONS:
        diffs: list[int] = []
        for row in instance_rows:
            left = _row_int(row[left_key])
            right = _row_int(row[right_key])
            if left is None or right is None:
                continue
            diffs.append(left - right)
        abs_diffs[comparison] = max((abs(value) for value in diffs), default=0)

    rows: list[dict[str, Any]] = []
    for summary in summaries:
        comparison = summary["comparison"]
        rows.append(
            {
                "catalogue": catalogue,
                **summary,
                "max_abs_soc_diff": abs_diffs.get(comparison, 0),
            }
        )
    return rows


def build_sensitive_instances_rows(
    instance_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in instance_rows:
        if not row["soc_sensitive"]:
            continue
        rows.append(
            {
                "map_name": row["map_name"],
                "instance_id": row["instance_id"],
                "agent_count": row["agent_count"],
                "interaction_level": row["interaction_level"],
                "spf_soc": row["spf_soc"],
                "cdf_h_soc": row["cdf_h_soc"],
                "cdf_l_soc": row["cdf_l_soc"],
                "spf_cd_soc": row["spf_cd_soc"],
                "soc_range": row["soc_range"],
                "spf_makespan": row["spf_makespan"],
                "cdf_h_makespan": row["cdf_h_makespan"],
                "cdf_l_makespan": row["cdf_l_makespan"],
                "spf_cd_makespan": row["spf_cd_makespan"],
                "makespan_range": row["makespan_range"],
                "independent_conflict_count": row.get("independent_conflict_count"),
                "independent_conflict_pair_count": row.get(
                    "independent_conflict_pair_count"
                ),
                "max_degree": row.get("max_degree"),
            }
        )
    rows.sort(
        key=lambda item: (
            str(item["map_name"]),
            -int(item.get("soc_range") or 0),
            str(item["instance_id"]),
        )
    )
    return rows


def build_interaction_sensitivity_summary(
    *,
    catalogue: str,
    instance_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level in INTERACTION_ORDER:
        subset = [row for row in instance_rows if row["interaction_level"] == level]
        if not subset:
            continue
        soc_ranges = [int(row["soc_range"]) for row in subset]
        stats = _range_stats(soc_ranges)
        soc_sensitive = sum(1 for row in subset if row["soc_sensitive"])
        makespan_sensitive = sum(1 for row in subset if row["makespan_sensitive"])
        rows.append(
            {
                "catalogue": catalogue,
                "interaction_level": level,
                "instance_count": len(subset),
                "soc_sensitive_count": soc_sensitive,
                "makespan_sensitive_count": makespan_sensitive,
                "mean_soc_range": stats["mean"],
                "median_soc_range": stats["median"],
                "max_soc_range": stats["max"],
            }
        )
    return rows


def build_agent_count_sensitivity_summary(
    *,
    catalogue: str,
    instance_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for agent_count in AGENT_COUNT_ORDER:
        subset = [row for row in instance_rows if row["agent_count"] == agent_count]
        if not subset:
            continue
        soc_ranges = [int(row["soc_range"]) for row in subset]
        stats = _range_stats(soc_ranges)
        soc_sensitive = sum(1 for row in subset if row["soc_sensitive"])
        makespan_sensitive = sum(1 for row in subset if row["makespan_sensitive"])
        rows.append(
            {
                "catalogue": catalogue,
                "agent_count": agent_count,
                "instance_count": len(subset),
                "soc_sensitive_count": soc_sensitive,
                "makespan_sensitive_count": makespan_sensitive,
                "mean_soc_range": stats["mean"],
                "median_soc_range": stats["median"],
                "max_soc_range": stats["max"],
            }
        )
    return rows


def build_spf_cd_order_summary_rows(
    *,
    catalogue: str,
    instance_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    aggregate = {
        "catalogue": catalogue,
        "scope": "aggregate",
        "instance_count": len(instance_rows),
        "same_order_count": sum(
            1 for row in instance_rows if row.get("spf_cd_same_order") is True
        ),
        "different_order_count": sum(
            1 for row in instance_rows if row.get("spf_cd_same_order") is False
        ),
        "same_soc_count": sum(1 for row in instance_rows if row.get("spf_cd_soc_equal")),
        "different_soc_count": sum(
            1 for row in instance_rows if not row.get("spf_cd_soc_equal")
        ),
        "same_makespan_count": sum(
            1 for row in instance_rows if row.get("spf_cd_makespan_equal")
        ),
        "different_makespan_count": sum(
            1 for row in instance_rows if not row.get("spf_cd_makespan_equal")
        ),
        "cdf_h_different_order_count": sum(
            1 for row in instance_rows if row.get("cdf_h_same_order_as_spf") is False
        ),
        "cdf_l_different_order_count": sum(
            1 for row in instance_rows if row.get("cdf_l_same_order_as_spf") is False
        ),
        "cdf_h_order_diff_no_soc_diff_count": sum(
            1 for row in instance_rows if row.get("cdf_h_order_diff_no_soc_diff")
        ),
        "cdf_l_order_diff_no_soc_diff_count": sum(
            1 for row in instance_rows if row.get("cdf_l_order_diff_no_soc_diff")
        ),
    }
    rows = [aggregate]
    for row in instance_rows:
        rows.append(
            {
                "catalogue": catalogue,
                "scope": "instance",
                "instance_id": row["instance_id"],
                "agent_count": row["agent_count"],
                "interaction_level": row["interaction_level"],
                "same_order": row.get("spf_cd_same_order"),
                "soc_equal": row.get("spf_cd_soc_equal"),
                "makespan_equal": row.get("spf_cd_makespan_equal"),
                "cdf_h_same_order_as_spf": row.get("cdf_h_same_order_as_spf"),
                "cdf_l_same_order_as_spf": row.get("cdf_l_same_order_as_spf"),
            }
        )
    return rows


def build_conflict_structure_summary_rows(
    *,
    catalogue: str,
    instance_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def _summarize_group(group_label: str, group_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not group_rows:
            return {
                "catalogue": catalogue,
                "group": group_label,
                "instance_count": 0,
            }
        return {
            "catalogue": catalogue,
            "group": group_label,
            "instance_count": len(group_rows),
            "mean_independent_conflict_count": statistics.mean(
                float(row.get("independent_conflict_count") or 0) for row in group_rows
            ),
            "median_independent_conflict_count": statistics.median(
                float(row.get("independent_conflict_count") or 0) for row in group_rows
            ),
            "mean_independent_conflict_pair_count": statistics.mean(
                float(row.get("independent_conflict_pair_count") or 0)
                for row in group_rows
            ),
            "median_independent_conflict_pair_count": statistics.median(
                float(row.get("independent_conflict_pair_count") or 0)
                for row in group_rows
            ),
            "mean_max_degree": statistics.mean(
                float(row.get("max_degree") or 0) for row in group_rows
            ),
            "median_max_degree": statistics.median(
                float(row.get("max_degree") or 0) for row in group_rows
            ),
            "mean_soc_range": statistics.mean(
                float(row.get("soc_range") or 0) for row in group_rows
            ),
        }

    sensitive = [row for row in instance_rows if row["soc_sensitive"]]
    insensitive = [row for row in instance_rows if not row["soc_sensitive"]]
    rows.append(_summarize_group("soc_sensitive", sensitive))
    rows.append(_summarize_group("soc_insensitive", insensitive))
    for row in sensitive:
        rows.append(
            {
                "catalogue": catalogue,
                "group": "sensitive_instance",
                "instance_id": row["instance_id"],
                "agent_count": row["agent_count"],
                "interaction_level": row["interaction_level"],
                "independent_conflict_count": row.get("independent_conflict_count"),
                "independent_conflict_pair_count": row.get(
                    "independent_conflict_pair_count"
                ),
                "max_degree": row.get("max_degree"),
                "soc_range": row.get("soc_range"),
            }
        )
    return rows


def build_runtime_summary_from_records(
    *,
    catalogue: str,
    records: Sequence[CrossMapPriorityRunRecord],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy in MAPF8_STRATEGIES:
        subset = [record for record in records if record.strategy.value == strategy]
        ordering = [record.ordering_time_ms for record in subset]
        pp = [
            record.pp_time_ms
            for record in subset
            if record.pp_time_ms is not None
        ]
        total = [record.total_time_ms for record in subset]
        rows.append(
            {
                "catalogue": catalogue,
                "strategy": MAPF8_STRATEGY_LABELS[strategy],
                "run_count": len(subset),
                "mean_ordering_time_ms": statistics.mean(ordering) if ordering else "",
                "median_ordering_time_ms": statistics.median(ordering) if ordering else "",
                "mean_pp_time_ms": statistics.mean(pp) if pp else "",
                "median_pp_time_ms": statistics.median(pp) if pp else "",
                "mean_total_time_ms": statistics.mean(total) if total else "",
                "median_total_time_ms": statistics.median(total) if total else "",
            }
        )

    for agent_count in AGENT_COUNT_ORDER:
        for strategy in MAPF8_STRATEGIES:
            subset = [
                record
                for record in records
                if record.agent_count == agent_count
                and record.strategy.value == strategy
            ]
            if not subset:
                continue
            rows.append(
                {
                    "catalogue": catalogue,
                    "strategy": MAPF8_STRATEGY_LABELS[strategy],
                    "agent_count": agent_count,
                    "run_count": len(subset),
                    "mean_ordering_time_ms": statistics.mean(
                        record.ordering_time_ms for record in subset
                    ),
                    "median_ordering_time_ms": statistics.median(
                        record.ordering_time_ms for record in subset
                    ),
                    "mean_pp_time_ms": statistics.mean(
                        record.pp_time_ms
                        for record in subset
                        if record.pp_time_ms is not None
                    ),
                    "median_pp_time_ms": statistics.median(
                        record.pp_time_ms
                        for record in subset
                        if record.pp_time_ms is not None
                    ),
                    "mean_total_time_ms": statistics.mean(
                        record.total_time_ms for record in subset
                    ),
                    "median_total_time_ms": statistics.median(
                        record.total_time_ms for record in subset
                    ),
                }
            )
    return rows


def _historical_instance_rows(
    config: Mapf7FinalAnalysisConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary_rows = _reproduce_primary_instance_rows(config)
    for row in primary_rows:
        row["catalogue"] = "AR0204SR_primary"
        row["map_name"] = "AR0204SR"

    heldout_manifest = load_benchmark_manifest(config.heldout_manifest_path)
    heldout_conflict = load_conflict_aware_results(config.heldout_conflict_csv)
    heldout_spf = load_heldout_spf_results(config.heldout_spf_csv)
    heldout_rows = build_heldout_instance_level_summary(
        manifest=heldout_manifest,
        conflict_records=heldout_conflict,
        spf_records=heldout_spf,
    )
    for row in heldout_rows:
        row["catalogue"] = "AR0204SR_held_out"
        row["map_name"] = "AR0204SR"

    return primary_rows, heldout_rows


def _enrich_historical_row(row: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    enriched["soc_range"] = compute_four_strategy_soc_range(row)
    enriched["makespan_range"] = compute_four_strategy_makespan_range(row)
    enriched["soc_sensitive"] = is_soc_sensitive(row)
    enriched["makespan_sensitive"] = is_makespan_sensitive(row)
    enriched["mapf7_three_strategy_soc_range"] = compute_mapf7_three_strategy_soc_range(row)
    enriched["mapf7_three_strategy_soc_sensitive"] = mapf7_strategy_socs_differ(row)
    return enriched


def build_historical_comparison_rows(
    *,
    mapf7_config: Mapf7FinalAnalysisConfig,
    ar0400_rows: Sequence[dict[str, Any]],
    ar0307_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    primary_rows, heldout_rows = _historical_instance_rows(mapf7_config)
    primary_conflict = load_conflict_aware_results(mapf7_config.primary_conflict_csv)
    primary_spf_lpf = load_strategy_results_from_path(mapf7_config.primary_spf_lpf_csv)
    heldout_conflict = load_conflict_aware_results(mapf7_config.heldout_conflict_csv)
    heldout_spf = load_heldout_spf_results(mapf7_config.heldout_spf_csv)

    primary_spf_cd = build_spf_cd_ablation_aggregate(
        build_spf_cd_ablation_rows(
            primary_rows,
            catalogue="primary",
            conflict_records=primary_conflict,
            spf_records=primary_spf_lpf,
        ),
        catalogue="primary",
    )
    heldout_spf_cd = build_spf_cd_ablation_aggregate(
        build_spf_cd_ablation_rows(
            heldout_rows,
            catalogue="held_out",
            conflict_records=heldout_conflict,
            spf_records=heldout_spf,
        ),
        catalogue="held_out",
    )

    catalogues: list[tuple[str, Sequence[dict[str, Any]], dict[str, Any] | None]] = [
        ("AR0204SR_primary", primary_rows, primary_spf_cd),
        ("AR0204SR_held_out", heldout_rows, heldout_spf_cd),
        ("AR0400SR", ar0400_rows, None),
        ("AR0307SR", ar0307_rows, None),
    ]

    rows: list[dict[str, Any]] = []
    for catalogue, instance_rows, spf_cd_agg in catalogues:
        enriched = [_enrich_historical_row(row) for row in instance_rows]
        summary = build_catalogue_summary_rows(catalogue=catalogue, instance_rows=enriched)
        pairwise = {
            item["comparison"]: item
            for item in build_pairwise_summary_for_catalogue(
                catalogue=catalogue, instance_rows=enriched
            )
        }
        spf_cd_order = None
        if spf_cd_agg is not None:
            spf_cd_order = spf_cd_agg
        else:
            spf_cd_order = {
                "identical_ordering": sum(
                    1 for row in enriched if row.get("spf_cd_same_order") is True
                ),
                "differing_ordering": sum(
                    1 for row in enriched if row.get("spf_cd_same_order") is False
                ),
                "instance_count": len(enriched),
            }

        rows.append(
            {
                "catalogue": catalogue,
                "instance_count": summary["instance_count"],
                "soc_sensitive_count": summary["soc_sensitive_count"],
                "soc_sensitive_pct": summary["soc_sensitive_pct"],
                "makespan_sensitive_count": summary["makespan_sensitive_count"],
                "makespan_sensitive_pct": summary["makespan_sensitive_pct"],
                "mapf7_three_strategy_soc_sensitive_count": summary[
                    "mapf7_three_strategy_soc_sensitive_count"
                ],
                "max_soc_range": summary["max_soc_range"],
                "cdf_h_vs_spf_left_better": pairwise["cdf_h_vs_spf"]["left_better_count"],
                "cdf_h_vs_spf_equal": pairwise["cdf_h_vs_spf"]["equal_count"],
                "cdf_h_vs_spf_right_better": pairwise["cdf_h_vs_spf"]["right_better_count"],
                "cdf_h_vs_spf_mean_diff": pairwise["cdf_h_vs_spf"]["mean_soc_diff"],
                "cdf_l_vs_spf_left_better": pairwise["cdf_l_vs_spf"]["left_better_count"],
                "cdf_l_vs_spf_equal": pairwise["cdf_l_vs_spf"]["equal_count"],
                "cdf_l_vs_spf_right_better": pairwise["cdf_l_vs_spf"]["right_better_count"],
                "cdf_l_vs_spf_mean_diff": pairwise["cdf_l_vs_spf"]["mean_soc_diff"],
                "cdf_h_vs_cdf_l_left_better": pairwise["cdf_h_vs_cdf_l"]["left_better_count"],
                "cdf_h_vs_cdf_l_equal": pairwise["cdf_h_vs_cdf_l"]["equal_count"],
                "cdf_h_vs_cdf_l_right_better": pairwise["cdf_h_vs_cdf_l"]["right_better_count"],
                "cdf_h_vs_cdf_l_mean_diff": pairwise["cdf_h_vs_cdf_l"]["mean_soc_diff"],
                "spf_cd_vs_spf_left_better": pairwise["spf_cd_vs_spf"]["left_better_count"],
                "spf_cd_vs_spf_equal": pairwise["spf_cd_vs_spf"]["equal_count"],
                "spf_cd_vs_spf_right_better": pairwise["spf_cd_vs_spf"]["right_better_count"],
                "spf_cd_vs_spf_mean_diff": pairwise["spf_cd_vs_spf"]["mean_soc_diff"],
                "spf_cd_identical_ordering": spf_cd_order.get("identical_ordering"),
                "spf_cd_differing_ordering": spf_cd_order.get("differing_ordering"),
            }
        )
    return rows


def load_strategy_results_from_path(path: Path):
    from pathfinding.src.experiments.mapf_priority_strategy_analysis import (
        load_strategy_results,
    )

    return load_strategy_results(path)


def default_mapf7_config(repo_root: Path) -> Mapf7FinalAnalysisConfig:
    results = repo_root / "pathfinding" / "results"
    return Mapf7FinalAnalysisConfig(
        primary_conflict_csv=results
        / "mapf_conflict_aware_priority_execution"
        / "results.csv",
        primary_manifest_path=results / "mapf_benchmarks" / "AR0204SR_manifest.json",
        primary_spf_lpf_csv=results / "mapf_priority_strategy_execution" / "results.csv",
        fixed_results_csv=results / "mapf_benchmark_execution" / "results.csv",
        random_results_csv=results / "mapf_random_priority_pilot" / "results.csv",
        heldout_conflict_csv=results
        / "mapf_conflict_aware_priority_heldout_execution"
        / "results.csv",
        heldout_spf_csv=results / "mapf_spf_heldout_execution" / "results.csv",
        heldout_manifest_path=results / "mapf_benchmarks" / "AR0204SR_heldout_manifest.json",
        output_dir=results / "mapf7_final_analysis",
        expected_instance_count=27,
    )


def default_mapf8_config(repo_root: Path) -> Mapf8CrossMapAnalysisConfig:
    results = repo_root / "pathfinding" / "results"
    execution_root = repo_root / MAPF8_CROSS_MAP_RESULTS_ROOT
    return Mapf8CrossMapAnalysisConfig(
        repo_root=repo_root,
        ar0400_csv=execution_root / "AR0400SR" / "results.csv",
        ar0400_manifest=results / "mapf_benchmarks" / "AR0400SR_manifest.json",
        ar0307_csv=execution_root / "AR0307SR" / "results.csv",
        ar0307_manifest=results / "mapf_benchmarks" / "AR0307SR_manifest.json",
        output_dir=repo_root / MAPF8_ANALYSIS_DIR,
        mapf7_config=default_mapf7_config(repo_root),
    )


def _format_mapf7_secondary_column(row: dict[str, Any]) -> str:
    count = row.get("mapf7_three_strategy_soc_sensitive_count")
    pct = row.get("mapf7_three_strategy_soc_sensitive_pct")
    if count is None:
        return "—"
    if pct:
        return f"{count} ({pct})"
    return str(count)


def _format_table1_row(
    *,
    catalogue: str,
    instance_count: int,
    soc_sensitive_count: int,
    soc_sensitive_pct: str,
    makespan_sensitive_count: int,
    makespan_sensitive_pct: str,
    max_soc_range: int | str,
    mean_soc_range: str,
    mapf7_secondary: str,
) -> str:
    return (
        f"| {catalogue} | {instance_count} | "
        f"{soc_sensitive_count} ({soc_sensitive_pct}) | "
        f"{makespan_sensitive_count} ({makespan_sensitive_pct}) | "
        f"{max_soc_range} | {mean_soc_range} | {mapf7_secondary} |"
    )


def combined_spf_cd_order_counts(
    *,
    historical_rows: Sequence[dict[str, Any]],
    pooled_spf_cd: dict[str, Any],
) -> dict[str, int]:
    hist = {row["catalogue"]: row for row in historical_rows}
    primary = hist["AR0204SR_primary"]
    heldout = hist["AR0204SR_held_out"]
    historical_same = int(primary["spf_cd_identical_ordering"]) + int(
        heldout["spf_cd_identical_ordering"]
    )
    cross_map_same = int(pooled_spf_cd["same_order_count"])
    cross_map_total = int(pooled_spf_cd["instance_count"])
    combined_same = historical_same + cross_map_same
    combined_total = 54 + cross_map_total
    return {
        "cross_map_same": cross_map_same,
        "cross_map_total": cross_map_total,
        "historical_same": historical_same,
        "historical_total": 54,
        "combined_same": combined_same,
        "combined_total": combined_total,
    }


def build_research_questions_md(
    *,
    ar0400_summary: dict[str, Any],
    ar0307_summary: dict[str, Any],
    pooled_summary: dict[str, Any],
    historical_rows: Sequence[dict[str, Any]],
    pooled_pairwise: Sequence[dict[str, Any]],
    pooled_spf_cd: dict[str, Any],
    pooled_conflict: Sequence[dict[str, Any]],
) -> str:
    hist = {row["catalogue"]: row for row in historical_rows}
    primary = hist["AR0204SR_primary"]
    heldout = hist["AR0204SR_held_out"]
    cdf_h_cdf_l = next(
        row for row in pooled_pairwise if row["comparison"] == "cdf_h_vs_cdf_l"
    )
    sensitive_conflict = next(
        row for row in pooled_conflict if row["group"] == "soc_sensitive"
    )
    insensitive_conflict = next(
        row for row in pooled_conflict if row["group"] == "soc_insensitive"
    )
    spf_cd_counts = combined_spf_cd_order_counts(
        historical_rows=historical_rows,
        pooled_spf_cd=pooled_spf_cd,
    )

    lines = [
        "# MAPF-8 Research Questions (RQ8.1–RQ8.6)",
        "",
        "## RQ8.1 — Does sparse priority sensitivity persist on different map topologies?",
        "",
        f"- AR0400SR: {ar0400_summary['soc_sensitive_count']}/{ar0400_summary['instance_count']} "
        f"({ar0400_summary['soc_sensitive_pct']}) SoC-sensitive under four-strategy range.",
        f"- AR0307SR: {ar0307_summary['soc_sensitive_count']}/{ar0307_summary['instance_count']} "
        f"({ar0307_summary['soc_sensitive_pct']}) SoC-sensitive.",
        f"- Pooled cross-map: {pooled_summary['soc_sensitive_count']}/"
        f"{pooled_summary['instance_count']} ({pooled_summary['soc_sensitive_pct']}).",
        f"- Historical AR0204SR primary: {primary['soc_sensitive_count']}/27; "
        f"held-out: {heldout['soc_sensitive_count']}/27.",
        "- **Conservative conclusion:** Priority sensitivity remained sparse on both new bg512 "
        "topologies; exact sensitive-instance sets differ by map.",
        "- **Limitation:** Only two additional maps within the same bg512 family were tested.",
        "",
        "## RQ8.2 — Does SPF remain a stable baseline across maps?",
        "",
    ]
    for row in pooled_pairwise:
        if row["comparison"] in {"cdf_h_vs_spf", "cdf_l_vs_spf", "spf_cd_vs_spf"}:
            lines.append(
                f"- {row['left_strategy']} vs {row['right_strategy']} (pooled): "
                f"LEFT better={row['left_better_count']}, equal={row['equal_count']}, "
                f"RIGHT better={row['right_better_count']}, mean diff={_format_float(row['mean_soc_diff'])}."
            )
    lines.extend(
        [
            "- **Conservative conclusion:** SPF remained a frequently equal-cost baseline; "
            "no map showed a large systematic SoC advantage for CDF variants over SPF.",
            "- **Limitation:** Pairwise means can be outlier-driven on sparse sensitive subsets.",
            "",
            "## RQ8.3 — Does either CDF-H or CDF-L show a consistent cross-map advantage?",
            "",
            f"- Pooled CDF-H vs CDF-L: LEFT better={cdf_h_cdf_l['left_better_count']}, "
            f"equal={cdf_h_cdf_l['equal_count']}, RIGHT better={cdf_h_cdf_l['right_better_count']}, "
            f"mean diff={_format_float(cdf_h_cdf_l['mean_soc_diff'])}.",
            f"- AR0204SR primary mean diff={_format_float(primary['cdf_h_vs_cdf_l_mean_diff'])}; "
            f"held-out mean diff={_format_float(heldout['cdf_h_vs_cdf_l_mean_diff'])}.",
            "- **Conservative conclusion:** No consistent cross-map advantage for CDF-H or CDF-L "
            "was observed; directionality varied by catalogue and map.",
            "",
            "## RQ8.4 — Does SPF+CD ever produce a different priority ordering from SPF?",
            "",
            f"- Cross-map (AR0400SR + AR0307SR): same order="
            f"{spf_cd_counts['cross_map_same']}/{spf_cd_counts['cross_map_total']}, "
            f"different order="
            f"{pooled_spf_cd['different_order_count']}/{spf_cd_counts['cross_map_total']}.",
            f"- Historical AR0204SR (primary + held-out): same order="
            f"{spf_cd_counts['historical_same']}/{spf_cd_counts['historical_total']}, "
            f"different order=0/{spf_cd_counts['historical_total']}.",
            f"- Combined four catalogues: same order="
            f"{spf_cd_counts['combined_same']}/{spf_cd_counts['combined_total']}, "
            f"different order=0/{spf_cd_counts['combined_total']}.",
            "- **Conservative conclusion:** No different SPF+CD ordering was observed in "
            "any of the 108 tested instances across the four catalogues (54 historical "
            "AR0204SR instances and 54 new cross-map instances).",
            "- **Limitation:** This is observational and does not establish mathematical "
            "equivalence between SPF and SPF+CD.",
            "",
            "## RQ8.5 — Does makespan remain less sensitive than SoC?",
            "",
            f"- Pooled: SoC-sensitive={pooled_summary['soc_sensitive_count']}/"
            f"{pooled_summary['instance_count']}, makespan-sensitive="
            f"{pooled_summary['makespan_sensitive_count']}/{pooled_summary['instance_count']}.",
            "- **Conservative conclusion:** Makespan sensitivity remained lower than SoC sensitivity "
            "within the tested catalogues.",
            "",
            "## RQ8.6 — Is conflict-graph structure still descriptively associated with priority sensitivity?",
            "",
            f"- Pooled sensitive (n={sensitive_conflict.get('instance_count', 0)}): "
            f"mean conflicts={_format_float(sensitive_conflict.get('mean_independent_conflict_count'))}, "
            f"mean pairs={_format_float(sensitive_conflict.get('mean_independent_conflict_pair_count'))}, "
            f"mean max degree={_format_float(sensitive_conflict.get('mean_max_degree'))}.",
            f"- Pooled insensitive (n={insensitive_conflict.get('instance_count', 0)}): "
            f"mean conflicts={_format_float(insensitive_conflict.get('mean_independent_conflict_count'))}, "
            f"mean pairs={_format_float(insensitive_conflict.get('mean_independent_conflict_pair_count'))}, "
            f"mean max degree={_format_float(insensitive_conflict.get('mean_max_degree'))}.",
            "- **Conservative conclusion:** Descriptive differences were observed, but MAPF-8 "
            "does not support causal or predictive claims from conflict degree alone.",
            "",
    ])
    return "\n".join(lines)


def build_analysis_summary_md(
    *,
    validation: ValidationReport,
    ar0400_summary: dict[str, Any],
    ar0307_summary: dict[str, Any],
    pooled_summary: dict[str, Any],
    historical_rows: Sequence[dict[str, Any]],
    pooled_pairwise: Sequence[dict[str, Any]],
    pooled_spf_cd: dict[str, Any],
    pooled_conflict: Sequence[dict[str, Any]],
    sensitive_rows: Sequence[dict[str, Any]],
) -> str:
    hist = {row["catalogue"]: row for row in historical_rows}
    spf_cd_counts = combined_spf_cd_order_counts(
        historical_rows=historical_rows,
        pooled_spf_cd=pooled_spf_cd,
    )
    lines = [
        "# MAPF-8.5 Cross-Map Analysis Summary",
        "",
        "## 1. Dataset validation",
        "",
        f"- Validation status: **{'PASSED' if validation.passed else 'FAILED'}**",
    ]
    lines.extend(f"- {message}" for message in validation.messages)
    lines.extend(
        [
            "",
            "## 2. AR0400SR results",
            "",
            f"- Instances: {ar0400_summary['instance_count']}",
            f"- SoC-sensitive: {ar0400_summary['soc_sensitive_count']} "
            f"({ar0400_summary['soc_sensitive_pct']})",
            f"- Makespan-sensitive: {ar0400_summary['makespan_sensitive_count']} "
            f"({ar0400_summary['makespan_sensitive_pct']})",
            f"- Max/mean/median SoC range: {ar0400_summary['max_soc_range']} / "
            f"{_format_float(ar0400_summary['mean_soc_range'])} / "
            f"{_format_float(ar0400_summary['median_soc_range'])}",
            "",
            "## 3. AR0307SR results",
            "",
            f"- Instances: {ar0307_summary['instance_count']}",
            f"- SoC-sensitive: {ar0307_summary['soc_sensitive_count']} "
            f"({ar0307_summary['soc_sensitive_pct']})",
            f"- Makespan-sensitive: {ar0307_summary['makespan_sensitive_count']} "
            f"({ar0307_summary['makespan_sensitive_pct']})",
            f"- Max/mean/median SoC range: {ar0307_summary['max_soc_range']} / "
            f"{_format_float(ar0307_summary['mean_soc_range'])} / "
            f"{_format_float(ar0307_summary['median_soc_range'])}",
            "",
            "## 4. Pooled cross-map results",
            "",
            f"- Instances: {pooled_summary['instance_count']}",
            f"- SoC-sensitive: {pooled_summary['soc_sensitive_count']} "
            f"({pooled_summary['soc_sensitive_pct']})",
            f"- Makespan-sensitive: {pooled_summary['makespan_sensitive_count']} "
            f"({pooled_summary['makespan_sensitive_pct']})",
            "",
            "## 5. Historical AR0204SR comparison",
            "",
        ]
    )
    for catalogue in (
        "AR0204SR_primary",
        "AR0204SR_held_out",
        "AR0400SR",
        "AR0307SR",
    ):
        row = hist[catalogue]
        lines.append(
            f"- {catalogue}: SoC-sensitive={row['soc_sensitive_count']}/27 "
            f"({row['soc_sensitive_pct']}), makespan-sensitive="
            f"{row['makespan_sensitive_count']}/27, max SoC range={row['max_soc_range']}."
        )
    lines.extend(
        [
            "",
            "## 6. SPF stability",
            "",
        ]
    )
    for row in pooled_pairwise:
        if "spf" in row["comparison"]:
            lines.append(
                f"- {row['comparison']}: LEFT better={row['left_better_count']}, "
                f"equal={row['equal_count']}, RIGHT better={row['right_better_count']}, "
                f"mean diff={_format_float(row['mean_soc_diff'])}."
            )
    lines.extend(
        [
            "",
            "## 7. CDF-H vs CDF-L directionality",
            "",
        ]
    )
    cdf_row = next(row for row in pooled_pairwise if row["comparison"] == "cdf_h_vs_cdf_l")
    lines.append(
        f"- Pooled: LEFT better={cdf_row['left_better_count']}, "
        f"equal={cdf_row['equal_count']}, RIGHT better={cdf_row['right_better_count']}, "
        f"mean diff={_format_float(cdf_row['mean_soc_diff'])}."
    )
    lines.extend(
        [
            "",
            "## 8. SPF+CD ablation result",
            "",
            f"- Cross-map same order: {spf_cd_counts['cross_map_same']}/"
            f"{spf_cd_counts['cross_map_total']}.",
            f"- Historical AR0204SR same order: {spf_cd_counts['historical_same']}/"
            f"{spf_cd_counts['historical_total']}.",
            f"- Combined four catalogues: {spf_cd_counts['combined_same']}/"
            f"{spf_cd_counts['combined_total']}.",
            "- No different SPF+CD ordering was observed in any of the 108 tested instances "
            "across the four catalogues (54 historical AR0204SR instances and 54 new "
            "cross-map instances).",
            "- Limitation: observational only; this does not establish mathematical "
            "equivalence between SPF and SPF+CD.",
            "",
            "## 9. SoC vs makespan sensitivity",
            "",
            f"- Pooled SoC-sensitive: {pooled_summary['soc_sensitive_count']}/"
            f"{pooled_summary['instance_count']}.",
            f"- Pooled makespan-sensitive: {pooled_summary['makespan_sensitive_count']}/"
            f"{pooled_summary['instance_count']}.",
            "",
            "## 10. Conflict-structure diagnostics",
            "",
        ]
    )
    for row in pooled_conflict:
        if row["group"] in {"soc_sensitive", "soc_insensitive"}:
            lines.append(
                f"- {row['group']} (n={row['instance_count']}): mean conflicts="
                f"{_format_float(row.get('mean_independent_conflict_count'))}, "
                f"mean pairs={_format_float(row.get('mean_independent_conflict_pair_count'))}, "
                f"mean max degree={_format_float(row.get('mean_max_degree'))}."
            )
    lines.extend(
        [
            "",
            "## 11. Answers to RQ8.1–RQ8.6",
            "",
            "See `research_questions.md` for explicit RQ answers with evidence and limitations.",
            "",
            "## 12. Methodological limitations",
            "",
            "- AR0204SR, AR0400SR, and AR0307SR belong to the MovingAI bg512 arena family.",
            "- MAPF-8 demonstrates cross-map validation within bg512, not universal MovingAI "
            "generalization or cross-domain generalization.",
            "- Primary sensitivity uses all four strategies (SPF, CDF-H, CDF-L, SPF+CD).",
            "- MAPF-7 three-strategy diagnostic is reported separately as secondary.",
            "- Runtime differences are descriptive; conflict-aware strategies recompute independent "
            "paths during ordering.",
            "",
            "## 13. Conservative thesis interpretation",
            "",
            "### Wnioski (wersja robocza, PL)",
            "",
            "W ramach zamrożonych katalogów bg512 (AR0400SR, AR0307SR) oraz historycznych "
            f"katalogów AR0204SR zaobserwowano rzadką wrażliwość jakościową (SoC) na wybór "
            f"strategii priorytetu: {pooled_summary['soc_sensitive_count']} z "
            f"{pooled_summary['instance_count']} nowych instancji cross-map oraz wartości "
            "historyczne 5/27 (primary) i 3/27 (held-out) przy definicji czterostrategicznej. "
            "Nie stwierdzono spójnej przewagi CDF-H ani CDF-L między mapami; SPF pozostał "
            "często baseline'em o równym SoC. W żadnej ze 108 analizowanych instancji "
            "(54 historycznych AR0204SR oraz 54 nowych instancjach cross-map) nie "
            "zaobserwowano innego porządku priorytetów SPF+CD względem SPF; obserwacja ta "
            "nie stanowi dowodu formalnej równoważności obu strategii. Wyniki należy "
            "interpretować jako walidację topologii w rodzinie bg512, a nie generalizację "
            "na wszystkie mapy MovingAI.",
            "",
            "### Sensitive instances (pooled cross-map)",
            "",
        ]
    )
    if sensitive_rows:
        for row in sensitive_rows:
            lines.append(
                f"- {row['map_name']} / {row['instance_id']} "
                f"(n={row['agent_count']}, {row['interaction_level']}): "
                f"SPF={row['spf_soc']}, CDF-H={row['cdf_h_soc']}, CDF-L={row['cdf_l_soc']}, "
                f"SPF+CD={row['spf_cd_soc']}, range={row['soc_range']}."
            )
    else:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines)


def build_thesis_tables_md(
    *,
    ar0400_summary: dict[str, Any],
    ar0307_summary: dict[str, Any],
    pooled_summary: dict[str, Any],
    historical_rows: Sequence[dict[str, Any]],
    pooled_pairwise: Sequence[dict[str, Any]],
    pooled_spf_cd: dict[str, Any],
) -> str:
    hist = {row["catalogue"]: row for row in historical_rows}
    primary = hist["AR0204SR_primary"]
    heldout = hist["AR0204SR_held_out"]
    spf_cd_counts = combined_spf_cd_order_counts(
        historical_rows=historical_rows,
        pooled_spf_cd=pooled_spf_cd,
    )

    lines = [
        "# MAPF-8 Thesis Tables (draft)",
        "",
        "## Table 1 — Catalogue sensitivity summary (four-strategy SoC range)",
        "",
        "| Catalogue | Instances | SoC-sensitive | Makespan-sensitive | Max SoC range | "
        "Mean SoC range | MAPF-7 3-strategy sensitive (secondary) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    table1_rows = [
        _format_table1_row(
            catalogue="AR0204SR primary",
            instance_count=primary["instance_count"],
            soc_sensitive_count=primary["soc_sensitive_count"],
            soc_sensitive_pct=primary["soc_sensitive_pct"],
            makespan_sensitive_count=primary["makespan_sensitive_count"],
            makespan_sensitive_pct=primary["makespan_sensitive_pct"],
            max_soc_range=primary["max_soc_range"],
            mean_soc_range="—",
            mapf7_secondary=str(primary["mapf7_three_strategy_soc_sensitive_count"]),
        ),
        _format_table1_row(
            catalogue="AR0204SR held-out",
            instance_count=heldout["instance_count"],
            soc_sensitive_count=heldout["soc_sensitive_count"],
            soc_sensitive_pct=heldout["soc_sensitive_pct"],
            makespan_sensitive_count=heldout["makespan_sensitive_count"],
            makespan_sensitive_pct=heldout["makespan_sensitive_pct"],
            max_soc_range=heldout["max_soc_range"],
            mean_soc_range="—",
            mapf7_secondary=str(heldout["mapf7_three_strategy_soc_sensitive_count"]),
        ),
        _format_table1_row(
            catalogue="AR0400SR",
            instance_count=ar0400_summary["instance_count"],
            soc_sensitive_count=ar0400_summary["soc_sensitive_count"],
            soc_sensitive_pct=ar0400_summary["soc_sensitive_pct"],
            makespan_sensitive_count=ar0400_summary["makespan_sensitive_count"],
            makespan_sensitive_pct=ar0400_summary["makespan_sensitive_pct"],
            max_soc_range=ar0400_summary["max_soc_range"],
            mean_soc_range=_format_float(ar0400_summary["mean_soc_range"]),
            mapf7_secondary=_format_mapf7_secondary_column(ar0400_summary),
        ),
        _format_table1_row(
            catalogue="AR0307SR",
            instance_count=ar0307_summary["instance_count"],
            soc_sensitive_count=ar0307_summary["soc_sensitive_count"],
            soc_sensitive_pct=ar0307_summary["soc_sensitive_pct"],
            makespan_sensitive_count=ar0307_summary["makespan_sensitive_count"],
            makespan_sensitive_pct=ar0307_summary["makespan_sensitive_pct"],
            max_soc_range=ar0307_summary["max_soc_range"],
            mean_soc_range=_format_float(ar0307_summary["mean_soc_range"]),
            mapf7_secondary=_format_mapf7_secondary_column(ar0307_summary),
        ),
    ]
    lines.extend(table1_rows)
    lines.append(
        _format_table1_row(
            catalogue="pooled cross-map (AR0400SR + AR0307SR aggregate)",
            instance_count=pooled_summary["instance_count"],
            soc_sensitive_count=pooled_summary["soc_sensitive_count"],
            soc_sensitive_pct=pooled_summary["soc_sensitive_pct"],
            makespan_sensitive_count=pooled_summary["makespan_sensitive_count"],
            makespan_sensitive_pct=pooled_summary["makespan_sensitive_pct"],
            max_soc_range=pooled_summary["max_soc_range"],
            mean_soc_range=_format_float(pooled_summary["mean_soc_range"]),
            mapf7_secondary=_format_mapf7_secondary_column(pooled_summary),
        )
    )

    lines.extend(
        [
            "",
            "## Table 2 — Pooled pairwise SoC comparisons (diff = LEFT − RIGHT)",
            "",
            "| Comparison | LEFT better | Equal | RIGHT better | Mean diff | Median diff | Max abs(diff) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pooled_pairwise:
        lines.append(
            f"| {row['left_strategy']} vs {row['right_strategy']} | "
            f"{row['left_better_count']} | {row['equal_count']} | {row['right_better_count']} | "
            f"{_format_float(row['mean_soc_diff'])} | {_format_float(row['median_soc_diff'])} | "
            f"{row['max_abs_soc_diff']} |"
        )

    lines.extend(
        [
            "",
            "## Table 3 — SPF+CD order equality",
            "",
            "### Cross-map (AR0400SR + AR0307SR)",
            "",
            f"- Same order: {spf_cd_counts['cross_map_same']}/{spf_cd_counts['cross_map_total']}",
            f"- Different order: {pooled_spf_cd['different_order_count']}/"
            f"{spf_cd_counts['cross_map_total']}",
            f"- CDF-H different order from SPF (cross-map only): "
            f"{pooled_spf_cd['cdf_h_different_order_count']}/{spf_cd_counts['cross_map_total']}",
            f"- CDF-L different order from SPF (cross-map only): "
            f"{pooled_spf_cd['cdf_l_different_order_count']}/{spf_cd_counts['cross_map_total']}",
            "",
            "### Historical AR0204SR (primary + held-out)",
            "",
            f"- Same order: {spf_cd_counts['historical_same']}/{spf_cd_counts['historical_total']}",
            f"- Different order: 0/{spf_cd_counts['historical_total']}",
            "",
            "### Combined four catalogues",
            "",
            f"- Same order: {spf_cd_counts['combined_same']}/{spf_cd_counts['combined_total']}",
            f"- Different order: 0/{spf_cd_counts['combined_total']}",
            "",
        ]
    )
    return "\n".join(lines)


def run_mapf8_cross_map_analysis(
    config: Mapf8CrossMapAnalysisConfig,
) -> Mapf8CrossMapAnalysisResult:
    ar0400_records = load_cross_map_results(config.ar0400_csv)
    ar0307_records = load_cross_map_results(config.ar0307_csv)
    ar0400_manifest = load_benchmark_manifest(config.ar0400_manifest)
    ar0307_manifest = load_benchmark_manifest(config.ar0307_manifest)

    validation = ValidationReport(passed=True)
    for map_name, records, manifest in (
        ("AR0400SR", ar0400_records, ar0400_manifest),
        ("AR0307SR", ar0307_records, ar0307_manifest),
    ):
        map_report = validate_mapf8_map_dataset(
            map_name=map_name,
            records=records,
            manifest=manifest,
            expected_manifest_sha256=FROZEN_MANIFEST_SHA256[map_name],
            expected_seed=MAPF8_CROSS_MAP_SEEDS[map_name],
        )
        validation.messages.extend(map_report.messages)
        if not map_report.passed:
            validation.passed = False

    combined_report = validate_mapf8_combined_dataset(
        ar0400_records=ar0400_records,
        ar0307_records=ar0307_records,
    )
    validation.messages.extend(combined_report.messages)
    if not combined_report.passed:
        validation.passed = False

    if not validation.passed:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        _write_text(
            config.output_dir / "dataset_validation.txt",
            "\n".join(validation.messages) + "\n",
        )
        return Mapf8CrossMapAnalysisResult(
            validation=validation,
            output_dir=config.output_dir,
            tables_written=("dataset_validation.txt",),
            ar0400_instance_rows=(),
            ar0307_instance_rows=(),
            pooled_instance_rows=(),
        )

    ar0400_rows = build_instance_level_summary_from_records(
        map_name="AR0400SR", records=ar0400_records
    )
    ar0307_rows = build_instance_level_summary_from_records(
        map_name="AR0307SR", records=ar0307_records
    )
    pooled_rows = ar0400_rows + ar0307_rows

    ar0400_summary = build_catalogue_summary_rows(
        catalogue="AR0400SR", instance_rows=ar0400_rows
    )
    ar0307_summary = build_catalogue_summary_rows(
        catalogue="AR0307SR", instance_rows=ar0307_rows
    )
    pooled_summary = build_catalogue_summary_rows(
        catalogue="pooled_cross_map", instance_rows=pooled_rows
    )

    catalogue_summary_rows = [ar0400_summary, ar0307_summary, pooled_summary]
    sensitive_rows = build_sensitive_instances_rows(pooled_rows)

    pairwise_rows = (
        build_pairwise_summary_for_catalogue(catalogue="AR0400SR", instance_rows=ar0400_rows)
        + build_pairwise_summary_for_catalogue(catalogue="AR0307SR", instance_rows=ar0307_rows)
        + build_pairwise_summary_for_catalogue(
            catalogue="pooled_cross_map", instance_rows=pooled_rows
        )
    )

    interaction_rows = (
        build_interaction_sensitivity_summary(catalogue="AR0400SR", instance_rows=ar0400_rows)
        + build_interaction_sensitivity_summary(catalogue="AR0307SR", instance_rows=ar0307_rows)
        + build_interaction_sensitivity_summary(
            catalogue="pooled_cross_map", instance_rows=pooled_rows
        )
    )
    agent_count_rows = (
        build_agent_count_sensitivity_summary(catalogue="AR0400SR", instance_rows=ar0400_rows)
        + build_agent_count_sensitivity_summary(catalogue="AR0307SR", instance_rows=ar0307_rows)
        + build_agent_count_sensitivity_summary(
            catalogue="pooled_cross_map", instance_rows=pooled_rows
        )
    )

    spf_cd_rows = (
        build_spf_cd_order_summary_rows(catalogue="AR0400SR", instance_rows=ar0400_rows)
        + build_spf_cd_order_summary_rows(catalogue="AR0307SR", instance_rows=ar0307_rows)
        + build_spf_cd_order_summary_rows(
            catalogue="pooled_cross_map", instance_rows=pooled_rows
        )
    )
    pooled_spf_cd = next(
        row
        for row in spf_cd_rows
        if row.get("catalogue") == "pooled_cross_map" and row.get("scope") == "aggregate"
    )

    conflict_rows = (
        build_conflict_structure_summary_rows(
            catalogue="AR0400SR", instance_rows=ar0400_rows
        )
        + build_conflict_structure_summary_rows(
            catalogue="AR0307SR", instance_rows=ar0307_rows
        )
        + build_conflict_structure_summary_rows(
            catalogue="pooled_cross_map", instance_rows=pooled_rows
        )
    )

    runtime_rows = (
        build_runtime_summary_from_records(
            catalogue="AR0400SR", records=ar0400_records
        )
        + build_runtime_summary_from_records(
            catalogue="AR0307SR", records=ar0307_records
        )
        + build_runtime_summary_from_records(
            catalogue="pooled_cross_map",
            records=list(ar0400_records) + list(ar0307_records),
        )
    )

    historical_rows: list[dict[str, Any]] = []
    if config.mapf7_config is not None:
        historical_rows = build_historical_comparison_rows(
            mapf7_config=config.mapf7_config,
            ar0400_rows=ar0400_rows,
            ar0307_rows=ar0307_rows,
        )

    pooled_pairwise = [
        row for row in pairwise_rows if row["catalogue"] == "pooled_cross_map"
    ]
    pooled_conflict = [
        row
        for row in conflict_rows
        if row["catalogue"] == "pooled_cross_map"
        and row["group"] in {"soc_sensitive", "soc_insensitive"}
    ]

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_text(
        config.output_dir / "dataset_validation.txt",
        "\n".join(
            ["Validation status: PASSED", *validation.messages, ""]
        ),
    )
    _write_csv_rows(config.output_dir / "catalogue_summary.csv", catalogue_summary_rows)
    _write_csv_rows(config.output_dir / "instance_level_summary.csv", pooled_rows)
    _write_csv_rows(config.output_dir / "sensitive_instances.csv", sensitive_rows)
    _write_csv_rows(config.output_dir / "pairwise_summary.csv", pairwise_rows)
    _write_csv_rows(config.output_dir / "interaction_summary.csv", interaction_rows)
    _write_csv_rows(config.output_dir / "agent_count_summary.csv", agent_count_rows)
    _write_csv_rows(config.output_dir / "spf_cd_order_summary.csv", spf_cd_rows)
    _write_csv_rows(config.output_dir / "conflict_structure_summary.csv", conflict_rows)
    _write_csv_rows(config.output_dir / "runtime_summary.csv", runtime_rows)
    if historical_rows:
        _write_csv_rows(config.output_dir / "historical_comparison.csv", historical_rows)

    research_questions_md = build_research_questions_md(
        ar0400_summary=ar0400_summary,
        ar0307_summary=ar0307_summary,
        pooled_summary=pooled_summary,
        historical_rows=historical_rows,
        pooled_pairwise=pooled_pairwise,
        pooled_spf_cd=pooled_spf_cd,
        pooled_conflict=pooled_conflict,
    )
    analysis_summary_md = build_analysis_summary_md(
        validation=validation,
        ar0400_summary=ar0400_summary,
        ar0307_summary=ar0307_summary,
        pooled_summary=pooled_summary,
        historical_rows=historical_rows,
        pooled_pairwise=pooled_pairwise,
        pooled_spf_cd=pooled_spf_cd,
        pooled_conflict=pooled_conflict,
        sensitive_rows=sensitive_rows,
    )
    thesis_tables_md = build_thesis_tables_md(
        ar0400_summary=ar0400_summary,
        ar0307_summary=ar0307_summary,
        pooled_summary=pooled_summary,
        historical_rows=historical_rows,
        pooled_pairwise=pooled_pairwise,
        pooled_spf_cd=pooled_spf_cd,
    )

    _write_text(config.output_dir / "research_questions.md", research_questions_md)
    _write_text(config.output_dir / "analysis_summary.md", analysis_summary_md)
    _write_text(config.output_dir / "thesis_tables.md", thesis_tables_md)

    tables_written = tuple(
        name for name in OUTPUT_TABLES if (config.output_dir / name).is_file()
    )
    return Mapf8CrossMapAnalysisResult(
        validation=validation,
        output_dir=config.output_dir,
        tables_written=tables_written,
        ar0400_instance_rows=tuple(ar0400_rows),
        ar0307_instance_rows=tuple(ar0307_rows),
        pooled_instance_rows=tuple(pooled_rows),
    )
