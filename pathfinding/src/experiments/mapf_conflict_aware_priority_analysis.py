from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Iterable, Sequence
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pathfinding.src.experiments.mapf_benchmark_analysis import count_directional_soc_better
from pathfinding.src.experiments.mapf_benchmark_execution import TERMINATION_SUCCESS
from pathfinding.src.experiments.mapf_benchmark_instances import (
    MAPFBenchmarkManifest,
    load_benchmark_manifest,
)
from pathfinding.src.experiments.mapf_priority_strategy_analysis import (
    AGENT_COUNT_ORDER,
    INTERACTION_ORDER,
    RANDOM_K_DEFAULT,
    STRATEGY_CBS_BASIC,
    STRATEGY_FIXED,
    STRATEGY_LPF,
    STRATEGY_SPF,
    FixedRunRecord,
    RandomInstanceAggregate,
    RandomRunRecord,
    StrategyRunRecord,
    _cbs_by_instance,
    _directional_cbs_gaps,
    _fixed_by_instance,
    _format_float,
    _format_percent,
    _instance_sort_key,
    _interaction_sort_key,
    _ms_to_seconds,
    _parse_agent_order,
    _parse_bool,
    _parse_float,
    _parse_int,
    _read_csv_rows,
    _row_bool,
    _row_int,
    _summary_stats,
    _write_csv,
    _write_text,
    aggregate_random_by_instance,
    load_fixed_results,
    load_random_results,
    load_strategy_results,
)

STRATEGY_CDF_H = "cdf_h"
STRATEGY_CDF_L = "cdf_l"
STRATEGY_SPF_CD = "spf_cd"

MAPF7_STRATEGIES: tuple[str, ...] = (STRATEGY_CDF_H, STRATEGY_CDF_L, STRATEGY_SPF_CD)

TIME_TOLERANCE_MS = 1.0

MAIN_CONFLICT_AWARE_ANALYSIS_DIR = Path(
    "pathfinding/results/mapf_conflict_aware_priority_analysis"
)

OUTPUT_TABLES: tuple[str, ...] = (
    "instance_level_summary.csv",
    "pairwise_quality_summary.csv",
    "cdf_h_vs_cdf_l_direction.csv",
    "spf_cd_vs_spf.csv",
    "priority_sensitive_subset.csv",
    "interaction_summary.csv",
    "agent_count_summary.csv",
    "conflict_graph_diagnostic.csv",
    "makespan_summary.csv",
    "runtime_summary.csv",
    "mapf7_differing_instances.csv",
    "cbs_quality_reference.csv",
    "analysis_summary.md",
    "thesis_tables.md",
)


@dataclass(frozen=True, slots=True)
class ConflictAwareAnalysisConfig:
    conflict_aware_results_csv: Path
    fixed_results_csv: Path
    random_results_csv: Path
    spf_lpf_results_csv: Path
    manifest_path: Path
    output_dir: Path
    expected_instance_count: int = 27
    expected_random_k: int = RANDOM_K_DEFAULT


@dataclass
class ValidationReport:
    passed: bool
    messages: list[str] = field(default_factory=list)

    def add(self, message: str, *, passed: bool) -> None:
        self.messages.append(message)
        if not passed:
            self.passed = False


@dataclass(frozen=True, slots=True)
class ConflictAwareAnalysisResult:
    validation: ValidationReport
    output_dir: Path
    tables_written: tuple[str, ...]
    plots_written: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConflictAwareRunRecord:
    instance_id: str
    agent_count: int
    interaction_level: str
    strategy: str
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


def _parse_int_tuple(value: str | None) -> tuple[int, ...] | None:
    if value is None or value == "":
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("expected JSON list for integer tuple field")
    return tuple(int(item) for item in parsed)


def load_conflict_aware_results(path: Path) -> tuple[ConflictAwareRunRecord, ...]:
    rows = _read_csv_rows(path)
    records: list[ConflictAwareRunRecord] = []
    for row in rows:
        records.append(
            ConflictAwareRunRecord(
                instance_id=row["instance_id"],
                agent_count=int(row["agent_count"]),
                interaction_level=row["interaction_level"].lower(),
                strategy=row["strategy"].lower(),
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
                pp_time_ms=_parse_float(row.get("pp_time_ms")),
                total_time_ms=float(row["total_time_ms"]),
                soc=_parse_int(row.get("soc")),
                makespan=_parse_int(row.get("makespan")),
                conflict_count=_parse_int(row.get("conflict_count")),
            )
        )
    return tuple(records)


def _is_valid_permutation(order: tuple[int, ...], agent_count: int) -> bool:
    return sorted(order) == list(range(agent_count))


def _degree_summary(degrees: tuple[int, ...] | None) -> dict[str, Any]:
    if not degrees:
        return {
            "max_degree": "",
            "mean_degree": "",
            "degree_zero_count": "",
            "distinct_degree_count": "",
            "degree_stddev": "",
        }
    distinct = len(set(degrees))
    return {
        "max_degree": max(degrees),
        "mean_degree": statistics.mean(degrees),
        "degree_zero_count": sum(1 for value in degrees if value == 0),
        "distinct_degree_count": distinct,
        "degree_stddev": statistics.pstdev(degrees) if len(degrees) > 1 else 0.0,
    }


def validate_datasets(
    *,
    manifest: MAPFBenchmarkManifest,
    conflict_records: Sequence[ConflictAwareRunRecord],
    fixed_records: Sequence[FixedRunRecord],
    random_records: Sequence[RandomRunRecord],
    spf_lpf_records: Sequence[StrategyRunRecord],
    config: ConflictAwareAnalysisConfig,
) -> ValidationReport:
    report = ValidationReport(passed=True)
    manifest_ids = {instance.instance_id for instance in manifest.instances}
    expected_total = config.expected_instance_count * len(MAPF7_STRATEGIES)

    report.add(
        f"MAPF-7 records: expected {expected_total}, found {len(conflict_records)}",
        passed=len(conflict_records) == expected_total,
    )

    conflict_keys = {
        (record.instance_id, record.strategy) for record in conflict_records
    }
    report.add(
        "MAPF-7 has no duplicate (instance_id, strategy)",
        passed=len(conflict_keys) == len(conflict_records),
    )

    conflict_ids = {record.instance_id for record in conflict_records}
    report.add(
        "MAPF-7 instance IDs match manifest",
        passed=conflict_ids == manifest_ids,
    )

    for strategy in MAPF7_STRATEGIES:
        count = sum(1 for record in conflict_records if record.strategy == strategy)
        report.add(
            f"MAPF-7 strategy {strategy}: expected {config.expected_instance_count}, found {count}",
            passed=count == config.expected_instance_count,
        )

    success_count = sum(1 for record in conflict_records if record.success)
    report.add(
        f"MAPF-7 successful runs: expected {expected_total}, found {success_count}",
        passed=success_count == expected_total,
    )

    ordering_failures = sum(
        1
        for record in conflict_records
        if record.termination_reason == "ordering_failure"
    )
    report.add(
        "MAPF-7 ordering failures: expected 0",
        passed=ordering_failures == 0,
    )

    for record in conflict_records:
        if not record.success:
            report.add(f"Unexpected failed MAPF-7 run: {record.instance_id}/{record.strategy}", passed=False)
            continue
        if record.conflict_count != 0:
            report.add(
                f"Successful MAPF-7 run has non-zero conflict_count: {record.instance_id}/{record.strategy}",
                passed=False,
            )
        if record.pp_time_ms is None:
            report.add(
                f"Successful MAPF-7 run missing pp_time_ms: {record.instance_id}/{record.strategy}",
                passed=False,
            )
            continue
        expected_total_ms = record.ordering_time_ms + record.pp_time_ms
        timing_ok = abs(record.total_time_ms - expected_total_ms) <= TIME_TOLERANCE_MS
        report.add(
            f"Timing identity {record.instance_id}/{record.strategy}: "
            f"total={record.total_time_ms:.3f}, ordering+pp={expected_total_ms:.3f}",
            passed=timing_ok,
        )
        if record.original_index_order is None:
            report.add(
                f"Missing original_index_order: {record.instance_id}/{record.strategy}",
                passed=False,
            )
            continue
        perm_ok = _is_valid_permutation(
            record.original_index_order,
            record.agent_count,
        )
        report.add(
            f"original_index_order permutation {record.instance_id}/{record.strategy}",
            passed=perm_ok,
        )
        if record.agent_degrees is None or len(record.agent_degrees) != record.agent_count:
            report.add(
                f"agent_degrees length mismatch {record.instance_id}/{record.strategy}",
                passed=False,
            )

    fixed_pp = [record for record in fixed_records if record.algorithm == STRATEGY_FIXED]
    report.add(
        f"Fixed-Priority PP records: expected {config.expected_instance_count}, found {len(fixed_pp)}",
        passed=len(fixed_pp) == config.expected_instance_count,
    )
    fixed_ids = {record.instance_id for record in fixed_pp}

    random_ids = {record.instance_id for record in random_records}
    spf_lpf_ids = {record.instance_id for record in spf_lpf_records}

    report.add(
        "Cross-dataset instance IDs match manifest",
        passed=fixed_ids == random_ids == spf_lpf_ids == conflict_ids == manifest_ids,
    )

    return report


def _conflict_by_instance(
    records: Sequence[ConflictAwareRunRecord],
    strategy: str,
) -> dict[str, ConflictAwareRunRecord]:
    return {
        record.instance_id: record
        for record in records
        if record.strategy == strategy
    }


def build_instance_level_summary(
    *,
    manifest: MAPFBenchmarkManifest,
    fixed_records: Sequence[FixedRunRecord],
    random_aggregates: dict[str, RandomInstanceAggregate],
    spf_lpf_records: Sequence[StrategyRunRecord],
    conflict_records: Sequence[ConflictAwareRunRecord],
) -> list[dict[str, Any]]:
    fixed_lookup = _fixed_by_instance(fixed_records)
    cbs_lookup = _cbs_by_instance(fixed_records)
    spf_lookup = {
        record.instance_id: record
        for record in spf_lpf_records
        if record.strategy == STRATEGY_SPF
    }
    lpf_lookup = {
        record.instance_id: record
        for record in spf_lpf_records
        if record.strategy == STRATEGY_LPF
    }
    cdf_h_lookup = _conflict_by_instance(conflict_records, STRATEGY_CDF_H)
    cdf_l_lookup = _conflict_by_instance(conflict_records, STRATEGY_CDF_L)
    spf_cd_lookup = _conflict_by_instance(conflict_records, STRATEGY_SPF_CD)

    rows: list[dict[str, Any]] = []
    for instance in sorted(
        manifest.instances,
        key=lambda item: _instance_sort_key(
            item.instance_id,
            item.agent_count,
            item.interaction_level.value,
        ),
    ):
        instance_id = instance.instance_id
        fixed = fixed_lookup[instance_id]
        random = random_aggregates[instance_id]
        spf = spf_lookup[instance_id]
        lpf = lpf_lookup[instance_id]
        cdf_h = cdf_h_lookup[instance_id]
        cdf_l = cdf_l_lookup[instance_id]
        spf_cd = spf_cd_lookup[instance_id]
        cbs = cbs_lookup.get(instance_id)
        degree_stats = _degree_summary(cdf_h.agent_degrees)

        rows.append(
            {
                "instance_id": instance_id,
                "agent_count": instance.agent_count,
                "interaction_level": instance.interaction_level.value,
                "fixed_soc": fixed.soc,
                "random_soc_median": random.soc_median,
                "random_soc_min": random.soc_min,
                "random_soc_max": random.soc_max,
                "random_sampled_best_of_k_soc": random.sampled_best_of_k_soc,
                "random_sampled_best_of_k_note": "sampled best-of-K diagnostic only",
                "spf_soc": spf.soc,
                "lpf_soc": lpf.soc,
                "cdf_h_soc": cdf_h.soc,
                "cdf_l_soc": cdf_l.soc,
                "spf_cd_soc": spf_cd.soc,
                "cbs_soc": cbs.soc if cbs is not None else "",
                "spf_makespan": spf.makespan,
                "lpf_makespan": lpf.makespan,
                "cdf_h_makespan": cdf_h.makespan,
                "cdf_l_makespan": cdf_l.makespan,
                "spf_cd_makespan": spf_cd.makespan,
                "fixed_makespan": fixed.makespan,
                "observed_priority_sensitive": random.observed_priority_sensitive,
                "random_soc_range": random.soc_range,
                "random_unique_soc_count": random.unique_soc_count,
                "random_makespan_range": random.makespan_range,
                "independent_conflict_count": cdf_h.independent_conflict_count,
                "independent_conflict_pair_count": cdf_h.independent_conflict_pair_count,
                **degree_stats,
                "cdf_h_agent_order": json.dumps(list(cdf_h.agent_order or ())),
                "cdf_h_original_index_order": json.dumps(
                    list(cdf_h.original_index_order or ())
                ),
                "spf_agent_order": json.dumps(list(spf.agent_order or ())),
                "spf_cd_original_index_order": json.dumps(
                    list(spf_cd.original_index_order or ())
                ),
            }
        )
    return rows


def _paired_soc_summary(
    *,
    comparison: str,
    left_label: str,
    right_label: str,
    paired_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    soc_diffs = [row["soc_diff"] for row in paired_rows]
    directional = count_directional_soc_better(soc_diffs)
    soc_stats = _summary_stats(soc_diffs)
    return {
        "comparison": comparison,
        "left_strategy": left_label,
        "right_strategy": right_label,
        "common_success_count": len(paired_rows),
        "mean_soc_diff": soc_stats["mean"],
        "median_soc_diff": soc_stats["median"],
        "min_soc_diff": soc_stats["min"],
        "max_soc_diff": soc_stats["max"],
        "left_better_count": directional.left_better,
        "equal_count": directional.equal,
        "right_better_count": directional.right_better,
        "soc_diff_definition": "left_soc - right_soc (negative => left better)",
    }


def _build_pairwise_rows(
    instance_rows: Sequence[dict[str, Any]],
    *,
    comparison: str,
    left_key: str,
    right_key: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in instance_rows:
        left_soc = _row_int(row[left_key])
        right_soc = _row_int(row[right_key])
        if left_soc is None or right_soc is None:
            continue
        rows.append(
            {
                "comparison": comparison,
                "instance_id": row["instance_id"],
                "agent_count": row["agent_count"],
                "interaction_level": row["interaction_level"],
                "left_soc": left_soc,
                "right_soc": right_soc,
                "soc_diff": left_soc - right_soc,
            }
        )
    return rows


def build_pairwise_quality_summary(
    instance_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    specs = (
        ("cdf_h_vs_fixed", "cdf_h_soc", "fixed_soc", "CDF-H", "Fixed"),
        ("cdf_h_vs_spf", "cdf_h_soc", "spf_soc", "CDF-H", "SPF"),
        ("cdf_h_vs_lpf", "cdf_h_soc", "lpf_soc", "CDF-H", "LPF"),
        ("cdf_h_vs_cdf_l", "cdf_h_soc", "cdf_l_soc", "CDF-H", "CDF-L"),
        ("cdf_h_vs_spf_cd", "cdf_h_soc", "spf_cd_soc", "CDF-H", "SPF+CD"),
        ("cdf_l_vs_spf", "cdf_l_soc", "spf_soc", "CDF-L", "SPF"),
        ("spf_cd_vs_spf", "spf_cd_soc", "spf_soc", "SPF+CD", "SPF"),
        ("cdf_l_vs_spf_cd", "cdf_l_soc", "spf_cd_soc", "CDF-L", "SPF+CD"),
    )
    summaries: list[dict[str, Any]] = []
    for comparison, left_key, right_key, left_label, right_label in specs:
        paired = _build_pairwise_rows(
            instance_rows,
            comparison=comparison,
            left_key=left_key,
            right_key=right_key,
        )
        summaries.append(
            _paired_soc_summary(
                comparison=comparison,
                left_label=left_label,
                right_label=right_label,
                paired_rows=paired,
            )
        )
    return summaries


def build_cdf_h_vs_cdf_l_direction(
    instance_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in instance_rows:
        cdf_h_soc = _row_int(row["cdf_h_soc"])
        cdf_l_soc = _row_int(row["cdf_l_soc"])
        if cdf_h_soc is None or cdf_l_soc is None:
            continue
        diff = cdf_h_soc - cdf_l_soc
        if diff < 0:
            direction = "cdf_h_better"
        elif diff > 0:
            direction = "cdf_l_better"
        else:
            direction = "equal"
        rows.append(
            {
                "instance_id": row["instance_id"],
                "agent_count": row["agent_count"],
                "interaction_level": row["interaction_level"],
                "cdf_h_soc": cdf_h_soc,
                "cdf_l_soc": cdf_l_soc,
                "soc_diff_cdf_h_minus_cdf_l": diff,
                "direction": direction,
                "max_degree": row.get("max_degree"),
                "independent_conflict_pair_count": row.get("independent_conflict_pair_count"),
            }
        )
    return rows


def build_spf_cd_vs_spf(
    instance_rows: Sequence[dict[str, Any]],
    *,
    spf_lpf_records: Sequence[StrategyRunRecord],
    conflict_records: Sequence[ConflictAwareRunRecord],
) -> list[dict[str, Any]]:
    spf_lookup = {
        record.instance_id: record
        for record in spf_lpf_records
        if record.strategy == STRATEGY_SPF
    }
    spf_cd_lookup = _conflict_by_instance(conflict_records, STRATEGY_SPF_CD)

    rows: list[dict[str, Any]] = []
    for row in instance_rows:
        instance_id = row["instance_id"]
        spf = spf_lookup[instance_id]
        spf_cd = spf_cd_lookup[instance_id]

        spf_order = spf.agent_order
        spf_cd_index_order = spf_cd.original_index_order
        same_order = (
            spf_order == spf_cd_index_order
            if spf_order is not None and spf_cd_index_order is not None
            else ""
        )
        order_note = (
            "On this benchmark, agent_id equals original scenario index; "
            "SPF agent_order is compared to SPF+CD original_index_order."
        )

        tie_break_active = ""
        if same_order is False and spf_order is not None and spf_cd_index_order is not None:
            tie_break_active = (
                "Orders differ; SPF+CD applies conflict-degree tie-break after "
                "independent cost (likely equal-cost agents with differing degree)."
            )

        spf_soc = spf.soc
        spf_cd_soc = spf_cd.soc
        rows.append(
            {
                "instance_id": instance_id,
                "agent_count": row["agent_count"],
                "interaction_level": row["interaction_level"],
                "spf_soc": spf_soc,
                "spf_cd_soc": spf_cd_soc,
                "soc_equal": spf_soc == spf_cd_soc,
                "soc_diff_spf_cd_minus_spf": (
                    (spf_cd_soc - spf_soc)
                    if spf_soc is not None and spf_cd_soc is not None
                    else ""
                ),
                "spf_makespan": spf.makespan,
                "spf_cd_makespan": spf_cd.makespan,
                "makespan_equal": spf.makespan == spf_cd.makespan,
                "spf_total_time_ms": spf.total_time_ms,
                "spf_cd_total_time_ms": spf_cd.total_time_ms,
                "spf_agent_order": json.dumps(list(spf_order or ())),
                "spf_cd_original_index_order": json.dumps(list(spf_cd_index_order or ())),
                "same_priority_order": same_order,
                "order_comparison_note": order_note,
                "tie_break_explanation": tie_break_active,
                "max_degree": row.get("max_degree"),
                "distinct_degree_count": row.get("distinct_degree_count"),
            }
        )
    return rows


def build_priority_sensitive_subset(
    instance_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in instance_rows:
        if not _row_bool(row.get("observed_priority_sensitive")):
            continue
        spf_soc = _row_int(row["spf_soc"])
        rows.append(
            {
                "instance_id": row["instance_id"],
                "agent_count": row["agent_count"],
                "interaction_level": row["interaction_level"],
                "fixed_soc": row["fixed_soc"],
                "random_soc_median": row["random_soc_median"],
                "random_sampled_best_of_k_soc": row["random_sampled_best_of_k_soc"],
                "spf_soc": spf_soc,
                "lpf_soc": row["lpf_soc"],
                "cdf_h_soc": row["cdf_h_soc"],
                "cdf_l_soc": row["cdf_l_soc"],
                "spf_cd_soc": row["spf_cd_soc"],
                "cbs_soc": row.get("cbs_soc", ""),
                "cdf_h_minus_spf": (
                    _row_int(row["cdf_h_soc"]) - spf_soc
                    if spf_soc is not None and _row_int(row["cdf_h_soc"]) is not None
                    else ""
                ),
                "cdf_l_minus_spf": (
                    _row_int(row["cdf_l_soc"]) - spf_soc
                    if spf_soc is not None and _row_int(row["cdf_l_soc"]) is not None
                    else ""
                ),
                "spf_cd_minus_spf": (
                    _row_int(row["spf_cd_soc"]) - spf_soc
                    if spf_soc is not None and _row_int(row["spf_cd_soc"]) is not None
                    else ""
                ),
                "mapf7_strategies_differ_in_soc": mapf7_strategy_socs_differ(row),
                "subset_source": "MAPF-6 Random K=10 observed SoC sensitivity (frozen)",
            }
        )
    for entry in rows:
        cdf_l_gap = entry.get("cdf_l_minus_spf")
        if isinstance(cdf_l_gap, int):
            entry["cdf_l_vs_spf_interpretation"] = cdf_l_vs_spf_interpretation(cdf_l_gap)
        else:
            entry["cdf_l_vs_spf_interpretation"] = ""
    rows.sort(
        key=lambda item: (
            _row_int(item.get("random_soc_median")) or 0,
            str(item["instance_id"]),
        )
    )
    return rows


def _strategy_vs_spf_summary(
    instance_rows: Sequence[dict[str, Any]],
    *,
    strategy_key: str,
    strategy_label: str,
    filter_rows: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    source = filter_rows if filter_rows is not None else instance_rows
    diffs: list[int] = []
    for row in source:
        left = _row_int(row[strategy_key])
        spf = _row_int(row["spf_soc"])
        if left is None or spf is None:
            continue
        diffs.append(left - spf)
    directional = count_directional_soc_better(diffs)
    stats = _summary_stats(diffs)
    return {
        "strategy": strategy_label,
        "instance_count": len(source),
        "paired_count": len(diffs),
        "mean_soc_diff_vs_spf": stats["mean"],
        "median_soc_diff_vs_spf": stats["median"],
        "left_better_count": directional.left_better,
        "equal_count": directional.equal,
        "right_better_count": directional.right_better,
    }


def build_interaction_summary(
    instance_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level in INTERACTION_ORDER:
        subset = [row for row in instance_rows if row["interaction_level"] == level]
        if not subset:
            continue
        for strategy_key, label in (
            ("cdf_h_soc", "CDF-H"),
            ("cdf_l_soc", "CDF-L"),
            ("spf_cd_soc", "SPF+CD"),
            ("spf_soc", "SPF"),
        ):
            socs = [_row_int(row[strategy_key]) for row in subset]
            socs = [value for value in socs if value is not None]
            vs_spf = _strategy_vs_spf_summary(
                instance_rows,
                strategy_key=strategy_key,
                strategy_label=label,
                filter_rows=subset,
            )
            rows.append(
                {
                    "interaction_level": level,
                    "strategy": label,
                    "instance_count": len(subset),
                    "mean_soc": statistics.mean(socs) if socs else "",
                    "median_soc": statistics.median(socs) if socs else "",
                    "mean_soc_diff_vs_spf": vs_spf["mean_soc_diff_vs_spf"],
                    "median_soc_diff_vs_spf": vs_spf["median_soc_diff_vs_spf"],
                    "better_vs_spf": vs_spf["left_better_count"],
                    "equal_vs_spf": vs_spf["equal_count"],
                    "worse_vs_spf": vs_spf["right_better_count"],
                }
            )
        cdf_diff_instances = sum(
            1
            for row in subset
            if _row_int(row["cdf_h_soc"]) != _row_int(row["cdf_l_soc"])
        )
        mapf7_diff_instances = sum(
            1
            for row in subset
            if len(
                {
                    _row_int(row["cdf_h_soc"]),
                    _row_int(row["cdf_l_soc"]),
                    _row_int(row["spf_cd_soc"]),
                }
                - {None}
            )
            > 1
        )
        rows.append(
            {
                "interaction_level": level,
                "strategy": "_stratum_meta",
                "instance_count": len(subset),
                "cdf_h_vs_cdf_l_differing_instances": cdf_diff_instances,
                "mapf7_any_strategy_differing_instances": mapf7_diff_instances,
            }
        )
    return rows


def build_agent_count_summary(
    instance_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for agent_count in AGENT_COUNT_ORDER:
        subset = [row for row in instance_rows if row["agent_count"] == agent_count]
        if not subset:
            continue
        for strategy_key, label in (
            ("cdf_h_soc", "CDF-H"),
            ("cdf_l_soc", "CDF-L"),
            ("spf_cd_soc", "SPF+CD"),
        ):
            summary = _strategy_vs_spf_summary(
                instance_rows,
                strategy_key=strategy_key,
                strategy_label=label,
                filter_rows=subset,
            )
            rows.append(
                {
                    "agent_count": agent_count,
                    "strategy": label,
                    "instance_count": len(subset),
                    "mean_soc_diff_vs_spf": summary["mean_soc_diff_vs_spf"],
                    "median_soc_diff_vs_spf": summary["median_soc_diff_vs_spf"],
                    "better_vs_spf": summary["left_better_count"],
                    "equal_vs_spf": summary["equal_count"],
                    "worse_vs_spf": summary["right_better_count"],
                }
            )
    return rows


def build_conflict_graph_diagnostic(
    instance_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in instance_rows:
        mapf7_socs = {
            _row_int(row["cdf_h_soc"]),
            _row_int(row["cdf_l_soc"]),
            _row_int(row["spf_cd_soc"]),
        }
        mapf7_socs.discard(None)
        mapf7_range = max(mapf7_socs) - min(mapf7_socs) if mapf7_socs else 0
        rows.append(
            {
                "instance_id": row["instance_id"],
                "agent_count": row["agent_count"],
                "interaction_level": row["interaction_level"],
                "independent_conflict_count": row.get("independent_conflict_count"),
                "independent_conflict_pair_count": row.get("independent_conflict_pair_count"),
                "max_degree": row.get("max_degree"),
                "mean_degree": row.get("mean_degree"),
                "degree_zero_count": row.get("degree_zero_count"),
                "distinct_degree_count": row.get("distinct_degree_count"),
                "degree_stddev": row.get("degree_stddev"),
                "mapf7_soc_range": mapf7_range,
                "mapf7_soc_all_equal": len(mapf7_socs) <= 1,
            }
        )
    return rows


def build_makespan_summary(
    instance_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comparisons = (
        ("cdf_h_vs_spf", "cdf_h_makespan", "spf_makespan"),
        ("cdf_l_vs_spf", "cdf_l_makespan", "spf_makespan"),
        ("spf_cd_vs_spf", "spf_cd_makespan", "spf_makespan"),
        ("cdf_h_vs_cdf_l", "cdf_h_makespan", "cdf_l_makespan"),
    )
    for comparison, left_key, right_key in comparisons:
        diffs: list[int] = []
        differing: list[dict[str, Any]] = []
        for row in instance_rows:
            left = _row_int(row[left_key])
            right = _row_int(row[right_key])
            if left is None or right is None:
                continue
            diff = left - right
            diffs.append(diff)
            if diff != 0:
                differing.append(
                    {
                        "comparison": comparison,
                        "instance_id": row["instance_id"],
                        "left_makespan": left,
                        "right_makespan": right,
                        "makespan_diff": diff,
                    }
                )
        directional = count_directional_soc_better(diffs)
        stats = _summary_stats(diffs)
        rows.append(
            {
                "comparison": comparison,
                "common_count": len(diffs),
                "mean_makespan_diff": stats["mean"],
                "median_makespan_diff": stats["median"],
                "left_better_count": directional.left_better,
                "equal_count": directional.equal,
                "right_better_count": directional.right_better,
                "differing_instance_count": len(differing),
            }
        )
    return rows


def build_runtime_summary(
    instance_rows: Sequence[dict[str, Any]],
    *,
    spf_lpf_records: Sequence[StrategyRunRecord],
    conflict_records: Sequence[ConflictAwareRunRecord],
) -> list[dict[str, Any]]:
    spf_lookup = {
        record.instance_id: record
        for record in spf_lpf_records
        if record.strategy == STRATEGY_SPF
    }
    cdf_h_lookup = _conflict_by_instance(conflict_records, STRATEGY_CDF_H)
    cdf_l_lookup = _conflict_by_instance(conflict_records, STRATEGY_CDF_L)
    spf_cd_lookup = _conflict_by_instance(conflict_records, STRATEGY_SPF_CD)

    per_instance: list[dict[str, Any]] = []
    for row in instance_rows:
        instance_id = row["instance_id"]
        spf = spf_lookup[instance_id]
        cdf_h = cdf_h_lookup[instance_id]
        cdf_l = cdf_l_lookup[instance_id]
        spf_cd = spf_cd_lookup[instance_id]
        per_instance.append(
            {
                "instance_id": instance_id,
                "agent_count": row["agent_count"],
                "interaction_level": row["interaction_level"],
                "spf_ordering_time_ms": spf.ordering_time_ms,
                "spf_pp_time_ms": spf.pp_time_ms,
                "spf_total_time_ms": spf.total_time_ms,
                "cdf_h_ordering_time_ms": cdf_h.ordering_time_ms,
                "cdf_h_pp_time_ms": cdf_h.pp_time_ms,
                "cdf_h_total_time_ms": cdf_h.total_time_ms,
                "cdf_l_ordering_time_ms": cdf_l.ordering_time_ms,
                "cdf_l_pp_time_ms": cdf_l.pp_time_ms,
                "cdf_l_total_time_ms": cdf_l.total_time_ms,
                "spf_cd_ordering_time_ms": spf_cd.ordering_time_ms,
                "spf_cd_pp_time_ms": spf_cd.pp_time_ms,
                "spf_cd_total_time_ms": spf_cd.total_time_ms,
                "cdf_h_ordering_minus_spf_ordering_ms": (
                    cdf_h.ordering_time_ms - spf.ordering_time_ms
                ),
                "spf_cd_ordering_minus_spf_ordering_ms": (
                    spf_cd.ordering_time_ms - spf.ordering_time_ms
                ),
            }
        )

    def _mean(values: Sequence[float]) -> float:
        return statistics.mean(values)

    def _median(values: Sequence[float]) -> float:
        return statistics.median(values)

    aggregate = {
        "metric_group": "runtime_aggregate",
        "mean_spf_ordering_ms": _mean([row["spf_ordering_time_ms"] for row in per_instance]),
        "median_spf_ordering_ms": _median([row["spf_ordering_time_ms"] for row in per_instance]),
        "mean_cdf_h_ordering_ms": _mean([row["cdf_h_ordering_time_ms"] for row in per_instance]),
        "median_cdf_h_ordering_ms": _median([row["cdf_h_ordering_time_ms"] for row in per_instance]),
        "mean_cdf_l_ordering_ms": _mean([row["cdf_l_ordering_time_ms"] for row in per_instance]),
        "median_cdf_l_ordering_ms": _median([row["cdf_l_ordering_time_ms"] for row in per_instance]),
        "mean_spf_cd_ordering_ms": _mean([row["spf_cd_ordering_time_ms"] for row in per_instance]),
        "median_spf_cd_ordering_ms": _median([row["spf_cd_ordering_time_ms"] for row in per_instance]),
        "mean_spf_pp_ms": _mean([float(row["spf_pp_time_ms"]) for row in per_instance]),
        "mean_cdf_h_pp_ms": _mean([float(row["cdf_h_pp_time_ms"]) for row in per_instance]),
        "mean_cdf_l_pp_ms": _mean([float(row["cdf_l_pp_time_ms"]) for row in per_instance]),
        "mean_spf_cd_pp_ms": _mean([float(row["spf_cd_pp_time_ms"]) for row in per_instance]),
        "mean_spf_total_ms": _mean([row["spf_total_time_ms"] for row in per_instance]),
        "mean_cdf_h_total_ms": _mean([row["cdf_h_total_time_ms"] for row in per_instance]),
        "mean_cdf_l_total_ms": _mean([row["cdf_l_total_time_ms"] for row in per_instance]),
        "mean_spf_cd_total_ms": _mean([row["spf_cd_total_time_ms"] for row in per_instance]),
        "mean_spf_ordering_s": _ms_to_seconds(_mean([row["spf_ordering_time_ms"] for row in per_instance])),
        "mean_cdf_h_ordering_s": _ms_to_seconds(_mean([row["cdf_h_ordering_time_ms"] for row in per_instance])),
        "mean_cdf_l_ordering_s": _ms_to_seconds(_mean([row["cdf_l_ordering_time_ms"] for row in per_instance])),
        "mean_spf_cd_ordering_s": _ms_to_seconds(_mean([row["spf_cd_ordering_time_ms"] for row in per_instance])),
        "mean_spf_pp_s": _ms_to_seconds(_mean([float(row["spf_pp_time_ms"]) for row in per_instance])),
        "mean_cdf_h_pp_s": _ms_to_seconds(_mean([float(row["cdf_h_pp_time_ms"]) for row in per_instance])),
        "mean_cdf_l_pp_s": _ms_to_seconds(_mean([float(row["cdf_l_pp_time_ms"]) for row in per_instance])),
        "mean_spf_cd_pp_s": _ms_to_seconds(_mean([float(row["spf_cd_pp_time_ms"]) for row in per_instance])),
        "mean_ordering_delta_cdf_h_minus_spf_ms": _mean(
            [row["cdf_h_ordering_minus_spf_ordering_ms"] for row in per_instance]
        ),
        "mean_ordering_delta_spf_cd_minus_spf_ms": _mean(
            [row["spf_cd_ordering_minus_spf_ordering_ms"] for row in per_instance]
        ),
        "note": (
            "Ordering phases include independent ST-A* + conflict detection/graph; "
            "isolated graph-construction timing was not recorded separately. "
            "SPF from MAPF-6, MAPF-7 from separate benchmark run."
        ),
    }
    return per_instance + [aggregate]


def build_mapf7_differing_instances(
    instance_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in instance_rows:
        cdf_h = _row_int(row["cdf_h_soc"])
        cdf_l = _row_int(row["cdf_l_soc"])
        spf_cd = _row_int(row["spf_cd_soc"])
        if cdf_h is None or cdf_l is None or spf_cd is None:
            continue
        if cdf_h == cdf_l == spf_cd:
            continue
        rows.append(
            {
                "instance_id": row["instance_id"],
                "agent_count": row["agent_count"],
                "interaction_level": row["interaction_level"],
                "cdf_h_soc": cdf_h,
                "cdf_l_soc": cdf_l,
                "spf_cd_soc": spf_cd,
                "spf_soc": row["spf_soc"],
                "mapf7_soc_range": max(cdf_h, cdf_l, spf_cd) - min(cdf_h, cdf_l, spf_cd),
                "max_degree": row.get("max_degree"),
                "independent_conflict_pair_count": row.get("independent_conflict_pair_count"),
            }
        )
    return rows


def build_cbs_quality_reference(
    *,
    instance_rows: Sequence[dict[str, Any]],
    fixed_records: Sequence[FixedRunRecord],
) -> list[dict[str, Any]]:
    cbs_lookup = _cbs_by_instance(fixed_records)
    rows: list[dict[str, Any]] = []
    for row in instance_rows:
        cbs = cbs_lookup.get(row["instance_id"])
        if cbs is None or cbs.soc is None:
            continue
        rows.append(
            {
                "instance_id": row["instance_id"],
                "agent_count": row["agent_count"],
                "interaction_level": row["interaction_level"],
                "cbs_soc": cbs.soc,
                "fixed_minus_cbs": _row_int(row["fixed_soc"]) - cbs.soc if row.get("fixed_soc") is not None else "",
                "spf_minus_cbs": _row_int(row["spf_soc"]) - cbs.soc,
                "cdf_h_minus_cbs": _row_int(row["cdf_h_soc"]) - cbs.soc,
                "cdf_l_minus_cbs": _row_int(row["cdf_l_soc"]) - cbs.soc,
                "spf_cd_minus_cbs": _row_int(row["spf_cd_soc"]) - cbs.soc,
                "random_median_minus_cbs": (
                    _row_int(row["random_soc_median"]) - cbs.soc
                    if row.get("random_soc_median") is not None
                    else ""
                ),
                "random_sampled_best_of_k_minus_cbs": (
                    _row_int(row["random_sampled_best_of_k_soc"]) - cbs.soc
                    if row.get("random_sampled_best_of_k_soc") is not None
                    else ""
                ),
                "random_sampled_best_of_k_note": "diagnostic only",
                "reference_note": "Basic CBS quality reference; common-success only",
            }
        )
    return rows


def build_cbs_aggregate_summary(cbs_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = (
        ("Fixed", "fixed_minus_cbs", False),
        ("SPF", "spf_minus_cbs", False),
        ("CDF-H", "cdf_h_minus_cbs", False),
        ("CDF-L", "cdf_l_minus_cbs", False),
        ("SPF+CD", "spf_cd_minus_cbs", False),
        ("Random K=10 median", "random_median_minus_cbs", False),
        ("Random sampled best-of-K (diagnostic)", "random_sampled_best_of_k_minus_cbs", True),
    )
    summaries: list[dict[str, Any]] = []
    for label, key, diagnostic in specs:
        gaps = [float(row[key]) for row in cbs_rows if row.get(key) not in ("", None)]
        directional = _directional_cbs_gaps(gaps)
        summaries.append(
            {
                "strategy": label,
                "common_success_count": len(cbs_rows),
                "mean_soc_gap": statistics.mean(gaps) if gaps else "",
                "median_soc_gap": statistics.median(gaps) if gaps else "",
                "strategy_better_count": directional["strategy_better_count"],
                "equal_count": directional["equal_count"],
                "basic_cbs_better_count": directional["basic_cbs_better_count"],
                "diagnostic_only": diagnostic,
            }
        )
    return summaries


def _lookup_pairwise(
    paired_summary: Sequence[dict[str, Any]],
    comparison: str,
) -> dict[str, Any]:
    for row in paired_summary:
        if row["comparison"] == comparison:
            return row
    return {}


def mapf7_strategy_socs_differ(row: dict[str, Any]) -> bool:
    socs = {
        _row_int(row.get("cdf_h_soc")),
        _row_int(row.get("cdf_l_soc")),
        _row_int(row.get("spf_cd_soc")),
    }
    socs.discard(None)
    return len(socs) > 1


def count_mapf7_differing_in_priority_sensitive(
    priority_sensitive_rows: Sequence[dict[str, Any]],
) -> int:
    return sum(
        1 for row in priority_sensitive_rows if mapf7_strategy_socs_differ(row)
    )


def cdf_l_vs_spf_interpretation(cdf_l_minus_spf: int) -> str:
    if cdf_l_minus_spf < 0:
        return "CDF-L better (lower SoC than SPF)"
    if cdf_l_minus_spf > 0:
        return "CDF-L worse (higher SoC than SPF)"
    return "equal to SPF"


def _runtime_strategy_seconds(
    runtime_aggregate: dict[str, Any],
    *,
    prefix: str,
) -> tuple[float, float, float]:
    ordering = float(runtime_aggregate[f"mean_{prefix}_ordering_s"])
    pp = float(runtime_aggregate[f"mean_{prefix}_pp_s"])
    return ordering, pp, ordering + pp


def build_runtime_summary_lines(runtime_aggregate: dict[str, Any]) -> list[str]:
    if not runtime_aggregate:
        return ["- Runtime aggregate unavailable."]

    spf_ord, spf_pp, spf_total = _runtime_strategy_seconds(runtime_aggregate, prefix="spf")
    cdf_h_ord, cdf_h_pp, cdf_h_total = _runtime_strategy_seconds(
        runtime_aggregate, prefix="cdf_h"
    )
    cdf_l_ord, cdf_l_pp, cdf_l_total = _runtime_strategy_seconds(
        runtime_aggregate, prefix="cdf_l"
    )
    spf_cd_ord, spf_cd_pp, spf_cd_total = _runtime_strategy_seconds(
        runtime_aggregate, prefix="spf_cd"
    )

    return [
        "### Runtime",
        "",
        f"- Mean SPF: ordering {_format_float(spf_ord)} s, PP {_format_float(spf_pp)} s, "
        f"total {_format_float(spf_total)} s.",
        f"- Mean CDF-H: ordering {_format_float(cdf_h_ord)} s, PP {_format_float(cdf_h_pp)} s, "
        f"total {_format_float(cdf_h_total)} s.",
        f"- Mean CDF-L: ordering {_format_float(cdf_l_ord)} s, PP {_format_float(cdf_l_pp)} s, "
        f"total {_format_float(cdf_l_total)} s.",
        f"- Mean SPF+CD: ordering {_format_float(spf_cd_ord)} s, PP {_format_float(spf_cd_pp)} s, "
        f"total {_format_float(spf_cd_total)} s.",
        "- Ordering-phase cost is very similar across SPF, CDF-H, CDF-L, and SPF+CD "
        f"(mean ordering within ~{_format_float(min(spf_ord, cdf_h_ord, cdf_l_ord, spf_cd_ord))}–"
        f"{_format_float(max(spf_ord, cdf_h_ord, cdf_l_ord, spf_cd_ord))} s).",
        "- CDF-H and CDF-L have higher mean PP runtime than SPF/SPF+CD because their generated "
        "priority orders can make downstream PP more expensive on this catalogue.",
        "- Isolated graph-construction time was not measured separately; comparisons use full "
        "ordering phases from separate MAPF-6 (SPF) and MAPF-7 benchmark runs.",
        "",
    ]


def build_analysis_summary_md(
    *,
    config: ConflictAwareAnalysisConfig,
    validation: ValidationReport,
    instance_rows: Sequence[dict[str, Any]],
    paired_summary: Sequence[dict[str, Any]],
    cdf_direction_rows: Sequence[dict[str, Any]],
    spf_cd_rows: Sequence[dict[str, Any]],
    priority_sensitive_rows: Sequence[dict[str, Any]],
    interaction_rows: Sequence[dict[str, Any]],
    conflict_diagnostic_rows: Sequence[dict[str, Any]],
    runtime_rows: Sequence[dict[str, Any]],
    cbs_aggregate: Sequence[dict[str, Any]],
    mapf7_differing_rows: Sequence[dict[str, Any]],
) -> str:
    cdf_h_vs_cdf_l = _lookup_pairwise(paired_summary, "cdf_h_vs_cdf_l")
    cdf_h_vs_spf = _lookup_pairwise(paired_summary, "cdf_h_vs_spf")
    cdf_l_vs_spf = _lookup_pairwise(paired_summary, "cdf_l_vs_spf")
    spf_cd_vs_spf = _lookup_pairwise(paired_summary, "spf_cd_vs_spf")

    cdf_l_better = cdf_h_vs_cdf_l.get("right_better_count", 0)
    cdf_h_better = cdf_h_vs_cdf_l.get("left_better_count", 0)
    cdf_equal = cdf_h_vs_cdf_l.get("equal_count", 0)

    spf_cd_soc_diff = sum(1 for row in spf_cd_rows if not row.get("soc_equal"))
    spf_cd_order_diff = sum(1 for row in spf_cd_rows if row.get("same_priority_order") is False)
    spf_cd_soc_same = sum(1 for row in spf_cd_rows if row.get("soc_equal"))

    sensitive_count = len(priority_sensitive_rows)
    mapf7_differ_in_sensitive = count_mapf7_differing_in_priority_sensitive(
        priority_sensitive_rows
    )

    equal_diag = [row for row in conflict_diagnostic_rows if row.get("mapf7_soc_all_equal")]
    differ_diag = [row for row in conflict_diagnostic_rows if not row.get("mapf7_soc_all_equal")]

    runtime_aggregate = next(
        (row for row in runtime_rows if row.get("metric_group") == "runtime_aggregate"),
        {},
    )

    sensitive_lookup = {
        str(row["instance_id"]): row for row in priority_sensitive_rows
    }
    n20_row = sensitive_lookup.get("AR0204SR_n20_high_001", {})
    n05_row = sensitive_lookup.get("AR0204SR_n05_high_002", {})

    direction_supported = cdf_h_better > cdf_l_better
    direction_text = (
        "The pre-declared high-conflict-first direction (CDF-H) shows equal or better SoC "
        "on at least as many instances as CDF-L."
        if direction_supported
        else "The pre-declared high-conflict-first direction (CDF-H) was not supported by "
        "aggregate SoC counts relative to the CDF-L ablation."
    )

    lines = [
        "# MAPF-7 Conflict-Aware Priority Analysis Summary",
        "",
        "## Dataset validation",
        "",
        f"- Validation status: {'PASSED' if validation.passed else 'FAILED'}",
        f"- MAPF-7 CSV: `{config.conflict_aware_results_csv}`",
        f"- Manifest: `{config.manifest_path}`",
        "",
        "## Methodological notes",
        "",
        "- CDF-H remains the pre-declared primary strategy; CDF-L and SPF+CD are required ablations.",
        "- Random sampled best-of-K is diagnostic only, not a fair single-run baseline.",
        "- Basic CBS is a quality reference on common-success instances, not a quality ceiling.",
        "- LOW/MEDIUM/HIGH are independent-path interaction strata, not general difficulty labels.",
        "- Because strata were defined using independent-path conflicts, concentration in HIGH is "
        "related to the same underlying interaction signal and is not independent validation.",
        "- SPF runtime comes from MAPF-6; MAPF-7 strategies from a separate benchmark run.",
        "",
        "## Confirmed facts",
        "",
        f"- All {len(instance_rows)} instances succeeded for CDF-H, CDF-L, and SPF+CD.",
        f"- MAPF-7 strategies differ in SoC on {len(mapf7_differing_rows)} of {len(instance_rows)} instances.",
        f"- CDF-H vs CDF-L: CDF-H better={cdf_h_better}, equal={cdf_equal}, CDF-L better={cdf_l_better}; "
        f"mean diff={_format_float(cdf_h_vs_cdf_l.get('mean_soc_diff'))}, "
        f"median diff={_format_float(cdf_h_vs_cdf_l.get('median_soc_diff'))}.",
        f"- {direction_text}",
        f"- CDF-H vs SPF: LEFT better={cdf_h_vs_spf.get('left_better_count', 'n/a')}, "
        f"equal={cdf_h_vs_spf.get('equal_count', 'n/a')}, RIGHT better={cdf_h_vs_spf.get('right_better_count', 'n/a')}.",
        f"- CDF-L vs SPF: LEFT better={cdf_l_vs_spf.get('left_better_count', 'n/a')}, "
        f"equal={cdf_l_vs_spf.get('equal_count', 'n/a')}, RIGHT better={cdf_l_vs_spf.get('right_better_count', 'n/a')}.",
        f"- SPF+CD vs SPF: identical SoC on {spf_cd_soc_same}/27; different SoC on {spf_cd_soc_diff}/27; "
        f"different priority order on {spf_cd_order_diff}/27.",
        f"- MAPF-6 priority-sensitive subset size remains {sensitive_count} instances (frozen from Random K=10).",
        f"- Within that subset, MAPF-7 strategies (CDF-H / CDF-L / SPF+CD) differ in SoC on "
        f"{mapf7_differ_in_sensitive} of {sensitive_count} instances.",
        "",
        *build_runtime_summary_lines(runtime_aggregate),
        "### Priority-sensitive subset examples (CDF-L vs SPF, diff = CDF-L SoC − SPF SoC)",
        "",
        *(
            [
                f"- {n05_row['instance_id']}: SPF={n05_row['spf_soc']}, CDF-L={n05_row['cdf_l_soc']}, "
                f"diff={n05_row['cdf_l_minus_spf']} → {n05_row.get('cdf_l_vs_spf_interpretation', '')}."
            ]
            if n05_row
            else []
        ),
        *(
            [
                f"- {n20_row['instance_id']}: SPF={n20_row['spf_soc']}, CDF-L={n20_row['cdf_l_soc']}, "
                f"diff={n20_row['cdf_l_minus_spf']} → {n20_row.get('cdf_l_vs_spf_interpretation', '')}."
            ]
            if n20_row
            else []
        ),
        "",
        "## Interpretation",
        "",
        "- Conflict-degree information can change PP outcomes when independent-path conflict structure "
        "is heterogeneous, but effects are sparse on this catalogue.",
        "- Direction ablation (CDF-H vs CDF-L) is the primary test of whether high-conflict-first ordering "
        "was preferable to low-conflict-first under the frozen methodology.",
        "- SPF+CD isolates whether conflict degree adds signal beyond path length when costs tie.",
        "",
        "## Limitations",
        "",
        "- Only 27 primary catalogue instances; design was motivated by observations from the same catalogue.",
        "- CDF-H was frozen before MAPF-7 evaluation; CDF-L must not be retroactively promoted to primary.",
        "- Random K=10 samples only 10 permutations per instance.",
        "- Runtime comparisons mix separate historical runs (MAPF-6 vs MAPF-7).",
        "- Isolated graph-construction timing was not recorded; ordering overhead comparisons use full ordering phases.",
        "",
        "## MAPF-7 decision support",
        "",
        "### A. Stop with CDF-H + ablations and report mixed/negative contribution",
        "",
        "**Evidence supporting:**",
        f"- MAPF-7 SoC differences occur on only {len(mapf7_differing_rows)}/27 instances.",
        f"- CDF-H does not clearly dominate SPF on aggregate counts "
        f"(equal={cdf_h_vs_spf.get('equal_count', 'n/a')}, SPF better={cdf_h_vs_spf.get('right_better_count', 'n/a')}).",
        "",
        "**Evidence against:**",
        f"- CDF-L may outperform CDF-H on some instances (CDF-L better on {cdf_l_better}).",
        f"- Effects appear on the frozen priority-sensitive subset "
        f"({mapf7_differ_in_sensitive}/{sensitive_count} with MAPF-7 SoC differences).",
        "",
        "**Methodological risk:** Accepting a null/mixed result without held-out validation.",
        "**Expected cost:** Analysis/write-up only.",
        "",
        "### B. Design one additional frozen conflict-aware variant from observed failure modes",
        "",
        "**Evidence supporting:**",
        f"- Observed SoC differences on {len(mapf7_differing_rows)} instances show the ordering signal matters on a subset.",
        f"- Within the priority-sensitive subset, MAPF-7 SoC differs on "
        f"{mapf7_differ_in_sensitive}/{sensitive_count} instances.",
        f"- Direction test: CDF-H better {cdf_h_better}, CDF-L better {cdf_l_better}.",
        f"- Graph diagnostics differ descriptively (equal-SoC n={len(equal_diag)}, differing n={len(differ_diag)}).",
        "",
        "**Evidence against:**",
        "- Additional variants risk post-hoc tuning unless pre-registered and still may not beat SPF.",
        "",
        "**Methodological risk:** Overfitting to 27 instances without held-out validation.",
        "**Expected cost:** Small implementation + another 81-run benchmark (or subset ablation).",
        "",
        "### C. Proceed to held-out validation of the current frozen family",
        "",
        "**Evidence supporting:**",
        "- Current family is fully implemented and benchmarked once on the primary catalogue.",
        "- Held-out catalogue was pre-declared in MAPF-7.1 as preferred over Random K=20.",
        "",
        "**Evidence against:**",
        "- If primary-catalogue effect is weak, held-out may confirm null result only.",
        "",
        "**Methodological risk:** Low if catalogue generation methodology is unchanged and not used for tuning.",
        "**Expected cost:** Catalogue generation + 81-run benchmark on held-out set.",
        "",
    ]
    return "\n".join(lines)


def build_thesis_tables_md(
    *,
    instance_rows: Sequence[dict[str, Any]],
    paired_summary: Sequence[dict[str, Any]],
    cdf_direction_rows: Sequence[dict[str, Any]],
    priority_sensitive_rows: Sequence[dict[str, Any]],
    interaction_rows: Sequence[dict[str, Any]],
    runtime_rows: Sequence[dict[str, Any]],
    cbs_aggregate: Sequence[dict[str, Any]],
    mapf7_differing_rows: Sequence[dict[str, Any]],
) -> str:
    runtime_aggregate = next(
        row for row in runtime_rows if row.get("metric_group") == "runtime_aggregate"
    )
    cdf_h_vs_cdf_l = _lookup_pairwise(paired_summary, "cdf_h_vs_cdf_l")
    cdf_h_vs_spf = _lookup_pairwise(paired_summary, "cdf_h_vs_spf")

    mapf7_differ_in_sensitive = count_mapf7_differing_in_priority_sensitive(
        priority_sensitive_rows
    )

    lines = [
        "# MAPF-7 Thesis Tables",
        "",
        "## Table A — MAPF-7 success",
        "",
        "| Strategy | Successful / total | Success rate |",
        "|---|---:|---:|",
        f"| CDF-H (primary) | 27 / 27 | 100.0% |",
        f"| CDF-L (ablation) | 27 / 27 | 100.0% |",
        f"| SPF+CD (ablation) | 27 / 27 | 100.0% |",
        "",
        "## Table B — Pairwise SoC comparisons (selected)",
        "",
        "| Comparison | Common success | Mean diff | Median diff | LEFT better | Equal | RIGHT better |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for comparison in (
        "cdf_h_vs_fixed",
        "cdf_h_vs_spf",
        "cdf_h_vs_cdf_l",
        "cdf_l_vs_spf",
        "spf_cd_vs_spf",
    ):
        summary = _lookup_pairwise(paired_summary, comparison)
        if not summary:
            continue
        lines.append(
            f"| {summary['left_strategy']} vs {summary['right_strategy']} | "
            f"{summary['common_success_count']} | "
            f"{_format_float(summary['mean_soc_diff'])} | "
            f"{_format_float(summary['median_soc_diff'])} | "
            f"{summary['left_better_count']} | {summary['equal_count']} | "
            f"{summary['right_better_count']} |"
        )

    lines.extend(
        [
            "",
            "*Sign convention: diff = LEFT SoC − RIGHT SoC.*",
            "",
            "## Table C — Priority-sensitive subset (MAPF-6 frozen, n="
            f"{len(priority_sensitive_rows)})",
            "",
            f"*Within this subset, MAPF-7 strategies differ in SoC on "
            f"{mapf7_differ_in_sensitive} of {len(priority_sensitive_rows)} instances "
            f"(derived from stored CDF-H / CDF-L / SPF+CD results).*",
            "",
            "| Instance | Agents | Interaction | SPF | CDF-H | CDF-L | SPF+CD | MAPF-7 differ |",
            "|---|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in priority_sensitive_rows:
        lines.append(
            f"| {row['instance_id']} | {row['agent_count']} | {row['interaction_level']} | "
            f"{row['spf_soc']} | {row['cdf_h_soc']} | {row['cdf_l_soc']} | {row['spf_cd_soc']} | "
            f"{'yes' if row.get('mapf7_strategies_differ_in_soc') else 'no'} |"
        )

    lines.extend(
        [
            "",
            "*CDF-L vs SPF sign convention: diff = CDF-L SoC − SPF SoC; negative → CDF-L better; "
            "positive → CDF-L worse.*",
            "",
            "## Table D — Interaction-stratified mean SoC diff vs SPF",
            "",
            "| Interaction | Strategy | Mean SoC diff vs SPF | Better / equal / worse vs SPF |",
            "|---|---|---:|---|",
        ]
    )
    for row in interaction_rows:
        if row.get("strategy") == "_stratum_meta":
            continue
        lines.append(
            f"| {row['interaction_level'].upper()} | {row['strategy']} | "
            f"{_format_float(row['mean_soc_diff_vs_spf'])} | "
            f"{row['better_vs_spf']} / {row['equal_vs_spf']} / {row['worse_vs_spf']} |"
        )

    lines.extend(
        [
            "",
            "## Table E — Runtime comparison (mean, seconds)",
            "",
            "| Strategy | Ordering | PP | Total | Ordering fraction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, ordering_key, pp_key, total_key in (
        ("SPF", "mean_spf_ordering_s", "mean_spf_pp_s", None),
        ("CDF-H", "mean_cdf_h_ordering_s", "mean_cdf_h_pp_s", None),
        ("CDF-L", "mean_cdf_l_ordering_s", "mean_cdf_l_pp_s", None),
        ("SPF+CD", "mean_spf_cd_ordering_s", "mean_spf_cd_pp_s", None),
    ):
        ordering = float(runtime_aggregate[ordering_key])
        pp = float(runtime_aggregate[pp_key])
        total = ordering + pp
        fraction = ordering / total if total else 0.0
        lines.append(
            f"| {label} | {_format_float(ordering)} | {_format_float(pp)} | "
            f"{_format_float(total)} | {_format_float(fraction * 100, digits=1)}% |"
        )

    lines.extend(
        [
            "",
            "## Table F — Basic CBS quality reference",
            "",
            "| Strategy | Common success | Mean gap | Median gap | Strategy better | Equal | Basic CBS better |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in cbs_aggregate:
        suffix = " *(diagnostic)*" if row.get("diagnostic_only") else ""
        lines.append(
            f"| {row['strategy']}{suffix} | {row['common_success_count']} | "
            f"{_format_float(row['mean_soc_gap'])} | {_format_float(row['median_soc_gap'])} | "
            f"{row['strategy_better_count']} | {row['equal_count']} | "
            f"{row['basic_cbs_better_count']} |"
        )

    lines.extend(
        [
            "",
            "## Table G — MAPF-7 differing instances",
            "",
            "| Instance | Agents | Interaction | CDF-H | CDF-L | SPF+CD | SPF | Range |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in mapf7_differing_rows:
        lines.append(
            f"| {row['instance_id']} | {row['agent_count']} | {row['interaction_level']} | "
            f"{row['cdf_h_soc']} | {row['cdf_l_soc']} | {row['spf_cd_soc']} | "
            f"{row['spf_soc']} | {row['mapf7_soc_range']} |"
        )

    lines.append("")
    return "\n".join(lines)


def run_conflict_aware_priority_analysis(
    config: ConflictAwareAnalysisConfig,
    *,
    plot_writer: Callable[..., tuple[str, ...]] | None = None,
) -> ConflictAwareAnalysisResult:
    manifest = load_benchmark_manifest(config.manifest_path)
    conflict_records = load_conflict_aware_results(config.conflict_aware_results_csv)
    fixed_records = load_fixed_results(config.fixed_results_csv)
    random_records = load_random_results(config.random_results_csv)
    spf_lpf_records = load_strategy_results(config.spf_lpf_results_csv)

    validation = validate_datasets(
        manifest=manifest,
        conflict_records=conflict_records,
        fixed_records=fixed_records,
        random_records=random_records,
        spf_lpf_records=spf_lpf_records,
        config=config,
    )

    if not validation.passed:
        return ConflictAwareAnalysisResult(
            validation=validation,
            output_dir=config.output_dir,
            tables_written=(),
            plots_written=(),
        )

    random_aggregates = aggregate_random_by_instance(random_records)
    instance_rows = build_instance_level_summary(
        manifest=manifest,
        fixed_records=fixed_records,
        random_aggregates=random_aggregates,
        spf_lpf_records=spf_lpf_records,
        conflict_records=conflict_records,
    )
    paired_summary = build_pairwise_quality_summary(instance_rows)
    cdf_direction_rows = build_cdf_h_vs_cdf_l_direction(instance_rows)
    spf_cd_rows = build_spf_cd_vs_spf(
        instance_rows,
        spf_lpf_records=spf_lpf_records,
        conflict_records=conflict_records,
    )
    priority_sensitive_rows = build_priority_sensitive_subset(instance_rows)
    interaction_rows = build_interaction_summary(instance_rows)
    agent_count_rows = build_agent_count_summary(instance_rows)
    conflict_diagnostic_rows = build_conflict_graph_diagnostic(instance_rows)
    makespan_rows = build_makespan_summary(instance_rows)
    runtime_rows = build_runtime_summary(
        instance_rows,
        spf_lpf_records=spf_lpf_records,
        conflict_records=conflict_records,
    )
    mapf7_differing_rows = build_mapf7_differing_instances(instance_rows)
    cbs_rows = build_cbs_quality_reference(
        instance_rows=instance_rows,
        fixed_records=fixed_records,
    )
    cbs_aggregate = build_cbs_aggregate_summary(cbs_rows)

    output_dir = config.output_dir
    tables_written: list[str] = []

    table_specs: list[tuple[str, list[dict[str, Any]]]] = [
        ("instance_level_summary.csv", instance_rows),
        ("pairwise_quality_summary.csv", paired_summary),
        ("cdf_h_vs_cdf_l_direction.csv", cdf_direction_rows),
        ("spf_cd_vs_spf.csv", spf_cd_rows),
        ("priority_sensitive_subset.csv", priority_sensitive_rows),
        ("interaction_summary.csv", interaction_rows),
        ("agent_count_summary.csv", agent_count_rows),
        ("conflict_graph_diagnostic.csv", conflict_diagnostic_rows),
        ("makespan_summary.csv", makespan_rows),
        ("runtime_summary.csv", runtime_rows),
        ("mapf7_differing_instances.csv", mapf7_differing_rows),
        ("cbs_quality_reference.csv", cbs_rows),
    ]

    for filename, rows in table_specs:
        if not rows:
            fieldnames: list[str] = []
        else:
            fieldnames = []
            for row in rows:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
        _write_csv(output_dir / filename, rows, fieldnames)
        tables_written.append(filename)

    _write_text(
        output_dir / "analysis_summary.md",
        build_analysis_summary_md(
            config=config,
            validation=validation,
            instance_rows=instance_rows,
            paired_summary=paired_summary,
            cdf_direction_rows=cdf_direction_rows,
            spf_cd_rows=spf_cd_rows,
            priority_sensitive_rows=priority_sensitive_rows,
            interaction_rows=interaction_rows,
            conflict_diagnostic_rows=conflict_diagnostic_rows,
            runtime_rows=runtime_rows,
            cbs_aggregate=cbs_aggregate,
            mapf7_differing_rows=mapf7_differing_rows,
        ),
    )
    tables_written.append("analysis_summary.md")

    _write_text(
        output_dir / "thesis_tables.md",
        build_thesis_tables_md(
            instance_rows=instance_rows,
            paired_summary=paired_summary,
            cdf_direction_rows=cdf_direction_rows,
            priority_sensitive_rows=priority_sensitive_rows,
            interaction_rows=interaction_rows,
            runtime_rows=runtime_rows,
            cbs_aggregate=cbs_aggregate,
            mapf7_differing_rows=mapf7_differing_rows,
        ),
    )
    tables_written.append("thesis_tables.md")

    plots_written: tuple[str, ...] = ()
    if plot_writer is not None:
        runtime_aggregate = next(
            row for row in runtime_rows if row.get("metric_group") == "runtime_aggregate"
        )
        plots_written = plot_writer(
            output_dir,
            instance_rows,
            cdf_direction_rows=cdf_direction_rows,
            conflict_diagnostic_rows=conflict_diagnostic_rows,
            runtime_aggregate=runtime_aggregate,
        )

    return ConflictAwareAnalysisResult(
        validation=validation,
        output_dir=output_dir,
        tables_written=tuple(tables_written),
        plots_written=plots_written,
    )
