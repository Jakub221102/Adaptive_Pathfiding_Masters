from __future__ import annotations

import csv
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pathfinding.src.experiments.mapf_benchmark_execution import (
    DEFAULT_BENCHMARK_ALGORITHMS,
    MAPFBenchmarkAlgorithm,
    MAPFBenchmarkRunKey,
    MAPFBenchmarkRunRecord,
    MAPFCBSSearchMetrics,
    MAPFPPSearchMetrics,
    TERMINATION_ERROR,
    TERMINATION_EXPANSION_LIMIT,
    TERMINATION_FAILURE,
    TERMINATION_SUCCESS,
    TERMINATION_TIME_LIMIT,
    load_checkpoint_records,
    run_key,
)
from pathfinding.src.experiments.mapf_benchmark_instances import (
    MAPFBenchmarkManifest,
    load_benchmark_manifest,
)

COMMON_BUDGET_MS = 180_000.0

ALGORITHM_ORDER: tuple[MAPFBenchmarkAlgorithm, ...] = DEFAULT_BENCHMARK_ALGORITHMS

AGENT_COUNT_ORDER: tuple[int, ...] = (5, 10, 20)

INTERACTION_ORDER: tuple[str, ...] = ("low", "medium", "high")

VALID_TERMINATION_REASONS: frozenset[str] = frozenset(
    {
        TERMINATION_SUCCESS,
        TERMINATION_FAILURE,
        TERMINATION_EXPANSION_LIMIT,
        TERMINATION_TIME_LIMIT,
        TERMINATION_ERROR,
    }
)

EXPECTED_TERMINATION_COUNTS: dict[str, dict[str, int]] = {
    MAPFBenchmarkAlgorithm.FIXED_PRIORITY_PP.value: {TERMINATION_SUCCESS: 27},
    MAPFBenchmarkAlgorithm.CBS_BASIC.value: {
        TERMINATION_SUCCESS: 22,
        TERMINATION_EXPANSION_LIMIT: 1,
        TERMINATION_TIME_LIMIT: 4,
    },
    MAPFBenchmarkAlgorithm.CBS_CARDINAL_FIRST.value: {
        TERMINATION_SUCCESS: 21,
        TERMINATION_TIME_LIMIT: 6,
    },
    "__overall__": {
        TERMINATION_SUCCESS: 70,
        TERMINATION_EXPANSION_LIMIT: 1,
        TERMINATION_TIME_LIMIT: 10,
        TERMINATION_FAILURE: 0,
        TERMINATION_ERROR: 0,
    },
}

OUTPUT_TABLES: tuple[str, ...] = (
    "dataset_validation.txt",
    "termination_summary.csv",
    "success_rate_by_group.csv",
    "runtime_success_summary.csv",
    "runtime_all_observed_summary.csv",
    "solution_quality_summary.csv",
    "paired_quality_comparison.csv",
    "pp_vs_basic_quality_summary.csv",
    "pp_vs_cardinal_quality_summary.csv",
    "basic_vs_cardinal_quality_summary.csv",
    "cbs_search_effort_summary.csv",
    "cardinal_overhead_summary.csv",
    "common_180s_budget_summary.csv",
    "non_success_runs.csv",
    "high_interaction_outcomes.csv",
    "instance_level_analysis.csv",
    "analysis_summary.md",
)


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    results_jsonl_path: Path
    manifest_path: Path
    output_dir: Path
    expected_record_count: int = 81
    expected_instance_count: int = 27
    common_budget_ms: float = COMMON_BUDGET_MS


@dataclass
class ValidationReport:
    passed: bool
    messages: list[str] = field(default_factory=list)

    def add(self, message: str, *, passed: bool) -> None:
        self.messages.append(message)
        if not passed:
            self.passed = False


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    validation: ValidationReport
    output_dir: Path
    tables_written: tuple[str, ...]


def _algorithm_sort_key(algorithm: MAPFBenchmarkAlgorithm) -> int:
    return ALGORITHM_ORDER.index(algorithm)


def _interaction_sort_key(level: str) -> int:
    normalized = level.lower()
    if normalized in INTERACTION_ORDER:
        return INTERACTION_ORDER.index(normalized)
    return len(INTERACTION_ORDER)


def _record_sort_key(record: MAPFBenchmarkRunRecord) -> tuple[int, int, int, str]:
    return (
        _algorithm_sort_key(record.algorithm),
        AGENT_COUNT_ORDER.index(record.agent_count)
        if record.agent_count in AGENT_COUNT_ORDER
        else record.agent_count,
        _interaction_sort_key(record.interaction_level),
        record.instance_id,
    )


def sort_records(
    records: Sequence[MAPFBenchmarkRunRecord],
) -> tuple[MAPFBenchmarkRunRecord, ...]:
    return tuple(sorted(records, key=_record_sort_key))


def load_analysis_records(jsonl_path: Path) -> tuple[MAPFBenchmarkRunRecord, ...]:
    records_by_key = load_checkpoint_records(jsonl_path)
    return sort_records(records_by_key.values())


def _group_sort_key(
    algorithm: str,
    agent_count: int | None = None,
    interaction_level: str | None = None,
) -> tuple[int, int, int]:
    algo = MAPFBenchmarkAlgorithm(algorithm)
    agent_index = (
        AGENT_COUNT_ORDER.index(agent_count)
        if agent_count is not None and agent_count in AGENT_COUNT_ORDER
        else -1
    )
    interaction_index = (
        _interaction_sort_key(interaction_level)
        if interaction_level is not None
        else -1
    )
    return (_algorithm_sort_key(algo), agent_index, interaction_index)


def _blank_if_none(value: Any) -> Any:
    if value is None:
        return ""
    return value


def _format_number(value: float | int | None, *, digits: int = 4) -> str | float | int:
    if value is None or value == "":
        return ""
    if isinstance(value, int):
        return value
    return round(float(value), digits)


def _summary_stats(values: Sequence[float | int]) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "mean": "",
            "median": "",
            "min": "",
            "max": "",
            "std": "",
        }

    numeric = [float(value) for value in values]
    result: dict[str, Any] = {
        "n": len(numeric),
        "mean": statistics.mean(numeric),
        "median": statistics.median(numeric),
        "min": min(numeric),
        "max": max(numeric),
    }
    if len(numeric) >= 2:
        result["std"] = statistics.stdev(numeric)
    else:
        result["std"] = ""
    return result


def _records_by_key(
    records: Sequence[MAPFBenchmarkRunRecord],
) -> dict[MAPFBenchmarkRunKey, MAPFBenchmarkRunRecord]:
    return {run_key(record): record for record in records}


def _records_by_instance(
    records: Sequence[MAPFBenchmarkRunRecord],
) -> dict[str, dict[MAPFBenchmarkAlgorithm, MAPFBenchmarkRunRecord]]:
    grouped: dict[str, dict[MAPFBenchmarkAlgorithm, MAPFBenchmarkRunRecord]] = {}
    for record in records:
        grouped.setdefault(record.instance_id, {})[record.algorithm] = record
    return grouped


def validate_dataset(
    records: Sequence[MAPFBenchmarkRunRecord],
    manifest: MAPFBenchmarkManifest,
    *,
    expected_record_count: int,
    expected_instance_count: int,
    expected_termination_counts: dict[str, dict[str, int]] | None = None,
) -> ValidationReport:
    report = ValidationReport(passed=True)
    expected_counts = expected_termination_counts or EXPECTED_TERMINATION_COUNTS
    manifest_ids = {instance.instance_id for instance in manifest.instances}

    report.add(
        f"record_count={len(records)} (expected {expected_record_count})",
        passed=len(records) == expected_record_count,
    )

    instance_ids = {record.instance_id for record in records}
    report.add(
        f"unique_instance_count={len(instance_ids)} (expected {expected_instance_count})",
        passed=len(instance_ids) == expected_instance_count,
    )

    seen_keys: set[MAPFBenchmarkRunKey] = set()
    duplicate_keys: list[str] = []
    invalid_algorithms: list[str] = []
    invalid_terminations: list[str] = []
    unknown_instances: list[str] = []

    for record in records:
        key = run_key(record)
        if key in seen_keys:
            duplicate_keys.append(f"{key.instance_id}/{key.algorithm.value}")
        seen_keys.add(key)

        if record.algorithm not in ALGORITHM_ORDER:
            invalid_algorithms.append(record.algorithm.value)

        if record.termination_reason not in VALID_TERMINATION_REASONS:
            invalid_terminations.append(
                f"{record.instance_id}/{record.algorithm.value}: "
                f"{record.termination_reason}"
            )

        if record.instance_id not in manifest_ids:
            unknown_instances.append(record.instance_id)

    report.add(
        f"duplicate_keys={len(duplicate_keys)}",
        passed=len(duplicate_keys) == 0,
    )
    if duplicate_keys:
        report.messages.append("  duplicates: " + ", ".join(sorted(duplicate_keys)))

    report.add(
        f"invalid_algorithms={len(invalid_algorithms)}",
        passed=len(invalid_algorithms) == 0,
    )

    report.add(
        f"invalid_termination_reasons={len(invalid_terminations)}",
        passed=len(invalid_terminations) == 0,
    )
    if invalid_terminations:
        report.messages.extend(f"  {item}" for item in invalid_terminations)

    report.add(
        f"unknown_instance_ids={len(unknown_instances)}",
        passed=len(unknown_instances) == 0,
    )

    missing_combinations: list[str] = []
    extra_per_instance: list[str] = []
    for instance_id in sorted(instance_ids):
        algorithms_for_instance = {
            record.algorithm
            for record in records
            if record.instance_id == instance_id
        }
        if len(algorithms_for_instance) != len(ALGORITHM_ORDER):
            extra_per_instance.append(
                f"{instance_id}: {len(algorithms_for_instance)} algorithms"
            )
        for algorithm in ALGORITHM_ORDER:
            if algorithm not in algorithms_for_instance:
                missing_combinations.append(f"{instance_id}/{algorithm.value}")

    report.add(
        f"missing_algorithm_combinations={len(missing_combinations)}",
        passed=len(missing_combinations) == 0,
    )
    if missing_combinations:
        report.messages.append(
            "  missing: " + ", ".join(missing_combinations[:10])
            + (" ..." if len(missing_combinations) > 10 else "")
        )

    report.add(
        f"instances_without_three_algorithms={len(extra_per_instance)}",
        passed=len(extra_per_instance) == 0,
    )

    termination_by_algorithm: dict[str, dict[str, int]] = {
        algorithm.value: {} for algorithm in ALGORITHM_ORDER
    }
    overall_counts: dict[str, int] = {}
    for record in records:
        algorithm_counts = termination_by_algorithm.setdefault(
            record.algorithm.value, {}
        )
        algorithm_counts[record.termination_reason] = (
            algorithm_counts.get(record.termination_reason, 0) + 1
        )
        overall_counts[record.termination_reason] = (
            overall_counts.get(record.termination_reason, 0) + 1
        )

    for algorithm_value, expected_for_algorithm in expected_counts.items():
        if algorithm_value == "__overall__":
            continue
        actual = termination_by_algorithm.get(algorithm_value, {})
        for reason, expected_count in expected_for_algorithm.items():
            actual_count = actual.get(reason, 0)
            report.add(
                f"{algorithm_value}.{reason}={actual_count} (expected {expected_count})",
                passed=actual_count == expected_count,
            )

    overall_expected = expected_counts["__overall__"]
    for reason, expected_count in overall_expected.items():
        actual_count = overall_counts.get(reason, 0)
        report.add(
            f"overall.{reason}={actual_count} (expected {expected_count})",
            passed=actual_count == expected_count,
        )

    return report


def within_common_budget(record: MAPFBenchmarkRunRecord, budget_ms: float) -> bool:
    if not record.success:
        return False
    if record.algorithm in (
        MAPFBenchmarkAlgorithm.CBS_BASIC,
        MAPFBenchmarkAlgorithm.CBS_CARDINAL_FIRST,
    ):
        return record.success
    return record.execution_time_ms <= budget_ms


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _blank_if_none(row.get(key)) for key in fieldnames})


def build_termination_summary(
    records: Sequence[MAPFBenchmarkRunRecord],
) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    for record in records:
        key = (record.algorithm.value, record.termination_reason)
        counts[key] = counts.get(key, 0) + 1

    rows = [
        {
            "algorithm": algorithm,
            "termination_reason": reason,
            "count": count,
        }
        for (algorithm, reason), count in counts.items()
    ]
    rows.sort(
        key=lambda row: (
            _group_sort_key(row["algorithm"]),
            list(VALID_TERMINATION_REASONS).index(row["termination_reason"])
            if row["termination_reason"] in VALID_TERMINATION_REASONS
            else 99,
        )
    )
    return rows


def build_success_rate_by_group(
    records: Sequence[MAPFBenchmarkRunRecord],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[MAPFBenchmarkRunRecord]] = {}
    for record in records:
        key = (
            record.algorithm.value,
            record.agent_count,
            record.interaction_level.lower(),
        )
        grouped.setdefault(key, []).append(record)

    rows: list[dict[str, Any]] = []
    for (algorithm, agent_count, interaction_level), group_records in grouped.items():
        success_count = sum(1 for record in group_records if record.success)
        total = len(group_records)
        rows.append(
            {
                "algorithm": algorithm,
                "agent_count": agent_count,
                "interaction_level": interaction_level,
                "total_runs": total,
                "success_count": success_count,
                "success_rate": success_count / total if total else "",
            }
        )

    rows.sort(
        key=lambda row: _group_sort_key(
            row["algorithm"],
            row["agent_count"],
            row["interaction_level"],
        )
    )
    return rows


def _runtime_group_rows(
    records: Sequence[MAPFBenchmarkRunRecord],
    *,
    success_only: bool,
    metric_label: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[float]] = {}
    for record in records:
        if success_only and not record.success:
            continue
        key = (
            record.algorithm.value,
            record.agent_count,
            record.interaction_level.lower(),
        )
        grouped.setdefault(key, []).append(record.execution_time_ms)

    rows: list[dict[str, Any]] = []
    for (algorithm, agent_count, interaction_level), values in grouped.items():
        stats = _summary_stats(values)
        rows.append(
            {
                "algorithm": algorithm,
                "agent_count": agent_count,
                "interaction_level": interaction_level,
                "metric": metric_label,
                "n": stats["n"],
                "mean_ms": stats["mean"],
                "median_ms": stats["median"],
                "min_ms": stats["min"],
                "max_ms": stats["max"],
                "std_ms": stats["std"],
            }
        )

    rows.sort(
        key=lambda row: _group_sort_key(
            row["algorithm"],
            row["agent_count"],
            row["interaction_level"],
        )
    )
    return rows


def build_runtime_success_summary(
    records: Sequence[MAPFBenchmarkRunRecord],
) -> list[dict[str, Any]]:
    return _runtime_group_rows(records, success_only=True, metric_label="solve_time")


def build_runtime_all_observed_summary(
    records: Sequence[MAPFBenchmarkRunRecord],
) -> list[dict[str, Any]]:
    return _runtime_group_rows(
        records,
        success_only=False,
        metric_label="observed_execution_time",
    )


def build_solution_quality_summary(
    records: Sequence[MAPFBenchmarkRunRecord],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[MAPFBenchmarkRunRecord]] = {}
    for record in records:
        if not record.success:
            continue
        key = (
            record.algorithm.value,
            record.agent_count,
            record.interaction_level.lower(),
        )
        grouped.setdefault(key, []).append(record)

    rows: list[dict[str, Any]] = []
    for (algorithm, agent_count, interaction_level), group_records in grouped.items():
        soc_stats = _summary_stats([record.soc for record in group_records if record.soc is not None])
        makespan_stats = _summary_stats(
            [record.makespan for record in group_records if record.makespan is not None]
        )
        rows.append(
            {
                "algorithm": algorithm,
                "agent_count": agent_count,
                "interaction_level": interaction_level,
                "n": soc_stats["n"],
                "mean_soc": soc_stats["mean"],
                "median_soc": soc_stats["median"],
                "min_soc": soc_stats["min"],
                "max_soc": soc_stats["max"],
                "std_soc": soc_stats["std"],
                "mean_makespan": makespan_stats["mean"],
                "median_makespan": makespan_stats["median"],
                "min_makespan": makespan_stats["min"],
                "max_makespan": makespan_stats["max"],
                "std_makespan": makespan_stats["std"],
            }
        )

    rows.sort(
        key=lambda row: _group_sort_key(
            row["algorithm"],
            row["agent_count"],
            row["interaction_level"],
        )
    )
    return rows


def _paired_rows_for_algorithms(
    records_by_instance: dict[str, dict[MAPFBenchmarkAlgorithm, MAPFBenchmarkRunRecord]],
    left: MAPFBenchmarkAlgorithm,
    right: MAPFBenchmarkAlgorithm,
    *,
    comparison: str,
    soc_diff_column: str,
    makespan_diff_column: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for instance_id in sorted(records_by_instance):
        instance_records = records_by_instance[instance_id]
        left_record = instance_records.get(left)
        right_record = instance_records.get(right)
        if left_record is None or right_record is None:
            continue
        if not left_record.success or not right_record.success:
            continue

        left_soc = left_record.soc
        right_soc = right_record.soc
        left_makespan = left_record.makespan
        right_makespan = right_record.makespan
        if (
            left_soc is None
            or right_soc is None
            or left_makespan is None
            or right_makespan is None
        ):
            continue

        rows.append(
            {
                "comparison": comparison,
                "instance_id": instance_id,
                "agent_count": left_record.agent_count,
                "interaction_level": left_record.interaction_level.lower(),
                "independent_conflict_count": left_record.independent_conflict_count,
                "left_algorithm": left.value,
                "right_algorithm": right.value,
                "left_soc": left_soc,
                "right_soc": right_soc,
                "left_makespan": left_makespan,
                "right_makespan": right_makespan,
                soc_diff_column: left_soc - right_soc,
                makespan_diff_column: left_makespan - right_makespan,
            }
        )
    return rows


def build_paired_quality_comparison(
    records: Sequence[MAPFBenchmarkRunRecord],
) -> list[dict[str, Any]]:
    records_by_instance = _records_by_instance(records)
    rows: list[dict[str, Any]] = []

    pair_specs = (
        (
            MAPFBenchmarkAlgorithm.FIXED_PRIORITY_PP,
            MAPFBenchmarkAlgorithm.CBS_BASIC,
            "pp_minus_basic",
            "pp_minus_basic_soc",
            "pp_minus_basic_makespan",
        ),
        (
            MAPFBenchmarkAlgorithm.FIXED_PRIORITY_PP,
            MAPFBenchmarkAlgorithm.CBS_CARDINAL_FIRST,
            "pp_minus_cardinal",
            "pp_minus_cardinal_soc",
            "pp_minus_cardinal_makespan",
        ),
        (
            MAPFBenchmarkAlgorithm.CBS_BASIC,
            MAPFBenchmarkAlgorithm.CBS_CARDINAL_FIRST,
            "basic_minus_cardinal",
            "basic_minus_cardinal_soc",
            "basic_minus_cardinal_makespan",
        ),
    )

    for left, right, comparison, soc_col, makespan_col in pair_specs:
        rows.extend(
            _paired_rows_for_algorithms(
                records_by_instance,
                left,
                right,
                comparison=comparison,
                soc_diff_column=soc_col,
                makespan_diff_column=makespan_col,
            )
        )

    rows.sort(
        key=lambda row: (
            row["comparison"],
            row["agent_count"],
            _interaction_sort_key(row["interaction_level"]),
            row["instance_id"],
        )
    )
    return rows


@dataclass(frozen=True, slots=True)
class DirectionalSocCounts:
    left_better: int
    equal: int
    right_better: int


def count_directional_soc_better(
    soc_diffs: Sequence[float | int],
) -> DirectionalSocCounts:
    """Count paired SoC outcomes where diff = left_soc - right_soc."""
    return DirectionalSocCounts(
        left_better=sum(1 for value in soc_diffs if value < 0),
        equal=sum(1 for value in soc_diffs if value == 0),
        right_better=sum(1 for value in soc_diffs if value > 0),
    )


def _quality_difference_summary(
    paired_rows: Sequence[dict[str, Any]],
    *,
    comparison: str,
    diff_prefix: str,
) -> list[dict[str, Any]]:
    filtered = [row for row in paired_rows if row["comparison"] == comparison]
    if not filtered:
        return [
            {
                "comparison": comparison,
                "paired_instance_count": 0,
                "mean_soc_diff": "",
                "median_soc_diff": "",
                "mean_makespan_diff": "",
                "median_makespan_diff": "",
                "left_better_soc_count": 0,
                "right_better_soc_count": 0,
                "equal_soc_count": 0,
            }
        ]

    soc_column = f"{diff_prefix}_soc"
    makespan_column = f"{diff_prefix}_makespan"
    soc_diffs = [row[soc_column] for row in filtered]
    makespan_diffs = [row[makespan_column] for row in filtered]
    soc_stats = _summary_stats(soc_diffs)
    makespan_stats = _summary_stats(makespan_diffs)

    directional_counts = count_directional_soc_better(soc_diffs)

    return [
        {
            "comparison": comparison,
            "paired_instance_count": len(filtered),
            "mean_soc_diff": soc_stats["mean"],
            "median_soc_diff": soc_stats["median"],
            "std_soc_diff": soc_stats["std"],
            "mean_makespan_diff": makespan_stats["mean"],
            "median_makespan_diff": makespan_stats["median"],
            "std_makespan_diff": makespan_stats["std"],
            "left_better_soc_count": directional_counts.left_better,
            "right_better_soc_count": directional_counts.right_better,
            "equal_soc_count": directional_counts.equal,
            "soc_diff_column": f"{diff_prefix}_soc",
            "makespan_diff_column": f"{diff_prefix}_makespan",
            "soc_diff_interpretation": (
                f"Positive {diff_prefix}_soc means the right-hand algorithm achieved lower SoC."
            ),
        }
    ]


def build_pp_vs_basic_quality_summary(
    paired_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _quality_difference_summary(
        paired_rows,
        comparison="pp_minus_basic",
        diff_prefix="pp_minus_basic",
    )


def build_pp_vs_cardinal_quality_summary(
    paired_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _quality_difference_summary(
        paired_rows,
        comparison="pp_minus_cardinal",
        diff_prefix="pp_minus_cardinal",
    )


def build_basic_vs_cardinal_quality_summary(
    paired_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _quality_difference_summary(
        paired_rows,
        comparison="basic_minus_cardinal",
        diff_prefix="basic_minus_cardinal",
    )


def _cbs_metric_values(
    record: MAPFBenchmarkRunRecord,
    accessor: Callable[[MAPFCBSSearchMetrics], int],
) -> int | None:
    if record.cbs_search_metrics is None:
        return None
    return accessor(record.cbs_search_metrics)


def build_cbs_search_effort_summary(
    records: Sequence[MAPFBenchmarkRunRecord],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[MAPFBenchmarkRunRecord]] = {}
    for record in records:
        if record.algorithm not in (
            MAPFBenchmarkAlgorithm.CBS_BASIC,
            MAPFBenchmarkAlgorithm.CBS_CARDINAL_FIRST,
        ):
            continue
        if not record.success:
            continue
        key = (
            record.algorithm.value,
            record.agent_count,
            record.interaction_level.lower(),
        )
        grouped.setdefault(key, []).append(record)

    metric_accessors: dict[str, Callable[[MAPFCBSSearchMetrics], int]] = {
        "expanded_ct_nodes": lambda metrics: metrics.expanded_ct_nodes,
        "generated_ct_nodes": lambda metrics: metrics.generated_ct_nodes,
        "low_level_replans": lambda metrics: metrics.low_level_replans,
        "max_open_size": lambda metrics: metrics.max_open_size,
        "classified_conflicts": lambda metrics: metrics.classified_conflicts,
        "classification_low_level_searches": (
            lambda metrics: metrics.classification_low_level_searches
        ),
        "selected_cardinal_conflicts": lambda metrics: metrics.selected_cardinal_conflicts,
        "selected_semi_cardinal_conflicts": (
            lambda metrics: metrics.selected_semi_cardinal_conflicts
        ),
        "selected_non_cardinal_conflicts": (
            lambda metrics: metrics.selected_non_cardinal_conflicts
        ),
        "combined_low_level_searches": lambda metrics: metrics.combined_low_level_searches,
    }

    rows: list[dict[str, Any]] = []
    for (algorithm, agent_count, interaction_level), group_records in grouped.items():
        row: dict[str, Any] = {
            "algorithm": algorithm,
            "agent_count": agent_count,
            "interaction_level": interaction_level,
            "n": len(group_records),
        }
        for metric_name, accessor in metric_accessors.items():
            values = [
                value
                for record in group_records
                if (value := _cbs_metric_values(record, accessor)) is not None
            ]
            stats = _summary_stats(values)
            row[f"mean_{metric_name}"] = stats["mean"]
            row[f"median_{metric_name}"] = stats["median"]
            row[f"min_{metric_name}"] = stats["min"]
            row[f"max_{metric_name}"] = stats["max"]
            row[f"std_{metric_name}"] = stats["std"]
        rows.append(row)

    rows.sort(
        key=lambda row: _group_sort_key(
            row["algorithm"],
            row["agent_count"],
            row["interaction_level"],
        )
    )
    return rows


def build_cardinal_overhead_summary(
    records: Sequence[MAPFBenchmarkRunRecord],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records_by_instance = _records_by_instance(records)
    instance_rows: list[dict[str, Any]] = []

    for instance_id in sorted(records_by_instance):
        instance_records = records_by_instance[instance_id]
        basic = instance_records.get(MAPFBenchmarkAlgorithm.CBS_BASIC)
        cardinal = instance_records.get(MAPFBenchmarkAlgorithm.CBS_CARDINAL_FIRST)
        if basic is None or cardinal is None:
            continue
        if not basic.success or not cardinal.success:
            continue
        if basic.cbs_search_metrics is None or cardinal.cbs_search_metrics is None:
            continue

        basic_metrics = basic.cbs_search_metrics
        cardinal_metrics = cardinal.cbs_search_metrics

        basic_runtime = basic.execution_time_ms
        cardinal_runtime = cardinal.execution_time_ms
        runtime_ratio = cardinal_runtime / basic_runtime if basic_runtime else ""

        basic_ct = basic_metrics.expanded_ct_nodes
        cardinal_ct = cardinal_metrics.expanded_ct_nodes
        ct_reduction = basic_ct - cardinal_ct

        basic_ll = basic_metrics.combined_low_level_searches
        cardinal_ll = cardinal_metrics.combined_low_level_searches
        additional_ll = cardinal_ll - basic_ll

        instance_rows.append(
            {
                "instance_id": instance_id,
                "agent_count": basic.agent_count,
                "interaction_level": basic.interaction_level.lower(),
                "independent_conflict_count": basic.independent_conflict_count,
                "basic_runtime_ms": basic_runtime,
                "cardinal_runtime_ms": cardinal_runtime,
                "runtime_ratio_cardinal_over_basic": runtime_ratio,
                "basic_expanded_ct_nodes": basic_ct,
                "cardinal_expanded_ct_nodes": cardinal_ct,
                "ct_expansion_reduction_basic_minus_cardinal": ct_reduction,
                "basic_combined_low_level_searches": basic_ll,
                "cardinal_low_level_replans": cardinal_metrics.low_level_replans,
                "cardinal_classification_low_level_searches": (
                    cardinal_metrics.classification_low_level_searches
                ),
                "cardinal_combined_low_level_searches": cardinal_ll,
                "additional_low_level_searches_cardinal_minus_basic": additional_ll,
                "cardinal_classified_conflicts": cardinal_metrics.classified_conflicts,
                "cardinal_selected_cardinal_conflicts": (
                    cardinal_metrics.selected_cardinal_conflicts
                ),
                "cardinal_selected_semi_cardinal_conflicts": (
                    cardinal_metrics.selected_semi_cardinal_conflicts
                ),
                "cardinal_selected_non_cardinal_conflicts": (
                    cardinal_metrics.selected_non_cardinal_conflicts
                ),
            }
        )

    aggregate_rows: list[dict[str, Any]] = []
    if instance_rows:
        aggregate_rows.append(
            {
                "summary_level": "common_success_instances",
                "paired_instance_count": len(instance_rows),
                "mean_runtime_ratio": _summary_stats(
                    [
                        row["runtime_ratio_cardinal_over_basic"]
                        for row in instance_rows
                        if row["runtime_ratio_cardinal_over_basic"] != ""
                    ]
                )["mean"],
                "mean_ct_reduction": _summary_stats(
                    [row["ct_expansion_reduction_basic_minus_cardinal"] for row in instance_rows]
                )["mean"],
                "mean_additional_low_level_searches": _summary_stats(
                    [
                        row["additional_low_level_searches_cardinal_minus_basic"]
                        for row in instance_rows
                    ]
                )["mean"],
                "median_runtime_ratio": _summary_stats(
                    [
                        row["runtime_ratio_cardinal_over_basic"]
                        for row in instance_rows
                        if row["runtime_ratio_cardinal_over_basic"] != ""
                    ]
                )["median"],
            }
        )

    combined_rows = instance_rows + aggregate_rows
    return instance_rows, combined_rows


def build_common_180s_budget_summary(
    records: Sequence[MAPFBenchmarkRunRecord],
    *,
    budget_ms: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for algorithm in ALGORITHM_ORDER:
        algorithm_records = [record for record in records if record.algorithm == algorithm]
        total = len(algorithm_records)
        observed_success = sum(1 for record in algorithm_records if record.success)
        within_budget = sum(
            1 for record in algorithm_records if within_common_budget(record, budget_ms)
        )

        if algorithm == MAPFBenchmarkAlgorithm.FIXED_PRIORITY_PP:
            methodology = (
                "Post-hoc secondary comparison: PP observed success AND "
                f"execution_time_ms <= {budget_ms:.0f}; original termination unchanged."
            )
        else:
            methodology = (
                "CBS runs were executed with an enforced "
                f"{budget_ms / 1000:.0f} s wall-clock limit; observed success equals "
                "within-budget success."
            )

        rows.append(
            {
                "algorithm": algorithm.value,
                "total_runs": total,
                "observed_success_count": observed_success,
                "observed_success_rate": observed_success / total if total else "",
                "within_common_budget_success_count": within_budget,
                "within_common_budget_success_rate": within_budget / total if total else "",
                "common_budget_ms": budget_ms,
                "comparison_scope": "secondary_post_hoc",
                "methodology_note": methodology,
            }
        )

    return rows


def build_non_success_runs(
    records: Sequence[MAPFBenchmarkRunRecord],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.success:
            continue
        row: dict[str, Any] = {
            "instance_id": record.instance_id,
            "algorithm": record.algorithm.value,
            "agent_count": record.agent_count,
            "interaction_level": record.interaction_level.lower(),
            "independent_conflict_count": record.independent_conflict_count,
            "termination_reason": record.termination_reason,
            "execution_time_ms": record.execution_time_ms,
            "error_message": record.error_message,
        }
        if record.cbs_search_metrics is not None:
            row["cbs_expanded_ct_nodes"] = record.cbs_search_metrics.expanded_ct_nodes
            row["cbs_generated_ct_nodes"] = record.cbs_search_metrics.generated_ct_nodes
            row["cbs_low_level_replans"] = record.cbs_search_metrics.low_level_replans
            row["cbs_combined_low_level_searches"] = (
                record.cbs_search_metrics.combined_low_level_searches
            )
        rows.append(row)

    rows.sort(
        key=lambda row: (
            _group_sort_key(row["algorithm"], row["agent_count"], row["interaction_level"]),
            row["instance_id"],
        )
    )
    return rows


def build_high_interaction_outcomes(
    records: Sequence[MAPFBenchmarkRunRecord],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.interaction_level.lower() != "high":
            continue
        row: dict[str, Any] = {
            "instance_id": record.instance_id,
            "agent_count": record.agent_count,
            "interaction_level": record.interaction_level.lower(),
            "independent_conflict_count": record.independent_conflict_count,
            "independent_soc": record.independent_soc,
            "independent_makespan": record.independent_makespan,
            "algorithm": record.algorithm.value,
            "termination_reason": record.termination_reason,
            "success": record.success,
            "execution_time_ms": record.execution_time_ms,
            "soc": record.soc,
            "makespan": record.makespan,
        }
        if record.cbs_search_metrics is not None:
            metrics = record.cbs_search_metrics
            row.update(
                {
                    "cbs_expanded_ct_nodes": metrics.expanded_ct_nodes,
                    "cbs_generated_ct_nodes": metrics.generated_ct_nodes,
                    "cbs_low_level_replans": metrics.low_level_replans,
                    "cbs_combined_low_level_searches": metrics.combined_low_level_searches,
                    "cbs_classified_conflicts": metrics.classified_conflicts,
                    "cbs_classification_low_level_searches": (
                        metrics.classification_low_level_searches
                    ),
                    "cbs_selected_cardinal_conflicts": metrics.selected_cardinal_conflicts,
                    "cbs_selected_semi_cardinal_conflicts": (
                        metrics.selected_semi_cardinal_conflicts
                    ),
                    "cbs_selected_non_cardinal_conflicts": (
                        metrics.selected_non_cardinal_conflicts
                    ),
                }
            )
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["agent_count"],
            row["instance_id"],
            _group_sort_key(row["algorithm"]),
        )
    )
    return rows


def build_instance_level_analysis(
    records: Sequence[MAPFBenchmarkRunRecord],
) -> list[dict[str, Any]]:
    records_by_instance = _records_by_instance(records)
    rows: list[dict[str, Any]] = []

    for instance_id in sorted(records_by_instance):
        instance_records = records_by_instance[instance_id]
        sample = next(iter(instance_records.values()))
        row: dict[str, Any] = {
            "instance_id": instance_id,
            "agent_count": sample.agent_count,
            "interaction_level": sample.interaction_level.lower(),
            "independent_conflict_count": sample.independent_conflict_count,
            "independent_soc": sample.independent_soc,
            "independent_makespan": sample.independent_makespan,
        }

        for algorithm in ALGORITHM_ORDER:
            prefix = algorithm.value
            record = instance_records.get(algorithm)
            if record is None:
                row[f"{prefix}_termination_reason"] = ""
                row[f"{prefix}_success"] = ""
                row[f"{prefix}_execution_time_ms"] = ""
                row[f"{prefix}_soc"] = ""
                row[f"{prefix}_makespan"] = ""
                row[f"{prefix}_within_180s"] = ""
                continue

            row[f"{prefix}_termination_reason"] = record.termination_reason
            row[f"{prefix}_success"] = record.success
            row[f"{prefix}_execution_time_ms"] = record.execution_time_ms
            row[f"{prefix}_soc"] = record.soc
            row[f"{prefix}_makespan"] = record.makespan
            row[f"{prefix}_within_180s"] = within_common_budget(
                record,
                COMMON_BUDGET_MS,
            )

            if record.cbs_search_metrics is not None:
                row[f"{prefix}_expanded_ct_nodes"] = (
                    record.cbs_search_metrics.expanded_ct_nodes
                )
                row[f"{prefix}_combined_low_level_searches"] = (
                    record.cbs_search_metrics.combined_low_level_searches
                )

        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["agent_count"],
            _interaction_sort_key(row["interaction_level"]),
            row["instance_id"],
        )
    )
    return rows


def _format_validation_report(report: ValidationReport) -> str:
    status = "PASSED" if report.passed else "FAILED"
    lines = [f"Dataset validation: {status}", ""]
    lines.extend(report.messages)
    return "\n".join(lines) + "\n"


def _summarize_group_success(
    records: Sequence[MAPFBenchmarkRunRecord],
) -> list[str]:
    lines: list[str] = []
    for agent_count in AGENT_COUNT_ORDER:
        for interaction in INTERACTION_ORDER:
            group = [
                record
                for record in records
                if record.agent_count == agent_count
                and record.interaction_level.lower() == interaction
            ]
            if not group:
                continue
            total = len(group)
            success = sum(1 for record in group if record.success)
            lines.append(
                f"- agents={agent_count}, interaction={interaction}: "
                f"{success}/{total} success across all algorithms"
            )
    return lines


def build_analysis_summary_markdown(
    records: Sequence[MAPFBenchmarkRunRecord],
    validation: ValidationReport,
    *,
    budget_ms: float,
    paired_rows: Sequence[dict[str, Any]],
    cardinal_instance_rows: Sequence[dict[str, Any]],
    common_budget_rows: Sequence[dict[str, Any]],
) -> str:
    lines: list[str] = [
        "# MAPF Benchmark Analysis Summary (MAPF-5C.1)",
        "",
        "## Dataset validation",
        "",
        f"- Validation status: **{'PASSED' if validation.passed else 'FAILED'}**",
        f"- Total runs analyzed: {len(records)}",
        f"- Unique instances: {len({record.instance_id for record in records})}",
        "",
        "## Success by algorithm",
        "",
    ]

    for algorithm in ALGORITHM_ORDER:
        algorithm_records = [record for record in records if record.algorithm == algorithm]
        success_count = sum(1 for record in algorithm_records if record.success)
        lines.append(
            f"- `{algorithm.value}`: {success_count}/{len(algorithm_records)} success"
        )

    lines.extend(["", "## Success by agent count and interaction level", ""])
    lines.extend(_summarize_group_success(records))

    non_success = [record for record in records if not record.success]
    timeout_count = sum(
        1 for record in non_success if record.termination_reason == TERMINATION_TIME_LIMIT
    )
    expansion_count = sum(
        1
        for record in non_success
        if record.termination_reason == TERMINATION_EXPANSION_LIMIT
    )
    failure_count = sum(
        1 for record in non_success if record.termination_reason == TERMINATION_FAILURE
    )
    error_count = sum(
        1 for record in non_success if record.termination_reason == TERMINATION_ERROR
    )

    lines.extend(
        [
            "",
            "## Non-success termination counts",
            "",
            f"- time_limit: {timeout_count}",
            f"- expansion_limit: {expansion_count}",
            f"- failure: {failure_count}",
            f"- error: {error_count}",
            "",
            "## Runtime observations",
            "",
            "- Successful solve-time statistics exclude censored timeout and expansion-limit runs.",
            "- Observed execution-time summaries include all completed runs regardless of termination.",
            "",
            "## Solution quality (paired common-success cases)",
            "",
        ]
    )

    for comparison, label in (
        ("pp_minus_basic", "PP vs Basic CBS"),
        ("pp_minus_cardinal", "PP vs Cardinal-First CBS"),
        ("basic_minus_cardinal", "Basic CBS vs Cardinal-First CBS"),
    ):
        filtered = [row for row in paired_rows if row["comparison"] == comparison]
        if not filtered:
            lines.append(f"- {label}: no common-success pairs")
            continue
        soc_column = {
            "pp_minus_basic": "pp_minus_basic_soc",
            "pp_minus_cardinal": "pp_minus_cardinal_soc",
            "basic_minus_cardinal": "basic_minus_cardinal_soc",
        }[comparison]
        soc_diffs = [row[soc_column] for row in filtered]
        mean_diff = statistics.mean(soc_diffs)
        lines.append(
            f"- {label}: {len(filtered)} paired instances; "
            f"mean SoC diff = {mean_diff:.2f} "
            f"(positive means right-hand algorithm lower SoC)"
        )

    if cardinal_instance_rows:
        runtime_ratios = [
            row["runtime_ratio_cardinal_over_basic"]
            for row in cardinal_instance_rows
            if row["runtime_ratio_cardinal_over_basic"] != ""
        ]
        ct_reductions = [
            row["ct_expansion_reduction_basic_minus_cardinal"]
            for row in cardinal_instance_rows
        ]
        additional_ll = [
            row["additional_low_level_searches_cardinal_minus_basic"]
            for row in cardinal_instance_rows
        ]
        lines.extend(
            [
                "",
                "## Basic vs Cardinal search effort",
                "",
                f"- Common-success CBS pairs: {len(cardinal_instance_rows)}",
                f"- Mean runtime ratio (Cardinal/Basic): {statistics.mean(runtime_ratios):.3f}",
                f"- Mean CT expansion reduction (Basic - Cardinal): {statistics.mean(ct_reductions):.2f}",
                f"- Mean additional combined low-level searches (Cardinal - Basic): {statistics.mean(additional_ll):.2f}",
                "",
                "Cardinal-First may reduce CT expansions while increasing low-level classification/replan work.",
            ]
        )

    pp_budget_row = next(
        row
        for row in common_budget_rows
        if row["algorithm"] == MAPFBenchmarkAlgorithm.FIXED_PRIORITY_PP.value
    )
    lines.extend(
        [
            "",
            "## Post-hoc common 180 s budget comparison (secondary)",
            "",
            f"- Fixed-Priority PP observed success: {pp_budget_row['observed_success_count']}/27",
            f"- Fixed-Priority PP within 180 s post-hoc success: "
            f"{pp_budget_row['within_common_budget_success_count']}/27",
            "- CBS algorithms were already executed under the 180 s wall-clock guard.",
            "- Original PP termination labels were not modified for this comparison.",
            "",
            "## Caveats",
            "",
            "- Interaction level (LOW/MEDIUM/HIGH) reflects independent-path conflict count, not absolute computational difficulty.",
            "- HIGH instances show substantial internal variation in runtime and CBS outcomes.",
            "- This summary is empirical and descriptive; final thesis conclusions are out of scope for MAPF-5C.1.",
        ]
    )

    return "\n".join(lines) + "\n"


def run_benchmark_analysis(config: AnalysisConfig) -> AnalysisResult:
    records = load_analysis_records(config.results_jsonl_path)
    manifest = load_benchmark_manifest(config.manifest_path)
    validation = validate_dataset(
        records,
        manifest,
        expected_record_count=config.expected_record_count,
        expected_instance_count=config.expected_instance_count,
    )

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_path = output_dir / "dataset_validation.txt"
    validation_path.write_text(_format_validation_report(validation), encoding="utf-8")

    if not validation.passed:
        return AnalysisResult(
            validation=validation,
            output_dir=output_dir,
            tables_written=("dataset_validation.txt",),
        )

    paired_rows = build_paired_quality_comparison(records)
    cardinal_instance_rows, cardinal_overhead_rows = build_cardinal_overhead_summary(records)
    common_budget_rows = build_common_180s_budget_summary(
        records,
        budget_ms=config.common_budget_ms,
    )

    table_builders: list[tuple[str, Sequence[str], Callable[[], list[dict[str, Any]]]]] = [
        (
            "termination_summary.csv",
            ["algorithm", "termination_reason", "count"],
            lambda: build_termination_summary(records),
        ),
        (
            "success_rate_by_group.csv",
            [
                "algorithm",
                "agent_count",
                "interaction_level",
                "total_runs",
                "success_count",
                "success_rate",
            ],
            lambda: build_success_rate_by_group(records),
        ),
        (
            "runtime_success_summary.csv",
            [
                "algorithm",
                "agent_count",
                "interaction_level",
                "metric",
                "n",
                "mean_ms",
                "median_ms",
                "min_ms",
                "max_ms",
                "std_ms",
            ],
            lambda: build_runtime_success_summary(records),
        ),
        (
            "runtime_all_observed_summary.csv",
            [
                "algorithm",
                "agent_count",
                "interaction_level",
                "metric",
                "n",
                "mean_ms",
                "median_ms",
                "min_ms",
                "max_ms",
                "std_ms",
            ],
            lambda: build_runtime_all_observed_summary(records),
        ),
        (
            "solution_quality_summary.csv",
            [
                "algorithm",
                "agent_count",
                "interaction_level",
                "n",
                "mean_soc",
                "median_soc",
                "min_soc",
                "max_soc",
                "std_soc",
                "mean_makespan",
                "median_makespan",
                "min_makespan",
                "max_makespan",
                "std_makespan",
            ],
            lambda: build_solution_quality_summary(records),
        ),
        (
            "paired_quality_comparison.csv",
            [
                "comparison",
                "instance_id",
                "agent_count",
                "interaction_level",
                "independent_conflict_count",
                "left_algorithm",
                "right_algorithm",
                "left_soc",
                "right_soc",
                "left_makespan",
                "right_makespan",
                "pp_minus_basic_soc",
                "pp_minus_basic_makespan",
                "pp_minus_cardinal_soc",
                "pp_minus_cardinal_makespan",
                "basic_minus_cardinal_soc",
                "basic_minus_cardinal_makespan",
            ],
            lambda: paired_rows,
        ),
        (
            "pp_vs_basic_quality_summary.csv",
            [
                "comparison",
                "paired_instance_count",
                "mean_soc_diff",
                "median_soc_diff",
                "std_soc_diff",
                "mean_makespan_diff",
                "median_makespan_diff",
                "std_makespan_diff",
                "left_better_soc_count",
                "right_better_soc_count",
                "equal_soc_count",
                "soc_diff_column",
                "makespan_diff_column",
                "soc_diff_interpretation",
            ],
            lambda: build_pp_vs_basic_quality_summary(paired_rows),
        ),
        (
            "pp_vs_cardinal_quality_summary.csv",
            [
                "comparison",
                "paired_instance_count",
                "mean_soc_diff",
                "median_soc_diff",
                "std_soc_diff",
                "mean_makespan_diff",
                "median_makespan_diff",
                "std_makespan_diff",
                "left_better_soc_count",
                "right_better_soc_count",
                "equal_soc_count",
                "soc_diff_column",
                "makespan_diff_column",
                "soc_diff_interpretation",
            ],
            lambda: build_pp_vs_cardinal_quality_summary(paired_rows),
        ),
        (
            "basic_vs_cardinal_quality_summary.csv",
            [
                "comparison",
                "paired_instance_count",
                "mean_soc_diff",
                "median_soc_diff",
                "std_soc_diff",
                "mean_makespan_diff",
                "median_makespan_diff",
                "std_makespan_diff",
                "left_better_soc_count",
                "right_better_soc_count",
                "equal_soc_count",
                "soc_diff_column",
                "makespan_diff_column",
                "soc_diff_interpretation",
            ],
            lambda: build_basic_vs_cardinal_quality_summary(paired_rows),
        ),
        (
            "cbs_search_effort_summary.csv",
            [
                "algorithm",
                "agent_count",
                "interaction_level",
                "n",
                "mean_expanded_ct_nodes",
                "median_expanded_ct_nodes",
                "min_expanded_ct_nodes",
                "max_expanded_ct_nodes",
                "std_expanded_ct_nodes",
                "mean_generated_ct_nodes",
                "median_generated_ct_nodes",
                "min_generated_ct_nodes",
                "max_generated_ct_nodes",
                "std_generated_ct_nodes",
                "mean_low_level_replans",
                "median_low_level_replans",
                "min_low_level_replans",
                "max_low_level_replans",
                "std_low_level_replans",
                "mean_max_open_size",
                "median_max_open_size",
                "min_max_open_size",
                "max_max_open_size",
                "std_max_open_size",
                "mean_classified_conflicts",
                "median_classified_conflicts",
                "min_classified_conflicts",
                "max_classified_conflicts",
                "std_classified_conflicts",
                "mean_classification_low_level_searches",
                "median_classification_low_level_searches",
                "min_classification_low_level_searches",
                "max_classification_low_level_searches",
                "std_classification_low_level_searches",
                "mean_selected_cardinal_conflicts",
                "median_selected_cardinal_conflicts",
                "min_selected_cardinal_conflicts",
                "max_selected_cardinal_conflicts",
                "std_selected_cardinal_conflicts",
                "mean_selected_semi_cardinal_conflicts",
                "median_selected_semi_cardinal_conflicts",
                "min_selected_semi_cardinal_conflicts",
                "max_selected_semi_cardinal_conflicts",
                "std_selected_semi_cardinal_conflicts",
                "mean_selected_non_cardinal_conflicts",
                "median_selected_non_cardinal_conflicts",
                "min_selected_non_cardinal_conflicts",
                "max_selected_non_cardinal_conflicts",
                "std_selected_non_cardinal_conflicts",
                "mean_combined_low_level_searches",
                "median_combined_low_level_searches",
                "min_combined_low_level_searches",
                "max_combined_low_level_searches",
                "std_combined_low_level_searches",
            ],
            lambda: build_cbs_search_effort_summary(records),
        ),
        (
            "cardinal_overhead_summary.csv",
            [
                "instance_id",
                "agent_count",
                "interaction_level",
                "independent_conflict_count",
                "basic_runtime_ms",
                "cardinal_runtime_ms",
                "runtime_ratio_cardinal_over_basic",
                "basic_expanded_ct_nodes",
                "cardinal_expanded_ct_nodes",
                "ct_expansion_reduction_basic_minus_cardinal",
                "basic_combined_low_level_searches",
                "cardinal_low_level_replans",
                "cardinal_classification_low_level_searches",
                "cardinal_combined_low_level_searches",
                "additional_low_level_searches_cardinal_minus_basic",
                "cardinal_classified_conflicts",
                "cardinal_selected_cardinal_conflicts",
                "cardinal_selected_semi_cardinal_conflicts",
                "cardinal_selected_non_cardinal_conflicts",
                "summary_level",
                "paired_instance_count",
                "mean_runtime_ratio",
                "mean_ct_reduction",
                "mean_additional_low_level_searches",
                "median_runtime_ratio",
            ],
            lambda: cardinal_overhead_rows,
        ),
        (
            "common_180s_budget_summary.csv",
            [
                "algorithm",
                "total_runs",
                "observed_success_count",
                "observed_success_rate",
                "within_common_budget_success_count",
                "within_common_budget_success_rate",
                "common_budget_ms",
                "comparison_scope",
                "methodology_note",
            ],
            lambda: common_budget_rows,
        ),
        (
            "non_success_runs.csv",
            [
                "instance_id",
                "algorithm",
                "agent_count",
                "interaction_level",
                "independent_conflict_count",
                "termination_reason",
                "execution_time_ms",
                "error_message",
                "cbs_expanded_ct_nodes",
                "cbs_generated_ct_nodes",
                "cbs_low_level_replans",
                "cbs_combined_low_level_searches",
            ],
            lambda: build_non_success_runs(records),
        ),
        (
            "high_interaction_outcomes.csv",
            [
                "instance_id",
                "agent_count",
                "interaction_level",
                "independent_conflict_count",
                "independent_soc",
                "independent_makespan",
                "algorithm",
                "termination_reason",
                "success",
                "execution_time_ms",
                "soc",
                "makespan",
                "cbs_expanded_ct_nodes",
                "cbs_generated_ct_nodes",
                "cbs_low_level_replans",
                "cbs_combined_low_level_searches",
                "cbs_classified_conflicts",
                "cbs_classification_low_level_searches",
                "cbs_selected_cardinal_conflicts",
                "cbs_selected_semi_cardinal_conflicts",
                "cbs_selected_non_cardinal_conflicts",
            ],
            lambda: build_high_interaction_outcomes(records),
        ),
        (
            "instance_level_analysis.csv",
            [
                "instance_id",
                "agent_count",
                "interaction_level",
                "independent_conflict_count",
                "independent_soc",
                "independent_makespan",
                "fixed_priority_pp_termination_reason",
                "fixed_priority_pp_success",
                "fixed_priority_pp_execution_time_ms",
                "fixed_priority_pp_soc",
                "fixed_priority_pp_makespan",
                "fixed_priority_pp_within_180s",
                "cbs_basic_termination_reason",
                "cbs_basic_success",
                "cbs_basic_execution_time_ms",
                "cbs_basic_soc",
                "cbs_basic_makespan",
                "cbs_basic_within_180s",
                "cbs_basic_expanded_ct_nodes",
                "cbs_basic_combined_low_level_searches",
                "cbs_cardinal_first_termination_reason",
                "cbs_cardinal_first_success",
                "cbs_cardinal_first_execution_time_ms",
                "cbs_cardinal_first_soc",
                "cbs_cardinal_first_makespan",
                "cbs_cardinal_first_within_180s",
                "cbs_cardinal_first_expanded_ct_nodes",
                "cbs_cardinal_first_combined_low_level_searches",
            ],
            lambda: build_instance_level_analysis(records),
        ),
    ]

    written: list[str] = ["dataset_validation.txt"]
    for filename, fieldnames, builder in table_builders:
        write_csv(output_dir / filename, builder(), fieldnames)
        written.append(filename)

    summary_md = build_analysis_summary_markdown(
        records,
        validation,
        budget_ms=config.common_budget_ms,
        paired_rows=paired_rows,
        cardinal_instance_rows=cardinal_instance_rows,
        common_budget_rows=common_budget_rows,
    )
    summary_path = output_dir / "analysis_summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")
    written.append("analysis_summary.md")

    return AnalysisResult(
        validation=validation,
        output_dir=output_dir,
        tables_written=tuple(written),
    )
