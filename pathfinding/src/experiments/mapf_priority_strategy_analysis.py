from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pathfinding.src.experiments.mapf_benchmark_analysis import count_directional_soc_better
from pathfinding.src.experiments.mapf_benchmark_execution import TERMINATION_SUCCESS
from pathfinding.src.experiments.mapf_benchmark_instances import (
    MAPFBenchmarkManifest,
    load_benchmark_manifest,
)

AGENT_COUNT_ORDER: tuple[int, ...] = (5, 10, 20)
INTERACTION_ORDER: tuple[str, ...] = ("low", "medium", "high")
RANDOM_K_DEFAULT = 10

STRATEGY_FIXED = "fixed_priority_pp"
STRATEGY_SPF = "spf"
STRATEGY_LPF = "lpf"
STRATEGY_CBS_BASIC = "cbs_basic"

OUTPUT_TABLES: tuple[str, ...] = (
    "instance_level_summary.csv",
    "random_sensitivity.csv",
    "paired_quality_summary.csv",
    "runtime_summary.csv",
    "cbs_quality_reference.csv",
    "analysis_summary.md",
    "thesis_tables.md",
)

FIGURE_BASENAMES: tuple[str, ...] = (
    "fig01_priority_sensitivity_by_instance",
    "fig02_spf_vs_lpf_soc_difference",
    "fig03_spf_lpf_runtime_cost",
    "fig04_priority_sensitive_case_study",
)


@dataclass(frozen=True, slots=True)
class PriorityStrategyAnalysisConfig:
    fixed_results_csv: Path
    random_results_csv: Path
    strategy_results_csv: Path
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
class PriorityStrategyAnalysisResult:
    validation: ValidationReport
    output_dir: Path
    tables_written: tuple[str, ...]
    plots_written: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FixedRunRecord:
    instance_id: str
    agent_count: int
    interaction_level: str
    algorithm: str
    success: bool
    termination_reason: str
    execution_time_ms: float
    soc: int | None
    makespan: int | None


@dataclass(frozen=True, slots=True)
class RandomRunRecord:
    instance_id: str
    agent_count: int
    interaction_level: str
    ordering_index: int
    ordering_seed: int
    agent_order: tuple[int, ...]
    success: bool
    termination_reason: str
    execution_time_ms: float
    soc: int | None
    makespan: int | None


@dataclass(frozen=True, slots=True)
class StrategyRunRecord:
    instance_id: str
    agent_count: int
    interaction_level: str
    strategy: str
    agent_order: tuple[int, ...] | None
    success: bool
    termination_reason: str
    ordering_time_ms: float
    pp_time_ms: float | None
    total_time_ms: float
    soc: int | None
    makespan: int | None


@dataclass(frozen=True, slots=True)
class RandomInstanceAggregate:
    instance_id: str
    agent_count: int
    interaction_level: str
    ordering_count: int
    success_count: int
    success_rate: float
    soc_mean: float | None
    soc_median: float | None
    soc_min: int | None
    soc_max: int | None
    soc_range: int | None
    unique_soc_count: int | None
    makespan_mean: float | None
    makespan_median: float | None
    makespan_min: int | None
    makespan_max: int | None
    makespan_range: int | None
    runtime_mean_ms: float | None
    runtime_median_ms: float | None
    runtime_min_ms: float | None
    runtime_max_ms: float | None
    sampled_best_of_k_soc: int | None
    observed_priority_sensitive: bool


def _row_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def _row_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"true", "1", "yes"}


def _parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _parse_agent_order(value: str | None) -> tuple[int, ...] | None:
    if value is None or value == "":
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("agent_order must be a JSON list")
    return tuple(int(item) for item in parsed)


def _interaction_sort_key(level: str) -> int:
    normalized = level.lower()
    if normalized in INTERACTION_ORDER:
        return INTERACTION_ORDER.index(normalized)
    return len(INTERACTION_ORDER)


def _instance_sort_key(
    instance_id: str,
    agent_count: int,
    interaction_level: str,
) -> tuple[int, int, str]:
    return (
        AGENT_COUNT_ORDER.index(agent_count)
        if agent_count in AGENT_COUNT_ORDER
        else agent_count,
        _interaction_sort_key(interaction_level),
        instance_id,
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _summary_stats(values: Sequence[float | int]) -> dict[str, float | int | str]:
    if not values:
        return {"count": 0, "mean": "", "median": "", "min": "", "max": ""}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def load_fixed_results(path: Path) -> tuple[FixedRunRecord, ...]:
    rows = _read_csv_rows(path)
    records: list[FixedRunRecord] = []
    for row in rows:
        records.append(
            FixedRunRecord(
                instance_id=row["instance_id"],
                agent_count=int(row["agent_count"]),
                interaction_level=row["interaction_level"].lower(),
                algorithm=row["algorithm"],
                success=_parse_bool(row.get("success")),
                termination_reason=row["termination_reason"],
                execution_time_ms=float(row["execution_time_ms"]),
                soc=_parse_int(row.get("soc")),
                makespan=_parse_int(row.get("makespan")),
            )
        )
    return tuple(records)


def load_random_results(path: Path) -> tuple[RandomRunRecord, ...]:
    rows = _read_csv_rows(path)
    records: list[RandomRunRecord] = []
    for row in rows:
        records.append(
            RandomRunRecord(
                instance_id=row["instance_id"],
                agent_count=int(row["agent_count"]),
                interaction_level=row["interaction_level"].lower(),
                ordering_index=int(row["ordering_index"]),
                ordering_seed=int(row["ordering_seed"]),
                agent_order=_parse_agent_order(row.get("agent_order")) or (),
                success=_parse_bool(row.get("success")),
                termination_reason=row["termination_reason"],
                execution_time_ms=float(row["execution_time_ms"]),
                soc=_parse_int(row.get("soc")),
                makespan=_parse_int(row.get("makespan")),
            )
        )
    return tuple(records)


def load_strategy_results(path: Path) -> tuple[StrategyRunRecord, ...]:
    rows = _read_csv_rows(path)
    records: list[StrategyRunRecord] = []
    for row in rows:
        agent_order = _parse_agent_order(row.get("agent_order"))
        records.append(
            StrategyRunRecord(
                instance_id=row["instance_id"],
                agent_count=int(row["agent_count"]),
                interaction_level=row["interaction_level"].lower(),
                strategy=row["strategy"].lower(),
                agent_order=agent_order,
                success=_parse_bool(row.get("success")),
                termination_reason=row["termination_reason"],
                ordering_time_ms=float(row["ordering_time_ms"]),
                pp_time_ms=_parse_float(row.get("pp_time_ms")),
                total_time_ms=float(row["total_time_ms"]),
                soc=_parse_int(row.get("soc")),
                makespan=_parse_int(row.get("makespan")),
            )
        )
    return tuple(records)


def aggregate_random_by_instance(
    random_records: Sequence[RandomRunRecord],
) -> dict[str, RandomInstanceAggregate]:
    grouped: dict[str, list[RandomRunRecord]] = defaultdict(list)
    for record in random_records:
        grouped[record.instance_id].append(record)

    aggregates: dict[str, RandomInstanceAggregate] = {}
    for instance_id, records in grouped.items():
        records = sorted(records, key=lambda item: item.ordering_index)
        success_records = [record for record in records if record.success]
        success_count = len(success_records)
        ordering_count = len(records)
        success_rate = success_count / ordering_count if ordering_count else 0.0

        successful_socs = [
            record.soc for record in success_records if record.soc is not None
        ]
        successful_makespans = [
            record.makespan
            for record in success_records
            if record.makespan is not None
        ]
        successful_runtimes = [
            record.execution_time_ms for record in success_records
        ]

        unique_soc_count = len(set(successful_socs)) if successful_socs else None
        soc_range = (
            max(successful_socs) - min(successful_socs)
            if successful_socs
            else None
        )
        makespan_range = (
            max(successful_makespans) - min(successful_makespans)
            if successful_makespans
            else None
        )

        aggregates[instance_id] = RandomInstanceAggregate(
            instance_id=instance_id,
            agent_count=records[0].agent_count,
            interaction_level=records[0].interaction_level,
            ordering_count=ordering_count,
            success_count=success_count,
            success_rate=success_rate,
            soc_mean=statistics.mean(successful_socs) if successful_socs else None,
            soc_median=statistics.median(successful_socs) if successful_socs else None,
            soc_min=min(successful_socs) if successful_socs else None,
            soc_max=max(successful_socs) if successful_socs else None,
            soc_range=soc_range,
            unique_soc_count=unique_soc_count,
            makespan_mean=(
                statistics.mean(successful_makespans)
                if successful_makespans
                else None
            ),
            makespan_median=(
                statistics.median(successful_makespans)
                if successful_makespans
                else None
            ),
            makespan_min=min(successful_makespans) if successful_makespans else None,
            makespan_max=max(successful_makespans) if successful_makespans else None,
            makespan_range=makespan_range,
            runtime_mean_ms=(
                statistics.mean(successful_runtimes) if successful_runtimes else None
            ),
            runtime_median_ms=(
                statistics.median(successful_runtimes)
                if successful_runtimes
                else None
            ),
            runtime_min_ms=min(successful_runtimes) if successful_runtimes else None,
            runtime_max_ms=max(successful_runtimes) if successful_runtimes else None,
            sampled_best_of_k_soc=min(successful_socs) if successful_socs else None,
            observed_priority_sensitive=bool(unique_soc_count and unique_soc_count > 1),
        )

    return aggregates


def validate_datasets(
    *,
    manifest: MAPFBenchmarkManifest,
    fixed_records: Sequence[FixedRunRecord],
    random_records: Sequence[RandomRunRecord],
    strategy_records: Sequence[StrategyRunRecord],
    config: PriorityStrategyAnalysisConfig,
) -> ValidationReport:
    report = ValidationReport(passed=True)
    manifest_ids = {instance.instance_id for instance in manifest.instances}
    expected_random_total = config.expected_instance_count * config.expected_random_k

    fixed_pp = [record for record in fixed_records if record.algorithm == STRATEGY_FIXED]
    report.add(
        f"Fixed-Priority PP records: expected {config.expected_instance_count}, "
        f"found {len(fixed_pp)}",
        passed=len(fixed_pp) == config.expected_instance_count,
    )

    fixed_ids = {record.instance_id for record in fixed_pp}
    report.add(
        "Fixed-Priority instance IDs match manifest",
        passed=fixed_ids == manifest_ids,
    )

    report.add(
        f"Random Priority records: expected {expected_random_total}, "
        f"found {len(random_records)}",
        passed=len(random_records) == expected_random_total,
    )

    random_keys = {
        (record.instance_id, record.ordering_index) for record in random_records
    }
    report.add(
        "Random Priority has no duplicate (instance_id, ordering_index)",
        passed=len(random_keys) == len(random_records),
    )

    random_by_instance: dict[str, set[int]] = defaultdict(set)
    random_orders_by_instance: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    for record in random_records:
        random_by_instance[record.instance_id].add(record.ordering_index)
        if record.agent_order:
            random_orders_by_instance[record.instance_id].add(record.agent_order)

    expected_indices = set(range(config.expected_random_k))
    random_instance_ok = all(
        indices == expected_indices for indices in random_by_instance.values()
    )
    report.add(
        f"Random Priority has exactly K={config.expected_random_k} ordering indices "
        f"per instance",
        passed=random_instance_ok and len(random_by_instance) == config.expected_instance_count,
    )

    unique_orders_ok = all(
        len(orders) == config.expected_random_k
        for orders in random_orders_by_instance.values()
    )
    report.add(
        "Random Priority agent orders are unique within each instance",
        passed=unique_orders_ok,
    )

    random_ids = {record.instance_id for record in random_records}
    report.add(
        "Random Priority instance IDs match manifest",
        passed=random_ids == manifest_ids,
    )

    report.add(
        f"SPF/LPF records: expected {config.expected_instance_count * 2}, "
        f"found {len(strategy_records)}",
        passed=len(strategy_records) == config.expected_instance_count * 2,
    )

    strategy_keys = {
        (record.instance_id, record.strategy) for record in strategy_records
    }
    report.add(
        "SPF/LPF has no duplicate (instance_id, strategy)",
        passed=len(strategy_keys) == len(strategy_records),
    )

    strategy_ids = {record.instance_id for record in strategy_records}
    report.add(
        "SPF/LPF instance IDs match manifest",
        passed=strategy_ids == manifest_ids,
    )

    for instance_id in manifest_ids:
        strategies = {
            record.strategy
            for record in strategy_records
            if record.instance_id == instance_id
        }
        if strategies != {STRATEGY_SPF, STRATEGY_LPF}:
            report.add(
                f"Instance {instance_id} missing SPF or LPF strategy",
                passed=False,
            )
            break
    else:
        report.add(
            "Each instance has exactly one SPF and one LPF record",
            passed=True,
        )

    cross_ok = fixed_ids == random_ids == strategy_ids == manifest_ids
    report.add(
        "Cross-dataset instance IDs are identical",
        passed=cross_ok,
    )

    return report


def _fixed_by_instance(
    fixed_records: Sequence[FixedRunRecord],
) -> dict[str, FixedRunRecord]:
    return {
        record.instance_id: record
        for record in fixed_records
        if record.algorithm == STRATEGY_FIXED
    }


def _strategy_by_instance(
    strategy_records: Sequence[StrategyRunRecord],
    strategy: str,
) -> dict[str, StrategyRunRecord]:
    return {
        record.instance_id: record
        for record in strategy_records
        if record.strategy == strategy
    }


def _cbs_by_instance(
    fixed_records: Sequence[FixedRunRecord],
) -> dict[str, FixedRunRecord]:
    return {
        record.instance_id: record
        for record in fixed_records
        if record.algorithm == STRATEGY_CBS_BASIC and record.success
    }


def build_instance_level_summary(
    *,
    manifest: MAPFBenchmarkManifest,
    fixed_records: Sequence[FixedRunRecord],
    random_aggregates: dict[str, RandomInstanceAggregate],
    strategy_records: Sequence[StrategyRunRecord],
) -> list[dict[str, Any]]:
    fixed_lookup = _fixed_by_instance(fixed_records)
    spf_lookup = _strategy_by_instance(strategy_records, STRATEGY_SPF)
    lpf_lookup = _strategy_by_instance(strategy_records, STRATEGY_LPF)

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

        rows.append(
            {
                "instance_id": instance_id,
                "agent_count": instance.agent_count,
                "interaction_level": instance.interaction_level.value,
                "fixed_success": fixed.success,
                "fixed_soc": fixed.soc,
                "fixed_makespan": fixed.makespan,
                "fixed_runtime_ms": fixed.execution_time_ms,
                "spf_success": spf.success,
                "spf_soc": spf.soc,
                "spf_makespan": spf.makespan,
                "spf_agent_order": json.dumps(list(spf.agent_order or ())),
                "spf_ordering_time_ms": spf.ordering_time_ms,
                "spf_pp_time_ms": spf.pp_time_ms,
                "spf_total_time_ms": spf.total_time_ms,
                "lpf_success": lpf.success,
                "lpf_soc": lpf.soc,
                "lpf_makespan": lpf.makespan,
                "lpf_agent_order": json.dumps(list(lpf.agent_order or ())),
                "lpf_ordering_time_ms": lpf.ordering_time_ms,
                "lpf_pp_time_ms": lpf.pp_time_ms,
                "lpf_total_time_ms": lpf.total_time_ms,
                "random_success_rate": random.success_rate,
                "random_success_count": random.success_count,
                "random_ordering_count": random.ordering_count,
                "random_soc_mean": random.soc_mean,
                "random_soc_median": random.soc_median,
                "random_soc_min": random.soc_min,
                "random_soc_max": random.soc_max,
                "random_soc_range": random.soc_range,
                "random_unique_soc_count": random.unique_soc_count,
                "random_makespan_mean": random.makespan_mean,
                "random_makespan_median": random.makespan_median,
                "random_makespan_min": random.makespan_min,
                "random_makespan_max": random.makespan_max,
                "random_makespan_range": random.makespan_range,
                "random_runtime_mean_ms": random.runtime_mean_ms,
                "random_runtime_median_ms": random.runtime_median_ms,
                "random_sampled_best_of_k_soc": random.sampled_best_of_k_soc,
                "observed_priority_sensitive": random.observed_priority_sensitive,
            }
        )
    return rows


def build_random_sensitivity(
    instance_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in instance_rows:
        fixed_soc = row["fixed_soc"]
        spf_soc = row["spf_soc"]
        lpf_soc = row["lpf_soc"]
        rows.append(
            {
                "instance_id": row["instance_id"],
                "agent_count": row["agent_count"],
                "interaction_level": row["interaction_level"],
                "observed_priority_sensitive": row["observed_priority_sensitive"],
                "random_unique_soc_count": row["random_unique_soc_count"],
                "random_soc_range": row["random_soc_range"],
                "random_soc_min": row["random_soc_min"],
                "random_soc_max": row["random_soc_max"],
                "random_soc_median": row["random_soc_median"],
                "random_makespan_range": row["random_makespan_range"],
                "spf_differs_from_fixed": (
                    fixed_soc is not None
                    and spf_soc is not None
                    and fixed_soc != spf_soc
                ),
                "lpf_differs_from_fixed": (
                    fixed_soc is not None
                    and lpf_soc is not None
                    and fixed_soc != lpf_soc
                ),
                "spf_differs_from_lpf": (
                    spf_soc is not None
                    and lpf_soc is not None
                    and spf_soc != lpf_soc
                ),
                "soc_changes_but_makespan_constant": (
                    row["observed_priority_sensitive"]
                    and row["random_makespan_range"] == 0
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
    makespan_diffs = [row["makespan_diff"] for row in paired_rows]
    directional = count_directional_soc_better(soc_diffs)
    soc_stats = _summary_stats(soc_diffs)
    makespan_stats = _summary_stats(makespan_diffs)
    max_abs_soc = max((abs(value) for value in soc_diffs), default=0)

    return {
        "comparison": comparison,
        "left_strategy": left_label,
        "right_strategy": right_label,
        "common_success_count": len(paired_rows),
        "mean_soc_diff": soc_stats["mean"],
        "median_soc_diff": soc_stats["median"],
        "max_abs_soc_diff": max_abs_soc,
        "left_better_count": directional.left_better,
        "equal_count": directional.equal,
        "right_better_count": directional.right_better,
        "mean_makespan_diff": makespan_stats["mean"],
        "median_makespan_diff": makespan_stats["median"],
        "soc_diff_definition": "left_soc - right_soc (positive => right better)",
        "makespan_diff_definition": "left_makespan - right_makespan (positive => right better)",
    }


def build_paired_instance_rows(
    instance_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comparisons = (
        ("fixed_vs_spf", "fixed_soc", "spf_soc", "fixed_makespan", "spf_makespan", "fixed_success", "spf_success"),
        ("fixed_vs_lpf", "fixed_soc", "lpf_soc", "fixed_makespan", "lpf_makespan", "fixed_success", "lpf_success"),
        ("spf_vs_lpf", "spf_soc", "lpf_soc", "spf_makespan", "lpf_makespan", "spf_success", "lpf_success"),
    )
    for row in instance_rows:
        for (
            comparison,
            left_soc_key,
            right_soc_key,
            left_ms_key,
            right_ms_key,
            left_success_key,
            right_success_key,
        ) in comparisons:
            if not _row_bool(row[left_success_key]) or not _row_bool(row[right_success_key]):
                continue
            left_soc = _row_int(row[left_soc_key])
            right_soc = _row_int(row[right_soc_key])
            left_makespan = _row_int(row[left_ms_key])
            right_makespan = _row_int(row[right_ms_key])
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
                    "instance_id": row["instance_id"],
                    "agent_count": row["agent_count"],
                    "interaction_level": row["interaction_level"],
                    "left_soc": left_soc,
                    "right_soc": right_soc,
                    "soc_diff": left_soc - right_soc,
                    "left_makespan": left_makespan,
                    "right_makespan": right_makespan,
                    "makespan_diff": left_makespan - right_makespan,
                }
            )
    return rows


def build_paired_quality_summary(
    instance_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    paired_rows = build_paired_instance_rows(instance_rows)
    summaries: list[dict[str, Any]] = []

    for comparison, left_label, right_label in (
        ("fixed_vs_spf", "fixed", "spf"),
        ("fixed_vs_lpf", "fixed", "lpf"),
        ("spf_vs_lpf", "spf", "lpf"),
    ):
        filtered = [row for row in paired_rows if row["comparison"] == comparison]
        summaries.append(
            _paired_soc_summary(
                comparison=comparison,
                left_label=left_label,
                right_label=right_label,
                paired_rows=filtered,
            )
        )
    return summaries


def _compare_deterministic_to_random(
    *,
    strategy_label: str,
    deterministic_soc: int | None,
    random_records: Sequence[RandomRunRecord],
) -> dict[str, Any]:
    successful = [
        record for record in random_records if record.success and record.soc is not None
    ]
    if deterministic_soc is None or not successful:
        return {
            "strategy": strategy_label,
            "deterministic_soc": deterministic_soc,
            "random_runs_lower_soc_count": "",
            "random_runs_equal_soc_count": "",
            "random_runs_higher_soc_count": "",
            "deterministic_vs_random_mean_soc_diff": "",
            "deterministic_vs_random_median_soc_diff": "",
            "deterministic_vs_random_sampled_best_of_k_soc_diff": "",
        }

    random_socs = [record.soc for record in successful if record.soc is not None]
    random_mean = statistics.mean(random_socs)
    random_median = statistics.median(random_socs)
    random_min = min(random_socs)

    return {
        "strategy": strategy_label,
        "deterministic_soc": deterministic_soc,
        "random_runs_lower_soc_count": sum(
            1 for soc in random_socs if soc < deterministic_soc
        ),
        "random_runs_equal_soc_count": sum(
            1 for soc in random_socs if soc == deterministic_soc
        ),
        "random_runs_higher_soc_count": sum(
            1 for soc in random_socs if soc > deterministic_soc
        ),
        "deterministic_vs_random_mean_soc_diff": deterministic_soc - random_mean,
        "deterministic_vs_random_median_soc_diff": deterministic_soc - random_median,
        "deterministic_vs_random_sampled_best_of_k_soc_diff": deterministic_soc - random_min,
    }


def build_deterministic_vs_random_summary(
    *,
    instance_rows: Sequence[dict[str, Any]],
    random_records: Sequence[RandomRunRecord],
) -> list[dict[str, Any]]:
    random_by_instance: dict[str, list[RandomRunRecord]] = defaultdict(list)
    for record in random_records:
        random_by_instance[record.instance_id].append(record)

    rows: list[dict[str, Any]] = []
    for row in instance_rows:
        instance_id = row["instance_id"]
        instance_random = random_by_instance[instance_id]
        for strategy_label, soc_key in (
            ("fixed", "fixed_soc"),
            ("spf", "spf_soc"),
            ("lpf", "lpf_soc"),
        ):
            comparison = _compare_deterministic_to_random(
                strategy_label=strategy_label,
                deterministic_soc=row[soc_key],
                random_records=instance_random,
            )
            comparison.update(
                {
                    "instance_id": instance_id,
                    "agent_count": row["agent_count"],
                    "interaction_level": row["interaction_level"],
                    "random_soc_mean": row["random_soc_mean"],
                    "random_soc_median": row["random_soc_median"],
                    "random_sampled_best_of_k_soc": row["random_sampled_best_of_k_soc"],
                    "sampled_best_of_k_note": (
                        "sampled best-of-K diagnostic; not the primary Random baseline"
                    ),
                }
            )
            rows.append(comparison)
    return rows


def build_runtime_summary(
    instance_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in instance_rows:
        rows.append(
            {
                "instance_id": row["instance_id"],
                "agent_count": row["agent_count"],
                "interaction_level": row["interaction_level"],
                "fixed_runtime_ms": row["fixed_runtime_ms"],
                "random_runtime_mean_ms": row["random_runtime_mean_ms"],
                "random_runtime_median_ms": row["random_runtime_median_ms"],
                "random_runtime_semantics": (
                    "execution_time_ms from Random pilot; permutation overhead not "
                    "separately timed; no fabricated ordering_time_ms"
                ),
                "spf_ordering_time_ms": row["spf_ordering_time_ms"],
                "spf_pp_time_ms": row["spf_pp_time_ms"],
                "spf_total_time_ms": row["spf_total_time_ms"],
                "spf_ordering_fraction": (
                    row["spf_ordering_time_ms"] / row["spf_total_time_ms"]
                    if row["spf_total_time_ms"]
                    else ""
                ),
                "lpf_ordering_time_ms": row["lpf_ordering_time_ms"],
                "lpf_pp_time_ms": row["lpf_pp_time_ms"],
                "lpf_total_time_ms": row["lpf_total_time_ms"],
                "lpf_ordering_fraction": (
                    row["lpf_ordering_time_ms"] / row["lpf_total_time_ms"]
                    if row["lpf_total_time_ms"]
                    else ""
                ),
                "spf_total_minus_lpf_total_ms": (
                    row["spf_total_time_ms"] - row["lpf_total_time_ms"]
                ),
                "lpf_total_over_spf_total_ratio": (
                    row["lpf_total_time_ms"] / row["spf_total_time_ms"]
                    if row["spf_total_time_ms"]
                    else ""
                ),
            }
        )

    aggregate_rows = [
        {
            "metric_group": "runtime_aggregate",
            "spf_total_faster_count": sum(
                1
                for item in rows
                if item["spf_total_time_ms"] < item["lpf_total_time_ms"]
            ),
            "lpf_total_faster_count": sum(
                1
                for item in rows
                if item["lpf_total_time_ms"] < item["spf_total_time_ms"]
            ),
            "equal_total_runtime_count": sum(
                1
                for item in rows
                if item["spf_total_time_ms"] == item["lpf_total_time_ms"]
            ),
            "mean_spf_ordering_time_ms": statistics.mean(
                item["spf_ordering_time_ms"] for item in rows
            ),
            "mean_lpf_ordering_time_ms": statistics.mean(
                item["lpf_ordering_time_ms"] for item in rows
            ),
            "mean_spf_pp_time_ms": statistics.mean(item["spf_pp_time_ms"] for item in rows),
            "mean_lpf_pp_time_ms": statistics.mean(item["lpf_pp_time_ms"] for item in rows),
            "mean_spf_total_time_ms": statistics.mean(
                item["spf_total_time_ms"] for item in rows
            ),
            "mean_lpf_total_time_ms": statistics.mean(
                item["lpf_total_time_ms"] for item in rows
            ),
            "median_spf_total_time_ms": statistics.median(
                item["spf_total_time_ms"] for item in rows
            ),
            "median_lpf_total_time_ms": statistics.median(
                item["lpf_total_time_ms"] for item in rows
            ),
            "median_lpf_over_spf_total_ratio": statistics.median(
                item["lpf_total_over_spf_total_ratio"] for item in rows
            ),
        }
    ]

    return rows + aggregate_rows


def build_cbs_quality_reference(
    *,
    instance_rows: Sequence[dict[str, Any]],
    fixed_records: Sequence[FixedRunRecord],
) -> list[dict[str, Any]]:
    cbs_lookup = _cbs_by_instance(fixed_records)
    rows: list[dict[str, Any]] = []

    for row in instance_rows:
        instance_id = row["instance_id"]
        cbs = cbs_lookup.get(instance_id)
        if cbs is None or cbs.soc is None:
            continue

        rows.append(
            {
                "instance_id": instance_id,
                "agent_count": row["agent_count"],
                "interaction_level": row["interaction_level"],
                "cbs_soc": cbs.soc,
                "cbs_makespan": cbs.makespan,
                "cbs_runtime_ms": cbs.execution_time_ms,
                "fixed_soc": row["fixed_soc"],
                "spf_soc": row["spf_soc"],
                "lpf_soc": row["lpf_soc"],
                "random_soc_median": row["random_soc_median"],
                "random_sampled_best_of_k_soc": row["random_sampled_best_of_k_soc"],
                "fixed_minus_cbs_soc": (
                    row["fixed_soc"] - cbs.soc if row["fixed_soc"] is not None else ""
                ),
                "spf_minus_cbs_soc": (
                    row["spf_soc"] - cbs.soc if row["spf_soc"] is not None else ""
                ),
                "lpf_minus_cbs_soc": (
                    row["lpf_soc"] - cbs.soc if row["lpf_soc"] is not None else ""
                ),
                "random_median_minus_cbs_soc": (
                    row["random_soc_median"] - cbs.soc
                    if row["random_soc_median"] is not None
                    else ""
                ),
                "random_sampled_best_of_k_minus_cbs_soc": (
                    row["random_sampled_best_of_k_soc"] - cbs.soc
                    if row["random_sampled_best_of_k_soc"] is not None
                    else ""
                ),
                "reference_note": (
                    "Basic CBS successful runs only; quality reference, not a "
                    "scalable mandatory baseline"
                ),
                "sampled_best_of_k_note": (
                    "Random minimum is a sampled best-of-K diagnostic only"
                ),
            }
        )
    return rows


def _format_float(value: Any, *, digits: int = 2) -> str:
    if value == "" or value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def build_analysis_summary_md(
    *,
    config: PriorityStrategyAnalysisConfig,
    validation: ValidationReport,
    instance_rows: Sequence[dict[str, Any]],
    sensitivity_rows: Sequence[dict[str, Any]],
    paired_summary: Sequence[dict[str, Any]],
    runtime_rows: Sequence[dict[str, Any]],
    cbs_rows: Sequence[dict[str, Any]],
) -> str:
    sensitive_count = sum(
        1 for row in sensitivity_rows if row["observed_priority_sensitive"]
    )
    soc_change_makespan_constant = sum(
        1 for row in sensitivity_rows if row["soc_changes_but_makespan_constant"]
    )
    spf_fixed_diff = sum(
        1 for row in sensitivity_rows if row["spf_differs_from_fixed"]
    )
    lpf_fixed_diff = sum(
        1 for row in sensitivity_rows if row["lpf_differs_from_fixed"]
    )
    spf_lpf_diff = sum(
        1 for row in sensitivity_rows if row["spf_differs_from_lpf"]
    )

    runtime_instance_rows = [
        row for row in runtime_rows if row.get("instance_id") is not None
    ]
    aggregate = next(
        (row for row in runtime_rows if row.get("metric_group") == "runtime_aggregate"),
        {},
    )

    lines = [
        "# MAPF-6 Priority Strategy Analysis Summary",
        "",
        "## Dataset validation",
        "",
        f"- Validation status: {'PASSED' if validation.passed else 'FAILED'}",
        f"- Manifest: `{config.manifest_path}`",
        f"- Fixed / CBS CSV: `{config.fixed_results_csv}`",
        f"- Random K={config.expected_random_k} CSV: `{config.random_results_csv}`",
        f"- SPF / LPF CSV: `{config.strategy_results_csv}`",
        "",
        "## Methodological notes",
        "",
        f"- Random Priority baseline uses K={config.expected_random_k} sampled orderings per instance.",
        "- K=20 is deferred as an optional later extension; this analysis infrastructure accepts additional Random rows automatically.",
        "- The MAPF instance is the unit of analysis; Random runs are aggregated per instance before cross-instance summaries.",
        "- Random minimum SoC is labeled as a sampled best-of-K diagnostic, not a fair single-run baseline.",
        "- Random runtime uses measured PP execution time only; permutation overhead was not separately instrumented.",
        "- SPF/LPF runtime uses `total_time_ms = ordering_time_ms + pp_time_ms`, including independent Space-Time A* ordering cost.",
        "- LOW/MEDIUM/HIGH remain interaction-level strata from independent-path conflicts, not absolute difficulty labels.",
        "",
        "## Direct empirical findings",
        "",
        f"- All {len(instance_rows)} catalogue instances succeeded for Fixed Priority, Random K={config.expected_random_k}, SPF, and LPF in the stored datasets.",
        f"- {sensitive_count} of {len(instance_rows)} instances show observed priority sensitivity "
        f"(>1 unique successful Random SoC among K={config.expected_random_k} orderings).",
        f"- SPF differs from Fixed on {spf_fixed_diff} instances; LPF differs from Fixed on {lpf_fixed_diff}; SPF differs from LPF on {spf_lpf_diff} instances.",
        f"- {soc_change_makespan_constant} priority-sensitive instances change SoC while successful Random makespan remains constant.",
        "",
        "### Pairwise SoC summaries (diff = LEFT - RIGHT; positive => RIGHT better)",
        "",
    ]

    for summary in paired_summary:
        lines.append(
            f"- {summary['comparison']}: common-success={summary['common_success_count']}, "
            f"mean diff={_format_float(summary['mean_soc_diff'])}, "
            f"median diff={_format_float(summary['median_soc_diff'])}, "
            f"LEFT better={summary['left_better_count']}, "
            f"equal={summary['equal_count']}, "
            f"RIGHT better={summary['right_better_count']}"
        )

    lines.extend(
        [
            "",
            "### Runtime",
            "",
            f"- Mean SPF total runtime: {_format_float(aggregate.get('mean_spf_total_time_ms'))} ms "
            f"(ordering {_format_float(aggregate.get('mean_spf_ordering_time_ms'))} ms, "
            f"PP {_format_float(aggregate.get('mean_spf_pp_time_ms'))} ms).",
            f"- Mean LPF total runtime: {_format_float(aggregate.get('mean_lpf_total_time_ms'))} ms "
            f"(ordering {_format_float(aggregate.get('mean_lpf_ordering_time_ms'))} ms, "
            f"PP {_format_float(aggregate.get('mean_lpf_pp_time_ms'))} ms).",
            f"- SPF total faster on {aggregate.get('spf_total_faster_count', 'n/a')} instances; "
            f"LPF total faster on {aggregate.get('lpf_total_faster_count', 'n/a')} instances.",
            "",
            "### CBS quality reference",
            "",
            f"- Common-success Basic CBS reference available on {len(cbs_rows)} instances.",
            "",
            "## Interpretation",
            "",
            "- Priority ordering can change PP solution quality on a subset of instances even when makespan remains unchanged, which supports studying adaptive or conflict-aware priority selection rather than assuming a single fixed order.",
            "- SPF and LPF do not uniformly dominate Fixed Priority or Random K=10; their value depends on instance-level interaction structure.",
            "- The extra independent-path ordering cost of SPF/LPF is non-trivial and must be accounted for when comparing runtime against Fixed or Random PP.",
            "- CBS remains useful as a quality reference on solvable instances, but timeout/expansion-limit records are excluded from quality means.",
            "",
        ]
    )
    return "\n".join(lines)


def build_thesis_tables_md(
    *,
    instance_rows: Sequence[dict[str, Any]],
    sensitivity_rows: Sequence[dict[str, Any]],
    paired_summary: Sequence[dict[str, Any]],
    runtime_rows: Sequence[dict[str, Any]],
    cbs_rows: Sequence[dict[str, Any]],
    expected_random_k: int,
) -> str:
    success_rates = {
        "fixed": sum(1 for row in instance_rows if row["fixed_success"]) / len(instance_rows),
        "spf": sum(1 for row in instance_rows if row["spf_success"]) / len(instance_rows),
        "lpf": sum(1 for row in instance_rows if row["lpf_success"]) / len(instance_rows),
        "random_mean": statistics.mean(row["random_success_rate"] for row in instance_rows),
    }

    sensitive_rows = [
        row for row in sensitivity_rows if row["observed_priority_sensitive"]
    ]
    sensitive_rows.sort(key=lambda row: row["random_soc_range"], reverse=True)

    aggregate = next(
        row for row in runtime_rows if row.get("metric_group") == "runtime_aggregate"
    )

    lines = [
        "# MAPF-6 Thesis Tables",
        "",
        "## Table A — Success rates",
        "",
        "| Strategy | Success rate |",
        "|---|---:|",
        f"| Fixed Priority | {success_rates['fixed']:.3f} |",
        f"| Random K={expected_random_k} (instance mean) | {success_rates['random_mean']:.3f} |",
        f"| SPF | {success_rates['spf']:.3f} |",
        f"| LPF | {success_rates['lpf']:.3f} |",
        "",
        "## Table B — Pairwise SoC comparisons",
        "",
        "| Comparison | Common success | Mean diff | Median diff | LEFT better | Equal | RIGHT better |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for summary in paired_summary:
        lines.append(
            f"| {summary['comparison']} | {summary['common_success_count']} | "
            f"{_format_float(summary['mean_soc_diff'])} | "
            f"{_format_float(summary['median_soc_diff'])} | "
            f"{summary['left_better_count']} | {summary['equal_count']} | "
            f"{summary['right_better_count']} |"
        )

    lines.extend(
        [
            "",
            "## Table C — Random sensitivity summary",
            "",
            f"| Metric | Value |",
            f"|---|---:|",
            f"| Priority-sensitive instances | {len(sensitive_rows)} / {len(instance_rows)} |",
            f"| Max Random SoC range | {max((row['random_soc_range'] or 0) for row in sensitivity_rows)} |",
            "",
            "## Table D — SPF vs LPF runtime cost",
            "",
            "| Metric | SPF | LPF |",
            "|---|---:|---:|",
            f"| Mean total runtime (ms) | {_format_float(aggregate['mean_spf_total_time_ms'])} | {_format_float(aggregate['mean_lpf_total_time_ms'])} |",
            f"| Mean ordering runtime (ms) | {_format_float(aggregate['mean_spf_ordering_time_ms'])} | {_format_float(aggregate['mean_lpf_ordering_time_ms'])} |",
            f"| Mean PP runtime (ms) | {_format_float(aggregate['mean_spf_pp_time_ms'])} | {_format_float(aggregate['mean_lpf_pp_time_ms'])} |",
            f"| Median total runtime (ms) | {_format_float(aggregate['median_spf_total_time_ms'])} | {_format_float(aggregate['median_lpf_total_time_ms'])} |",
            f"| Instances where strategy is faster (total) | {aggregate['spf_total_faster_count']} (SPF) | {aggregate['lpf_total_faster_count']} (LPF) |",
            "",
            "## Table E — Selected priority-sensitive cases",
            "",
            "| Instance | Agents | Interaction | Random SoC range | Fixed | SPF | LPF | Random median |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )

    lookup = {row["instance_id"]: row for row in instance_rows}
    for row in sensitive_rows[:10]:
        instance = lookup[row["instance_id"]]
        lines.append(
            f"| {row['instance_id']} | {row['agent_count']} | {row['interaction_level']} | "
            f"{row['random_soc_range']} | {instance['fixed_soc']} | {instance['spf_soc']} | "
            f"{instance['lpf_soc']} | {instance['random_soc_median']} |"
        )

    lines.extend(
        [
            "",
            "## Table F — CBS quality reference (common success only)",
            "",
            "| Instance | CBS SoC | Fixed | SPF | LPF | Random median | Sampled best-of-K |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in cbs_rows[:10]:
        lines.append(
            f"| {row['instance_id']} | {row['cbs_soc']} | {row['fixed_soc']} | "
            f"{row['spf_soc']} | {row['lpf_soc']} | {row['random_soc_median']} | "
            f"{row['random_sampled_best_of_k_soc']} |"
        )

    lines.append("")
    return "\n".join(lines)


def run_priority_strategy_analysis(
    config: PriorityStrategyAnalysisConfig,
    *,
    plot_writer: Callable[[Path, Sequence[dict[str, Any]], Sequence[dict[str, Any]]], tuple[str, ...]]
    | None = None,
) -> PriorityStrategyAnalysisResult:
    manifest = load_benchmark_manifest(config.manifest_path)
    fixed_records = load_fixed_results(config.fixed_results_csv)
    random_records = load_random_results(config.random_results_csv)
    strategy_records = load_strategy_results(config.strategy_results_csv)

    validation = validate_datasets(
        manifest=manifest,
        fixed_records=fixed_records,
        random_records=random_records,
        strategy_records=strategy_records,
        config=config,
    )

    if not validation.passed:
        return PriorityStrategyAnalysisResult(
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
        strategy_records=strategy_records,
    )
    sensitivity_rows = build_random_sensitivity(instance_rows)
    paired_summary = build_paired_quality_summary(instance_rows)
    runtime_rows = build_runtime_summary(instance_rows)
    cbs_rows = build_cbs_quality_reference(
        instance_rows=instance_rows,
        fixed_records=fixed_records,
    )
    deterministic_vs_random = build_deterministic_vs_random_summary(
        instance_rows=instance_rows,
        random_records=random_records,
    )

    output_dir = config.output_dir
    tables_written: list[str] = []

    table_specs: list[tuple[str, list[dict[str, Any]], Sequence[str] | None]] = [
        ("instance_level_summary.csv", instance_rows, None),
        ("random_sensitivity.csv", sensitivity_rows, None),
        ("paired_quality_summary.csv", paired_summary, None),
        ("runtime_summary.csv", runtime_rows, None),
        ("cbs_quality_reference.csv", cbs_rows, None),
    ]

    for filename, rows, fieldnames in table_specs:
        if fieldnames is None and rows:
            fieldnames_list: list[str] = []
            for row in rows:
                for key in row:
                    if key not in fieldnames_list:
                        fieldnames_list.append(key)
            fieldnames = fieldnames_list
        if fieldnames is not None:
            _write_csv(output_dir / filename, rows, fieldnames)
            tables_written.append(filename)

    _write_csv(
        output_dir / "deterministic_vs_random.csv",
        deterministic_vs_random,
        list(deterministic_vs_random[0].keys()) if deterministic_vs_random else [],
    )
    tables_written.append("deterministic_vs_random.csv")

    _write_text(
        output_dir / "analysis_summary.md",
        build_analysis_summary_md(
            config=config,
            validation=validation,
            instance_rows=instance_rows,
            sensitivity_rows=sensitivity_rows,
            paired_summary=paired_summary,
            runtime_rows=runtime_rows,
            cbs_rows=cbs_rows,
        ),
    )
    tables_written.append("analysis_summary.md")

    _write_text(
        output_dir / "thesis_tables.md",
        build_thesis_tables_md(
            instance_rows=instance_rows,
            sensitivity_rows=sensitivity_rows,
            paired_summary=paired_summary,
            runtime_rows=runtime_rows,
            cbs_rows=cbs_rows,
            expected_random_k=config.expected_random_k,
        ),
    )
    tables_written.append("thesis_tables.md")

    plots_written: tuple[str, ...] = ()
    if plot_writer is not None:
        plots_written = plot_writer(output_dir, instance_rows, sensitivity_rows)

    return PriorityStrategyAnalysisResult(
        validation=validation,
        output_dir=output_dir,
        tables_written=tuple(tables_written),
        plots_written=plots_written,
    )
