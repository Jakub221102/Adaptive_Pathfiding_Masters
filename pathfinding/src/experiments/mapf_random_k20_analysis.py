from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pathfinding.src.experiments.mapf_benchmark_execution import (
    TERMINATION_FAILURE,
    TERMINATION_SUCCESS,
)
from pathfinding.src.experiments.mapf_benchmark_instances import (
    MAPFBenchmarkManifest,
    load_benchmark_manifest,
)
from pathfinding.src.experiments.mapf_random_priority_execution import (
    DEFAULT_RANDOM_PRIORITY_K,
    RandomPriorityRunKey,
    RandomPriorityRunRecord,
    SUPPLEMENTAL_ORDERING_COUNT,
    SUPPLEMENTAL_ORDERING_INDEX_START,
    load_checkpoint_records,
)

K10_ORDERING_INDICES = tuple(range(DEFAULT_RANDOM_PRIORITY_K))
SUPPLEMENTAL_ORDERING_INDICES = tuple(
    range(
        SUPPLEMENTAL_ORDERING_INDEX_START,
        SUPPLEMENTAL_ORDERING_INDEX_START + SUPPLEMENTAL_ORDERING_COUNT,
    )
)
K20_ORDERING_INDICES = tuple(range(DEFAULT_RANDOM_PRIORITY_K + SUPPLEMENTAL_ORDERING_COUNT))

EXPECTED_INSTANCE_COUNT = 27
EXPECTED_K10_RECORDS = EXPECTED_INSTANCE_COUNT * DEFAULT_RANDOM_PRIORITY_K
EXPECTED_SUPPLEMENTAL_RECORDS = (
    EXPECTED_INSTANCE_COUNT * SUPPLEMENTAL_ORDERING_COUNT
)
EXPECTED_K20_RECORDS = EXPECTED_K10_RECORDS + EXPECTED_SUPPLEMENTAL_RECORDS

AGENT_COUNT_ORDER: tuple[int, ...] = (5, 10, 20)
INTERACTION_ORDER: tuple[str, ...] = ("low", "medium", "high")

OUTPUT_TABLES: tuple[str, ...] = (
    "k10_vs_k20_summary.csv",
    "sensitive_instances.csv",
    "distinct_soc_values.csv",
    "analysis_summary.md",
)

OUTPUT_FIGURES: tuple[str, ...] = ("fig_random_k10_vs_k20_soc_range",)


@dataclass(frozen=True, slots=True)
class RandomK20AnalysisConfig:
    historical_k10_jsonl: Path
    supplemental_jsonl: Path
    manifest_path: Path
    output_dir: Path
    expected_instance_count: int = EXPECTED_INSTANCE_COUNT


@dataclass
class ValidationReport:
    passed: bool
    messages: list[str] = field(default_factory=list)
    k10_record_count: int = 0
    supplemental_record_count: int = 0
    combined_record_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    def add(self, message: str, *, passed: bool) -> None:
        self.messages.append(message)
        if not passed:
            self.passed = False


@dataclass(frozen=True, slots=True)
class InstanceSampleMetrics:
    instance_id: str
    agent_count: int
    interaction_level: str
    successful_run_count: int
    soc_min: int | None
    soc_max: int | None
    soc_range: int
    soc_median: float | None
    soc_mean: float | None
    unique_soc_count: int
    makespan_min: int | None
    makespan_max: int | None
    makespan_range: int
    unique_makespan_count: int
    distinct_socs: tuple[int, ...]
    soc_sensitive: bool
    makespan_sensitive: bool


@dataclass(frozen=True, slots=True)
class RandomK20AnalysisResult:
    validation: ValidationReport
    output_dir: Path
    tables_written: tuple[str, ...]
    figures_written: tuple[str, ...]
    k10_global: dict[str, Any]
    k20_global: dict[str, Any]


def _format_sorted_ints(values: set[int] | frozenset[int]) -> str:
    return json.dumps(sorted(values))


def _instance_sort_key(instance_id: str, agent_count: int, interaction_level: str) -> tuple:
    return (
        AGENT_COUNT_ORDER.index(agent_count)
        if agent_count in AGENT_COUNT_ORDER
        else agent_count,
        INTERACTION_ORDER.index(interaction_level)
        if interaction_level in INTERACTION_ORDER
        else interaction_level,
        instance_id,
    )


def _records_by_instance(
    records: Sequence[RandomPriorityRunRecord],
) -> dict[str, tuple[RandomPriorityRunRecord, ...]]:
    grouped: dict[str, list[RandomPriorityRunRecord]] = {}
    for record in records:
        grouped.setdefault(record.instance_id, []).append(record)
    return {
        instance_id: tuple(sorted(items, key=lambda item: item.ordering_index))
        for instance_id, items in grouped.items()
    }


def _validate_record_partition(
    *,
    records: dict[RandomPriorityRunKey, RandomPriorityRunRecord],
    label: str,
    expected_total: int,
    expected_indices: tuple[int, ...],
    expected_instance_ids: set[str],
    report: ValidationReport,
) -> None:
    report.add(
        f"{label}: exactly {expected_total} records "
        f"(found {len(records)})",
        passed=len(records) == expected_total,
    )

    by_instance = _records_by_instance(records.values())
    instance_ids = set(by_instance)
    report.add(
        f"{label}: exactly {len(expected_instance_ids)} unique instance IDs "
        f"(found {len(instance_ids)})",
        passed=instance_ids == expected_instance_ids,
    )

    if instance_ids != expected_instance_ids:
        missing = sorted(expected_instance_ids - instance_ids)
        extra = sorted(instance_ids - expected_instance_ids)
        if missing:
            report.add(f"{label}: missing instance IDs: {missing}", passed=False)
        if extra:
            report.add(f"{label}: unexpected instance IDs: {extra}", passed=False)

    per_instance_count = len(expected_indices)
    for instance_id in sorted(expected_instance_ids):
        instance_records = by_instance.get(instance_id, ())
        report.add(
            f"{label}: {instance_id} has exactly {per_instance_count} records "
            f"(found {len(instance_records)})",
            passed=len(instance_records) == per_instance_count,
        )
        observed_indices = {record.ordering_index for record in instance_records}
        report.add(
            f"{label}: {instance_id} ordering indices exactly "
            f"{list(expected_indices)} (found {sorted(observed_indices)})",
            passed=observed_indices == set(expected_indices),
        )

    keys = list(records.keys())
    report.add(
        f"{label}: all (instance_id, ordering_index) keys unique",
        passed=len(keys) == len(set(keys)),
    )


def _validate_combined_records(
    *,
    combined: dict[RandomPriorityRunKey, RandomPriorityRunRecord],
    expected_instance_ids: set[str],
    report: ValidationReport,
) -> None:
    report.combined_record_count = len(combined)
    report.add(
        f"Combined K=20: exactly {EXPECTED_K20_RECORDS} records "
        f"(found {len(combined)})",
        passed=len(combined) == EXPECTED_K20_RECORDS,
    )

    by_instance = _records_by_instance(combined.values())
    report.add(
        f"Combined K=20: exactly {EXPECTED_INSTANCE_COUNT} unique instances "
        f"(found {len(by_instance)})",
        passed=len(by_instance) == EXPECTED_INSTANCE_COUNT
        and set(by_instance) == expected_instance_ids,
    )

    for instance_id in sorted(expected_instance_ids):
        instance_records = by_instance[instance_id]
        report.add(
            f"Combined K=20: {instance_id} has exactly 20 records "
            f"(found {len(instance_records)})",
            passed=len(instance_records) == 20,
        )
        observed_indices = {record.ordering_index for record in instance_records}
        report.add(
            f"Combined K=20: {instance_id} ordering indices exactly 0..19 "
            f"(found {sorted(observed_indices)})",
            passed=observed_indices == set(K20_ORDERING_INDICES),
        )

        agent_orders = [record.agent_order for record in instance_records]
        report.add(
            f"Combined K=20: {instance_id} has no duplicate agent_order "
            f"({len(agent_orders)} orders, {len(set(agent_orders))} unique)",
            passed=len(agent_orders) == len(set(agent_orders)),
        )

    allowed_terminations = {TERMINATION_SUCCESS, TERMINATION_FAILURE}
    success_count = 0
    failure_count = 0
    for record in combined.values():
        if record.termination_reason not in allowed_terminations:
            report.add(
                f"Invalid termination_reason for {record.instance_id} / "
                f"ordering {record.ordering_index}: {record.termination_reason}",
                passed=False,
            )
        if record.success:
            success_count += 1
            if record.conflict_count != 0:
                report.add(
                    f"Successful run {record.instance_id} / ordering "
                    f"{record.ordering_index} has conflict_count="
                    f"{record.conflict_count}, expected 0",
                    passed=False,
                )
        else:
            failure_count += 1

    report.success_count = success_count
    report.failure_count = failure_count
    report.add(
        f"Combined K=20: success/failure counts = "
        f"{success_count}/{failure_count}",
        passed=True,
    )


def validate_random_k20_datasets(
    *,
    manifest: MAPFBenchmarkManifest,
    k10_records: dict[RandomPriorityRunKey, RandomPriorityRunRecord],
    supplemental_records: dict[RandomPriorityRunKey, RandomPriorityRunRecord],
) -> ValidationReport:
    report = ValidationReport(passed=True)
    expected_instance_ids = {instance.instance_id for instance in manifest.instances}

    _validate_record_partition(
        records=k10_records,
        label="Historical K=10",
        expected_total=EXPECTED_K10_RECORDS,
        expected_indices=K10_ORDERING_INDICES,
        expected_instance_ids=expected_instance_ids,
        report=report,
    )
    report.k10_record_count = len(k10_records)

    _validate_record_partition(
        records=supplemental_records,
        label="Supplemental K=20 extension",
        expected_total=EXPECTED_SUPPLEMENTAL_RECORDS,
        expected_indices=SUPPLEMENTAL_ORDERING_INDICES,
        expected_instance_ids=expected_instance_ids,
        report=report,
    )
    report.supplemental_record_count = len(supplemental_records)

    overlap = set(k10_records) & set(supplemental_records)
    report.add(
        f"Historical and supplemental partitions are disjoint "
        f"({len(overlap)} overlapping keys)",
        passed=not overlap,
    )

    combined = dict(k10_records)
    combined.update(supplemental_records)
    _validate_combined_records(
        combined=combined,
        expected_instance_ids=expected_instance_ids,
        report=report,
    )

    return report


def compute_instance_sample_metrics(
    *,
    instance_id: str,
    agent_count: int,
    interaction_level: str,
    records: Sequence[RandomPriorityRunRecord],
) -> InstanceSampleMetrics:
    successful = [record for record in records if record.success]
    successful_socs = [
        record.soc for record in successful if record.soc is not None
    ]
    successful_makespans = [
        record.makespan for record in successful if record.makespan is not None
    ]
    distinct_socs = tuple(sorted(set(successful_socs)))

    soc_range = (
        max(successful_socs) - min(successful_socs) if successful_socs else 0
    )
    makespan_range = (
        max(successful_makespans) - min(successful_makespans)
        if successful_makespans
        else 0
    )

    return InstanceSampleMetrics(
        instance_id=instance_id,
        agent_count=agent_count,
        interaction_level=interaction_level,
        successful_run_count=len(successful),
        soc_min=min(successful_socs) if successful_socs else None,
        soc_max=max(successful_socs) if successful_socs else None,
        soc_range=soc_range,
        soc_median=(
            statistics.median(successful_socs) if successful_socs else None
        ),
        soc_mean=statistics.mean(successful_socs) if successful_socs else None,
        unique_soc_count=len(distinct_socs),
        makespan_min=min(successful_makespans) if successful_makespans else None,
        makespan_max=max(successful_makespans) if successful_makespans else None,
        makespan_range=makespan_range,
        unique_makespan_count=len(set(successful_makespans)),
        distinct_socs=distinct_socs,
        soc_sensitive=soc_range > 0,
        makespan_sensitive=makespan_range > 0,
    )


def _distinct_socs_for_indices(
    records: Sequence[RandomPriorityRunRecord],
    indices: set[int],
) -> set[int]:
    values: set[int] = set()
    for record in records:
        if record.ordering_index not in indices:
            continue
        if record.success and record.soc is not None:
            values.add(record.soc)
    return values


def compute_global_sample_summary(
    metrics_by_instance: dict[str, InstanceSampleMetrics],
) -> dict[str, Any]:
    instance_count = len(metrics_by_instance)
    soc_ranges = [metrics.soc_range for metrics in metrics_by_instance.values()]
    soc_sensitive_count = sum(
        1 for metrics in metrics_by_instance.values() if metrics.soc_sensitive
    )
    makespan_sensitive_count = sum(
        1
        for metrics in metrics_by_instance.values()
        if metrics.makespan_sensitive
    )

    return {
        "instance_count": instance_count,
        "soc_sensitive_count": soc_sensitive_count,
        "soc_sensitive_fraction": f"{soc_sensitive_count}/{instance_count}",
        "makespan_sensitive_count": makespan_sensitive_count,
        "makespan_sensitive_fraction": (
            f"{makespan_sensitive_count}/{instance_count}"
        ),
        "max_soc_range": max(soc_ranges) if soc_ranges else 0,
        "mean_soc_range": statistics.mean(soc_ranges) if soc_ranges else 0.0,
        "median_soc_range": statistics.median(soc_ranges) if soc_ranges else 0.0,
    }


def build_k10_vs_k20_comparison_rows(
    *,
    k10_by_instance: dict[str, tuple[RandomPriorityRunRecord, ...]],
    k20_by_instance: dict[str, tuple[RandomPriorityRunRecord, ...]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for instance_id in sorted(
        k20_by_instance,
        key=lambda item: _instance_sort_key(
            item,
            k20_by_instance[item][0].agent_count,
            k20_by_instance[item][0].interaction_level,
        ),
    ):
        k10_records = k10_by_instance[instance_id]
        k20_records = k20_by_instance[instance_id]
        k10 = compute_instance_sample_metrics(
            instance_id=instance_id,
            agent_count=k10_records[0].agent_count,
            interaction_level=k10_records[0].interaction_level,
            records=k10_records,
        )
        k20 = compute_instance_sample_metrics(
            instance_id=instance_id,
            agent_count=k20_records[0].agent_count,
            interaction_level=k20_records[0].interaction_level,
            records=k20_records,
        )

        k10_socs = _distinct_socs_for_indices(k10_records, set(K10_ORDERING_INDICES))
        supplemental_socs = _distinct_socs_for_indices(
            k20_records,
            set(SUPPLEMENTAL_ORDERING_INDICES),
        )
        new_soc_values = supplemental_socs - k10_socs

        rows.append(
            {
                "instance_id": instance_id,
                "agent_count": k20.agent_count,
                "interaction_level": k20.interaction_level,
                "K10_min_soc": k10.soc_min,
                "K10_max_soc": k10.soc_max,
                "K10_soc_range": k10.soc_range,
                "K10_unique_soc_count": k10.unique_soc_count,
                "K20_min_soc": k20.soc_min,
                "K20_max_soc": k20.soc_max,
                "K20_soc_range": k20.soc_range,
                "K20_unique_soc_count": k20.unique_soc_count,
                "delta_soc_range": k20.soc_range - k10.soc_range,
                "new_soc_values_found_in_10_19": _format_sorted_ints(new_soc_values),
                "K10_makespan_range": k10.makespan_range,
                "K20_makespan_range": k20.makespan_range,
                "became_newly_soc_sensitive": (not k10.soc_sensitive) and k20.soc_sensitive,
                "became_newly_makespan_sensitive": (
                    (not k10.makespan_sensitive) and k20.makespan_sensitive
                ),
                "new_internal_soc_without_range_increase": (
                    k20.unique_soc_count > k10.unique_soc_count
                    and k20.soc_range == k10.soc_range
                ),
            }
        )

    return rows


def build_sensitive_instance_rows(
    comparison_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    sensitive = [row for row in comparison_rows if (row["K20_soc_range"] or 0) > 0]
    sensitive.sort(
        key=lambda row: (
            -(row["K20_soc_range"] or 0),
            row["agent_count"],
            INTERACTION_ORDER.index(row["interaction_level"])
            if row["interaction_level"] in INTERACTION_ORDER
            else row["interaction_level"],
            row["instance_id"],
        )
    )
    return [
        {
            "instance_id": row["instance_id"],
            "agent_count": row["agent_count"],
            "interaction_level": row["interaction_level"],
            "K10_soc_min": row["K10_min_soc"],
            "K10_soc_max": row["K10_max_soc"],
            "K10_soc_range": row["K10_soc_range"],
            "K10_unique_socs": row["K10_unique_soc_count"],
            "K20_soc_min": row["K20_min_soc"],
            "K20_soc_max": row["K20_max_soc"],
            "K20_soc_range": row["K20_soc_range"],
            "K20_unique_socs": row["K20_unique_soc_count"],
            "new_soc_values_from_indices_10_19": row["new_soc_values_found_in_10_19"],
        }
        for row in sensitive
    ]


def build_distinct_soc_value_rows(
    *,
    k10_by_instance: dict[str, tuple[RandomPriorityRunRecord, ...]],
    k20_by_instance: dict[str, tuple[RandomPriorityRunRecord, ...]],
    comparison_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sensitive_ids = {
        row["instance_id"] for row in comparison_rows if (row["K20_soc_range"] or 0) > 0
    }

    for instance_id in sorted(
        sensitive_ids,
        key=lambda item: next(
            (
                (
                    -(row["K20_soc_range"] or 0),
                    row["agent_count"],
                    INTERACTION_ORDER.index(row["interaction_level"])
                    if row["interaction_level"] in INTERACTION_ORDER
                    else row["interaction_level"],
                    row["instance_id"],
                )
                for row in comparison_rows
                if row["instance_id"] == item
            ),
            (0, 0, "", item),
        ),
    ):
        k10_records = k10_by_instance[instance_id]
        k20_records = k20_by_instance[instance_id]
        k10_socs = _distinct_socs_for_indices(k10_records, set(K10_ORDERING_INDICES))
        supplemental_socs = _distinct_socs_for_indices(
            k20_records,
            set(SUPPLEMENTAL_ORDERING_INDICES),
        )
        k20_socs = _distinct_socs_for_indices(k20_records, set(K20_ORDERING_INDICES))

        rows.append(
            {
                "instance_id": instance_id,
                "k10_distinct_socs_0_9": _format_sorted_ints(k10_socs),
                "supplemental_distinct_socs_10_19": _format_sorted_ints(
                    supplemental_socs
                ),
                "k20_distinct_socs_0_19": _format_sorted_ints(k20_socs),
                "new_socs_in_10_19_not_in_0_9": _format_sorted_ints(
                    supplemental_socs - k10_socs
                ),
            }
        )

    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
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


def _count_truthy(rows: Sequence[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if bool(row.get(key)))


def _format_global_block(title: str, summary: dict[str, Any]) -> list[str]:
    return [
        f"### {title}",
        "",
        f"- SoC-sensitive instances: {summary['soc_sensitive_fraction']}",
        f"- Makespan-sensitive instances: {summary['makespan_sensitive_fraction']}",
        f"- Maximum observed SoC range: {summary['max_soc_range']}",
        f"- Mean SoC range: {summary['mean_soc_range']:.2f}",
        f"- Median SoC range: {summary['median_soc_range']:.2f}",
        "",
    ]


def build_analysis_summary_markdown(
    *,
    config: RandomK20AnalysisConfig,
    validation: ValidationReport,
    k10_global: dict[str, Any],
    k20_global: dict[str, Any],
    comparison_rows: Sequence[dict[str, Any]],
    sensitive_rows: Sequence[dict[str, Any]],
) -> str:
    newly_soc_sensitive = [
        row["instance_id"]
        for row in comparison_rows
        if row["became_newly_soc_sensitive"]
    ]
    newly_makespan_sensitive = [
        row["instance_id"]
        for row in comparison_rows
        if row["became_newly_makespan_sensitive"]
    ]
    increased_soc_range = [
        row["instance_id"]
        for row in comparison_rows
        if (row["delta_soc_range"] or 0) > 0
    ]
    new_internal_soc = [
        row["instance_id"]
        for row in comparison_rows
        if row["new_internal_soc_without_range_increase"]
    ]
    new_soc_value_instances = [
        row["instance_id"]
        for row in comparison_rows
        if json.loads(row["new_soc_values_found_in_10_19"])
    ]

    lines = [
        "# Random K=20 Robustness Analysis Summary",
        "",
        "## 1. Data validation",
        "",
        f"- Validation status: {'PASSED' if validation.passed else 'FAILED'}",
        f"- Manifest: `{config.manifest_path}`",
        f"- Historical K=10 JSONL: `{config.historical_k10_jsonl}`",
        f"- Supplemental JSONL: `{config.supplemental_jsonl}`",
        f"- Historical K=10 record count: {validation.k10_record_count}",
        f"- Supplemental record count: {validation.supplemental_record_count}",
        f"- Combined K=20 record count: {validation.combined_record_count}",
        f"- Instance count: {EXPECTED_INSTANCE_COUNT}",
        f"- Success/failure counts: {validation.success_count}/{validation.failure_count}",
        "- Successful runs validated with `conflict_count == 0`",
        "",
    ]

    for message in validation.messages:
        lines.append(f"- {message}")
    lines.append("")

    lines.extend(_format_global_block("2. K=10 summary", k10_global))
    lines.extend(_format_global_block("3. K=20 summary", k20_global))

    lines.extend(
        [
            "## 4. K=10 -> K=20 changes",
            "",
            f"- Newly SoC-sensitive instances: {len(newly_soc_sensitive)}"
            + (
                f" ({', '.join(newly_soc_sensitive)})"
                if newly_soc_sensitive
                else " (none)"
            ),
            f"- Newly makespan-sensitive instances: {len(newly_makespan_sensitive)}"
            + (
                f" ({', '.join(newly_makespan_sensitive)})"
                if newly_makespan_sensitive
                else " (none)"
            ),
            f"- Maximum observed SoC range changed: "
            f"{k10_global['max_soc_range']} -> {k20_global['max_soc_range']}",
            f"- Instances with increased SoC range: {len(increased_soc_range)}"
            + (
                f" ({', '.join(increased_soc_range)})"
                if increased_soc_range
                else " (none)"
            ),
            f"- Instances with new SoC values in indices 10..19: "
            f"{len(new_soc_value_instances)}"
            + (
                f" ({', '.join(new_soc_value_instances)})"
                if new_soc_value_instances
                else " (none)"
            ),
            f"- Instances with additional unique SoC levels but unchanged min/max range: "
            f"{len(new_internal_soc)}"
            + (
                f" ({', '.join(new_internal_soc)})"
                if new_internal_soc
                else " (none)"
            ),
            f"- Makespan-sensitive instance count: "
            f"{k10_global['makespan_sensitive_fraction']} -> "
            f"{k20_global['makespan_sensitive_fraction']}",
            "",
            "## 5. Scientific interpretation",
            "",
            "This analysis compares the original Random K=10 sample (ordering indices 0..9) "
            "with an expanded deterministic K=20 sample (indices 0..19) on the same frozen "
            "27-instance primary catalogue. The goal is robustness checking: whether the "
            "priority-order sensitivity conclusions remain stable when the random ordering "
            "sample depth is doubled.",
            "",
            "Additional sampled orderings may reveal additional internal SoC levels even when "
            "the observed min/max SoC range does not expand. Unchanged extrema under K=20 do "
            "not prove that the full permutation space has been exhausted. K=20 remains a finite "
            "sample of possible priority permutations, not an exhaustive enumeration.",
            "",
            "No causal claims are made beyond the directly observed sample statistics.",
            "",
            "## 6. Thesis-ready paragraph (Polish)",
            "",
            _build_polish_thesis_paragraph(
                k10_global=k10_global,
                k20_global=k20_global,
                newly_soc_sensitive_count=len(newly_soc_sensitive),
                increased_soc_range_count=len(increased_soc_range),
                new_internal_soc_count=len(new_internal_soc),
            ),
            "",
        ]
    )

    if sensitive_rows:
        lines.extend(
            [
                "## Appendix — K=20 SoC-sensitive instances",
                "",
                f"{len(sensitive_rows)} instances remain SoC-sensitive under K=20 "
                "(see `sensitive_instances.csv` and `distinct_soc_values.csv`).",
                "",
            ]
        )

    return "\n".join(lines)


def _build_polish_thesis_paragraph(
    *,
    k10_global: dict[str, Any],
    k20_global: dict[str, Any],
    newly_soc_sensitive_count: int,
    increased_soc_range_count: int,
    new_internal_soc_count: int,
) -> str:
    return (
        "Aby ocenić stabilność wniosków z losowego doboru kolejności priorytetów, "
        "uzupełniono pierwotny deterministyczny próbnik K=10 (indeksy 0–9) o dodatkowe "
        "10 unikalnych permutacji na instancję (indeksy 10–19), tworząc łącznie próbkę "
        f"K=20 obejmującą 540 uruchomień PP na 27 instancjach katalogu głównego. "
        f"W próbce K=10 zaobserwowano {k10_global['soc_sensitive_count']} instancji "
        f"wrażliwych na SoC (maksymalny zakres SoC: {k10_global['max_soc_range']}), "
        f"natomiast w próbce K=20 — {k20_global['soc_sensitive_count']} takich instancji "
        f"(maksymalny zakres SoC: {k20_global['max_soc_range']}). "
        f"Po rozszerzeniu próbki nie wykryto nowych instancji wrażliwych na SoC "
        f"({newly_soc_sensitive_count}), zakresy min/max SoC nie uległy poszerzeniu "
        f"({increased_soc_range_count} instancji ze wzrostem zakresu), "
        f"choć w {new_internal_soc_count} przypadkach dodatkowe uporządkowania ujawniły "
        "nowe pośrednie wartości SoC przy niezmienionych ekstremach. "
        "Wnioski należy traktować konserwatywnie: K=20 nadal stanowi skończoną próbę "
        "permutacji priorytetów, a nie pełną enumerację przestrzeni kolejności."
    )


def render_k10_vs_k20_soc_range_figure(
    *,
    sensitive_rows: Sequence[dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, Path]:
    from pathfinding.src.experiments.mapf_conflict_aware_priority_plots import (
        _apply_plot_style,
        _save_figure,
        short_instance_label,
    )

    import matplotlib.pyplot as plt
    import numpy as np

    _apply_plot_style()
    labels = [short_instance_label(row["instance_id"]) for row in sensitive_rows]
    k10_ranges = [row["K10_soc_range"] for row in sensitive_rows]
    k20_ranges = [row["K20_soc_range"] for row in sensitive_rows]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 0.55), 5.0))
    ax.bar(x - width / 2, k10_ranges, width, label="K=10", color="#777777")
    ax.bar(x + width / 2, k20_ranges, width, label="K=20", color="#4C72B0")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Observed SoC range")
    ax.set_xlabel("SoC-sensitive instance")
    ax.set_title("Random K=10 vs K=20 observed SoC range (sensitive instances)")
    ax.legend(loc="upper right")
    fig.tight_layout()

    return _save_figure(
        fig,
        output_dir,
        "fig_random_k10_vs_k20_soc_range",
    )


def run_random_k20_analysis(config: RandomK20AnalysisConfig) -> RandomK20AnalysisResult:
    manifest = load_benchmark_manifest(config.manifest_path)
    k10_records = load_checkpoint_records(config.historical_k10_jsonl)
    supplemental_records = load_checkpoint_records(config.supplemental_jsonl)

    validation = validate_random_k20_datasets(
        manifest=manifest,
        k10_records=k10_records,
        supplemental_records=supplemental_records,
    )
    if not validation.passed:
        messages = "\n".join(f"  - {message}" for message in validation.messages)
        raise RuntimeError(
            "Random K=20 analysis validation failed:\n" + messages
        )

    combined = dict(k10_records)
    combined.update(supplemental_records)
    k10_by_instance = _records_by_instance(k10_records.values())
    k20_by_instance = _records_by_instance(combined.values())

    k10_metrics = {
        instance_id: compute_instance_sample_metrics(
            instance_id=instance_id,
            agent_count=records[0].agent_count,
            interaction_level=records[0].interaction_level,
            records=records,
        )
        for instance_id, records in k10_by_instance.items()
    }
    k20_metrics = {
        instance_id: compute_instance_sample_metrics(
            instance_id=instance_id,
            agent_count=records[0].agent_count,
            interaction_level=records[0].interaction_level,
            records=records,
        )
        for instance_id, records in k20_by_instance.items()
    }

    k10_global = compute_global_sample_summary(k10_metrics)
    k20_global = compute_global_sample_summary(k20_metrics)
    comparison_rows = build_k10_vs_k20_comparison_rows(
        k10_by_instance=k10_by_instance,
        k20_by_instance=k20_by_instance,
    )
    sensitive_rows = build_sensitive_instance_rows(comparison_rows)
    distinct_soc_rows = build_distinct_soc_value_rows(
        k10_by_instance=k10_by_instance,
        k20_by_instance=k20_by_instance,
        comparison_rows=comparison_rows,
    )

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(output_dir / "k10_vs_k20_summary.csv", comparison_rows)
    _write_csv(output_dir / "sensitive_instances.csv", sensitive_rows)
    _write_csv(output_dir / "distinct_soc_values.csv", distinct_soc_rows)

    summary_path = output_dir / "analysis_summary.md"
    summary_path.write_text(
        build_analysis_summary_markdown(
            config=config,
            validation=validation,
            k10_global=k10_global,
            k20_global=k20_global,
            comparison_rows=comparison_rows,
            sensitive_rows=sensitive_rows,
        ),
        encoding="utf-8",
    )

    figures_written: list[str] = []
    if sensitive_rows:
        png_path, _ = render_k10_vs_k20_soc_range_figure(
            sensitive_rows=sensitive_rows,
            output_dir=output_dir,
        )
        figures_written.append(png_path.stem)

    return RandomK20AnalysisResult(
        validation=validation,
        output_dir=output_dir,
        tables_written=OUTPUT_TABLES,
        figures_written=tuple(figures_written),
        k10_global=k10_global,
        k20_global=k20_global,
    )
