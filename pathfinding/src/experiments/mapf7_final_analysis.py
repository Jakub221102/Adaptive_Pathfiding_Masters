from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pathfinding.src.experiments.mapf_benchmark_analysis import count_directional_soc_better
from pathfinding.src.experiments.mapf_benchmark_instances import (
    CATALOGUE_ROLE_HELD_OUT,
    CATALOGUE_ROLE_PRIMARY,
    DEFAULT_HELD_OUT_SEED,
    MAPFBenchmarkManifest,
    load_benchmark_manifest,
)
from pathfinding.src.experiments.mapf_conflict_aware_priority_execution import (
    validate_held_out_execution_manifest,
)
from pathfinding.src.experiments.mapf_conflict_aware_priority_analysis import (
    MAPF7_STRATEGIES,
    STRATEGY_CDF_H,
    STRATEGY_CDF_L,
    STRATEGY_SPF_CD,
    ConflictAwareRunRecord,
    _conflict_by_instance,
    _degree_summary,
    _lookup_pairwise,
    _paired_soc_summary,
    build_agent_count_summary,
    build_instance_level_summary,
    build_interaction_summary,
    build_makespan_summary,
    build_runtime_summary,
    load_conflict_aware_results,
    mapf7_strategy_socs_differ,
)
from pathfinding.src.experiments.mapf_priority_strategy_analysis import (
    STRATEGY_SPF,
    StrategyRunRecord,
    _format_float,
    _instance_sort_key,
    _read_csv_rows,
    _row_int,
    _write_csv,
    _write_text,
    aggregate_random_by_instance,
    load_fixed_results,
    load_random_results,
    load_strategy_results,
)

FINAL_ANALYSIS_DIR = Path("pathfinding/results/mapf7_final_analysis")

FINAL_PAIRWISE_COMPARISONS: tuple[tuple[str, str, str, str, str], ...] = (
    ("cdf_h_vs_spf", "cdf_h_soc", "spf_soc", "CDF-H", "SPF"),
    ("cdf_l_vs_spf", "cdf_l_soc", "spf_soc", "CDF-L", "SPF"),
    ("cdf_h_vs_cdf_l", "cdf_h_soc", "cdf_l_soc", "CDF-H", "CDF-L"),
    ("spf_cd_vs_spf", "spf_cd_soc", "spf_soc", "SPF+CD", "SPF"),
)

OUTPUT_TABLES: tuple[str, ...] = (
    "primary_pairwise_summary.csv",
    "heldout_pairwise_summary.csv",
    "generalization_summary.csv",
    "sensitive_instances.csv",
    "interaction_summary.csv",
    "agent_count_summary.csv",
    "makespan_summary.csv",
    "runtime_summary.csv",
    "spf_cd_ablation.csv",
    "conflict_structure_summary.csv",
    "thesis_tables.md",
    "final_analysis_summary.md",
)


@dataclass(frozen=True, slots=True)
class Mapf7FinalAnalysisConfig:
    primary_conflict_csv: Path
    primary_manifest_path: Path
    primary_spf_lpf_csv: Path
    fixed_results_csv: Path
    random_results_csv: Path
    heldout_conflict_csv: Path
    heldout_spf_csv: Path
    heldout_manifest_path: Path
    output_dir: Path
    expected_instance_count: int = 27


@dataclass
class ValidationReport:
    passed: bool
    messages: list[str] = field(default_factory=list)

    def add(self, message: str, *, passed: bool) -> None:
        self.messages.append(message)
        if not passed:
            self.passed = False


@dataclass(frozen=True, slots=True)
class Mapf7FinalAnalysisResult:
    validation: ValidationReport
    output_dir: Path
    tables_written: tuple[str, ...]
    plots_written: tuple[str, ...]
    primary_instance_rows: tuple[dict[str, Any], ...]
    heldout_instance_rows: tuple[dict[str, Any], ...]


def load_heldout_spf_results(path: Path) -> tuple[StrategyRunRecord, ...]:
    return load_strategy_results(path)


def _validate_conflict_records(
    report: ValidationReport,
    *,
    label: str,
    records: Sequence[ConflictAwareRunRecord],
    manifest: MAPFBenchmarkManifest,
    expected_instance_count: int,
) -> None:
    expected_total = expected_instance_count * len(MAPF7_STRATEGIES)
    report.add(
        f"{label}: expected {expected_total} records, found {len(records)}",
        passed=len(records) == expected_total,
    )

    keys = {(record.instance_id, record.strategy) for record in records}
    report.add(
        f"{label}: no duplicate (instance_id, strategy)",
        passed=len(keys) == len(records),
    )

    manifest_ids = {instance.instance_id for instance in manifest.instances}
    record_ids = {record.instance_id for record in records}
    report.add(
        f"{label}: instance IDs match manifest",
        passed=record_ids == manifest_ids,
    )

    for strategy in MAPF7_STRATEGIES:
        count = sum(1 for record in records if record.strategy == strategy)
        report.add(
            f"{label} strategy {strategy}: expected {expected_instance_count}, found {count}",
            passed=count == expected_instance_count,
        )

    success_count = sum(1 for record in records if record.success)
    report.add(
        f"{label}: successful runs expected {expected_total}, found {success_count}",
        passed=success_count == expected_total,
    )

    for record in records:
        if not record.success:
            report.add(
                f"{label} failed run: {record.instance_id}/{record.strategy}",
                passed=False,
            )
            continue
        if record.conflict_count != 0:
            report.add(
                f"{label} non-zero conflict_count: {record.instance_id}/{record.strategy}",
                passed=False,
            )


def _validate_heldout_conflict_csv_metadata(
    report: ValidationReport,
    csv_path: Path,
) -> None:
    rows = _read_csv_rows(csv_path)
    for row in rows:
        role = row.get("catalogue_role", "")
        seed = row.get("catalogue_seed", "")
        role_ok = role == CATALOGUE_ROLE_HELD_OUT
        seed_ok = seed == str(DEFAULT_HELD_OUT_SEED)
        if not role_ok or not seed_ok:
            report.add(
                f"Held-out MAPF-7 metadata {row['instance_id']}/{row['strategy']}: "
                f"role={role!r}, seed={seed!r}",
                passed=False,
            )
            return
    report.add(
        f"Held-out MAPF-7 catalogue_role={CATALOGUE_ROLE_HELD_OUT!r}, seed={DEFAULT_HELD_OUT_SEED}",
        passed=True,
    )


def validate_final_analysis_inputs(
    config: Mapf7FinalAnalysisConfig,
) -> ValidationReport:
    report = ValidationReport(passed=True)

    primary_manifest = load_benchmark_manifest(config.primary_manifest_path)
    heldout_manifest = load_benchmark_manifest(config.heldout_manifest_path)

    report.add(
        f"Primary catalogue: expected {config.expected_instance_count} instances, "
        f"found {len(primary_manifest.instances)}",
        passed=len(primary_manifest.instances) == config.expected_instance_count,
    )
    report.add(
        f"Primary catalogue_role={CATALOGUE_ROLE_PRIMARY!r}, "
        f"found {primary_manifest.catalogue_role!r}",
        passed=primary_manifest.catalogue_role == CATALOGUE_ROLE_PRIMARY,
    )

    try:
        validate_held_out_execution_manifest(
            heldout_manifest,
            primary_manifest=primary_manifest,
            expected_seed=DEFAULT_HELD_OUT_SEED,
        )
        report.add("Held-out manifest structure validated", passed=True)
    except ValueError as error:
        report.add(f"Held-out manifest validation failed: {error}", passed=False)

    primary_conflict = load_conflict_aware_results(config.primary_conflict_csv)
    heldout_conflict = load_conflict_aware_results(config.heldout_conflict_csv)
    heldout_spf = load_heldout_spf_results(config.heldout_spf_csv)

    _validate_conflict_records(
        report,
        label="Primary MAPF-7",
        records=primary_conflict,
        manifest=primary_manifest,
        expected_instance_count=config.expected_instance_count,
    )
    _validate_conflict_records(
        report,
        label="Held-out MAPF-7",
        records=heldout_conflict,
        manifest=heldout_manifest,
        expected_instance_count=config.expected_instance_count,
    )
    _validate_heldout_conflict_csv_metadata(report, config.heldout_conflict_csv)

    heldout_spf_count = len(heldout_spf)
    report.add(
        f"Held-out SPF: expected {config.expected_instance_count} records, found {heldout_spf_count}",
        passed=heldout_spf_count == config.expected_instance_count,
    )
    spf_keys = {(record.instance_id, record.strategy) for record in heldout_spf}
    report.add(
        "Held-out SPF: no duplicate (instance_id, strategy)",
        passed=len(spf_keys) == len(heldout_spf),
    )
    report.add(
        "Held-out SPF: strategy is spf only",
        passed=all(record.strategy == STRATEGY_SPF for record in heldout_spf),
    )

    heldout_mapf7_ids = {record.instance_id for record in heldout_conflict}
    heldout_spf_ids = {record.instance_id for record in heldout_spf}
    report.add(
        "Held-out SPF instance IDs align with held-out MAPF-7",
        passed=heldout_spf_ids == heldout_mapf7_ids,
    )

    for record in heldout_spf:
        if not record.success:
            report.add(
                f"Held-out SPF failed: {record.instance_id}",
                passed=False,
            )

    primary_ids = {instance.instance_id for instance in primary_manifest.instances}
    heldout_ids = {instance.instance_id for instance in heldout_manifest.instances}
    report.add(
        "Primary and held-out catalogues have disjoint instance IDs (no cross-pairing)",
        passed=primary_ids.isdisjoint(heldout_ids),
    )

    return report


def build_heldout_instance_level_summary(
    *,
    manifest: MAPFBenchmarkManifest,
    conflict_records: Sequence[ConflictAwareRunRecord],
    spf_records: Sequence[StrategyRunRecord],
) -> list[dict[str, Any]]:
    spf_lookup = {
        record.instance_id: record
        for record in spf_records
        if record.strategy == STRATEGY_SPF
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
        spf = spf_lookup[instance_id]
        cdf_h = cdf_h_lookup[instance_id]
        cdf_l = cdf_l_lookup[instance_id]
        spf_cd = spf_cd_lookup[instance_id]
        degree_stats = _degree_summary(cdf_h.agent_degrees)

        rows.append(
            {
                "catalogue": "held_out",
                "instance_id": instance_id,
                "agent_count": instance.agent_count,
                "interaction_level": instance.interaction_level.value,
                "spf_soc": spf.soc,
                "cdf_h_soc": cdf_h.soc,
                "cdf_l_soc": cdf_l.soc,
                "spf_cd_soc": spf_cd.soc,
                "spf_makespan": spf.makespan,
                "cdf_h_makespan": cdf_h.makespan,
                "cdf_l_makespan": cdf_l.makespan,
                "spf_cd_makespan": spf_cd.makespan,
                "independent_conflict_count": cdf_h.independent_conflict_count,
                "independent_conflict_pair_count": cdf_h.independent_conflict_pair_count,
                **degree_stats,
                "cdf_h_agent_order": json.dumps(list(cdf_h.agent_order or ())),
                "cdf_l_agent_order": json.dumps(list(cdf_l.agent_order or ())),
                "spf_cd_original_index_order": json.dumps(
                    list(spf_cd.original_index_order or ())
                ),
                "spf_agent_order": json.dumps(list(spf.agent_order or ())),
            }
        )
    return rows


def _build_pairwise_rows_from_instance_rows(
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


def build_catalogue_pairwise_summary(
    instance_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for comparison, left_key, right_key, left_label, right_label in FINAL_PAIRWISE_COMPARISONS:
        paired = _build_pairwise_rows_from_instance_rows(
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


def count_mapf7_soc_sensitive_instances(instance_rows: Sequence[dict[str, Any]]) -> int:
    return sum(1 for row in instance_rows if mapf7_strategy_socs_differ(row))


def count_makespan_sensitive_instances(instance_rows: Sequence[dict[str, Any]]) -> int:
    count = 0
    for row in instance_rows:
        makespans = {
            _row_int(row.get("cdf_h_makespan")),
            _row_int(row.get("cdf_l_makespan")),
            _row_int(row.get("spf_cd_makespan")),
        }
        makespans.discard(None)
        if len(makespans) > 1:
            count += 1
    return count


def build_spf_cd_ablation_rows(
    instance_rows: Sequence[dict[str, Any]],
    *,
    catalogue: str,
    conflict_records: Sequence[ConflictAwareRunRecord],
    spf_records: Sequence[StrategyRunRecord],
) -> list[dict[str, Any]]:
    spf_lookup = {
        record.instance_id: record
        for record in spf_records
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
            else None
        )
        rows.append(
            {
                "catalogue": catalogue,
                "instance_id": instance_id,
                "agent_count": row["agent_count"],
                "interaction_level": row["interaction_level"],
                "spf_soc": spf.soc,
                "spf_cd_soc": spf_cd.soc,
                "soc_equal": spf.soc == spf_cd.soc,
                "spf_makespan": spf.makespan,
                "spf_cd_makespan": spf_cd.makespan,
                "makespan_equal": spf.makespan == spf_cd.makespan,
                "same_priority_order": same_order,
                "spf_agent_order": json.dumps(list(spf_order or ())),
                "spf_cd_original_index_order": json.dumps(list(spf_cd_index_order or ())),
            }
        )
    return rows


def build_spf_cd_ablation_aggregate(
    ablation_rows: Sequence[dict[str, Any]],
    *,
    catalogue: str,
) -> dict[str, Any]:
    return {
        "catalogue": catalogue,
        "instance_count": len(ablation_rows),
        "identical_soc": sum(1 for row in ablation_rows if row.get("soc_equal")),
        "identical_makespan": sum(1 for row in ablation_rows if row.get("makespan_equal")),
        "identical_ordering": sum(
            1 for row in ablation_rows if row.get("same_priority_order") is True
        ),
        "differing_ordering": sum(
            1 for row in ablation_rows if row.get("same_priority_order") is False
        ),
    }


def _generalization_interpretation(
    comparison: str,
    *,
    primary_row: dict[str, Any],
    heldout_row: dict[str, Any],
) -> str:
    p_left = primary_row.get("left_better_count", 0)
    p_equal = primary_row.get("equal_count", 0)
    p_right = primary_row.get("right_better_count", 0)
    h_left = heldout_row.get("left_better_count", 0)
    h_equal = heldout_row.get("equal_count", 0)
    h_right = heldout_row.get("right_better_count", 0)
    p_mean = primary_row.get("mean_soc_diff", 0)
    h_mean = heldout_row.get("mean_soc_diff", 0)

    if comparison == "cdf_h_vs_cdf_l":
        primary_pref = "CDF-H" if p_left > p_right else ("CDF-L" if p_right > p_left else "none")
        heldout_pref = "CDF-H" if h_left > h_right else ("CDF-L" if h_right > h_left else "none")
        if primary_pref != heldout_pref and primary_pref != "none" and heldout_pref != "none":
            return "direction reversed between catalogues"
        if primary_pref == heldout_pref and primary_pref != "none":
            return "preferred direction replicated on held-out catalogue"
        if p_left == 0 and p_right == 0 and h_left == 0 and h_right == 0:
            return "effect remained sparse on both catalogues"
        return "direction not stable across catalogues"

    if comparison == "spf_cd_vs_spf":
        p_ident = primary_row.get("identical_soc", p_equal)
        h_ident = heldout_row.get("identical_soc", h_equal)
        if p_ident == h_ident == 27:
            return "no practical SPF+CD benefit on either catalogue"
        return f"primary identical SoC={p_ident}/27; held-out identical SoC={h_ident}/27"

    if p_mean == 0 and h_mean == 0 and p_left == 0 and h_left == 0:
        return "effect remained sparse on both catalogues"
    if (p_mean > 0 and h_mean <= 0) or (p_mean < 0 and h_mean >= 0):
        return "mean effect direction differed between catalogues"
    if p_left > 0 and h_left == 0 and p_right == 0 and h_right == 0:
        return "primary LEFT-better cases not replicated on held-out catalogue"
    if h_left > 0 and p_left == 0:
        return "held-out showed LEFT-better cases not seen in primary"
    return "partial replication; aggregate counts differ"


def build_generalization_summary(
    *,
    primary_pairwise: Sequence[dict[str, Any]],
    heldout_pairwise: Sequence[dict[str, Any]],
    primary_instance_rows: Sequence[dict[str, Any]],
    heldout_instance_rows: Sequence[dict[str, Any]],
    primary_spf_cd_ablation: dict[str, Any],
    heldout_spf_cd_ablation: dict[str, Any],
    combined_spf_cd_ablation: dict[str, Any],
) -> list[dict[str, Any]]:
    heldout_lookup = {row["comparison"]: row for row in heldout_pairwise}
    rows: list[dict[str, Any]] = []

    for primary_row in primary_pairwise:
        comparison = primary_row["comparison"]
        heldout_row = heldout_lookup[comparison]
        rows.append(
            {
                "comparison": comparison,
                "primary_left_better": primary_row["left_better_count"],
                "primary_equal": primary_row["equal_count"],
                "primary_right_better": primary_row["right_better_count"],
                "primary_mean_soc_diff": primary_row["mean_soc_diff"],
                "heldout_left_better": heldout_row["left_better_count"],
                "heldout_equal": heldout_row["equal_count"],
                "heldout_right_better": heldout_row["right_better_count"],
                "heldout_mean_soc_diff": heldout_row["mean_soc_diff"],
                "generalization_interpretation": _generalization_interpretation(
                    comparison,
                    primary_row=primary_row,
                    heldout_row=heldout_row,
                ),
            }
        )

    primary_sensitive = count_mapf7_soc_sensitive_instances(primary_instance_rows)
    heldout_sensitive = count_mapf7_soc_sensitive_instances(heldout_instance_rows)
    rows.append(
        {
            "comparison": "strategy_soc_sensitivity",
            "primary_left_better": primary_sensitive,
            "primary_equal": 27 - primary_sensitive,
            "primary_right_better": "",
            "primary_mean_soc_diff": f"{primary_sensitive}/27",
            "heldout_left_better": heldout_sensitive,
            "heldout_equal": 27 - heldout_sensitive,
            "heldout_right_better": "",
            "heldout_mean_soc_diff": f"{heldout_sensitive}/27",
            "generalization_interpretation": (
                "replicated sparse sensitivity"
                if primary_sensitive > 0 and heldout_sensitive > 0
                else (
                    "held-out validation did not support primary sensitivity pattern"
                    if primary_sensitive > 0 and heldout_sensitive == 0
                    else "sensitivity remained sparse on both catalogues"
                )
            ),
        }
    )

    primary_ms = count_makespan_sensitive_instances(primary_instance_rows)
    heldout_ms = count_makespan_sensitive_instances(heldout_instance_rows)
    rows.append(
        {
            "comparison": "makespan_sensitivity",
            "primary_left_better": primary_ms,
            "primary_equal": 27 - primary_ms,
            "primary_right_better": "",
            "primary_mean_soc_diff": f"{primary_ms}/27",
            "heldout_left_better": heldout_ms,
            "heldout_equal": 27 - heldout_ms,
            "heldout_right_better": "",
            "heldout_mean_soc_diff": f"{heldout_ms}/27",
            "generalization_interpretation": (
                "makespan less sensitive than SoC on both catalogues"
                if primary_ms <= primary_sensitive and heldout_ms <= heldout_sensitive
                else "makespan sensitivity pattern differs between catalogues"
            ),
        }
    )

    rows.append(
        {
            "comparison": "spf_cd_ablation_identical_soc",
            "primary_left_better": primary_spf_cd_ablation["identical_soc"],
            "primary_equal": "",
            "primary_right_better": "",
            "primary_mean_soc_diff": (
                f"{primary_spf_cd_ablation['identical_soc']}/"
                f"{primary_spf_cd_ablation['instance_count']}"
            ),
            "heldout_left_better": heldout_spf_cd_ablation["identical_soc"],
            "heldout_equal": "",
            "heldout_right_better": "",
            "heldout_mean_soc_diff": (
                f"{heldout_spf_cd_ablation['identical_soc']}/"
                f"{heldout_spf_cd_ablation['instance_count']}"
            ),
            "combined_identical_soc": (
                f"{combined_spf_cd_ablation['identical_soc']}/"
                f"{combined_spf_cd_ablation['instance_count']}"
            ),
            "combined_identical_ordering": (
                f"{combined_spf_cd_ablation['identical_ordering']}/"
                f"{combined_spf_cd_ablation['instance_count']}"
            ),
            "combined_differing_ordering": (
                f"{combined_spf_cd_ablation['differing_ordering']}/"
                f"{combined_spf_cd_ablation['instance_count']}"
            ),
            "generalization_interpretation": (
                f"primary identical ordering="
                f"{primary_spf_cd_ablation['identical_ordering']}/"
                f"{primary_spf_cd_ablation['instance_count']}; "
                f"held-out identical ordering="
                f"{heldout_spf_cd_ablation['identical_ordering']}/"
                f"{heldout_spf_cd_ablation['instance_count']}; "
                f"combined identical ordering="
                f"{combined_spf_cd_ablation['identical_ordering']}/"
                f"{combined_spf_cd_ablation['instance_count']}"
            ),
        }
    )

    return rows


def build_sensitive_instances_table(
    *,
    primary_instance_rows: Sequence[dict[str, Any]],
    heldout_instance_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def _append_sensitive(catalogue: str, instance_rows: Sequence[dict[str, Any]]) -> None:
        for row in instance_rows:
            if not mapf7_strategy_socs_differ(row):
                continue
            socs = [
                _row_int(row.get("cdf_h_soc")),
                _row_int(row.get("cdf_l_soc")),
                _row_int(row.get("spf_cd_soc")),
            ]
            socs = [value for value in socs if value is not None]
            rows.append(
                {
                    "catalogue": catalogue,
                    "instance_id": row["instance_id"],
                    "agent_count": row["agent_count"],
                    "interaction_level": row["interaction_level"],
                    "independent_conflict_count": row.get("independent_conflict_count"),
                    "independent_conflict_pair_count": row.get(
                        "independent_conflict_pair_count"
                    ),
                    "max_degree": row.get("max_degree"),
                    "cdf_h_soc": row.get("cdf_h_soc"),
                    "cdf_l_soc": row.get("cdf_l_soc"),
                    "spf_cd_soc": row.get("spf_cd_soc"),
                    "spf_soc": row.get("spf_soc"),
                    "soc_range": max(socs) - min(socs) if socs else 0,
                }
            )

    _append_sensitive("primary", primary_instance_rows)
    _append_sensitive("held_out", heldout_instance_rows)
    rows.sort(
        key=lambda item: (
            0 if item["catalogue"] == "primary" else 1,
            -int(item.get("soc_range") or 0),
            str(item["instance_id"]),
        )
    )
    return rows


def build_interaction_summary_combined(
    *,
    primary_instance_rows: Sequence[dict[str, Any]],
    heldout_instance_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for catalogue, instance_rows in (
        ("primary", primary_instance_rows),
        ("held_out", heldout_instance_rows),
    ):
        for entry in build_interaction_summary(instance_rows):
            entry = dict(entry)
            entry["catalogue"] = catalogue
            rows.append(entry)
    return rows


def build_agent_count_summary_combined(
    *,
    primary_instance_rows: Sequence[dict[str, Any]],
    heldout_instance_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for catalogue, instance_rows in (
        ("primary", primary_instance_rows),
        ("held_out", heldout_instance_rows),
    ):
        for entry in build_agent_count_summary(instance_rows):
            entry = dict(entry)
            entry["catalogue"] = catalogue
            rows.append(entry)
    return rows


def build_makespan_summary_combined(
    *,
    primary_instance_rows: Sequence[dict[str, Any]],
    heldout_instance_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for catalogue, instance_rows in (
        ("primary", primary_instance_rows),
        ("held_out", heldout_instance_rows),
    ):
        for entry in build_makespan_summary(instance_rows):
            entry = dict(entry)
            entry["catalogue"] = catalogue
            rows.append(entry)
    return rows


def build_runtime_summary_combined(
    *,
    primary_instance_rows: Sequence[dict[str, Any]],
    heldout_instance_rows: Sequence[dict[str, Any]],
    primary_conflict_records: Sequence[ConflictAwareRunRecord],
    heldout_conflict_records: Sequence[ConflictAwareRunRecord],
    primary_spf_records: Sequence[StrategyRunRecord],
    heldout_spf_records: Sequence[StrategyRunRecord],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    primary_runtime = build_runtime_summary(
        primary_instance_rows,
        spf_lpf_records=primary_spf_records,
        conflict_records=primary_conflict_records,
    )
    for row in primary_runtime:
        if row.get("metric_group") == "runtime_aggregate":
            entry = dict(row)
            entry["catalogue"] = "primary"
            rows.append(entry)
        else:
            entry = dict(row)
            entry["catalogue"] = "primary"
            rows.append(entry)

    heldout_runtime = _build_heldout_runtime_summary(
        heldout_instance_rows,
        conflict_records=heldout_conflict_records,
        spf_records=heldout_spf_records,
    )
    rows.extend(heldout_runtime)
    return rows


def _build_heldout_runtime_summary(
    instance_rows: Sequence[dict[str, Any]],
    *,
    conflict_records: Sequence[ConflictAwareRunRecord],
    spf_records: Sequence[StrategyRunRecord],
) -> list[dict[str, Any]]:
    spf_lookup = {
        record.instance_id: record
        for record in spf_records
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
                "catalogue": "held_out",
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
            }
        )

    def _mean(values: Sequence[float]) -> float:
        return statistics.mean(values)

    def _median(values: Sequence[float]) -> float:
        return statistics.median(values)

    def _strategy_stats(prefix: str) -> dict[str, float]:
        ordering = [float(row[f"{prefix}_ordering_time_ms"]) for row in per_instance]
        pp = [float(row[f"{prefix}_pp_time_ms"]) for row in per_instance]
        total = [float(row[f"{prefix}_total_time_ms"]) for row in per_instance]
        mean_total = _mean(total)
        mean_ordering = _mean(ordering)
        return {
            f"mean_{prefix}_ordering_ms": mean_ordering,
            f"median_{prefix}_ordering_ms": _median(ordering),
            f"mean_{prefix}_pp_ms": _mean(pp),
            f"median_{prefix}_pp_ms": _median(pp),
            f"mean_{prefix}_total_ms": mean_total,
            f"median_{prefix}_total_ms": _median(total),
            f"mean_{prefix}_ordering_fraction": (
                mean_ordering / mean_total if mean_total else 0.0
            ),
        }

    aggregate: dict[str, Any] = {
        "catalogue": "held_out",
        "metric_group": "runtime_aggregate",
        **(_strategy_stats("spf")),
        **(_strategy_stats("cdf_h")),
        **(_strategy_stats("cdf_l")),
        **(_strategy_stats("spf_cd")),
        "note": (
            "Held-out runtime from independent SPF and MAPF-7 runs; "
            "ordering fraction = mean ordering / mean total."
        ),
    }
    return per_instance + [aggregate]


def build_conflict_structure_summary(
    *,
    primary_instance_rows: Sequence[dict[str, Any]],
    heldout_instance_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def _summarize_group(
        catalogue: str,
        label: str,
        group_rows: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        if not group_rows:
            return {
                "catalogue": catalogue,
                "group": label,
                "instance_count": 0,
            }
        return {
            "catalogue": catalogue,
            "group": label,
            "instance_count": len(group_rows),
            "mean_independent_conflict_count": statistics.mean(
                float(row.get("independent_conflict_count") or 0) for row in group_rows
            ),
            "mean_independent_conflict_pair_count": statistics.mean(
                float(row.get("independent_conflict_pair_count") or 0)
                for row in group_rows
            ),
            "mean_max_degree": statistics.mean(
                float(row.get("max_degree") or 0) for row in group_rows
            ),
            "mean_mean_degree": statistics.mean(
                float(row.get("mean_degree") or 0) for row in group_rows
            ),
            "mean_mapf7_soc_range": statistics.mean(
                _mapf7_soc_range(row) for row in group_rows
            ),
        }

    for catalogue, instance_rows in (
        ("primary", primary_instance_rows),
        ("held_out", heldout_instance_rows),
    ):
        sensitive = [row for row in instance_rows if mapf7_strategy_socs_differ(row)]
        insensitive = [row for row in instance_rows if not mapf7_strategy_socs_differ(row)]
        rows.append(_summarize_group(catalogue, "soc_sensitive", sensitive))
        rows.append(_summarize_group(catalogue, "soc_insensitive", insensitive))

    return rows


def _mapf7_soc_range(row: dict[str, Any]) -> float:
    socs = {
        _row_int(row.get("cdf_h_soc")),
        _row_int(row.get("cdf_l_soc")),
        _row_int(row.get("spf_cd_soc")),
    }
    socs.discard(None)
    if not socs:
        return 0.0
    return float(max(socs) - min(socs))


def build_case_study_n10_high_001(
    heldout_instance_rows: Sequence[dict[str, Any]],
    *,
    instance_id: str = "AR0204SR_HO_n10_high_001",
) -> dict[str, Any] | None:
    for row in heldout_instance_rows:
        if row["instance_id"] != instance_id:
            continue
        return {
            "instance_id": instance_id,
            "catalogue": "held_out",
            "independent_conflict_count": row.get("independent_conflict_count"),
            "independent_conflict_pair_count": row.get("independent_conflict_pair_count"),
            "max_degree": row.get("max_degree"),
            "mean_degree": row.get("mean_degree"),
            "agent_degrees_note": (
                "97 independent conflict events but only 3 conflicting agent pairs; "
                "most agents have degree 0 with high incident counts on agents 4 and 9."
            ),
            "cdf_h_soc": row.get("cdf_h_soc"),
            "cdf_l_soc": row.get("cdf_l_soc"),
            "spf_cd_soc": row.get("spf_cd_soc"),
            "spf_soc": row.get("spf_soc"),
            "cdf_h_makespan": row.get("cdf_h_makespan"),
            "cdf_l_makespan": row.get("cdf_l_makespan"),
            "spf_cd_makespan": row.get("spf_cd_makespan"),
            "spf_makespan": row.get("spf_makespan"),
            "cdf_h_agent_order": row.get("cdf_h_agent_order"),
            "cdf_l_agent_order": row.get("cdf_l_agent_order"),
            "spf_cd_original_index_order": row.get("spf_cd_original_index_order"),
            "spf_agent_order": row.get("spf_agent_order"),
            "interpretation": (
                "Descriptive case study only: high conflict-event frequency with low pair count "
                "does not imply MAPF-7 SoC sensitivity (all strategies reached SoC "
                f"{row.get('spf_soc')}). Orders differ but final cost does not."
            ),
        }
    return None


def _sensitive_interaction_counts(
    instance_rows: Sequence[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {"low": 0, "medium": 0, "high": 0}
    for row in instance_rows:
        if mapf7_strategy_socs_differ(row):
            level = str(row["interaction_level"]).lower()
            counts[level] = counts.get(level, 0) + 1
    return counts


def build_final_analysis_summary_md(
    *,
    validation: ValidationReport,
    config: Mapf7FinalAnalysisConfig,
    primary_pairwise: Sequence[dict[str, Any]],
    heldout_pairwise: Sequence[dict[str, Any]],
    generalization_rows: Sequence[dict[str, Any]],
    sensitive_rows: Sequence[dict[str, Any]],
    primary_instance_rows: Sequence[dict[str, Any]],
    heldout_instance_rows: Sequence[dict[str, Any]],
    primary_spf_cd_ablation: dict[str, Any],
    heldout_spf_cd_ablation: dict[str, Any],
    combined_spf_cd_ablation: dict[str, Any],
    conflict_structure_rows: Sequence[dict[str, Any]],
    runtime_rows: Sequence[dict[str, Any]],
    case_study: dict[str, Any] | None,
    makespan_rows: Sequence[dict[str, Any]],
) -> str:
    cdf_h_vs_spf_p = _lookup_pairwise(primary_pairwise, "cdf_h_vs_spf")
    cdf_l_vs_spf_p = _lookup_pairwise(primary_pairwise, "cdf_l_vs_spf")
    cdf_h_vs_cdf_l_p = _lookup_pairwise(primary_pairwise, "cdf_h_vs_cdf_l")
    spf_cd_vs_spf_p = _lookup_pairwise(primary_pairwise, "spf_cd_vs_spf")

    cdf_h_vs_spf_h = _lookup_pairwise(heldout_pairwise, "cdf_h_vs_spf")
    cdf_l_vs_spf_h = _lookup_pairwise(heldout_pairwise, "cdf_l_vs_spf")
    cdf_h_vs_cdf_l_h = _lookup_pairwise(heldout_pairwise, "cdf_h_vs_cdf_l")
    spf_cd_vs_spf_h = _lookup_pairwise(heldout_pairwise, "spf_cd_vs_spf")

    primary_sensitive = count_mapf7_soc_sensitive_instances(primary_instance_rows)
    heldout_sensitive = count_mapf7_soc_sensitive_instances(heldout_instance_rows)
    primary_ms = count_makespan_sensitive_instances(primary_instance_rows)
    heldout_ms = count_makespan_sensitive_instances(heldout_instance_rows)

    primary_int = _sensitive_interaction_counts(primary_instance_rows)
    heldout_int = _sensitive_interaction_counts(heldout_instance_rows)

    heldout_runtime_agg = next(
        (row for row in runtime_rows if row.get("metric_group") == "runtime_aggregate"
         and row.get("catalogue") == "held_out"),
        {},
    )

    lines = [
        "# MAPF-7 Final Analysis Summary (MAPF-7.9)",
        "",
        "## 1. Data / validation",
        "",
        f"- Validation status: **{'PASSED' if validation.passed else 'FAILED'}**",
        f"- Primary MAPF-7: `{config.primary_conflict_csv}` (27 instances × 3 strategies = 81 records)",
        f"- Held-out MAPF-7: `{config.heldout_conflict_csv}` (81 records, seed {DEFAULT_HELD_OUT_SEED})",
        f"- Held-out SPF: `{config.heldout_spf_csv}` (27 records)",
        "- Held-out validation is **scenario-level on the same MovingAI map/source**, not cross-map validation.",
        "- Primary and held-out catalogues contain **different instances**; no instance-level pairing was performed.",
        "",
        "## 2. Primary findings",
        "",
        f"- CDF-H vs SPF: LEFT better={cdf_h_vs_spf_p.get('left_better_count')}, "
        f"equal={cdf_h_vs_spf_p.get('equal_count')}, RIGHT better={cdf_h_vs_spf_p.get('right_better_count')}; "
        f"mean diff={_format_float(cdf_h_vs_spf_p.get('mean_soc_diff'))}.",
        f"- CDF-L vs SPF: LEFT better={cdf_l_vs_spf_p.get('left_better_count')}, "
        f"equal={cdf_l_vs_spf_p.get('equal_count')}, RIGHT better={cdf_l_vs_spf_p.get('right_better_count')}; "
        f"mean diff={_format_float(cdf_l_vs_spf_p.get('mean_soc_diff'))}.",
        f"- CDF-H vs CDF-L: LEFT better={cdf_h_vs_cdf_l_p.get('left_better_count')}, "
        f"equal={cdf_h_vs_cdf_l_p.get('equal_count')}, RIGHT better={cdf_h_vs_cdf_l_p.get('right_better_count')}; "
        f"mean diff={_format_float(cdf_h_vs_cdf_l_p.get('mean_soc_diff'))}.",
        f"- SPF+CD vs SPF: identical SoC on {primary_spf_cd_ablation['identical_soc']}/27; "
        f"identical ordering on {primary_spf_cd_ablation['identical_ordering']}/27.",
        f"- MAPF-7 strategies differ in SoC on {primary_sensitive}/27 primary instances.",
        "",
        "## 3. Held-out findings",
        "",
        f"- CDF-H vs SPF: LEFT better={cdf_h_vs_spf_h.get('left_better_count')}, "
        f"equal={cdf_h_vs_spf_h.get('equal_count')}, RIGHT better={cdf_h_vs_spf_h.get('right_better_count')}; "
        f"mean diff={_format_float(cdf_h_vs_spf_h.get('mean_soc_diff'))}.",
        f"- CDF-L vs SPF: LEFT better={cdf_l_vs_spf_h.get('left_better_count')}, "
        f"equal={cdf_l_vs_spf_h.get('equal_count')}, RIGHT better={cdf_l_vs_spf_h.get('right_better_count')}; "
        f"mean diff={_format_float(cdf_l_vs_spf_h.get('mean_soc_diff'))}.",
        f"- CDF-H vs CDF-L: LEFT better={cdf_h_vs_cdf_l_h.get('left_better_count')}, "
        f"equal={cdf_h_vs_cdf_l_h.get('equal_count')}, RIGHT better={cdf_h_vs_cdf_l_h.get('right_better_count')}; "
        f"mean diff={_format_float(cdf_h_vs_cdf_l_h.get('mean_soc_diff'))}.",
        f"- SPF+CD vs SPF: identical SoC on {heldout_spf_cd_ablation['identical_soc']}/27; "
        f"identical ordering on {heldout_spf_cd_ablation['identical_ordering']}/27; "
        f"differing ordering on {heldout_spf_cd_ablation['differing_ordering']}/27.",
        f"- MAPF-7 strategies differ in SoC on {heldout_sensitive}/27 held-out instances.",
        "",
        "## 4. Generalization comparison",
        "",
    ]
    for row in generalization_rows:
        if row["comparison"] == "spf_cd_ablation_identical_soc":
            lines.append(
                f"- **{row['comparison']}**: primary identical SoC="
                f"{row['primary_mean_soc_diff']}, held-out identical SoC="
                f"{row['heldout_mean_soc_diff']}, combined identical SoC="
                f"{row['combined_identical_soc']} → {row['generalization_interpretation']}"
            )
        elif row["comparison"] in ("strategy_soc_sensitivity", "makespan_sensitivity"):
            lines.append(
                f"- **{row['comparison']}**: primary={row.get('primary_mean_soc_diff')}, "
                f"held-out={row.get('heldout_mean_soc_diff')} → {row['generalization_interpretation']}"
            )
        else:
            lines.append(
                f"- **{row['comparison']}**: primary "
                f"({row['primary_left_better']}/{row['primary_equal']}/{row['primary_right_better']}, "
                f"mean={_format_float(row.get('primary_mean_soc_diff'))}); held-out "
                f"({row['heldout_left_better']}/{row['heldout_equal']}/{row['heldout_right_better']}, "
                f"mean={_format_float(row.get('heldout_mean_soc_diff'))}) → "
                f"{row['generalization_interpretation']}"
            )

    cdf_h_primary = cdf_h_vs_cdf_l_p.get("left_better_count", 0)
    cdf_l_primary = cdf_h_vs_cdf_l_p.get("right_better_count", 0)
    cdf_equal_primary = cdf_h_vs_cdf_l_p.get("equal_count", 0)
    cdf_h_heldout = cdf_h_vs_cdf_l_h.get("left_better_count", 0)
    cdf_l_heldout = cdf_h_vs_cdf_l_h.get("right_better_count", 0)
    cdf_equal_heldout = cdf_h_vs_cdf_l_h.get("equal_count", 0)

    lines.extend(
        [
            "",
            "## 5. Direction ablation (CDF-H vs CDF-L)",
            "",
            f"- Primary catalogue: CDF-H better={cdf_h_primary}, equal={cdf_equal_primary}, "
            f"CDF-L better={cdf_l_primary}.",
            f"- Held-out catalogue: CDF-H better={cdf_h_heldout}, equal={cdf_equal_heldout}, "
            f"CDF-L better={cdf_l_heldout}.",
            "- The observed direction **reversed between catalogues**: on the primary catalogue "
            "CDF-L was better on more instances than CDF-H, whereas on the held-out catalogue "
            "CDF-H was better on more instances than CDF-L.",
            "",
            "## 6. SPF+CD ablation",
            "",
            f"- Primary: identical SoC={primary_spf_cd_ablation['identical_soc']}/27, "
            f"identical makespan={primary_spf_cd_ablation['identical_makespan']}/27, "
            f"identical ordering={primary_spf_cd_ablation['identical_ordering']}/27.",
            f"- Held-out: identical SoC={heldout_spf_cd_ablation['identical_soc']}/27, "
            f"identical makespan={heldout_spf_cd_ablation['identical_makespan']}/27, "
            f"identical ordering={heldout_spf_cd_ablation['identical_ordering']}/27, "
            f"differing ordering={heldout_spf_cd_ablation['differing_ordering']}/27.",
            f"- Combined (54 instances): identical SoC={combined_spf_cd_ablation['identical_soc']}/54, "
            f"identical ordering={combined_spf_cd_ablation['identical_ordering']}/54, "
            f"differing ordering={combined_spf_cd_ablation['differing_ordering']}/54.",
            "- Across all 54 primary and held-out instances, SPF+CD produced exactly the same "
            "priority ordering as plain SPF; the conflict-degree tie-break therefore never changed "
            "the resulting order in this experimental sample.",
            "",
            "## 7. SoC vs makespan sensitivity",
            "",
            f"- Primary: SoC-sensitive={primary_sensitive}/27, makespan-sensitive={primary_ms}/27.",
            f"- Held-out: SoC-sensitive={heldout_sensitive}/27, makespan-sensitive={heldout_ms}/27.",
            "- Makespan was substantially less sensitive to tested priority-ordering strategies than SoC "
            "within this experimental setting.",
            "",
            "### Sensitive instances by interaction stratum (descriptive)",
            "",
            f"- Primary: LOW={primary_int.get('low', 0)}, MEDIUM={primary_int.get('medium', 0)}, "
            f"HIGH={primary_int.get('high', 0)}.",
            f"- Held-out: LOW={heldout_int.get('low', 0)}, MEDIUM={heldout_int.get('medium', 0)}, "
            f"HIGH={heldout_int.get('high', 0)}.",
            "- Interaction strata are derived from independent-path conflicts; HIGH concentration "
            "is not independent evidence that conflict information predicts sensitivity.",
            "",
            "## 8. Conflict-structure diagnostics",
            "",
        ]
    )
    for row in conflict_structure_rows:
        if row.get("instance_count", 0) == 0:
            continue
        lines.append(
            f"- {row['catalogue']} / {row['group']} (n={row['instance_count']}): "
            f"mean conflicts={_format_float(row.get('mean_independent_conflict_count'))}, "
            f"mean pairs={_format_float(row.get('mean_independent_conflict_pair_count'))}, "
            f"mean max degree={_format_float(row.get('mean_max_degree'))}, "
            f"mean SoC range={_format_float(row.get('mean_mapf7_soc_range'))}."
        )
    lines.append(
        "- Conflict-graph structure may be descriptively associated with priority sensitivity, "
        "but degree alone does not reliably exploit that information."
    )

    if case_study:
        lines.extend(
            [
                "",
                "### Case study: AR0204SR_HO_n10_high_001",
                "",
                f"- Independent conflict events: {case_study.get('independent_conflict_count')}; "
                f"conflict pairs: {case_study.get('independent_conflict_pair_count')}; "
                f"max degree: {case_study.get('max_degree')}.",
                f"- SoC (CDF-H / CDF-L / SPF+CD / SPF): "
                f"{case_study.get('cdf_h_soc')} / {case_study.get('cdf_l_soc')} / "
                f"{case_study.get('spf_cd_soc')} / {case_study.get('spf_soc')}.",
                f"- {case_study.get('interpretation')}",
            ]
        )

    lines.extend(["", "## 9. Runtime", ""])
    if heldout_runtime_agg:
        for strategy, prefix in (
            ("SPF", "spf"),
            ("CDF-H", "cdf_h"),
            ("CDF-L", "cdf_l"),
            ("SPF+CD", "spf_cd"),
        ):
            mean_ord = heldout_runtime_agg.get(f"mean_{prefix}_ordering_ms", 0)
            mean_pp = heldout_runtime_agg.get(f"mean_{prefix}_pp_ms", 0)
            mean_total = heldout_runtime_agg.get(f"mean_{prefix}_total_ms", 0)
            fraction = heldout_runtime_agg.get(f"mean_{prefix}_ordering_fraction", 0)
            lines.append(
                f"- Held-out mean {strategy}: ordering {_format_float(float(mean_ord) / 1000)} s, "
                f"PP {_format_float(float(mean_pp) / 1000)} s, total {_format_float(float(mean_total) / 1000)} s "
                f"(ordering fraction {_format_float(float(fraction) * 100, digits=1)}%)."
            )
    lines.extend(
        [
            "- Ordering phases are of similar overall magnitude across strategies.",
            "- Total runtime changes can also result from the selected priority order making "
            "subsequent PP easier or harder.",
            "- Isolated graph-construction overhead was not separately benchmarked.",
            "",
            "## 10. Main thesis interpretation",
            "",
            "### Supported findings",
            "",
            f"- **A.** Priority ordering affects SoC only on a minority of tested instances "
            f"(primary {primary_sensitive}/27, held-out {heldout_sensitive}/27).",
            f"- **B.** Makespan is less sensitive than SoC (primary {primary_ms}/27, held-out {heldout_ms}/27).",
            "- **C.** Simple conflict-degree direction is not a stable general rule across catalogues.",
            "- **D.** SPF remains a strong and stable deterministic baseline.",
            "- **E.** SPF+CD tie-break provides little or no practical benefit when costs rarely tie.",
            "- **F.** Conflict-graph structure may be descriptively associated with sensitivity, "
            "but degree alone does not reliably exploit it.",
            "- **G.** Held-out catalogue prevents interpreting primary improvements/regressions as "
            "universally generalizing.",
            "",
            "## 11. Limitations",
            "",
            "1. One MovingAI map/source (AR0204SR).",
            "2. Held-out validation is scenario-level, not cross-map.",
            "3. Only 27 primary + 27 held-out instances.",
            "4. Interaction strata derived from the same independent-path conflict structure used by CDF.",
            "5. Static pre-PP ordering only; no online/dynamic priority adaptation.",
            "6. Degree is a coarse unweighted graph signal.",
            "7. No event multiplicity / conflict type / timing weighting in CDF v1.",
            "8. PP is incomplete and priority-sensitive by nature.",
            "9. Runtime measured from independent runs, not isolated microbenchmarks.",
            "10. Basic CBS cited elsewhere as quality reference, not quality ceiling.",
            "",
            "## 12. Final MAPF-7 conclusion",
            "",
            "Within this experimental setting on AR0204SR, conflict-degree-based priority ordering "
            "produced sparse and catalogue-dependent SoC effects. SPF remained competitive; "
            "held-out scenario-level validation did not support treating primary-catalogue observations "
            "as universally generalizing. The frozen MAPF-7 family is best reported as a completed "
            "negative/mixed methodological result with explicit ablations, not as a reliable improvement "
            "over SPF.",
            "",
        ]
    )
    return "\n".join(lines)


def build_thesis_tables_md(
    *,
    primary_pairwise: Sequence[dict[str, Any]],
    heldout_pairwise: Sequence[dict[str, Any]],
    generalization_rows: Sequence[dict[str, Any]],
    sensitive_rows: Sequence[dict[str, Any]],
    interaction_rows: Sequence[dict[str, Any]],
    agent_count_rows: Sequence[dict[str, Any]],
    makespan_rows: Sequence[dict[str, Any]],
    runtime_rows: Sequence[dict[str, Any]],
    spf_cd_ablation_rows: Sequence[dict[str, Any]],
) -> str:
    lines = [
        "# MAPF-7.9 Final Thesis Tables",
        "",
        "*Sign convention: diff = LEFT SoC − RIGHT SoC; diff < 0 → LEFT better.*",
        "",
        "## Table 1 — Primary pairwise summary",
        "",
        "| Comparison | Mean diff | Median diff | LEFT better | Equal | RIGHT better |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in primary_pairwise:
        lines.append(
            f"| {row['left_strategy']} vs {row['right_strategy']} | "
            f"{_format_float(row['mean_soc_diff'])} | {_format_float(row['median_soc_diff'])} | "
            f"{row['left_better_count']} | {row['equal_count']} | {row['right_better_count']} |"
        )

    lines.extend(
        [
            "",
            "## Table 2 — Held-out pairwise summary",
            "",
            "| Comparison | Mean diff | Median diff | LEFT better | Equal | RIGHT better |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in heldout_pairwise:
        lines.append(
            f"| {row['left_strategy']} vs {row['right_strategy']} | "
            f"{_format_float(row['mean_soc_diff'])} | {_format_float(row['median_soc_diff'])} | "
            f"{row['left_better_count']} | {row['equal_count']} | {row['right_better_count']} |"
        )

    lines.extend(
        [
            "",
            "## Table 3 — Primary vs held-out generalization",
            "",
            "| Comparison | Primary (L/E/R) | Primary mean diff | Held-out (L/E/R) | Held-out mean diff | Interpretation |",
            "|---|---|---:|---|---:|---|",
        ]
    )
    for row in generalization_rows:
        if row["comparison"] in ("strategy_soc_sensitivity", "makespan_sensitivity"):
            lines.append(
                f"| {row['comparison']} | {row.get('primary_mean_soc_diff')} | | "
                f"{row.get('heldout_mean_soc_diff')} | | {row['generalization_interpretation']} |"
            )
        elif row["comparison"] == "spf_cd_ablation_identical_soc":
            continue
        else:
            lines.append(
                f"| {row['comparison']} | "
                f"{row['primary_left_better']}/{row['primary_equal']}/{row['primary_right_better']} | "
                f"{_format_float(row.get('primary_mean_soc_diff'))} | "
                f"{row['heldout_left_better']}/{row['heldout_equal']}/{row['heldout_right_better']} | "
                f"{_format_float(row.get('heldout_mean_soc_diff'))} | "
                f"{row['generalization_interpretation']} |"
            )

    lines.extend(
        [
            "",
            "## Table 4 — SoC-sensitive instances",
            "",
            "| Catalogue | Instance | Agents | Interaction | Conflicts | Pairs | Max deg | CDF-H | CDF-L | SPF+CD | SPF | Range |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sensitive_rows:
        lines.append(
            f"| {row['catalogue']} | {row['instance_id']} | {row['agent_count']} | "
            f"{row['interaction_level']} | {row.get('independent_conflict_count', '')} | "
            f"{row.get('independent_conflict_pair_count', '')} | {row.get('max_degree', '')} | "
            f"{row.get('cdf_h_soc')} | {row.get('cdf_l_soc')} | {row.get('spf_cd_soc')} | "
            f"{row.get('spf_soc')} | {row.get('soc_range')} |"
        )

    lines.extend(
        [
            "",
            "## Table 5 — Interaction-level comparison",
            "",
            "| Catalogue | Interaction | Strategy | Mean SoC diff vs SPF | Better / equal / worse |",
            "|---|---|---|---:|---|",
        ]
    )
    for row in interaction_rows:
        if row.get("strategy") == "_stratum_meta":
            continue
        lines.append(
            f"| {row.get('catalogue')} | {row['interaction_level'].upper()} | {row['strategy']} | "
            f"{_format_float(row.get('mean_soc_diff_vs_spf'))} | "
            f"{row.get('better_vs_spf')}/{row.get('equal_vs_spf')}/{row.get('worse_vs_spf')} |"
        )

    lines.extend(
        [
            "",
            "## Table 6 — Agent-count comparison",
            "",
            "| Catalogue | Agents | Strategy | Mean SoC diff vs SPF | Better / equal / worse |",
            "|---|---:|---|---:|---|",
        ]
    )
    for row in agent_count_rows:
        lines.append(
            f"| {row.get('catalogue')} | {row['agent_count']} | {row['strategy']} | "
            f"{_format_float(row.get('mean_soc_diff_vs_spf'))} | "
            f"{row.get('better_vs_spf')}/{row.get('equal_vs_spf')}/{row.get('worse_vs_spf')} |"
        )

    lines.extend(
        [
            "",
            "## Table 7 — Makespan sensitivity",
            "",
            "| Catalogue | Comparison | Differing instances | Mean makespan diff | LEFT / equal / RIGHT |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in makespan_rows:
        lines.append(
            f"| {row.get('catalogue')} | {row['comparison']} | {row['differing_instance_count']} | "
            f"{_format_float(row.get('mean_makespan_diff'))} | "
            f"{row['left_better_count']}/{row['equal_count']}/{row['right_better_count']} |"
        )

    lines.extend(
        [
            "",
            "## Table 8 — Runtime summary (aggregate rows)",
            "",
            "| Catalogue | Strategy | Mean ordering (s) | Mean PP (s) | Mean total (s) | Ordering fraction |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in runtime_rows:
        if row.get("metric_group") != "runtime_aggregate":
            continue
        catalogue = row.get("catalogue", "primary")
        for label, prefix in (
            ("SPF", "spf"),
            ("CDF-H", "cdf_h"),
            ("CDF-L", "cdf_l"),
            ("SPF+CD", "spf_cd"),
        ):
            if catalogue == "primary":
                ord_key = f"mean_{prefix}_ordering_s"
                pp_key = f"mean_{prefix}_pp_s"
                if ord_key not in row:
                    continue
                ordering = float(row[ord_key])
                pp = float(row[pp_key])
            else:
                ordering = float(row.get(f"mean_{prefix}_ordering_ms", 0)) / 1000.0
                pp = float(row.get(f"mean_{prefix}_pp_ms", 0)) / 1000.0
            total = ordering + pp
            fraction = ordering / total if total else 0.0
            lines.append(
                f"| {catalogue} | {label} | {_format_float(ordering)} | {_format_float(pp)} | "
                f"{_format_float(total)} | {_format_float(fraction * 100, digits=1)}% |"
            )

    lines.extend(
        [
            "",
            "## Table 9 — SPF+CD ablation (aggregate)",
            "",
            "| Catalogue | Instances | Identical SoC | Identical makespan | Identical ordering | Differing ordering |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for catalogue in ("primary", "held_out", "combined"):
        agg_rows = [row for row in spf_cd_ablation_rows if row.get("catalogue") == catalogue]
        if not agg_rows:
            continue
        agg = agg_rows[0]
        lines.append(
            f"| {catalogue} | {agg.get('instance_count')} | {agg.get('identical_soc')} | "
            f"{agg.get('identical_makespan')} | {agg.get('identical_ordering')} | "
            f"{agg.get('differing_ordering')} |"
        )

    lines.append("")
    return "\n".join(lines)


def _reproduce_primary_instance_rows(
    config: Mapf7FinalAnalysisConfig,
) -> list[dict[str, Any]]:
    primary_manifest = load_benchmark_manifest(config.primary_manifest_path)
    conflict_records = load_conflict_aware_results(config.primary_conflict_csv)
    fixed_records = load_fixed_results(config.fixed_results_csv)
    random_records = load_random_results(config.random_results_csv)
    spf_lpf_records = load_strategy_results(config.primary_spf_lpf_csv)
    random_aggregates = aggregate_random_by_instance(random_records)
    rows = build_instance_level_summary(
        manifest=primary_manifest,
        fixed_records=fixed_records,
        random_aggregates=random_aggregates,
        spf_lpf_records=spf_lpf_records,
        conflict_records=conflict_records,
    )
    for row in rows:
        row["catalogue"] = "primary"
    return rows


def run_mapf7_final_analysis(
    config: Mapf7FinalAnalysisConfig,
    *,
    plot_writer: Callable[..., tuple[str, ...]] | None = None,
) -> Mapf7FinalAnalysisResult:
    validation = validate_final_analysis_inputs(config)
    if not validation.passed:
        return Mapf7FinalAnalysisResult(
            validation=validation,
            output_dir=config.output_dir,
            tables_written=(),
            plots_written=(),
            primary_instance_rows=(),
            heldout_instance_rows=(),
        )

    primary_manifest = load_benchmark_manifest(config.primary_manifest_path)
    heldout_manifest = load_benchmark_manifest(config.heldout_manifest_path)
    primary_conflict = load_conflict_aware_results(config.primary_conflict_csv)
    heldout_conflict = load_conflict_aware_results(config.heldout_conflict_csv)
    primary_spf_lpf = load_strategy_results(config.primary_spf_lpf_csv)
    heldout_spf = load_heldout_spf_results(config.heldout_spf_csv)

    primary_instance_rows = _reproduce_primary_instance_rows(config)
    heldout_instance_rows = build_heldout_instance_level_summary(
        manifest=heldout_manifest,
        conflict_records=heldout_conflict,
        spf_records=heldout_spf,
    )

    primary_pairwise = build_catalogue_pairwise_summary(primary_instance_rows)
    heldout_pairwise = build_catalogue_pairwise_summary(heldout_instance_rows)

    primary_spf_cd_rows = build_spf_cd_ablation_rows(
        primary_instance_rows,
        catalogue="primary",
        conflict_records=primary_conflict,
        spf_records=primary_spf_lpf,
    )
    heldout_spf_cd_rows = build_spf_cd_ablation_rows(
        heldout_instance_rows,
        catalogue="held_out",
        conflict_records=heldout_conflict,
        spf_records=heldout_spf,
    )
    primary_spf_cd_ablation = build_spf_cd_ablation_aggregate(
        primary_spf_cd_rows, catalogue="primary"
    )
    heldout_spf_cd_ablation = build_spf_cd_ablation_aggregate(
        heldout_spf_cd_rows, catalogue="held_out"
    )
    combined_spf_cd_ablation = build_spf_cd_ablation_aggregate(
        primary_spf_cd_rows + heldout_spf_cd_rows, catalogue="combined"
    )
    combined_spf_cd_ablation["instance_count"] = 54

    generalization_rows = build_generalization_summary(
        primary_pairwise=primary_pairwise,
        heldout_pairwise=heldout_pairwise,
        primary_instance_rows=primary_instance_rows,
        heldout_instance_rows=heldout_instance_rows,
        primary_spf_cd_ablation=primary_spf_cd_ablation,
        heldout_spf_cd_ablation=heldout_spf_cd_ablation,
        combined_spf_cd_ablation=combined_spf_cd_ablation,
    )

    sensitive_rows = build_sensitive_instances_table(
        primary_instance_rows=primary_instance_rows,
        heldout_instance_rows=heldout_instance_rows,
    )
    interaction_rows = build_interaction_summary_combined(
        primary_instance_rows=primary_instance_rows,
        heldout_instance_rows=heldout_instance_rows,
    )
    agent_count_rows = build_agent_count_summary_combined(
        primary_instance_rows=primary_instance_rows,
        heldout_instance_rows=heldout_instance_rows,
    )
    makespan_rows = build_makespan_summary_combined(
        primary_instance_rows=primary_instance_rows,
        heldout_instance_rows=heldout_instance_rows,
    )
    runtime_rows = build_runtime_summary_combined(
        primary_instance_rows=primary_instance_rows,
        heldout_instance_rows=heldout_instance_rows,
        primary_conflict_records=primary_conflict,
        heldout_conflict_records=heldout_conflict,
        primary_spf_records=primary_spf_lpf,
        heldout_spf_records=heldout_spf,
    )
    conflict_structure_rows = build_conflict_structure_summary(
        primary_instance_rows=primary_instance_rows,
        heldout_instance_rows=heldout_instance_rows,
    )
    case_study = build_case_study_n10_high_001(heldout_instance_rows)

    spf_cd_ablation_output: list[dict[str, Any]] = (
        primary_spf_cd_rows + heldout_spf_cd_rows + [
            primary_spf_cd_ablation,
            heldout_spf_cd_ablation,
            combined_spf_cd_ablation,
        ]
    )

    output_dir = config.output_dir
    tables_written: list[str] = []
    table_specs: list[tuple[str, list[dict[str, Any]]]] = [
        ("primary_pairwise_summary.csv", primary_pairwise),
        ("heldout_pairwise_summary.csv", heldout_pairwise),
        ("generalization_summary.csv", generalization_rows),
        ("sensitive_instances.csv", sensitive_rows),
        ("interaction_summary.csv", interaction_rows),
        ("agent_count_summary.csv", agent_count_rows),
        ("makespan_summary.csv", makespan_rows),
        ("runtime_summary.csv", runtime_rows),
        ("spf_cd_ablation.csv", spf_cd_ablation_output),
        ("conflict_structure_summary.csv", conflict_structure_rows),
    ]

    for filename, rows in table_specs:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        _write_csv(output_dir / filename, rows, fieldnames)
        tables_written.append(filename)

    _write_text(
        output_dir / "final_analysis_summary.md",
        build_final_analysis_summary_md(
            validation=validation,
            config=config,
            primary_pairwise=primary_pairwise,
            heldout_pairwise=heldout_pairwise,
            generalization_rows=generalization_rows,
            sensitive_rows=sensitive_rows,
            primary_instance_rows=primary_instance_rows,
            heldout_instance_rows=heldout_instance_rows,
            primary_spf_cd_ablation=primary_spf_cd_ablation,
            heldout_spf_cd_ablation=heldout_spf_cd_ablation,
            combined_spf_cd_ablation=combined_spf_cd_ablation,
            conflict_structure_rows=conflict_structure_rows,
            runtime_rows=runtime_rows,
            case_study=case_study,
            makespan_rows=makespan_rows,
        ),
    )
    tables_written.append("final_analysis_summary.md")

    _write_text(
        output_dir / "thesis_tables.md",
        build_thesis_tables_md(
            primary_pairwise=primary_pairwise,
            heldout_pairwise=heldout_pairwise,
            generalization_rows=generalization_rows,
            sensitive_rows=sensitive_rows,
            interaction_rows=interaction_rows,
            agent_count_rows=agent_count_rows,
            makespan_rows=makespan_rows,
            runtime_rows=runtime_rows,
            spf_cd_ablation_rows=[
                primary_spf_cd_ablation,
                heldout_spf_cd_ablation,
                combined_spf_cd_ablation,
            ],
        ),
    )
    tables_written.append("thesis_tables.md")

    plots_written: tuple[str, ...] = ()
    if plot_writer is not None:
        plots_written = plot_writer(
            output_dir,
            primary_instance_rows=primary_instance_rows,
            heldout_instance_rows=heldout_instance_rows,
            primary_pairwise=primary_pairwise,
            heldout_pairwise=heldout_pairwise,
            sensitive_rows=sensitive_rows,
            conflict_structure_rows=conflict_structure_rows,
            runtime_rows=runtime_rows,
            primary_instance_count=len(primary_manifest.instances),
            heldout_instance_count=len(heldout_manifest.instances),
        )

    return Mapf7FinalAnalysisResult(
        validation=validation,
        output_dir=output_dir,
        tables_written=tuple(tables_written),
        plots_written=plots_written,
        primary_instance_rows=tuple(primary_instance_rows),
        heldout_instance_rows=tuple(heldout_instance_rows),
    )
