"""MAPF-9.7 final thesis figures, tables, and interpretation (presentation only)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pathfinding.src.experiments.mapf9_primary_analysis import MAPF9_PRIMARY_ANALYSIS_DIR
from pathfinding.src.experiments.mapf_conflict_aware_priority_plots import (
    _apply_plot_style,
    _save_figure,
)
from pathfinding.src.experiments.mapf_priority_strategy_analysis import (
    _format_float,
    _read_csv_rows,
    _write_text,
)

MAPF9_FINAL_ANALYSIS_DIR = Path("pathfinding/results/mapf9_final_analysis")

MANDATORY_FIGURE_BASENAMES: tuple[str, ...] = (
    "fig01_quality_improvements_over_spf",
    "fig02_matched_budget_cglps_vs_ubls",
    "fig03_bounded_search_cost",
)

SEMANTIC_COLORS: dict[str, str] = {
    "cglps_better": "#4C72B0",
    "equal": "#B0B0B0",
    "ubls_better": "#C44E52",
}

METHOD_BAR_COLORS: dict[str, str] = {
    "CGLPS": "#4C72B0",
    "UBLS": "#8172B2",
}

EXPECTED_CGLPS_LEX_INSTANCES: frozenset[str] = frozenset(
    {
        "AR0400SR_n05_high_000",
        "AR0307SR_n05_high_000",
        "AR0307SR_n05_high_002",
        "AR0307SR_n20_high_000",
        "AR0307SR_n20_medium_002",
    }
)

FROZEN_PRESENTATION_CHECKS: dict[str, Any] = {
    "primary_instances": 54,
    "cglps_soc_improved": 2,
    "cglps_makespan_only": 3,
    "cglps_lex": 5,
    "ubls_soc_improved": 1,
    "ubls_makespan_only": 0,
    "ubls_lex": 1,
    "cglps_vs_ubls_soc": (1, 53, 0),
    "cglps_vs_ubls_lex": (4, 50, 0),
    "cglps_unique_soc": 1,
    "cglps_unique_lex": 4,
    "ubls_unique_soc": 0,
    "ubls_unique_lex": 0,
    "per_map_cglps_soc": {"AR0400SR": 1, "AR0307SR": 1},
    "cglps_candidate_distribution": {0: 49, 1: 4, 2: 1},
    "ubls_candidate_distribution": {0: 53, 1: 1},
    "sum_a_i": 75,
    "physical_pp": 204,
    "search_active": 36,
    "logical_pp": {"SPF": 54, "CGLPS": 129, "UBLS": 129},
}


@dataclass(frozen=True, slots=True)
class Mapf9FinalAnalysisConfig:
    primary_analysis_dir: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class Mapf9FinalPresentationData:
    method_summary: list[dict[str, str]]
    soc_pairwise: list[dict[str, str]]
    lex_pairwise: list[dict[str, str]]
    cost_benefit: list[dict[str, str]]
    budget: list[dict[str, str]]
    instance_level: list[dict[str, str]]
    candidate_selection: list[dict[str, str]]
    runtime: list[dict[str, str]]


@dataclass(frozen=True, slots=True)
class Mapf9FinalAnalysisResult:
    output_dir: Path
    figures_written: tuple[str, ...]
    markdown_written: tuple[str, ...]
    consistency_errors: tuple[str, ...]


def default_mapf9_final_config(repo_root: Path) -> Mapf9FinalAnalysisConfig:
    return Mapf9FinalAnalysisConfig(
        primary_analysis_dir=repo_root / MAPF9_PRIMARY_ANALYSIS_DIR,
        output_dir=repo_root / MAPF9_FINAL_ANALYSIS_DIR,
    )


def _load_csv(path: Path) -> list[dict[str, str]]:
    return _read_csv_rows(path)


def load_mapf9_final_presentation_data(primary_analysis_dir: Path) -> Mapf9FinalPresentationData:
    return Mapf9FinalPresentationData(
        method_summary=_load_csv(primary_analysis_dir / "method_summary.csv"),
        soc_pairwise=_load_csv(primary_analysis_dir / "soc_pairwise_summary.csv"),
        lex_pairwise=_load_csv(primary_analysis_dir / "lexicographic_pairwise_summary.csv"),
        cost_benefit=_load_csv(primary_analysis_dir / "cost_benefit_summary.csv"),
        budget=_load_csv(primary_analysis_dir / "budget_summary.csv"),
        instance_level=_load_csv(primary_analysis_dir / "instance_level_summary.csv"),
        candidate_selection=_load_csv(primary_analysis_dir / "candidate_selection_summary.csv"),
        runtime=_load_csv(primary_analysis_dir / "runtime_summary.csv"),
    )


def _soc_row(rows: Sequence[Mapping[str, str]], scope: str, comparison: str) -> dict[str, str]:
    for row in rows:
        if row.get("scope") == scope and row.get("comparison") == comparison:
            return dict(row)
    raise ValueError(f"missing soc row scope={scope} comparison={comparison}")


def _lex_row(rows: Sequence[Mapping[str, str]], scope: str, comparison: str) -> dict[str, str]:
    for row in rows:
        if row.get("scope") == scope and row.get("comparison") == comparison:
            return dict(row)
    raise ValueError(f"missing lex row scope={scope} comparison={comparison}")


def _unique_row(rows: Sequence[Mapping[str, str]], scope: str, beneficiary: str) -> dict[str, str]:
    for row in rows:
        if row.get("scope") == scope and row.get("beneficiary") == beneficiary:
            return dict(row)
    raise ValueError(f"missing unique row scope={scope} beneficiary={beneficiary}")


def _budget_metric(rows: Sequence[Mapping[str, str]], scope: str, metric: str) -> float | int:
    for row in rows:
        if row.get("scope") == scope and row.get("metric") == metric:
            value = row.get("value")
            if value in (None, ""):
                raise ValueError(f"empty budget metric {scope}/{metric}")
            if "." in str(value):
                return float(value)
            return int(value)
    raise ValueError(f"missing budget metric scope={scope} metric={metric}")


def _method_pooled(rows: Sequence[Mapping[str, str]], method: str) -> dict[str, str]:
    for row in rows:
        if row.get("scope") == "pooled" and row.get("method") == method:
            return dict(row)
    raise ValueError(f"missing pooled method row for {method}")


def _cost_pooled(rows: Sequence[Mapping[str, str]], method: str) -> dict[str, str]:
    for row in rows:
        if row.get("scope") == "pooled" and row.get("method") == method:
            return dict(row)
    raise ValueError(f"missing pooled cost row for {method}")


def _runtime_elapsed(rows: Sequence[Mapping[str, str]], scope: str) -> float:
    for row in rows:
        if (
            row.get("scope") == scope
            and row.get("method") == "physical_execution"
            and row.get("metric") == "run_log_elapsed_s"
        ):
            return float(row["mean"])
    raise ValueError(f"missing run_log elapsed for {scope}")


def _pct(count: int, total: int) -> str:
    return f"{count}/{total} ({100.0 * count / total:.1f}%)"


def build_cglps_lex_improved_table_rows(
    data: Mapf9FinalPresentationData,
) -> list[dict[str, Any]]:
    selection_lookup = {
        row["instance_id"]: row
        for row in data.candidate_selection
        if row["method"] == "cglps"
    }
    rows: list[dict[str, Any]] = []
    for instance_id in sorted(EXPECTED_CGLPS_LEX_INSTANCES):
        inst = next(row for row in data.instance_level if row["instance_id"] == instance_id)
        sel = selection_lookup[instance_id]
        soc_delta = int(inst["spf_soc"]) - int(inst["cglps_soc"])
        makespan_only = inst["cglps_makespan_only_vs_spf"] == "True"
        effect = "Makespan-only" if makespan_only and soc_delta == 0 else "SoC"
        rows.append(
            {
                "map_name": inst["map_name"],
                "instance_id": instance_id,
                "agent_count": int(inst["agent_count"]),
                "interaction_level": inst["interaction_level"],
                "actual_a_i": int(inst["actual_a_i"]),
                "selected_candidate_id": int(inst["cglps_selected_candidate_id"]),
                "spf_soc": int(inst["spf_soc"]),
                "cglps_soc": int(inst["cglps_soc"]),
                "soc_delta": soc_delta,
                "spf_makespan": int(inst["spf_makespan"]),
                "cglps_makespan": int(inst["cglps_makespan"]),
                "effect_type": effect,
                "winning_candidate_rank": sel.get("winning_candidate_rank") or "—",
                "winning_pair_conflict_event_count": sel.get("winning_pair_conflict_event_count") or "—",
            }
        )
    return rows


def build_fig01_quality_points(data: Mapf9FinalPresentationData) -> list[dict[str, Any]]:
    cglps = _soc_row(data.soc_pairwise, "pooled", "CGLPS_vs_SPF")
    ubls = _soc_row(data.soc_pairwise, "pooled", "UBLS_vs_SPF")
    return [
        {
            "method": "CGLPS",
            "soc_improved": int(cglps["soc_improved_count"]),
            "makespan_only": int(cglps["makespan_only_improved_count"]),
        },
        {
            "method": "UBLS",
            "soc_improved": int(ubls["soc_improved_count"]),
            "makespan_only": int(ubls["makespan_only_improved_count"]),
        },
    ]


def build_fig02_matched_budget_points(data: Mapf9FinalPresentationData) -> list[dict[str, Any]]:
    soc = _soc_row(data.soc_pairwise, "pooled", "CGLPS_vs_UBLS_soc")
    lex = _lex_row(data.lex_pairwise, "pooled", "CGLPS_vs_UBLS")
    return [
        {
            "objective": "SoC-only",
            "cglps_better": int(soc["cglps_better_count"]),
            "equal": int(soc["equal_count"]),
            "ubls_better": int(soc["ubls_better_count"]),
        },
        {
            "objective": "Lexicographic",
            "cglps_better": int(lex["left_better_count"]),
            "equal": int(lex["equal_count"]),
            "ubls_better": int(lex["right_better_count"]),
        },
    ]


def build_fig03_cost_points(data: Mapf9FinalPresentationData) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for method in ("SPF", "CGLPS", "UBLS"):
        row = _cost_pooled(data.cost_benefit, method)
        points.append(
            {
                "method": method,
                "logical_pp_candidates": int(row["logical_pp_candidates"]),
                "logical_runtime_ms_total": float(row["logical_runtime_ms_total"]),
                "logical_runtime_ratio_vs_spf": float(row["logical_runtime_ratio_vs_spf"]),
            }
        )
    return points


def verify_final_presentation_consistency(data: Mapf9FinalPresentationData) -> list[str]:
    errors: list[str] = []
    checks = FROZEN_PRESENTATION_CHECKS

    if len(data.instance_level) != checks["primary_instances"]:
        errors.append(f"primary instances={len(data.instance_level)}")

    cglps_spf = _soc_row(data.soc_pairwise, "pooled", "CGLPS_vs_SPF")
    ubls_spf = _soc_row(data.soc_pairwise, "pooled", "UBLS_vs_SPF")
    cglps_lex = _lex_row(data.lex_pairwise, "pooled", "CGLPS_vs_SPF")
    ubls_lex = _lex_row(data.lex_pairwise, "pooled", "UBLS_vs_SPF")
    cglps_vs_ubls_soc = _soc_row(data.soc_pairwise, "pooled", "CGLPS_vs_UBLS_soc")
    cglps_vs_ubls_lex = _lex_row(data.lex_pairwise, "pooled", "CGLPS_vs_UBLS")
    cglps_unique = _unique_row(data.soc_pairwise, "pooled", "CGLPS")

    numeric_checks = [
        ("cglps_soc_improved", int(cglps_spf["soc_improved_count"])),
        ("cglps_makespan_only", int(cglps_spf["makespan_only_improved_count"])),
        ("cglps_lex", int(cglps_lex["left_better_count"])),
        ("ubls_soc_improved", int(ubls_spf["soc_improved_count"])),
        ("ubls_makespan_only", int(ubls_spf["makespan_only_improved_count"])),
        ("ubls_lex", int(ubls_lex["left_better_count"])),
        ("cglps_unique_soc", int(cglps_unique["unique_soc_advantage_count"])),
        ("cglps_unique_lex", int(cglps_unique["unique_lex_advantage_count"])),
    ]
    for key, actual in numeric_checks:
        if actual != checks[key]:
            errors.append(f"{key}: expected {checks[key]}, got {actual}")

    soc_triple = (
        int(cglps_vs_ubls_soc["cglps_better_count"]),
        int(cglps_vs_ubls_soc["equal_count"]),
        int(cglps_vs_ubls_soc["ubls_better_count"]),
    )
    if soc_triple != checks["cglps_vs_ubls_soc"]:
        errors.append(f"cglps_vs_ubls_soc: expected {checks['cglps_vs_ubls_soc']}, got {soc_triple}")

    lex_triple = (
        int(cglps_vs_ubls_lex["left_better_count"]),
        int(cglps_vs_ubls_lex["equal_count"]),
        int(cglps_vs_ubls_lex["right_better_count"]),
    )
    if lex_triple != checks["cglps_vs_ubls_lex"]:
        errors.append(f"cglps_vs_ubls_lex: expected {checks['cglps_vs_ubls_lex']}, got {lex_triple}")

    for map_name, expected in checks["per_map_cglps_soc"].items():
        actual = int(_soc_row(data.soc_pairwise, map_name, "CGLPS_vs_SPF")["soc_improved_count"])
        if actual != expected:
            errors.append(f"{map_name} cglps soc improved: expected {expected}, got {actual}")

    cglps_counts = Counter(int(row["cglps_selected_candidate_id"]) for row in data.instance_level)
    ubls_counts = Counter(int(row["ubls_selected_candidate_id"]) for row in data.instance_level)
    if dict(sorted(cglps_counts.items())) != checks["cglps_candidate_distribution"]:
        errors.append(f"cglps candidate distribution mismatch: {dict(cglps_counts)}")
    if dict(sorted(ubls_counts.items())) != checks["ubls_candidate_distribution"]:
        errors.append(f"ubls candidate distribution mismatch: {dict(ubls_counts)}")

    if _budget_metric(data.budget, "pooled", "sum_actual_a_i") != checks["sum_a_i"]:
        errors.append("sum A_i mismatch")
    if _budget_metric(data.budget, "pooled", "physical_pp_eval_count") != checks["physical_pp"]:
        errors.append("physical PP mismatch")

    search_active = sum(1 for row in data.instance_level if row["search_active"] == "True")
    if search_active != checks["search_active"]:
        errors.append(f"search_active={search_active}")

    for method, expected in checks["logical_pp"].items():
        actual = int(_method_pooled(data.method_summary, method)["logical_pp_candidates"])
        if actual != expected:
            errors.append(f"logical PP {method}: expected {expected}, got {actual}")

    improved_rows = build_cglps_lex_improved_table_rows(data)
    if len(improved_rows) != 5:
        errors.append(f"table 4 row count={len(improved_rows)}")
    if {row["instance_id"] for row in improved_rows} != EXPECTED_CGLPS_LEX_INSTANCES:
        errors.append("table 4 instance set mismatch")

    return errors


def build_thesis_tables_md(data: Mapf9FinalPresentationData) -> str:
    cglps_spf = _soc_row(data.soc_pairwise, "pooled", "CGLPS_vs_SPF")
    ubls_spf = _soc_row(data.soc_pairwise, "pooled", "UBLS_vs_SPF")
    cglps_lex = _lex_row(data.lex_pairwise, "pooled", "CGLPS_vs_SPF")
    ubls_lex = _lex_row(data.lex_pairwise, "pooled", "UBLS_vs_SPF")
    cglps_vs_ubls_soc = _soc_row(data.soc_pairwise, "pooled", "CGLPS_vs_UBLS_soc")
    cglps_vs_ubls_lex = _lex_row(data.lex_pairwise, "pooled", "CGLPS_vs_UBLS")
    cglps_unique = _unique_row(data.soc_pairwise, "pooled", "CGLPS")
    ubls_unique = _unique_row(data.soc_pairwise, "pooled", "UBLS")
    improved_rows = build_cglps_lex_improved_table_rows(data)

    lines = [
        "# MAPF-9 Final Thesis Tables",
        "",
        "## Table 1 — Primary MAPF-9 dataset and execution integrity",
        "",
        "| Map | Instances | Seed | SPF success | CGLPS success | UBLS success | Sum A_i | Physical PP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for map_name, seed in (("AR0400SR", 2030), ("AR0307SR", 2031)):
        map_rows = {row["method"]: row for row in data.method_summary if row["scope"] == map_name}
        lines.append(
            f"| {map_name} | 27 | {seed} | {map_rows['SPF']['success_count']}/27 | "
            f"{map_rows['CGLPS']['success_count']}/27 | {map_rows['UBLS']['success_count']}/27 | "
            f"{_budget_metric(data.budget, map_name, 'sum_actual_a_i')} | "
            f"{_budget_metric(data.budget, map_name, 'physical_pp_eval_count')} |"
        )
    pooled_methods = {row["method"]: row for row in data.method_summary if row["scope"] == "pooled"}
    lines.append(
        f"| **Pooled** | **54** | — | **{pooled_methods['SPF']['success_count']}/54** | "
        f"**{pooled_methods['CGLPS']['success_count']}/54** | "
        f"**{pooled_methods['UBLS']['success_count']}/54** | "
        f"**{_budget_metric(data.budget, 'pooled', 'sum_actual_a_i')}** | "
        f"**{_budget_metric(data.budget, 'pooled', 'physical_pp_eval_count')}** |"
    )
    lines.extend(
        [
            "",
            "## Table 2 — Quality outcomes vs SPF (n = 54)",
            "",
            "| Method | SoC improved | SoC equal | SoC worse | Total SoC gain | Makespan-only | Lex better | Lex equal | Lex worse |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| CGLPS | {_pct(int(cglps_spf['soc_improved_count']), 54)} | "
            f"{int(cglps_spf['soc_equal_count'])} | {int(cglps_spf['soc_structurally_worse_count'])} | "
            f"{int(cglps_spf['total_soc_improvement'])} | {int(cglps_spf['makespan_only_improved_count'])} | "
            f"{int(cglps_lex['left_better_count'])} | {int(cglps_lex['equal_count'])} | "
            f"{int(cglps_lex['right_better_count'])} |",
            f"| UBLS | {_pct(int(ubls_spf['soc_improved_count']), 54)} | "
            f"{int(ubls_spf['soc_equal_count'])} | {int(ubls_spf['soc_structurally_worse_count'])} | "
            f"{int(ubls_spf['total_soc_improvement'])} | {int(ubls_spf['makespan_only_improved_count'])} | "
            f"{int(ubls_lex['left_better_count'])} | {int(ubls_lex['equal_count'])} | "
            f"{int(ubls_lex['right_better_count'])} |",
            "",
            "## Table 3 — Matched-budget CGLPS vs UBLS (n = 54)",
            "",
            "### SoC-only",
            "",
            f"- CGLPS better: **{int(cglps_vs_ubls_soc['cglps_better_count'])}**",
            f"- Equal: **{int(cglps_vs_ubls_soc['equal_count'])}**",
            f"- UBLS better: **{int(cglps_vs_ubls_soc['ubls_better_count'])}**",
            "",
            "### Frozen lexicographic objective (success → SoC → makespan)",
            "",
            f"- CGLPS better: **{int(cglps_vs_ubls_lex['left_better_count'])}**",
            f"- Equal: **{int(cglps_vs_ubls_lex['equal_count'])}**",
            f"- UBLS better: **{int(cglps_vs_ubls_lex['right_better_count'])}**",
            "",
            "### Unique benefit relative to SPF (descriptive, not statistical superiority)",
            "",
            f"- CGLPS unique SoC: **{int(cglps_unique['unique_soc_advantage_count'])}**",
            f"- CGLPS unique lex: **{int(cglps_unique['unique_lex_advantage_count'])}**",
            f"- UBLS unique SoC: **{int(ubls_unique['unique_soc_advantage_count'])}**",
            f"- UBLS unique lex: **{int(ubls_unique['unique_lex_advantage_count'])}**",
            "",
            "## Table 4 — CGLPS lexicographically improved instances (distinct set)",
            "",
            "| Map | Instance | n | Interaction | A_i | Cand. | SPF SoC | CGLPS SoC | ΔSoC | SPF MS | CGLPS MS | Effect | Pair rank | Pair events |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    for row in improved_rows:
        lines.append(
            f"| {row['map_name']} | {row['instance_id']} | {row['agent_count']} | "
            f"{row['interaction_level']} | {row['actual_a_i']} | {row['selected_candidate_id']} | "
            f"{row['spf_soc']} | {row['cglps_soc']} | {row['soc_delta']} | "
            f"{row['spf_makespan']} | {row['cglps_makespan']} | {row['effect_type']} | "
            f"{row['winning_candidate_rank']} | {row['winning_pair_conflict_event_count']} |"
        )
    lines.extend(
        [
            "",
            "*Note:* UBLS matched the SoC improvement on `AR0400SR_n05_high_000`; the four remaining "
            "CGLPS-vs-UBLS lex wins occurred on AR0307SR.",
            "",
            "## Table 5 — Cost-benefit (logical method accounting, n = 54)",
            "",
            "| Method | Logical PP candidates | PP ratio vs SPF | Total logical runtime (ms) | Runtime ratio vs SPF | SoC improved | Makespan-only | Total SoC gain |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in ("SPF", "CGLPS", "UBLS"):
        row = _cost_pooled(data.cost_benefit, method)
        pp_candidates = int(row["logical_pp_candidates"])
        pp_ratio = (
            "1.0"
            if method == "SPF"
            else _format_float(pp_candidates / 54, digits=3)
        )
        lines.append(
            f"| {method} | {pp_candidates} | {pp_ratio} | "
            f"{_format_float(float(row['logical_runtime_ms_total']), digits=1)} | "
            f"{_format_float(float(row['logical_runtime_ratio_vs_spf']), digits=3)} | "
            f"{int(row['soc_improved_instances'])} | {int(row['makespan_only_improved_instances'])} | "
            f"{int(row['total_soc_improvement'])} |"
        )
    combined_elapsed = _runtime_elapsed(data.runtime, "AR0400SR") + _runtime_elapsed(
        data.runtime, "AR0307SR"
    )
    lines.extend(
        [
            "",
            f"*Physical combined experiment:* 204 PP evaluations total; run.log elapsed "
            f"AR0400SR = {_runtime_elapsed(data.runtime, 'AR0400SR')} s, "
            f"AR0307SR = {_runtime_elapsed(data.runtime, 'AR0307SR')} s, "
            f"combined = {combined_elapsed} s. Physical combined PP count is not directly comparable "
            "to a single-method logical-runtime row above.",
            "",
        ]
    )
    return "\n".join(lines)


def build_figure_captions_md(data: Mapf9FinalPresentationData) -> str:
    fig1 = build_fig01_quality_points(data)
    fig2 = build_fig02_matched_budget_points(data)
    fig3 = build_fig03_cost_points(data)
    cglps = next(point for point in fig3 if point["method"] == "CGLPS")
    ubls = next(point for point in fig3 if point["method"] == "UBLS")
    return f"""# MAPF-9 Final Figure Captions

## Figure 1 — `fig01_quality_improvements_over_spf`

**English caption:** Count of instances with SoC improvement or makespan-only improvement relative to SPF for CGLPS and UBLS on the fresh 54-instance primary set (n = 54). SoC improvement requires strictly lower sum of costs; makespan-only improvement requires equal SoC and strictly lower makespan. Categories are reported separately.

**Polish caption:** Liczba instancji z poprawą SoC lub wyłączną poprawą makespan względem SPF dla CGLPS i UBLS w świeżym zbiorze 54 instancji (n = 54). Poprawa SoC oznacza ściśle niższą sumę kosztów; poprawa wyłącznie makespan — równy SoC i niższy makespan. Kategorie podano osobno.

**Interpretation caveat:** Quality changes were sparse (CGLPS: {fig1[0]['soc_improved']} SoC + {fig1[0]['makespan_only']} makespan-only; UBLS: {fig1[1]['soc_improved']} SoC). Descriptive counts only; no inferential significance claim.

---

## Figure 2 — `fig02_matched_budget_cglps_vs_ubls`

**English caption:** Matched-budget comparison of CGLPS and UBLS on n = 54 primary instances. Each instance uses the same actual additional PP-evaluation budget A_i. Stacked bars show SoC-only directional counts (CGLPS better / equal / UBLS better) and frozen lexicographic directional counts under success → SoC → makespan.

**Polish caption:** Porównanie CGLPS i UBLS przy dopasowanym budżecie na n = 54 instancjach pierwotnych. Każda instancja ma ten sam rzeczywisty budżet dodatkowych ewaluacji PP (A_i). Skumulowane słupki pokazują liczności kierunkowe dla samego SoC oraz dla zamrożonego kryterium leksykograficznego (sukces → SoC → makespan).

**Interpretation caveat:** SoC-only counts: {fig2[0]['cglps_better']}/{fig2[0]['equal']}/{fig2[0]['ubls_better']}. Lexicographic counts: {fig2[1]['cglps_better']}/{fig2[1]['equal']}/{fig2[1]['ubls_better']}. This compares methods under matched budget; it does not establish statistical significance.

---

## Figure 3 — `fig03_bounded_search_cost`

**English caption:** Total logical method runtime normalized to SPF = 1.00 on the pooled 54-instance primary set. Annotation shows logical PP candidate counts (SPF 54; CGLPS 129; UBLS 129). Values shown: CGLPS ≈ {cglps['logical_runtime_ratio_vs_spf']:.2f}×, UBLS ≈ {ubls['logical_runtime_ratio_vs_spf']:.2f}×.

**Polish caption:** Całkowity logiczny czas wykonania metody znormalizowany do SPF = 1,00 dla puli 54 instancji pierwotnych. Adnotacja podaje liczbę logicznych ewaluacji kandydatów PP (SPF 54; CGLPS 129; UBLS 129).

**Interpretation caveat:** Logical runtime accounts per-method preprocessing and candidate evaluation. It is distinct from the physical combined experiment wall-clock (AR0400SR + AR0307SR ≈ 17 236 s combined run.log elapsed with 204 physical PP evaluations shared across methods).
"""


def build_final_mapf9_interpretation_md(data: Mapf9FinalPresentationData) -> str:
    cglps_spf = _soc_row(data.soc_pairwise, "pooled", "CGLPS_vs_SPF")
    ubls_spf = _soc_row(data.soc_pairwise, "pooled", "UBLS_vs_SPF")
    cglps_lex = _lex_row(data.lex_pairwise, "pooled", "CGLPS_vs_SPF")
    cglps_vs_ubls_soc = _soc_row(data.soc_pairwise, "pooled", "CGLPS_vs_UBLS_soc")
    cglps_vs_ubls_lex = _lex_row(data.lex_pairwise, "pooled", "CGLPS_vs_UBLS")
    cglps_cost = _cost_pooled(data.cost_benefit, "CGLPS")
    search_active = sum(1 for row in data.instance_level if row["search_active"] == "True")
    ar0307_lex_wins = sum(
        1
        for row in data.instance_level
        if row["map_name"] == "AR0307SR" and row["cglps_vs_ubls_lex"] == "left_better"
    )

    return f"""# MAPF-9 Final Interpretation

## Purpose

MAPF-7 showed that static conflict-degree ordering direction was unstable across
priority permutations. MAPF-8 generalized this within bg512: priority order changes
were frequent (36/54 and 34/54 cross-map for CDF-H and CDF-L), while SoC changes
remained sparse (2/54 and 3/54). MAPF-9 therefore tested whether conflict information
is more useful as a **search-guidance signal** for bounded local exploration around
SPF than as a static ordering rule.

## Primary findings

On the fresh 54-instance primary set:

- All methods succeeded on **54/54** instances (SPF, CGLPS, UBLS).
- **CGLPS vs SPF:** SoC improved **{int(cglps_spf['soc_improved_count'])}/54**; total SoC gain **{int(cglps_spf['total_soc_improvement'])}**; makespan-only **{int(cglps_spf['makespan_only_improved_count'])}**; lexicographic better **{int(cglps_lex['left_better_count'])}**.
- **UBLS vs SPF:** SoC improved **{int(ubls_spf['soc_improved_count'])}/54**; total SoC gain **{int(ubls_spf['total_soc_improvement'])}**; makespan-only **{int(ubls_spf['makespan_only_improved_count'])}**.
- **CGLPS vs UBLS (matched A_i):** SoC **{int(cglps_vs_ubls_soc['cglps_better_count'])}/{int(cglps_vs_ubls_soc['equal_count'])}/{int(cglps_vs_ubls_soc['ubls_better_count'])}**; lex **{int(cglps_vs_ubls_lex['left_better_count'])}/{int(cglps_vs_ubls_lex['equal_count'])}/{int(cglps_vs_ubls_lex['right_better_count'])}**.

## Interpretation

Conflict guidance produced a **small observed advantage** over the deterministic
unguided bounded control (UBLS), but improvements remained **sparse**. One CGLPS SoC
improvement (`AR0400SR_n05_high_000`) was **shared** with UBLS; one SoC improvement
(`AR0307SR_n20_high_000`) was **unique** to CGLPS. All **{ar0307_lex_wins}** CGLPS-vs-UBLS
lexicographic wins occurred on AR0307SR. The result demonstrates **possibility**, not
universal superiority.

## Conflict structure

All observed improvements occurred among **search-active** instances (A_i > 0;
**{search_active}/54** search-active overall). LOW-interaction instances had A_i = 0
by construction and therefore identical SPF/CGLPS/UBLS outcomes. Some improved cases
involved many conflict events; many high-conflict cases did not improve. Conflict
magnitude alone did **not** provide a reliable predictor. No causal claim is made.

## Cost

CGLPS used **129** logical PP candidates vs **54** for SPF; pooled logical runtime
ratio ≈ **{float(cglps_cost['logical_runtime_ratio_vs_spf']):.2f}×** SPF for total SoC
gain **{int(cglps_spf['total_soc_improvement'])}** over 54 instances. Cost-benefit under
the tested B = 4 transposition neighbourhood is **weak**.

This does **not** imply CGLPS is useless. It shows that conflict-guided local
exploration **can** expose beneficial order changes missed by SPF, but a fixed
B = 4 neighbourhood spends substantial computation on instances where no benefit
is obtained.

## Scope limitations

- bg512 arena family only; two fresh maps (AR0400SR, AR0307SR).
- 54 primary instances; B = 4 frozen, not optimized.
- Single-transposition neighbourhood only.
- Deterministic UBLS control with one frozen seed convention.
- No exhaustive priority permutations; no global optimality claim.
- Descriptive counts only; no inferential significance claim.
- Historical MAPF-7/8 (108 instances) used as context only, not combined denominators.

## Overall conclusion

MAPF-9 provides evidence that conflict information can be useful for directing a
bounded local search over priority orders, but within the tested bg512 setting the
benefit was sparse and modest relative to its computational cost. The result
therefore supports conflict guidance as an **informative search signal** rather
than as a generally superior static prioritization rule.
"""


def build_thesis_subsection_pl_md(data: Mapf9FinalPresentationData) -> str:
    cglps_spf = _soc_row(data.soc_pairwise, "pooled", "CGLPS_vs_SPF")
    ubls_spf = _soc_row(data.soc_pairwise, "pooled", "UBLS_vs_SPF")
    cglps_vs_ubls_soc = _soc_row(data.soc_pairwise, "pooled", "CGLPS_vs_UBLS_soc")
    cglps_vs_ubls_lex = _lex_row(data.lex_pairwise, "pooled", "CGLPS_vs_UBLS")
    cglps_cost = _cost_pooled(data.cost_benefit, "CGLPS")
    ubls_cost = _cost_pooled(data.cost_benefit, "UBLS")

    return f"""# Konfliktowo sterowane lokalne przeszukiwanie kolejności priorytetów

## Motywacja

W MAPF-7 wykazano niestabilność kierunku statycznego porządkowania według stopnia
konfliktu. MAPF-8 uogólnił ten wniosek w rodzinie bg512: zmiany kolejności priorytetów
były częste, natomiast zmiany sumy kosztów (SoC) pozostawały rzadkie. MAPF-9 bada
zatem, czy informacja o konfliktach jest użyteczniejsza jako sygnał **sterowania
przeszukiwaniem** wokół punktu startowego SPF niż jako stała reguła priorytetu.

## Metoda

Punktem odniesienia pozostaje SPF. Dla każdej instancji budowany jest graf konfliktów
na niezależnych ścieżkach; pary agentów rankowane są według liczby zdarzeń konfliktowych.
CGLPS generuje do B = 4 kandydatów przez pojedynczą transpozycję w kolejności SPF,
wybierając pary o najwyższym rankingu konfliktowym. UBLS stanowi dopasowany kontrolnie
deterministyczny odpowiednik z tym samym budżetem A_i, lecz bez rankingu konfliktowego.
Zamrożone kryterium wyboru rozwiązania: sukces → niższy SoC → niższy makespan.

## Zbiór ewaluacyjny

Primary obejmuje **54 świeże instancje** MAPF na mapach AR0400SR (seed 2030) i
AR0307SR (seed 2031), rozłącznych względem katalogów MAPF-8 na tych samych mapach.
Walidacja ograniczona jest do rodziny scenariuszy MovingAI **bg512**.

## Wyniki

- Sukces wszystkich metod: **54/54**.
- **CGLPS vs SPF:** poprawa SoC **{int(cglps_spf['soc_improved_count'])}/54**; wyłączna poprawa makespan **{int(cglps_spf['makespan_only_improved_count'])}/54**; łącznie **{int(cglps_spf['soc_improved_count']) + int(cglps_spf['makespan_only_improved_count'])}** leksykograficznych usprawnień (przy braku odzyskiwania sukcesu).
- **UBLS vs SPF:** poprawa SoC **{int(ubls_spf['soc_improved_count'])}/54**; wyłączna poprawa makespan **{int(ubls_spf['makespan_only_improved_count'])}/54**.
- **CGLPS vs UBLS (dopasowany A_i):** SoC **{int(cglps_vs_ubls_soc['cglps_better_count'])}/{int(cglps_vs_ubls_soc['equal_count'])}/{int(cglps_vs_ubls_soc['ubls_better_count'])}**; leksykograficznie **{int(cglps_vs_ubls_lex['left_better_count'])}/{int(cglps_vs_ubls_lex['equal_count'])}/{int(cglps_vs_ubls_lex['right_better_count'])}**.
- Per mapa (CGLPS SoC): AR0400SR **1/27**, AR0307SR **1/27**.

## Koszt obliczeniowy

Logiczne ewaluacje kandydatów PP: SPF **54**, CGLPS **129**, UBLS **129**. Całkowity
logiczny czas wykonania w puli: CGLPS ≈ **{float(cglps_cost['logical_runtime_ratio_vs_spf']):.2f}×** SPF,
UBLS ≈ **{float(ubls_cost['logical_runtime_ratio_vs_spf']):.2f}×** SPF. Fizyczny łączny
eksperyment MAPF-9 wykonał **204** ewaluacje PP (oba mapy, wszystkie metody).

## Interpretacja

W badanej próbie zaobserwowano **niewielką, lecz realną** przewagę CGLPS nad UBLS
przy dopasowanym budżecie, przy jednoczesnej **rzadkości** zmian jakości względem SPF.
Jeden wspólny przypadek poprawy SoC (`AR0400SR_n05_high_000`) pokazuje, że sama lokalna
eksploracja może czasem wystarczyć bez rankingu konfliktowego; pozostałe cztery
leksykograficzne przewagi CGLPS nad UBLS wystąpiły na AR0307SR. Wyniki wskazują na
możliwość korzystnego wykorzystania informacji konfliktowej, ale **nie pozwalają**
wnioskować o ogólnej wyższości reguły statycznej ani o istotności statystycznej.

## Ograniczenia

- tylko bg512; dwa mapy; 54 instancje primary;
- zamrożone B = 4 i sąsiedztwo pojedynczej transpozycji;
- brak optymalizacji budżetu; brak permutacji wyczerpujących;
- UBLS z jedną zamrożoną konwencją seed;
- wyłącznie opisy ilościowe; MAPF-7/8 wyłącznie jako kontekst historyczny.

## Wniosek

MAPF-9 dostarcza dowodu, że konflikt może kierować ograniczonym lokalnym przeszukiwaniem
kolejności priorytetów, lecz w badanej konfiguracji korzyść jakościowa była rzadka,
a koszt obliczeniowy wysoki. Informacja konfliktowa pełni więc rolę sygnału
przeszukiwania, a nie pewnego deterministycznego usprawnienia priorytetu globalnego.
"""


def build_mapf9_close_md(data: Mapf9FinalPresentationData) -> str:
    cglps_spf = _soc_row(data.soc_pairwise, "pooled", "CGLPS_vs_SPF")
    cglps_lex = _lex_row(data.lex_pairwise, "pooled", "CGLPS_vs_SPF")
    cglps_vs_ubls_lex = _lex_row(data.lex_pairwise, "pooled", "CGLPS_vs_UBLS")

    return f"""# MAPF-9 Close

## Status

**MAPF-9 CLOSED**

## Frozen primary evidence

- Fresh primary set: **54** bg512 instances (AR0400SR seed 2030, AR0307SR seed 2031).
- Success: SPF/CGLPS/UBLS **54/54**.
- CGLPS vs SPF: SoC **{int(cglps_spf['soc_improved_count'])}/54**, makespan-only **{int(cglps_spf['makespan_only_improved_count'])}/54**, lex **{int(cglps_lex['left_better_count'])}/54**.
- CGLPS vs UBLS (matched budget): lex **{int(cglps_vs_ubls_lex['left_better_count'])}/{int(cglps_vs_ubls_lex['equal_count'])}/{int(cglps_vs_ubls_lex['right_better_count'])}**.
- Sum A_i = **75**; physical PP = **204**; logical PP candidates SPF/CGLPS/UBLS = **54/129/129**.

## Final interpretation

- Conflict-guided bounded search can occasionally find better priority orders than SPF.
- Advantage over matched UBLS was small and mostly on AR0307SR.
- Quality gains were sparse relative to ≈1.94× logical runtime overhead.
- No success recovery was observed because SPF already solved all primary instances.

## Frozen artifacts

- `pathfinding/docs/mapf9_design_freeze.md`
- `pathfinding/results/mapf_benchmarks/*_mapf9_manifest.json`
- `pathfinding/results/mapf9_primary_execution/`
- `pathfinding/results/mapf9_primary_analysis/` (MAPF-9.6a authoritative analysis)
- `pathfinding/results/mapf9_final_analysis/` (MAPF-9.7 thesis presentation)

## No further tuning

B = 4, CGLPS, UBLS, fresh catalogues, and primary production results are **frozen**.
Any future work on alternative budgets, neighbourhoods, or new execution is **outside MAPF-9**.
"""


def render_mapf9_final_figures(
    *,
    output_dir: Path,
    data: Mapf9FinalPresentationData,
) -> tuple[str, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    _apply_plot_style()
    plots_dir = output_dir / "figures"
    written: list[str] = []

    # Figure 1
    fig1_points = build_fig01_quality_points(data)
    categories = ["SoC improved", "Makespan-only"]
    x = np.arange(len(categories))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for idx, point in enumerate(fig1_points):
        offset = (idx - 0.5) * width
        values = [point["soc_improved"], point["makespan_only"]]
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=point["method"],
            color=METHOD_BAR_COLORS[point["method"]],
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.08,
                str(value),
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Instance count")
    ax.set_title("Quality improvements over SPF (n = 54)")
    ax.set_ylim(0, max(max(p["soc_improved"], p["makespan_only"]) for p in fig1_points) + 1.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, plots_dir, MANDATORY_FIGURE_BASENAMES[0])
    plt.close(fig)
    written.extend([f"{MANDATORY_FIGURE_BASENAMES[0]}.png", f"{MANDATORY_FIGURE_BASENAMES[0]}.pdf"])

    # Figure 2
    fig2_points = build_fig02_matched_budget_points(data)
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    y = np.arange(len(fig2_points))
    left = np.array([point["cglps_better"] for point in fig2_points], dtype=float)
    mid = np.array([point["equal"] for point in fig2_points], dtype=float)
    right = np.array([point["ubls_better"] for point in fig2_points], dtype=float)
    ax.barh(y, left, color=SEMANTIC_COLORS["cglps_better"], label="CGLPS better", height=0.55)
    ax.barh(y, mid, left=left, color=SEMANTIC_COLORS["equal"], label="Equal", height=0.55)
    ax.barh(
        y,
        right,
        left=left + mid,
        color=SEMANTIC_COLORS["ubls_better"],
        label="UBLS better",
        height=0.55,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([point["objective"] for point in fig2_points])
    ax.invert_yaxis()
    ax.set_xlabel("Instance count (n = 54, matched A_i)")
    ax.set_title("Matched-budget CGLPS vs UBLS")
    ax.set_xlim(0, 54)
    for idx, point in enumerate(fig2_points):
        ax.text(point["cglps_better"] / 2, idx, str(point["cglps_better"]), ha="center", va="center", color="white", fontsize=8)
        ax.text(left[idx] + mid[idx] / 2, idx, str(point["equal"]), ha="center", va="center", fontsize=8)
        if point["ubls_better"]:
            ax.text(left[idx] + mid[idx] + right[idx] / 2, idx, str(point["ubls_better"]), ha="center", va="center", color="white", fontsize=8)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    _save_figure(fig, plots_dir, MANDATORY_FIGURE_BASENAMES[1])
    plt.close(fig)
    written.extend([f"{MANDATORY_FIGURE_BASENAMES[1]}.png", f"{MANDATORY_FIGURE_BASENAMES[1]}.pdf"])

    # Figure 3
    fig3_points = build_fig03_cost_points(data)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    methods = [point["method"] for point in fig3_points]
    ratios = [point["logical_runtime_ratio_vs_spf"] for point in fig3_points]
    colors = ["#55A868", METHOD_BAR_COLORS["CGLPS"], METHOD_BAR_COLORS["UBLS"]]
    bars = ax.bar(methods, ratios, color=colors, width=0.58, edgecolor="white", linewidth=0.6)
    ax.axhline(1.0, color="#666666", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Logical runtime relative to SPF")
    ax.set_title("Cost of bounded search (pooled logical runtime)")
    ax.set_ylim(0, max(ratios) * 1.18)
    for bar, point in zip(bars, fig3_points, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{point['logical_runtime_ratio_vs_spf']:.2f}×",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    pp_text = "; ".join(
        f"{point['method']} {point['logical_pp_candidates']}" for point in fig3_points
    )
    ax.text(
        0.5,
        -0.16,
        f"Logical PP candidates: {pp_text}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    _save_figure(fig, plots_dir, MANDATORY_FIGURE_BASENAMES[2])
    plt.close(fig)
    written.extend([f"{MANDATORY_FIGURE_BASENAMES[2]}.png", f"{MANDATORY_FIGURE_BASENAMES[2]}.pdf"])

    return tuple(written)


def run_mapf9_final_analysis(config: Mapf9FinalAnalysisConfig) -> Mapf9FinalAnalysisResult:
    data = load_mapf9_final_presentation_data(config.primary_analysis_dir)
    consistency_errors = verify_final_presentation_consistency(data)
    if consistency_errors:
        raise ValueError(
            "MAPF-9.7 final presentation consistency failed:\n"
            + "\n".join(f"- {error}" for error in consistency_errors)
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    markdown_files = {
        "thesis_tables.md": build_thesis_tables_md(data),
        "figure_captions.md": build_figure_captions_md(data),
        "final_mapf9_interpretation.md": build_final_mapf9_interpretation_md(data),
        "thesis_subsection_pl.md": build_thesis_subsection_pl_md(data),
        "mapf9_close.md": build_mapf9_close_md(data),
    }
    for name, content in markdown_files.items():
        _write_text(config.output_dir / name, content)

    figures_written = render_mapf9_final_figures(output_dir=config.output_dir, data=data)
    return Mapf9FinalAnalysisResult(
        output_dir=config.output_dir,
        figures_written=figures_written,
        markdown_written=tuple(markdown_files.keys()),
        consistency_errors=tuple(consistency_errors),
    )
