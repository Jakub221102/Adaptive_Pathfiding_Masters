"""Tests for MAPF-9.6 formal primary analysis."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pathfinding.src.experiments.mapf9_primary_analysis import (
    EXPECTED_PHYSICAL_PP,
    EXPECTED_POOLED_INSTANCES,
    EXPECTED_POOLED_METHOD_ROWS,
    EXPECTED_POOLED_PHYSICAL_PP,
    EXPECTED_SUM_A_I,
    MAPF9InstanceKey,
    MAPF9_MAPS,
    MethodOutcome,
    _candidate_selection_distribution,
    _lex_comparison_summary,
    _soc_comparison_summary,
    _unique_benefit_counts,
    build_budget_rows,
    build_candidate_selection_rows,
    build_improved_instances_rows,
    build_instance_level_rows,
    compare_lexicographic,
    default_mapf9_primary_analysis_config,
    is_makespan_only_improvement,
    load_map_dataset,
    run_mapf9_primary_analysis,
    soc_improvement,
    validate_and_load_all,
    verify_analysis_consistency,
)
from pathfinding.src.experiments.mapf_benchmark_instances import load_benchmark_manifest
from pathfinding.src.experiments.mapf_cross_map_catalogue_validation import (
    manifest_file_sha256,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _production_available() -> bool:
    root = _repo_root() / "pathfinding/results/mapf9_primary_execution"
    return all(
        (root / map_stem / "instance_results.csv").is_file()
        for map_stem in ("AR0400SR", "AR0307SR")
    )


def _method(
    *,
    success: bool = True,
    soc: int = 10,
    makespan: int = 5,
    candidate_id: int = 0,
    additional: int = 0,
) -> MethodOutcome:
    return MethodOutcome(
        success=success,
        termination_reason="success" if success else "failure",
        selected_candidate_id=candidate_id,
        soc=soc if success else None,
        makespan=makespan if success else None,
        final_conflict_count=0 if success else None,
        additional_pp_eval_count=additional,
        logical_pp_candidate_count=1 + additional,
        logical_runtime_ms=100.0,
        baseline_spf_pp_time_ms=50.0,
        method_specific_runtime_ms=50.0,
    )


def test_soc_improvement_sign_convention() -> None:
    assert soc_improvement(10, 8) == 2
    assert soc_improvement(10, 10) == 0
    assert soc_improvement(10, 12) == -2


def test_makespan_only_classification() -> None:
    spf = _method(soc=100, makespan=50)
    improved = _method(soc=100, makespan=49, candidate_id=1, additional=1)
    soc_only = _method(soc=99, makespan=49, candidate_id=1, additional=1)
    assert is_makespan_only_improvement(spf, improved)
    assert not is_makespan_only_improvement(spf, soc_only)


def test_lexicographic_prefers_lower_soc_then_makespan() -> None:
    spf = _method(soc=100, makespan=50)
    better_soc = _method(soc=99, makespan=50, candidate_id=1, additional=1)
    better_ms = _method(soc=100, makespan=49, candidate_id=1, additional=1)
    assert compare_lexicographic(better_soc, spf) == "left_better"
    assert compare_lexicographic(better_ms, spf) == "left_better"
    assert compare_lexicographic(spf, better_ms) == "right_better"


def test_instance_key_requires_manifest_sha256() -> None:
    key_a = MAPF9InstanceKey("sha-a", "AR0400SR_n05_low_000")
    key_b = MAPF9InstanceKey("sha-b", "AR0400SR_n05_low_000")
    assert key_a != key_b


@pytest.mark.skipif(not _production_available(), reason="Production results missing")
def test_pooled_dataset_counts_from_production() -> None:
    config = default_mapf9_primary_analysis_config(_repo_root())
    datasets, report, _run_logs = validate_and_load_all(config)
    assert report.passed
    pooled = [bundle for bundles in datasets.values() for bundle in bundles]
    assert len(pooled) == EXPECTED_POOLED_INSTANCES == 54
    method_rows = sum(len(bundle.methods) for bundle in pooled)
    assert method_rows == EXPECTED_POOLED_METHOD_ROWS == 162
    assert len({(b.manifest_sha256, b.instance_id) for b in pooled}) == 54


@pytest.mark.skipif(not _production_available(), reason="Production results missing")
def test_budget_totals_match_frozen_expectations() -> None:
    config = default_mapf9_primary_analysis_config(_repo_root())
    datasets, report, _run_logs = validate_and_load_all(config)
    assert report.passed
    for map_stem in ("AR0400SR", "AR0307SR"):
        bundles = datasets[map_stem]
        assert sum(bundle.actual_a_i for bundle in bundles) == EXPECTED_SUM_A_I[map_stem]
        assert (
            sum(bundle.physical_pp_eval_count for bundle in bundles)
            == EXPECTED_PHYSICAL_PP[map_stem]
        )
    pooled = [bundle for bundles in datasets.values() for bundle in bundles]
    assert sum(bundle.actual_a_i for bundle in pooled) == EXPECTED_SUM_A_I["pooled"]
    assert sum(bundle.physical_pp_eval_count for bundle in pooled) == EXPECTED_POOLED_PHYSICAL_PP


@pytest.mark.skipif(not _production_available(), reason="Production results missing")
def test_each_instance_has_exactly_three_methods() -> None:
    config = default_mapf9_primary_analysis_config(_repo_root())
    datasets, report, _ = validate_and_load_all(config)
    assert report.passed
    for bundle in (b for bs in datasets.values() for b in bs):
        assert set(bundle.methods) == {"spf", "cglps", "ubls"}


@pytest.mark.skipif(not _production_available(), reason="Production results missing")
def test_successful_methods_are_conflict_free() -> None:
    config = default_mapf9_primary_analysis_config(_repo_root())
    datasets, report, _ = validate_and_load_all(config)
    assert report.passed
    for bundle in (b for bs in datasets.values() for b in bs):
        for method in bundle.methods.values():
            if method.success:
                assert method.final_conflict_count == 0
                assert method.soc is not None
                assert method.makespan is not None


@pytest.mark.skipif(not _production_available(), reason="Production results missing")
def test_physical_pp_formula_on_production() -> None:
    config = default_mapf9_primary_analysis_config(_repo_root())
    datasets, report, _ = validate_and_load_all(config)
    assert report.passed
    for bundle in (b for bs in datasets.values() for b in bs):
        assert bundle.physical_pp_eval_count == 1 + 2 * bundle.actual_a_i


@pytest.mark.skipif(not _production_available(), reason="Production results missing")
def test_active_subset_size_recomputed() -> None:
    config = default_mapf9_primary_analysis_config(_repo_root())
    datasets, report, _ = validate_and_load_all(config)
    assert report.passed
    pooled = [bundle for bundles in datasets.values() for bundle in bundles]
    active = [bundle for bundle in pooled if bundle.actual_a_i > 0]
    assert len(active) == 36


@pytest.mark.skipif(not _production_available(), reason="Production results missing")
def test_cglps_ubls_additional_pp_counts_match() -> None:
    config = default_mapf9_primary_analysis_config(_repo_root())
    datasets, report, _ = validate_and_load_all(config)
    assert report.passed
    for bundle in (b for bs in datasets.values() for b in bs):
        assert bundle.methods["cglps"].additional_pp_eval_count == bundle.actual_a_i
        assert bundle.methods["ubls"].additional_pp_eval_count == bundle.actual_a_i


@pytest.mark.skipif(not _production_available(), reason="Production results missing")
def test_run_analysis_generates_all_artifacts() -> None:
    config = default_mapf9_primary_analysis_config(_repo_root())
    assert run_mapf9_primary_analysis(config) == 0
    expected = (
        "dataset_validation.txt",
        "instance_level_summary.csv",
        "method_summary.csv",
        "soc_pairwise_summary.csv",
        "lexicographic_pairwise_summary.csv",
        "improved_instances.csv",
        "budget_summary.csv",
        "candidate_selection_summary.csv",
        "candidate_level_summary.csv",
        "conflict_structure_summary.csv",
        "runtime_summary.csv",
        "cost_benefit_summary.csv",
        "research_questions.md",
        "analysis_summary.md",
    )
    for name in expected:
        assert (config.output_dir / name).is_file(), name


@pytest.mark.skipif(not _production_available(), reason="Production results missing")
def test_instance_level_rows_use_manifest_sha_identity() -> None:
    config = default_mapf9_primary_analysis_config(_repo_root())
    datasets, report, _ = validate_and_load_all(config)
    assert report.passed
    pooled = [bundle for bundles in datasets.values() for bundle in bundles]
    rows = build_instance_level_rows(pooled)
    assert len(rows) == 54
    keys = {(row["manifest_sha256"], row["instance_id"]) for row in rows}
    assert len(keys) == 54


@pytest.mark.skipif(not _production_available(), reason="Production results missing")
def test_pooled_logical_pp_candidate_counts() -> None:
    config = default_mapf9_primary_analysis_config(_repo_root())
    datasets, report, _ = validate_and_load_all(config)
    assert report.passed
    pooled = [bundle for bundles in datasets.values() for bundle in bundles]
    assert sum(bundle.methods["spf"].logical_pp_candidate_count for bundle in pooled) == 54
    assert sum(bundle.methods["cglps"].logical_pp_candidate_count for bundle in pooled) == 129
    assert sum(bundle.methods["ubls"].logical_pp_candidate_count for bundle in pooled) == 129


@pytest.mark.skipif(not _production_available(), reason="Production results missing")
def test_map_level_budget_rows_sum_to_pooled() -> None:
    config = default_mapf9_primary_analysis_config(_repo_root())
    datasets, report, _ = validate_and_load_all(config)
    assert report.passed
    per_map = []
    for map_stem in ("AR0400SR", "AR0307SR"):
        rows = build_budget_rows(datasets[map_stem], scope=map_stem)
        per_map.append(next(row["value"] for row in rows if row.get("metric") == "sum_actual_a_i"))
    pooled = build_budget_rows(
        [bundle for bundles in datasets.values() for bundle in bundles],
        scope="pooled",
    )
    pooled_sum = next(row["value"] for row in pooled if row.get("metric") == "sum_actual_a_i")
    assert sum(per_map) == pooled_sum == 75


def test_join_by_instance_id_alone_is_insufficient() -> None:
    key_same_text = MAPF9InstanceKey("hash-a", "AR0400SR_n05_low_000")
    key_same_text_other = MAPF9InstanceKey("hash-b", "AR0400SR_n05_low_000")
    assert key_same_text.instance_id == key_same_text_other.instance_id
    assert key_same_text != key_same_text_other


@pytest.mark.skipif(not _production_available(), reason="Production results missing")
def test_manifest_sha_verified_before_analysis() -> None:
    repo = _repo_root()
    for map_stem, expected_sha in (
        (
            "AR0400SR",
            "14964ee449b826c6280299f68a6cfc0cc42d34f3045d87070971027009dbb71c",
        ),
        (
            "AR0307SR",
            "a8bec75ead3a9f18a6d4e99fa05aa5a2f4291ee289b47debf5533ec06dc18403",
        ),
    ):
        path = repo / "pathfinding/results/mapf_benchmarks" / f"{map_stem}_mapf9_manifest.json"
        assert manifest_file_sha256(path) == expected_sha


@pytest.mark.skipif(not _production_available(), reason="Production results missing")
def test_per_map_row_counts() -> None:
    repo = _repo_root()
    for map_stem in ("AR0400SR", "AR0307SR"):
        execution_dir = repo / "pathfinding/results/mapf9_primary_execution" / map_stem
        manifest = load_benchmark_manifest(
            repo / "pathfinding/results/mapf_benchmarks" / f"{map_stem}_mapf9_manifest.json"
        )
        sha = manifest_file_sha256(
            repo / "pathfinding/results/mapf_benchmarks" / f"{map_stem}_mapf9_manifest.json"
        )
        bundles, report = load_map_dataset(
            map_stem=map_stem,
            execution_dir=execution_dir,
            manifest=manifest,
            manifest_sha256=sha,
        )
        assert report.passed
        assert len(bundles) == 27


def test_spf_failure_recovery_classification() -> None:
    from pathfinding.src.experiments.mapf9_primary_analysis import InstanceBundle

    spf = _method(success=False, soc=0, makespan=0)
    recovered = _method(soc=10, makespan=5, candidate_id=1, additional=1)
    bundle = InstanceBundle(
        map_stem="AR0400SR",
        manifest_sha256="abc",
        catalogue_seed=2030,
        instance_id="test",
        agent_count=5,
        interaction_level="low",
        preprocessing_success=True,
        actual_a_i=1,
        expected_a_i=1,
        physical_pp_eval_count=3,
        budget_match_ok=True,
        manifest_independent_conflict_count=0,
        manifest_conflicting_agent_pair_count=0,
        max_conflict_degree=0,
        independent_conflict_count=0,
        conflicting_agent_pair_count=0,
        independent_path_time_ms=1.0,
        conflict_detection_time_ms=1.0,
        spf_ordering_time_ms=1.0,
        baseline_spf_pp_time_ms=1.0,
        cglps_candidate_ranking_time_ms=1.0,
        cglps_additional_pp_time_ms=1.0,
        ubls_candidate_generation_time_ms=1.0,
        ubls_additional_pp_time_ms=1.0,
        physical_instance_wall_time_ms=1.0,
        methods={"spf": spf, "cglps": recovered, "ubls": _method(success=False, soc=0, makespan=0)},
        audit={},
    )
    row = build_instance_level_rows([bundle])[0]
    assert row["recovered_by_cglps"] is True
    assert row["recovered_by_ubls"] is False


def test_cglps_vs_ubls_soc_sign_convention() -> None:
    cglps = _method(soc=10)
    ubls = _method(soc=12, candidate_id=1, additional=1)
    assert cglps.soc - ubls.soc < 0


@pytest.mark.skipif(not _production_available(), reason="Production results missing")
def test_historical_denominator_not_mixed_into_primary() -> None:
    config = default_mapf9_primary_analysis_config(_repo_root())
    datasets, report, _ = validate_and_load_all(config)
    assert report.passed
    pooled = [bundle for bundles in datasets.values() for bundle in bundles]
    assert len(pooled) == 54
    assert all(bundle.catalogue_seed in {2030, 2031} for bundle in pooled)


@pytest.mark.skipif(not _production_available(), reason="Production results missing")
def test_analysis_is_deterministic() -> None:
    config = default_mapf9_primary_analysis_config(_repo_root())
    assert run_mapf9_primary_analysis(config) == 0
    first = (config.output_dir / "instance_level_summary.csv").read_text(encoding="utf-8")
    assert run_mapf9_primary_analysis(config) == 0
    second = (config.output_dir / "instance_level_summary.csv").read_text(encoding="utf-8")
    assert first == second


def _production_analysis_artifacts():
    config = default_mapf9_primary_analysis_config(_repo_root())
    datasets, report, _ = validate_and_load_all(config)
    assert report.passed
    pooled = [bundle for bundles in datasets.values() for bundle in bundles]
    instance_rows = build_instance_level_rows(pooled)
    improved_rows = build_improved_instances_rows(pooled)
    candidate_selection_rows = build_candidate_selection_rows(pooled)
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
                    if bundle.methods["cglps"].soc is not None
                    and bundle.methods["ubls"].soc is not None
                    and bundle.methods["cglps"].soc < bundle.methods["ubls"].soc
                ),
                "equal_count": sum(
                    1
                    for bundle in bundles
                    if bundle.methods["cglps"].soc == bundle.methods["ubls"].soc
                ),
                "ubls_better_count": sum(
                    1
                    for bundle in bundles
                    if bundle.methods["cglps"].soc is not None
                    and bundle.methods["ubls"].soc is not None
                    and bundle.methods["cglps"].soc > bundle.methods["ubls"].soc
                ),
            }
        )
        for beneficiary, other in (("cglps", "ubls"), ("ubls", "cglps")):
            soc_rows.append(
                _unique_benefit_counts(
                    bundles,
                    beneficiary=beneficiary,
                    other=other,
                    scope=map_stem,
                )
            )
    return datasets, instance_rows, improved_rows, soc_rows, lex_rows, candidate_selection_rows


@pytest.mark.skipif(not _production_available(), reason="Production results missing")
def test_internal_consistency_invariants_pass() -> None:
    datasets, instance_rows, improved_rows, soc_rows, lex_rows, candidate_selection_rows = (
        _production_analysis_artifacts()
    )
    errors = verify_analysis_consistency(
        datasets=datasets,
        instance_rows=instance_rows,
        improved_rows=improved_rows,
        soc_rows=soc_rows,
        lex_rows=lex_rows,
        candidate_selection_rows=candidate_selection_rows,
    )
    assert errors == []


@pytest.mark.skipif(not _production_available(), reason="Production results missing")
def test_per_map_cglps_soc_counts_sum_to_pooled() -> None:
    datasets, _, _, soc_rows, _, _ = _production_analysis_artifacts()
    per_map_sum = sum(
        int(_soc_comparison_summary(datasets[m], method_name="cglps", scope=m)["soc_improved_count"])
        for m in MAPF9_MAPS
    )
    pooled = next(
        row for row in soc_rows if row["scope"] == "pooled" and row["comparison"] == "CGLPS_vs_SPF"
    )
    assert per_map_sum == int(pooled["soc_improved_count"]) == 2
    assert (
        int(_soc_comparison_summary(datasets["AR0400SR"], method_name="cglps", scope="AR0400SR")["soc_improved_count"])
        == 1
    )
    assert (
        int(_soc_comparison_summary(datasets["AR0307SR"], method_name="cglps", scope="AR0307SR")["soc_improved_count"])
        == 1
    )


@pytest.mark.skipif(not _production_available(), reason="Production results missing")
def test_unique_cglps_lex_benefit_is_four() -> None:
    datasets, _, _, soc_rows, _, _ = _production_analysis_artifacts()
    unique = next(
        row
        for row in soc_rows
        if row.get("scope") == "pooled"
        and row.get("beneficiary") == "CGLPS"
        and row.get("other_method") == "UBLS"
    )
    assert int(unique["unique_lex_advantage_count"]) == 4
    assert int(unique["unique_soc_advantage_count"]) == 1
    pooled_bundles = [bundle for bundles in datasets.values() for bundle in bundles]
    recomputed = _unique_benefit_counts(
        pooled_bundles, beneficiary="cglps", other="ubls", scope="pooled"
    )
    assert int(recomputed["unique_lex_advantage_count"]) == 4


@pytest.mark.skipif(not _production_available(), reason="Production results missing")
def test_cglps_candidate_selection_distribution() -> None:
    datasets, _, _, _, _, candidate_selection_rows = _production_analysis_artifacts()
    pooled = [bundle for bundles in datasets.values() for bundle in bundles]
    assert _candidate_selection_distribution(pooled, "cglps") == {0: 49, 1: 4, 2: 1}
    assert _candidate_selection_distribution(pooled, "ubls") == {0: 53, 1: 1}
    non_baseline = sum(
        1 for row in candidate_selection_rows if row["method"] == "cglps" and row["selected_non_baseline"]
    )
    assert non_baseline == 5


@pytest.mark.skipif(not _production_available(), reason="Production results missing")
def test_cglps_lex_improved_instances_match_expected_set() -> None:
    _, instance_rows, improved_rows, _, lex_rows, _ = _production_analysis_artifacts()
    expected = {
        "AR0400SR_n05_high_000",
        "AR0307SR_n05_high_000",
        "AR0307SR_n05_high_002",
        "AR0307SR_n20_high_000",
        "AR0307SR_n20_medium_002",
    }
    actual = {
        row["instance_id"]
        for row in instance_rows
        if row.get("cglps_lex_vs_spf") == "left_better"
    }
    assert actual == expected
    pooled_lex = next(
        row for row in lex_rows if row["scope"] == "pooled" and row["comparison"] == "CGLPS_vs_SPF"
    )
    assert int(pooled_lex["left_better_count"]) == 5
    cglps_improved = {
        row["instance_id"] for row in improved_rows if row["method"] == "cglps"
    }
    assert cglps_improved == expected


@pytest.mark.skipif(not _production_available(), reason="Production results missing")
def test_cglps_vs_ubls_direct_comparison() -> None:
    _, instance_rows, _, soc_rows, lex_rows, _ = _production_analysis_artifacts()
    soc = next(row for row in soc_rows if row["scope"] == "pooled" and row["comparison"] == "CGLPS_vs_UBLS_soc")
    lex = next(row for row in lex_rows if row["scope"] == "pooled" and row["comparison"] == "CGLPS_vs_UBLS")
    assert int(soc["cglps_better_count"]) == 1
    assert int(soc["equal_count"]) == 53
    assert int(soc["ubls_better_count"]) == 0
    assert int(lex["left_better_count"]) == 4
    assert int(lex["equal_count"]) == 50
    assert int(lex["right_better_count"]) == 0


@pytest.mark.skipif(not _production_available(), reason="Production results missing")
def test_soc_plus_makespan_only_equals_lex_without_recovery() -> None:
    datasets, _, _, _, _, _ = _production_analysis_artifacts()
    pooled = [bundle for bundles in datasets.values() for bundle in bundles]
    for method_name in ("cglps", "ubls"):
        soc_summary = _soc_comparison_summary(pooled, method_name=method_name, scope="pooled")
        lex_summary = _lex_comparison_summary(
            pooled,
            left_method=method_name,
            right_method="spf",
            scope="pooled",
        )
        assert int(soc_summary["soc_improved_count"]) + int(
            soc_summary["makespan_only_improved_count"]
        ) == int(lex_summary["left_better_count"])

