"""Tests for MAPF-9.4 primary production execution harness."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pathfinding.src.algorithms.mapf.models import MAPFAgent, MAPFScenario
from pathfinding.src.core.models import Position
from pathfinding.src.experiments.mapf_benchmark_instances import (
    CATALOGUE_ROLE_MAPF9,
    MAPF9_CATALOGUE_GENERATION_VERSION,
    MAPF9_CATALOGUE_SEEDS,
    MAPFBenchmarkManifest,
    load_benchmark_manifest,
    save_benchmark_manifest,
)
from pathfinding.src.experiments.mapf_bounded_local_priority_search import (
    MAPF9CandidateEvaluation,
    MAPF9IntegrationResult,
    MAPF9Method,
    MAPF9MethodResult,
    MAPF9MethodTimings,
    MAPF9PreprocessingTimings,
    MAPF9SharedInstance,
)
from pathfinding.src.experiments.mapf_benchmark_execution import TERMINATION_SUCCESS
from pathfinding.src.experiments.mapf9_primary_execution import (
    MAPF9_EXPECTED_INSTANCES_PER_MAP,
    MAPF9_FROZEN_MANIFEST_SHA256,
    MAPF9PrimaryCheckpointStore,
    MAPF9PrimaryExecutionConfig,
    BudgetDiagnosticMismatchError,
    build_budget_plan,
    build_records_from_integration_result,
    instance_key,
    mapf9_primary_execution_config,
    resolve_mapf9_primary_target,
    run_mapf9_primary_benchmark,
    validate_checkpoint_state,
    verify_frozen_mapf9_manifest,
)
from pathfinding.src.experiments.mapf_cross_map_catalogue_validation import (
    manifest_file_sha256,
)
from pathfinding.tests.unit.test_mapf_cross_map_catalogue_validation import (
    _synthetic_27_instance_manifest,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _mock_grid_map(map_stem: str) -> MagicMock:
    grid_map = MagicMock()
    grid_map.name = f"{map_stem}.map"
    return grid_map


def _agent(agent_id: int) -> MAPFAgent:
    return MAPFAgent(
        agent_id=agent_id,
        start=Position(row=0, col=0),
        goal=Position(row=0, col=1),
    )


def _evaluation(
    candidate_id: int,
    *,
    success: bool = True,
    soc: int = 10,
    makespan: int = 5,
) -> MAPF9CandidateEvaluation:
    return MAPF9CandidateEvaluation(
        candidate_id=candidate_id,
        agent_order=(0, 1),
        source_agent_a_id=None,
        source_agent_b_id=None,
        canonical_original_index_pair=None,
        pair_conflict_event_count=None,
        success=success,
        termination_reason=TERMINATION_SUCCESS if success else "failure",
        soc=soc if success else None,
        makespan=makespan if success else None,
        conflict_count=0 if success else None,
        pp_time_ms=1.0,
        pp_search_metrics=None,
    )


def _method_result(
    method: MAPF9Method,
    *,
    additional: int = 0,
    success: bool = True,
    soc: int = 10,
) -> MAPF9MethodResult:
    baseline = _evaluation(0, success=success, soc=soc)
    extras = tuple(
        _evaluation(index + 1, success=success, soc=soc - 1)
        for index in range(additional)
    )
    evaluations = (baseline, *extras)
    selected = evaluations[0]
    timings = MAPF9MethodTimings(baseline_spf_pp_time_ms=2.0)
    if method == MAPF9Method.CGLPS:
        timings = MAPF9MethodTimings(
            baseline_spf_pp_time_ms=2.0,
            cglps_candidate_ranking_time_ms=0.1,
            cglps_additional_pp_time_ms=float(additional),
        )
    elif method == MAPF9Method.UBLS:
        timings = MAPF9MethodTimings(
            baseline_spf_pp_time_ms=2.0,
            ubls_candidate_generation_time_ms=0.2,
            ubls_additional_pp_time_ms=float(additional),
        )
    return MAPF9MethodResult(
        method=method,
        selected_candidate_id=selected.candidate_id,
        selected_agent_order=selected.agent_order,
        success=selected.success,
        termination_reason=selected.termination_reason,
        soc=selected.soc,
        makespan=selected.makespan,
        conflict_count=selected.conflict_count,
        baseline_spf_evaluation=baseline,
        additional_pp_eval_count=additional,
        total_logical_pp_candidate_count=1 + additional,
        additional_candidate_specs=(),
        candidate_evaluations=evaluations,
        independent_conflict_count=1,
        independent_conflict_pair_count=1,
        preprocessing_timings=MAPF9PreprocessingTimings(1.0, 0.5, 0.1),
        method_timings=timings,
    )


def _shared_stub() -> MAPF9SharedInstance:
    from pathfinding.tests.helpers import build_grid_map

    scenario = MAPFScenario(agents=(_agent(0), _agent(1)))
    from pathfinding.src.algorithms.mapf.priority_ordering import (
        ConflictAwareOrderingInputs,
    )

    return MAPF9SharedInstance(
        grid_map=build_grid_map([[0, 0], [0, 0]]),
        scenario=scenario,
        max_timestep=10,
        inputs=ConflictAwareOrderingInputs(
            paths=(),
            independent_costs=(1, 1),
            degrees=(1, 1),
            incident_counts=(1, 1),
            conflicts=(),
            conflict_pair_count=1,
        ),
        spf_order=(0, 1),
        original_index_by_agent_id={0: 0, 1: 1},
        agent_id_by_original_index={0: 0, 1: 1},
        conflict_edges=frozenset({(0, 1)}),
        pair_event_counts={(0, 1): 1},
        preprocessing_timings=MAPF9PreprocessingTimings(1.0, 0.5, 0.1),
    )


def _integration(
    instance_id: str,
    *,
    actual_a_i: int = 1,
    spf_soc: int = 10,
    cglps_soc: int = 10,
) -> MAPF9IntegrationResult:
    return MAPF9IntegrationResult(
        instance_id=instance_id,
        shared=_shared_stub(),
        ordering_failure=None,
        spf=_method_result(MAPF9Method.SPF, success=True, soc=spf_soc),
        cglps=_method_result(
            MAPF9Method.CGLPS,
            additional=actual_a_i,
            success=True,
            soc=cglps_soc,
        ),
        ubls=_method_result(
            MAPF9Method.UBLS,
            additional=actual_a_i,
            success=True,
            soc=spf_soc,
        ),
        actual_a_i=actual_a_i,
        physical_pp_eval_count=1 + 2 * actual_a_i,
    )


def _mapf9_manifest_for_tests(map_stem: str) -> MAPFBenchmarkManifest:
    base = _synthetic_27_instance_manifest(
        map_stem=map_stem,
        seed=MAPF9_CATALOGUE_SEEDS[map_stem],
    )
    return replace(
        base,
        map_name=f"{map_stem}.map",
        scenario_name=f"{map_stem}.map.scen",
        catalogue_role=CATALOGUE_ROLE_MAPF9,
        generation_version=MAPF9_CATALOGUE_GENERATION_VERSION,
    )


def _execution_config(tmp_path: Path, map_stem: str) -> MAPF9PrimaryExecutionConfig:
    results_dir = (
        tmp_path / "pathfinding" / "results" / "mapf9_primary_execution" / map_stem
    )
    manifest_path = tmp_path / f"{map_stem}_mapf9_manifest.json"
    save_benchmark_manifest(_mapf9_manifest_for_tests(map_stem), manifest_path)
    manifest_sha256 = manifest_file_sha256(manifest_path)
    return MAPF9PrimaryExecutionConfig(
        map_stem=map_stem,
        manifest_path=manifest_path,
        map_path=tmp_path / f"{map_stem}.map",
        scen_path=tmp_path / f"{map_stem}.map.scen",
        results_dir=results_dir,
        instance_csv_path=results_dir / "instance_results.csv",
        method_csv_path=results_dir / "method_results.csv",
        candidate_jsonl_path=results_dir / "candidate_details.jsonl",
        log_file_path=results_dir / "run.log",
        expected_manifest_sha256=manifest_sha256,
        expected_seed=MAPF9_CATALOGUE_SEEDS[map_stem],
    )


def _config_for_manifest_path(
    tmp_path: Path,
    map_stem: str,
    manifest_path: Path,
) -> MAPF9PrimaryExecutionConfig:
    base = _execution_config(tmp_path, map_stem)
    return replace(
        base,
        manifest_path=manifest_path,
        expected_manifest_sha256=manifest_file_sha256(manifest_path),
    )


def test_allowed_maps_and_frozen_registry() -> None:
    assert resolve_mapf9_primary_target("AR0400SR") == "AR0400SR"
    assert resolve_mapf9_primary_target("AR0307SR") == "AR0307SR"
    with pytest.raises(ValueError, match="AR0204SR"):
        resolve_mapf9_primary_target("AR0204SR")
    with pytest.raises(ValueError, match="AR0404SR"):
        resolve_mapf9_primary_target("AR0404SR")


def test_frozen_config_hashes_and_seeds() -> None:
    repo = _repo_root()
    cfg400 = mapf9_primary_execution_config("AR0400SR", repo)
    cfg307 = mapf9_primary_execution_config("AR0307SR", repo)
    assert cfg400.expected_seed == 2030
    assert cfg307.expected_seed == 2031
    assert cfg400.expected_manifest_sha256 == MAPF9_FROZEN_MANIFEST_SHA256["AR0400SR"]
    assert cfg307.expected_manifest_sha256 == MAPF9_FROZEN_MANIFEST_SHA256["AR0307SR"]
    assert cfg400.manifest_path.name == "AR0400SR_mapf9_manifest.json"


def test_wrong_mapf8_manifest_rejected(tmp_path: Path) -> None:
    mapf8 = _synthetic_27_instance_manifest(map_stem="AR0400SR", seed=2028)
    path = tmp_path / "AR0400SR_manifest.json"
    save_benchmark_manifest(mapf8, path)
    config = replace(
        _execution_config(tmp_path, "AR0400SR"),
        manifest_path=path,
    )
    with pytest.raises(ValueError, match="Refusing MAPF-8"):
        verify_frozen_mapf9_manifest(config)


def test_wrong_hash_rejected(tmp_path: Path) -> None:
    config = _execution_config(tmp_path, "AR0400SR")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_frozen_mapf9_manifest(
            replace(config, expected_manifest_sha256="0" * 64)
        )


def test_wrong_seed_rejected(tmp_path: Path) -> None:
    manifest = replace(_mapf9_manifest_for_tests("AR0400SR"), seed=9999)
    path = tmp_path / "bad_seed.json"
    save_benchmark_manifest(manifest, path)
    config = _config_for_manifest_path(tmp_path, "AR0400SR", path)
    with pytest.raises(ValueError, match="Expected seed 2030"):
        verify_frozen_mapf9_manifest(config)


def test_wrong_catalogue_role_rejected(tmp_path: Path) -> None:
    manifest = replace(
        _mapf9_manifest_for_tests("AR0400SR"),
        catalogue_role="primary",
    )
    path = tmp_path / "bad_role.json"
    save_benchmark_manifest(manifest, path)
    config = _config_for_manifest_path(tmp_path, "AR0400SR", path)
    with pytest.raises(ValueError, match="catalogue_role"):
        verify_frozen_mapf9_manifest(config)


def test_manifest_requires_27_instances(tmp_path: Path) -> None:
    manifest = replace(
        _mapf9_manifest_for_tests("AR0400SR"),
        instances=_mapf9_manifest_for_tests("AR0400SR").instances[:26],
    )
    path = tmp_path / "short_manifest.json"
    save_benchmark_manifest(manifest, path)
    config = _config_for_manifest_path(tmp_path, "AR0400SR", path)
    with pytest.raises(ValueError, match=str(MAPF9_EXPECTED_INSTANCES_PER_MAP)):
        verify_frozen_mapf9_manifest(config)


def test_combined_expected_physical_pp_is_204() -> None:
    bench = _repo_root() / "pathfinding/results/mapf_benchmarks"
    if not (bench / "AR0400SR_mapf9_manifest.json").is_file():
        pytest.skip("Frozen MAPF-9 manifests not present")
    total_pp = 0
    for map_stem in ("AR0400SR", "AR0307SR"):
        manifest = load_benchmark_manifest(bench / f"{map_stem}_mapf9_manifest.json")
        total_pp += build_budget_plan(manifest).expected_physical_pp_eval_count
    assert total_pp == 204


def test_budget_plan_ar0400_and_ar0307_from_repo_manifests() -> None:
    bench = _repo_root() / "pathfinding/results/mapf_benchmarks"
    if not (bench / "AR0400SR_mapf9_manifest.json").is_file():
        pytest.skip("Frozen MAPF-9 manifests not present")
    for map_stem, expected_sum, expected_pp in (
        ("AR0400SR", 33, 93),
        ("AR0307SR", 42, 111),
    ):
        manifest = load_benchmark_manifest(bench / f"{map_stem}_mapf9_manifest.json")
        plan = build_budget_plan(manifest)
        assert plan.expected_sum_a_i == expected_sum
        assert plan.expected_physical_pp_eval_count == expected_pp


def test_plan_only_writes_no_results_and_no_planner(tmp_path: Path) -> None:
    config = replace(_execution_config(tmp_path, "AR0400SR"), plan_only=True)
    verify_frozen_mapf9_manifest(config)
    with patch(
        "pathfinding.src.experiments.mapf9_primary_execution.load_moving_ai_map"
    ) as load_map:
        exit_code = run_mapf9_primary_benchmark(
            config,
            repo_root=tmp_path,
        )
        load_map.assert_not_called()
    assert exit_code == 0
    assert not config.instance_csv_path.exists()
    assert not config.method_csv_path.exists()
    assert not config.candidate_jsonl_path.exists()
    assert not config.log_file_path.exists()


def test_complete_instance_produces_rows_and_audit(tmp_path: Path) -> None:
    manifest = _mapf9_manifest_for_tests("AR0400SR")
    instance = manifest.instances[0]
    integration = _integration(instance.instance_id, actual_a_i=1)
    instance_record, method_records, audit = build_records_from_integration_result(
        manifest=manifest,
        manifest_sha256="abc",
        benchmark_instance=instance,
        integration=integration,
        expected_a_i=1,
        wall_time_ms=12.3,
    )
    assert instance_record.physical_pp_eval_count == 3
    assert len(method_records) == 3
    assert audit.actual_a_i == 1
    assert method_records[0].additional_pp_eval_count == 0
    assert method_records[1].additional_pp_eval_count == 1
    assert method_records[2].additional_pp_eval_count == 1


def test_successful_result_requires_zero_conflicts() -> None:
    manifest = _mapf9_manifest_for_tests("AR0400SR")
    instance = manifest.instances[0]
    integration = _integration(instance.instance_id)
    bad_spf = replace(
        integration.spf,
        success=True,
        conflict_count=1,
        soc=10,
        makespan=5,
    )
    integration = replace(integration, spf=bad_spf)
    with pytest.raises(RuntimeError, match="conflict_count"):
        build_records_from_integration_result(
            manifest=manifest,
            manifest_sha256="abc",
            benchmark_instance=instance,
            integration=integration,
            expected_a_i=1,
            wall_time_ms=1.0,
        )


def test_budget_mismatch_stops_execution(tmp_path: Path) -> None:
    config = _execution_config(tmp_path, "AR0400SR")
    manifest, _sha = verify_frozen_mapf9_manifest(config)
    instance = manifest.instances[0]

    def evaluate(**_kwargs):
        return _integration(instance.instance_id, actual_a_i=99)

    with patch(
        "pathfinding.src.experiments.mapf9_primary_execution.load_moving_ai_map",
        return_value=_mock_grid_map("AR0400SR"),
    ), patch(
        "pathfinding.src.experiments.mapf9_primary_execution.load_moving_ai_scenarios",
        return_value=[],
    ), patch(
        "pathfinding.src.experiments.mapf9_primary_execution.reconstruct_mapf_scenario_from_instance",
        return_value=MagicMock(scenario=MAPFScenario(agents=(_agent(0),))),
    ):
        with pytest.raises(BudgetDiagnosticMismatchError):
            run_mapf9_primary_benchmark(
                config,
                repo_root=tmp_path,
                evaluate_instance=evaluate,
            )
    assert not config.candidate_jsonl_path.exists()


def test_checkpoint_one_instance_three_methods(tmp_path: Path) -> None:
    config = _execution_config(tmp_path, "AR0400SR")
    manifest = _mapf9_manifest_for_tests("AR0400SR")
    instance = manifest.instances[0]
    integration = _integration(instance.instance_id, actual_a_i=1)
    instance_record, method_records, audit = build_records_from_integration_result(
        manifest=manifest,
        manifest_sha256=config.expected_manifest_sha256,
        benchmark_instance=instance,
        integration=integration,
        expected_a_i=1,
        wall_time_ms=1.0,
    )
    store = MAPF9PrimaryCheckpointStore(config=config)
    store.append_complete_instance(instance_record, method_records, audit)
    assert config.candidate_jsonl_path.is_file()
    assert config.instance_csv_path.is_file()
    assert config.method_csv_path.is_file()
    import csv

    with config.method_csv_path.open(encoding="utf-8") as file:
        method_rows = list(csv.DictReader(file))
    assert len(method_rows) == 3


def test_resume_rejects_duplicate_instance(tmp_path: Path) -> None:
    config = _execution_config(tmp_path, "AR0400SR")
    manifest = _mapf9_manifest_for_tests("AR0400SR")
    instance = manifest.instances[0]
    integration = _integration(instance.instance_id, actual_a_i=0)
    instance_record, method_records, audit = build_records_from_integration_result(
        manifest=manifest,
        manifest_sha256=config.expected_manifest_sha256,
        benchmark_instance=instance,
        integration=integration,
        expected_a_i=0,
        wall_time_ms=1.0,
    )
    store = MAPF9PrimaryCheckpointStore(config=config)
    store.append_complete_instance(instance_record, method_records, audit)
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        store.append_complete_instance(instance_record, method_records, audit)


def test_resume_key_uses_manifest_sha256() -> None:
    record = build_records_from_integration_result(
        manifest=_mapf9_manifest_for_tests("AR0400SR"),
        manifest_sha256="deadbeef",
        benchmark_instance=_mapf9_manifest_for_tests("AR0400SR").instances[0],
        integration=_integration("AR0400SR_n05_low_000", actual_a_i=0),
        expected_a_i=0,
        wall_time_ms=1.0,
    )[2]
    key = instance_key(record)
    assert key.manifest_sha256 == "deadbeef"
    assert key.instance_id == "AR0400SR_n05_low_000"


def test_reset_only_clears_target_map_directory(tmp_path: Path) -> None:
    map_a = tmp_path / "pathfinding" / "results" / "mapf9_primary_execution" / "AR0400SR"
    map_b = tmp_path / "pathfinding" / "results" / "mapf9_primary_execution" / "AR0307SR"
    map_a.mkdir(parents=True)
    map_b.mkdir(parents=True)
    old = map_a / "candidate_details.jsonl"
    old.write_text("{}\n", encoding="utf-8")
    other = map_b / "candidate_details.jsonl"
    other.write_text("{}\n", encoding="utf-8")
    config = replace(
        _execution_config(tmp_path, "AR0400SR"),
        results_dir=map_a,
        candidate_jsonl_path=map_a / "candidate_details.jsonl",
        reset_results=True,
    )
    MAPF9PrimaryCheckpointStore(config=config, reset_results=True)
    assert not old.exists()
    assert other.exists()


def test_frozen_mapf8_manifest_bytes_unchanged() -> None:
    repo = _repo_root()
    path = repo / "pathfinding/results/mapf_benchmarks/AR0400SR_manifest.json"
    if not path.is_file():
        pytest.skip("MAPF-8 manifest missing")
    expected = "ab652d77ad99c251d7827684b7b2978a3bd5151609c5b557840eec8522a54259"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_plan_only_on_frozen_ar0400_manifest() -> None:
    repo = _repo_root()
    manifest_path = repo / "pathfinding/results/mapf_benchmarks/AR0400SR_mapf9_manifest.json"
    if not manifest_path.is_file():
        pytest.skip("Frozen MAPF-9 manifest missing")
    config = replace(
        mapf9_primary_execution_config("AR0400SR", repo, plan_only=True),
        results_dir=repo / "pathfinding/results/mapf9_primary_execution/AR0400SR",
    )
    exit_code = run_mapf9_primary_benchmark(config, repo_root=repo)
    assert exit_code == 0


def test_mocked_run_calls_evaluate_once_per_instance(tmp_path: Path) -> None:
    config = _execution_config(tmp_path, "AR0400SR")
    manifest, _sha = verify_frozen_mapf9_manifest(config)
    evaluate = MagicMock(
        side_effect=lambda **kwargs: _integration(
            kwargs["instance_id"],
            actual_a_i=build_budget_plan(manifest).expected_a_i_by_instance_id[
                kwargs["instance_id"]
            ],
        )
    )
    with patch(
        "pathfinding.src.experiments.mapf9_primary_execution.load_moving_ai_map",
        return_value=_mock_grid_map("AR0400SR"),
    ), patch(
        "pathfinding.src.experiments.mapf9_primary_execution.load_moving_ai_scenarios",
        return_value=[],
    ), patch(
        "pathfinding.src.experiments.mapf9_primary_execution.reconstruct_mapf_scenario_from_instance",
        return_value=MagicMock(scenario=MAPFScenario(agents=(_agent(0),))),
    ):
        run_mapf9_primary_benchmark(
            config,
            repo_root=tmp_path,
            evaluate_instance=evaluate,
        )
    assert evaluate.call_count == len(manifest.instances)


def test_validate_checkpoint_state_rejects_wrong_manifest_hash(tmp_path: Path) -> None:
    config = _execution_config(tmp_path, "AR0400SR")
    manifest = _mapf9_manifest_for_tests("AR0400SR")
    store = MAPF9PrimaryCheckpointStore(config=config)
    integration = _integration(manifest.instances[0].instance_id, actual_a_i=0)
    instance_record, method_records, audit = build_records_from_integration_result(
        manifest=manifest,
        manifest_sha256="wrong",
        benchmark_instance=manifest.instances[0],
        integration=integration,
        expected_a_i=0,
        wall_time_ms=1.0,
    )
    store.append_complete_instance(instance_record, method_records, audit)
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        validate_checkpoint_state(
            store,
            manifest=manifest,
            manifest_sha256=config.expected_manifest_sha256,
            budget_plan=build_budget_plan(manifest),
        )


def test_incomplete_method_checkpoint_rejected(tmp_path: Path) -> None:
    config = _execution_config(tmp_path, "AR0400SR")
    manifest = _mapf9_manifest_for_tests("AR0400SR")
    integration = _integration(manifest.instances[0].instance_id, actual_a_i=0)
    instance_record, method_records, audit = build_records_from_integration_result(
        manifest=manifest,
        manifest_sha256=config.expected_manifest_sha256,
        benchmark_instance=manifest.instances[0],
        integration=integration,
        expected_a_i=0,
        wall_time_ms=1.0,
    )
    with pytest.raises(RuntimeError, match="exactly 3 method rows"):
        MAPF9PrimaryCheckpointStore(config=config).append_complete_instance(
            instance_record,
            method_records[:1],
            audit,
        )

    store = MAPF9PrimaryCheckpointStore(config=config)
    store.append_complete_instance(instance_record, method_records, audit)
    config.method_csv_path.unlink()
    config.method_csv_path.write_text(
        "manifest_sha256,map_name,catalogue_role,catalogue_seed,instance_id,method,"
        "success,termination_reason,selected_candidate_id,selected_agent_order_json,"
        "soc,makespan,final_conflict_count,additional_pp_eval_count,"
        "logical_pp_candidate_count,logical_runtime_ms,baseline_spf_pp_time_ms,"
        "method_specific_runtime_ms\n",
        encoding="utf-8",
    )
    reloaded = MAPF9PrimaryCheckpointStore(config=config)
    with pytest.raises(ValueError, match="expected 3"):
        validate_checkpoint_state(
            reloaded,
            manifest=manifest,
            manifest_sha256=config.expected_manifest_sha256,
            budget_plan=build_budget_plan(manifest),
        )


def test_resume_skips_completed_and_reruns_remaining(tmp_path: Path) -> None:
    config = _execution_config(tmp_path, "AR0400SR")
    manifest, manifest_sha256 = verify_frozen_mapf9_manifest(config)
    first = manifest.instances[0]
    integration = _integration(first.instance_id, actual_a_i=0)
    instance_record, method_records, audit = build_records_from_integration_result(
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        benchmark_instance=first,
        integration=integration,
        expected_a_i=0,
        wall_time_ms=1.0,
    )
    store = MAPF9PrimaryCheckpointStore(config=config)
    store.append_complete_instance(instance_record, method_records, audit)

    evaluate = MagicMock(
        side_effect=lambda **kwargs: _integration(
            kwargs["instance_id"],
            actual_a_i=build_budget_plan(manifest).expected_a_i_by_instance_id[
                kwargs["instance_id"]
            ],
        )
    )
    resume_config = replace(config, resume=True)
    with patch(
        "pathfinding.src.experiments.mapf9_primary_execution.load_moving_ai_map",
        return_value=_mock_grid_map("AR0400SR"),
    ), patch(
        "pathfinding.src.experiments.mapf9_primary_execution.load_moving_ai_scenarios",
        return_value=[],
    ), patch(
        "pathfinding.src.experiments.mapf9_primary_execution.reconstruct_mapf_scenario_from_instance",
        return_value=MagicMock(scenario=MAPFScenario(agents=(_agent(0),))),
    ):
        run_mapf9_primary_benchmark(
            resume_config,
            repo_root=tmp_path,
            evaluate_instance=evaluate,
        )
    assert evaluate.call_count == len(manifest.instances) - 1


def test_deterministic_csv_row_order(tmp_path: Path) -> None:
    config = _execution_config(tmp_path, "AR0400SR")
    manifest = _mapf9_manifest_for_tests("AR0400SR")
    store = MAPF9PrimaryCheckpointStore(config=config)
    for instance in reversed(manifest.instances[:3]):
        integration = _integration(
            instance.instance_id,
            actual_a_i=build_budget_plan(manifest).expected_a_i_by_instance_id[
                instance.instance_id
            ],
        )
        records = build_records_from_integration_result(
            manifest=manifest,
            manifest_sha256=config.expected_manifest_sha256,
            benchmark_instance=instance,
            integration=integration,
            expected_a_i=build_budget_plan(manifest).expected_a_i_by_instance_id[
                instance.instance_id
            ],
            wall_time_ms=1.0,
        )
        store.append_complete_instance(*records)

    import csv

    with config.instance_csv_path.open(encoding="utf-8") as file:
        instance_ids = [row["instance_id"] for row in csv.DictReader(file)]
    assert instance_ids == sorted(instance_ids)

    with config.method_csv_path.open(encoding="utf-8") as file:
        method_rows = list(csv.DictReader(file))
    method_order = {"spf": 0, "cglps": 1, "ubls": 2}
    grouped = [
        (row["instance_id"], row["method"]) for row in method_rows
    ]
    assert grouped == sorted(
        grouped,
        key=lambda item: (item[0], method_order[item[1]]),
    )


def test_plan_only_ar0307_from_repo() -> None:
    repo = _repo_root()
    manifest_path = repo / "pathfinding/results/mapf_benchmarks/AR0307SR_mapf9_manifest.json"
    if not manifest_path.is_file():
        pytest.skip("Frozen MAPF-9 manifest missing")
    config = replace(
        mapf9_primary_execution_config("AR0307SR", repo, plan_only=True),
        results_dir=repo / "pathfinding/results/mapf9_primary_execution/AR0307SR",
    )
    exit_code = run_mapf9_primary_benchmark(config, repo_root=repo)
    assert exit_code == 0
    plan = build_budget_plan(load_benchmark_manifest(config.manifest_path))
    assert plan.expected_sum_a_i == 42
    assert plan.expected_physical_pp_eval_count == 111
