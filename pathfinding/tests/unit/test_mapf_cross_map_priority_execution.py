from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from pathfinding.src.core.models import Position, Scenario
from pathfinding.src.experiments.mapf_benchmark_instances import (
    MAPFBenchmarkInstance,
    MAPFBenchmarkManifest,
    MAPFInteractionLevel,
    load_benchmark_manifest,
)
from pathfinding.src.experiments.mapf_conflict_aware_priority_execution import (
    MAIN_CONFLICT_AWARE_RESULTS_DIR,
)
from pathfinding.src.experiments.mapf_cross_map_priority_execution import (
    DEFAULT_CROSS_MAP_STRATEGIES,
    EXPECTED_RUNS_PER_MAP,
    FROZEN_MANIFEST_SHA256,
    FORBIDDEN_HISTORICAL_RESULTS_DIRS,
    MAPF8_CROSS_MAP_RESULTS_ROOT,
    CrossMapPriorityCheckpointStore,
    CrossMapPriorityRunKey,
    CrossMapPriorityStrategy,
    assert_results_dir_isolated,
    build_cross_map_priority_run_plan,
    execute_cross_map_priority_run,
    load_checkpoint_records,
    mapf8_cross_map_execution_config,
    resolve_mapf8_cross_map_execution_target,
    run_key,
    validate_checkpoint_records,
    verify_frozen_manifest,
)
from pathfinding.src.experiments.mapf_priority_strategy_execution import (
    MAIN_PRIORITY_STRATEGY_RESULTS_DIR,
)
from pathfinding.tests.helpers import build_grid_map


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _instance(
    instance_id: str,
    agent_count: int,
    level: MAPFInteractionLevel = MAPFInteractionLevel.LOW,
) -> MAPFBenchmarkInstance:
    return MAPFBenchmarkInstance(
        instance_id=instance_id,
        agent_count=agent_count,
        interaction_level=level,
        scenario_indices=tuple(range(agent_count)),
        independent_conflict_count=0,
        conflicting_agent_pair_count=0,
        independent_soc=4,
        independent_makespan=2,
        vertex_conflict_count=0,
        edge_conflict_count=0,
    )


def _scenario(
    start_row: int,
    start_col: int,
    goal_row: int,
    goal_col: int,
) -> Scenario:
    return Scenario(
        map_name="bench.map",
        width=5,
        height=5,
        start=Position(row=start_row, col=start_col),
        goal=Position(row=goal_row, col=goal_col),
        optimal_length=25.0,
    )


def _tiny_manifest(*instances: MAPFBenchmarkInstance) -> MAPFBenchmarkManifest:
    return MAPFBenchmarkManifest(
        map_name="bench.map",
        scenario_name="bench.map.scen",
        seed=2028,
        max_timestep=8,
        min_reference_length=1.0,
        instances=instances or (_instance("bench_n02_low_000", 2),),
    )


def _tiny_grid_and_scenarios() -> tuple:
    grid_map = build_grid_map(
        [[0, 0, 0, 0, 0] for _ in range(5)],
        name="bench.map",
    )
    scenarios = [
        _scenario(0, 0, 0, 4),
        _scenario(1, 0, 1, 4),
    ]
    return grid_map, scenarios


@pytest.mark.parametrize(
    ("map_stem", "expected_hash"),
    [
        ("AR0400SR", FROZEN_MANIFEST_SHA256["AR0400SR"]),
        ("AR0307SR", FROZEN_MANIFEST_SHA256["AR0307SR"]),
    ],
)
def test_frozen_manifest_binds_to_expected_hash(map_stem: str, expected_hash: str) -> None:
    repo = _repo_root()
    config = mapf8_cross_map_execution_config(map_stem, repo)
    manifest_path = config.manifest_path
    if not manifest_path.is_file():
        pytest.skip(f"Frozen manifest not present: {manifest_path}")

    assert config.expected_manifest_sha256 == expected_hash
    _, sha256 = verify_frozen_manifest(config)
    assert sha256 == expected_hash


def test_wrong_manifest_hash_is_rejected(tmp_path: Path) -> None:
    repo = _repo_root()
    config = mapf8_cross_map_execution_config("AR0400SR", repo)
    manifest_path = config.manifest_path
    if not manifest_path.is_file():
        pytest.skip("Frozen AR0400SR manifest not present")

    bad_manifest = tmp_path / "bad_manifest.json"
    bad_manifest.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
    bad_config = replace(
        mapf8_cross_map_execution_config("AR0400SR", repo),
        manifest_path=bad_manifest,
        expected_manifest_sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_frozen_manifest(bad_config)


def test_ar0404sr_is_rejected() -> None:
    with pytest.raises(ValueError, match="AR0404SR"):
        resolve_mapf8_cross_map_execution_target("AR0404SR")


def test_ar0204sr_is_rejected() -> None:
    with pytest.raises(ValueError, match="AR0204SR"):
        resolve_mapf8_cross_map_execution_target("AR0204SR")


def test_strategy_set_is_exactly_four_frozen_strategies() -> None:
    assert DEFAULT_CROSS_MAP_STRATEGIES == (
        CrossMapPriorityStrategy.SPF,
        CrossMapPriorityStrategy.CDF_H,
        CrossMapPriorityStrategy.CDF_L,
        CrossMapPriorityStrategy.SPF_CD,
    )


@pytest.mark.parametrize("map_stem", ["AR0400SR", "AR0307SR"])
def test_expected_run_count_is_108_per_map(map_stem: str) -> None:
    manifest_path = (
        _repo_root()
        / "pathfinding"
        / "results"
        / "mapf_benchmarks"
        / f"{map_stem}_manifest.json"
    )
    if not manifest_path.is_file():
        pytest.skip(f"Frozen manifest not present: {manifest_path}")

    manifest = load_benchmark_manifest(manifest_path)
    plan = build_cross_map_priority_run_plan(manifest)
    assert len(manifest.instances) == 27
    assert len(plan) == EXPECTED_RUNS_PER_MAP == 108


def test_deterministic_run_ordering() -> None:
    manifest_path = (
        _repo_root()
        / "pathfinding"
        / "results"
        / "mapf_benchmarks"
        / "AR0400SR_manifest.json"
    )
    if not manifest_path.is_file():
        pytest.skip("Frozen AR0400SR manifest not present")

    manifest = load_benchmark_manifest(manifest_path)
    plan = build_cross_map_priority_run_plan(manifest)

    first_instance = manifest.instances[0].instance_id
    assert plan[0].instance.instance_id == first_instance
    assert plan[0].strategy == CrossMapPriorityStrategy.SPF
    assert plan[1].strategy == CrossMapPriorityStrategy.CDF_H
    assert plan[2].strategy == CrossMapPriorityStrategy.CDF_L
    assert plan[3].strategy == CrossMapPriorityStrategy.SPF_CD

    strategies_for_first = [
        entry.strategy
        for entry in plan
        if entry.instance.instance_id == first_instance
    ]
    assert strategies_for_first == list(DEFAULT_CROSS_MAP_STRATEGIES)


def test_checkpoint_resume_skips_completed_keys(tmp_path: Path) -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    manifest = _tiny_manifest(instance)
    plan = build_cross_map_priority_run_plan(manifest)

    store = CrossMapPriorityCheckpointStore(
        results_dir=tmp_path,
        csv_path=tmp_path / "results.csv",
        jsonl_path=tmp_path / "results_details.jsonl",
    )
    first_record = execute_cross_map_priority_run(
        grid_map=grid_map,
        scenarios=scenarios,
        instance=plan[0].instance,
        strategy=plan[0].strategy,
        max_timestep=manifest.max_timestep,
        map_name="bench",
        manifest_sha256="abc",
    )
    store.append_record(first_record)

    completed_keys = load_checkpoint_records(tmp_path / "results_details.jsonl")
    executed: list[CrossMapPriorityStrategy] = []

    for entry in plan:
        key = CrossMapPriorityRunKey("bench", entry.instance.instance_id, entry.strategy)
        if key in completed_keys:
            continue
        record = execute_cross_map_priority_run(
            grid_map=grid_map,
            scenarios=scenarios,
            instance=entry.instance,
            strategy=entry.strategy,
            max_timestep=manifest.max_timestep,
            map_name="bench",
            manifest_sha256="abc",
        )
        store.append_record(record)
        executed.append(entry.strategy)

    assert executed == [
        CrossMapPriorityStrategy.CDF_H,
        CrossMapPriorityStrategy.CDF_L,
        CrossMapPriorityStrategy.SPF_CD,
    ]


def test_duplicate_checkpoint_key_rejected(tmp_path: Path) -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    manifest = _tiny_manifest(instance)
    plan = build_cross_map_priority_run_plan(
        manifest,
        strategies=(CrossMapPriorityStrategy.SPF,),
    )
    record = execute_cross_map_priority_run(
        grid_map=grid_map,
        scenarios=scenarios,
        instance=plan[0].instance,
        strategy=plan[0].strategy,
        max_timestep=manifest.max_timestep,
        map_name="bench",
        manifest_sha256="abc",
    )

    jsonl_path = tmp_path / "results_details.jsonl"
    line = json.dumps(
        {
            "map_name": record.map_name,
            "manifest_sha256": record.manifest_sha256,
            "instance_id": record.instance_id,
            "agent_count": record.agent_count,
            "interaction_level": record.interaction_level,
            "strategy": record.strategy.value,
            "agent_order": list(record.agent_order or ()),
            "original_index_order": list(record.original_index_order or ()),
            "success": record.success,
            "termination_reason": record.termination_reason,
            "ordering_time_ms": record.ordering_time_ms,
            "pp_time_ms": record.pp_time_ms,
            "total_time_ms": record.total_time_ms,
            "soc": record.soc,
            "makespan": record.makespan,
            "conflict_count": record.conflict_count,
            "pp_low_level_searches": record.pp_low_level_searches,
            "pp_agents_planned": record.pp_agents_planned,
        }
    )
    jsonl_path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Duplicate cross-map priority run keys"):
        load_checkpoint_records(jsonl_path)


def test_wrong_map_checkpoint_rejected(tmp_path: Path) -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    manifest = _tiny_manifest(instance)
    record = execute_cross_map_priority_run(
        grid_map=grid_map,
        scenarios=scenarios,
        instance=instance,
        strategy=CrossMapPriorityStrategy.SPF,
        max_timestep=manifest.max_timestep,
        map_name="AR0400SR",
        manifest_sha256="abc",
    )

    store = CrossMapPriorityCheckpointStore(
        results_dir=tmp_path,
        csv_path=tmp_path / "results.csv",
        jsonl_path=tmp_path / "results_details.jsonl",
    )
    store.append_record(record)

    with pytest.raises(ValueError, match="map mismatch"):
        validate_checkpoint_records(
            store.records,
            map_name="AR0307SR",
            manifest_sha256="abc",
            allowed_instance_ids={instance.instance_id},
            allowed_strategies=set(DEFAULT_CROSS_MAP_STRATEGIES),
        )


def test_unknown_strategy_checkpoint_rejected(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "results_details.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "map_name": "bench",
                "manifest_sha256": "abc",
                "instance_id": "bench_n02_low_000",
                "agent_count": 2,
                "interaction_level": "low",
                "strategy": "not_a_strategy",
                "success": True,
                "termination_reason": "success",
                "ordering_time_ms": 1.0,
                "pp_time_ms": 2.0,
                "total_time_ms": 3.0,
                "soc": 1,
                "makespan": 1,
                "conflict_count": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_checkpoint_records(jsonl_path)


def test_records_joinable_by_map_instance_and_strategy(tmp_path: Path) -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    manifest = _tiny_manifest(instance)
    plan = build_cross_map_priority_run_plan(manifest)

    store = CrossMapPriorityCheckpointStore(
        results_dir=tmp_path,
        csv_path=tmp_path / "results.csv",
        jsonl_path=tmp_path / "results_details.jsonl",
    )
    for entry in plan:
        store.append_record(
            execute_cross_map_priority_run(
                grid_map=grid_map,
                scenarios=scenarios,
                instance=entry.instance,
                strategy=entry.strategy,
                max_timestep=manifest.max_timestep,
                map_name="bench",
                manifest_sha256="abc",
            )
        )

    by_instance: dict[str, dict[CrossMapPriorityStrategy, object]] = {}
    for record in store.records.values():
        by_instance.setdefault(record.instance_id, {})[record.strategy] = record

    assert set(by_instance["bench_n02_low_000"]) == set(DEFAULT_CROSS_MAP_STRATEGIES)


def test_result_output_isolated_per_map() -> None:
    repo = _repo_root()
    config_0400 = mapf8_cross_map_execution_config("AR0400SR", repo)
    config_0307 = mapf8_cross_map_execution_config("AR0307SR", repo)

    assert config_0400.results_dir != config_0307.results_dir
    assert config_0400.results_dir.name == "AR0400SR"
    assert config_0307.results_dir.name == "AR0307SR"
    assert str(MAPF8_CROSS_MAP_RESULTS_ROOT) in str(config_0400.results_dir)


def test_historical_result_paths_cannot_be_targeted() -> None:
    repo = _repo_root()
    for forbidden in FORBIDDEN_HISTORICAL_RESULTS_DIRS:
        with pytest.raises(ValueError, match="historical"):
            assert_results_dir_isolated(repo / forbidden, repo)

    assert MAIN_PRIORITY_STRATEGY_RESULTS_DIR != str(MAPF8_CROSS_MAP_RESULTS_ROOT)
    assert MAIN_CONFLICT_AWARE_RESULTS_DIR != MAPF8_CROSS_MAP_RESULTS_ROOT


def test_tiny_smoke_execution_produces_conflict_free_success() -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    manifest = _tiny_manifest(instance)

    record = execute_cross_map_priority_run(
        grid_map=grid_map,
        scenarios=scenarios,
        instance=instance,
        strategy=CrossMapPriorityStrategy.SPF,
        max_timestep=manifest.max_timestep,
        map_name="bench",
        manifest_sha256="abc",
    )

    assert record.success is True
    assert record.conflict_count == 0
    assert record.agent_order is not None
    assert record.original_index_order is not None


def test_run_key_uses_map_instance_and_strategy() -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    manifest = _tiny_manifest(instance)

    record = execute_cross_map_priority_run(
        grid_map=grid_map,
        scenarios=scenarios,
        instance=instance,
        strategy=CrossMapPriorityStrategy.CDF_H,
        max_timestep=manifest.max_timestep,
        map_name="AR0400SR",
        manifest_sha256="abc",
    )

    assert run_key(record) == CrossMapPriorityRunKey(
        "AR0400SR",
        "bench_n02_low_000",
        CrossMapPriorityStrategy.CDF_H,
    )


def test_manifest_hash_mismatch_in_checkpoint_rejected(tmp_path: Path) -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    manifest = _tiny_manifest(instance)
    record = execute_cross_map_priority_run(
        grid_map=grid_map,
        scenarios=scenarios,
        instance=instance,
        strategy=CrossMapPriorityStrategy.SPF,
        max_timestep=manifest.max_timestep,
        map_name="bench",
        manifest_sha256="abc",
    )

    store = CrossMapPriorityCheckpointStore(
        results_dir=tmp_path,
        csv_path=tmp_path / "results.csv",
        jsonl_path=tmp_path / "results_details.jsonl",
    )
    store.append_record(record)

    with pytest.raises(ValueError, match="manifest hash mismatch"):
        validate_checkpoint_records(
            store.records,
            map_name="bench",
            manifest_sha256="def",
            allowed_instance_ids={instance.instance_id},
            allowed_strategies=set(DEFAULT_CROSS_MAP_STRATEGIES),
        )


def test_frozen_manifest_file_hashes_match_constants() -> None:
    repo = _repo_root()
    for map_stem, expected in FROZEN_MANIFEST_SHA256.items():
        path = (
            repo
            / "pathfinding"
            / "results"
            / "mapf_benchmarks"
            / f"{map_stem}_manifest.json"
        )
        if not path.is_file():
            pytest.skip(f"Manifest missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected
