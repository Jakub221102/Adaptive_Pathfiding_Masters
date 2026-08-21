from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pathfinding.src.experiments.mapf_benchmark_instances import (
    CATALOGUE_ROLE_HELD_OUT,
    DEFAULT_HELD_OUT_SEED,
    load_benchmark_manifest,
)
from pathfinding.src.experiments.mapf_conflict_aware_priority_execution import (
    DEFAULT_CONFLICT_AWARE_STRATEGIES,
    HELD_OUT_CONFLICT_AWARE_RESULTS_DIR,
    HELD_OUT_MANIFEST_PATH,
    MAIN_CONFLICT_AWARE_RESULTS_DIR,
    ConflictAwarePriorityCheckpointStore,
    ConflictAwarePriorityRunKey,
    build_conflict_aware_priority_run_plan,
    execute_conflict_aware_priority_plan_entry,
    load_checkpoint_records,
    validate_conflict_aware_priority_run_record,
    validate_held_out_execution_manifest,
)
from pathfinding.src.loaders.map_loader import load_moving_ai_map
from pathfinding.src.loaders.scen_loader import load_moving_ai_scenarios


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _held_out_manifest_path() -> Path:
    return _repo_root() / HELD_OUT_MANIFEST_PATH


def _primary_manifest_path() -> Path:
    return _repo_root() / "pathfinding/results/mapf_benchmarks/AR0204SR_manifest.json"


def _map_path() -> Path:
    return _repo_root() / "Data" / "bg512-map" / "AR0204SR.map"


def _scen_path() -> Path:
    return _repo_root() / "Data" / "bg512-scen" / "AR0204SR.map.scen"


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_held_out_manifest_loads_and_validates() -> None:
    held_out_path = _held_out_manifest_path()
    primary_path = _primary_manifest_path()
    if not held_out_path.is_file():
        pytest.skip("Frozen held-out manifest not present")

    held_out = load_benchmark_manifest(held_out_path)
    primary = (
        load_benchmark_manifest(primary_path) if primary_path.is_file() else None
    )

    validate_held_out_execution_manifest(
        held_out,
        primary_manifest=primary,
        expected_seed=DEFAULT_HELD_OUT_SEED,
    )
    assert held_out.catalogue_role == CATALOGUE_ROLE_HELD_OUT
    assert len(held_out.instances) == 27


def test_held_out_full_plan_is_81_runs_with_distinct_ho_ids() -> None:
    held_out_path = _held_out_manifest_path()
    if not held_out_path.is_file():
        pytest.skip("Frozen held-out manifest not present")

    held_out = load_benchmark_manifest(held_out_path)
    plan = build_conflict_aware_priority_run_plan(
        held_out,
        strategies=DEFAULT_CONFLICT_AWARE_STRATEGIES,
        agent_counts=(5, 10, 20),
        interaction_levels=("low", "medium", "high"),
    )

    assert len(plan) == 81
    instance_ids = {entry.instance.instance_id for entry in plan}
    assert len(instance_ids) == 27
    assert all("_HO_" in instance_id for instance_id in instance_ids)
    assert all(
        not instance_id.startswith("AR0204SR_n")
        or instance_id.startswith("AR0204SR_HO_")
        for instance_id in instance_ids
    )

    keys = {
        ConflictAwarePriorityRunKey(entry.instance.instance_id, entry.strategy)
        for entry in plan
    }
    assert len(keys) == 81


def test_held_out_output_directory_differs_from_primary() -> None:
    assert HELD_OUT_CONFLICT_AWARE_RESULTS_DIR != MAIN_CONFLICT_AWARE_RESULTS_DIR
    assert str(HELD_OUT_CONFLICT_AWARE_RESULTS_DIR).endswith(
        "mapf_conflict_aware_priority_heldout_execution"
    )


def test_held_out_resume_skips_completed_strategies(tmp_path: Path) -> None:
    held_out_path = _held_out_manifest_path()
    if not held_out_path.is_file():
        pytest.skip("Frozen held-out manifest not present")

    held_out = load_benchmark_manifest(held_out_path)
    smoke_id = "AR0204SR_HO_n05_low_000"
    plan = build_conflict_aware_priority_run_plan(
        held_out,
        strategies=DEFAULT_CONFLICT_AWARE_STRATEGIES,
        instance_ids=(smoke_id,),
    )
    assert len(plan) == 3

    jsonl_path = tmp_path / "results_details.jsonl"
    first_strategy = plan[0].strategy
    jsonl_path.write_text(
        json.dumps(
            {
                "instance_id": smoke_id,
                "agent_count": 5,
                "interaction_level": "low",
                "strategy": first_strategy.value,
                "agent_order": [0, 1, 2, 3, 4],
                "original_index_order": [0, 1, 2, 3, 4],
                "agent_degrees": [0, 0, 0, 0, 0],
                "independent_conflict_count": 0,
                "independent_conflict_pair_count": 0,
                "ordering_low_level_searches": 5,
                "incident_conflict_counts": [0, 0, 0, 0, 0],
                "success": True,
                "termination_reason": "success",
                "ordering_time_ms": 1.0,
                "pp_time_ms": 2.0,
                "total_time_ms": 3.0,
                "soc": 100,
                "makespan": 20,
                "conflict_count": 0,
                "pp_search_metrics": {
                    "low_level_searches": 5,
                    "agents_planned": 5,
                },
                "catalogue_role": CATALOGUE_ROLE_HELD_OUT,
                "catalogue_seed": DEFAULT_HELD_OUT_SEED,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_checkpoint_records(jsonl_path)
    completed = set(records)
    remaining = [
        entry
        for entry in plan
        if ConflictAwarePriorityRunKey(entry.instance.instance_id, entry.strategy)
        not in completed
    ]
    assert len(remaining) == 2


def test_held_out_smoke_one_instance_three_strategies(tmp_path: Path) -> None:
    held_out_path = _held_out_manifest_path()
    map_path = _map_path()
    scen_path = _scen_path()
    if not held_out_path.is_file() or not map_path.is_file() or not scen_path.is_file():
        pytest.skip("Held-out manifest or AR0204SR map/scen not available")

    primary_results = (
        _repo_root()
        / "pathfinding/results/mapf_conflict_aware_priority_execution/results_details.jsonl"
    )
    primary_digest_before = (
        _file_digest(primary_results) if primary_results.is_file() else None
    )
    held_out_digest_before = _file_digest(held_out_path)

    held_out = load_benchmark_manifest(held_out_path)
    smoke_id = "AR0204SR_HO_n05_low_000"
    plan = build_conflict_aware_priority_run_plan(
        held_out,
        strategies=DEFAULT_CONFLICT_AWARE_STRATEGIES,
        instance_ids=(smoke_id,),
    )
    assert len(plan) == 3

    grid_map = load_moving_ai_map(map_path)
    scenarios = load_moving_ai_scenarios(scen_path)
    store = ConflictAwarePriorityCheckpointStore(
        results_dir=tmp_path,
        csv_path=tmp_path / "results.csv",
        jsonl_path=tmp_path / "results_details.jsonl",
        reset_results=True,
    )

    for entry in plan:
        record = execute_conflict_aware_priority_plan_entry(
            grid_map=grid_map,
            scenarios=scenarios,
            entry=entry,
            max_timestep=held_out.max_timestep,
            catalogue_role=held_out.catalogue_role,
            catalogue_seed=held_out.seed,
        )
        validate_conflict_aware_priority_run_record(record)
        assert record.catalogue_role == CATALOGUE_ROLE_HELD_OUT
        assert record.catalogue_seed == DEFAULT_HELD_OUT_SEED
        store.append_record(record)

    assert (tmp_path / "results.csv").is_file()
    assert (tmp_path / "results_details.jsonl").read_text(encoding="utf-8").count("\n") == 3

    if primary_digest_before is not None:
        assert _file_digest(primary_results) == primary_digest_before
    assert _file_digest(held_out_path) == held_out_digest_before
