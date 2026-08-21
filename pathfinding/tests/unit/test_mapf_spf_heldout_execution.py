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
    HELD_OUT_CONFLICT_AWARE_RESULTS_DIR,
    validate_held_out_execution_manifest,
)
from pathfinding.src.experiments.mapf_priority_strategy_execution import (
    HELD_OUT_SPF_RESULTS_DIR,
    MAIN_PRIORITY_STRATEGY_RESULTS_DIR,
    PriorityStrategy,
    PriorityStrategyCheckpointStore,
    PriorityStrategyRunKey,
    build_priority_strategy_run_plan,
    execute_priority_strategy_plan_entry,
    load_checkpoint_records,
    validate_priority_strategy_run_record,
)
from pathfinding.src.loaders.map_loader import load_moving_ai_map
from pathfinding.src.loaders.scen_loader import load_moving_ai_scenarios


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _held_out_manifest_path() -> Path:
    return _repo_root() / "pathfinding/results/mapf_benchmarks/AR0204SR_heldout_manifest.json"


def _primary_manifest_path() -> Path:
    return _repo_root() / "pathfinding/results/mapf_benchmarks/AR0204SR_manifest.json"


def _map_path() -> Path:
    return _repo_root() / "Data" / "bg512-map" / "AR0204SR.map"


def _scen_path() -> Path:
    return _repo_root() / "Data" / "bg512-scen" / "AR0204SR.map.scen"


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_held_out_spf_manifest_loads_and_validates() -> None:
    held_out_path = _held_out_manifest_path()
    if not held_out_path.is_file():
        pytest.skip("Frozen held-out manifest not present")

    held_out = load_benchmark_manifest(held_out_path)
    primary = (
        load_benchmark_manifest(_primary_manifest_path())
        if _primary_manifest_path().is_file()
        else None
    )
    validate_held_out_execution_manifest(
        held_out,
        primary_manifest=primary,
        expected_seed=DEFAULT_HELD_OUT_SEED,
    )


def test_held_out_spf_plan_is_27_runs_spf_only() -> None:
    held_out_path = _held_out_manifest_path()
    if not held_out_path.is_file():
        pytest.skip("Frozen held-out manifest not present")

    held_out = load_benchmark_manifest(held_out_path)
    plan = build_priority_strategy_run_plan(
        held_out,
        strategies=(PriorityStrategy.SPF,),
        agent_counts=(5, 10, 20),
        interaction_levels=("low", "medium", "high"),
    )

    assert len(plan) == 27
    assert all(entry.strategy == PriorityStrategy.SPF for entry in plan)
    assert len({entry.instance.instance_id for entry in plan}) == 27
    assert all("_HO_" in entry.instance.instance_id for entry in plan)


def test_held_out_spf_output_directory_is_separate() -> None:
    assert HELD_OUT_SPF_RESULTS_DIR != MAIN_PRIORITY_STRATEGY_RESULTS_DIR
    assert HELD_OUT_SPF_RESULTS_DIR != HELD_OUT_CONFLICT_AWARE_RESULTS_DIR
    assert str(HELD_OUT_SPF_RESULTS_DIR).endswith("mapf_spf_heldout_execution")


def test_held_out_spf_does_not_use_manifest_independent_soc_cache() -> None:
    source = (
        Path("pathfinding/src/experiments/mapf_priority_strategy_execution.py")
        .read_text(encoding="utf-8")
    )
    fn_start = source.index("def execute_priority_strategy_run")
    fn_end = source.index("\ndef execute_priority_strategy_plan_entry")
    body = source[fn_start:fn_end]
    for forbidden in (
        "independent_soc",
        "independent_makespan",
        "independent_conflict_count",
        "vertex_conflict_count",
        "edge_conflict_count",
    ):
        assert forbidden not in body


def test_held_out_spf_resume_skips_completed_instance(tmp_path: Path) -> None:
    smoke_id = "AR0204SR_HO_n05_low_000"
    jsonl_path = tmp_path / "results_details.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "instance_id": smoke_id,
                "agent_count": 5,
                "interaction_level": "low",
                "strategy": "spf",
                "agent_order": [0, 1, 2, 3, 4],
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
    assert PriorityStrategyRunKey(smoke_id, PriorityStrategy.SPF) in records


def test_held_out_spf_smoke_one_instance(tmp_path: Path) -> None:
    held_out_path = _held_out_manifest_path()
    map_path = _map_path()
    scen_path = _scen_path()
    if not held_out_path.is_file() or not map_path.is_file() or not scen_path.is_file():
        pytest.skip("Held-out manifest or AR0204SR map/scen not available")

    frozen_paths = [
        _repo_root()
        / "pathfinding/results/mapf_priority_strategy_execution/results_details.jsonl",
        _repo_root()
        / "pathfinding/results/mapf_conflict_aware_priority_heldout_execution/results_details.jsonl",
    ]
    digests_before = {path: _file_digest(path) for path in frozen_paths if path.is_file()}
    held_out_digest_before = _file_digest(held_out_path)

    held_out = load_benchmark_manifest(held_out_path)
    entry = build_priority_strategy_run_plan(
        held_out,
        strategies=(PriorityStrategy.SPF,),
        instance_ids=("AR0204SR_HO_n05_low_000",),
    )[0]

    grid_map = load_moving_ai_map(map_path)
    scenarios = load_moving_ai_scenarios(scen_path)
    store = PriorityStrategyCheckpointStore(
        results_dir=tmp_path,
        csv_path=tmp_path / "results.csv",
        jsonl_path=tmp_path / "results_details.jsonl",
        reset_results=True,
    )

    record = execute_priority_strategy_plan_entry(
        grid_map=grid_map,
        scenarios=scenarios,
        entry=entry,
        max_timestep=held_out.max_timestep,
        catalogue_role=held_out.catalogue_role,
        catalogue_seed=held_out.seed,
    )
    validate_priority_strategy_run_record(record)
    assert record.strategy == PriorityStrategy.SPF
    assert record.catalogue_role == CATALOGUE_ROLE_HELD_OUT
    assert record.catalogue_seed == DEFAULT_HELD_OUT_SEED
    assert record.total_time_ms == pytest.approx(
        record.ordering_time_ms + (record.pp_time_ms or 0.0)
    )
    store.append_record(record)

    assert (tmp_path / "results.csv").is_file()
    assert (tmp_path / "results_details.jsonl").read_text(encoding="utf-8").count("\n") == 1

    for path, digest in digests_before.items():
        assert _file_digest(path) == digest
    assert _file_digest(held_out_path) == held_out_digest_before
