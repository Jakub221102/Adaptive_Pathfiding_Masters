from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from pathfinding.src.experiments.mapf_benchmark_execution import TERMINATION_SUCCESS
from pathfinding.src.experiments.mapf_benchmark_instances import (
    MAPFBenchmarkInstance,
    MAPFBenchmarkManifest,
    MAPFInteractionLevel,
    load_benchmark_manifest,
)
from pathfinding.src.experiments.mapf_random_priority_execution import (
    DEFAULT_RANDOM_PRIORITY_BASE_SEED,
    DEFAULT_RANDOM_PRIORITY_K,
    MAIN_RANDOM_PRIORITY_JSONL,
    SUPPLEMENTAL_ORDERING_COUNT,
    SUPPLEMENTAL_ORDERING_INDEX_START,
    RandomPriorityCheckpointStore,
    RandomPriorityRunKey,
    build_supplemental_random_priority_run_plan,
    execute_random_priority_plan_entry,
    generate_supplemental_orderings_for_instance,
    generate_unique_orderings_for_instance,
    load_checkpoint_records,
    load_historical_k10_records_by_instance,
)
from pathfinding.tests.helpers import build_grid_map
from pathfinding.tests.unit.test_mapf_random_priority_execution import (
    _instance,
    _scenario,
    _tiny_manifest,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HISTORICAL_K10_JSONL = _REPO_ROOT / MAIN_RANDOM_PRIORITY_JSONL
_PRIMARY_MANIFEST = (
    _REPO_ROOT / "pathfinding" / "results" / "mapf_benchmarks" / "AR0204SR_manifest.json"
)


def _snapshot_file(path: Path) -> tuple[int, str]:
    return path.stat().st_mtime_ns, path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def primary_manifest() -> MAPFBenchmarkManifest:
    return load_benchmark_manifest(_PRIMARY_MANIFEST)


def test_supplemental_ordering_indices_are_exactly_10_through_19(
    primary_manifest: MAPFBenchmarkManifest,
) -> None:
    plan = build_supplemental_random_priority_run_plan(
        primary_manifest,
        base_seed=DEFAULT_RANDOM_PRIORITY_BASE_SEED,
        historical_k10_jsonl_path=_HISTORICAL_K10_JSONL,
    )

    expected = list(
        range(
            SUPPLEMENTAL_ORDERING_INDEX_START,
            SUPPLEMENTAL_ORDERING_INDEX_START + SUPPLEMENTAL_ORDERING_COUNT,
        )
    )
    assert sorted({entry.ordering.ordering_index for entry in plan}) == expected

    for instance_id in {entry.instance.instance_id for entry in plan}:
        instance_indices = sorted(
            entry.ordering.ordering_index
            for entry in plan
            if entry.instance.instance_id == instance_id
        )
        assert instance_indices == expected


def test_supplemental_plan_has_ten_orderings_per_primary_instance(
    primary_manifest: MAPFBenchmarkManifest,
) -> None:
    plan = build_supplemental_random_priority_run_plan(
        primary_manifest,
        base_seed=DEFAULT_RANDOM_PRIORITY_BASE_SEED,
        historical_k10_jsonl_path=_HISTORICAL_K10_JSONL,
    )

    assert len(primary_manifest.instances) == 27
    assert len(plan) == 27 * SUPPLEMENTAL_ORDERING_COUNT

    counts: dict[str, int] = {}
    for entry in plan:
        counts[entry.instance.instance_id] = (
            counts.get(entry.instance.instance_id, 0) + 1
        )

    assert len(counts) == 27
    assert all(count == SUPPLEMENTAL_ORDERING_COUNT for count in counts.values())


def test_supplemental_seeds_are_deterministic(
    primary_manifest: MAPFBenchmarkManifest,
) -> None:
    first_plan = build_supplemental_random_priority_run_plan(
        primary_manifest,
        base_seed=DEFAULT_RANDOM_PRIORITY_BASE_SEED,
        historical_k10_jsonl_path=_HISTORICAL_K10_JSONL,
    )
    second_plan = build_supplemental_random_priority_run_plan(
        primary_manifest,
        base_seed=DEFAULT_RANDOM_PRIORITY_BASE_SEED,
        historical_k10_jsonl_path=_HISTORICAL_K10_JSONL,
    )

    first_specs = {
        (entry.instance.instance_id, entry.ordering.ordering_index): entry.ordering
        for entry in first_plan
    }
    second_specs = {
        (entry.instance.instance_id, entry.ordering.ordering_index): entry.ordering
        for entry in second_plan
    }

    assert first_specs.keys() == second_specs.keys()
    for key, first_ordering in first_specs.items():
        second_ordering = second_specs[key]
        assert first_ordering.ordering_seed == second_ordering.ordering_seed
        assert first_ordering.agent_order == second_ordering.agent_order


def test_supplemental_orderings_do_not_duplicate_historical_k10(
    primary_manifest: MAPFBenchmarkManifest,
) -> None:
    historical = load_historical_k10_records_by_instance(_HISTORICAL_K10_JSONL)
    plan = build_supplemental_random_priority_run_plan(
        primary_manifest,
        base_seed=DEFAULT_RANDOM_PRIORITY_BASE_SEED,
        historical_k10_jsonl_path=_HISTORICAL_K10_JSONL,
    )

    for entry in plan:
        instance_id = entry.instance.instance_id
        historical_orders = {
            record.agent_order
            for record in historical[instance_id].values()
        }
        assert entry.ordering.agent_order not in historical_orders


def test_supplemental_orderings_are_mutually_unique_per_instance(
    primary_manifest: MAPFBenchmarkManifest,
) -> None:
    plan = build_supplemental_random_priority_run_plan(
        primary_manifest,
        base_seed=DEFAULT_RANDOM_PRIORITY_BASE_SEED,
        historical_k10_jsonl_path=_HISTORICAL_K10_JSONL,
    )

    orders_by_instance: dict[str, list[tuple[int, ...]]] = {}
    for entry in plan:
        orders_by_instance.setdefault(entry.instance.instance_id, []).append(
            entry.ordering.agent_order
        )

    for instance_id, orders in orders_by_instance.items():
        assert len(orders) == SUPPLEMENTAL_ORDERING_COUNT
        assert len(set(orders)) == SUPPLEMENTAL_ORDERING_COUNT, instance_id


def _write_synthetic_k10_jsonl(
    *,
    path: Path,
    instance: MAPFBenchmarkInstance,
) -> None:
    k10_orderings = generate_unique_orderings_for_instance(
        instance=instance,
        base_seed=DEFAULT_RANDOM_PRIORITY_BASE_SEED,
        k=DEFAULT_RANDOM_PRIORITY_K,
    )
    with path.open("w", encoding="utf-8") as file:
        for ordering in k10_orderings:
            payload = {
                "instance_id": instance.instance_id,
                "agent_count": instance.agent_count,
                "interaction_level": instance.interaction_level.value,
                "ordering_index": ordering.ordering_index,
                "ordering_seed": ordering.ordering_seed,
                "agent_order": list(ordering.agent_order),
                "base_seed": DEFAULT_RANDOM_PRIORITY_BASE_SEED,
                "success": True,
                "termination_reason": TERMINATION_SUCCESS,
                "execution_time_ms": 1.0,
                "soc": 4,
                "makespan": 2,
                "conflict_count": 0,
                "pp_search_metrics": {
                    "low_level_searches": instance.agent_count,
                    "agents_planned": instance.agent_count,
                },
            }
            file.write(json.dumps(payload) + "\n")


def test_supplemental_resume_skips_completed_ordering_runs(tmp_path: Path) -> None:
    instance = _instance("bench_n05_low_000", 5)
    manifest = _tiny_manifest(instance)
    grid_map = build_grid_map(
        [[0, 0, 0, 0, 0] for _ in range(5)],
        name="bench.map",
    )
    scenarios = [
        _scenario(row, 0, row, 4, optimal_length=25.0 + row)
        for row in range(5)
    ]

    historical_jsonl = tmp_path / "historical_k10.jsonl"
    _write_synthetic_k10_jsonl(path=historical_jsonl, instance=instance)

    plan = build_supplemental_random_priority_run_plan(
        manifest,
        base_seed=DEFAULT_RANDOM_PRIORITY_BASE_SEED,
        historical_k10_jsonl_path=historical_jsonl,
        supplemental_count=2,
        ordering_index_start=SUPPLEMENTAL_ORDERING_INDEX_START,
    )
    assert [entry.ordering.ordering_index for entry in plan] == [10, 11]

    supplemental_jsonl = tmp_path / "supplemental.jsonl"
    supplemental_csv = tmp_path / "supplemental.csv"
    store = RandomPriorityCheckpointStore(
        results_dir=tmp_path,
        csv_path=supplemental_csv,
        jsonl_path=supplemental_jsonl,
    )

    first_record = execute_random_priority_plan_entry(
        grid_map=grid_map,
        scenarios=scenarios,
        entry=plan[0],
        base_seed=DEFAULT_RANDOM_PRIORITY_BASE_SEED,
        max_timestep=manifest.max_timestep,
    )
    store.append_record(first_record)

    completed_keys = load_checkpoint_records(supplemental_jsonl)
    executed_indices: list[int] = []

    for entry in plan:
        key = RandomPriorityRunKey(
            entry.instance.instance_id,
            entry.ordering.ordering_index,
        )
        if key in completed_keys:
            continue
        record = execute_random_priority_plan_entry(
            grid_map=grid_map,
            scenarios=scenarios,
            entry=entry,
            base_seed=DEFAULT_RANDOM_PRIORITY_BASE_SEED,
            max_timestep=manifest.max_timestep,
        )
        store.append_record(record)
        executed_indices.append(entry.ordering.ordering_index)

    assert executed_indices == [11]
    assert len(load_checkpoint_records(supplemental_jsonl)) == 2


def test_historical_k10_files_remain_untouched_during_supplemental_planning(
    primary_manifest: MAPFBenchmarkManifest,
    tmp_path: Path,
) -> None:
    if not _HISTORICAL_K10_JSONL.is_file():
        pytest.skip("Historical K=10 results not present in workspace")

    historical_copy_dir = tmp_path / "historical_copy"
    historical_copy_dir.mkdir()
    copied_jsonl = historical_copy_dir / "results_details.jsonl"
    copied_csv = historical_copy_dir / "results.csv"
    shutil.copy2(_HISTORICAL_K10_JSONL, copied_jsonl)
    shutil.copy2(_REPO_ROOT / MAIN_RANDOM_PRIORITY_JSONL.parent / "results.csv", copied_csv)

    before_jsonl = _snapshot_file(_HISTORICAL_K10_JSONL)
    before_csv = _snapshot_file(_REPO_ROOT / MAIN_RANDOM_PRIORITY_JSONL.parent / "results.csv")

    build_supplemental_random_priority_run_plan(
        primary_manifest,
        base_seed=DEFAULT_RANDOM_PRIORITY_BASE_SEED,
        historical_k10_jsonl_path=copied_jsonl,
    )

    after_jsonl = _snapshot_file(_HISTORICAL_K10_JSONL)
    after_csv = _snapshot_file(_REPO_ROOT / MAIN_RANDOM_PRIORITY_JSONL.parent / "results.csv")

    assert before_jsonl == after_jsonl
    assert before_csv == after_csv


def test_generate_supplemental_orderings_rejects_k10_duplicates() -> None:
    instance = _instance("bench_n05_low_000", 5)
    k10 = generate_unique_orderings_for_instance(
        instance=instance,
        base_seed=DEFAULT_RANDOM_PRIORITY_BASE_SEED,
        k=DEFAULT_RANDOM_PRIORITY_K,
    )

    supplemental = generate_supplemental_orderings_for_instance(
        instance=instance,
        base_seed=DEFAULT_RANDOM_PRIORITY_BASE_SEED,
        excluded_agent_orders=tuple(ordering.agent_order for ordering in k10),
    )

    k10_orders = {ordering.agent_order for ordering in k10}
    supplemental_orders = {ordering.agent_order for ordering in supplemental}
    assert supplemental_orders.isdisjoint(k10_orders)


def test_smoke_supplemental_execution_on_one_instance(tmp_path: Path) -> None:
    instance = _instance("bench_n05_low_000", 5)
    manifest = _tiny_manifest(instance)
    grid_map = build_grid_map(
        [[0, 0, 0, 0, 0] for _ in range(5)],
        name="bench.map",
    )
    scenarios = [
        _scenario(row, 0, row, 4, optimal_length=25.0 + row)
        for row in range(5)
    ]

    historical_jsonl = tmp_path / "historical_k10.jsonl"
    _write_synthetic_k10_jsonl(path=historical_jsonl, instance=instance)

    plan = build_supplemental_random_priority_run_plan(
        manifest,
        base_seed=DEFAULT_RANDOM_PRIORITY_BASE_SEED,
        historical_k10_jsonl_path=historical_jsonl,
        supplemental_count=1,
        ordering_index_start=SUPPLEMENTAL_ORDERING_INDEX_START,
        instance_ids=(instance.instance_id,),
    )
    assert len(plan) == 1
    assert plan[0].ordering.ordering_index == SUPPLEMENTAL_ORDERING_INDEX_START

    record = execute_random_priority_plan_entry(
        grid_map=grid_map,
        scenarios=scenarios,
        entry=plan[0],
        base_seed=DEFAULT_RANDOM_PRIORITY_BASE_SEED,
        max_timestep=manifest.max_timestep,
    )

    assert record.success is True
    assert record.ordering_index == SUPPLEMENTAL_ORDERING_INDEX_START
    assert record.agent_order == plan[0].ordering.agent_order
    assert record.soc is not None
    assert record.makespan is not None
    assert record.conflict_count == 0
