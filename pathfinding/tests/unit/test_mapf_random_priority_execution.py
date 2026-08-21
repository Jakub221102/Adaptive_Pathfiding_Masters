from __future__ import annotations

import json
from pathlib import Path

import pytest

from pathfinding.src.algorithms.mapf.models import MAPFAgent, MAPFScenario
from pathfinding.src.core.models import Position, Scenario
from pathfinding.src.experiments.mapf_benchmark_execution import TERMINATION_SUCCESS
from pathfinding.src.experiments.mapf_benchmark_instances import (
    MAPFBenchmarkInstance,
    MAPFBenchmarkManifest,
    MAPFInteractionLevel,
)
from pathfinding.src.experiments.mapf_random_priority_execution import (
    RandomPriorityCheckpointStore,
    RandomPriorityRunKey,
    build_random_priority_run_plan,
    derive_ordering_seed,
    execute_random_priority_plan_entry,
    execute_random_priority_run,
    generate_unique_orderings_for_instance,
    load_checkpoint_records,
    run_key,
)
from pathfinding.tests.helpers import build_grid_map


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
    *,
    optimal_length: float = 25.0,
) -> Scenario:
    return Scenario(
        map_name="bench.map",
        width=5,
        height=5,
        start=Position(row=start_row, col=start_col),
        goal=Position(row=goal_row, col=goal_col),
        optimal_length=optimal_length,
    )


def _tiny_manifest(*instances: MAPFBenchmarkInstance) -> MAPFBenchmarkManifest:
    return MAPFBenchmarkManifest(
        map_name="bench.map",
        scenario_name="bench.map.scen",
        seed=2026,
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


def test_same_inputs_produce_identical_ordering_seed_and_agent_order() -> None:
    instance = _instance("bench_n05_low_000", 5)
    base_seed = 2026
    ordering_index = 3

    first_seed = derive_ordering_seed(
        base_seed=base_seed,
        instance_id=instance.instance_id,
        ordering_index=ordering_index,
    )
    second_seed = derive_ordering_seed(
        base_seed=base_seed,
        instance_id=instance.instance_id,
        ordering_index=ordering_index,
    )
    assert first_seed == second_seed

    first_orderings = generate_unique_orderings_for_instance(
        instance=instance,
        base_seed=base_seed,
        k=ordering_index + 1,
    )
    second_orderings = generate_unique_orderings_for_instance(
        instance=instance,
        base_seed=base_seed,
        k=ordering_index + 1,
    )

    assert first_orderings == second_orderings
    assert (
        first_orderings[ordering_index].agent_order
        == second_orderings[ordering_index].agent_order
    )


def test_generated_k_orderings_are_unique_for_instance() -> None:
    instance = _instance("bench_n05_low_001", 5)
    orderings = generate_unique_orderings_for_instance(
        instance=instance,
        base_seed=2026,
        k=10,
    )

    agent_orders = [ordering.agent_order for ordering in orderings]
    assert len(agent_orders) == 10
    assert len(set(agent_orders)) == 10
    assert [ordering.ordering_index for ordering in orderings] == list(range(10))


def test_resume_key_distinguishes_ordering_runs() -> None:
    instance = _instance("bench_n05_low_000", 5)
    plan = build_random_priority_run_plan(
        _tiny_manifest(instance),
        k=3,
        base_seed=2026,
    )

    keys = [
        RandomPriorityRunKey(entry.instance.instance_id, entry.ordering.ordering_index)
        for entry in plan
    ]
    assert len(keys) == len(set(keys)) == 3


def test_resume_skips_completed_ordering_runs(tmp_path: Path) -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    manifest = _tiny_manifest(instance)
    jsonl_path = tmp_path / "results_details.jsonl"
    csv_path = tmp_path / "results.csv"

    plan = build_random_priority_run_plan(manifest, k=2, base_seed=2026)
    assert len(plan) == 2

    first_record = execute_random_priority_plan_entry(
        grid_map=grid_map,
        scenarios=scenarios,
        entry=plan[0],
        base_seed=2026,
        max_timestep=manifest.max_timestep,
    )

    store = RandomPriorityCheckpointStore(
        results_dir=tmp_path,
        csv_path=csv_path,
        jsonl_path=jsonl_path,
    )
    store.append_record(first_record)

    completed_keys = load_checkpoint_records(jsonl_path)
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
            base_seed=2026,
            max_timestep=manifest.max_timestep,
        )
        store.append_record(record)
        executed_indices.append(entry.ordering.ordering_index)

    assert executed_indices == [1]
    assert len(load_checkpoint_records(jsonl_path)) == 2


def test_jsonl_roundtrip_preserves_ordering_fields(tmp_path: Path) -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    manifest = _tiny_manifest(instance)
    entry = build_random_priority_run_plan(manifest, k=1, base_seed=2026)[0]

    record = execute_random_priority_plan_entry(
        grid_map=grid_map,
        scenarios=scenarios,
        entry=entry,
        base_seed=2026,
        max_timestep=manifest.max_timestep,
    )

    store = RandomPriorityCheckpointStore(
        results_dir=tmp_path,
        csv_path=tmp_path / "results.csv",
        jsonl_path=tmp_path / "results_details.jsonl",
    )
    store.append_record(record)

    restored = next(iter(load_checkpoint_records(tmp_path / "results_details.jsonl").values()))
    assert restored.ordering_index == record.ordering_index
    assert restored.ordering_seed == record.ordering_seed
    assert restored.agent_order == record.agent_order
    assert restored.instance_id == record.instance_id


def test_smoke_configuration_executes_with_prioritized_planning() -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    manifest = _tiny_manifest(instance)
    entry = build_random_priority_run_plan(
        manifest,
        k=2,
        base_seed=2026,
        instance_ids=(instance.instance_id,),
    )[0]

    record = execute_random_priority_plan_entry(
        grid_map=grid_map,
        scenarios=scenarios,
        entry=entry,
        base_seed=2026,
        max_timestep=manifest.max_timestep,
    )

    assert record.success is True
    assert record.termination_reason == TERMINATION_SUCCESS
    assert record.ordering_index == 0
    assert record.agent_order == entry.ordering.agent_order
    assert record.soc is not None
    assert record.makespan is not None
    assert record.conflict_count == 0
    assert record.pp_search_metrics is not None


def test_run_key_uses_instance_and_ordering_index() -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    manifest = _tiny_manifest(instance)
    entry = build_random_priority_run_plan(manifest, k=1, base_seed=2026)[0]

    scenario = MAPFScenario(
        agents=(
            MAPFAgent(agent_id=0, start=scenarios[0].start, goal=scenarios[0].goal),
            MAPFAgent(agent_id=1, start=scenarios[1].start, goal=scenarios[1].goal),
        )
    )
    record = execute_random_priority_run(
        grid_map=grid_map,
        scenario=scenario,
        instance=instance,
        ordering=entry.ordering,
        base_seed=2026,
        max_timestep=manifest.max_timestep,
    )

    assert run_key(record) == RandomPriorityRunKey("bench_n02_low_000", 0)


def test_checkpoint_store_detects_duplicate_ordering_keys(tmp_path: Path) -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    manifest = _tiny_manifest(instance)
    entry = build_random_priority_run_plan(manifest, k=1, base_seed=2026)[0]
    record = execute_random_priority_plan_entry(
        grid_map=grid_map,
        scenarios=scenarios,
        entry=entry,
        base_seed=2026,
        max_timestep=manifest.max_timestep,
    )

    jsonl_path = tmp_path / "results_details.jsonl"
    line = json.dumps(
        {
            "instance_id": record.instance_id,
            "agent_count": record.agent_count,
            "interaction_level": record.interaction_level,
            "ordering_index": record.ordering_index,
            "ordering_seed": record.ordering_seed,
            "agent_order": list(record.agent_order),
            "base_seed": record.base_seed,
            "success": record.success,
            "termination_reason": record.termination_reason,
            "execution_time_ms": record.execution_time_ms,
            "soc": record.soc,
            "makespan": record.makespan,
            "conflict_count": record.conflict_count,
            "pp_search_metrics": {
                "low_level_searches": record.pp_search_metrics.low_level_searches,
                "agents_planned": record.pp_search_metrics.agents_planned,
            },
        }
    )
    jsonl_path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Duplicate random-priority run keys"):
        load_checkpoint_records(jsonl_path)
