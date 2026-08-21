from __future__ import annotations

import json
from pathlib import Path

import pytest

from pathfinding.src.algorithms.mapf.models import MAPFAgent, MAPFScenario
from pathfinding.src.core.models import Position, Scenario
from pathfinding.src.experiments.mapf_benchmark_execution import (
    MAIN_BENCHMARK_RESULTS_DIR,
    TERMINATION_SUCCESS,
)
from pathfinding.src.experiments.mapf_benchmark_instances import (
    MAPFBenchmarkInstance,
    MAPFBenchmarkManifest,
    MAPFInteractionLevel,
    load_benchmark_manifest,
)
from pathfinding.src.experiments.mapf_priority_strategy_execution import (
    DEFAULT_PRIORITY_STRATEGIES,
    MAIN_PRIORITY_STRATEGY_RESULTS_DIR,
    TERMINATION_ORDERING_FAILURE,
    PriorityStrategy,
    PriorityStrategyCheckpointStore,
    PriorityStrategyRunKey,
    build_priority_strategy_run_plan,
    execute_priority_strategy_plan_entry,
    execute_priority_strategy_run,
    load_checkpoint_records,
    run_key,
    validate_priority_strategy_run_record,
)
from pathfinding.src.experiments.mapf_random_priority_execution import (
    MAIN_RANDOM_PRIORITY_RESULTS_DIR,
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


def test_full_plan_contains_54_run_identities() -> None:
    manifest_path = Path(
        "pathfinding/results/mapf_benchmarks/AR0204SR_manifest.json"
    )
    manifest = load_benchmark_manifest(manifest_path)

    plan = build_priority_strategy_run_plan(
        manifest,
        strategies=DEFAULT_PRIORITY_STRATEGIES,
        agent_counts=(5, 10, 20),
        interaction_levels=("low", "medium", "high"),
    )

    assert len(manifest.instances) == 27
    assert len(plan) == 54

    keys = {
        PriorityStrategyRunKey(entry.instance.instance_id, entry.strategy)
        for entry in plan
    }
    assert len(keys) == 54


def test_resume_key_is_instance_and_strategy() -> None:
    instance = _instance("bench_n02_low_000", 2)
    plan = build_priority_strategy_run_plan(
        _tiny_manifest(instance),
        strategies=DEFAULT_PRIORITY_STRATEGIES,
    )

    keys = [
        PriorityStrategyRunKey(entry.instance.instance_id, entry.strategy)
        for entry in plan
    ]
    assert keys == [
        PriorityStrategyRunKey("bench_n02_low_000", PriorityStrategy.SPF),
        PriorityStrategyRunKey("bench_n02_low_000", PriorityStrategy.LPF),
    ]


def test_completed_spf_does_not_skip_lpf_for_same_instance(tmp_path: Path) -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    manifest = _tiny_manifest(instance)
    plan = build_priority_strategy_run_plan(manifest, strategies=DEFAULT_PRIORITY_STRATEGIES)

    store = PriorityStrategyCheckpointStore(
        results_dir=tmp_path,
        csv_path=tmp_path / "results.csv",
        jsonl_path=tmp_path / "results_details.jsonl",
    )
    store.append_record(
        execute_priority_strategy_plan_entry(
            grid_map=grid_map,
            scenarios=scenarios,
            entry=plan[0],
            max_timestep=manifest.max_timestep,
        )
    )

    completed_keys = load_checkpoint_records(tmp_path / "results_details.jsonl")
    executed_strategies: list[PriorityStrategy] = []

    for entry in plan:
        key = PriorityStrategyRunKey(entry.instance.instance_id, entry.strategy)
        if key in completed_keys:
            continue
        record = execute_priority_strategy_plan_entry(
            grid_map=grid_map,
            scenarios=scenarios,
            entry=entry,
            max_timestep=manifest.max_timestep,
        )
        store.append_record(record)
        executed_strategies.append(entry.strategy)

    assert executed_strategies == [PriorityStrategy.LPF]


def test_resume_skips_completed_strategy_runs(tmp_path: Path) -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    manifest = _tiny_manifest(instance)
    plan = build_priority_strategy_run_plan(manifest, strategies=DEFAULT_PRIORITY_STRATEGIES)

    first_record = execute_priority_strategy_plan_entry(
        grid_map=grid_map,
        scenarios=scenarios,
        entry=plan[0],
        max_timestep=manifest.max_timestep,
    )

    store = PriorityStrategyCheckpointStore(
        results_dir=tmp_path,
        csv_path=tmp_path / "results.csv",
        jsonl_path=tmp_path / "results_details.jsonl",
    )
    store.append_record(first_record)

    completed_keys = load_checkpoint_records(tmp_path / "results_details.jsonl")
    executed_strategies: list[PriorityStrategy] = []

    for entry in plan:
        key = PriorityStrategyRunKey(entry.instance.instance_id, entry.strategy)
        if key in completed_keys:
            continue
        record = execute_priority_strategy_plan_entry(
            grid_map=grid_map,
            scenarios=scenarios,
            entry=entry,
            max_timestep=manifest.max_timestep,
        )
        store.append_record(record)
        executed_strategies.append(entry.strategy)

    assert executed_strategies == [PriorityStrategy.LPF]
    assert len(load_checkpoint_records(tmp_path / "results_details.jsonl")) == 2


def test_jsonl_roundtrip_preserves_timing_and_strategy_fields(tmp_path: Path) -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    manifest = _tiny_manifest(instance)
    entry = build_priority_strategy_run_plan(
        manifest,
        strategies=(PriorityStrategy.SPF,),
    )[0]

    record = execute_priority_strategy_plan_entry(
        grid_map=grid_map,
        scenarios=scenarios,
        entry=entry,
        max_timestep=manifest.max_timestep,
    )

    store = PriorityStrategyCheckpointStore(
        results_dir=tmp_path,
        csv_path=tmp_path / "results.csv",
        jsonl_path=tmp_path / "results_details.jsonl",
    )
    store.append_record(record)

    restored = next(iter(load_checkpoint_records(tmp_path / "results_details.jsonl").values()))
    assert restored.strategy == PriorityStrategy.SPF
    assert restored.agent_order == record.agent_order
    assert restored.ordering_time_ms == record.ordering_time_ms
    assert restored.pp_time_ms == record.pp_time_ms
    assert restored.total_time_ms == record.total_time_ms


def test_total_time_ms_equals_ordering_plus_pp() -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    manifest = _tiny_manifest(instance)
    scenario = MAPFScenario(
        agents=(
            MAPFAgent(agent_id=0, start=scenarios[0].start, goal=scenarios[0].goal),
            MAPFAgent(agent_id=1, start=scenarios[1].start, goal=scenarios[1].goal),
        )
    )

    for strategy in DEFAULT_PRIORITY_STRATEGIES:
        record = execute_priority_strategy_run(
            grid_map=grid_map,
            scenario=scenario,
            instance=instance,
            strategy=strategy,
            max_timestep=manifest.max_timestep,
        )
        validate_priority_strategy_run_record(record)
        assert record.pp_time_ms is not None
        assert record.total_time_ms == pytest.approx(
            record.ordering_time_ms + record.pp_time_ms
        )


def test_tiny_spf_run_executes_successfully() -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    manifest = _tiny_manifest(instance)
    entry = build_priority_strategy_run_plan(
        manifest,
        strategies=(PriorityStrategy.SPF,),
    )[0]

    record = execute_priority_strategy_plan_entry(
        grid_map=grid_map,
        scenarios=scenarios,
        entry=entry,
        max_timestep=manifest.max_timestep,
    )

    assert record.success is True
    assert record.strategy == PriorityStrategy.SPF
    assert record.termination_reason == TERMINATION_SUCCESS
    assert record.agent_order is not None
    assert record.soc is not None
    validate_priority_strategy_run_record(record)


def test_tiny_lpf_run_executes_successfully() -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    manifest = _tiny_manifest(instance)
    entry = build_priority_strategy_run_plan(
        manifest,
        strategies=(PriorityStrategy.LPF,),
    )[0]

    record = execute_priority_strategy_plan_entry(
        grid_map=grid_map,
        scenarios=scenarios,
        entry=entry,
        max_timestep=manifest.max_timestep,
    )

    assert record.success is True
    assert record.strategy == PriorityStrategy.LPF
    assert record.termination_reason == TERMINATION_SUCCESS
    assert record.agent_order is not None
    validate_priority_strategy_run_record(record)


def test_failed_ordering_has_no_fake_solution_metrics() -> None:
    grid_map = build_grid_map([[0, 0, 0]], name="bench.map")
    instance = _instance("bench_n02_low_000", 1)
    scenario = MAPFScenario(
        agents=(
            MAPFAgent(
                agent_id=0,
                start=Position(row=0, col=0),
                goal=Position(row=0, col=2),
            ),
        )
    )

    record = execute_priority_strategy_run(
        grid_map=grid_map,
        scenario=scenario,
        instance=instance,
        strategy=PriorityStrategy.SPF,
        max_timestep=1,
    )

    assert record.success is False
    assert record.termination_reason == TERMINATION_ORDERING_FAILURE
    assert record.pp_time_ms is None
    assert record.total_time_ms == record.ordering_time_ms
    assert record.agent_order is None
    assert record.soc is None
    assert record.makespan is None
    assert record.conflict_count is None
    assert record.pp_search_metrics is None
    assert record.error_message is not None
    validate_priority_strategy_run_record(record)


def test_historical_result_directories_are_distinct() -> None:
    assert MAIN_BENCHMARK_RESULTS_DIR != MAIN_PRIORITY_STRATEGY_RESULTS_DIR
    assert MAIN_RANDOM_PRIORITY_RESULTS_DIR != MAIN_PRIORITY_STRATEGY_RESULTS_DIR
    assert str(MAIN_PRIORITY_STRATEGY_RESULTS_DIR).endswith(
        "mapf_priority_strategy_execution"
    )


def test_checkpoint_store_detects_duplicate_strategy_keys(tmp_path: Path) -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    manifest = _tiny_manifest(instance)
    entry = build_priority_strategy_run_plan(
        manifest,
        strategies=(PriorityStrategy.SPF,),
    )[0]
    record = execute_priority_strategy_plan_entry(
        grid_map=grid_map,
        scenarios=scenarios,
        entry=entry,
        max_timestep=manifest.max_timestep,
    )

    jsonl_path = tmp_path / "results_details.jsonl"
    line = json.dumps(
        {
            "instance_id": record.instance_id,
            "agent_count": record.agent_count,
            "interaction_level": record.interaction_level,
            "strategy": record.strategy.value,
            "agent_order": list(record.agent_order or ()),
            "success": record.success,
            "termination_reason": record.termination_reason,
            "ordering_time_ms": record.ordering_time_ms,
            "pp_time_ms": record.pp_time_ms,
            "total_time_ms": record.total_time_ms,
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

    with pytest.raises(RuntimeError, match="Duplicate priority-strategy run keys"):
        load_checkpoint_records(jsonl_path)


def test_run_key_uses_instance_and_strategy() -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    manifest = _tiny_manifest(instance)
    entry = build_priority_strategy_run_plan(
        manifest,
        strategies=(PriorityStrategy.LPF,),
    )[0]
    record = execute_priority_strategy_plan_entry(
        grid_map=grid_map,
        scenarios=scenarios,
        entry=entry,
        max_timestep=manifest.max_timestep,
    )

    assert run_key(record) == PriorityStrategyRunKey("bench_n02_low_000", PriorityStrategy.LPF)
