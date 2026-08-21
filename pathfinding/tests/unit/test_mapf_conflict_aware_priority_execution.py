from __future__ import annotations

import json
from pathlib import Path

import pytest

from pathfinding.src.algorithms.mapf.models import MAPFAgent, MAPFResult, MAPFScenario
from pathfinding.src.algorithms.mapf.prioritized_planning import (
    PrioritizedPlanningRunResult,
    PrioritizedPlanningStats,
)
from pathfinding.src.core.models import Position, Scenario
from pathfinding.src.experiments.mapf_benchmark_execution import (
    MAIN_BENCHMARK_RESULTS_DIR,
    TERMINATION_FAILURE,
    TERMINATION_SUCCESS,
)
from pathfinding.src.experiments.mapf_benchmark_instances import (
    MAPFBenchmarkInstance,
    MAPFBenchmarkManifest,
    MAPFInteractionLevel,
    load_benchmark_manifest,
)
from pathfinding.src.experiments.mapf_conflict_aware_priority_execution import (
    DEFAULT_CONFLICT_AWARE_STRATEGIES,
    MAIN_CONFLICT_AWARE_RESULTS_DIR,
    TERMINATION_ORDERING_FAILURE,
    ConflictAwarePriorityCheckpointStore,
    ConflictAwarePriorityRunKey,
    ConflictAwareStrategy,
    _ordering_sort_key,
    build_conflict_aware_priority_run_plan,
    execute_conflict_aware_priority_plan_entry,
    execute_conflict_aware_priority_run,
    load_checkpoint_records,
    original_index_order,
    run_key,
    validate_conflict_aware_priority_run_record,
)
from pathfinding.src.experiments.mapf_priority_strategy_execution import (
    MAIN_PRIORITY_STRATEGY_RESULTS_DIR,
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


def test_full_plan_contains_81_run_identities() -> None:
    manifest_path = Path(
        "pathfinding/results/mapf_benchmarks/AR0204SR_manifest.json"
    )
    manifest = load_benchmark_manifest(manifest_path)

    plan = build_conflict_aware_priority_run_plan(
        manifest,
        strategies=DEFAULT_CONFLICT_AWARE_STRATEGIES,
        agent_counts=(5, 10, 20),
        interaction_levels=("low", "medium", "high"),
    )

    assert len(manifest.instances) == 27
    assert len(plan) == 81

    keys = {
        ConflictAwarePriorityRunKey(entry.instance.instance_id, entry.strategy)
        for entry in plan
    }
    assert len(keys) == 81


def test_strategy_dispatch_maps_all_frozen_strategies() -> None:
    for strategy in DEFAULT_CONFLICT_AWARE_STRATEGIES:
        sort_key = _ordering_sort_key(strategy)
        assert callable(sort_key)


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported strategy"):
        _ordering_sort_key("not_a_strategy")  # type: ignore[arg-type]


def test_resume_key_is_instance_and_strategy() -> None:
    instance = _instance("bench_n02_low_000", 2)
    plan = build_conflict_aware_priority_run_plan(
        _tiny_manifest(instance),
        strategies=DEFAULT_CONFLICT_AWARE_STRATEGIES,
    )

    keys = [
        ConflictAwarePriorityRunKey(entry.instance.instance_id, entry.strategy)
        for entry in plan
    ]
    assert keys == [
        ConflictAwarePriorityRunKey("bench_n02_low_000", ConflictAwareStrategy.CDF_H),
        ConflictAwarePriorityRunKey("bench_n02_low_000", ConflictAwareStrategy.CDF_L),
        ConflictAwarePriorityRunKey("bench_n02_low_000", ConflictAwareStrategy.SPF_CD),
    ]


def test_completed_cdf_h_does_not_skip_other_strategies_for_same_instance(
    tmp_path: Path,
) -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    manifest = _tiny_manifest(instance)
    plan = build_conflict_aware_priority_run_plan(
        manifest,
        strategies=DEFAULT_CONFLICT_AWARE_STRATEGIES,
    )

    store = ConflictAwarePriorityCheckpointStore(
        results_dir=tmp_path,
        csv_path=tmp_path / "results.csv",
        jsonl_path=tmp_path / "results_details.jsonl",
    )
    store.append_record(
        execute_conflict_aware_priority_plan_entry(
            grid_map=grid_map,
            scenarios=scenarios,
            entry=plan[0],
            max_timestep=manifest.max_timestep,
        )
    )

    completed_keys = load_checkpoint_records(tmp_path / "results_details.jsonl")
    executed_strategies: list[ConflictAwareStrategy] = []

    for entry in plan:
        key = ConflictAwarePriorityRunKey(entry.instance.instance_id, entry.strategy)
        if key in completed_keys:
            continue
        record = execute_conflict_aware_priority_plan_entry(
            grid_map=grid_map,
            scenarios=scenarios,
            entry=entry,
            max_timestep=manifest.max_timestep,
        )
        store.append_record(record)
        executed_strategies.append(entry.strategy)

    assert executed_strategies == [
        ConflictAwareStrategy.CDF_L,
        ConflictAwareStrategy.SPF_CD,
    ]


def test_original_index_order_uses_scenario_position_not_agent_id() -> None:
    original = MAPFScenario(
        agents=(
            MAPFAgent(
                agent_id=17,
                start=Position(row=0, col=0),
                goal=Position(row=0, col=1),
            ),
            MAPFAgent(
                agent_id=42,
                start=Position(row=1, col=0),
                goal=Position(row=1, col=1),
            ),
            MAPFAgent(
                agent_id=8,
                start=Position(row=2, col=0),
                goal=Position(row=2, col=1),
            ),
        )
    )
    reordered = MAPFScenario(
        agents=(
            original.agents[2],
            original.agents[0],
            original.agents[1],
        )
    )

    assert original_index_order(original, reordered) == (2, 0, 1)
    assert tuple(agent.agent_id for agent in reordered.agents) == (8, 17, 42)


def test_agent_order_and_original_index_order_on_successful_run() -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    original = MAPFScenario(
        agents=(
            MAPFAgent(agent_id=17, start=scenarios[0].start, goal=scenarios[0].goal),
            MAPFAgent(agent_id=42, start=scenarios[1].start, goal=scenarios[1].goal),
        )
    )

    record = execute_conflict_aware_priority_run(
        grid_map=grid_map,
        scenario=original,
        instance=instance,
        strategy=ConflictAwareStrategy.CDF_H,
        max_timestep=8,
    )

    assert record.success is True
    assert record.agent_order is not None
    assert record.original_index_order is not None
    assert len(record.agent_order) == len(record.original_index_order)
    for agent_id, original_index in zip(
        record.agent_order,
        record.original_index_order,
        strict=True,
    ):
        assert original.agents[original_index].agent_id == agent_id


def test_conflict_diagnostics_are_recorded_on_success() -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    manifest = _tiny_manifest(instance)
    entry = build_conflict_aware_priority_run_plan(
        manifest,
        strategies=(ConflictAwareStrategy.CDF_H,),
    )[0]

    record = execute_conflict_aware_priority_plan_entry(
        grid_map=grid_map,
        scenarios=scenarios,
        entry=entry,
        max_timestep=manifest.max_timestep,
    )

    assert record.agent_degrees is not None
    assert len(record.agent_degrees) == instance.agent_count
    assert record.independent_conflict_count is not None
    assert record.independent_conflict_pair_count is not None
    assert record.ordering_low_level_searches == instance.agent_count
    assert record.incident_conflict_counts is not None
    assert len(record.incident_conflict_counts) == instance.agent_count


def test_ordering_failure_prevents_pp_execution() -> None:
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

    record = execute_conflict_aware_priority_run(
        grid_map=grid_map,
        scenario=scenario,
        instance=instance,
        strategy=ConflictAwareStrategy.CDF_H,
        max_timestep=1,
    )

    assert record.success is False
    assert record.termination_reason == TERMINATION_ORDERING_FAILURE
    assert record.pp_time_ms is None
    assert record.total_time_ms == record.ordering_time_ms
    assert record.agent_order is None
    assert record.original_index_order is None
    assert record.agent_degrees is None
    assert record.pp_search_metrics is None
    validate_conflict_aware_priority_run_record(record)


def test_pp_failure_preserves_ordering_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    scenario = MAPFScenario(
        agents=(
            MAPFAgent(agent_id=0, start=scenarios[0].start, goal=scenarios[0].goal),
            MAPFAgent(agent_id=1, start=scenarios[1].start, goal=scenarios[1].goal),
        )
    )

    def fake_pp(**kwargs: object) -> PrioritizedPlanningRunResult:
        return PrioritizedPlanningRunResult(
            result=MAPFResult(success=False, paths=()),
            stats=PrioritizedPlanningStats(
                low_level_searches=1,
                agents_planned=0,
            ),
            termination_reason=TERMINATION_FAILURE,
        )

    monkeypatch.setattr(
        "pathfinding.src.experiments.mapf_conflict_aware_priority_execution.plan_prioritized_with_stats",
        fake_pp,
    )

    record = execute_conflict_aware_priority_run(
        grid_map=grid_map,
        scenario=scenario,
        instance=instance,
        strategy=ConflictAwareStrategy.SPF_CD,
        max_timestep=8,
    )

    assert record.success is False
    assert record.termination_reason == TERMINATION_FAILURE
    assert record.agent_order is not None
    assert record.original_index_order is not None
    assert record.agent_degrees is not None
    assert record.independent_conflict_count is not None
    assert record.soc is None
    assert record.makespan is None
    assert record.conflict_count is None
    validate_conflict_aware_priority_run_record(record)


def test_total_time_ms_equals_ordering_plus_pp() -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    scenario = MAPFScenario(
        agents=(
            MAPFAgent(agent_id=0, start=scenarios[0].start, goal=scenarios[0].goal),
            MAPFAgent(agent_id=1, start=scenarios[1].start, goal=scenarios[1].goal),
        )
    )

    for strategy in DEFAULT_CONFLICT_AWARE_STRATEGIES:
        record = execute_conflict_aware_priority_run(
            grid_map=grid_map,
            scenario=scenario,
            instance=instance,
            strategy=strategy,
            max_timestep=8,
        )
        validate_conflict_aware_priority_run_record(record)
        assert record.pp_time_ms is not None
        assert record.total_time_ms == pytest.approx(
            record.ordering_time_ms + record.pp_time_ms
        )


def test_successful_run_records_soc_makespan_and_zero_conflicts() -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    manifest = _tiny_manifest(instance)
    entry = build_conflict_aware_priority_run_plan(
        manifest,
        strategies=(ConflictAwareStrategy.CDF_L,),
    )[0]

    record = execute_conflict_aware_priority_plan_entry(
        grid_map=grid_map,
        scenarios=scenarios,
        entry=entry,
        max_timestep=manifest.max_timestep,
    )

    assert record.success is True
    assert record.termination_reason == TERMINATION_SUCCESS
    assert record.soc is not None
    assert record.makespan is not None
    assert record.conflict_count == 0
    validate_conflict_aware_priority_run_record(record)


def test_deterministic_strategy_execution() -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    scenario = MAPFScenario(
        agents=(
            MAPFAgent(agent_id=0, start=scenarios[0].start, goal=scenarios[0].goal),
            MAPFAgent(agent_id=1, start=scenarios[1].start, goal=scenarios[1].goal),
        )
    )

    first = execute_conflict_aware_priority_run(
        grid_map=grid_map,
        scenario=scenario,
        instance=instance,
        strategy=ConflictAwareStrategy.CDF_H,
        max_timestep=8,
    )
    second = execute_conflict_aware_priority_run(
        grid_map=grid_map,
        scenario=scenario,
        instance=instance,
        strategy=ConflictAwareStrategy.CDF_H,
        max_timestep=8,
    )

    assert first.agent_order == second.agent_order
    assert first.original_index_order == second.original_index_order
    assert first.agent_degrees == second.agent_degrees


def test_jsonl_roundtrip_preserves_ordering_and_index_fields(tmp_path: Path) -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    manifest = _tiny_manifest(instance)
    entry = build_conflict_aware_priority_run_plan(
        manifest,
        strategies=(ConflictAwareStrategy.SPF_CD,),
    )[0]

    record = execute_conflict_aware_priority_plan_entry(
        grid_map=grid_map,
        scenarios=scenarios,
        entry=entry,
        max_timestep=manifest.max_timestep,
    )

    store = ConflictAwarePriorityCheckpointStore(
        results_dir=tmp_path,
        csv_path=tmp_path / "results.csv",
        jsonl_path=tmp_path / "results_details.jsonl",
    )
    store.append_record(record)

    restored = next(
        iter(load_checkpoint_records(tmp_path / "results_details.jsonl").values())
    )
    assert restored.strategy == ConflictAwareStrategy.SPF_CD
    assert restored.agent_order == record.agent_order
    assert restored.original_index_order == record.original_index_order
    assert restored.agent_degrees == record.agent_degrees
    assert restored.independent_conflict_count == record.independent_conflict_count
    assert restored.ordering_time_ms == record.ordering_time_ms
    assert restored.total_time_ms == record.total_time_ms


def test_atomic_csv_is_written(tmp_path: Path) -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    manifest = _tiny_manifest(instance)
    plan = build_conflict_aware_priority_run_plan(
        manifest,
        strategies=DEFAULT_CONFLICT_AWARE_STRATEGIES,
    )

    store = ConflictAwarePriorityCheckpointStore(
        results_dir=tmp_path,
        csv_path=tmp_path / "results.csv",
        jsonl_path=tmp_path / "results_details.jsonl",
    )
    for entry in plan:
        store.append_record(
            execute_conflict_aware_priority_plan_entry(
                grid_map=grid_map,
                scenarios=scenarios,
                entry=entry,
                max_timestep=manifest.max_timestep,
            )
        )

    csv_path = tmp_path / "results.csv"
    assert csv_path.is_file()
    assert not csv_path.with_suffix(".csv.tmp").exists()
    assert csv_path.read_text(encoding="utf-8").count("\n") >= 4


def test_checkpoint_store_detects_duplicate_strategy_keys(tmp_path: Path) -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    manifest = _tiny_manifest(instance)
    entry = build_conflict_aware_priority_run_plan(
        manifest,
        strategies=(ConflictAwareStrategy.CDF_H,),
    )[0]
    record = execute_conflict_aware_priority_plan_entry(
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
            "original_index_order": list(record.original_index_order or ()),
            "agent_degrees": list(record.agent_degrees or ()),
            "independent_conflict_count": record.independent_conflict_count,
            "independent_conflict_pair_count": record.independent_conflict_pair_count,
            "ordering_low_level_searches": record.ordering_low_level_searches,
            "incident_conflict_counts": list(record.incident_conflict_counts or ()),
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

    with pytest.raises(RuntimeError, match="Duplicate conflict-aware priority run keys"):
        load_checkpoint_records(jsonl_path)


def test_run_key_uses_instance_and_strategy() -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    instance = _instance("bench_n02_low_000", 2)
    manifest = _tiny_manifest(instance)
    entry = build_conflict_aware_priority_run_plan(
        manifest,
        strategies=(ConflictAwareStrategy.CDF_L,),
    )[0]
    record = execute_conflict_aware_priority_plan_entry(
        grid_map=grid_map,
        scenarios=scenarios,
        entry=entry,
        max_timestep=manifest.max_timestep,
    )

    assert run_key(record) == ConflictAwarePriorityRunKey(
        "bench_n02_low_000",
        ConflictAwareStrategy.CDF_L,
    )


def test_result_directories_are_distinct_from_mapf6() -> None:
    assert MAIN_BENCHMARK_RESULTS_DIR != MAIN_CONFLICT_AWARE_RESULTS_DIR
    assert MAIN_PRIORITY_STRATEGY_RESULTS_DIR != MAIN_CONFLICT_AWARE_RESULTS_DIR
    assert str(MAIN_CONFLICT_AWARE_RESULTS_DIR).endswith(
        "mapf_conflict_aware_priority_execution"
    )
