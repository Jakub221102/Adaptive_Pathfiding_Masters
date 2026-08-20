from __future__ import annotations

import json
from pathlib import Path

import pytest

from pathfinding.src.algorithms.mapf.models import MAPFAgent, MAPFScenario
from pathfinding.src.core.models import Position, Scenario
from pathfinding.src.experiments.mapf_benchmark_execution import (
    DEFAULT_BENCHMARK_ALGORITHMS,
    TERMINATION_ERROR,
    TERMINATION_EXPANSION_LIMIT,
    TERMINATION_FAILURE,
    TERMINATION_SUCCESS,
    BenchmarkCheckpointStore,
    MAPFBenchmarkAlgorithm,
    MAPFBenchmarkCBSLimits,
    MAPFBenchmarkExecutionReport,
    MAPFBenchmarkRunKey,
    build_benchmark_run_plan,
    create_error_benchmark_record,
    execute_benchmark_run,
    execute_benchmark_suite,
    export_execution_report_csv,
    load_checkpoint_records,
    load_execution_report,
    rewrite_checkpoint_csv,
    save_execution_report,
    validate_benchmark_run_record,
    validate_execution_report,
)
from pathfinding.src.experiments.mapf_benchmark_instances import (
    MAPFBenchmarkInstance,
    MAPFBenchmarkManifest,
    MAPFInteractionLevel,
    save_benchmark_manifest,
)
from pathfinding.tests.helpers import build_grid_map


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


def _instance(
    instance_id: str,
    agent_count: int,
    level: MAPFInteractionLevel,
    scenario_indices: tuple[int, ...],
) -> MAPFBenchmarkInstance:
    return MAPFBenchmarkInstance(
        instance_id=instance_id,
        agent_count=agent_count,
        interaction_level=level,
        scenario_indices=scenario_indices,
        independent_conflict_count=0,
        conflicting_agent_pair_count=0,
        independent_soc=4,
        independent_makespan=2,
        vertex_conflict_count=0,
        edge_conflict_count=0,
    )


def _tiny_manifest() -> MAPFBenchmarkManifest:
    return MAPFBenchmarkManifest(
        map_name="bench.map",
        scenario_name="bench.map.scen",
        seed=2026,
        max_timestep=8,
        min_reference_length=1.0,
        instances=(
            _instance("bench_n02_low_000", 2, MAPFInteractionLevel.LOW, (0, 1)),
        ),
    )


def _two_instance_manifest() -> MAPFBenchmarkManifest:
    return MAPFBenchmarkManifest(
        map_name="bench.map",
        scenario_name="bench.map.scen",
        seed=2026,
        max_timestep=8,
        min_reference_length=1.0,
        instances=(
            _instance("bench_n02_low_001", 2, MAPFInteractionLevel.LOW, (0, 1)),
            _instance("bench_n02_low_000", 2, MAPFInteractionLevel.LOW, (0, 1)),
        ),
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


def test_execute_benchmark_run_all_algorithms_succeed_on_low_interaction() -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    manifest = _tiny_manifest()
    instance = manifest.instances[0]
    scenario = MAPFScenario(
        agents=(
            MAPFAgent(agent_id=0, start=scenarios[0].start, goal=scenarios[0].goal),
            MAPFAgent(agent_id=1, start=scenarios[1].start, goal=scenarios[1].goal),
        )
    )

    for algorithm in DEFAULT_BENCHMARK_ALGORITHMS:
        record = execute_benchmark_run(
            grid_map=grid_map,
            scenario=scenario,
            instance=instance,
            algorithm=algorithm,
            max_timestep=manifest.max_timestep,
        )

        assert record.success is True
        assert record.termination_reason == TERMINATION_SUCCESS
        assert record.remaining_conflicts == 0
        assert record.soc is not None
        assert record.makespan is not None
        assert record.execution_time_ms >= 0.0

        if algorithm == MAPFBenchmarkAlgorithm.FIXED_PRIORITY_PP:
            assert record.pp_search_metrics is not None
            assert record.pp_search_metrics.low_level_searches == 2
            assert record.cbs_search_metrics is None
        else:
            assert record.cbs_search_metrics is not None
            assert record.pp_search_metrics is None


def test_execute_benchmark_suite_runs_instance_algorithm_cross_product() -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    manifest = _tiny_manifest()

    report = execute_benchmark_suite(
        grid_map=grid_map,
        scenarios=scenarios,
        manifest=manifest,
        manifest_path=Path("bench_manifest.json"),
        algorithms=DEFAULT_BENCHMARK_ALGORITHMS,
    )

    assert len(report.runs) == len(DEFAULT_BENCHMARK_ALGORITHMS)
    assert report.max_timestep == manifest.max_timestep
    assert all(record.success for record in report.runs)
    validate_execution_report(report)


def test_build_benchmark_run_plan_uses_algorithm_phases() -> None:
    manifest = _two_instance_manifest()

    plan = build_benchmark_run_plan(
        manifest,
        algorithms=(
            MAPFBenchmarkAlgorithm.FIXED_PRIORITY_PP,
            MAPFBenchmarkAlgorithm.CBS_BASIC,
        ),
    )

    assert [entry.algorithm for entry in plan] == [
        MAPFBenchmarkAlgorithm.FIXED_PRIORITY_PP,
        MAPFBenchmarkAlgorithm.FIXED_PRIORITY_PP,
        MAPFBenchmarkAlgorithm.CBS_BASIC,
        MAPFBenchmarkAlgorithm.CBS_BASIC,
    ]
    assert [entry.instance.instance_id for entry in plan[:2]] == [
        "bench_n02_low_000",
        "bench_n02_low_001",
    ]


def test_checkpoint_store_writes_after_each_record(tmp_path: Path) -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    manifest = _tiny_manifest()
    instance = manifest.instances[0]
    scenario = MAPFScenario(
        agents=(
            MAPFAgent(agent_id=0, start=scenarios[0].start, goal=scenarios[0].goal),
            MAPFAgent(agent_id=1, start=scenarios[1].start, goal=scenarios[1].goal),
        )
    )

    store = BenchmarkCheckpointStore(
        results_dir=tmp_path,
        csv_path=tmp_path / "results.csv",
        jsonl_path=tmp_path / "results_details.jsonl",
    )

    record = execute_benchmark_run(
        grid_map=grid_map,
        scenario=scenario,
        instance=instance,
        algorithm=MAPFBenchmarkAlgorithm.FIXED_PRIORITY_PP,
        max_timestep=manifest.max_timestep,
    )
    store.append_record(record)

    assert (tmp_path / "results_details.jsonl").is_file()
    assert (tmp_path / "results.csv").is_file()
    loaded = load_checkpoint_records(tmp_path / "results_details.jsonl")
    assert len(loaded) == 1
    assert loaded[
        MAPFBenchmarkRunKey("bench_n02_low_000", MAPFBenchmarkAlgorithm.FIXED_PRIORITY_PP)
    ].soc == record.soc


def test_resume_skips_completed_run_and_executes_remaining(tmp_path: Path) -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    manifest = _two_instance_manifest()
    jsonl_path = tmp_path / "results_details.jsonl"
    csv_path = tmp_path / "results.csv"

    first_instance = manifest.instances[0]
    scenario = MAPFScenario(
        agents=(
            MAPFAgent(agent_id=0, start=scenarios[0].start, goal=scenarios[0].goal),
            MAPFAgent(agent_id=1, start=scenarios[1].start, goal=scenarios[1].goal),
        )
    )
    completed = execute_benchmark_run(
        grid_map=grid_map,
        scenario=scenario,
        instance=first_instance,
        algorithm=MAPFBenchmarkAlgorithm.FIXED_PRIORITY_PP,
        max_timestep=manifest.max_timestep,
    )

    store = BenchmarkCheckpointStore(
        results_dir=tmp_path,
        csv_path=csv_path,
        jsonl_path=jsonl_path,
    )
    store.append_record(completed)

    plan = build_benchmark_run_plan(
        manifest,
        algorithms=(MAPFBenchmarkAlgorithm.FIXED_PRIORITY_PP,),
    )
    completed_keys = load_checkpoint_records(jsonl_path)
    executed: list[str] = []

    for entry in plan:
        key = MAPFBenchmarkRunKey(entry.instance.instance_id, entry.algorithm)
        if key in completed_keys:
            continue
        record = execute_benchmark_run(
            grid_map=grid_map,
            scenario=scenario,
            instance=entry.instance,
            algorithm=entry.algorithm,
            max_timestep=manifest.max_timestep,
        )
        store.append_record(record)
        executed.append(entry.instance.instance_id)

    assert executed == ["bench_n02_low_000"]
    assert len(load_checkpoint_records(jsonl_path)) == 2


def test_checkpoint_store_detects_duplicate_run_keys(tmp_path: Path) -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    manifest = _tiny_manifest()
    instance = manifest.instances[0]
    scenario = MAPFScenario(
        agents=(
            MAPFAgent(agent_id=0, start=scenarios[0].start, goal=scenarios[0].goal),
            MAPFAgent(agent_id=1, start=scenarios[1].start, goal=scenarios[1].goal),
        )
    )
    record = execute_benchmark_run(
        grid_map=grid_map,
        scenario=scenario,
        instance=instance,
        algorithm=MAPFBenchmarkAlgorithm.FIXED_PRIORITY_PP,
        max_timestep=manifest.max_timestep,
    )

    jsonl_path = tmp_path / "results_details.jsonl"
    line = json.dumps(
        {
            "instance_id": record.instance_id,
            "algorithm": record.algorithm.value,
            "agent_count": record.agent_count,
            "interaction_level": record.interaction_level,
            "success": record.success,
            "termination_reason": record.termination_reason,
            "execution_time_ms": record.execution_time_ms,
            "soc": record.soc,
            "makespan": record.makespan,
            "remaining_conflicts": record.remaining_conflicts,
            "independent_soc": record.independent_soc,
            "independent_makespan": record.independent_makespan,
            "independent_conflict_count": record.independent_conflict_count,
            "pp_search_metrics": {
                "low_level_searches": record.pp_search_metrics.low_level_searches,
                "agents_planned": record.pp_search_metrics.agents_planned,
            },
            "cbs_search_metrics": None,
        }
    )
    jsonl_path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Duplicate benchmark run keys"):
        load_checkpoint_records(jsonl_path)


def test_execution_report_json_roundtrip() -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    manifest = _tiny_manifest()

    report = execute_benchmark_suite(
        grid_map=grid_map,
        scenarios=scenarios,
        manifest=manifest,
        manifest_path=Path("bench_manifest.json"),
        algorithms=DEFAULT_BENCHMARK_ALGORITHMS,
    )

    output_path = Path("test_execution_report.json")
    save_execution_report(report, output_path)
    loaded = load_execution_report(output_path)

    assert loaded.manifest_path == report.manifest_path
    assert len(loaded.runs) == len(report.runs)
    assert loaded.runs[0].execution_time_ms == report.runs[0].execution_time_ms
    assert loaded.runs[0].remaining_conflicts == 0

    output_path.unlink(missing_ok=True)


def test_jsonl_roundtrip_preserves_important_fields(tmp_path: Path) -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    manifest = _tiny_manifest()
    instance = manifest.instances[0]
    scenario = MAPFScenario(
        agents=(
            MAPFAgent(agent_id=0, start=scenarios[0].start, goal=scenarios[0].goal),
            MAPFAgent(agent_id=1, start=scenarios[1].start, goal=scenarios[1].goal),
        )
    )
    record = execute_benchmark_run(
        grid_map=grid_map,
        scenario=scenario,
        instance=instance,
        algorithm=MAPFBenchmarkAlgorithm.CBS_BASIC,
        max_timestep=manifest.max_timestep,
    )

    store = BenchmarkCheckpointStore(
        results_dir=tmp_path,
        csv_path=tmp_path / "results.csv",
        jsonl_path=tmp_path / "results_details.jsonl",
    )
    store.append_record(record)

    loaded = load_checkpoint_records(tmp_path / "results_details.jsonl")
    restored = next(iter(loaded.values()))
    assert restored.instance_id == record.instance_id
    assert restored.algorithm == record.algorithm
    assert restored.success == record.success
    assert restored.termination_reason == TERMINATION_SUCCESS
    assert restored.soc == record.soc
    assert restored.cbs_search_metrics is not None


def test_export_execution_report_csv_writes_rows(tmp_path: Path) -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    manifest = _tiny_manifest()

    report = execute_benchmark_suite(
        grid_map=grid_map,
        scenarios=scenarios,
        manifest=manifest,
        manifest_path=tmp_path / "manifest.json",
        algorithms=(MAPFBenchmarkAlgorithm.CBS_BASIC,),
    )

    csv_path = tmp_path / "results.csv"
    export_execution_report_csv(report, csv_path)

    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert "execution_time_ms" in lines[0]
    assert "remaining_conflicts" in lines[0]


def test_execution_is_deterministic_for_fixed_inputs() -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    manifest = _tiny_manifest()

    first = execute_benchmark_suite(
        grid_map=grid_map,
        scenarios=scenarios,
        manifest=manifest,
        manifest_path=Path("bench_manifest.json"),
        algorithms=DEFAULT_BENCHMARK_ALGORITHMS,
    )
    second = execute_benchmark_suite(
        grid_map=grid_map,
        scenarios=scenarios,
        manifest=manifest,
        manifest_path=Path("bench_manifest.json"),
        algorithms=DEFAULT_BENCHMARK_ALGORITHMS,
    )

    assert len(first.runs) == len(second.runs)
    for left, right in zip(first.runs, second.runs, strict=True):
        assert left.instance_id == right.instance_id
        assert left.algorithm == right.algorithm
        assert left.success == right.success
        assert left.soc == right.soc
        assert left.makespan == right.makespan


def test_create_error_record_has_distinct_termination_reason() -> None:
    instance = _instance("bench_n02_low_000", 2, MAPFInteractionLevel.LOW, (0, 1))
    record = create_error_benchmark_record(
        instance=instance,
        algorithm=MAPFBenchmarkAlgorithm.CBS_BASIC,
        execution_time_ms=12.5,
        error_message="RuntimeError: synthetic failure",
    )

    assert record.success is False
    assert record.termination_reason == TERMINATION_ERROR
    assert record.error_message is not None
    validate_benchmark_run_record(record)


def test_cbs_expansion_limit_is_distinct_from_failure() -> None:
    grid_map = build_grid_map([[0, 0, 0]], name="bench.map")
    scenario = MAPFScenario(
        agents=(
            MAPFAgent(
                agent_id=0,
                start=Position(row=0, col=0),
                goal=Position(row=0, col=2),
            ),
            MAPFAgent(
                agent_id=1,
                start=Position(row=0, col=2),
                goal=Position(row=0, col=0),
            ),
        )
    )
    instance = _instance("bench_n02_high_000", 2, MAPFInteractionLevel.HIGH, (0, 1))

    record = execute_benchmark_run(
        grid_map=grid_map,
        scenario=scenario,
        instance=instance,
        algorithm=MAPFBenchmarkAlgorithm.CBS_BASIC,
        max_timestep=6,
        cbs_limits=MAPFBenchmarkCBSLimits(
            basic_max_expanded_nodes=0,
            cardinal_first_max_expanded_nodes=0,
        ),
    )

    assert record.success is False
    assert record.termination_reason == TERMINATION_EXPANSION_LIMIT
    assert record.termination_reason != TERMINATION_FAILURE
    assert record.cbs_search_metrics is not None
    validate_benchmark_run_record(record)


def test_pp_failure_uses_failure_termination_reason() -> None:
    grid_map = build_grid_map([[0, 0], [0, 0]], name="bench.map")
    scenario = MAPFScenario(
        agents=(
            MAPFAgent(
                agent_id=0,
                start=Position(row=0, col=0),
                goal=Position(row=1, col=1),
            ),
            MAPFAgent(
                agent_id=1,
                start=Position(row=1, col=0),
                goal=Position(row=0, col=1),
            ),
        )
    )
    instance = _instance("bench_n02_high_000", 2, MAPFInteractionLevel.HIGH, (0, 1))

    record = execute_benchmark_run(
        grid_map=grid_map,
        scenario=scenario,
        instance=instance,
        algorithm=MAPFBenchmarkAlgorithm.FIXED_PRIORITY_PP,
        max_timestep=1,
    )

    assert record.success is False
    assert record.termination_reason == TERMINATION_FAILURE
    assert record.pp_search_metrics is not None
    validate_benchmark_run_record(record)


def test_validate_execution_report_rejects_duplicate_runs() -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    manifest = _tiny_manifest()
    report = execute_benchmark_suite(
        grid_map=grid_map,
        scenarios=scenarios,
        manifest=manifest,
        manifest_path=Path("bench_manifest.json"),
        algorithms=(MAPFBenchmarkAlgorithm.CBS_BASIC,),
    )

    duplicate = MAPFBenchmarkExecutionReport(
        manifest_path=report.manifest_path,
        map_name=report.map_name,
        scenario_name=report.scenario_name,
        max_timestep=report.max_timestep,
        algorithms=report.algorithms,
        executed_at_utc=report.executed_at_utc,
        total_runtime_s=report.total_runtime_s,
        runs=report.runs + report.runs,
    )

    with pytest.raises(RuntimeError, match="duplicate execution record"):
        validate_execution_report(duplicate)


def test_manifest_file_execution_uses_stored_instances_not_regeneration(
    tmp_path: Path,
) -> None:
    grid_map, scenarios = _tiny_grid_and_scenarios()
    manifest = _tiny_manifest()
    manifest_path = tmp_path / "manifest.json"
    save_benchmark_manifest(manifest, manifest_path)

    report = execute_benchmark_suite(
        grid_map=grid_map,
        scenarios=scenarios,
        manifest=manifest,
        manifest_path=manifest_path,
        algorithms=DEFAULT_BENCHMARK_ALGORITHMS,
    )

    assert report.runs[0].instance_id == "bench_n02_low_000"
    assert report.runs[0].independent_soc == 4
    assert report.runs[0].independent_makespan == 2


def test_rewrite_checkpoint_csv_handles_none_fields(tmp_path: Path) -> None:
    instance = _instance("bench_n02_low_000", 2, MAPFInteractionLevel.LOW, (0, 1))
    record = create_error_benchmark_record(
        instance=instance,
        algorithm=MAPFBenchmarkAlgorithm.CBS_BASIC,
        execution_time_ms=1.0,
        error_message="ValueError: test",
    )
    csv_path = tmp_path / "results.csv"
    rewrite_checkpoint_csv((record,), csv_path)
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "error_message" in header
