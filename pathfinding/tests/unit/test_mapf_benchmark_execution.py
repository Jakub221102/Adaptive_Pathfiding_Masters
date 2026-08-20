from __future__ import annotations

import json
from pathlib import Path

import pytest

from pathfinding.src.algorithms.mapf.models import MAPFAgent, MAPFScenario
from pathfinding.src.core.models import Position, Scenario
from pathfinding.src.experiments.mapf_benchmark_execution import (
    DEFAULT_BENCHMARK_ALGORITHMS,
    MAPFBenchmarkAlgorithm,
    MAPFBenchmarkExecutionReport,
    execute_benchmark_run,
    execute_benchmark_suite,
    export_execution_report_csv,
    load_execution_report,
    save_execution_report,
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


def _tiny_manifest() -> MAPFBenchmarkManifest:
    instance = MAPFBenchmarkInstance(
        instance_id="bench_n02_low_000",
        agent_count=2,
        interaction_level=MAPFInteractionLevel.LOW,
        scenario_indices=(0, 1),
        independent_conflict_count=0,
        conflicting_agent_pair_count=0,
        independent_soc=4,
        independent_makespan=2,
        vertex_conflict_count=0,
        edge_conflict_count=0,
    )
    return MAPFBenchmarkManifest(
        map_name="bench.map",
        scenario_name="bench.map.scen",
        seed=2026,
        max_timestep=8,
        min_reference_length=1.0,
        instances=(instance,),
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
        assert record.termination_reason == "success"
        assert record.conflict_count == 0
        assert record.soc is not None
        assert record.makespan is not None
        assert record.runtime_s >= 0.0

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
    assert loaded.map_name == report.map_name
    assert len(loaded.runs) == len(report.runs)
    assert loaded.runs[0].instance_id == report.runs[0].instance_id
    assert loaded.runs[0].algorithm == report.runs[0].algorithm
    assert loaded.runs[0].soc == report.runs[0].soc

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert isinstance(payload["runs"], list)
    output_path.unlink(missing_ok=True)


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
    assert "instance_id" in lines[0]
    assert "cbs_expanded_ct_nodes" in lines[0]


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
