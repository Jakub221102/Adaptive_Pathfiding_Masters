from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pathfinding.src.algorithms.mapf.cbs import (
    CBSRunResult,
    CBSStats,
    solve_cbs_cardinal_first_with_stats,
    solve_cbs_with_stats,
)
from pathfinding.src.algorithms.mapf.conflicts import detect_conflicts
from pathfinding.src.algorithms.mapf.metrics import makespan, sum_of_costs
from pathfinding.src.algorithms.mapf.models import MAPFResult, MAPFScenario
from pathfinding.src.algorithms.mapf.prioritized_planning import (
    PrioritizedPlanningRunResult,
    PrioritizedPlanningStats,
    plan_prioritized_with_stats,
)
from pathfinding.src.core.models import GridMap, Scenario
from pathfinding.src.experiments.mapf_benchmark_instances import (
    MAPFBenchmarkInstance,
    MAPFBenchmarkManifest,
    load_benchmark_manifest,
    reconstruct_mapf_scenario_from_instance,
)


class MAPFBenchmarkAlgorithm(str, Enum):
    FIXED_PRIORITY_PP = "fixed_priority_pp"
    CBS_BASIC = "cbs_basic"
    CBS_CARDINAL_FIRST = "cbs_cardinal_first"


DEFAULT_BENCHMARK_ALGORITHMS: tuple[MAPFBenchmarkAlgorithm, ...] = (
    MAPFBenchmarkAlgorithm.FIXED_PRIORITY_PP,
    MAPFBenchmarkAlgorithm.CBS_BASIC,
    MAPFBenchmarkAlgorithm.CBS_CARDINAL_FIRST,
)


@dataclass(frozen=True, slots=True)
class MAPFPPSearchMetrics:
    low_level_searches: int
    agents_planned: int


@dataclass(frozen=True, slots=True)
class MAPFCBSSearchMetrics:
    expanded_ct_nodes: int
    generated_ct_nodes: int
    low_level_replans: int
    max_open_size: int
    unique_constraint_signatures: int
    duplicate_constraint_signatures: int
    unique_path_signatures: int
    duplicate_path_signatures: int
    classified_conflicts: int
    classification_low_level_searches: int
    selected_cardinal_conflicts: int
    selected_semi_cardinal_conflicts: int
    selected_non_cardinal_conflicts: int
    combined_low_level_searches: int


@dataclass(frozen=True, slots=True)
class MAPFBenchmarkRunRecord:
    instance_id: str
    algorithm: MAPFBenchmarkAlgorithm
    agent_count: int
    interaction_level: str

    success: bool
    termination_reason: str
    runtime_s: float

    soc: int | None
    makespan: int | None
    conflict_count: int | None

    independent_soc: int
    independent_makespan: int
    independent_conflict_count: int

    pp_search_metrics: MAPFPPSearchMetrics | None = None
    cbs_search_metrics: MAPFCBSSearchMetrics | None = None


@dataclass(frozen=True, slots=True)
class MAPFBenchmarkExecutionReport:
    manifest_path: str
    map_name: str
    scenario_name: str
    max_timestep: int
    algorithms: tuple[MAPFBenchmarkAlgorithm, ...]
    executed_at_utc: str
    total_runtime_s: float
    runs: tuple[MAPFBenchmarkRunRecord, ...]


def _combined_low_level_searches(stats: CBSStats) -> int:
    return stats.low_level_replans + stats.classification_low_level_searches


def _cbs_search_metrics_from_stats(stats: CBSStats) -> MAPFCBSSearchMetrics:
    return MAPFCBSSearchMetrics(
        expanded_ct_nodes=stats.expanded_ct_nodes,
        generated_ct_nodes=stats.generated_ct_nodes,
        low_level_replans=stats.low_level_replans,
        max_open_size=stats.max_open_size,
        unique_constraint_signatures=stats.unique_constraint_signatures,
        duplicate_constraint_signatures=stats.duplicate_constraint_signatures,
        unique_path_signatures=stats.unique_path_signatures,
        duplicate_path_signatures=stats.duplicate_path_signatures,
        classified_conflicts=stats.classified_conflicts,
        classification_low_level_searches=stats.classification_low_level_searches,
        selected_cardinal_conflicts=stats.selected_cardinal_conflicts,
        selected_semi_cardinal_conflicts=stats.selected_semi_cardinal_conflicts,
        selected_non_cardinal_conflicts=stats.selected_non_cardinal_conflicts,
        combined_low_level_searches=_combined_low_level_searches(stats),
    )


def _pp_search_metrics_from_stats(stats: PrioritizedPlanningStats) -> MAPFPPSearchMetrics:
    return MAPFPPSearchMetrics(
        low_level_searches=stats.low_level_searches,
        agents_planned=stats.agents_planned,
    )


def _validate_successful_solution(result: MAPFResult) -> None:
    if not result.success:
        raise RuntimeError("expected successful MAPF result")

    conflicts = detect_conflicts(result.paths)
    if conflicts:
        raise RuntimeError(
            f"successful MAPF result still has {len(conflicts)} conflicts"
        )


def _quality_from_result(result: MAPFResult) -> tuple[int, int, int]:
    _validate_successful_solution(result)
    return (
        sum_of_costs(result.paths),
        makespan(result.paths),
        0,
    )


def _run_fixed_priority_pp(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
) -> tuple[MAPFResult | None, str, MAPFPPSearchMetrics]:
    run = plan_prioritized_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
    )
    return (
        run.result if run.result.success else None,
        run.termination_reason,
        _pp_search_metrics_from_stats(run.stats),
    )


def _run_cbs_basic(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
) -> tuple[MAPFResult | None, str, MAPFCBSSearchMetrics]:
    run = solve_cbs_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
    )
    return _cbs_run_to_tuple(run)


def _run_cbs_cardinal_first(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
) -> tuple[MAPFResult | None, str, MAPFCBSSearchMetrics]:
    run = solve_cbs_cardinal_first_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
    )
    return _cbs_run_to_tuple(run)


def _cbs_run_to_tuple(
    run: CBSRunResult,
) -> tuple[MAPFResult | None, str, MAPFCBSSearchMetrics]:
    result = run.result
    if result is not None and result.success:
        return (
            result,
            run.termination_reason,
            _cbs_search_metrics_from_stats(run.stats),
        )

    return (
        None,
        run.termination_reason,
        _cbs_search_metrics_from_stats(run.stats),
    )


def execute_benchmark_run(
    *,
    grid_map: GridMap,
    scenario: MAPFScenario,
    instance: MAPFBenchmarkInstance,
    algorithm: MAPFBenchmarkAlgorithm,
    max_timestep: int,
) -> MAPFBenchmarkRunRecord:
    if max_timestep < 0:
        raise ValueError("max_timestep must be non-negative")

    start = time.perf_counter()

    pp_metrics: MAPFPPSearchMetrics | None = None
    cbs_metrics: MAPFCBSSearchMetrics | None = None
    result: MAPFResult | None = None
    termination_reason: str

    if algorithm == MAPFBenchmarkAlgorithm.FIXED_PRIORITY_PP:
        result, termination_reason, pp_metrics = _run_fixed_priority_pp(
            grid_map=grid_map,
            scenario=scenario,
            max_timestep=max_timestep,
        )
    elif algorithm == MAPFBenchmarkAlgorithm.CBS_BASIC:
        result, termination_reason, cbs_metrics = _run_cbs_basic(
            grid_map=grid_map,
            scenario=scenario,
            max_timestep=max_timestep,
        )
    elif algorithm == MAPFBenchmarkAlgorithm.CBS_CARDINAL_FIRST:
        result, termination_reason, cbs_metrics = _run_cbs_cardinal_first(
            grid_map=grid_map,
            scenario=scenario,
            max_timestep=max_timestep,
        )
    else:
        raise ValueError(f"unsupported benchmark algorithm: {algorithm}")

    runtime_s = time.perf_counter() - start

    if result is None:
        return MAPFBenchmarkRunRecord(
            instance_id=instance.instance_id,
            algorithm=algorithm,
            agent_count=instance.agent_count,
            interaction_level=instance.interaction_level.value,
            success=False,
            termination_reason=termination_reason,
            runtime_s=runtime_s,
            soc=None,
            makespan=None,
            conflict_count=None,
            independent_soc=instance.independent_soc,
            independent_makespan=instance.independent_makespan,
            independent_conflict_count=instance.independent_conflict_count,
            pp_search_metrics=pp_metrics,
            cbs_search_metrics=cbs_metrics,
        )

    soc, solution_makespan, conflict_count = _quality_from_result(result)

    return MAPFBenchmarkRunRecord(
        instance_id=instance.instance_id,
        algorithm=algorithm,
        agent_count=instance.agent_count,
        interaction_level=instance.interaction_level.value,
        success=True,
        termination_reason=termination_reason,
        runtime_s=runtime_s,
        soc=soc,
        makespan=solution_makespan,
        conflict_count=conflict_count,
        independent_soc=instance.independent_soc,
        independent_makespan=instance.independent_makespan,
        independent_conflict_count=instance.independent_conflict_count,
        pp_search_metrics=pp_metrics,
        cbs_search_metrics=cbs_metrics,
    )


def execute_benchmark_suite(
    *,
    grid_map: GridMap,
    scenarios: Sequence[Scenario],
    manifest: MAPFBenchmarkManifest,
    manifest_path: Path,
    algorithms: Sequence[MAPFBenchmarkAlgorithm] = DEFAULT_BENCHMARK_ALGORITHMS,
    progress_callback: Callable[[MAPFBenchmarkRunRecord, int, int], None] | None = None,
) -> MAPFBenchmarkExecutionReport:
    if not algorithms:
        raise ValueError("algorithms must not be empty")
    if not manifest.instances:
        raise ValueError("manifest must contain at least one instance")

    suite_start = time.perf_counter()
    runs: list[MAPFBenchmarkRunRecord] = []
    total_runs = len(manifest.instances) * len(algorithms)
    completed = 0

    for instance in manifest.instances:
        scenario = reconstruct_mapf_scenario_from_instance(
            scenarios=scenarios,
            grid_map=grid_map,
            instance=instance,
        )
        for algorithm in algorithms:
            record = execute_benchmark_run(
                grid_map=grid_map,
                scenario=scenario,
                instance=instance,
                algorithm=algorithm,
                max_timestep=manifest.max_timestep,
            )
            runs.append(record)
            completed += 1
            if progress_callback is not None:
                progress_callback(record, completed, total_runs)

    return MAPFBenchmarkExecutionReport(
        manifest_path=str(manifest_path),
        map_name=manifest.map_name,
        scenario_name=manifest.scenario_name,
        max_timestep=manifest.max_timestep,
        algorithms=tuple(algorithms),
        executed_at_utc=datetime.now(UTC).replace(microsecond=0).isoformat(),
        total_runtime_s=time.perf_counter() - suite_start,
        runs=tuple(runs),
    )


def execute_benchmark_manifest_file(
    *,
    manifest_path: Path,
    map_path: Path,
    scen_path: Path,
    algorithms: Sequence[MAPFBenchmarkAlgorithm] = DEFAULT_BENCHMARK_ALGORITHMS,
    progress_callback: Callable[[MAPFBenchmarkRunRecord, int, int], None] | None = None,
) -> MAPFBenchmarkExecutionReport:
    from pathfinding.src.loaders.map_loader import load_moving_ai_map
    from pathfinding.src.loaders.scen_loader import load_moving_ai_scenarios

    manifest = load_benchmark_manifest(manifest_path)
    grid_map = load_moving_ai_map(map_path)
    scenarios = load_moving_ai_scenarios(scen_path)

    if grid_map.name != manifest.map_name:
        raise ValueError(
            f"map name mismatch: manifest expects {manifest.map_name!r}, "
            f"loaded {grid_map.name!r}"
        )

    return execute_benchmark_suite(
        grid_map=grid_map,
        scenarios=scenarios,
        manifest=manifest,
        manifest_path=manifest_path,
        algorithms=algorithms,
        progress_callback=progress_callback,
    )


def _pp_metrics_to_json(metrics: MAPFPPSearchMetrics | None) -> dict[str, int] | None:
    if metrics is None:
        return None
    return {
        "low_level_searches": metrics.low_level_searches,
        "agents_planned": metrics.agents_planned,
    }


def _cbs_metrics_to_json(metrics: MAPFCBSSearchMetrics | None) -> dict[str, int] | None:
    if metrics is None:
        return None
    return {
        "expanded_ct_nodes": metrics.expanded_ct_nodes,
        "generated_ct_nodes": metrics.generated_ct_nodes,
        "low_level_replans": metrics.low_level_replans,
        "max_open_size": metrics.max_open_size,
        "unique_constraint_signatures": metrics.unique_constraint_signatures,
        "duplicate_constraint_signatures": metrics.duplicate_constraint_signatures,
        "unique_path_signatures": metrics.unique_path_signatures,
        "duplicate_path_signatures": metrics.duplicate_path_signatures,
        "classified_conflicts": metrics.classified_conflicts,
        "classification_low_level_searches": metrics.classification_low_level_searches,
        "selected_cardinal_conflicts": metrics.selected_cardinal_conflicts,
        "selected_semi_cardinal_conflicts": metrics.selected_semi_cardinal_conflicts,
        "selected_non_cardinal_conflicts": metrics.selected_non_cardinal_conflicts,
        "combined_low_level_searches": metrics.combined_low_level_searches,
    }


def _run_record_to_json(record: MAPFBenchmarkRunRecord) -> dict[str, object]:
    return {
        "instance_id": record.instance_id,
        "algorithm": record.algorithm.value,
        "agent_count": record.agent_count,
        "interaction_level": record.interaction_level,
        "success": record.success,
        "termination_reason": record.termination_reason,
        "runtime_s": record.runtime_s,
        "soc": record.soc,
        "makespan": record.makespan,
        "conflict_count": record.conflict_count,
        "independent_soc": record.independent_soc,
        "independent_makespan": record.independent_makespan,
        "independent_conflict_count": record.independent_conflict_count,
        "pp_search_metrics": _pp_metrics_to_json(record.pp_search_metrics),
        "cbs_search_metrics": _cbs_metrics_to_json(record.cbs_search_metrics),
    }


def _run_record_from_json(data: dict[str, object]) -> MAPFBenchmarkRunRecord:
    pp_data = data.get("pp_search_metrics")
    cbs_data = data.get("cbs_search_metrics")

    pp_metrics: MAPFPPSearchMetrics | None = None
    if isinstance(pp_data, dict):
        pp_metrics = MAPFPPSearchMetrics(
            low_level_searches=int(pp_data["low_level_searches"]),
            agents_planned=int(pp_data["agents_planned"]),
        )

    cbs_metrics: MAPFCBSSearchMetrics | None = None
    if isinstance(cbs_data, dict):
        cbs_metrics = MAPFCBSSearchMetrics(
            expanded_ct_nodes=int(cbs_data["expanded_ct_nodes"]),
            generated_ct_nodes=int(cbs_data["generated_ct_nodes"]),
            low_level_replans=int(cbs_data["low_level_replans"]),
            max_open_size=int(cbs_data["max_open_size"]),
            unique_constraint_signatures=int(cbs_data["unique_constraint_signatures"]),
            duplicate_constraint_signatures=int(cbs_data["duplicate_constraint_signatures"]),
            unique_path_signatures=int(cbs_data["unique_path_signatures"]),
            duplicate_path_signatures=int(cbs_data["duplicate_path_signatures"]),
            classified_conflicts=int(cbs_data["classified_conflicts"]),
            classification_low_level_searches=int(
                cbs_data["classification_low_level_searches"]
            ),
            selected_cardinal_conflicts=int(cbs_data["selected_cardinal_conflicts"]),
            selected_semi_cardinal_conflicts=int(cbs_data["selected_semi_cardinal_conflicts"]),
            selected_non_cardinal_conflicts=int(cbs_data["selected_non_cardinal_conflicts"]),
            combined_low_level_searches=int(cbs_data["combined_low_level_searches"]),
        )

    soc_value = data.get("soc")
    makespan_value = data.get("makespan")
    conflict_count_value = data.get("conflict_count")

    return MAPFBenchmarkRunRecord(
        instance_id=str(data["instance_id"]),
        algorithm=MAPFBenchmarkAlgorithm(str(data["algorithm"])),
        agent_count=int(data["agent_count"]),
        interaction_level=str(data["interaction_level"]),
        success=bool(data["success"]),
        termination_reason=str(data["termination_reason"]),
        runtime_s=float(data["runtime_s"]),
        soc=None if soc_value is None else int(soc_value),
        makespan=None if makespan_value is None else int(makespan_value),
        conflict_count=(
            None if conflict_count_value is None else int(conflict_count_value)
        ),
        independent_soc=int(data["independent_soc"]),
        independent_makespan=int(data["independent_makespan"]),
        independent_conflict_count=int(data["independent_conflict_count"]),
        pp_search_metrics=pp_metrics,
        cbs_search_metrics=cbs_metrics,
    )


def _report_to_json(report: MAPFBenchmarkExecutionReport) -> dict[str, object]:
    return {
        "manifest_path": report.manifest_path,
        "map_name": report.map_name,
        "scenario_name": report.scenario_name,
        "max_timestep": report.max_timestep,
        "algorithms": [algorithm.value for algorithm in report.algorithms],
        "executed_at_utc": report.executed_at_utc,
        "total_runtime_s": report.total_runtime_s,
        "runs": [_run_record_to_json(record) for record in report.runs],
    }


def _report_from_json(data: dict[str, object]) -> MAPFBenchmarkExecutionReport:
    algorithms_data = data["algorithms"]
    if not isinstance(algorithms_data, list):
        raise ValueError("report algorithms must be a list")

    runs_data = data["runs"]
    if not isinstance(runs_data, list):
        raise ValueError("report runs must be a list")

    return MAPFBenchmarkExecutionReport(
        manifest_path=str(data["manifest_path"]),
        map_name=str(data["map_name"]),
        scenario_name=str(data["scenario_name"]),
        max_timestep=int(data["max_timestep"]),
        algorithms=tuple(
            MAPFBenchmarkAlgorithm(str(algorithm))
            for algorithm in algorithms_data
        ),
        executed_at_utc=str(data["executed_at_utc"]),
        total_runtime_s=float(data["total_runtime_s"]),
        runs=tuple(
            _run_record_from_json(run_data)
            for run_data in runs_data
            if isinstance(run_data, dict)
        ),
    )


def save_execution_report(report: MAPFBenchmarkExecutionReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _report_to_json(report)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_execution_report(path: Path) -> MAPFBenchmarkExecutionReport:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid benchmark execution report format: {path}")
    return _report_from_json(data)


def _flatten_run_record(record: MAPFBenchmarkRunRecord) -> dict[str, object]:
    row: dict[str, object] = {
        "instance_id": record.instance_id,
        "algorithm": record.algorithm.value,
        "agent_count": record.agent_count,
        "interaction_level": record.interaction_level,
        "success": record.success,
        "termination_reason": record.termination_reason,
        "runtime_s": record.runtime_s,
        "soc": record.soc,
        "makespan": record.makespan,
        "conflict_count": record.conflict_count,
        "independent_soc": record.independent_soc,
        "independent_makespan": record.independent_makespan,
        "independent_conflict_count": record.independent_conflict_count,
    }

    if record.pp_search_metrics is not None:
        row["pp_low_level_searches"] = record.pp_search_metrics.low_level_searches
        row["pp_agents_planned"] = record.pp_search_metrics.agents_planned

    if record.cbs_search_metrics is not None:
        row["cbs_expanded_ct_nodes"] = record.cbs_search_metrics.expanded_ct_nodes
        row["cbs_generated_ct_nodes"] = record.cbs_search_metrics.generated_ct_nodes
        row["cbs_low_level_replans"] = record.cbs_search_metrics.low_level_replans
        row["cbs_max_open_size"] = record.cbs_search_metrics.max_open_size
        row["cbs_unique_constraint_signatures"] = (
            record.cbs_search_metrics.unique_constraint_signatures
        )
        row["cbs_duplicate_constraint_signatures"] = (
            record.cbs_search_metrics.duplicate_constraint_signatures
        )
        row["cbs_unique_path_signatures"] = (
            record.cbs_search_metrics.unique_path_signatures
        )
        row["cbs_duplicate_path_signatures"] = (
            record.cbs_search_metrics.duplicate_path_signatures
        )
        row["cbs_classified_conflicts"] = record.cbs_search_metrics.classified_conflicts
        row["cbs_classification_low_level_searches"] = (
            record.cbs_search_metrics.classification_low_level_searches
        )
        row["cbs_selected_cardinal_conflicts"] = (
            record.cbs_search_metrics.selected_cardinal_conflicts
        )
        row["cbs_selected_semi_cardinal_conflicts"] = (
            record.cbs_search_metrics.selected_semi_cardinal_conflicts
        )
        row["cbs_selected_non_cardinal_conflicts"] = (
            record.cbs_search_metrics.selected_non_cardinal_conflicts
        )
        row["cbs_combined_low_level_searches"] = (
            record.cbs_search_metrics.combined_low_level_searches
        )

    return row


def export_execution_report_csv(
    report: MAPFBenchmarkExecutionReport,
    path: Path,
) -> None:
    if not report.runs:
        return

    rows = [_flatten_run_record(record) for record in report.runs]
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_execution_report(report: MAPFBenchmarkExecutionReport) -> None:
    if not report.runs:
        raise RuntimeError("execution report contains no runs")

    seen_keys: set[tuple[str, MAPFBenchmarkAlgorithm]] = set()
    for record in report.runs:
        key = (record.instance_id, record.algorithm)
        if key in seen_keys:
            raise RuntimeError(
                f"duplicate execution record for {record.instance_id} / "
                f"{record.algorithm.value}"
            )
        seen_keys.add(key)

        if record.success:
            if record.soc is None or record.makespan is None or record.conflict_count is None:
                raise RuntimeError(
                    f"successful run {record.instance_id} / "
                    f"{record.algorithm.value} missing quality metrics"
                )
            if record.conflict_count != 0:
                raise RuntimeError(
                    f"successful run {record.instance_id} / "
                    f"{record.algorithm.value} reports conflicts"
                )
        else:
            if (
                record.soc is not None
                or record.makespan is not None
                or record.conflict_count is not None
            ):
                raise RuntimeError(
                    f"failed run {record.instance_id} / "
                    f"{record.algorithm.value} must not include quality metrics"
                )

        if record.algorithm == MAPFBenchmarkAlgorithm.FIXED_PRIORITY_PP:
            if record.pp_search_metrics is None:
                raise RuntimeError(
                    f"PP run {record.instance_id} missing search metrics"
                )
            if record.cbs_search_metrics is not None:
                raise RuntimeError(
                    f"PP run {record.instance_id} must not include CBS metrics"
                )
        else:
            if record.cbs_search_metrics is None:
                raise RuntimeError(
                    f"CBS run {record.instance_id} / {record.algorithm.value} "
                    f"missing search metrics"
                )
            if record.pp_search_metrics is not None:
                raise RuntimeError(
                    f"CBS run {record.instance_id} / {record.algorithm.value} "
                    f"must not include PP metrics"
                )
