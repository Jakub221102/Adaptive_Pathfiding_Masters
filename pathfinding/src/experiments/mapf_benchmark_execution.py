from __future__ import annotations

import csv
import json
import os
import time
from collections.abc import Callable, Iterable, Sequence
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
    PrioritizedPlanningStats,
    plan_prioritized_with_stats,
)
from pathfinding.src.core.models import GridMap, Scenario
from pathfinding.src.experiments.mapf_benchmark_instances import (
    MAPFBenchmarkInstance,
    MAPFBenchmarkManifest,
    MAPFInteractionLevel,
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

MAIN_BENCHMARK_RESULTS_DIR = Path("pathfinding/results/mapf_benchmark_execution")
MAIN_BENCHMARK_CSV = MAIN_BENCHMARK_RESULTS_DIR / "results.csv"
MAIN_BENCHMARK_JSONL = MAIN_BENCHMARK_RESULTS_DIR / "results_details.jsonl"

TERMINATION_SUCCESS = "success"
TERMINATION_FAILURE = "failure"
TERMINATION_EXPANSION_LIMIT = "expansion_limit"
TERMINATION_ERROR = "error"

_INTERACTION_ORDER: dict[MAPFInteractionLevel, int] = {
    MAPFInteractionLevel.LOW: 0,
    MAPFInteractionLevel.MEDIUM: 1,
    MAPFInteractionLevel.HIGH: 2,
}


@dataclass(frozen=True, slots=True)
class MAPFBenchmarkRunKey:
    instance_id: str
    algorithm: MAPFBenchmarkAlgorithm


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
    execution_time_ms: float

    soc: int | None
    makespan: int | None
    remaining_conflicts: int | None

    independent_soc: int
    independent_makespan: int
    independent_conflict_count: int

    pp_search_metrics: MAPFPPSearchMetrics | None = None
    cbs_search_metrics: MAPFCBSSearchMetrics | None = None
    error_message: str | None = None

    @property
    def runtime_s(self) -> float:
        return self.execution_time_ms / 1000.0


@dataclass(frozen=True, slots=True)
class MAPFBenchmarkRunPlanEntry:
    instance: MAPFBenchmarkInstance
    algorithm: MAPFBenchmarkAlgorithm


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


@dataclass(frozen=True, slots=True)
class MAPFBenchmarkCBSLimits:
    basic_max_expanded_nodes: int | None
    cardinal_first_max_expanded_nodes: int | None


def run_key(record: MAPFBenchmarkRunRecord) -> MAPFBenchmarkRunKey:
    return MAPFBenchmarkRunKey(
        instance_id=record.instance_id,
        algorithm=record.algorithm,
    )


def sort_benchmark_instances(
    instances: Sequence[MAPFBenchmarkInstance],
) -> tuple[MAPFBenchmarkInstance, ...]:
    return tuple(
        sorted(
            instances,
            key=lambda instance: (
                instance.agent_count,
                _INTERACTION_ORDER[instance.interaction_level],
                instance.instance_id,
            ),
        )
    )


def filter_benchmark_instances(
    instances: Sequence[MAPFBenchmarkInstance],
    *,
    agent_counts: Sequence[int] | None = None,
    interaction_levels: Sequence[str] | None = None,
) -> tuple[MAPFBenchmarkInstance, ...]:
    filtered = list(instances)

    if agent_counts is not None:
        allowed_counts = set(agent_counts)
        filtered = [
            instance
            for instance in filtered
            if instance.agent_count in allowed_counts
        ]

    if interaction_levels is not None:
        allowed_levels = {level.lower() for level in interaction_levels}
        filtered = [
            instance
            for instance in filtered
            if instance.interaction_level.value in allowed_levels
        ]

    return sort_benchmark_instances(filtered)


def build_benchmark_run_plan(
    manifest: MAPFBenchmarkManifest,
    *,
    algorithms: Sequence[MAPFBenchmarkAlgorithm] = DEFAULT_BENCHMARK_ALGORITHMS,
    agent_counts: Sequence[int] | None = None,
    interaction_levels: Sequence[str] | None = None,
    run_index_filter: Sequence[int] | None = None,
) -> tuple[MAPFBenchmarkRunPlanEntry, ...]:
    """Build deterministic run order: algorithm phases, then sorted instances.

    Phase 1: Fixed-Priority PP on all selected instances
    Phase 2: Basic CBS on all selected instances
    Phase 3: Cardinal-First CBS on all selected instances

    Within each phase, instances are ordered by agent_count, interaction level,
    then instance_id.
    """
    if not algorithms:
        raise ValueError("algorithms must not be empty")

    instances = filter_benchmark_instances(
        manifest.instances,
        agent_counts=agent_counts,
        interaction_levels=interaction_levels,
    )
    if not instances:
        raise ValueError("no benchmark instances matched the selection filters")

    plan: list[MAPFBenchmarkRunPlanEntry] = []
    for algorithm in algorithms:
        for instance in instances:
            plan.append(
                MAPFBenchmarkRunPlanEntry(instance=instance, algorithm=algorithm)
            )

    if run_index_filter is not None:
        selected_indices = set(run_index_filter)
        plan = [entry for index, entry in enumerate(plan) if index in selected_indices]

    if not plan:
        raise ValueError("run plan is empty after applying run_index_filter")

    return tuple(plan)


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
    *,
    max_expanded_nodes: int | None,
) -> tuple[MAPFResult | None, str, MAPFCBSSearchMetrics]:
    run = solve_cbs_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
        max_expanded_nodes=max_expanded_nodes,
    )
    return _cbs_run_to_tuple(run)


def _run_cbs_cardinal_first(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
    *,
    max_expanded_nodes: int | None,
) -> tuple[MAPFResult | None, str, MAPFCBSSearchMetrics]:
    run = solve_cbs_cardinal_first_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
        max_expanded_nodes=max_expanded_nodes,
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


def create_error_benchmark_record(
    *,
    instance: MAPFBenchmarkInstance,
    algorithm: MAPFBenchmarkAlgorithm,
    execution_time_ms: float,
    error_message: str,
) -> MAPFBenchmarkRunRecord:
    return MAPFBenchmarkRunRecord(
        instance_id=instance.instance_id,
        algorithm=algorithm,
        agent_count=instance.agent_count,
        interaction_level=instance.interaction_level.value,
        success=False,
        termination_reason=TERMINATION_ERROR,
        execution_time_ms=execution_time_ms,
        soc=None,
        makespan=None,
        remaining_conflicts=None,
        independent_soc=instance.independent_soc,
        independent_makespan=instance.independent_makespan,
        independent_conflict_count=instance.independent_conflict_count,
        error_message=error_message,
    )


def execute_benchmark_run(
    *,
    grid_map: GridMap,
    scenario: MAPFScenario,
    instance: MAPFBenchmarkInstance,
    algorithm: MAPFBenchmarkAlgorithm,
    max_timestep: int,
    cbs_limits: MAPFBenchmarkCBSLimits | None = None,
) -> MAPFBenchmarkRunRecord:
    if max_timestep < 0:
        raise ValueError("max_timestep must be non-negative")

    limits = cbs_limits or MAPFBenchmarkCBSLimits(
        basic_max_expanded_nodes=None,
        cardinal_first_max_expanded_nodes=None,
    )

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
            max_expanded_nodes=limits.basic_max_expanded_nodes,
        )
    elif algorithm == MAPFBenchmarkAlgorithm.CBS_CARDINAL_FIRST:
        result, termination_reason, cbs_metrics = _run_cbs_cardinal_first(
            grid_map=grid_map,
            scenario=scenario,
            max_timestep=max_timestep,
            max_expanded_nodes=limits.cardinal_first_max_expanded_nodes,
        )
    else:
        raise ValueError(f"unsupported benchmark algorithm: {algorithm}")

    execution_time_ms = (time.perf_counter() - start) * 1000.0

    if result is None:
        return MAPFBenchmarkRunRecord(
            instance_id=instance.instance_id,
            algorithm=algorithm,
            agent_count=instance.agent_count,
            interaction_level=instance.interaction_level.value,
            success=False,
            termination_reason=termination_reason,
            execution_time_ms=execution_time_ms,
            soc=None,
            makespan=None,
            remaining_conflicts=None,
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
        execution_time_ms=execution_time_ms,
        soc=soc,
        makespan=solution_makespan,
        remaining_conflicts=conflict_count,
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
    cbs_limits: MAPFBenchmarkCBSLimits | None = None,
    progress_callback: Callable[[MAPFBenchmarkRunRecord, int, int], None] | None = None,
) -> MAPFBenchmarkExecutionReport:
    if not algorithms:
        raise ValueError("algorithms must not be empty")
    if not manifest.instances:
        raise ValueError("manifest must contain at least one instance")

    plan = build_benchmark_run_plan(manifest, algorithms=algorithms)

    suite_start = time.perf_counter()
    runs: list[MAPFBenchmarkRunRecord] = []
    total_runs = len(plan)

    for completed, entry in enumerate(plan, start=1):
        scenario = reconstruct_mapf_scenario_from_instance(
            scenarios=scenarios,
            grid_map=grid_map,
            instance=entry.instance,
        )
        record = execute_benchmark_run(
            grid_map=grid_map,
            scenario=scenario,
            instance=entry.instance,
            algorithm=entry.algorithm,
            max_timestep=manifest.max_timestep,
            cbs_limits=cbs_limits,
        )
        runs.append(record)
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
    cbs_limits: MAPFBenchmarkCBSLimits | None = None,
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
        cbs_limits=cbs_limits,
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
    payload: dict[str, object] = {
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
        "pp_search_metrics": _pp_metrics_to_json(record.pp_search_metrics),
        "cbs_search_metrics": _cbs_metrics_to_json(record.cbs_search_metrics),
    }
    if record.error_message is not None:
        payload["error_message"] = record.error_message
    return payload


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
    remaining_conflicts_value = data.get("remaining_conflicts")
    if remaining_conflicts_value is None and "conflict_count" in data:
        remaining_conflicts_value = data.get("conflict_count")

    execution_time_ms = data.get("execution_time_ms")
    if execution_time_ms is None and "runtime_s" in data:
        execution_time_ms = float(data["runtime_s"]) * 1000.0

    error_message = data.get("error_message")

    return MAPFBenchmarkRunRecord(
        instance_id=str(data["instance_id"]),
        algorithm=MAPFBenchmarkAlgorithm(str(data["algorithm"])),
        agent_count=int(data["agent_count"]),
        interaction_level=str(data["interaction_level"]),
        success=bool(data["success"]),
        termination_reason=str(data["termination_reason"]),
        execution_time_ms=float(execution_time_ms),
        soc=None if soc_value is None else int(soc_value),
        makespan=None if makespan_value is None else int(makespan_value),
        remaining_conflicts=(
            None
            if remaining_conflicts_value is None
            else int(remaining_conflicts_value)
        ),
        independent_soc=int(data["independent_soc"]),
        independent_makespan=int(data["independent_makespan"]),
        independent_conflict_count=int(data["independent_conflict_count"]),
        pp_search_metrics=pp_metrics,
        cbs_search_metrics=cbs_metrics,
        error_message=None if error_message is None else str(error_message),
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


def load_checkpoint_records(
    jsonl_path: Path,
) -> dict[MAPFBenchmarkRunKey, MAPFBenchmarkRunRecord]:
    if not jsonl_path.is_file():
        return {}

    records: dict[MAPFBenchmarkRunKey, MAPFBenchmarkRunRecord] = {}
    duplicate_keys: list[str] = []

    with jsonl_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            data = json.loads(stripped)
            if not isinstance(data, dict):
                raise ValueError(
                    f"Invalid JSONL record at {jsonl_path}:{line_number}"
                )

            record = _run_record_from_json(data)
            key = run_key(record)
            if key in records:
                duplicate_keys.append(
                    f"{key.instance_id} / {key.algorithm.value} (line {line_number})"
                )
            records[key] = record

    if duplicate_keys:
        joined = "\n  - ".join(duplicate_keys)
        raise RuntimeError(
            "Duplicate benchmark run keys found in checkpoint file "
            f"{jsonl_path}:\n  - {joined}"
        )

    return records


def _flatten_run_record(record: MAPFBenchmarkRunRecord) -> dict[str, object]:
    row: dict[str, object] = {
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
        "error_message": record.error_message,
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


def rewrite_checkpoint_csv(
    records: Iterable[MAPFBenchmarkRunRecord],
    csv_path: Path,
) -> None:
    rows = [_flatten_run_record(record) for record in records]
    if not rows:
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    os.replace(temp_path, csv_path)


def export_execution_report_csv(
    report: MAPFBenchmarkExecutionReport,
    path: Path,
) -> None:
    rewrite_checkpoint_csv(report.runs, path)


class BenchmarkCheckpointStore:
    """Append-only JSONL checkpoint store with atomic CSV rewrite."""

    def __init__(
        self,
        *,
        results_dir: Path,
        csv_path: Path,
        jsonl_path: Path,
        reset_results: bool = False,
    ) -> None:
        self.results_dir = results_dir
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path
        self.results_dir.mkdir(parents=True, exist_ok=True)

        if reset_results:
            self.csv_path.unlink(missing_ok=True)
            self.jsonl_path.unlink(missing_ok=True)

        self._records = load_checkpoint_records(self.jsonl_path)

    @property
    def records(self) -> dict[MAPFBenchmarkRunKey, MAPFBenchmarkRunRecord]:
        return dict(self._records)

    def completed_keys(self) -> set[MAPFBenchmarkRunKey]:
        return set(self._records)

    def append_record(self, record: MAPFBenchmarkRunRecord) -> None:
        key = run_key(record)
        if key in self._records:
            raise RuntimeError(
                f"Refusing to overwrite existing checkpoint for "
                f"{key.instance_id} / {key.algorithm.value}"
            )

        line = json.dumps(_run_record_to_json(record), sort_keys=True)
        with self.jsonl_path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
            file.flush()
            os.fsync(file.fileno())

        self._records[key] = record
        rewrite_checkpoint_csv(self._ordered_records(), self.csv_path)

    def _ordered_records(self) -> tuple[MAPFBenchmarkRunRecord, ...]:
        return tuple(
            sorted(
                self._records.values(),
                key=lambda record: (
                    DEFAULT_BENCHMARK_ALGORITHMS.index(record.algorithm)
                    if record.algorithm in DEFAULT_BENCHMARK_ALGORITHMS
                    else len(DEFAULT_BENCHMARK_ALGORITHMS),
                    record.agent_count,
                    record.interaction_level,
                    record.instance_id,
                ),
            )
        )


def validate_execution_report(report: MAPFBenchmarkExecutionReport) -> None:
    if not report.runs:
        raise RuntimeError("execution report contains no runs")

    seen_keys: set[MAPFBenchmarkRunKey] = set()
    for record in report.runs:
        key = run_key(record)
        if key in seen_keys:
            raise RuntimeError(
                f"duplicate execution record for {record.instance_id} / "
                f"{record.algorithm.value}"
            )
        seen_keys.add(key)
        validate_benchmark_run_record(record)


def validate_benchmark_run_record(record: MAPFBenchmarkRunRecord) -> None:
    allowed_terminations = {
        TERMINATION_SUCCESS,
        TERMINATION_FAILURE,
        TERMINATION_EXPANSION_LIMIT,
        TERMINATION_ERROR,
    }
    if record.termination_reason not in allowed_terminations:
        raise RuntimeError(
            f"unexpected termination_reason: {record.termination_reason}"
        )

    if record.success:
        if record.termination_reason != TERMINATION_SUCCESS:
            raise RuntimeError(
                f"successful run {record.instance_id} / "
                f"{record.algorithm.value} must have termination_reason=success"
            )
        if (
            record.soc is None
            or record.makespan is None
            or record.remaining_conflicts is None
        ):
            raise RuntimeError(
                f"successful run {record.instance_id} / "
                f"{record.algorithm.value} missing quality metrics"
            )
        if record.remaining_conflicts != 0:
            raise RuntimeError(
                f"successful run {record.instance_id} / "
                f"{record.algorithm.value} reports remaining conflicts"
            )
        if record.error_message is not None:
            raise RuntimeError(
                f"successful run {record.instance_id} / "
                f"{record.algorithm.value} must not include error_message"
            )
    else:
        if (
            record.soc is not None
            or record.makespan is not None
            or record.remaining_conflicts is not None
        ):
            raise RuntimeError(
                f"non-successful run {record.instance_id} / "
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
    elif record.termination_reason != TERMINATION_ERROR:
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

    if record.termination_reason == TERMINATION_ERROR and record.error_message is None:
        raise RuntimeError(
            f"error run {record.instance_id} / {record.algorithm.value} "
            f"missing error_message"
        )
