from __future__ import annotations

import csv
import json
import os
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pathfinding.src.algorithms.mapf.conflicts import detect_conflicts
from pathfinding.src.algorithms.mapf.metrics import makespan, sum_of_costs
from pathfinding.src.algorithms.mapf.models import MAPFResult, MAPFScenario
from pathfinding.src.algorithms.mapf.priority_ordering import (
    ConflictAwareOrderingInputs,
    build_conflict_aware_ordering_inputs,
    cdf_h_sort_key,
    cdf_l_sort_key,
    reorder_scenario_from_conflict_aware_inputs,
    spf_cd_sort_key,
)
from pathfinding.src.algorithms.mapf.prioritized_planning import plan_prioritized_with_stats
from pathfinding.src.core.models import GridMap, Scenario
from pathfinding.src.experiments.mapf_benchmark_execution import (
    MAPFPPSearchMetrics,
    TERMINATION_FAILURE,
    TERMINATION_SUCCESS,
    _pp_search_metrics_from_stats,
    filter_benchmark_instances,
)
from pathfinding.src.experiments.mapf_benchmark_instances import (
    CATALOGUE_ROLE_HELD_OUT,
    DEFAULT_HELD_OUT_SEED,
    MAPFBenchmarkInstance,
    MAPFBenchmarkManifest,
    MAPFInteractionLevel,
    reconstruct_mapf_scenario_from_instance,
    validate_held_out_catalogue,
)

TERMINATION_ORDERING_FAILURE = "ordering_failure"


class ConflictAwareStrategy(str, Enum):
    CDF_H = "cdf_h"
    CDF_L = "cdf_l"
    SPF_CD = "spf_cd"


DEFAULT_CONFLICT_AWARE_STRATEGIES: tuple[ConflictAwareStrategy, ...] = (
    ConflictAwareStrategy.CDF_H,
    ConflictAwareStrategy.CDF_L,
    ConflictAwareStrategy.SPF_CD,
)

MAIN_CONFLICT_AWARE_RESULTS_DIR = Path(
    "pathfinding/results/mapf_conflict_aware_priority_execution"
)
MAIN_CONFLICT_AWARE_CSV = MAIN_CONFLICT_AWARE_RESULTS_DIR / "results.csv"
MAIN_CONFLICT_AWARE_JSONL = MAIN_CONFLICT_AWARE_RESULTS_DIR / "results_details.jsonl"
MAIN_CONFLICT_AWARE_LOG = MAIN_CONFLICT_AWARE_RESULTS_DIR / "run.log"

HELD_OUT_CONFLICT_AWARE_RESULTS_DIR = Path(
    "pathfinding/results/mapf_conflict_aware_priority_heldout_execution"
)
HELD_OUT_CONFLICT_AWARE_CSV = HELD_OUT_CONFLICT_AWARE_RESULTS_DIR / "results.csv"
HELD_OUT_CONFLICT_AWARE_JSONL = (
    HELD_OUT_CONFLICT_AWARE_RESULTS_DIR / "results_details.jsonl"
)
HELD_OUT_CONFLICT_AWARE_LOG = HELD_OUT_CONFLICT_AWARE_RESULTS_DIR / "run.log"
HELD_OUT_MANIFEST_PATH = Path(
    "pathfinding/results/mapf_benchmarks/AR0204SR_heldout_manifest.json"
)
PRIMARY_MANIFEST_PATH = Path(
    "pathfinding/results/mapf_benchmarks/AR0204SR_manifest.json"
)


@dataclass(frozen=True, slots=True)
class ConflictAwarePriorityRunKey:
    instance_id: str
    strategy: ConflictAwareStrategy


@dataclass(frozen=True, slots=True)
class ConflictAwarePriorityRunPlanEntry:
    instance: MAPFBenchmarkInstance
    strategy: ConflictAwareStrategy


@dataclass(frozen=True, slots=True)
class ConflictAwarePriorityRunRecord:
    instance_id: str
    agent_count: int
    interaction_level: str
    strategy: ConflictAwareStrategy

    agent_order: tuple[int, ...] | None
    original_index_order: tuple[int, ...] | None
    agent_degrees: tuple[int, ...] | None
    independent_conflict_count: int | None
    independent_conflict_pair_count: int | None
    ordering_low_level_searches: int | None
    incident_conflict_counts: tuple[int, ...] | None

    success: bool
    termination_reason: str

    ordering_time_ms: float
    pp_time_ms: float | None
    total_time_ms: float

    soc: int | None
    makespan: int | None
    conflict_count: int | None

    pp_search_metrics: MAPFPPSearchMetrics | None = None
    error_message: str | None = None
    catalogue_role: str | None = None
    catalogue_seed: int | None = None


def run_key(record: ConflictAwarePriorityRunRecord) -> ConflictAwarePriorityRunKey:
    return ConflictAwarePriorityRunKey(
        instance_id=record.instance_id,
        strategy=record.strategy,
    )


def original_index_order(
    original_scenario: MAPFScenario,
    reordered_scenario: MAPFScenario,
) -> tuple[int, ...]:
    return tuple(
        original_scenario.agents.index(agent) for agent in reordered_scenario.agents
    )


def validate_held_out_execution_manifest(
    manifest: MAPFBenchmarkManifest,
    *,
    primary_manifest: MAPFBenchmarkManifest | None = None,
    expected_seed: int = DEFAULT_HELD_OUT_SEED,
    agent_counts: Sequence[int] = (5, 10, 20),
    instances_per_level: int = 3,
) -> None:
    """Validate frozen held-out manifest before MAPF-7 execution."""
    if manifest.catalogue_role != CATALOGUE_ROLE_HELD_OUT:
        raise ValueError(
            f"Expected catalogue_role={CATALOGUE_ROLE_HELD_OUT!r}, "
            f"found {manifest.catalogue_role!r}"
        )
    if manifest.seed != expected_seed:
        raise ValueError(
            f"Expected held-out seed {expected_seed}, found {manifest.seed}"
        )

    expected_total = len(agent_counts) * len(MAPFInteractionLevel) * instances_per_level
    if len(manifest.instances) != expected_total:
        raise ValueError(
            f"Expected {expected_total} held-out instances, found {len(manifest.instances)}"
        )

    for instance in manifest.instances:
        if "_HO_" not in instance.instance_id:
            raise ValueError(
                f"Held-out instance ID missing _HO_ marker: {instance.instance_id}"
            )

    counts_by_cell: dict[tuple[int, MAPFInteractionLevel], int] = {}
    for instance in manifest.instances:
        key = (instance.agent_count, instance.interaction_level)
        counts_by_cell[key] = counts_by_cell.get(key, 0) + 1

    for agent_count in agent_counts:
        for level in MAPFInteractionLevel:
            count = counts_by_cell.get((agent_count, level), 0)
            if count != instances_per_level:
                raise ValueError(
                    f"Expected {instances_per_level} instances for "
                    f"agent_count={agent_count}, interaction={level.value}; "
                    f"found {count}"
                )

    if primary_manifest is not None:
        validate_held_out_catalogue(
            manifest,
            primary_manifest,
            agent_counts=agent_counts,
            instances_per_level=instances_per_level,
        )


def build_conflict_aware_priority_run_plan(
    manifest: MAPFBenchmarkManifest,
    *,
    strategies: Sequence[ConflictAwareStrategy] = DEFAULT_CONFLICT_AWARE_STRATEGIES,
    agent_counts: Sequence[int] | None = None,
    interaction_levels: Sequence[str] | None = None,
    instance_ids: Sequence[str] | None = None,
) -> tuple[ConflictAwarePriorityRunPlanEntry, ...]:
    if not strategies:
        raise ValueError("strategies must not be empty")

    instances = filter_benchmark_instances(
        manifest.instances,
        agent_counts=agent_counts,
        interaction_levels=interaction_levels,
    )

    if instance_ids is not None:
        allowed_ids = set(instance_ids)
        instances = tuple(
            instance for instance in instances if instance.instance_id in allowed_ids
        )

    if not instances:
        raise ValueError("no benchmark instances matched the selection filters")

    plan: list[ConflictAwarePriorityRunPlanEntry] = []
    for instance in instances:
        for strategy in strategies:
            plan.append(
                ConflictAwarePriorityRunPlanEntry(instance=instance, strategy=strategy)
            )

    return tuple(plan)


def _ordering_sort_key(
    strategy: ConflictAwareStrategy,
) -> Callable[[int, ConflictAwareOrderingInputs], tuple[int, ...]]:
    if strategy == ConflictAwareStrategy.CDF_H:
        return cdf_h_sort_key
    if strategy == ConflictAwareStrategy.CDF_L:
        return cdf_l_sort_key
    if strategy == ConflictAwareStrategy.SPF_CD:
        return spf_cd_sort_key
    raise ValueError(f"unsupported strategy: {strategy}")


def _ordering_diagnostics_from_inputs(
    inputs: ConflictAwareOrderingInputs,
) -> dict[str, object]:
    return {
        "agent_degrees": inputs.degrees,
        "independent_conflict_count": len(inputs.conflicts),
        "independent_conflict_pair_count": inputs.conflict_pair_count,
        "ordering_low_level_searches": len(inputs.paths),
        "incident_conflict_counts": inputs.incident_counts,
    }


def _validate_successful_solution(result: MAPFResult) -> None:
    if not result.success:
        raise RuntimeError("expected successful MAPF result")

    conflicts = detect_conflicts(result.paths)
    if conflicts:
        raise RuntimeError(
            f"successful MAPF result still has {len(conflicts)} conflicts"
        )


def execute_conflict_aware_priority_run(
    *,
    grid_map: GridMap,
    scenario: MAPFScenario,
    instance: MAPFBenchmarkInstance,
    strategy: ConflictAwareStrategy,
    max_timestep: int,
    catalogue_role: str | None = None,
    catalogue_seed: int | None = None,
) -> ConflictAwarePriorityRunRecord:
    if max_timestep < 0:
        raise ValueError("max_timestep must be non-negative")

    sort_key = _ordering_sort_key(strategy)

    ordering_start = time.perf_counter()
    try:
        inputs = build_conflict_aware_ordering_inputs(
            grid_map=grid_map,
            scenario=scenario,
            max_timestep=max_timestep,
        )
        reordered_scenario = reorder_scenario_from_conflict_aware_inputs(
            scenario,
            inputs,
            sort_key=sort_key,
        )
    except ValueError as error:
        ordering_time_ms = (time.perf_counter() - ordering_start) * 1000.0
        return ConflictAwarePriorityRunRecord(
            instance_id=instance.instance_id,
            agent_count=instance.agent_count,
            interaction_level=instance.interaction_level.value,
            strategy=strategy,
            agent_order=None,
            original_index_order=None,
            agent_degrees=None,
            independent_conflict_count=None,
            independent_conflict_pair_count=None,
            ordering_low_level_searches=None,
            incident_conflict_counts=None,
            success=False,
            termination_reason=TERMINATION_ORDERING_FAILURE,
            ordering_time_ms=ordering_time_ms,
            pp_time_ms=None,
            total_time_ms=ordering_time_ms,
            soc=None,
            makespan=None,
            conflict_count=None,
            pp_search_metrics=None,
            error_message=str(error),
            catalogue_role=catalogue_role,
            catalogue_seed=catalogue_seed,
        )

    ordering_time_ms = (time.perf_counter() - ordering_start) * 1000.0
    diagnostics = _ordering_diagnostics_from_inputs(inputs)
    agent_order = tuple(agent.agent_id for agent in reordered_scenario.agents)
    index_order = original_index_order(scenario, reordered_scenario)

    pp_start = time.perf_counter()
    run = plan_prioritized_with_stats(
        grid_map=grid_map,
        scenario=reordered_scenario,
        max_timestep=max_timestep,
    )
    pp_time_ms = (time.perf_counter() - pp_start) * 1000.0
    total_time_ms = ordering_time_ms + pp_time_ms
    pp_metrics = _pp_search_metrics_from_stats(run.stats)

    base_fields = {
        "instance_id": instance.instance_id,
        "agent_count": instance.agent_count,
        "interaction_level": instance.interaction_level.value,
        "strategy": strategy,
        "agent_order": agent_order,
        "original_index_order": index_order,
        "agent_degrees": diagnostics["agent_degrees"],
        "independent_conflict_count": diagnostics["independent_conflict_count"],
        "independent_conflict_pair_count": diagnostics["independent_conflict_pair_count"],
        "ordering_low_level_searches": diagnostics["ordering_low_level_searches"],
        "incident_conflict_counts": diagnostics["incident_conflict_counts"],
        "ordering_time_ms": ordering_time_ms,
        "pp_time_ms": pp_time_ms,
        "total_time_ms": total_time_ms,
        "pp_search_metrics": pp_metrics,
        "catalogue_role": catalogue_role,
        "catalogue_seed": catalogue_seed,
    }

    if not run.result.success:
        return ConflictAwarePriorityRunRecord(
            **base_fields,
            success=False,
            termination_reason=run.termination_reason,
            soc=None,
            makespan=None,
            conflict_count=None,
        )

    _validate_successful_solution(run.result)
    soc = sum_of_costs(run.result.paths)
    solution_makespan = makespan(run.result.paths)

    return ConflictAwarePriorityRunRecord(
        **base_fields,
        success=True,
        termination_reason=TERMINATION_SUCCESS,
        soc=soc,
        makespan=solution_makespan,
        conflict_count=0,
    )


def execute_conflict_aware_priority_plan_entry(
    *,
    grid_map: GridMap,
    scenarios: Sequence[Scenario],
    entry: ConflictAwarePriorityRunPlanEntry,
    max_timestep: int,
    catalogue_role: str | None = None,
    catalogue_seed: int | None = None,
) -> ConflictAwarePriorityRunRecord:
    scenario = reconstruct_mapf_scenario_from_instance(
        scenarios=scenarios,
        grid_map=grid_map,
        instance=entry.instance,
    )
    return execute_conflict_aware_priority_run(
        grid_map=grid_map,
        scenario=scenario.scenario,
        instance=entry.instance,
        strategy=entry.strategy,
        max_timestep=max_timestep,
        catalogue_role=catalogue_role,
        catalogue_seed=catalogue_seed,
    )


def _pp_metrics_to_json(metrics: MAPFPPSearchMetrics | None) -> dict[str, int] | None:
    if metrics is None:
        return None
    return {
        "low_level_searches": metrics.low_level_searches,
        "agents_planned": metrics.agents_planned,
    }


def _int_tuple_to_json(values: tuple[int, ...] | None) -> list[int] | None:
    if values is None:
        return None
    return list(values)


def _run_record_to_json(record: ConflictAwarePriorityRunRecord) -> dict[str, object]:
    payload: dict[str, object] = {
        "instance_id": record.instance_id,
        "agent_count": record.agent_count,
        "interaction_level": record.interaction_level,
        "strategy": record.strategy.value,
        "agent_order": _int_tuple_to_json(record.agent_order),
        "original_index_order": _int_tuple_to_json(record.original_index_order),
        "agent_degrees": _int_tuple_to_json(record.agent_degrees),
        "independent_conflict_count": record.independent_conflict_count,
        "independent_conflict_pair_count": record.independent_conflict_pair_count,
        "ordering_low_level_searches": record.ordering_low_level_searches,
        "incident_conflict_counts": _int_tuple_to_json(record.incident_conflict_counts),
        "success": record.success,
        "termination_reason": record.termination_reason,
        "ordering_time_ms": record.ordering_time_ms,
        "pp_time_ms": record.pp_time_ms,
        "total_time_ms": record.total_time_ms,
        "soc": record.soc,
        "makespan": record.makespan,
        "conflict_count": record.conflict_count,
        "pp_search_metrics": _pp_metrics_to_json(record.pp_search_metrics),
    }
    if record.error_message is not None:
        payload["error_message"] = record.error_message
    if record.catalogue_role is not None:
        payload["catalogue_role"] = record.catalogue_role
    if record.catalogue_seed is not None:
        payload["catalogue_seed"] = record.catalogue_seed
    return payload


def _parse_int_tuple(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("expected list for integer tuple field")
    return tuple(int(item) for item in value)


def _run_record_from_json(data: dict[str, object]) -> ConflictAwarePriorityRunRecord:
    pp_data = data.get("pp_search_metrics")
    pp_metrics: MAPFPPSearchMetrics | None = None
    if isinstance(pp_data, dict):
        pp_metrics = MAPFPPSearchMetrics(
            low_level_searches=int(pp_data["low_level_searches"]),
            agents_planned=int(pp_data["agents_planned"]),
        )

    soc_value = data.get("soc")
    makespan_value = data.get("makespan")
    conflict_count_value = data.get("conflict_count")
    pp_time_value = data.get("pp_time_ms")
    error_message = data.get("error_message")
    independent_conflict_count = data.get("independent_conflict_count")
    independent_conflict_pair_count = data.get("independent_conflict_pair_count")
    ordering_low_level_searches = data.get("ordering_low_level_searches")
    catalogue_role = data.get("catalogue_role")
    catalogue_seed = data.get("catalogue_seed")

    return ConflictAwarePriorityRunRecord(
        instance_id=str(data["instance_id"]),
        agent_count=int(data["agent_count"]),
        interaction_level=str(data["interaction_level"]),
        strategy=ConflictAwareStrategy(str(data["strategy"])),
        agent_order=_parse_int_tuple(data.get("agent_order")),
        original_index_order=_parse_int_tuple(data.get("original_index_order")),
        agent_degrees=_parse_int_tuple(data.get("agent_degrees")),
        independent_conflict_count=(
            None
            if independent_conflict_count is None
            else int(independent_conflict_count)
        ),
        independent_conflict_pair_count=(
            None
            if independent_conflict_pair_count is None
            else int(independent_conflict_pair_count)
        ),
        ordering_low_level_searches=(
            None
            if ordering_low_level_searches is None
            else int(ordering_low_level_searches)
        ),
        incident_conflict_counts=_parse_int_tuple(data.get("incident_conflict_counts")),
        success=bool(data["success"]),
        termination_reason=str(data["termination_reason"]),
        ordering_time_ms=float(data["ordering_time_ms"]),
        pp_time_ms=None if pp_time_value is None else float(pp_time_value),
        total_time_ms=float(data["total_time_ms"]),
        soc=None if soc_value is None else int(soc_value),
        makespan=None if makespan_value is None else int(makespan_value),
        conflict_count=(
            None if conflict_count_value is None else int(conflict_count_value)
        ),
        pp_search_metrics=pp_metrics,
        error_message=None if error_message is None else str(error_message),
        catalogue_role=None if catalogue_role is None else str(catalogue_role),
        catalogue_seed=None if catalogue_seed is None else int(catalogue_seed),
    )


def load_checkpoint_records(
    jsonl_path: Path,
) -> dict[ConflictAwarePriorityRunKey, ConflictAwarePriorityRunRecord]:
    if not jsonl_path.is_file():
        return {}

    records: dict[ConflictAwarePriorityRunKey, ConflictAwarePriorityRunRecord] = {}
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
                    f"{key.instance_id} / {key.strategy.value} (line {line_number})"
                )
            records[key] = record

    if duplicate_keys:
        joined = "\n  - ".join(duplicate_keys)
        raise RuntimeError(
            "Duplicate conflict-aware priority run keys found in checkpoint file "
            f"{jsonl_path}:\n  - {joined}"
        )

    return records


def _flatten_run_record(record: ConflictAwarePriorityRunRecord) -> dict[str, object]:
    row: dict[str, object] = {
        "instance_id": record.instance_id,
        "agent_count": record.agent_count,
        "interaction_level": record.interaction_level,
        "strategy": record.strategy.value,
        "agent_order": (
            None if record.agent_order is None else json.dumps(list(record.agent_order))
        ),
        "original_index_order": (
            None
            if record.original_index_order is None
            else json.dumps(list(record.original_index_order))
        ),
        "agent_degrees": (
            None if record.agent_degrees is None else json.dumps(list(record.agent_degrees))
        ),
        "independent_conflict_count": record.independent_conflict_count,
        "independent_conflict_pair_count": record.independent_conflict_pair_count,
        "ordering_low_level_searches": record.ordering_low_level_searches,
        "incident_conflict_counts": (
            None
            if record.incident_conflict_counts is None
            else json.dumps(list(record.incident_conflict_counts))
        ),
        "success": record.success,
        "termination_reason": record.termination_reason,
        "ordering_time_ms": record.ordering_time_ms,
        "pp_time_ms": record.pp_time_ms,
        "total_time_ms": record.total_time_ms,
        "soc": record.soc,
        "makespan": record.makespan,
        "conflict_count": record.conflict_count,
        "error_message": record.error_message,
        "catalogue_role": record.catalogue_role,
        "catalogue_seed": record.catalogue_seed,
    }

    if record.pp_search_metrics is not None:
        row["pp_low_level_searches"] = record.pp_search_metrics.low_level_searches
        row["pp_agents_planned"] = record.pp_search_metrics.agents_planned

    return row


def rewrite_checkpoint_csv(
    records: Iterable[ConflictAwarePriorityRunRecord],
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


class ConflictAwarePriorityCheckpointStore:
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
    def records(self) -> dict[ConflictAwarePriorityRunKey, ConflictAwarePriorityRunRecord]:
        return dict(self._records)

    def completed_keys(self) -> set[ConflictAwarePriorityRunKey]:
        return set(self._records)

    def append_record(self, record: ConflictAwarePriorityRunRecord) -> None:
        key = run_key(record)
        if key in self._records:
            raise RuntimeError(
                f"Refusing to overwrite existing checkpoint for "
                f"{key.instance_id} / {key.strategy.value}"
            )

        line = json.dumps(_run_record_to_json(record), sort_keys=True)
        with self.jsonl_path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
            file.flush()
            os.fsync(file.fileno())

        self._records[key] = record
        rewrite_checkpoint_csv(self._ordered_records(), self.csv_path)

    def _ordered_records(self) -> tuple[ConflictAwarePriorityRunRecord, ...]:
        strategy_order = {
            ConflictAwareStrategy.CDF_H: 0,
            ConflictAwareStrategy.CDF_L: 1,
            ConflictAwareStrategy.SPF_CD: 2,
        }
        return tuple(
            sorted(
                self._records.values(),
                key=lambda record: (
                    record.agent_count,
                    record.interaction_level,
                    record.instance_id,
                    strategy_order[record.strategy],
                ),
            )
        )


def validate_conflict_aware_priority_run_record(
    record: ConflictAwarePriorityRunRecord,
) -> None:
    allowed_terminations = {
        TERMINATION_SUCCESS,
        TERMINATION_FAILURE,
        TERMINATION_ORDERING_FAILURE,
    }
    if record.termination_reason not in allowed_terminations:
        raise RuntimeError(
            f"unexpected termination_reason: {record.termination_reason}"
        )

    if record.ordering_time_ms < 0:
        raise RuntimeError("ordering_time_ms must be non-negative")
    if record.total_time_ms < 0:
        raise RuntimeError("total_time_ms must be non-negative")
    if record.pp_time_ms is not None and record.pp_time_ms < 0:
        raise RuntimeError("pp_time_ms must be non-negative")

    if record.termination_reason == TERMINATION_ORDERING_FAILURE:
        if record.pp_time_ms is not None:
            raise RuntimeError("ordering failure must not include pp_time_ms")
        if record.total_time_ms != record.ordering_time_ms:
            raise RuntimeError(
                "ordering failure total_time_ms must equal ordering_time_ms"
            )
        if record.agent_order is not None:
            raise RuntimeError("ordering failure must not include agent_order")
        if record.original_index_order is not None:
            raise RuntimeError(
                "ordering failure must not include original_index_order"
            )
        if record.agent_degrees is not None:
            raise RuntimeError("ordering failure must not include agent_degrees")
        if record.independent_conflict_count is not None:
            raise RuntimeError(
                "ordering failure must not include independent_conflict_count"
            )
        if record.ordering_low_level_searches is not None:
            raise RuntimeError(
                "ordering failure must not include ordering_low_level_searches"
            )
        if record.pp_search_metrics is not None:
            raise RuntimeError("ordering failure must not include PP metrics")
        if (
            record.soc is not None
            or record.makespan is not None
            or record.conflict_count is not None
        ):
            raise RuntimeError(
                "ordering failure must not include solution quality metrics"
            )
        return

    if record.pp_time_ms is None:
        raise RuntimeError("PP phase must include pp_time_ms")
    expected_total = record.ordering_time_ms + record.pp_time_ms
    if abs(record.total_time_ms - expected_total) > 1e-6:
        raise RuntimeError(
            f"total_time_ms must equal ordering_time_ms + pp_time_ms "
            f"({record.total_time_ms} != {expected_total})"
        )
    if record.agent_order is None:
        raise RuntimeError("completed PP run must include agent_order")
    if record.original_index_order is None:
        raise RuntimeError("completed PP run must include original_index_order")
    if record.agent_degrees is None:
        raise RuntimeError("completed PP run must include agent_degrees")
    if record.independent_conflict_count is None:
        raise RuntimeError("completed PP run must include independent_conflict_count")
    if record.independent_conflict_pair_count is None:
        raise RuntimeError(
            "completed PP run must include independent_conflict_pair_count"
        )
    if record.ordering_low_level_searches is None:
        raise RuntimeError("completed PP run must include ordering_low_level_searches")
    if record.incident_conflict_counts is None:
        raise RuntimeError("completed PP run must include incident_conflict_counts")
    if record.pp_search_metrics is None:
        raise RuntimeError("completed PP run must include PP search metrics")

    if record.success:
        if record.termination_reason != TERMINATION_SUCCESS:
            raise RuntimeError("successful run must have termination_reason=success")
        if (
            record.soc is None
            or record.makespan is None
            or record.conflict_count is None
        ):
            raise RuntimeError("successful run missing quality metrics")
        if record.conflict_count != 0:
            raise RuntimeError("successful run reports remaining conflicts")
    elif (
        record.soc is not None
        or record.makespan is not None
        or record.conflict_count is not None
    ):
        raise RuntimeError("non-successful PP run must not include quality metrics")
