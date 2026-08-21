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
    longest_path_first_order,
    shortest_path_first_order,
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
    MAPFBenchmarkInstance,
    MAPFBenchmarkManifest,
    reconstruct_mapf_scenario_from_instance,
)

TERMINATION_ORDERING_FAILURE = "ordering_failure"


class PriorityStrategy(str, Enum):
    SPF = "spf"
    LPF = "lpf"


DEFAULT_PRIORITY_STRATEGIES: tuple[PriorityStrategy, ...] = (
    PriorityStrategy.SPF,
    PriorityStrategy.LPF,
)

MAIN_PRIORITY_STRATEGY_RESULTS_DIR = Path(
    "pathfinding/results/mapf_priority_strategy_execution"
)
MAIN_PRIORITY_STRATEGY_CSV = MAIN_PRIORITY_STRATEGY_RESULTS_DIR / "results.csv"
MAIN_PRIORITY_STRATEGY_JSONL = (
    MAIN_PRIORITY_STRATEGY_RESULTS_DIR / "results_details.jsonl"
)

HELD_OUT_SPF_RESULTS_DIR = Path("pathfinding/results/mapf_spf_heldout_execution")
HELD_OUT_SPF_CSV = HELD_OUT_SPF_RESULTS_DIR / "results.csv"
HELD_OUT_SPF_JSONL = HELD_OUT_SPF_RESULTS_DIR / "results_details.jsonl"
HELD_OUT_SPF_LOG = HELD_OUT_SPF_RESULTS_DIR / "run.log"
HELD_OUT_MANIFEST_PATH = Path(
    "pathfinding/results/mapf_benchmarks/AR0204SR_heldout_manifest.json"
)


@dataclass(frozen=True, slots=True)
class PriorityStrategyRunKey:
    instance_id: str
    strategy: PriorityStrategy


@dataclass(frozen=True, slots=True)
class PriorityStrategyRunPlanEntry:
    instance: MAPFBenchmarkInstance
    strategy: PriorityStrategy


@dataclass(frozen=True, slots=True)
class PriorityStrategyRunRecord:
    instance_id: str
    agent_count: int
    interaction_level: str
    strategy: PriorityStrategy
    agent_order: tuple[int, ...] | None

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


def run_key(record: PriorityStrategyRunRecord) -> PriorityStrategyRunKey:
    return PriorityStrategyRunKey(
        instance_id=record.instance_id,
        strategy=record.strategy,
    )


def build_priority_strategy_run_plan(
    manifest: MAPFBenchmarkManifest,
    *,
    strategies: Sequence[PriorityStrategy] = DEFAULT_PRIORITY_STRATEGIES,
    agent_counts: Sequence[int] | None = None,
    interaction_levels: Sequence[str] | None = None,
    instance_ids: Sequence[str] | None = None,
) -> tuple[PriorityStrategyRunPlanEntry, ...]:
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

    plan: list[PriorityStrategyRunPlanEntry] = []
    for instance in instances:
        for strategy in strategies:
            plan.append(
                PriorityStrategyRunPlanEntry(instance=instance, strategy=strategy)
            )

    return tuple(plan)


def _ordering_function(
    strategy: PriorityStrategy,
) -> Callable[[GridMap, MAPFScenario, int], MAPFScenario]:
    if strategy == PriorityStrategy.SPF:
        return shortest_path_first_order
    if strategy == PriorityStrategy.LPF:
        return longest_path_first_order
    raise ValueError(f"unsupported strategy: {strategy}")


def _validate_successful_solution(result: MAPFResult) -> None:
    if not result.success:
        raise RuntimeError("expected successful MAPF result")

    conflicts = detect_conflicts(result.paths)
    if conflicts:
        raise RuntimeError(
            f"successful MAPF result still has {len(conflicts)} conflicts"
        )


def execute_priority_strategy_run(
    *,
    grid_map: GridMap,
    scenario: MAPFScenario,
    instance: MAPFBenchmarkInstance,
    strategy: PriorityStrategy,
    max_timestep: int,
    catalogue_role: str | None = None,
    catalogue_seed: int | None = None,
) -> PriorityStrategyRunRecord:
    if max_timestep < 0:
        raise ValueError("max_timestep must be non-negative")

    order_agents = _ordering_function(strategy)

    ordering_start = time.perf_counter()
    try:
        reordered_scenario = order_agents(
            grid_map=grid_map,
            scenario=scenario,
            max_timestep=max_timestep,
        )
    except ValueError as error:
        ordering_time_ms = (time.perf_counter() - ordering_start) * 1000.0
        return PriorityStrategyRunRecord(
            instance_id=instance.instance_id,
            agent_count=instance.agent_count,
            interaction_level=instance.interaction_level.value,
            strategy=strategy,
            agent_order=None,
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
    agent_order = tuple(agent.agent_id for agent in reordered_scenario.agents)

    pp_start = time.perf_counter()
    run = plan_prioritized_with_stats(
        grid_map=grid_map,
        scenario=reordered_scenario,
        max_timestep=max_timestep,
    )
    pp_time_ms = (time.perf_counter() - pp_start) * 1000.0
    total_time_ms = ordering_time_ms + pp_time_ms
    pp_metrics = _pp_search_metrics_from_stats(run.stats)

    if not run.result.success:
        return PriorityStrategyRunRecord(
            instance_id=instance.instance_id,
            agent_count=instance.agent_count,
            interaction_level=instance.interaction_level.value,
            strategy=strategy,
            agent_order=agent_order,
            success=False,
            termination_reason=run.termination_reason,
            ordering_time_ms=ordering_time_ms,
            pp_time_ms=pp_time_ms,
            total_time_ms=total_time_ms,
            soc=None,
            makespan=None,
            conflict_count=None,
            pp_search_metrics=pp_metrics,
            catalogue_role=catalogue_role,
            catalogue_seed=catalogue_seed,
        )

    _validate_successful_solution(run.result)
    soc = sum_of_costs(run.result.paths)
    solution_makespan = makespan(run.result.paths)

    return PriorityStrategyRunRecord(
        instance_id=instance.instance_id,
        agent_count=instance.agent_count,
        interaction_level=instance.interaction_level.value,
        strategy=strategy,
        agent_order=agent_order,
        success=True,
        termination_reason=TERMINATION_SUCCESS,
        ordering_time_ms=ordering_time_ms,
        pp_time_ms=pp_time_ms,
        total_time_ms=total_time_ms,
        soc=soc,
        makespan=solution_makespan,
        conflict_count=0,
        pp_search_metrics=pp_metrics,
        catalogue_role=catalogue_role,
        catalogue_seed=catalogue_seed,
    )


def execute_priority_strategy_plan_entry(
    *,
    grid_map: GridMap,
    scenarios: Sequence[Scenario],
    entry: PriorityStrategyRunPlanEntry,
    max_timestep: int,
    catalogue_role: str | None = None,
    catalogue_seed: int | None = None,
) -> PriorityStrategyRunRecord:
    scenario = reconstruct_mapf_scenario_from_instance(
        scenarios=scenarios,
        grid_map=grid_map,
        instance=entry.instance,
    )
    return execute_priority_strategy_run(
        grid_map=grid_map,
        scenario=scenario,
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


def _run_record_to_json(record: PriorityStrategyRunRecord) -> dict[str, object]:
    payload: dict[str, object] = {
        "instance_id": record.instance_id,
        "agent_count": record.agent_count,
        "interaction_level": record.interaction_level,
        "strategy": record.strategy.value,
        "agent_order": None if record.agent_order is None else list(record.agent_order),
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


def _run_record_from_json(data: dict[str, object]) -> PriorityStrategyRunRecord:
    pp_data = data.get("pp_search_metrics")
    pp_metrics: MAPFPPSearchMetrics | None = None
    if isinstance(pp_data, dict):
        pp_metrics = MAPFPPSearchMetrics(
            low_level_searches=int(pp_data["low_level_searches"]),
            agents_planned=int(pp_data["agents_planned"]),
        )

    agent_order_data = data.get("agent_order")
    agent_order: tuple[int, ...] | None = None
    if isinstance(agent_order_data, list):
        agent_order = tuple(int(agent_id) for agent_id in agent_order_data)

    soc_value = data.get("soc")
    makespan_value = data.get("makespan")
    conflict_count_value = data.get("conflict_count")
    pp_time_value = data.get("pp_time_ms")
    error_message = data.get("error_message")
    catalogue_role = data.get("catalogue_role")
    catalogue_seed = data.get("catalogue_seed")

    return PriorityStrategyRunRecord(
        instance_id=str(data["instance_id"]),
        agent_count=int(data["agent_count"]),
        interaction_level=str(data["interaction_level"]),
        strategy=PriorityStrategy(str(data["strategy"])),
        agent_order=agent_order,
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
) -> dict[PriorityStrategyRunKey, PriorityStrategyRunRecord]:
    if not jsonl_path.is_file():
        return {}

    records: dict[PriorityStrategyRunKey, PriorityStrategyRunRecord] = {}
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
            "Duplicate priority-strategy run keys found in checkpoint file "
            f"{jsonl_path}:\n  - {joined}"
        )

    return records


def _flatten_run_record(record: PriorityStrategyRunRecord) -> dict[str, object]:
    row: dict[str, object] = {
        "instance_id": record.instance_id,
        "agent_count": record.agent_count,
        "interaction_level": record.interaction_level,
        "strategy": record.strategy.value,
        "agent_order": (
            None if record.agent_order is None else json.dumps(list(record.agent_order))
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
    records: Iterable[PriorityStrategyRunRecord],
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


class PriorityStrategyCheckpointStore:
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
    def records(self) -> dict[PriorityStrategyRunKey, PriorityStrategyRunRecord]:
        return dict(self._records)

    def completed_keys(self) -> set[PriorityStrategyRunKey]:
        return set(self._records)

    def append_record(self, record: PriorityStrategyRunRecord) -> None:
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

    def _ordered_records(self) -> tuple[PriorityStrategyRunRecord, ...]:
        return tuple(
            sorted(
                self._records.values(),
                key=lambda record: (
                    record.agent_count,
                    record.interaction_level,
                    record.instance_id,
                    0 if record.strategy == PriorityStrategy.SPF else 1,
                ),
            )
        )


def validate_priority_strategy_run_record(record: PriorityStrategyRunRecord) -> None:
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
