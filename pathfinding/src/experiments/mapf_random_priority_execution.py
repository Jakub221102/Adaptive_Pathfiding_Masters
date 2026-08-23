from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pathfinding.src.algorithms.mapf.conflicts import detect_conflicts
from pathfinding.src.algorithms.mapf.metrics import makespan, sum_of_costs
from pathfinding.src.algorithms.mapf.models import MAPFAgent, MAPFResult, MAPFScenario
from pathfinding.src.algorithms.mapf.priority_ordering import random_priority_order
from pathfinding.src.algorithms.mapf.prioritized_planning import plan_prioritized_with_stats
from pathfinding.src.core.models import GridMap, Position, Scenario
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

DEFAULT_RANDOM_PRIORITY_K = 10
DEFAULT_RANDOM_PRIORITY_BASE_SEED = 2026
SUPPLEMENTAL_ORDERING_INDEX_START = 10
SUPPLEMENTAL_ORDERING_COUNT = 10

MAIN_RANDOM_PRIORITY_RESULTS_DIR = Path(
    "pathfinding/results/mapf_random_priority_pilot"
)
MAIN_RANDOM_PRIORITY_CSV = MAIN_RANDOM_PRIORITY_RESULTS_DIR / "results.csv"
MAIN_RANDOM_PRIORITY_JSONL = MAIN_RANDOM_PRIORITY_RESULTS_DIR / "results_details.jsonl"

SUPPLEMENTAL_RANDOM_PRIORITY_RESULTS_DIR = Path(
    "pathfinding/results/mapf_random_k20_supplemental"
)
SUPPLEMENTAL_RANDOM_PRIORITY_CSV = (
    SUPPLEMENTAL_RANDOM_PRIORITY_RESULTS_DIR / "results.csv"
)
SUPPLEMENTAL_RANDOM_PRIORITY_JSONL = (
    SUPPLEMENTAL_RANDOM_PRIORITY_RESULTS_DIR / "results_details.jsonl"
)


@dataclass(frozen=True, slots=True)
class RandomPriorityRunKey:
    instance_id: str
    ordering_index: int


@dataclass(frozen=True, slots=True)
class RandomPriorityOrderingSpec:
    ordering_index: int
    ordering_seed: int
    agent_order: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RandomPriorityRunPlanEntry:
    instance: MAPFBenchmarkInstance
    ordering: RandomPriorityOrderingSpec


@dataclass(frozen=True, slots=True)
class RandomPriorityRunRecord:
    instance_id: str
    agent_count: int
    interaction_level: str
    ordering_index: int
    ordering_seed: int
    agent_order: tuple[int, ...]
    base_seed: int

    success: bool
    termination_reason: str
    execution_time_ms: float

    soc: int | None
    makespan: int | None
    conflict_count: int | None

    pp_search_metrics: MAPFPPSearchMetrics | None = None


def run_key(record: RandomPriorityRunRecord) -> RandomPriorityRunKey:
    return RandomPriorityRunKey(
        instance_id=record.instance_id,
        ordering_index=record.ordering_index,
    )


def derive_ordering_seed(
    *,
    base_seed: int,
    instance_id: str,
    ordering_index: int,
    attempt: int = 0,
) -> int:
    payload = (
        f"{base_seed}:{instance_id}:{ordering_index}:{attempt}".encode("utf-8")
    )
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def _synthetic_scenario_for_ordering(agent_count: int) -> MAPFScenario:
    agents = tuple(
        MAPFAgent(
            agent_id=agent_id,
            start=Position(row=0, col=0),
            goal=Position(row=0, col=1),
        )
        for agent_id in range(agent_count)
    )
    return MAPFScenario(agents=agents)


def generate_unique_orderings_for_instance(
    *,
    instance: MAPFBenchmarkInstance,
    base_seed: int,
    k: int,
) -> tuple[RandomPriorityOrderingSpec, ...]:
    if k <= 0:
        raise ValueError("k must be positive")

    max_permutations = math.factorial(instance.agent_count)
    if k > max_permutations:
        raise ValueError(
            f"cannot generate {k} unique orderings for "
            f"{instance.agent_count} agents (max {max_permutations})"
        )

    scenario = _synthetic_scenario_for_ordering(instance.agent_count)
    seen_orders: set[tuple[int, ...]] = set()
    orderings: list[RandomPriorityOrderingSpec] = []

    for ordering_index in range(k):
        attempt = 0
        while True:
            ordering_seed = derive_ordering_seed(
                base_seed=base_seed,
                instance_id=instance.instance_id,
                ordering_index=ordering_index,
                attempt=attempt,
            )
            ordered = random_priority_order(scenario, ordering_seed)
            agent_order = tuple(agent.agent_id for agent in ordered.agents)
            if agent_order not in seen_orders:
                seen_orders.add(agent_order)
                orderings.append(
                    RandomPriorityOrderingSpec(
                        ordering_index=ordering_index,
                        ordering_seed=ordering_seed,
                        agent_order=agent_order,
                    )
                )
                break
            attempt += 1

    return tuple(orderings)


def load_historical_k10_records_by_instance(
    jsonl_path: Path,
) -> dict[str, dict[int, RandomPriorityRunRecord]]:
    """Load K=10 historical records grouped by instance and ordering index."""
    records = load_checkpoint_records(jsonl_path)
    grouped: dict[str, dict[int, RandomPriorityRunRecord]] = {}

    for record in records.values():
        if record.ordering_index >= DEFAULT_RANDOM_PRIORITY_K:
            continue
        instance_records = grouped.setdefault(record.instance_id, {})
        if record.ordering_index in instance_records:
            raise RuntimeError(
                f"Duplicate historical K=10 record for "
                f"{record.instance_id} / ordering {record.ordering_index}"
            )
        instance_records[record.ordering_index] = record

    return grouped


def verify_historical_k10_orderings_match_reconstruction(
    *,
    instance: MAPFBenchmarkInstance,
    base_seed: int,
    historical_records: dict[int, RandomPriorityRunRecord],
) -> tuple[tuple[int, ...], ...]:
    reconstructed = generate_unique_orderings_for_instance(
        instance=instance,
        base_seed=base_seed,
        k=DEFAULT_RANDOM_PRIORITY_K,
    )

    if len(historical_records) != DEFAULT_RANDOM_PRIORITY_K:
        raise RuntimeError(
            f"Historical K=10 records for {instance.instance_id} contain "
            f"{len(historical_records)} orderings, expected "
            f"{DEFAULT_RANDOM_PRIORITY_K}"
        )

    excluded_orders: list[tuple[int, ...]] = []
    for ordering in reconstructed:
        historical = historical_records.get(ordering.ordering_index)
        if historical is None:
            raise RuntimeError(
                f"Missing historical K=10 record for "
                f"{instance.instance_id} / ordering {ordering.ordering_index}"
            )
        if (
            historical.agent_order != ordering.agent_order
            or historical.ordering_seed != ordering.ordering_seed
        ):
            raise RuntimeError(
                f"Historical K=10 ordering mismatch for "
                f"{instance.instance_id} / ordering {ordering.ordering_index}: "
                f"historical agent_order={historical.agent_order}, "
                f"reconstructed agent_order={ordering.agent_order}"
            )
        excluded_orders.append(ordering.agent_order)

    return tuple(excluded_orders)


def generate_supplemental_orderings_for_instance(
    *,
    instance: MAPFBenchmarkInstance,
    base_seed: int,
    excluded_agent_orders: Sequence[tuple[int, ...]],
    ordering_index_start: int = SUPPLEMENTAL_ORDERING_INDEX_START,
    supplemental_count: int = SUPPLEMENTAL_ORDERING_COUNT,
) -> tuple[RandomPriorityOrderingSpec, ...]:
    if supplemental_count <= 0:
        raise ValueError("supplemental_count must be positive")

    max_permutations = math.factorial(instance.agent_count)
    total_required = len(excluded_agent_orders) + supplemental_count
    if total_required > max_permutations:
        raise ValueError(
            f"cannot generate {supplemental_count} supplemental unique orderings "
            f"for {instance.agent_count} agents with "
            f"{len(excluded_agent_orders)} excluded orderings "
            f"(max {max_permutations})"
        )

    scenario = _synthetic_scenario_for_ordering(instance.agent_count)
    seen_orders = set(excluded_agent_orders)
    if len(seen_orders) != len(excluded_agent_orders):
        raise RuntimeError(
            f"excluded_agent_orders for {instance.instance_id} contain duplicates"
        )

    orderings: list[RandomPriorityOrderingSpec] = []
    ordering_index_end = ordering_index_start + supplemental_count

    for ordering_index in range(ordering_index_start, ordering_index_end):
        attempt = 0
        while True:
            ordering_seed = derive_ordering_seed(
                base_seed=base_seed,
                instance_id=instance.instance_id,
                ordering_index=ordering_index,
                attempt=attempt,
            )
            ordered = random_priority_order(scenario, ordering_seed)
            agent_order = tuple(agent.agent_id for agent in ordered.agents)
            if agent_order not in seen_orders:
                seen_orders.add(agent_order)
                orderings.append(
                    RandomPriorityOrderingSpec(
                        ordering_index=ordering_index,
                        ordering_seed=ordering_seed,
                        agent_order=agent_order,
                    )
                )
                break
            attempt += 1

    return tuple(orderings)


def build_supplemental_random_priority_run_plan(
    manifest: MAPFBenchmarkManifest,
    *,
    base_seed: int,
    historical_k10_jsonl_path: Path,
    supplemental_count: int = SUPPLEMENTAL_ORDERING_COUNT,
    ordering_index_start: int = SUPPLEMENTAL_ORDERING_INDEX_START,
    agent_counts: Sequence[int] | None = None,
    interaction_levels: Sequence[str] | None = None,
    instance_ids: Sequence[str] | None = None,
    verify_historical_k10: bool = True,
) -> tuple[RandomPriorityRunPlanEntry, ...]:
    if supplemental_count <= 0:
        raise ValueError("supplemental_count must be positive")

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

    historical_by_instance: dict[str, dict[int, RandomPriorityRunRecord]] = {}
    if verify_historical_k10:
        if not historical_k10_jsonl_path.is_file():
            raise FileNotFoundError(
                f"Historical K=10 checkpoint not found: {historical_k10_jsonl_path}"
            )
        historical_by_instance = load_historical_k10_records_by_instance(
            historical_k10_jsonl_path
        )

    plan: list[RandomPriorityRunPlanEntry] = []
    for instance in instances:
        if verify_historical_k10:
            historical_records = historical_by_instance.get(instance.instance_id, {})
            excluded_orders = verify_historical_k10_orderings_match_reconstruction(
                instance=instance,
                base_seed=base_seed,
                historical_records=historical_records,
            )
        else:
            k10_orderings = generate_unique_orderings_for_instance(
                instance=instance,
                base_seed=base_seed,
                k=DEFAULT_RANDOM_PRIORITY_K,
            )
            excluded_orders = tuple(ordering.agent_order for ordering in k10_orderings)

        orderings = generate_supplemental_orderings_for_instance(
            instance=instance,
            base_seed=base_seed,
            excluded_agent_orders=excluded_orders,
            ordering_index_start=ordering_index_start,
            supplemental_count=supplemental_count,
        )
        for ordering in orderings:
            plan.append(
                RandomPriorityRunPlanEntry(instance=instance, ordering=ordering)
            )

    return tuple(plan)


def build_random_priority_run_plan(
    manifest: MAPFBenchmarkManifest,
    *,
    k: int,
    base_seed: int,
    agent_counts: Sequence[int] | None = None,
    interaction_levels: Sequence[str] | None = None,
    instance_ids: Sequence[str] | None = None,
) -> tuple[RandomPriorityRunPlanEntry, ...]:
    if k <= 0:
        raise ValueError("k must be positive")

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

    plan: list[RandomPriorityRunPlanEntry] = []
    for instance in instances:
        orderings = generate_unique_orderings_for_instance(
            instance=instance,
            base_seed=base_seed,
            k=k,
        )
        for ordering in orderings:
            plan.append(
                RandomPriorityRunPlanEntry(instance=instance, ordering=ordering)
            )

    return tuple(plan)


def _validate_successful_solution(result: MAPFResult) -> None:
    if not result.success:
        raise RuntimeError("expected successful MAPF result")

    conflicts = detect_conflicts(result.paths)
    if conflicts:
        raise RuntimeError(
            f"successful MAPF result still has {len(conflicts)} conflicts"
        )


def execute_random_priority_run(
    *,
    grid_map: GridMap,
    scenario: MAPFScenario,
    instance: MAPFBenchmarkInstance,
    ordering: RandomPriorityOrderingSpec,
    base_seed: int,
    max_timestep: int,
) -> RandomPriorityRunRecord:
    if max_timestep < 0:
        raise ValueError("max_timestep must be non-negative")

    ordered_scenario = random_priority_order(scenario, ordering.ordering_seed)
    actual_order = tuple(agent.agent_id for agent in ordered_scenario.agents)
    if actual_order != ordering.agent_order:
        raise RuntimeError(
            f"ordering mismatch for {instance.instance_id} "
            f"index {ordering.ordering_index}: "
            f"expected {ordering.agent_order}, got {actual_order}"
        )

    start = time.perf_counter()
    run = plan_prioritized_with_stats(
        grid_map=grid_map,
        scenario=ordered_scenario,
        max_timestep=max_timestep,
    )
    execution_time_ms = (time.perf_counter() - start) * 1000.0
    pp_metrics = _pp_search_metrics_from_stats(run.stats)

    if not run.result.success:
        return RandomPriorityRunRecord(
            instance_id=instance.instance_id,
            agent_count=instance.agent_count,
            interaction_level=instance.interaction_level.value,
            ordering_index=ordering.ordering_index,
            ordering_seed=ordering.ordering_seed,
            agent_order=ordering.agent_order,
            base_seed=base_seed,
            success=False,
            termination_reason=run.termination_reason,
            execution_time_ms=execution_time_ms,
            soc=None,
            makespan=None,
            conflict_count=None,
            pp_search_metrics=pp_metrics,
        )

    _validate_successful_solution(run.result)
    soc = sum_of_costs(run.result.paths)
    solution_makespan = makespan(run.result.paths)

    return RandomPriorityRunRecord(
        instance_id=instance.instance_id,
        agent_count=instance.agent_count,
        interaction_level=instance.interaction_level.value,
        ordering_index=ordering.ordering_index,
        ordering_seed=ordering.ordering_seed,
        agent_order=ordering.agent_order,
        base_seed=base_seed,
        success=True,
        termination_reason=TERMINATION_SUCCESS,
        execution_time_ms=execution_time_ms,
        soc=soc,
        makespan=solution_makespan,
        conflict_count=0,
        pp_search_metrics=pp_metrics,
    )


def execute_random_priority_plan_entry(
    *,
    grid_map: GridMap,
    scenarios: Sequence[Scenario],
    entry: RandomPriorityRunPlanEntry,
    base_seed: int,
    max_timestep: int,
) -> RandomPriorityRunRecord:
    scenario = reconstruct_mapf_scenario_from_instance(
        scenarios=scenarios,
        grid_map=grid_map,
        instance=entry.instance,
    )
    return execute_random_priority_run(
        grid_map=grid_map,
        scenario=scenario,
        instance=entry.instance,
        ordering=entry.ordering,
        base_seed=base_seed,
        max_timestep=max_timestep,
    )


def _pp_metrics_to_json(metrics: MAPFPPSearchMetrics | None) -> dict[str, int] | None:
    if metrics is None:
        return None
    return {
        "low_level_searches": metrics.low_level_searches,
        "agents_planned": metrics.agents_planned,
    }


def _run_record_to_json(record: RandomPriorityRunRecord) -> dict[str, object]:
    return {
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
        "pp_search_metrics": _pp_metrics_to_json(record.pp_search_metrics),
    }


def _run_record_from_json(data: dict[str, object]) -> RandomPriorityRunRecord:
    pp_data = data.get("pp_search_metrics")
    pp_metrics: MAPFPPSearchMetrics | None = None
    if isinstance(pp_data, dict):
        pp_metrics = MAPFPPSearchMetrics(
            low_level_searches=int(pp_data["low_level_searches"]),
            agents_planned=int(pp_data["agents_planned"]),
        )

    agent_order_data = data["agent_order"]
    if not isinstance(agent_order_data, list):
        raise ValueError("agent_order must be a list")

    soc_value = data.get("soc")
    makespan_value = data.get("makespan")
    conflict_count_value = data.get("conflict_count")

    return RandomPriorityRunRecord(
        instance_id=str(data["instance_id"]),
        agent_count=int(data["agent_count"]),
        interaction_level=str(data["interaction_level"]),
        ordering_index=int(data["ordering_index"]),
        ordering_seed=int(data["ordering_seed"]),
        agent_order=tuple(int(agent_id) for agent_id in agent_order_data),
        base_seed=int(data["base_seed"]),
        success=bool(data["success"]),
        termination_reason=str(data["termination_reason"]),
        execution_time_ms=float(data["execution_time_ms"]),
        soc=None if soc_value is None else int(soc_value),
        makespan=None if makespan_value is None else int(makespan_value),
        conflict_count=(
            None if conflict_count_value is None else int(conflict_count_value)
        ),
        pp_search_metrics=pp_metrics,
    )


def load_checkpoint_records(
    jsonl_path: Path,
) -> dict[RandomPriorityRunKey, RandomPriorityRunRecord]:
    if not jsonl_path.is_file():
        return {}

    records: dict[RandomPriorityRunKey, RandomPriorityRunRecord] = {}
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
                    f"{key.instance_id} / ordering {key.ordering_index} "
                    f"(line {line_number})"
                )
            records[key] = record

    if duplicate_keys:
        joined = "\n  - ".join(duplicate_keys)
        raise RuntimeError(
            "Duplicate random-priority run keys found in checkpoint file "
            f"{jsonl_path}:\n  - {joined}"
        )

    return records


def _flatten_run_record(record: RandomPriorityRunRecord) -> dict[str, object]:
    row: dict[str, object] = {
        "instance_id": record.instance_id,
        "agent_count": record.agent_count,
        "interaction_level": record.interaction_level,
        "ordering_index": record.ordering_index,
        "ordering_seed": record.ordering_seed,
        "agent_order": json.dumps(list(record.agent_order)),
        "base_seed": record.base_seed,
        "success": record.success,
        "termination_reason": record.termination_reason,
        "execution_time_ms": record.execution_time_ms,
        "soc": record.soc,
        "makespan": record.makespan,
        "conflict_count": record.conflict_count,
    }

    if record.pp_search_metrics is not None:
        row["pp_low_level_searches"] = record.pp_search_metrics.low_level_searches
        row["pp_agents_planned"] = record.pp_search_metrics.agents_planned

    return row


def rewrite_checkpoint_csv(
    records: Iterable[RandomPriorityRunRecord],
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


class RandomPriorityCheckpointStore:
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
    def records(self) -> dict[RandomPriorityRunKey, RandomPriorityRunRecord]:
        return dict(self._records)

    def completed_keys(self) -> set[RandomPriorityRunKey]:
        return set(self._records)

    def append_record(self, record: RandomPriorityRunRecord) -> None:
        key = run_key(record)
        if key in self._records:
            raise RuntimeError(
                f"Refusing to overwrite existing checkpoint for "
                f"{key.instance_id} / ordering {key.ordering_index}"
            )

        line = json.dumps(_run_record_to_json(record), sort_keys=True)
        with self.jsonl_path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
            file.flush()
            os.fsync(file.fileno())

        self._records[key] = record
        rewrite_checkpoint_csv(self._ordered_records(), self.csv_path)

    def _ordered_records(self) -> tuple[RandomPriorityRunRecord, ...]:
        return tuple(
            sorted(
                self._records.values(),
                key=lambda record: (
                    record.agent_count,
                    record.interaction_level,
                    record.instance_id,
                    record.ordering_index,
                ),
            )
        )


def validate_random_priority_run_record(record: RandomPriorityRunRecord) -> None:
    allowed_terminations = {TERMINATION_SUCCESS, TERMINATION_FAILURE}
    if record.termination_reason not in allowed_terminations:
        raise RuntimeError(
            f"unexpected termination_reason: {record.termination_reason}"
        )

    if record.success:
        if record.termination_reason != TERMINATION_SUCCESS:
            raise RuntimeError(
                f"successful run {record.instance_id} / ordering "
                f"{record.ordering_index} must have termination_reason=success"
            )
        if (
            record.soc is None
            or record.makespan is None
            or record.conflict_count is None
        ):
            raise RuntimeError(
                f"successful run {record.instance_id} / ordering "
                f"{record.ordering_index} missing quality metrics"
            )
        if record.conflict_count != 0:
            raise RuntimeError(
                f"successful run {record.instance_id} / ordering "
                f"{record.ordering_index} reports conflicts"
            )
    elif (
        record.soc is not None
        or record.makespan is not None
        or record.conflict_count is not None
    ):
        raise RuntimeError(
            f"non-successful run {record.instance_id} / ordering "
            f"{record.ordering_index} must not include quality metrics"
        )

    if record.pp_search_metrics is None:
        raise RuntimeError(
            f"run {record.instance_id} / ordering {record.ordering_index} "
            f"missing PP search metrics"
        )
