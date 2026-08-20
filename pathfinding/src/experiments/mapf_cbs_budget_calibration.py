from __future__ import annotations

import csv
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pathfinding.src.core.models import GridMap, Scenario
from pathfinding.src.experiments.mapf_benchmark_execution import (
    MAPFBenchmarkAlgorithm,
    MAPFBenchmarkCBSLimits,
    MAPFBenchmarkRunRecord,
    MAPFCBSSearchMetrics,
    TERMINATION_ERROR,
    TERMINATION_EXPANSION_LIMIT,
    TERMINATION_FAILURE,
    TERMINATION_SUCCESS,
    execute_benchmark_run,
    validate_benchmark_run_record,
)
from pathfinding.src.experiments.mapf_benchmark_instances import (
    MAPFBenchmarkInstance,
    MAPFBenchmarkManifest,
    MAPFInteractionLevel,
    load_benchmark_manifest,
    reconstruct_mapf_scenario_from_instance,
)


DEFAULT_CALIBRATION_INSTANCE_IDS: tuple[str, ...] = (
    "AR0204SR_n05_low_000",
    "AR0204SR_n05_medium_000",
    "AR0204SR_n05_high_000",
    "AR0204SR_n10_low_000",
    "AR0204SR_n10_medium_000",
    "AR0204SR_n10_high_000",
    "AR0204SR_n20_low_000",
    "AR0204SR_n20_medium_000",
    "AR0204SR_n20_high_000",
)

DEFAULT_EXPANSION_BUDGETS: tuple[int, ...] = (10, 25, 50, 100, 250)

DEFAULT_CALIBRATION_ALGORITHMS: tuple[MAPFBenchmarkAlgorithm, ...] = (
    MAPFBenchmarkAlgorithm.CBS_BASIC,
    MAPFBenchmarkAlgorithm.CBS_CARDINAL_FIRST,
)

DEFAULT_CALIBRATION_RESULTS_DIR = Path("pathfinding/results/mapf_cbs_calibration")

CALIBRATION_RESULTS_CSV_NAME = "calibration_results.csv"
CALIBRATION_RESULTS_JSONL_NAME = "calibration_results.jsonl"

_INTERACTION_ORDER: dict[MAPFInteractionLevel, int] = {
    MAPFInteractionLevel.LOW: 0,
    MAPFInteractionLevel.MEDIUM: 1,
    MAPFInteractionLevel.HIGH: 2,
}


@dataclass(frozen=True, slots=True)
class MAPFCBSCalibrationRunKey:
    instance_id: str
    algorithm: MAPFBenchmarkAlgorithm
    max_expanded_nodes: int


@dataclass(frozen=True, slots=True)
class MAPFCBSCalibrationPairKey:
    instance_id: str
    algorithm: MAPFBenchmarkAlgorithm


@dataclass(frozen=True, slots=True)
class MAPFCBSCalibrationRecord:
    instance_id: str
    algorithm: MAPFBenchmarkAlgorithm
    max_expanded_nodes: int
    agent_count: int
    interaction_level: str

    success: bool
    termination_reason: str
    execution_time_ms: float

    soc: int | None
    makespan: int | None

    independent_soc: int
    independent_makespan: int
    independent_conflict_count: int

    cbs_search_metrics: MAPFCBSSearchMetrics
    error_message: str | None = None

    @property
    def runtime_s(self) -> float:
        return self.execution_time_ms / 1000.0


@dataclass(frozen=True, slots=True)
class MAPFCBSCalibrationPlanEntry:
    instance: MAPFBenchmarkInstance
    algorithm: MAPFBenchmarkAlgorithm
    max_expanded_nodes: int


@dataclass(frozen=True, slots=True)
class MAPFCBSCalibrationPairSummary:
    instance_id: str
    algorithm: MAPFBenchmarkAlgorithm
    agent_count: int
    interaction_level: str
    outcome: str
    smallest_successful_budget: int | None
    largest_tested_budget: int | None
    runtime_at_outcome_ms: float | None
    expanded_ct_nodes_at_outcome: int | None
    combined_low_level_searches_at_outcome: int | None


def calibration_run_key(record: MAPFCBSCalibrationRecord) -> MAPFCBSCalibrationRunKey:
    return MAPFCBSCalibrationRunKey(
        instance_id=record.instance_id,
        algorithm=record.algorithm,
        max_expanded_nodes=record.max_expanded_nodes,
    )


def calibration_pair_key(
    instance_id: str,
    algorithm: MAPFBenchmarkAlgorithm,
) -> MAPFCBSCalibrationPairKey:
    return MAPFCBSCalibrationPairKey(
        instance_id=instance_id,
        algorithm=algorithm,
    )


def resolve_calibration_instances(
    manifest: MAPFBenchmarkManifest,
    instance_ids: Sequence[str],
) -> tuple[MAPFBenchmarkInstance, ...]:
    lookup = {instance.instance_id: instance for instance in manifest.instances}
    missing = [instance_id for instance_id in instance_ids if instance_id not in lookup]
    if missing:
        raise ValueError(
            "Calibration instance IDs not found in manifest: "
            + ", ".join(missing)
        )

    return tuple(lookup[instance_id] for instance_id in instance_ids)


def sort_calibration_instances(
    instances: Sequence[MAPFBenchmarkInstance],
    instance_id_order: Sequence[str],
) -> tuple[MAPFBenchmarkInstance, ...]:
    order_index = {instance_id: index for index, instance_id in enumerate(instance_id_order)}
    return tuple(
        sorted(
            instances,
            key=lambda instance: order_index.get(
                instance.instance_id,
                (
                    instance.agent_count,
                    _INTERACTION_ORDER[instance.interaction_level],
                    instance.instance_id,
                ),
            ),
        )
    )


def should_stop_budget_escalation(
    record: MAPFCBSCalibrationRecord,
    *,
    budgets: Sequence[int],
) -> bool:
    if record.termination_reason == TERMINATION_SUCCESS:
        return True
    if record.termination_reason == TERMINATION_FAILURE:
        return True
    if record.termination_reason == TERMINATION_ERROR:
        return True
    if (
        record.termination_reason == TERMINATION_EXPANSION_LIMIT
        and record.max_expanded_nodes == budgets[-1]
    ):
        return True
    return False


def budgets_to_execute_for_pair(
    *,
    budgets: Sequence[int],
    existing_records: Mapping[int, MAPFCBSCalibrationRecord],
) -> tuple[int, ...]:
    """Return budgets that still need execution under progressive escalation."""
    if not budgets:
        raise ValueError("budgets must not be empty")

    to_run: list[int] = []

    for budget in budgets:
        if budget in existing_records:
            record = existing_records[budget]
            if should_stop_budget_escalation(record, budgets=budgets):
                break
            continue

        if budget == budgets[0]:
            to_run.append(budget)
            continue

        previous_budget = budgets[budgets.index(budget) - 1]
        previous = existing_records.get(previous_budget)
        if previous is None:
            break
        if previous.termination_reason != TERMINATION_EXPANSION_LIMIT:
            break

        to_run.append(budget)

    return tuple(to_run)


def is_calibration_pair_complete(
    *,
    budgets: Sequence[int],
    existing_records: Mapping[int, MAPFCBSCalibrationRecord],
) -> bool:
    return (
        len(
            budgets_to_execute_for_pair(
                budgets=budgets,
                existing_records=existing_records,
            )
        )
        == 0
    )


def group_calibration_records_by_pair(
    records: Mapping[MAPFCBSCalibrationRunKey, MAPFCBSCalibrationRecord],
) -> dict[MAPFCBSCalibrationPairKey, dict[int, MAPFCBSCalibrationRecord]]:
    grouped: dict[MAPFCBSCalibrationPairKey, dict[int, MAPFCBSCalibrationRecord]] = {}
    for record in records.values():
        pair = calibration_pair_key(record.instance_id, record.algorithm)
        grouped.setdefault(pair, {})[record.max_expanded_nodes] = record
    return grouped


def build_calibration_run_plan(
    instances: Sequence[MAPFBenchmarkInstance],
    *,
    algorithms: Sequence[MAPFBenchmarkAlgorithm] = DEFAULT_CALIBRATION_ALGORITHMS,
    budgets: Sequence[int] = DEFAULT_EXPANSION_BUDGETS,
    existing_records: Mapping[MAPFCBSCalibrationRunKey, MAPFCBSCalibrationRecord]
    | None = None,
) -> tuple[MAPFCBSCalibrationPlanEntry, ...]:
    """Build deterministic calibration plan with progressive budget escalation.

    Order:
      agent_count 5 → 10 → 20 (via instance_id_order in caller)
      interaction LOW → MEDIUM → HIGH
      algorithm Basic CBS → Cardinal-First CBS
      budget ascending within each pair (only budgets still required)
    """
    if not instances:
        raise ValueError("instances must not be empty")
    if not algorithms:
        raise ValueError("algorithms must not be empty")
    if not budgets:
        raise ValueError("budgets must not be empty")

    records_by_pair = group_calibration_records_by_pair(existing_records or {})

    plan: list[MAPFCBSCalibrationPlanEntry] = []
    for instance in instances:
        for algorithm in algorithms:
            pair = calibration_pair_key(instance.instance_id, algorithm)
            pair_records = records_by_pair.get(pair, {})
            if is_calibration_pair_complete(budgets=budgets, existing_records=pair_records):
                continue

            for budget in budgets_to_execute_for_pair(
                budgets=budgets,
                existing_records=pair_records,
            ):
                plan.append(
                    MAPFCBSCalibrationPlanEntry(
                        instance=instance,
                        algorithm=algorithm,
                        max_expanded_nodes=budget,
                    )
                )

    return tuple(plan)


def calibration_record_from_benchmark(
    record: MAPFBenchmarkRunRecord,
    *,
    max_expanded_nodes: int,
) -> MAPFCBSCalibrationRecord:
    if record.cbs_search_metrics is None:
        raise ValueError("calibration record requires CBS search metrics")

    return MAPFCBSCalibrationRecord(
        instance_id=record.instance_id,
        algorithm=record.algorithm,
        max_expanded_nodes=max_expanded_nodes,
        agent_count=record.agent_count,
        interaction_level=record.interaction_level,
        success=record.success,
        termination_reason=record.termination_reason,
        execution_time_ms=record.execution_time_ms,
        soc=record.soc,
        makespan=record.makespan,
        independent_soc=record.independent_soc,
        independent_makespan=record.independent_makespan,
        independent_conflict_count=record.independent_conflict_count,
        cbs_search_metrics=record.cbs_search_metrics,
        error_message=record.error_message,
    )


def create_error_calibration_record(
    *,
    instance: MAPFBenchmarkInstance,
    algorithm: MAPFBenchmarkAlgorithm,
    max_expanded_nodes: int,
    execution_time_ms: float,
    error_message: str,
) -> MAPFCBSCalibrationRecord:
    return MAPFCBSCalibrationRecord(
        instance_id=instance.instance_id,
        algorithm=algorithm,
        max_expanded_nodes=max_expanded_nodes,
        agent_count=instance.agent_count,
        interaction_level=instance.interaction_level.value,
        success=False,
        termination_reason=TERMINATION_ERROR,
        execution_time_ms=execution_time_ms,
        soc=None,
        makespan=None,
        independent_soc=instance.independent_soc,
        independent_makespan=instance.independent_makespan,
        independent_conflict_count=instance.independent_conflict_count,
        cbs_search_metrics=MAPFCBSSearchMetrics(
            expanded_ct_nodes=0,
            generated_ct_nodes=0,
            low_level_replans=0,
            max_open_size=0,
            unique_constraint_signatures=0,
            duplicate_constraint_signatures=0,
            unique_path_signatures=0,
            duplicate_path_signatures=0,
            classified_conflicts=0,
            classification_low_level_searches=0,
            selected_cardinal_conflicts=0,
            selected_semi_cardinal_conflicts=0,
            selected_non_cardinal_conflicts=0,
            combined_low_level_searches=0,
        ),
        error_message=error_message,
    )


def execute_cbs_calibration_run_from_loaded(
    *,
    grid_map: GridMap,
    scenarios: Sequence[Scenario],
    instance: MAPFBenchmarkInstance,
    algorithm: MAPFBenchmarkAlgorithm,
    max_timestep: int,
    max_expanded_nodes: int,
) -> MAPFCBSCalibrationRecord:
    mapf_scenario = reconstruct_mapf_scenario_from_instance(
        scenarios=scenarios,
        grid_map=grid_map,
        instance=instance,
    )

    limits = MAPFBenchmarkCBSLimits(
        basic_max_expanded_nodes=(
            max_expanded_nodes if algorithm == MAPFBenchmarkAlgorithm.CBS_BASIC else None
        ),
        cardinal_first_max_expanded_nodes=(
            max_expanded_nodes
            if algorithm == MAPFBenchmarkAlgorithm.CBS_CARDINAL_FIRST
            else None
        ),
    )

    record = execute_benchmark_run(
        grid_map=grid_map,
        scenario=mapf_scenario,
        instance=instance,
        algorithm=algorithm,
        max_timestep=max_timestep,
        cbs_limits=limits,
    )
    validate_benchmark_run_record(record)
    return calibration_record_from_benchmark(
        record,
        max_expanded_nodes=max_expanded_nodes,
    )


def _calibration_record_to_json(record: MAPFCBSCalibrationRecord) -> dict[str, object]:
    metrics = record.cbs_search_metrics
    payload: dict[str, object] = {
        "instance_id": record.instance_id,
        "algorithm": record.algorithm.value,
        "max_expanded_nodes": record.max_expanded_nodes,
        "agent_count": record.agent_count,
        "interaction_level": record.interaction_level,
        "success": record.success,
        "termination_reason": record.termination_reason,
        "execution_time_ms": record.execution_time_ms,
        "soc": record.soc,
        "makespan": record.makespan,
        "independent_soc": record.independent_soc,
        "independent_makespan": record.independent_makespan,
        "independent_conflict_count": record.independent_conflict_count,
        "cbs_expanded_ct_nodes": metrics.expanded_ct_nodes,
        "cbs_generated_ct_nodes": metrics.generated_ct_nodes,
        "cbs_low_level_replans": metrics.low_level_replans,
        "cbs_max_open_size": metrics.max_open_size,
        "cbs_classified_conflicts": metrics.classified_conflicts,
        "cbs_classification_low_level_searches": (
            metrics.classification_low_level_searches
        ),
        "cbs_selected_cardinal_conflicts": metrics.selected_cardinal_conflicts,
        "cbs_selected_semi_cardinal_conflicts": metrics.selected_semi_cardinal_conflicts,
        "cbs_selected_non_cardinal_conflicts": metrics.selected_non_cardinal_conflicts,
        "cbs_combined_low_level_searches": metrics.combined_low_level_searches,
    }
    if record.error_message is not None:
        payload["error_message"] = record.error_message
    return payload


def _calibration_record_from_json(data: dict[str, object]) -> MAPFCBSCalibrationRecord:
    metrics = MAPFCBSSearchMetrics(
        expanded_ct_nodes=int(data["cbs_expanded_ct_nodes"]),
        generated_ct_nodes=int(data["cbs_generated_ct_nodes"]),
        low_level_replans=int(data["cbs_low_level_replans"]),
        max_open_size=int(data["cbs_max_open_size"]),
        unique_constraint_signatures=int(data.get("cbs_unique_constraint_signatures", 0)),
        duplicate_constraint_signatures=int(
            data.get("cbs_duplicate_constraint_signatures", 0)
        ),
        unique_path_signatures=int(data.get("cbs_unique_path_signatures", 0)),
        duplicate_path_signatures=int(data.get("cbs_duplicate_path_signatures", 0)),
        classified_conflicts=int(data["cbs_classified_conflicts"]),
        classification_low_level_searches=int(
            data["cbs_classification_low_level_searches"]
        ),
        selected_cardinal_conflicts=int(data["cbs_selected_cardinal_conflicts"]),
        selected_semi_cardinal_conflicts=int(data["cbs_selected_semi_cardinal_conflicts"]),
        selected_non_cardinal_conflicts=int(data["cbs_selected_non_cardinal_conflicts"]),
        combined_low_level_searches=int(data["cbs_combined_low_level_searches"]),
    )

    soc_value = data.get("soc")
    makespan_value = data.get("makespan")
    error_message = data.get("error_message")

    return MAPFCBSCalibrationRecord(
        instance_id=str(data["instance_id"]),
        algorithm=MAPFBenchmarkAlgorithm(str(data["algorithm"])),
        max_expanded_nodes=int(data["max_expanded_nodes"]),
        agent_count=int(data["agent_count"]),
        interaction_level=str(data["interaction_level"]),
        success=bool(data["success"]),
        termination_reason=str(data["termination_reason"]),
        execution_time_ms=float(data["execution_time_ms"]),
        soc=None if soc_value is None else int(soc_value),
        makespan=None if makespan_value is None else int(makespan_value),
        independent_soc=int(data["independent_soc"]),
        independent_makespan=int(data["independent_makespan"]),
        independent_conflict_count=int(data["independent_conflict_count"]),
        cbs_search_metrics=metrics,
        error_message=None if error_message is None else str(error_message),
    )


def load_calibration_records(
    jsonl_path: Path,
) -> dict[MAPFCBSCalibrationRunKey, MAPFCBSCalibrationRecord]:
    if not jsonl_path.is_file():
        return {}

    records: dict[MAPFCBSCalibrationRunKey, MAPFCBSCalibrationRecord] = {}
    duplicate_keys: list[str] = []

    with jsonl_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            data = json.loads(stripped)
            if not isinstance(data, dict):
                raise ValueError(
                    f"Invalid calibration JSONL record at {jsonl_path}:{line_number}"
                )

            record = _calibration_record_from_json(data)
            key = calibration_run_key(record)
            if key in records:
                duplicate_keys.append(
                    f"{key.instance_id} / {key.algorithm.value} / "
                    f"budget={key.max_expanded_nodes} (line {line_number})"
                )
            records[key] = record

    if duplicate_keys:
        joined = "\n  - ".join(duplicate_keys)
        raise RuntimeError(
            "Duplicate calibration run keys found in checkpoint file "
            f"{jsonl_path}:\n  - {joined}"
        )

    return records


def _flatten_calibration_record(record: MAPFCBSCalibrationRecord) -> dict[str, object]:
    metrics = record.cbs_search_metrics
    row: dict[str, object] = {
        "instance_id": record.instance_id,
        "algorithm": record.algorithm.value,
        "max_expanded_nodes": record.max_expanded_nodes,
        "agent_count": record.agent_count,
        "interaction_level": record.interaction_level,
        "success": record.success,
        "termination_reason": record.termination_reason,
        "execution_time_ms": record.execution_time_ms,
        "soc": record.soc,
        "makespan": record.makespan,
        "independent_soc": record.independent_soc,
        "independent_makespan": record.independent_makespan,
        "independent_conflict_count": record.independent_conflict_count,
        "cbs_expanded_ct_nodes": metrics.expanded_ct_nodes,
        "cbs_generated_ct_nodes": metrics.generated_ct_nodes,
        "cbs_low_level_replans": metrics.low_level_replans,
        "cbs_max_open_size": metrics.max_open_size,
        "cbs_classified_conflicts": metrics.classified_conflicts,
        "cbs_classification_low_level_searches": (
            metrics.classification_low_level_searches
        ),
        "cbs_selected_cardinal_conflicts": metrics.selected_cardinal_conflicts,
        "cbs_selected_semi_cardinal_conflicts": metrics.selected_semi_cardinal_conflicts,
        "cbs_selected_non_cardinal_conflicts": metrics.selected_non_cardinal_conflicts,
        "cbs_combined_low_level_searches": metrics.combined_low_level_searches,
        "error_message": record.error_message,
    }
    return row


def rewrite_calibration_csv(
    records: Iterable[MAPFCBSCalibrationRecord],
    csv_path: Path,
) -> None:
    rows = [_flatten_calibration_record(record) for record in records]
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


class CalibrationCheckpointStore:
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

        self._records = load_calibration_records(self.jsonl_path)

    @property
    def records(self) -> dict[MAPFCBSCalibrationRunKey, MAPFCBSCalibrationRecord]:
        return dict(self._records)

    def completed_keys(self) -> set[MAPFCBSCalibrationRunKey]:
        return set(self._records)

    def append_record(self, record: MAPFCBSCalibrationRecord) -> None:
        key = calibration_run_key(record)
        if key in self._records:
            raise RuntimeError(
                f"Refusing to overwrite existing calibration checkpoint for "
                f"{key.instance_id} / {key.algorithm.value} / budget={key.max_expanded_nodes}"
            )

        line = json.dumps(_calibration_record_to_json(record), sort_keys=True)
        with self.jsonl_path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
            file.flush()
            os.fsync(file.fileno())

        self._records[key] = record
        rewrite_calibration_csv(self._ordered_records(), self.csv_path)

    def _ordered_records(self) -> tuple[MAPFCBSCalibrationRecord, ...]:
        return tuple(
            sorted(
                self._records.values(),
                key=lambda record: (
                    record.agent_count,
                    record.interaction_level,
                    record.instance_id,
                    DEFAULT_CALIBRATION_ALGORITHMS.index(record.algorithm)
                    if record.algorithm in DEFAULT_CALIBRATION_ALGORITHMS
                    else len(DEFAULT_CALIBRATION_ALGORITHMS),
                    record.max_expanded_nodes,
                ),
            )
        )


def validate_calibration_record(record: MAPFCBSCalibrationRecord) -> None:
    if record.algorithm not in DEFAULT_CALIBRATION_ALGORITHMS:
        raise RuntimeError(f"unexpected calibration algorithm: {record.algorithm}")

    if record.success and record.termination_reason != TERMINATION_SUCCESS:
        raise RuntimeError("successful calibration run must have termination_reason=success")
    if record.success and (record.soc is None or record.makespan is None):
        raise RuntimeError("successful calibration run missing quality metrics")
    if not record.success and (record.soc is not None or record.makespan is not None):
        raise RuntimeError("non-successful calibration run must not include quality metrics")
    if record.termination_reason == TERMINATION_ERROR and record.error_message is None:
        raise RuntimeError("error calibration run missing error_message")


def summarize_calibration_pair(
    *,
    instance_id: str,
    algorithm: MAPFBenchmarkAlgorithm,
    budgets: Sequence[int],
    records: Mapping[int, MAPFCBSCalibrationRecord],
) -> MAPFCBSCalibrationPairSummary | None:
    if not records:
        return None

    ordered = [records[budget] for budget in budgets if budget in records]
    if not ordered:
        return None

    first = ordered[0]
    success_records = [
        record for record in ordered if record.termination_reason == TERMINATION_SUCCESS
    ]
    if success_records:
        outcome_record = min(success_records, key=lambda record: record.max_expanded_nodes)
        outcome = TERMINATION_SUCCESS
        smallest_successful_budget = outcome_record.max_expanded_nodes
        largest_tested_budget = max(record.max_expanded_nodes for record in ordered)
    else:
        outcome_record = ordered[-1]
        outcome = outcome_record.termination_reason
        smallest_successful_budget = None
        largest_tested_budget = outcome_record.max_expanded_nodes

    return MAPFCBSCalibrationPairSummary(
        instance_id=instance_id,
        algorithm=algorithm,
        agent_count=first.agent_count,
        interaction_level=first.interaction_level,
        outcome=outcome,
        smallest_successful_budget=smallest_successful_budget,
        largest_tested_budget=largest_tested_budget,
        runtime_at_outcome_ms=outcome_record.execution_time_ms,
        expanded_ct_nodes_at_outcome=outcome_record.cbs_search_metrics.expanded_ct_nodes,
        combined_low_level_searches_at_outcome=(
            outcome_record.cbs_search_metrics.combined_low_level_searches
        ),
    )


def build_pair_summaries(
    instances: Sequence[MAPFBenchmarkInstance],
    *,
    algorithms: Sequence[MAPFBenchmarkAlgorithm],
    budgets: Sequence[int],
    records: Mapping[MAPFCBSCalibrationRunKey, MAPFCBSCalibrationRecord],
) -> tuple[MAPFCBSCalibrationPairSummary, ...]:
    grouped = group_calibration_records_by_pair(records)
    summaries: list[MAPFCBSCalibrationPairSummary] = []

    for instance in instances:
        for algorithm in algorithms:
            pair = calibration_pair_key(instance.instance_id, algorithm)
            pair_records = grouped.get(pair, {})
            summary = summarize_calibration_pair(
                instance_id=instance.instance_id,
                algorithm=algorithm,
                budgets=budgets,
                records=pair_records,
            )
            if summary is not None:
                summaries.append(summary)

    return tuple(summaries)


CalibrationRunExecutor = Callable[
    [MAPFCBSCalibrationPlanEntry],
    MAPFCBSCalibrationRecord,
]


def run_progressive_calibration(
    instances: Sequence[MAPFBenchmarkInstance],
    *,
    algorithms: Sequence[MAPFBenchmarkAlgorithm],
    budgets: Sequence[int],
    executor: CalibrationRunExecutor,
    checkpoint: CalibrationCheckpointStore | None = None,
    resume: bool = True,
    on_before_run: Callable[[MAPFCBSCalibrationPlanEntry, int], None] | None = None,
    on_record: Callable[[MAPFCBSCalibrationRecord, int], None] | None = None,
    stop_on_error: bool = False,
) -> tuple[MAPFCBSCalibrationRecord, ...]:
    """Execute progressive calibration, one budget step at a time per pair."""
    if checkpoint is not None:
        pair_state = group_calibration_records_by_pair(checkpoint.records)
        completed_keys = checkpoint.completed_keys()
    else:
        pair_state = {}
        completed_keys = set()

    executed: list[MAPFCBSCalibrationRecord] = []
    run_counter = 0

    for instance in instances:
        for algorithm in algorithms:
            while True:
                pair = calibration_pair_key(instance.instance_id, algorithm)
                pair_records = pair_state.get(pair, {})

                if is_calibration_pair_complete(
                    budgets=budgets,
                    existing_records=pair_records,
                ):
                    break

                pending = budgets_to_execute_for_pair(
                    budgets=budgets,
                    existing_records=pair_records,
                )
                if not pending:
                    break

                budget = pending[0]
                entry = MAPFCBSCalibrationPlanEntry(
                    instance=instance,
                    algorithm=algorithm,
                    max_expanded_nodes=budget,
                )
                key = MAPFCBSCalibrationRunKey(
                    instance_id=instance.instance_id,
                    algorithm=algorithm,
                    max_expanded_nodes=budget,
                )
                if resume and key in completed_keys:
                    existing = pair_records[budget]
                    if should_stop_budget_escalation(existing, budgets=budgets):
                        break
                    continue

                if on_before_run is not None:
                    on_before_run(entry, run_counter + 1)

                record = executor(entry)
                validate_calibration_record(record)

                if checkpoint is not None:
                    checkpoint.append_record(record)

                pair_records = pair_state.setdefault(pair, {})
                pair_records[record.max_expanded_nodes] = record
                completed_keys.add(calibration_run_key(record))
                executed.append(record)
                run_counter += 1

                if on_record is not None:
                    on_record(record, run_counter)

                if should_stop_budget_escalation(record, budgets=budgets):
                    break

                if record.termination_reason == TERMINATION_ERROR and stop_on_error:
                    return tuple(executed)

    return tuple(executed)
