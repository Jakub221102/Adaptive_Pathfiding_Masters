"""MAPF-9 core: Conflict-Guided Bounded Local Priority Search (CGLPS) and UBLS.

Implements the frozen MAPF-9.0 design for reusable instance-level evaluation.
Production benchmark orchestration belongs in MAPF-9.4.
"""

from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

from pathfinding.src.algorithms.mapf.conflicts import detect_conflicts
from pathfinding.src.algorithms.mapf.metrics import makespan, sum_of_costs
from pathfinding.src.algorithms.mapf.models import MAPFResult, MAPFScenario
from pathfinding.src.algorithms.mapf.priority_ordering import (
    ConflictAwareOrderingInputs,
    build_conflict_aware_ordering_inputs,
    conflict_graph_edges,
    pair_conflict_event_counts,
)
from pathfinding.src.algorithms.mapf.prioritized_planning import (
    PrioritizedPlanningRunResult,
    plan_prioritized_with_stats,
)
from pathfinding.src.core.models import GridMap
from pathfinding.src.experiments.mapf_benchmark_execution import (
    MAPFPPSearchMetrics,
    TERMINATION_FAILURE,
    TERMINATION_SUCCESS,
    _pp_search_metrics_from_stats,
)

TERMINATION_ORDERING_FAILURE = "ordering_failure"

MAPF9_CGLPS_BUDGET = 4
MAPF9_UBLS_GLOBAL_SEED = 9031


class MAPF9Method(str, Enum):
    SPF = "spf"
    CGLPS = "cglps"
    UBLS = "ubls"


@dataclass(frozen=True, slots=True)
class MAPF9PreprocessingTimings:
    independent_path_time_ms: float
    spf_ordering_time_ms: float


@dataclass(frozen=True, slots=True)
class MAPF9SharedInstance:
    grid_map: GridMap
    scenario: MAPFScenario
    max_timestep: int
    inputs: ConflictAwareOrderingInputs
    spf_order: tuple[int, ...]
    original_index_by_agent_id: dict[int, int]
    agent_id_by_original_index: dict[int, int]
    conflict_edges: frozenset[tuple[int, int]]
    pair_event_counts: dict[tuple[int, int], int]
    preprocessing_timings: MAPF9PreprocessingTimings


@dataclass(frozen=True, slots=True)
class MAPF9OrderingFailure:
    termination_reason: str
    error_message: str
    preprocessing_timings: MAPF9PreprocessingTimings | None = None


@dataclass(frozen=True, slots=True)
class MAPF9CandidateSpec:
    candidate_id: int
    agent_order: tuple[int, ...]
    source_agent_a_id: int | None
    source_agent_b_id: int | None
    canonical_original_index_pair: tuple[int, int] | None
    pair_conflict_event_count: int | None


@dataclass(frozen=True, slots=True)
class MAPF9CandidateEvaluation:
    candidate_id: int
    agent_order: tuple[int, ...]
    source_agent_a_id: int | None
    source_agent_b_id: int | None
    canonical_original_index_pair: tuple[int, int] | None
    pair_conflict_event_count: int | None
    success: bool
    termination_reason: str
    soc: int | None
    makespan: int | None
    conflict_count: int | None
    pp_time_ms: float
    pp_search_metrics: MAPFPPSearchMetrics | None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class MAPF9MethodTimings:
    baseline_spf_pp_time_ms: float
    conflict_detection_and_ranking_time_ms: float = 0.0
    cglps_additional_pp_time_ms: float = 0.0
    ubls_candidate_generation_time_ms: float = 0.0
    ubls_additional_pp_time_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class MAPF9MethodResult:
    method: MAPF9Method
    selected_candidate_id: int
    selected_agent_order: tuple[int, ...]
    success: bool
    termination_reason: str
    soc: int | None
    makespan: int | None
    conflict_count: int | None
    baseline_spf_evaluation: MAPF9CandidateEvaluation | None
    additional_pp_eval_count: int
    total_logical_pp_candidate_count: int
    additional_candidate_specs: tuple[MAPF9CandidateSpec, ...]
    candidate_evaluations: tuple[MAPF9CandidateEvaluation, ...]
    independent_conflict_count: int | None
    independent_conflict_pair_count: int | None
    preprocessing_timings: MAPF9PreprocessingTimings | None
    method_timings: MAPF9MethodTimings | None
    error_message: str | None = None


def original_index_by_agent_id(scenario: MAPFScenario) -> dict[int, int]:
    return {agent.agent_id: index for index, agent in enumerate(scenario.agents)}


def agent_id_by_original_index(scenario: MAPFScenario) -> dict[int, int]:
    return {index: agent.agent_id for index, agent in enumerate(scenario.agents)}


def spf_agent_order_from_inputs(
    scenario: MAPFScenario,
    inputs: ConflictAwareOrderingInputs,
) -> tuple[int, ...]:
    ranked_indices = sorted(
        range(len(scenario.agents)),
        key=lambda index: (inputs.independent_costs[index], index),
    )
    return tuple(scenario.agents[index].agent_id for index in ranked_indices)


def transpose_order(
    base_order: tuple[int, ...],
    agent_a_id: int,
    agent_b_id: int,
) -> tuple[int, ...]:
    if agent_a_id == agent_b_id:
        raise ValueError("agent_a_id and agent_b_id must differ")

    order = list(base_order)
    try:
        pos_a = order.index(agent_a_id)
        pos_b = order.index(agent_b_id)
    except ValueError as error:
        raise ValueError(
            f"agent id not found in order: {error.args[0]}"
        ) from error

    order[pos_a], order[pos_b] = order[pos_b], order[pos_a]
    return tuple(order)


def scenario_from_agent_order(
    scenario: MAPFScenario,
    agent_order: tuple[int, ...],
) -> MAPFScenario:
    agent_by_id = {agent.agent_id: agent for agent in scenario.agents}
    missing = [agent_id for agent_id in agent_order if agent_id not in agent_by_id]
    if missing:
        raise ValueError(f"unknown agent ids in order: {missing}")
    if len(agent_order) != len(scenario.agents):
        raise ValueError("agent_order length must match scenario agent count")
    return MAPFScenario(agents=tuple(agent_by_id[agent_id] for agent_id in agent_order))


def ubls_random_generator(instance_id: str) -> random.Random:
    text = f"{MAPF9_UBLS_GLOBAL_SEED}:{instance_id}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return random.Random(seed)


def prepare_mapf9_instance(
    *,
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
) -> MAPF9SharedInstance | MAPF9OrderingFailure:
    if max_timestep < 0:
        raise ValueError("max_timestep must be non-negative")

    independent_start = time.perf_counter()
    try:
        inputs = build_conflict_aware_ordering_inputs(
            grid_map=grid_map,
            scenario=scenario,
            max_timestep=max_timestep,
        )
    except ValueError as error:
        independent_time_ms = (time.perf_counter() - independent_start) * 1000.0
        return MAPF9OrderingFailure(
            termination_reason=TERMINATION_ORDERING_FAILURE,
            error_message=str(error),
            preprocessing_timings=MAPF9PreprocessingTimings(
                independent_path_time_ms=independent_time_ms,
                spf_ordering_time_ms=0.0,
            ),
        )
    independent_time_ms = (time.perf_counter() - independent_start) * 1000.0

    spf_start = time.perf_counter()
    spf_order = spf_agent_order_from_inputs(scenario, inputs)
    spf_ordering_time_ms = (time.perf_counter() - spf_start) * 1000.0

    id_to_index = original_index_by_agent_id(scenario)
    index_to_id = agent_id_by_original_index(scenario)
    pair_counts = pair_conflict_event_counts(scenario, inputs.conflicts)
    edges = conflict_graph_edges(scenario, inputs.conflicts)

    return MAPF9SharedInstance(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
        inputs=inputs,
        spf_order=spf_order,
        original_index_by_agent_id=id_to_index,
        agent_id_by_original_index=index_to_id,
        conflict_edges=edges,
        pair_event_counts=pair_counts,
        preprocessing_timings=MAPF9PreprocessingTimings(
            independent_path_time_ms=independent_time_ms,
            spf_ordering_time_ms=spf_ordering_time_ms,
        ),
    )


def generate_cglps_candidate_specs(
    shared: MAPF9SharedInstance,
    *,
    budget: int = MAPF9_CGLPS_BUDGET,
) -> tuple[MAPF9CandidateSpec, ...]:
    if budget < 0:
        raise ValueError("budget must be non-negative")

    ranked_edges = sorted(
        shared.conflict_edges,
        key=lambda pair: (
            -shared.pair_event_counts.get(pair, 0),
            pair[0],
            pair[1],
        ),
    )

    seen_orders: set[tuple[int, ...]] = {shared.spf_order}
    specs: list[MAPF9CandidateSpec] = []
    candidate_id = 1

    for first_index, second_index in ranked_edges:
        if len(specs) >= budget:
            break

        agent_a_id = shared.agent_id_by_original_index[first_index]
        agent_b_id = shared.agent_id_by_original_index[second_index]
        swap_order = transpose_order(shared.spf_order, agent_a_id, agent_b_id)
        if swap_order in seen_orders:
            continue

        seen_orders.add(swap_order)
        specs.append(
            MAPF9CandidateSpec(
                candidate_id=candidate_id,
                agent_order=swap_order,
                source_agent_a_id=agent_a_id,
                source_agent_b_id=agent_b_id,
                canonical_original_index_pair=(first_index, second_index),
                pair_conflict_event_count=shared.pair_event_counts.get(
                    (first_index, second_index),
                    0,
                ),
            )
        )
        candidate_id += 1

    return tuple(specs)


def generate_ubls_candidate_specs(
    shared: MAPF9SharedInstance,
    *,
    instance_id: str,
    additional_count: int,
) -> tuple[MAPF9CandidateSpec, ...]:
    if additional_count < 0:
        raise ValueError("additional_count must be non-negative")

    agent_count = len(shared.scenario.agents)
    all_pairs = [
        (first_index, second_index)
        for first_index in range(agent_count)
        for second_index in range(first_index + 1, agent_count)
    ]
    rng = ubls_random_generator(instance_id)
    shuffled_pairs = list(all_pairs)
    rng.shuffle(shuffled_pairs)

    seen_orders: set[tuple[int, ...]] = {shared.spf_order}
    specs: list[MAPF9CandidateSpec] = []
    candidate_id = 1

    for first_index, second_index in shuffled_pairs:
        if len(specs) >= additional_count:
            break

        agent_a_id = shared.agent_id_by_original_index[first_index]
        agent_b_id = shared.agent_id_by_original_index[second_index]
        swap_order = transpose_order(shared.spf_order, agent_a_id, agent_b_id)
        if swap_order in seen_orders:
            continue

        seen_orders.add(swap_order)
        specs.append(
            MAPF9CandidateSpec(
                candidate_id=candidate_id,
                agent_order=swap_order,
                source_agent_a_id=agent_a_id,
                source_agent_b_id=agent_b_id,
                canonical_original_index_pair=(first_index, second_index),
                pair_conflict_event_count=None,
            )
        )
        candidate_id += 1

    return tuple(specs)


def _validate_successful_solution(result: MAPFResult) -> None:
    if not result.success:
        raise RuntimeError("expected successful MAPF result")

    conflicts = detect_conflicts(result.paths)
    if conflicts:
        raise RuntimeError(
            f"successful MAPF result still has {len(conflicts)} conflicts"
        )


def _evaluate_pp_for_order(
    *,
    shared: MAPF9SharedInstance,
    candidate_id: int,
    agent_order: tuple[int, ...],
    spec: MAPF9CandidateSpec | None,
) -> MAPF9CandidateEvaluation:
    pp_start = time.perf_counter()
    ordered_scenario = scenario_from_agent_order(shared.scenario, agent_order)
    run = plan_prioritized_with_stats(
        grid_map=shared.grid_map,
        scenario=ordered_scenario,
        max_timestep=shared.max_timestep,
    )
    pp_time_ms = (time.perf_counter() - pp_start) * 1000.0
    pp_metrics = _pp_search_metrics_from_stats(run.stats)

    source_a = None if spec is None else spec.source_agent_a_id
    source_b = None if spec is None else spec.source_agent_b_id
    canonical_pair = None if spec is None else spec.canonical_original_index_pair
    pair_count = None if spec is None else spec.pair_conflict_event_count

    if not run.result.success:
        return MAPF9CandidateEvaluation(
            candidate_id=candidate_id,
            agent_order=agent_order,
            source_agent_a_id=source_a,
            source_agent_b_id=source_b,
            canonical_original_index_pair=canonical_pair,
            pair_conflict_event_count=pair_count,
            success=False,
            termination_reason=run.termination_reason,
            soc=None,
            makespan=None,
            conflict_count=None,
            pp_time_ms=pp_time_ms,
            pp_search_metrics=pp_metrics,
        )

    _validate_successful_solution(run.result)
    return MAPF9CandidateEvaluation(
        candidate_id=candidate_id,
        agent_order=agent_order,
        source_agent_a_id=source_a,
        source_agent_b_id=source_b,
        canonical_original_index_pair=canonical_pair,
        pair_conflict_event_count=pair_count,
        success=True,
        termination_reason=TERMINATION_SUCCESS,
        soc=sum_of_costs(run.result.paths),
        makespan=makespan(run.result.paths),
        conflict_count=0,
        pp_time_ms=pp_time_ms,
        pp_search_metrics=pp_metrics,
    )


def evaluate_spf_baseline(
    shared: MAPF9SharedInstance,
    *,
    pp_runner: Callable[..., PrioritizedPlanningRunResult] | None = None,
) -> MAPF9CandidateEvaluation:
    if pp_runner is None:
        return _evaluate_pp_for_order(
            shared=shared,
            candidate_id=0,
            agent_order=shared.spf_order,
            spec=None,
        )

    pp_start = time.perf_counter()
    ordered_scenario = scenario_from_agent_order(shared.scenario, shared.spf_order)
    run = pp_runner(
        grid_map=shared.grid_map,
        scenario=ordered_scenario,
        max_timestep=shared.max_timestep,
    )
    pp_time_ms = (time.perf_counter() - pp_start) * 1000.0
    pp_metrics = _pp_search_metrics_from_stats(run.stats)

    if not run.result.success:
        return MAPF9CandidateEvaluation(
            candidate_id=0,
            agent_order=shared.spf_order,
            source_agent_a_id=None,
            source_agent_b_id=None,
            canonical_original_index_pair=None,
            pair_conflict_event_count=None,
            success=False,
            termination_reason=run.termination_reason,
            soc=None,
            makespan=None,
            conflict_count=None,
            pp_time_ms=pp_time_ms,
            pp_search_metrics=pp_metrics,
        )

    _validate_successful_solution(run.result)
    return MAPF9CandidateEvaluation(
        candidate_id=0,
        agent_order=shared.spf_order,
        source_agent_a_id=None,
        source_agent_b_id=None,
        canonical_original_index_pair=None,
        pair_conflict_event_count=None,
        success=True,
        termination_reason=TERMINATION_SUCCESS,
        soc=sum_of_costs(run.result.paths),
        makespan=makespan(run.result.paths),
        conflict_count=0,
        pp_time_ms=pp_time_ms,
        pp_search_metrics=pp_metrics,
    )


def _selection_sort_key(evaluation: MAPF9CandidateEvaluation) -> tuple[int, ...]:
    failure_rank = 0 if evaluation.success else 1
    if evaluation.success:
        assert evaluation.soc is not None
        assert evaluation.makespan is not None
        return (
            failure_rank,
            evaluation.soc,
            evaluation.makespan,
            evaluation.candidate_id,
        )
    return (failure_rank, evaluation.candidate_id)


def select_best_candidate_evaluation(
    evaluations: Sequence[MAPF9CandidateEvaluation],
) -> MAPF9CandidateEvaluation:
    if not evaluations:
        raise ValueError("evaluations must not be empty")
    return min(evaluations, key=_selection_sort_key)


def _method_result_from_evaluations(
    *,
    method: MAPF9Method,
    shared: MAPF9SharedInstance,
    baseline: MAPF9CandidateEvaluation,
    additional_specs: tuple[MAPF9CandidateSpec, ...],
    additional_evaluations: tuple[MAPF9CandidateEvaluation, ...],
    method_timings: MAPF9MethodTimings,
) -> MAPF9MethodResult:
    all_evaluations = (baseline, *additional_evaluations)
    selected = select_best_candidate_evaluation(all_evaluations)
    additional_pp_time = sum(
        evaluation.pp_time_ms for evaluation in additional_evaluations
    )
    if method == MAPF9Method.CGLPS:
        method_timings = MAPF9MethodTimings(
            baseline_spf_pp_time_ms=method_timings.baseline_spf_pp_time_ms,
            conflict_detection_and_ranking_time_ms=(
                method_timings.conflict_detection_and_ranking_time_ms
            ),
            cglps_additional_pp_time_ms=additional_pp_time,
            ubls_candidate_generation_time_ms=(
                method_timings.ubls_candidate_generation_time_ms
            ),
            ubls_additional_pp_time_ms=method_timings.ubls_additional_pp_time_ms,
        )
    elif method == MAPF9Method.UBLS:
        method_timings = MAPF9MethodTimings(
            baseline_spf_pp_time_ms=method_timings.baseline_spf_pp_time_ms,
            conflict_detection_and_ranking_time_ms=(
                method_timings.conflict_detection_and_ranking_time_ms
            ),
            cglps_additional_pp_time_ms=method_timings.cglps_additional_pp_time_ms,
            ubls_candidate_generation_time_ms=(
                method_timings.ubls_candidate_generation_time_ms
            ),
            ubls_additional_pp_time_ms=additional_pp_time,
        )

    return MAPF9MethodResult(
        method=method,
        selected_candidate_id=selected.candidate_id,
        selected_agent_order=selected.agent_order,
        success=selected.success,
        termination_reason=selected.termination_reason,
        soc=selected.soc,
        makespan=selected.makespan,
        conflict_count=selected.conflict_count,
        baseline_spf_evaluation=baseline,
        additional_pp_eval_count=len(additional_evaluations),
        total_logical_pp_candidate_count=1 + len(additional_evaluations),
        additional_candidate_specs=additional_specs,
        candidate_evaluations=all_evaluations,
        independent_conflict_count=len(shared.inputs.conflicts),
        independent_conflict_pair_count=shared.inputs.conflict_pair_count,
        preprocessing_timings=shared.preprocessing_timings,
        method_timings=method_timings,
    )


def evaluate_cglps(
    shared: MAPF9SharedInstance,
    baseline: MAPF9CandidateEvaluation,
    *,
    budget: int = MAPF9_CGLPS_BUDGET,
    pp_runner: Callable[..., PrioritizedPlanningRunResult] | None = None,
) -> MAPF9MethodResult:
    if baseline.candidate_id != 0:
        raise ValueError("baseline must be SPF candidate 0")
    if baseline.agent_order != shared.spf_order:
        raise ValueError("baseline agent_order must match shared SPF order")

    ranking_start = time.perf_counter()
    additional_specs = generate_cglps_candidate_specs(shared, budget=budget)
    ranking_time_ms = (time.perf_counter() - ranking_start) * 1000.0

    additional_evaluations: list[MAPF9CandidateEvaluation] = []
    for spec in additional_specs:
        if pp_runner is None:
            evaluation = _evaluate_pp_for_order(
                shared=shared,
                candidate_id=spec.candidate_id,
                agent_order=spec.agent_order,
                spec=spec,
            )
        else:
            pp_start = time.perf_counter()
            ordered_scenario = scenario_from_agent_order(
                shared.scenario,
                spec.agent_order,
            )
            run = pp_runner(
                grid_map=shared.grid_map,
                scenario=ordered_scenario,
                max_timestep=shared.max_timestep,
            )
            pp_time_ms = (time.perf_counter() - pp_start) * 1000.0
            pp_metrics = _pp_search_metrics_from_stats(run.stats)
            if not run.result.success:
                evaluation = MAPF9CandidateEvaluation(
                    candidate_id=spec.candidate_id,
                    agent_order=spec.agent_order,
                    source_agent_a_id=spec.source_agent_a_id,
                    source_agent_b_id=spec.source_agent_b_id,
                    canonical_original_index_pair=spec.canonical_original_index_pair,
                    pair_conflict_event_count=spec.pair_conflict_event_count,
                    success=False,
                    termination_reason=run.termination_reason,
                    soc=None,
                    makespan=None,
                    conflict_count=None,
                    pp_time_ms=pp_time_ms,
                    pp_search_metrics=pp_metrics,
                )
            else:
                _validate_successful_solution(run.result)
                evaluation = MAPF9CandidateEvaluation(
                    candidate_id=spec.candidate_id,
                    agent_order=spec.agent_order,
                    source_agent_a_id=spec.source_agent_a_id,
                    source_agent_b_id=spec.source_agent_b_id,
                    canonical_original_index_pair=spec.canonical_original_index_pair,
                    pair_conflict_event_count=spec.pair_conflict_event_count,
                    success=True,
                    termination_reason=TERMINATION_SUCCESS,
                    soc=sum_of_costs(run.result.paths),
                    makespan=makespan(run.result.paths),
                    conflict_count=0,
                    pp_time_ms=pp_time_ms,
                    pp_search_metrics=pp_metrics,
                )
        additional_evaluations.append(evaluation)

    return _method_result_from_evaluations(
        method=MAPF9Method.CGLPS,
        shared=shared,
        baseline=baseline,
        additional_specs=additional_specs,
        additional_evaluations=tuple(additional_evaluations),
        method_timings=MAPF9MethodTimings(
            baseline_spf_pp_time_ms=baseline.pp_time_ms,
            conflict_detection_and_ranking_time_ms=ranking_time_ms,
        ),
    )


def evaluate_ubls(
    shared: MAPF9SharedInstance,
    baseline: MAPF9CandidateEvaluation,
    *,
    instance_id: str,
    additional_count: int,
    pp_runner: Callable[..., PrioritizedPlanningRunResult] | None = None,
) -> MAPF9MethodResult:
    if baseline.candidate_id != 0:
        raise ValueError("baseline must be SPF candidate 0")
    if baseline.agent_order != shared.spf_order:
        raise ValueError("baseline agent_order must match shared SPF order")
    if additional_count < 0:
        raise ValueError("additional_count must be non-negative")

    generation_start = time.perf_counter()
    additional_specs = generate_ubls_candidate_specs(
        shared,
        instance_id=instance_id,
        additional_count=additional_count,
    )
    generation_time_ms = (time.perf_counter() - generation_start) * 1000.0

    additional_evaluations: list[MAPF9CandidateEvaluation] = []
    for spec in additional_specs:
        if pp_runner is None:
            evaluation = _evaluate_pp_for_order(
                shared=shared,
                candidate_id=spec.candidate_id,
                agent_order=spec.agent_order,
                spec=spec,
            )
        else:
            pp_start = time.perf_counter()
            ordered_scenario = scenario_from_agent_order(
                shared.scenario,
                spec.agent_order,
            )
            run = pp_runner(
                grid_map=shared.grid_map,
                scenario=ordered_scenario,
                max_timestep=shared.max_timestep,
            )
            pp_time_ms = (time.perf_counter() - pp_start) * 1000.0
            pp_metrics = _pp_search_metrics_from_stats(run.stats)
            if not run.result.success:
                evaluation = MAPF9CandidateEvaluation(
                    candidate_id=spec.candidate_id,
                    agent_order=spec.agent_order,
                    source_agent_a_id=spec.source_agent_a_id,
                    source_agent_b_id=spec.source_agent_b_id,
                    canonical_original_index_pair=spec.canonical_original_index_pair,
                    pair_conflict_event_count=None,
                    success=False,
                    termination_reason=run.termination_reason,
                    soc=None,
                    makespan=None,
                    conflict_count=None,
                    pp_time_ms=pp_time_ms,
                    pp_search_metrics=pp_metrics,
                )
            else:
                _validate_successful_solution(run.result)
                evaluation = MAPF9CandidateEvaluation(
                    candidate_id=spec.candidate_id,
                    agent_order=spec.agent_order,
                    source_agent_a_id=spec.source_agent_a_id,
                    source_agent_b_id=spec.source_agent_b_id,
                    canonical_original_index_pair=spec.canonical_original_index_pair,
                    pair_conflict_event_count=None,
                    success=True,
                    termination_reason=TERMINATION_SUCCESS,
                    soc=sum_of_costs(run.result.paths),
                    makespan=makespan(run.result.paths),
                    conflict_count=0,
                    pp_time_ms=pp_time_ms,
                    pp_search_metrics=pp_metrics,
                )
        additional_evaluations.append(evaluation)

    return _method_result_from_evaluations(
        method=MAPF9Method.UBLS,
        shared=shared,
        baseline=baseline,
        additional_specs=additional_specs,
        additional_evaluations=tuple(additional_evaluations),
        method_timings=MAPF9MethodTimings(
            baseline_spf_pp_time_ms=baseline.pp_time_ms,
            ubls_candidate_generation_time_ms=generation_time_ms,
        ),
    )


def evaluate_spf_method_result(
    shared: MAPF9SharedInstance,
    baseline: MAPF9CandidateEvaluation,
) -> MAPF9MethodResult:
    if baseline.candidate_id != 0:
        raise ValueError("baseline must be SPF candidate 0")

    return MAPF9MethodResult(
        method=MAPF9Method.SPF,
        selected_candidate_id=0,
        selected_agent_order=baseline.agent_order,
        success=baseline.success,
        termination_reason=baseline.termination_reason,
        soc=baseline.soc,
        makespan=baseline.makespan,
        conflict_count=baseline.conflict_count,
        baseline_spf_evaluation=baseline,
        additional_pp_eval_count=0,
        total_logical_pp_candidate_count=1,
        additional_candidate_specs=(),
        candidate_evaluations=(baseline,),
        independent_conflict_count=len(shared.inputs.conflicts),
        independent_conflict_pair_count=shared.inputs.conflict_pair_count,
        preprocessing_timings=shared.preprocessing_timings,
        method_timings=MAPF9MethodTimings(
            baseline_spf_pp_time_ms=baseline.pp_time_ms,
        ),
    )


def ordering_failure_method_result(
    failure: MAPF9OrderingFailure,
    *,
    method: MAPF9Method,
) -> MAPF9MethodResult:
    return MAPF9MethodResult(
        method=method,
        selected_candidate_id=0,
        selected_agent_order=(),
        success=False,
        termination_reason=failure.termination_reason,
        soc=None,
        makespan=None,
        conflict_count=None,
        baseline_spf_evaluation=None,
        additional_pp_eval_count=0,
        total_logical_pp_candidate_count=0,
        additional_candidate_specs=(),
        candidate_evaluations=(),
        independent_conflict_count=None,
        independent_conflict_pair_count=None,
        preprocessing_timings=failure.preprocessing_timings,
        method_timings=None,
        error_message=failure.error_message,
    )
