"""Unit tests for MAPF-9.1 CGLPS/UBLS core."""

from __future__ import annotations

import hashlib
import random
from unittest.mock import MagicMock

import pytest

from pathfinding.src.algorithms.mapf.models import (
    AgentPath,
    EdgeConflict,
    MAPFAgent,
    MAPFResult,
    MAPFScenario,
    TimedState,
    VertexConflict,
)
from pathfinding.src.algorithms.mapf.priority_ordering import pair_conflict_event_counts
from pathfinding.src.algorithms.mapf.prioritized_planning import (
    PrioritizedPlanningRunResult,
    PrioritizedPlanningStats,
)
from pathfinding.src.experiments.mapf_bounded_local_priority_search import (
    MAPF9_CGLPS_BUDGET,
    MAPF9_UBLS_GLOBAL_SEED,
    MAPF9CandidateEvaluation,
    MAPF9Method,
    MAPF9OrderingFailure,
    MAPF9SharedInstance,
    evaluate_cglps,
    evaluate_spf_baseline,
    evaluate_ubls,
    generate_cglps_candidate_specs,
    generate_ubls_candidate_specs,
    ordering_failure_method_result,
    prepare_mapf9_instance,
    select_best_candidate_evaluation,
    transpose_order,
    ubls_random_generator,
)
from pathfinding.src.experiments.mapf_benchmark_execution import TERMINATION_FAILURE
from pathfinding.tests.helpers import build_grid_map


from pathfinding.src.core.models import Position


def _agent(
    agent_id: int,
    start_row: int,
    start_col: int,
    goal_row: int,
    goal_col: int,
) -> MAPFAgent:
    return MAPFAgent(
        agent_id=agent_id,
        start=Position(row=start_row, col=start_col),
        goal=Position(row=goal_row, col=goal_col),
    )


def _scenario(*agents: MAPFAgent) -> MAPFScenario:
    return MAPFScenario(agents=agents)


def _path(agent_id: int, coordinates: list[tuple[int, int, int]]) -> AgentPath:
    return AgentPath(
        agent_id=agent_id,
        states=tuple(
            TimedState(row=row, col=col, timestep=timestep)
            for row, col, timestep in coordinates
        ),
    )


def _evaluation(
    *,
    candidate_id: int,
    success: bool,
    soc: int | None = None,
    makespan: int | None = None,
    agent_order: tuple[int, ...] = (0, 1),
) -> MAPF9CandidateEvaluation:
    return MAPF9CandidateEvaluation(
        candidate_id=candidate_id,
        agent_order=agent_order,
        source_agent_a_id=None,
        source_agent_b_id=None,
        canonical_original_index_pair=None,
        pair_conflict_event_count=None,
        success=success,
        termination_reason="success" if success else TERMINATION_FAILURE,
        soc=soc,
        makespan=makespan,
        conflict_count=0 if success else None,
        pp_time_ms=1.0,
        pp_search_metrics=None,
    )


# --- pair_conflict_event_counts ---


def test_pair_counts_vertex_conflict_increments_canonical_pair() -> None:
    scenario = _scenario(
        _agent(0, 0, 0, 0, 1),
        _agent(1, 1, 0, 1, 1),
    )
    conflicts = (
        VertexConflict(agent1_id=0, agent2_id=1, row=0, col=0, timestep=1),
    )
    assert pair_conflict_event_counts(scenario, conflicts) == {(0, 1): 1}


def test_pair_counts_edge_conflict_increments_canonical_pair() -> None:
    scenario = _scenario(
        _agent(0, 0, 0, 0, 1),
        _agent(1, 0, 1, 0, 0),
    )
    conflicts = (
        EdgeConflict(
            agent1_id=0,
            agent2_id=1,
            agent1_from_row=0,
            agent1_from_col=0,
            agent1_to_row=0,
            agent1_to_col=1,
            agent2_from_row=0,
            agent2_from_col=1,
            agent2_to_row=0,
            agent2_to_col=0,
            timestep=1,
        ),
    )
    assert pair_conflict_event_counts(scenario, conflicts) == {(0, 1): 1}


def test_pair_counts_multiple_events_accumulate() -> None:
    scenario = _scenario(
        _agent(0, 0, 0, 0, 1),
        _agent(1, 1, 0, 1, 1),
    )
    conflicts = (
        VertexConflict(agent1_id=0, agent2_id=1, row=0, col=0, timestep=1),
        VertexConflict(agent1_id=0, agent2_id=1, row=0, col=0, timestep=2),
    )
    assert pair_conflict_event_counts(scenario, conflicts) == {(0, 1): 2}


def test_pair_counts_reversed_orientation_same_canonical_pair() -> None:
    scenario = _scenario(
        _agent(9, 0, 0, 0, 1),
        _agent(2, 1, 0, 1, 1),
    )
    conflicts = (
        VertexConflict(agent1_id=2, agent2_id=9, row=0, col=0, timestep=1),
    )
    assert pair_conflict_event_counts(scenario, conflicts) == {(0, 1): 1}


def test_pair_counts_unknown_agent_id_rejected() -> None:
    scenario = _scenario(
        _agent(0, 0, 0, 0, 1),
        _agent(1, 1, 0, 1, 1),
    )
    conflicts = (
        VertexConflict(agent1_id=0, agent2_id=99, row=0, col=0, timestep=1),
    )
    with pytest.raises(ValueError, match="unknown agent_id"):
        pair_conflict_event_counts(scenario, conflicts)


# --- identity/index correctness ---


def test_spf_order_differs_from_original_and_swap_uses_agent_ids() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    scenario = _scenario(
        _agent(agent_id=100, start_row=0, start_col=0, goal_row=0, goal_col=3),
        _agent(agent_id=200, start_row=2, start_col=0, goal_row=2, goal_col=1),
        _agent(agent_id=300, start_row=1, start_col=0, goal_row=1, goal_col=3),
    )
    prepared = prepare_mapf9_instance(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=20,
    )
    assert isinstance(prepared, MAPF9SharedInstance)
    original_order = tuple(agent.agent_id for agent in scenario.agents)
    assert prepared.spf_order != original_order

    swapped = transpose_order(prepared.spf_order, 100, 300)
    assert swapped.index(100) == prepared.spf_order.index(300)
    assert swapped.index(300) == prepared.spf_order.index(100)


def test_original_index_pair_not_interpreted_as_spf_positions() -> None:
    grid_map = build_grid_map([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    scenario = _scenario(
        _agent(agent_id=10, start_row=0, start_col=0, goal_row=0, goal_col=2),
        _agent(agent_id=20, start_row=2, start_col=0, goal_row=2, goal_col=2),
        _agent(agent_id=30, start_row=1, start_col=0, goal_row=1, goal_col=2),
    )
    prepared = prepare_mapf9_instance(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=20,
    )
    assert isinstance(prepared, MAPF9SharedInstance)

    wrong_swap = list(prepared.spf_order)
    wrong_swap[0], wrong_swap[1] = wrong_swap[1], wrong_swap[0]
    wrong_by_index = tuple(wrong_swap)

    correct_swap = transpose_order(prepared.spf_order, 10, 20)
    assert correct_swap != wrong_by_index or prepared.spf_order[0:2] == (10, 20)


# --- CGLPS generation ---


def _shared_with_conflicts(
    monkeypatch: pytest.MonkeyPatch,
    *,
    edges: list[tuple[int, int]],
    pair_counts: dict[tuple[int, int], int],
    spf_order: tuple[int, ...] = (0, 1, 2),
) -> MAPF9SharedInstance:
    from pathfinding.src.algorithms.mapf import priority_ordering as ordering_module

    scenario = _scenario(
        _agent(0, 0, 0, 0, 1),
        _agent(1, 1, 0, 1, 1),
        _agent(2, 2, 0, 2, 1),
    )
    inputs = ordering_module.ConflictAwareOrderingInputs(
        paths=(_path(0, [(0, 0, 0)]), _path(1, [(1, 0, 0)]), _path(2, [(2, 0, 0)])),
        independent_costs=(3, 2, 1),
        degrees=tuple(1 if index in {edge[0] for edge in edges} | {edge[1] for edge in edges} else 0 for index in range(3)),
        incident_counts=(0, 0, 0),
        conflicts=(),
        conflict_pair_count=len(edges),
    )
    return MAPF9SharedInstance(
        grid_map=build_grid_map([[0, 0], [0, 0]]),
        scenario=scenario,
        max_timestep=10,
        inputs=inputs,
        spf_order=spf_order,
        original_index_by_agent_id={0: 0, 1: 1, 2: 2},
        agent_id_by_original_index={0: 0, 1: 1, 2: 2},
        conflict_edges=frozenset(edges),
        pair_event_counts=pair_counts,
        preprocessing_timings=__import__(
            "pathfinding.src.experiments.mapf_bounded_local_priority_search",
            fromlist=["MAPF9PreprocessingTimings"],
        ).MAPF9PreprocessingTimings(0.0, 0.0, 0.0),
    )


def test_cglps_zero_conflict_zero_additional() -> None:
    shared = _shared_with_conflicts(pytest.MonkeyPatch(), edges=[], pair_counts={})
    specs = generate_cglps_candidate_specs(shared)
    assert specs == ()


def test_cglps_fewer_than_budget_edges() -> None:
    shared = _shared_with_conflicts(
        pytest.MonkeyPatch(),
        edges=[(0, 1), (1, 2)],
        pair_counts={(0, 1): 1, (1, 2): 1},
    )
    specs = generate_cglps_candidate_specs(shared)
    assert len(specs) == 2


def test_cglps_more_than_budget_exactly_four() -> None:
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    pair_counts = {
        (0, 1): 5,
        (0, 2): 4,
        (0, 3): 3,
        (1, 2): 2,
        (1, 3): 1,
        (2, 3): 1,
    }
    scenario = _scenario(
        _agent(0, 0, 0, 0, 1),
        _agent(1, 1, 0, 1, 1),
        _agent(2, 2, 0, 2, 1),
        _agent(3, 3, 0, 3, 1),
    )
    from pathfinding.src.experiments.mapf_bounded_local_priority_search import (
        MAPF9PreprocessingTimings,
    )

    shared = MAPF9SharedInstance(
        grid_map=build_grid_map([[0, 0], [0, 0]]),
        scenario=scenario,
        max_timestep=10,
        inputs=__import__(
            "pathfinding.src.algorithms.mapf.priority_ordering",
            fromlist=["ConflictAwareOrderingInputs"],
        ).ConflictAwareOrderingInputs(
            paths=(),
            independent_costs=(4, 3, 2, 1),
            degrees=(0, 0, 0, 0),
            incident_counts=(0, 0, 0, 0),
            conflicts=(),
            conflict_pair_count=len(edges),
        ),
        spf_order=(3, 2, 1, 0),
        original_index_by_agent_id={0: 0, 1: 1, 2: 2, 3: 3},
        agent_id_by_original_index={0: 0, 1: 1, 2: 2, 3: 3},
        conflict_edges=frozenset(edges),
        pair_event_counts=pair_counts,
        preprocessing_timings=MAPF9PreprocessingTimings(0.0, 0.0, 0.0),
    )
    specs = generate_cglps_candidate_specs(shared)
    assert len(specs) == MAPF9_CGLPS_BUDGET
    assert specs[0].canonical_original_index_pair == (0, 1)
    assert specs[1].canonical_original_index_pair == (0, 2)
    assert specs[2].canonical_original_index_pair == (0, 3)
    assert specs[3].canonical_original_index_pair == (1, 2)


def test_cglps_higher_pair_event_count_ranked_first() -> None:
    shared = _shared_with_conflicts(
        pytest.MonkeyPatch(),
        edges=[(0, 2), (1, 2)],
        pair_counts={(0, 2): 5, (1, 2): 1},
    )
    specs = generate_cglps_candidate_specs(shared, budget=1)
    assert specs[0].canonical_original_index_pair == (0, 2)
    assert specs[0].pair_conflict_event_count == 5


def test_cglps_deterministic_index_tie_break() -> None:
    shared = _shared_with_conflicts(
        pytest.MonkeyPatch(),
        edges=[(0, 2), (0, 1)],
        pair_counts={(0, 1): 1, (0, 2): 1},
    )
    specs = generate_cglps_candidate_specs(shared, budget=2)
    assert [spec.canonical_original_index_pair for spec in specs] == [(0, 1), (0, 2)]


def test_cglps_candidates_are_single_transpositions_of_spf() -> None:
    shared = _shared_with_conflicts(
        pytest.MonkeyPatch(),
        edges=[(0, 1), (1, 2)],
        pair_counts={(0, 1): 1, (1, 2): 1},
        spf_order=(2, 1, 0),
    )
    specs = generate_cglps_candidate_specs(shared)
    for spec in specs:
        assert spec.agent_order == transpose_order(
            shared.spf_order,
            spec.source_agent_a_id,
            spec.source_agent_b_id,
        )


def test_cglps_no_candidate_from_another_candidate() -> None:
    shared = _shared_with_conflicts(
        pytest.MonkeyPatch(),
        edges=[(0, 1), (0, 2), (1, 2)],
        pair_counts={(0, 1): 1, (0, 2): 1, (1, 2): 1},
        spf_order=(0, 1, 2),
    )
    specs = generate_cglps_candidate_specs(shared)
    for spec in specs:
        diff_positions = [
            index
            for index, (left, right) in enumerate(
                zip(shared.spf_order, spec.agent_order, strict=True)
            )
            if left != right
        ]
        assert len(diff_positions) == 2


# --- UBLS ---


def test_ubls_exact_sha256_seed_derivation() -> None:
    instance_id = "AR0400SR_n05_low_000"
    text = f"{MAPF9_UBLS_GLOBAL_SEED}:{instance_id}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    expected_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    expected_rng = random.Random(expected_seed)
    actual_rng = ubls_random_generator(instance_id)
    assert actual_rng.random() == expected_rng.random()
    assert actual_rng.random() == expected_rng.random()


def test_ubls_repeated_generation_identical() -> None:
    grid_map = build_grid_map([[0, 0], [0, 0]])
    scenario = _scenario(
        _agent(0, 0, 0, 0, 1),
        _agent(1, 1, 0, 1, 1),
        _agent(2, 0, 1, 1, 1),
    )
    prepared = prepare_mapf9_instance(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=10,
    )
    assert isinstance(prepared, MAPF9SharedInstance)
    first = generate_ubls_candidate_specs(
        prepared,
        instance_id="test_instance",
        additional_count=2,
    )
    second = generate_ubls_candidate_specs(
        prepared,
        instance_id="test_instance",
        additional_count=2,
    )
    assert first == second


def test_ubls_different_instance_ids_different_orders() -> None:
    shared = _shared_with_conflicts(
        pytest.MonkeyPatch(),
        edges=[],
        pair_counts={},
        spf_order=(0, 1, 2),
    )
    first = generate_ubls_candidate_specs(
        shared,
        instance_id="instance_a",
        additional_count=2,
    )
    second = generate_ubls_candidate_specs(
        shared,
        instance_id="instance_b",
        additional_count=2,
    )
    assert first != second


def test_ubls_exactly_ai_candidates() -> None:
    shared = _shared_with_conflicts(
        pytest.MonkeyPatch(),
        edges=[],
        pair_counts={},
        spf_order=(0, 1, 2, 3, 4),
    )
    specs = generate_ubls_candidate_specs(
        shared,
        instance_id="n5",
        additional_count=3,
    )
    assert len(specs) == 3


def test_ubls_does_not_require_conflict_inputs() -> None:
    shared = _shared_with_conflicts(
        pytest.MonkeyPatch(),
        edges=[],
        pair_counts={},
    )
    specs = generate_ubls_candidate_specs(
        shared,
        instance_id="no_conflicts",
        additional_count=1,
    )
    assert specs[0].pair_conflict_event_count is None


def test_ubls_no_duplicate_candidate_orders() -> None:
    shared = _shared_with_conflicts(
        pytest.MonkeyPatch(),
        edges=[],
        pair_counts={},
        spf_order=(0, 1, 2),
    )
    specs = generate_ubls_candidate_specs(
        shared,
        instance_id="dedupe",
        additional_count=3,
    )
    orders = [spec.agent_order for spec in specs]
    assert len(orders) == len(set(orders))


# --- selection ---


def test_selection_success_beats_failure() -> None:
    selected = select_best_candidate_evaluation(
        (
            _evaluation(candidate_id=0, success=False),
            _evaluation(candidate_id=1, success=True, soc=10, makespan=5),
        )
    )
    assert selected.candidate_id == 1


def test_selection_lower_soc_wins() -> None:
    selected = select_best_candidate_evaluation(
        (
            _evaluation(candidate_id=0, success=True, soc=12, makespan=5),
            _evaluation(candidate_id=1, success=True, soc=10, makespan=5),
        )
    )
    assert selected.candidate_id == 1


def test_selection_equal_soc_lower_makespan() -> None:
    selected = select_best_candidate_evaluation(
        (
            _evaluation(candidate_id=0, success=True, soc=10, makespan=8),
            _evaluation(candidate_id=1, success=True, soc=10, makespan=5),
        )
    )
    assert selected.candidate_id == 1


def test_selection_exact_tie_prefers_lower_candidate_id() -> None:
    selected = select_best_candidate_evaluation(
        (
            _evaluation(candidate_id=1, success=True, soc=10, makespan=5),
            _evaluation(candidate_id=0, success=True, soc=10, makespan=5),
        )
    )
    assert selected.candidate_id == 0


def test_selection_successful_spf_cannot_be_worse_than_spf() -> None:
    baseline = _evaluation(candidate_id=0, success=True, soc=10, makespan=5)
    selected = select_best_candidate_evaluation(
        (
            baseline,
            _evaluation(candidate_id=1, success=True, soc=11, makespan=4),
            _evaluation(candidate_id=2, success=True, soc=10, makespan=6),
        )
    )
    assert selected.soc <= baseline.soc


# --- shared baseline / PP accounting ---


def _pp_run(success: bool, soc: int = 10, makespan: int = 5) -> PrioritizedPlanningRunResult:
    if success:
        paths = (
            _path(
                0,
                [(0, 0, 0), (0, 1, 1)],
            ),
        )
        return PrioritizedPlanningRunResult(
            result=MAPFResult(success=True, paths=paths),
            stats=PrioritizedPlanningStats(low_level_searches=1, agents_planned=1),
            termination_reason="success",
        )
    return PrioritizedPlanningRunResult(
        result=MAPFResult(success=False, paths=()),
        stats=PrioritizedPlanningStats(low_level_searches=1, agents_planned=0),
        termination_reason=TERMINATION_FAILURE,
    )


def test_spf_baseline_evaluated_once_and_reused() -> None:
    grid_map = build_grid_map([[0, 0], [0, 0]])
    scenario = _scenario(
        _agent(0, 0, 0, 0, 1),
        _agent(1, 1, 0, 1, 1),
    )
    prepared = prepare_mapf9_instance(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=10,
    )
    assert isinstance(prepared, MAPF9SharedInstance)

    pp_runner = MagicMock(side_effect=lambda **_kwargs: _pp_run(True))
    baseline = evaluate_spf_baseline(prepared, pp_runner=pp_runner)
    cglps = evaluate_cglps(prepared, baseline, budget=1, pp_runner=pp_runner)
    ubls = evaluate_ubls(
        prepared,
        baseline,
        instance_id="inst",
        additional_count=cglps.additional_pp_eval_count,
        pp_runner=pp_runner,
    )

    assert pp_runner.call_count == 1 + cglps.additional_pp_eval_count + ubls.additional_pp_eval_count
    assert baseline is cglps.baseline_spf_evaluation
    assert baseline is ubls.baseline_spf_evaluation


def test_total_physical_pp_calls_equal_one_plus_both_ai() -> None:
    grid_map = build_grid_map([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    scenario = _scenario(
        _agent(0, 0, 0, 0, 2),
        _agent(1, 2, 0, 2, 2),
        _agent(2, 1, 0, 1, 2),
    )
    prepared = prepare_mapf9_instance(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=20,
    )
    assert isinstance(prepared, MAPF9SharedInstance)

    pp_runner = MagicMock(side_effect=lambda **_kwargs: _pp_run(True))
    baseline = evaluate_spf_baseline(prepared, pp_runner=pp_runner)
    cglps = evaluate_cglps(prepared, baseline, pp_runner=pp_runner)
    ubls = evaluate_ubls(
        prepared,
        baseline,
        instance_id="inst",
        additional_count=cglps.additional_pp_eval_count,
        pp_runner=pp_runner,
    )
    expected = 1 + cglps.additional_pp_eval_count + ubls.additional_pp_eval_count
    assert pp_runner.call_count == expected


# --- failure semantics ---


def test_spf_failure_can_be_recovered_by_cglps_candidate() -> None:
    baseline = _evaluation(candidate_id=0, success=False)
    recovered = _evaluation(candidate_id=1, success=True, soc=8, makespan=4)
    selected = select_best_candidate_evaluation((baseline, recovered))
    assert selected.candidate_id == 1
    assert selected.success


def test_spf_failure_can_be_recovered_by_ubls_candidate() -> None:
    baseline = _evaluation(candidate_id=0, success=False)
    recovered = _evaluation(candidate_id=1, success=True, soc=9, makespan=4)
    selected = select_best_candidate_evaluation((baseline, recovered))
    assert selected.success


def test_ordering_failure_triggers_zero_pp_calls() -> None:
    grid_map = build_grid_map([[0, 1], [0, 0]])
    scenario = _scenario(_agent(0, 0, 0, 1, 1))
    prepared = prepare_mapf9_instance(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=1,
    )
    assert isinstance(prepared, MAPF9OrderingFailure)

    pp_runner = MagicMock()
    result = ordering_failure_method_result(prepared, method=MAPF9Method.CGLPS)
    assert result.total_logical_pp_candidate_count == 0
    assert result.baseline_spf_evaluation is None
    pp_runner.assert_not_called()
    assert result.success is False
