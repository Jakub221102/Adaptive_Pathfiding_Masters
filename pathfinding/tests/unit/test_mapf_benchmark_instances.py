from __future__ import annotations

import json
from pathlib import Path

import pytest

from pathfinding.src.algorithms.mapf.conflicts import detect_conflicts
from pathfinding.src.algorithms.mapf.demo_scenario import build_mapf_scenario_from_indices
from pathfinding.src.algorithms.mapf.metrics import makespan, sum_of_costs
from pathfinding.src.algorithms.mapf.models import (
    AgentPath,
    EdgeConflict,
    MAPFAgent,
    TimedState,
    VertexConflict,
)
from pathfinding.src.experiments.mapf_independent_static_path import (
    evaluate_candidate_with_independent_paths,
    find_independent_static_path,
    find_independent_static_path_bounded_result,
    independent_path_fits_horizon,
)
from pathfinding.src.algorithms.mapf.space_time_astar import find_path
from pathfinding.src.core.models import GridMap, Position, Scenario
from pathfinding.src.experiments.mapf_benchmark_instances import (
    MAPFBenchmarkInstance,
    MAPFBenchmarkManifest,
    MAPFInteractionLevel,
    _candidate_signature,
    _eligible_scenario_indices,
    _evaluate_candidate,
    _validate_candidate_indices,
    agent_path_from_precomputed,
    build_precomputed_path_lookup,
    classify_interaction_level,
    count_conflicting_agent_pairs,
    evaluate_benchmark_candidate,
    generate_benchmark_instances,
    generate_benchmark_instances_from_precomputed,
    generate_benchmark_instances_legacy,
    generate_benchmark_manifest,
    load_benchmark_manifest,
    precompute_independent_paths,
    prepare_benchmark_source_pool,
    reconstruct_mapf_scenario_from_instance,
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
    map_name: str = "test.map",
    width: int = 20,
    height: int = 20,
) -> Scenario:
    return Scenario(
        map_name=map_name,
        width=width,
        height=height,
        start=Position(row=start_row, col=start_col),
        goal=Position(row=goal_row, col=goal_col),
        optimal_length=optimal_length,
    )


def _open_grid(size: int = 20) -> GridMap:
    return build_grid_map(
        [[0 for _ in range(size)] for _ in range(size)],
        name="bench.map",
    )


def _build_rich_scenario_pool(size: int = 9) -> tuple[GridMap, list[Scenario]]:
    """Synthetic pool with parallel and crossing paths for all interaction levels."""
    grid_map = build_grid_map(
        [[0 for _ in range(size)] for _ in range(size)],
        name="bench.map",
    )
    scenarios: list[Scenario] = []
    center = size // 2

    for row in range(size):
        scenarios.append(
            _scenario(
                row, 0, row, size - 1,
                optimal_length=25.0 + row,
                width=size,
                height=size,
            )
        )
        scenarios.append(
            _scenario(
                row, size - 1, row, 0,
                optimal_length=26.0 + row,
                width=size,
                height=size,
            )
        )

    for col in range(size):
        scenarios.append(
            _scenario(
                0, col, size - 1, col,
                optimal_length=35.0 + col,
                width=size,
                height=size,
            )
        )
        scenarios.append(
            _scenario(
                size - 1, col, 0, col,
                optimal_length=36.0 + col,
                width=size,
                height=size,
            )
        )

    for offset in range(3):
        scenarios.append(
            _scenario(
                0, offset, size - 1, size - 1 - offset,
                optimal_length=40.0 + offset,
                width=size,
                height=size,
            )
        )
        scenarios.append(
            _scenario(
                size - 1, offset, 0, size - 1 - offset,
                optimal_length=45.0 + offset,
                width=size,
                height=size,
            )
        )
        scenarios.append(
            _scenario(
                offset, 0, size - 1 - offset, size - 1,
                optimal_length=50.0 + offset,
                width=size,
                height=size,
            )
        )
        scenarios.append(
            _scenario(
                center,
                offset,
                center,
                size - 1 - offset,
                optimal_length=55.0 + offset,
                width=size,
                height=size,
            )
        )
        scenarios.append(
            _scenario(
                offset,
                center,
                size - 1 - offset,
                center,
                optimal_length=60.0 + offset,
                width=size,
                height=size,
            )
        )

    return grid_map, scenarios


def _build_crossing_scenario_pool(size: int = 20) -> tuple[GridMap, list[Scenario]]:
    return _build_rich_scenario_pool(size=9)


@pytest.mark.parametrize(
    ("conflict_count", "expected_level"),
    [
        (0, MAPFInteractionLevel.LOW),
        (1, MAPFInteractionLevel.MEDIUM),
        (2, MAPFInteractionLevel.MEDIUM),
        (3, MAPFInteractionLevel.HIGH),
        (7, MAPFInteractionLevel.HIGH),
    ],
)
def test_classify_interaction_level(
    conflict_count: int,
    expected_level: MAPFInteractionLevel,
) -> None:
    assert classify_interaction_level(conflict_count) == expected_level


def test_count_conflicting_agent_pairs_with_multiple_conflicts_per_pair() -> None:
    conflicts = (
        VertexConflict(agent1_id=0, agent2_id=3, row=1, col=1, timestep=1),
        VertexConflict(agent1_id=0, agent2_id=3, row=2, col=2, timestep=2),
        EdgeConflict(
            agent1_id=2,
            agent2_id=7,
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

    assert len(conflicts) == 3
    assert count_conflicting_agent_pairs(conflicts) == 2


def test_candidate_signature_is_order_independent() -> None:
    assert _candidate_signature((1, 4, 7)) == _candidate_signature((7, 1, 4))


def test_deterministic_generation_with_same_seed() -> None:
    grid_map, scenarios = _build_crossing_scenario_pool()

    result_a = generate_benchmark_instances(
        grid_map=grid_map,
        scenarios=scenarios,
        agent_count=3,
        instances_per_level=1,
        max_timestep=64,
        seed=42,
        min_reference_length=20.0,
        max_attempts=5000,
    )
    result_b = generate_benchmark_instances(
        grid_map=grid_map,
        scenarios=scenarios,
        agent_count=3,
        instances_per_level=1,
        max_timestep=64,
        seed=42,
        min_reference_length=20.0,
        max_attempts=5000,
    )

    assert result_a.instances == result_b.instances


def test_different_seed_can_produce_different_instances() -> None:
    grid_map, scenarios = _build_crossing_scenario_pool()

    result_a = generate_benchmark_instances(
        grid_map=grid_map,
        scenarios=scenarios,
        agent_count=3,
        instances_per_level=1,
        max_timestep=64,
        seed=1,
        min_reference_length=20.0,
        max_attempts=5000,
    )
    result_b = generate_benchmark_instances(
        grid_map=grid_map,
        scenarios=scenarios,
        agent_count=3,
        instances_per_level=1,
        max_timestep=64,
        seed=999,
        min_reference_length=20.0,
        max_attempts=5000,
    )

    assert result_a.instances != result_b.instances


def test_per_agent_count_seed_independence() -> None:
    grid_map, scenarios = _build_crossing_scenario_pool()
    config = {
        "instances_per_level": 1,
        "max_timestep": 64,
        "seed": 2026,
        "min_reference_length": 20.0,
        "max_attempts": 5000,
    }

    only_eight = generate_benchmark_instances(
        grid_map=grid_map,
        scenarios=scenarios,
        agent_count=8,
        **config,
    ).instances

    generate_benchmark_instances(
        grid_map=grid_map,
        scenarios=scenarios,
        agent_count=5,
        **config,
    )
    mixed_eight = generate_benchmark_instances(
        grid_map=grid_map,
        scenarios=scenarios,
        agent_count=8,
        **config,
    ).instances
    generate_benchmark_instances(
        grid_map=grid_map,
        scenarios=scenarios,
        agent_count=6,
        **config,
    )

    assert only_eight == mixed_eight


def test_accepted_instances_have_valid_candidates_and_feasible_paths() -> None:
    grid_map, scenarios = _build_crossing_scenario_pool()

    result = generate_benchmark_instances(
        grid_map=grid_map,
        scenarios=scenarios,
        agent_count=3,
        instances_per_level=1,
        max_timestep=64,
        seed=7,
        min_reference_length=20.0,
        max_attempts=5000,
    )

    for instance in result.instances:
        assert len(instance.scenario_indices) == instance.agent_count
        assert len(set(instance.scenario_indices)) == instance.agent_count
        assert _validate_candidate_indices(
            instance.scenario_indices,
            scenarios,
            grid_map,
        )

        selection = build_mapf_scenario_from_indices(
            scenarios=scenarios,
            grid_map=grid_map,
            scenario_indices=instance.scenario_indices,
        )
        starts = {
            (agent.start.row, agent.start.col) for agent in selection.scenario.agents
        }
        goals = {
            (agent.goal.row, agent.goal.col) for agent in selection.scenario.agents
        }
        assert len(starts) == instance.agent_count
        assert len(goals) == instance.agent_count

        for agent in selection.scenario.agents:
            path = find_path(
                grid_map=grid_map,
                agent=agent,
                max_timestep=64,
                constraints=(),
            )
            assert path is not None


def test_agent_ordering_maps_to_zero_based_agent_ids() -> None:
    grid_map, scenarios = _build_rich_scenario_pool()
    scenario_indices = (17, 4, 22)

    selection = build_mapf_scenario_from_indices(
        scenarios=scenarios,
        grid_map=grid_map,
        scenario_indices=scenario_indices,
    )

    assert [spec.agent_id for spec in selection.agents] == [0, 1, 2]
    assert [spec.scenario_index for spec in selection.agents] == list(scenario_indices)


def test_generator_rejects_duplicate_scenario_sets_in_different_order() -> None:
    grid_map, scenarios = _build_crossing_scenario_pool()

    result = generate_benchmark_instances(
        grid_map=grid_map,
        scenarios=scenarios,
        agent_count=3,
        instances_per_level=2,
        max_timestep=64,
        seed=12345,
        min_reference_length=20.0,
        max_attempts=10000,
    )

    signatures = [
        _candidate_signature(instance.scenario_indices)
        for instance in result.instances
    ]
    assert len(signatures) == len(set(signatures))


def test_stored_interaction_metadata_matches_recomputation() -> None:
    grid_map, scenarios = _build_crossing_scenario_pool()

    result = generate_benchmark_instances(
        grid_map=grid_map,
        scenarios=scenarios,
        agent_count=3,
        instances_per_level=1,
        max_timestep=64,
        seed=11,
        min_reference_length=20.0,
        max_attempts=5000,
    )

    for instance in result.instances:
        selection = reconstruct_mapf_scenario_from_instance(
            scenarios=scenarios,
            grid_map=grid_map,
            instance=instance,
        )
        independent_paths: list[AgentPath] = []
        for agent in selection.scenario.agents:
            path = find_path(
                grid_map=grid_map,
                agent=agent,
                max_timestep=64,
                constraints=(),
            )
            assert path is not None
            independent_paths.append(path)

        conflicts = detect_conflicts(tuple(independent_paths))
        vertex_count = sum(
            1 for conflict in conflicts if isinstance(conflict, VertexConflict)
        )
        edge_count = sum(
            1 for conflict in conflicts if isinstance(conflict, EdgeConflict)
        )

        assert instance.independent_conflict_count == len(conflicts)
        assert instance.conflicting_agent_pair_count == count_conflicting_agent_pairs(
            conflicts
        )
        assert instance.vertex_conflict_count == vertex_count
        assert instance.edge_conflict_count == edge_count
        assert instance.independent_soc == sum_of_costs(independent_paths)
        assert instance.independent_makespan == makespan(independent_paths)
        assert instance.interaction_level == classify_interaction_level(len(conflicts))


def test_reference_length_filter_excludes_short_scenarios() -> None:
    grid_map = _open_grid(10)
    scenarios = [
        _scenario(0, 0, 0, 9, optimal_length=5.0),
        _scenario(1, 0, 1, 9, optimal_length=25.0),
    ]

    eligible = _eligible_scenario_indices(
        scenarios=scenarios,
        grid_map=grid_map,
        min_reference_length=20.0,
    )

    assert eligible == (1,)


def test_infeasible_independent_path_rejects_candidate() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
        ],
        name="blocked.map",
    )
    scenarios = [
        _scenario(0, 0, 2, 4, optimal_length=25.0, width=5, height=3),
        _scenario(0, 4, 2, 0, optimal_length=25.0, width=5, height=3),
    ]

    evaluation = _evaluate_candidate(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_indices=(0, 1),
        max_timestep=2,
        path_cache={},
    )

    assert evaluation is None


def test_insufficient_category_raises_clear_error() -> None:
    grid_map = _open_grid(30)
    scenarios = [
        _scenario(0, 0, 0, 29, optimal_length=30.0, width=30, height=30),
        _scenario(1, 0, 1, 29, optimal_length=31.0, width=30, height=30),
        _scenario(2, 0, 2, 29, optimal_length=32.0, width=30, height=30),
        _scenario(3, 0, 3, 29, optimal_length=33.0, width=30, height=30),
    ]

    with pytest.raises(ValueError, match="agent_count=2") as error_info:
        generate_benchmark_instances(
            grid_map=grid_map,
            scenarios=scenarios,
            agent_count=2,
            instances_per_level=1,
            max_timestep=64,
            seed=1,
            min_reference_length=20.0,
            max_attempts=200,
        )

    message = str(error_info.value)
    assert "found LOW=" in message
    assert "MEDIUM=0" in message
    assert "HIGH=0" in message
    assert "attempts" in message


def test_manifest_json_round_trip(tmp_path: Path) -> None:
    instance = MAPFBenchmarkInstance(
        instance_id="bench_n03_low_000",
        agent_count=3,
        interaction_level=MAPFInteractionLevel.LOW,
        scenario_indices=(1, 4, 7),
        independent_conflict_count=0,
        conflicting_agent_pair_count=0,
        independent_soc=90,
        independent_makespan=35,
        vertex_conflict_count=0,
        edge_conflict_count=0,
    )
    manifest = MAPFBenchmarkManifest(
        map_name="bench.map",
        scenario_name="bench.map.scen",
        seed=2026,
        max_timestep=512,
        min_reference_length=20.0,
        instances=(instance,),
    )

    output_path = tmp_path / "manifest.json"
    save_benchmark_manifest(manifest, output_path)
    loaded = load_benchmark_manifest(output_path)

    assert loaded == manifest
    assert loaded.instances[0].scenario_indices == (1, 4, 7)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["instances"][0]["scenario_indices"] == [1, 4, 7]


def test_negative_max_timestep_raises() -> None:
    grid_map, scenarios = _build_crossing_scenario_pool()

    with pytest.raises(ValueError, match="max_timestep must be non-negative"):
        generate_benchmark_instances(
            grid_map=grid_map,
            scenarios=scenarios,
            agent_count=2,
            instances_per_level=1,
            max_timestep=-1,
            seed=1,
            min_reference_length=20.0,
            max_attempts=100,
        )


def test_instance_ids_contain_agent_count_level_and_sequence() -> None:
    grid_map, scenarios = _build_crossing_scenario_pool()

    result = generate_benchmark_instances(
        grid_map=grid_map,
        scenarios=scenarios,
        agent_count=3,
        instances_per_level=1,
        max_timestep=64,
        seed=5,
        min_reference_length=20.0,
        max_attempts=5000,
    )

    for instance in result.instances:
        assert "_n03_" in instance.instance_id
        assert instance.interaction_level.value in instance.instance_id
        assert instance.instance_id.endswith("_000")


def test_precompute_preserves_input_order() -> None:
    grid_map, scenarios = _build_crossing_scenario_pool()
    indices = (3, 7, 11)

    result = precompute_independent_paths(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_indices=indices,
        max_timestep=64,
    )

    assert [path.scenario_index for path in result.paths] == list(indices)
    assert result.failed_scenario_indices == ()


def test_precompute_plans_each_source_scenario_once() -> None:
    grid_map, scenarios = _build_crossing_scenario_pool()
    indices = (2, 2, 5)
    plan_calls = 0
    original_bounded = find_independent_static_path_bounded_result

    def counting_bounded(*args, **kwargs):
        nonlocal plan_calls
        plan_calls += 1
        return original_bounded(*args, **kwargs)

    import pathfinding.src.experiments.mapf_independent_static_path as static_module

    static_module.find_independent_static_path_bounded_result = counting_bounded
    try:
        result = precompute_independent_paths(
            grid_map=grid_map,
            scenarios=scenarios,
            scenario_indices=indices,
            max_timestep=64,
        )
    finally:
        static_module.find_independent_static_path_bounded_result = original_bounded

    assert [path.scenario_index for path in result.paths] == [2, 2, 5]
    assert result.paths[0].states == result.paths[1].states
    assert plan_calls == 2


def test_precomputed_states_reconstruct_with_different_agent_ids() -> None:
    grid_map, scenarios = _build_crossing_scenario_pool()
    result = precompute_independent_paths(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_indices=(4,),
        max_timestep=64,
    )
    precomputed = result.paths[0]

    path_a = agent_path_from_precomputed(precomputed, agent_id=0)
    path_b = agent_path_from_precomputed(precomputed, agent_id=7)

    assert path_a.states == path_b.states
    assert path_a.agent_id == 0
    assert path_b.agent_id == 7


def test_precomputed_candidate_evaluation_matches_static_direct_evaluation() -> None:
    grid_map, scenarios = _build_crossing_scenario_pool()
    scenario_indices = (1, 4, 8)

    precomputed = precompute_independent_paths(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_indices=scenario_indices,
        max_timestep=64,
    )
    lookup = build_precomputed_path_lookup(precomputed.paths)

    static_paths = []
    for agent_id, scenario_index in enumerate(scenario_indices):
        scenario = scenarios[scenario_index]
        path = find_independent_static_path(
            grid_map,
            MAPFAgent(
                agent_id=agent_id,
                start=scenario.start,
                goal=scenario.goal,
            ),
        )
        assert path is not None
        static_paths.append(path)

    direct = evaluate_candidate_with_independent_paths(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_indices=scenario_indices,
        independent_paths=static_paths,
    )
    cached = evaluate_benchmark_candidate(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_indices=scenario_indices,
        max_timestep=64,
        precomputed_lookup=lookup,
    )

    assert direct is not None
    assert cached is not None
    assert direct == cached


def test_precomputed_lookup_uses_scenario_index_not_agent_id() -> None:
    grid_map, scenarios = _build_crossing_scenario_pool()
    precomputed = precompute_independent_paths(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_indices=(5, 6),
        max_timestep=64,
    )
    lookup = build_precomputed_path_lookup(precomputed.paths)

    evaluation = evaluate_benchmark_candidate(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_indices=(6, 5),
        max_timestep=64,
        precomputed_lookup=lookup,
    )

    assert evaluation is not None
    assert evaluation.scenario_indices == (6, 5)


def test_repeated_precompute_is_deterministic() -> None:
    grid_map, scenarios = _build_crossing_scenario_pool()
    indices = (1, 3, 5)

    first = precompute_independent_paths(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_indices=indices,
        max_timestep=64,
    )
    second = precompute_independent_paths(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_indices=indices,
        max_timestep=64,
    )

    assert first == second


def test_precompute_negative_horizon_raises() -> None:
    grid_map, scenarios = _build_crossing_scenario_pool()

    with pytest.raises(ValueError, match="max_timestep must be non-negative"):
        precompute_independent_paths(
            grid_map=grid_map,
            scenarios=scenarios,
            scenario_indices=(0,),
            max_timestep=-1,
        )


def test_precompute_records_failed_scenario_indices() -> None:
    grid_map = build_grid_map(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
        ],
        name="blocked.map",
    )
    scenarios = [
        _scenario(0, 0, 2, 4, optimal_length=25.0, width=5, height=3),
    ]

    result = precompute_independent_paths(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_indices=(0, 999),
        max_timestep=2,
    )

    assert result.paths == ()
    assert 0 in result.failed_scenario_indices
    assert 999 in result.failed_scenario_indices


def test_precomputed_generation_is_deterministic_with_static_precompute() -> None:
    grid_map, scenarios = _build_crossing_scenario_pool()
    config = {
        "agent_count": 3,
        "instances_per_level": 1,
        "max_timestep": 64,
        "seed": 42,
        "min_reference_length": 20.0,
        "max_attempts": 5000,
    }

    first = generate_benchmark_instances(
        grid_map=grid_map,
        scenarios=scenarios,
        **config,
    )
    second = generate_benchmark_instances(
        grid_map=grid_map,
        scenarios=scenarios,
        **config,
    )

    assert first == second


def test_sampling_performs_zero_pathfinding_calls_after_precompute() -> None:
    grid_map, scenarios = _build_crossing_scenario_pool()
    source_pool = prepare_benchmark_source_pool(
        grid_map=grid_map,
        scenarios=scenarios,
        min_reference_length=20.0,
        max_timestep=64,
    )

    import pathfinding.src.experiments.mapf_benchmark_instances as benchmark_module
    import pathfinding.src.experiments.mapf_independent_static_path as static_module

    def raising_find_path(*args, **kwargs):
        raise AssertionError("find_path must not be called during candidate sampling")

    def raising_static_path(*args, **kwargs):
        raise AssertionError(
            "find_independent_static_path must not be called during candidate sampling"
        )

    def raising_bounded_static_path(*args, **kwargs):
        raise AssertionError(
            "find_independent_static_path_bounded must not be called during candidate sampling"
        )

    original_find_path = benchmark_module.find_path
    original_static_path = static_module.find_independent_static_path
    original_bounded_static_path = static_module.find_independent_static_path_bounded
    benchmark_module.find_path = raising_find_path
    static_module.find_independent_static_path = raising_static_path
    static_module.find_independent_static_path_bounded = raising_bounded_static_path
    try:
        result = generate_benchmark_instances_from_precomputed(
            grid_map=grid_map,
            scenarios=scenarios,
            agent_count=3,
            instances_per_level=1,
            max_timestep=64,
            seed=7,
            max_attempts=5000,
            source_pool=source_pool,
        )
    finally:
        benchmark_module.find_path = original_find_path
        static_module.find_independent_static_path = original_static_path
        static_module.find_independent_static_path_bounded = original_bounded_static_path

    assert len(result.instances) == 3


def test_full_pool_precomputed_once_per_generation_call() -> None:
    grid_map, scenarios = _build_crossing_scenario_pool()
    plan_calls = 0
    original_bounded = find_independent_static_path_bounded_result

    def counting_bounded(*args, **kwargs):
        nonlocal plan_calls
        plan_calls += 1
        return original_bounded(*args, **kwargs)

    import pathfinding.src.experiments.mapf_independent_static_path as static_module

    static_module.find_independent_static_path_bounded_result = counting_bounded
    try:
        source_pool = prepare_benchmark_source_pool(
            grid_map=grid_map,
            scenarios=scenarios,
            min_reference_length=20.0,
            max_timestep=64,
        )
        precompute_calls = plan_calls
        plan_calls = 0

        generate_benchmark_instances_from_precomputed(
            grid_map=grid_map,
            scenarios=scenarios,
            agent_count=3,
            instances_per_level=1,
            max_timestep=64,
            seed=11,
            max_attempts=5000,
            source_pool=source_pool,
        )
    finally:
        static_module.find_independent_static_path_bounded_result = original_bounded

    assert plan_calls == 0
    assert precompute_calls == len(set(source_pool.eligible_indices))


def test_interaction_levels_share_one_precompute() -> None:
    grid_map, scenarios = _build_crossing_scenario_pool()
    precompute_calls = 0
    original_precompute = precompute_independent_paths

    def counting_precompute(*args, **kwargs):
        nonlocal precompute_calls
        precompute_calls += 1
        return original_precompute(*args, **kwargs)

    import pathfinding.src.experiments.mapf_benchmark_instances as benchmark_module

    benchmark_module.precompute_independent_paths = counting_precompute
    try:
        generate_benchmark_instances(
            grid_map=grid_map,
            scenarios=scenarios,
            agent_count=3,
            instances_per_level=1,
            max_timestep=64,
            seed=13,
            min_reference_length=20.0,
            max_attempts=5000,
        )
    finally:
        benchmark_module.precompute_independent_paths = original_precompute

    assert precompute_calls == 1


def test_manifest_reuses_source_precompute_across_agent_counts() -> None:
    grid_map, scenarios = _build_crossing_scenario_pool()
    precompute_calls = 0
    original_precompute = precompute_independent_paths

    def counting_precompute(*args, **kwargs):
        nonlocal precompute_calls
        precompute_calls += 1
        return original_precompute(*args, **kwargs)

    import pathfinding.src.experiments.mapf_benchmark_instances as benchmark_module

    benchmark_module.precompute_independent_paths = counting_precompute
    try:
        result = generate_benchmark_manifest(
            grid_map=grid_map,
            scenarios=scenarios,
            scenario_name="bench.map.scen",
            agent_counts=(3, 4),
            instances_per_level=1,
            max_timestep=64,
            seed=2026,
            min_reference_length=20.0,
            max_attempts=5000,
        )
    finally:
        benchmark_module.precompute_independent_paths = original_precompute

    assert precompute_calls == 1
    assert len(result.manifest.instances) == 6


def test_progress_callbacks_do_not_affect_generated_instances() -> None:
    grid_map, scenarios = _build_crossing_scenario_pool()
    config = {
        "grid_map": grid_map,
        "scenarios": scenarios,
        "agent_count": 3,
        "instances_per_level": 1,
        "max_timestep": 64,
        "seed": 17,
        "min_reference_length": 20.0,
        "max_attempts": 5000,
    }

    silent = generate_benchmark_instances(**config)
    noisy = generate_benchmark_instances(
        **config,
        precompute_progress_every=1,
        precompute_progress_callback=lambda *_args: None,
        sampling_progress_every=1,
        sampling_progress_callback=lambda *_args: None,
        accepted_instance_callback=lambda _instance, _attempt: None,
    )

    assert silent.instances == noisy.instances


def test_precompute_horizon_includes_cost_equal_to_max_timestep() -> None:
    grid_map = build_grid_map(
        [[0, 0, 0, 0, 0] for _ in range(3)],
        name="exact_horizon.map",
    )
    scenarios = [
        _scenario(1, 0, 1, 4, optimal_length=25.0, width=5, height=3),
    ]

    result = precompute_independent_paths(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_indices=(0,),
        max_timestep=4,
    )

    assert len(result.paths) == 1
    assert result.over_horizon_count == 0
    assert len(result.paths[0].states) - 1 == 4


def test_precompute_horizon_excludes_cost_above_max_timestep() -> None:
    grid_map = build_grid_map(
        [[0, 0, 0, 0, 0] for _ in range(3)],
        name="over_horizon.map",
    )
    scenarios = [
        _scenario(1, 0, 1, 4, optimal_length=25.0, width=5, height=3),
    ]

    result = precompute_independent_paths(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_indices=(0,),
        max_timestep=3,
    )

    assert result.paths == ()
    assert 0 in result.failed_scenario_indices
    assert result.over_horizon_count == 1
    assert result.no_spatial_path_count == 0


def test_production_precompute_does_not_call_space_time_astar() -> None:
    grid_map, scenarios = _build_crossing_scenario_pool()

    import pathfinding.src.experiments.mapf_benchmark_instances as benchmark_module

    def raising_find_path(*args, **kwargs):
        raise AssertionError(
            "Space-Time A* must not be used for benchmark source precompute"
        )

    original_find_path = benchmark_module.find_path
    benchmark_module.find_path = raising_find_path
    try:
        result = prepare_benchmark_source_pool(
            grid_map=grid_map,
            scenarios=scenarios,
            min_reference_length=20.0,
            max_timestep=64,
        )
        generate_benchmark_instances_from_precomputed(
            grid_map=grid_map,
            scenarios=scenarios,
            agent_count=3,
            instances_per_level=1,
            max_timestep=64,
            seed=3,
            max_attempts=5000,
            source_pool=result,
        )
    finally:
        benchmark_module.find_path = original_find_path


def _reference_unbounded_classification(
    grid_map: GridMap,
    scenarios: list[Scenario],
    scenario_indices: tuple[int, ...],
    max_timestep: int,
) -> tuple[set[int], set[int], int, int]:
    feasible: set[int] = set()
    failed: set[int] = set()
    no_spatial_path_count = 0
    over_horizon_count = 0

    for scenario_index in scenario_indices:
        scenario = scenarios[scenario_index]
        agent = MAPFAgent(
            agent_id=0,
            start=scenario.start,
            goal=scenario.goal,
        )
        path = find_independent_static_path(grid_map=grid_map, agent=agent)
        if path is None:
            failed.add(scenario_index)
            no_spatial_path_count += 1
        elif not independent_path_fits_horizon(path, max_timestep):
            failed.add(scenario_index)
            over_horizon_count += 1
        else:
            feasible.add(scenario_index)

    return feasible, failed, no_spatial_path_count, over_horizon_count


def test_precompute_classifies_over_horizon_not_no_path() -> None:
    grid_map = build_grid_map([[0, 0, 0, 0, 0]], name="line.map")
    scenarios = [
        _scenario(0, 0, 0, 4, optimal_length=25.0, width=5, height=1),
    ]

    result = precompute_independent_paths(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_indices=(0,),
        max_timestep=3,
    )

    assert result.paths == ()
    assert result.over_horizon_count == 1
    assert result.no_spatial_path_count == 0
    assert result.spatially_reachable_count == 1


def test_precompute_classifies_no_spatial_path_not_over_horizon() -> None:
    grid_map = build_grid_map(
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ],
        name="split.map",
    )
    scenarios = [
        _scenario(0, 0, 0, 2, optimal_length=25.0, width=3, height=3),
    ]

    result = precompute_independent_paths(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_indices=(0,),
        max_timestep=64,
    )

    assert result.paths == ()
    assert result.no_spatial_path_count == 1
    assert result.over_horizon_count == 0
    assert result.spatially_reachable_count == 0


def test_bounded_precompute_matches_unbounded_reference_classification() -> None:
    grid_map, scenarios = _build_crossing_scenario_pool()
    eligible = _eligible_scenario_indices(
        scenarios=scenarios,
        grid_map=grid_map,
        min_reference_length=20.0,
    )
    max_timestep = 64

    ref_feasible, ref_failed, ref_no_path, ref_over = _reference_unbounded_classification(
        grid_map,
        scenarios,
        tuple(eligible),
        max_timestep,
    )
    bounded = precompute_independent_paths(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_indices=tuple(eligible),
        max_timestep=max_timestep,
    )

    bounded_feasible = {path.scenario_index for path in bounded.paths}
    bounded_failed = set(bounded.failed_scenario_indices)

    assert bounded_feasible == ref_feasible
    assert bounded_failed == ref_failed
    assert bounded.no_spatial_path_count == ref_no_path
    assert bounded.over_horizon_count == ref_over


def test_bounded_precompute_matches_unbounded_reference_paths() -> None:
    grid_map, scenarios = _build_crossing_scenario_pool()
    eligible = _eligible_scenario_indices(
        scenarios=scenarios,
        grid_map=grid_map,
        min_reference_length=20.0,
    )
    max_timestep = 64

    bounded = precompute_independent_paths(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_indices=tuple(eligible),
        max_timestep=max_timestep,
    )
    lookup = build_precomputed_path_lookup(bounded.paths)

    for scenario_index in lookup:
        scenario = scenarios[scenario_index]
        reference = find_independent_static_path(
            grid_map,
            MAPFAgent(agent_id=0, start=scenario.start, goal=scenario.goal),
        )
        assert reference is not None
        assert lookup[scenario_index].states == reference.states


def test_accepted_instance_callback_receives_attempt_number() -> None:
    grid_map, scenarios = _build_crossing_scenario_pool()
    source_pool = prepare_benchmark_source_pool(
        grid_map=grid_map,
        scenarios=scenarios,
        min_reference_length=20.0,
        max_timestep=64,
    )
    attempts_seen: list[int] = []

    def record_attempt(_instance: MAPFBenchmarkInstance, attempt: int) -> None:
        attempts_seen.append(attempt)

    result = generate_benchmark_instances_from_precomputed(
        grid_map=grid_map,
        scenarios=scenarios,
        agent_count=3,
        instances_per_level=1,
        max_timestep=64,
        seed=5,
        max_attempts=5000,
        source_pool=source_pool,
        accepted_instance_callback=record_attempt,
    )

    assert len(result.instances) == 3
    assert len(attempts_seen) == 3
    assert all(attempt > 0 for attempt in attempts_seen)
    assert max(attempts_seen) <= result.attempts
