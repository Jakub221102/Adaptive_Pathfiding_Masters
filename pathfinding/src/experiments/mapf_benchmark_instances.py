from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pathfinding.src.algorithms.mapf.conflicts import detect_conflicts
from pathfinding.src.algorithms.mapf.demo_scenario import build_mapf_scenario_from_indices
from pathfinding.src.algorithms.mapf.metrics import makespan, sum_of_costs
from pathfinding.src.algorithms.mapf.models import (
    AgentPath,
    Conflict,
    EdgeConflict,
    MAPFAgent,
    TimedState,
    VertexConflict,
)
from pathfinding.src.algorithms.mapf.smoke_scenario import _is_valid_agent_candidate
from pathfinding.src.algorithms.mapf.space_time_astar import find_path
from pathfinding.src.core.models import GridMap, Scenario


class MAPFInteractionLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Benchmark-stratification thresholds based on independent conflict count.
# These are explicit and easy to modify; do not silently change by agent count.
INTERACTION_LOW_MAX_CONFLICTS = 0
INTERACTION_MEDIUM_MIN_CONFLICTS = 1
INTERACTION_MEDIUM_MAX_CONFLICTS = 2
INTERACTION_HIGH_MIN_CONFLICTS = 3


@dataclass(frozen=True, slots=True)
class MAPFBenchmarkInstance:
    instance_id: str
    agent_count: int
    interaction_level: MAPFInteractionLevel
    scenario_indices: tuple[int, ...]

    independent_conflict_count: int
    conflicting_agent_pair_count: int

    independent_soc: int
    independent_makespan: int

    vertex_conflict_count: int
    edge_conflict_count: int


@dataclass(frozen=True, slots=True)
class MAPFBenchmarkManifest:
    map_name: str
    scenario_name: str

    seed: int
    max_timestep: int
    min_reference_length: float

    instances: tuple[MAPFBenchmarkInstance, ...]

    interaction_low_max_conflicts: int = INTERACTION_LOW_MAX_CONFLICTS
    interaction_medium_min_conflicts: int = INTERACTION_MEDIUM_MIN_CONFLICTS
    interaction_medium_max_conflicts: int = INTERACTION_MEDIUM_MAX_CONFLICTS
    interaction_high_min_conflicts: int = INTERACTION_HIGH_MIN_CONFLICTS


@dataclass(frozen=True, slots=True)
class MAPFBenchmarkGenerationResult:
    instances: tuple[MAPFBenchmarkInstance, ...]
    attempts: int


@dataclass(frozen=True, slots=True)
class MAPFPrecomputedSourcePath:
    scenario_index: int
    states: tuple[TimedState, ...]


@dataclass(frozen=True, slots=True)
class MAPFPrecomputePathsResult:
    paths: tuple[MAPFPrecomputedSourcePath, ...]
    failed_scenario_indices: tuple[int, ...]
    no_spatial_path_count: int = 0
    over_horizon_count: int = 0
    spatial_paths_found: int = 0


@dataclass(frozen=True, slots=True)
class StaticPrecomputeProgress:
    completed: int
    total: int
    feasible: int
    no_spatial_path: int
    over_horizon: int
    spatial_paths_found: int


@dataclass(frozen=True, slots=True)
class MAPFBenchmarkSourcePool:
    eligible_indices: tuple[int, ...]
    precompute_result: MAPFPrecomputePathsResult
    precomputed_lookup: dict[int, MAPFPrecomputedSourcePath]
    feasible_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MAPFBenchmarkManifestGenerationResult:
    manifest: MAPFBenchmarkManifest
    attempts_by_agent_count: dict[int, int]
    source_pool: MAPFBenchmarkSourcePool


def agent_path_from_precomputed(
    precomputed: MAPFPrecomputedSourcePath,
    agent_id: int,
) -> AgentPath:
    return AgentPath(agent_id=agent_id, states=precomputed.states)


def build_precomputed_path_lookup(
    paths: Sequence[MAPFPrecomputedSourcePath],
) -> dict[int, MAPFPrecomputedSourcePath]:
    lookup: dict[int, MAPFPrecomputedSourcePath] = {}
    for path in paths:
        if path.scenario_index in lookup:
            raise ValueError(
                f"duplicate precomputed path for scenario_index={path.scenario_index}"
            )
        lookup[path.scenario_index] = path
    return lookup


def precompute_independent_paths(
    grid_map: GridMap,
    scenarios: Sequence[Scenario],
    scenario_indices: Sequence[int],
    *,
    max_timestep: int,
    progress_every: int = 0,
    progress_callback: Callable[[StaticPrecomputeProgress], None] | None = None,
) -> MAPFPrecomputePathsResult:
    from pathfinding.src.experiments.mapf_independent_static_path import (
        find_independent_static_path,
        independent_path_fits_horizon,
    )

    if max_timestep < 0:
        raise ValueError("max_timestep must be non-negative")
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")

    succeeded: list[MAPFPrecomputedSourcePath] = []
    failed: list[int] = []
    total = len(scenario_indices)
    planned: dict[int, MAPFPrecomputedSourcePath | None] = {}
    no_spatial_path_count = 0
    over_horizon_count = 0
    spatial_paths_found = 0
    feasible_count = 0

    def _emit_progress(completed: int) -> None:
        if progress_callback is None:
            return
        progress_callback(
            StaticPrecomputeProgress(
                completed=completed,
                total=total,
                feasible=feasible_count,
                no_spatial_path=no_spatial_path_count,
                over_horizon=over_horizon_count,
                spatial_paths_found=spatial_paths_found,
            )
        )

    for completed, scenario_index in enumerate(scenario_indices, start=1):
        if scenario_index in planned:
            cached = planned[scenario_index]
            if cached is None:
                failed.append(scenario_index)
            else:
                succeeded.append(cached)
            if progress_every > 0 and completed % progress_every == 0:
                _emit_progress(completed)
            continue

        if scenario_index < 0 or scenario_index >= len(scenarios):
            planned[scenario_index] = None
            failed.append(scenario_index)
            if progress_every > 0 and completed % progress_every == 0:
                _emit_progress(completed)
            continue

        scenario = scenarios[scenario_index]
        agent = MAPFAgent(
            agent_id=0,
            start=scenario.start,
            goal=scenario.goal,
        )
        path = find_independent_static_path(grid_map=grid_map, agent=agent)
        if path is None:
            planned[scenario_index] = None
            failed.append(scenario_index)
            no_spatial_path_count += 1
        elif not independent_path_fits_horizon(path, max_timestep):
            planned[scenario_index] = None
            failed.append(scenario_index)
            spatial_paths_found += 1
            over_horizon_count += 1
        else:
            precomputed = MAPFPrecomputedSourcePath(
                scenario_index=scenario_index,
                states=path.states,
            )
            planned[scenario_index] = precomputed
            succeeded.append(precomputed)
            spatial_paths_found += 1
            feasible_count += 1

        if progress_every > 0 and completed % progress_every == 0:
            _emit_progress(completed)

    if total > 0:
        _emit_progress(total)

    return MAPFPrecomputePathsResult(
        paths=tuple(succeeded),
        failed_scenario_indices=tuple(failed),
        no_spatial_path_count=no_spatial_path_count,
        over_horizon_count=over_horizon_count,
        spatial_paths_found=spatial_paths_found,
    )


def classify_interaction_level(conflict_count: int) -> MAPFInteractionLevel:
    if conflict_count < 0:
        raise ValueError("conflict_count must be non-negative")

    if conflict_count <= INTERACTION_LOW_MAX_CONFLICTS:
        return MAPFInteractionLevel.LOW
    if (
        INTERACTION_MEDIUM_MIN_CONFLICTS
        <= conflict_count
        <= INTERACTION_MEDIUM_MAX_CONFLICTS
    ):
        return MAPFInteractionLevel.MEDIUM
    if conflict_count >= INTERACTION_HIGH_MIN_CONFLICTS:
        return MAPFInteractionLevel.HIGH

    raise ValueError(f"unexpected conflict_count: {conflict_count}")


def count_conflicting_agent_pairs(conflicts: Sequence[Conflict]) -> int:
    pairs: set[tuple[int, int]] = set()

    for conflict in conflicts:
        agent1_id = conflict.agent1_id
        agent2_id = conflict.agent2_id
        pairs.add((min(agent1_id, agent2_id), max(agent1_id, agent2_id)))

    return len(pairs)


def _count_conflict_types(conflicts: Sequence[Conflict]) -> tuple[int, int]:
    vertex_count = sum(1 for conflict in conflicts if isinstance(conflict, VertexConflict))
    edge_count = sum(1 for conflict in conflicts if isinstance(conflict, EdgeConflict))
    return vertex_count, edge_count


def _map_stem(grid_map: GridMap) -> str:
    return Path(grid_map.name).stem


def _make_instance_id(
    map_stem: str,
    agent_count: int,
    interaction_level: MAPFInteractionLevel,
    sequence: int,
) -> str:
    return f"{map_stem}_n{agent_count:02d}_{interaction_level.value}_{sequence:03d}"


def _candidate_signature(scenario_indices: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted(scenario_indices))


def eligible_scenario_indices(
    scenarios: Sequence[Scenario],
    grid_map: GridMap,
    min_reference_length: float,
) -> tuple[int, ...]:
    eligible: list[int] = []

    for index, scenario in enumerate(scenarios):
        if scenario.optimal_length is None:
            continue
        if scenario.optimal_length < min_reference_length:
            continue

        if not _is_valid_agent_candidate(
            scenario,
            grid_map,
            used_starts=set(),
            used_goals=set(),
        ):
            continue

        eligible.append(index)

    return tuple(eligible)


# Backward-compatible alias for scripts/tests that import the private name.
_eligible_scenario_indices = eligible_scenario_indices


def _feasible_scenario_indices(
    eligible_indices: Sequence[int],
    precomputed_lookup: Mapping[int, MAPFPrecomputedSourcePath],
) -> tuple[int, ...]:
    return tuple(
        scenario_index
        for scenario_index in eligible_indices
        if scenario_index in precomputed_lookup
    )


def prepare_benchmark_source_pool(
    grid_map: GridMap,
    scenarios: Sequence[Scenario],
    *,
    min_reference_length: float,
    max_timestep: int,
    precompute_progress_every: int = 0,
    precompute_progress_callback: Callable[[StaticPrecomputeProgress], None] | None = None,
) -> MAPFBenchmarkSourcePool:
    if max_timestep < 0:
        raise ValueError("max_timestep must be non-negative")
    if min_reference_length < 0:
        raise ValueError("min_reference_length must be non-negative")

    eligible_indices = eligible_scenario_indices(
        scenarios=scenarios,
        grid_map=grid_map,
        min_reference_length=min_reference_length,
    )
    precompute_result = precompute_independent_paths(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_indices=eligible_indices,
        max_timestep=max_timestep,
        progress_every=precompute_progress_every,
        progress_callback=precompute_progress_callback,
    )
    precomputed_lookup = build_precomputed_path_lookup(precompute_result.paths)
    feasible_indices = _feasible_scenario_indices(
        eligible_indices=eligible_indices,
        precomputed_lookup=precomputed_lookup,
    )
    return MAPFBenchmarkSourcePool(
        eligible_indices=eligible_indices,
        precompute_result=precompute_result,
        precomputed_lookup=precomputed_lookup,
        feasible_indices=feasible_indices,
    )


def _validate_candidate_indices(
    scenario_indices: Sequence[int],
    scenarios: Sequence[Scenario],
    grid_map: GridMap,
) -> bool:
    if len(scenario_indices) == 0:
        return False

    if len(set(scenario_indices)) != len(scenario_indices):
        return False

    used_starts: set[tuple[int, int]] = set()
    used_goals: set[tuple[int, int]] = set()

    for scenario_index in scenario_indices:
        if scenario_index < 0 or scenario_index >= len(scenarios):
            return False

        scenario = scenarios[scenario_index]
        if not _is_valid_agent_candidate(
            scenario,
            grid_map,
            used_starts,
            used_goals,
        ):
            return False

        used_starts.add((scenario.start.row, scenario.start.col))
        used_goals.add((scenario.goal.row, scenario.goal.col))

    return True


def _plan_independent_paths_from_precomputed(
    scenario_indices: Sequence[int],
    precomputed_lookup: Mapping[int, MAPFPrecomputedSourcePath],
) -> tuple[AgentPath, ...] | None:
    independent_paths: list[AgentPath] = []

    for agent_id, scenario_index in enumerate(scenario_indices):
        precomputed = precomputed_lookup.get(scenario_index)
        if precomputed is None:
            return None

        independent_paths.append(
            agent_path_from_precomputed(precomputed, agent_id=agent_id)
        )

    return tuple(independent_paths)


def _plan_cached_independent_paths(
    grid_map: GridMap,
    scenarios: Sequence[Scenario],
    scenario_indices: Sequence[int],
    max_timestep: int,
    path_cache: dict[int, tuple[TimedState, ...] | None],
) -> tuple[AgentPath, ...] | None:
    independent_paths: list[AgentPath] = []

    for agent_id, scenario_index in enumerate(scenario_indices):
        if scenario_index not in path_cache:
            scenario = scenarios[scenario_index]
            agent = MAPFAgent(
                agent_id=0,
                start=scenario.start,
                goal=scenario.goal,
            )
            path = find_path(
                grid_map=grid_map,
                agent=agent,
                max_timestep=max_timestep,
                constraints=(),
            )
            if path is None:
                path_cache[scenario_index] = None
            else:
                path_cache[scenario_index] = path.states

        cached_states = path_cache[scenario_index]
        if cached_states is None:
            return None

        independent_paths.append(
            AgentPath(agent_id=agent_id, states=cached_states)
        )

    return tuple(independent_paths)


def _evaluate_candidate(
    grid_map: GridMap,
    scenarios: Sequence[Scenario],
    scenario_indices: Sequence[int],
    max_timestep: int,
    path_cache: dict[int, tuple[TimedState, ...] | None],
    *,
    precomputed_lookup: Mapping[int, MAPFPrecomputedSourcePath] | None = None,
) -> MAPFBenchmarkInstance | None:
    if not _validate_candidate_indices(scenario_indices, scenarios, grid_map):
        return None

    if precomputed_lookup is not None:
        independent_paths = _plan_independent_paths_from_precomputed(
            scenario_indices=scenario_indices,
            precomputed_lookup=precomputed_lookup,
        )
    else:
        independent_paths = _plan_cached_independent_paths(
            grid_map=grid_map,
            scenarios=scenarios,
            scenario_indices=scenario_indices,
            max_timestep=max_timestep,
            path_cache=path_cache,
        )
    if independent_paths is None:
        return None

    conflicts = detect_conflicts(independent_paths)
    conflict_count = len(conflicts)
    interaction_level = classify_interaction_level(conflict_count)
    vertex_count, edge_count = _count_conflict_types(conflicts)

    indices = tuple(scenario_indices)
    return MAPFBenchmarkInstance(
        instance_id="",
        agent_count=len(indices),
        interaction_level=interaction_level,
        scenario_indices=indices,
        independent_conflict_count=conflict_count,
        conflicting_agent_pair_count=count_conflicting_agent_pairs(conflicts),
        independent_soc=sum_of_costs(independent_paths),
        independent_makespan=makespan(independent_paths),
        vertex_conflict_count=vertex_count,
        edge_conflict_count=edge_count,
    )


def evaluate_benchmark_candidate(
    grid_map: GridMap,
    scenarios: Sequence[Scenario],
    scenario_indices: Sequence[int],
    max_timestep: int,
    *,
    precomputed_lookup: Mapping[int, MAPFPrecomputedSourcePath] | None = None,
) -> MAPFBenchmarkInstance | None:
    return _evaluate_candidate(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_indices=scenario_indices,
        max_timestep=max_timestep,
        path_cache={},
        precomputed_lookup=precomputed_lookup,
    )


def _evaluate_candidate_direct(
    grid_map: GridMap,
    scenarios: Sequence[Scenario],
    scenario_indices: Sequence[int],
    max_timestep: int,
) -> MAPFBenchmarkInstance | None:
    return _evaluate_candidate(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_indices=scenario_indices,
        max_timestep=max_timestep,
        path_cache={},
        precomputed_lookup=None,
    )


def _sample_benchmark_instances(
    grid_map: GridMap,
    scenarios: Sequence[Scenario],
    *,
    agent_count: int,
    instances_per_level: int,
    max_timestep: int,
    seed: int,
    max_attempts: int,
    sample_indices: Sequence[int],
    precomputed_lookup: Mapping[int, MAPFPrecomputedSourcePath],
    sampling_progress_every: int = 0,
    sampling_progress_callback: Callable[
        [int, int, dict[MAPFInteractionLevel, int], dict[MAPFInteractionLevel, int]],
        None,
    ]
    | None = None,
    accepted_instance_callback: Callable[[MAPFBenchmarkInstance], None] | None = None,
) -> MAPFBenchmarkGenerationResult:
    if agent_count <= 0:
        raise ValueError("agent_count must be positive")
    if instances_per_level <= 0:
        raise ValueError("instances_per_level must be positive")
    if max_timestep < 0:
        raise ValueError("max_timestep must be non-negative")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if len(sample_indices) < agent_count:
        raise ValueError(
            f"Only {len(sample_indices)} scenarios available for agent_count={agent_count}; "
            f"need at least {agent_count}."
        )

    rng = random.Random(_agent_count_seed(seed, agent_count))
    map_stem = _map_stem(grid_map)

    needed = {
        MAPFInteractionLevel.LOW: instances_per_level,
        MAPFInteractionLevel.MEDIUM: instances_per_level,
        MAPFInteractionLevel.HIGH: instances_per_level,
    }
    sequence_by_level = {
        MAPFInteractionLevel.LOW: 0,
        MAPFInteractionLevel.MEDIUM: 0,
        MAPFInteractionLevel.HIGH: 0,
    }
    accepted_by_level: dict[MAPFInteractionLevel, list[MAPFBenchmarkInstance]] = {
        MAPFInteractionLevel.LOW: [],
        MAPFInteractionLevel.MEDIUM: [],
        MAPFInteractionLevel.HIGH: [],
    }
    accepted_signatures: set[tuple[int, ...]] = set()
    conflict_count_distribution: dict[int, int] = {}

    attempts = 0
    while attempts < max_attempts and any(count > 0 for count in needed.values()):
        attempts += 1

        if (
            sampling_progress_every > 0
            and sampling_progress_callback is not None
            and attempts % sampling_progress_every == 0
        ):
            found = {
                level: instances_per_level - needed[level]
                for level in MAPFInteractionLevel
            }
            sampling_progress_callback(
                attempts,
                max_attempts,
                needed,
                found,
            )

        sampled_indices = tuple(
            rng.sample(list(sample_indices), agent_count)
        )
        signature = _candidate_signature(sampled_indices)
        if signature in accepted_signatures:
            continue

        evaluation = _evaluate_candidate(
            grid_map=grid_map,
            scenarios=scenarios,
            scenario_indices=sampled_indices,
            max_timestep=max_timestep,
            path_cache={},
            precomputed_lookup=precomputed_lookup,
        )
        if evaluation is None:
            continue

        conflict_count_distribution[evaluation.independent_conflict_count] = (
            conflict_count_distribution.get(evaluation.independent_conflict_count, 0) + 1
        )

        level = evaluation.interaction_level
        if needed[level] <= 0:
            continue

        instance = MAPFBenchmarkInstance(
            instance_id=_make_instance_id(
                map_stem=map_stem,
                agent_count=agent_count,
                interaction_level=level,
                sequence=sequence_by_level[level],
            ),
            agent_count=evaluation.agent_count,
            interaction_level=evaluation.interaction_level,
            scenario_indices=evaluation.scenario_indices,
            independent_conflict_count=evaluation.independent_conflict_count,
            conflicting_agent_pair_count=evaluation.conflicting_agent_pair_count,
            independent_soc=evaluation.independent_soc,
            independent_makespan=evaluation.independent_makespan,
            vertex_conflict_count=evaluation.vertex_conflict_count,
            edge_conflict_count=evaluation.edge_conflict_count,
        )

        accepted_signatures.add(signature)
        accepted_by_level[level].append(instance)
        sequence_by_level[level] += 1
        needed[level] -= 1

        if accepted_instance_callback is not None:
            accepted_instance_callback(instance)

    if any(count > 0 for count in needed.values()):
        found = _count_by_level(
            [
                *accepted_by_level[MAPFInteractionLevel.LOW],
                *accepted_by_level[MAPFInteractionLevel.MEDIUM],
                *accepted_by_level[MAPFInteractionLevel.HIGH],
            ]
        )
        distribution_text = ", ".join(
            f"{count}={frequency}"
            for count, frequency in sorted(conflict_count_distribution.items())
        )
        raise ValueError(
            "Failed to generate requested benchmark instances for "
            f"agent_count={agent_count}: requested {instances_per_level} per level "
            f"(LOW/MEDIUM/HIGH); found LOW={found[MAPFInteractionLevel.LOW]}, "
            f"MEDIUM={found[MAPFInteractionLevel.MEDIUM]}, "
            f"HIGH={found[MAPFInteractionLevel.HIGH]} after {attempts} attempts. "
            f"Observed independent conflict-count distribution: "
            f"{{{distribution_text}}}"
        )

    ordered_instances = (
        *accepted_by_level[MAPFInteractionLevel.LOW],
        *accepted_by_level[MAPFInteractionLevel.MEDIUM],
        *accepted_by_level[MAPFInteractionLevel.HIGH],
    )
    return MAPFBenchmarkGenerationResult(
        instances=ordered_instances,
        attempts=attempts,
    )


def generate_benchmark_instances_from_precomputed(
    grid_map: GridMap,
    scenarios: Sequence[Scenario],
    *,
    agent_count: int,
    instances_per_level: int,
    max_timestep: int,
    seed: int,
    max_attempts: int,
    source_pool: MAPFBenchmarkSourcePool,
    sampling_progress_every: int = 0,
    sampling_progress_callback: Callable[
        [int, int, dict[MAPFInteractionLevel, int], dict[MAPFInteractionLevel, int]],
        None,
    ]
    | None = None,
    accepted_instance_callback: Callable[[MAPFBenchmarkInstance], None] | None = None,
) -> MAPFBenchmarkGenerationResult:
    return _sample_benchmark_instances(
        grid_map=grid_map,
        scenarios=scenarios,
        agent_count=agent_count,
        instances_per_level=instances_per_level,
        max_timestep=max_timestep,
        seed=seed,
        max_attempts=max_attempts,
        sample_indices=source_pool.feasible_indices,
        precomputed_lookup=source_pool.precomputed_lookup,
        sampling_progress_every=sampling_progress_every,
        sampling_progress_callback=sampling_progress_callback,
        accepted_instance_callback=accepted_instance_callback,
    )


def generate_benchmark_instances_legacy(
    grid_map: GridMap,
    scenarios: Sequence[Scenario],
    *,
    agent_count: int,
    instances_per_level: int,
    max_timestep: int,
    seed: int,
    min_reference_length: float,
    max_attempts: int,
) -> MAPFBenchmarkGenerationResult:
    """Legacy lazy path-cache sampling for equivalence testing only."""
    eligible_indices = eligible_scenario_indices(
        scenarios=scenarios,
        grid_map=grid_map,
        min_reference_length=min_reference_length,
    )
    if len(eligible_indices) < agent_count:
        raise ValueError(
            f"Only {len(eligible_indices)} eligible scenarios for agent_count={agent_count}; "
            f"need at least {agent_count} with min_reference_length={min_reference_length}."
        )

    rng = random.Random(_agent_count_seed(seed, agent_count))
    map_stem = _map_stem(grid_map)

    needed = {
        MAPFInteractionLevel.LOW: instances_per_level,
        MAPFInteractionLevel.MEDIUM: instances_per_level,
        MAPFInteractionLevel.HIGH: instances_per_level,
    }
    sequence_by_level = {
        MAPFInteractionLevel.LOW: 0,
        MAPFInteractionLevel.MEDIUM: 0,
        MAPFInteractionLevel.HIGH: 0,
    }
    accepted_by_level: dict[MAPFInteractionLevel, list[MAPFBenchmarkInstance]] = {
        MAPFInteractionLevel.LOW: [],
        MAPFInteractionLevel.MEDIUM: [],
        MAPFInteractionLevel.HIGH: [],
    }
    accepted_signatures: set[tuple[int, ...]] = set()
    conflict_count_distribution: dict[int, int] = {}
    path_cache: dict[int, tuple[TimedState, ...] | None] = {}

    attempts = 0
    while attempts < max_attempts and any(count > 0 for count in needed.values()):
        attempts += 1

        sampled_indices = tuple(
            rng.sample(list(eligible_indices), agent_count)
        )
        signature = _candidate_signature(sampled_indices)
        if signature in accepted_signatures:
            continue

        evaluation = _evaluate_candidate(
            grid_map=grid_map,
            scenarios=scenarios,
            scenario_indices=sampled_indices,
            max_timestep=max_timestep,
            path_cache=path_cache,
            precomputed_lookup=None,
        )
        if evaluation is None:
            continue

        conflict_count_distribution[evaluation.independent_conflict_count] = (
            conflict_count_distribution.get(evaluation.independent_conflict_count, 0) + 1
        )

        level = evaluation.interaction_level
        if needed[level] <= 0:
            continue

        instance = MAPFBenchmarkInstance(
            instance_id=_make_instance_id(
                map_stem=map_stem,
                agent_count=agent_count,
                interaction_level=level,
                sequence=sequence_by_level[level],
            ),
            agent_count=evaluation.agent_count,
            interaction_level=evaluation.interaction_level,
            scenario_indices=evaluation.scenario_indices,
            independent_conflict_count=evaluation.independent_conflict_count,
            conflicting_agent_pair_count=evaluation.conflicting_agent_pair_count,
            independent_soc=evaluation.independent_soc,
            independent_makespan=evaluation.independent_makespan,
            vertex_conflict_count=evaluation.vertex_conflict_count,
            edge_conflict_count=evaluation.edge_conflict_count,
        )

        accepted_signatures.add(signature)
        accepted_by_level[level].append(instance)
        sequence_by_level[level] += 1
        needed[level] -= 1

    if any(count > 0 for count in needed.values()):
        found = _count_by_level(
            [
                *accepted_by_level[MAPFInteractionLevel.LOW],
                *accepted_by_level[MAPFInteractionLevel.MEDIUM],
                *accepted_by_level[MAPFInteractionLevel.HIGH],
            ]
        )
        distribution_text = ", ".join(
            f"{count}={frequency}"
            for count, frequency in sorted(conflict_count_distribution.items())
        )
        raise ValueError(
            "Failed to generate requested benchmark instances for "
            f"agent_count={agent_count}: requested {instances_per_level} per level "
            f"(LOW/MEDIUM/HIGH); found LOW={found[MAPFInteractionLevel.LOW]}, "
            f"MEDIUM={found[MAPFInteractionLevel.MEDIUM]}, "
            f"HIGH={found[MAPFInteractionLevel.HIGH]} after {attempts} attempts. "
            f"Observed independent conflict-count distribution: "
            f"{{{distribution_text}}}"
        )

    ordered_instances = (
        *accepted_by_level[MAPFInteractionLevel.LOW],
        *accepted_by_level[MAPFInteractionLevel.MEDIUM],
        *accepted_by_level[MAPFInteractionLevel.HIGH],
    )
    return MAPFBenchmarkGenerationResult(
        instances=ordered_instances,
        attempts=attempts,
    )


def _agent_count_seed(base_seed: int, agent_count: int) -> int:
    return base_seed + agent_count


def _count_by_level(
    instances: Sequence[MAPFBenchmarkInstance],
) -> dict[MAPFInteractionLevel, int]:
    counts = {
        MAPFInteractionLevel.LOW: 0,
        MAPFInteractionLevel.MEDIUM: 0,
        MAPFInteractionLevel.HIGH: 0,
    }
    for instance in instances:
        counts[instance.interaction_level] += 1
    return counts


def generate_benchmark_instances(
    grid_map: GridMap,
    scenarios: Sequence[Scenario],
    *,
    agent_count: int,
    instances_per_level: int,
    max_timestep: int,
    seed: int,
    min_reference_length: float,
    max_attempts: int,
    precompute_progress_every: int = 0,
    precompute_progress_callback: Callable[[StaticPrecomputeProgress], None] | None = None,
    sampling_progress_every: int = 0,
    sampling_progress_callback: Callable[
        [int, int, dict[MAPFInteractionLevel, int], dict[MAPFInteractionLevel, int]],
        None,
    ]
    | None = None,
    accepted_instance_callback: Callable[[MAPFBenchmarkInstance], None] | None = None,
) -> MAPFBenchmarkGenerationResult:
    if agent_count <= 0:
        raise ValueError("agent_count must be positive")
    if instances_per_level <= 0:
        raise ValueError("instances_per_level must be positive")
    if max_timestep < 0:
        raise ValueError("max_timestep must be non-negative")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if min_reference_length < 0:
        raise ValueError("min_reference_length must be non-negative")

    source_pool = prepare_benchmark_source_pool(
        grid_map=grid_map,
        scenarios=scenarios,
        min_reference_length=min_reference_length,
        max_timestep=max_timestep,
        precompute_progress_every=precompute_progress_every,
        precompute_progress_callback=precompute_progress_callback,
    )
    return generate_benchmark_instances_from_precomputed(
        grid_map=grid_map,
        scenarios=scenarios,
        agent_count=agent_count,
        instances_per_level=instances_per_level,
        max_timestep=max_timestep,
        seed=seed,
        max_attempts=max_attempts,
        source_pool=source_pool,
        sampling_progress_every=sampling_progress_every,
        sampling_progress_callback=sampling_progress_callback,
        accepted_instance_callback=accepted_instance_callback,
    )


def generate_benchmark_manifest(
    grid_map: GridMap,
    scenarios: Sequence[Scenario],
    scenario_name: str,
    *,
    agent_counts: Sequence[int],
    instances_per_level: int,
    max_timestep: int,
    seed: int,
    min_reference_length: float,
    max_attempts: int,
    source_pool: MAPFBenchmarkSourcePool | None = None,
    precompute_progress_every: int = 0,
    precompute_progress_callback: Callable[[StaticPrecomputeProgress], None] | None = None,
    sampling_progress_every: int = 0,
    sampling_progress_callback: Callable[
        [int, int, int, dict[MAPFInteractionLevel, int], dict[MAPFInteractionLevel, int]],
        None,
    ]
    | None = None,
    accepted_instance_callback: Callable[[MAPFBenchmarkInstance], None] | None = None,
    generation_start_callback: Callable[[int], None] | None = None,
) -> MAPFBenchmarkManifestGenerationResult:
    if not agent_counts:
        raise ValueError("agent_counts must not be empty")

    if source_pool is None:
        source_pool = prepare_benchmark_source_pool(
            grid_map=grid_map,
            scenarios=scenarios,
            min_reference_length=min_reference_length,
            max_timestep=max_timestep,
            precompute_progress_every=precompute_progress_every,
            precompute_progress_callback=precompute_progress_callback,
        )

    all_instances: list[MAPFBenchmarkInstance] = []
    attempts_by_agent_count: dict[int, int] = {}

    for agent_count in agent_counts:
        if agent_count <= 0:
            raise ValueError("agent_counts must contain only positive values")

        if generation_start_callback is not None:
            generation_start_callback(agent_count)

        def _sampling_callback(
            attempts: int,
            max_attempts_local: int,
            needed: dict[MAPFInteractionLevel, int],
            found: dict[MAPFInteractionLevel, int],
            *,
            current_agent_count: int = agent_count,
        ) -> None:
            if sampling_progress_callback is not None:
                sampling_progress_callback(
                    current_agent_count,
                    attempts,
                    max_attempts_local,
                    needed,
                    found,
                )

        result = generate_benchmark_instances_from_precomputed(
            grid_map=grid_map,
            scenarios=scenarios,
            agent_count=agent_count,
            instances_per_level=instances_per_level,
            max_timestep=max_timestep,
            seed=seed,
            max_attempts=max_attempts,
            source_pool=source_pool,
            sampling_progress_every=sampling_progress_every,
            sampling_progress_callback=_sampling_callback
            if sampling_progress_every > 0
            else None,
            accepted_instance_callback=accepted_instance_callback,
        )
        all_instances.extend(result.instances)
        attempts_by_agent_count[agent_count] = result.attempts

    manifest = MAPFBenchmarkManifest(
        map_name=grid_map.name,
        scenario_name=scenario_name,
        seed=seed,
        max_timestep=max_timestep,
        min_reference_length=min_reference_length,
        instances=tuple(all_instances),
    )
    return MAPFBenchmarkManifestGenerationResult(
        manifest=manifest,
        attempts_by_agent_count=attempts_by_agent_count,
        source_pool=source_pool,
    )


def _instance_to_json(instance: MAPFBenchmarkInstance) -> dict[str, object]:
    return {
        "instance_id": instance.instance_id,
        "agent_count": instance.agent_count,
        "interaction_level": instance.interaction_level.value,
        "scenario_indices": list(instance.scenario_indices),
        "independent_conflict_count": instance.independent_conflict_count,
        "conflicting_agent_pair_count": instance.conflicting_agent_pair_count,
        "independent_soc": instance.independent_soc,
        "independent_makespan": instance.independent_makespan,
        "vertex_conflict_count": instance.vertex_conflict_count,
        "edge_conflict_count": instance.edge_conflict_count,
    }


def _instance_from_json(data: dict[str, object]) -> MAPFBenchmarkInstance:
    return MAPFBenchmarkInstance(
        instance_id=str(data["instance_id"]),
        agent_count=int(data["agent_count"]),
        interaction_level=MAPFInteractionLevel(str(data["interaction_level"])),
        scenario_indices=tuple(int(index) for index in data["scenario_indices"]),  # type: ignore[arg-type]
        independent_conflict_count=int(data["independent_conflict_count"]),
        conflicting_agent_pair_count=int(data["conflicting_agent_pair_count"]),
        independent_soc=int(data["independent_soc"]),
        independent_makespan=int(data["independent_makespan"]),
        vertex_conflict_count=int(data["vertex_conflict_count"]),
        edge_conflict_count=int(data["edge_conflict_count"]),
    )


def _manifest_to_json(manifest: MAPFBenchmarkManifest) -> dict[str, object]:
    return {
        "map_name": manifest.map_name,
        "scenario_name": manifest.scenario_name,
        "seed": manifest.seed,
        "max_timestep": manifest.max_timestep,
        "min_reference_length": manifest.min_reference_length,
        "interaction_thresholds": {
            "low_max": manifest.interaction_low_max_conflicts,
            "medium_min": manifest.interaction_medium_min_conflicts,
            "medium_max": manifest.interaction_medium_max_conflicts,
            "high_min": manifest.interaction_high_min_conflicts,
        },
        "instances": [_instance_to_json(instance) for instance in manifest.instances],
    }


def _manifest_from_json(data: dict[str, object]) -> MAPFBenchmarkManifest:
    thresholds = data.get("interaction_thresholds", {})
    if not isinstance(thresholds, dict):
        thresholds = {}

    instances_data = data["instances"]
    if not isinstance(instances_data, list):
        raise ValueError("manifest instances must be a list")

    return MAPFBenchmarkManifest(
        map_name=str(data["map_name"]),
        scenario_name=str(data["scenario_name"]),
        seed=int(data["seed"]),
        max_timestep=int(data["max_timestep"]),
        min_reference_length=float(data["min_reference_length"]),
        instances=tuple(
            _instance_from_json(instance_data)
            for instance_data in instances_data
            if isinstance(instance_data, dict)
        ),
        interaction_low_max_conflicts=int(
            thresholds.get("low_max", INTERACTION_LOW_MAX_CONFLICTS)
        ),
        interaction_medium_min_conflicts=int(
            thresholds.get("medium_min", INTERACTION_MEDIUM_MIN_CONFLICTS)
        ),
        interaction_medium_max_conflicts=int(
            thresholds.get("medium_max", INTERACTION_MEDIUM_MAX_CONFLICTS)
        ),
        interaction_high_min_conflicts=int(
            thresholds.get("high_min", INTERACTION_HIGH_MIN_CONFLICTS)
        ),
    )


def save_benchmark_manifest(manifest: MAPFBenchmarkManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _manifest_to_json(manifest)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_benchmark_manifest(path: Path) -> MAPFBenchmarkManifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid benchmark manifest format: {path}")
    return _manifest_from_json(data)


def reconstruct_mapf_scenario_from_instance(
    scenarios: Sequence[Scenario],
    grid_map: GridMap,
    instance: MAPFBenchmarkInstance,
):
    return build_mapf_scenario_from_indices(
        scenarios=scenarios,
        grid_map=grid_map,
        scenario_indices=instance.scenario_indices,
    )
