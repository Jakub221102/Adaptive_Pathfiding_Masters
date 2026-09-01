from __future__ import annotations

import json
import random
import time
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

CATALOGUE_ROLE_PRIMARY = "primary"
CATALOGUE_ROLE_HELD_OUT = "held_out_validation"
CATALOGUE_ROLE_MAPF9 = "mapf9_primary"
HELD_OUT_CATALOGUE_GENERATION_VERSION = "MAPF-7.6"
MAPF9_CATALOGUE_GENERATION_VERSION = "MAPF-9.3"
DEFAULT_HELD_OUT_SEED = 2027
MAPF9_CATALOGUE_SEEDS: dict[str, int] = {
    "AR0400SR": 2030,
    "AR0307SR": 2031,
}
MAPF9_CROSS_MAP_TARGETS: frozenset[str] = frozenset({"AR0400SR", "AR0307SR"})
MAPF8_CROSS_MAP_REFERENCE_SEEDS: dict[str, int] = {
    "AR0400SR": 2028,
    "AR0307SR": 2029,
}


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

    catalogue_role: str = CATALOGUE_ROLE_PRIMARY
    generation_version: str | None = None
    primary_manifest_reference: str | None = None


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
    spatially_reachable_count: int = 0
    reachability_component_count: int = 0
    reachability_walkable_cells: int = 0
    reachability_preprocess_s: float = 0.0
    bounded_preprocess_s: float = 0.0


@dataclass(frozen=True, slots=True)
class StaticPrecomputeProgress:
    completed: int
    total: int
    feasible: int
    no_spatial_path: int
    over_horizon: int
    spatially_reachable: int


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
        StaticPathSearchStatus,
        are_spatially_connected,
        build_static_reachability_index,
        find_independent_static_path_bounded_result,
    )

    if max_timestep < 0:
        raise ValueError("max_timestep must be non-negative")
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")

    reachability_start = time.perf_counter()
    reachability_index = build_static_reachability_index(grid_map)
    reachability_preprocess_s = time.perf_counter() - reachability_start

    bounded_start = time.perf_counter()
    succeeded: list[MAPFPrecomputedSourcePath] = []
    failed: list[int] = []
    total = len(scenario_indices)
    planned: dict[int, MAPFPrecomputedSourcePath | None] = {}
    no_spatial_path_count = 0
    over_horizon_count = 0
    spatially_reachable_count = 0
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
                spatially_reachable=spatially_reachable_count,
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
        if not are_spatially_connected(
            reachability_index,
            scenario.start,
            scenario.goal,
        ):
            planned[scenario_index] = None
            failed.append(scenario_index)
            no_spatial_path_count += 1
            if progress_every > 0 and completed % progress_every == 0:
                _emit_progress(completed)
            continue

        bounded_result = find_independent_static_path_bounded_result(
            grid_map=grid_map,
            agent=agent,
            max_cost=max_timestep,
            reachability_index=reachability_index,
        )
        if bounded_result.status == StaticPathSearchStatus.SUCCESS:
            assert bounded_result.path is not None
            precomputed = MAPFPrecomputedSourcePath(
                scenario_index=scenario_index,
                states=bounded_result.path.states,
            )
            planned[scenario_index] = precomputed
            succeeded.append(precomputed)
            spatially_reachable_count += 1
            feasible_count += 1
        elif bounded_result.status == StaticPathSearchStatus.OVER_COST_BOUND:
            planned[scenario_index] = None
            failed.append(scenario_index)
            spatially_reachable_count += 1
            over_horizon_count += 1
        else:
            planned[scenario_index] = None
            failed.append(scenario_index)
            no_spatial_path_count += 1

        if progress_every > 0 and completed % progress_every == 0:
            _emit_progress(completed)

    if total > 0:
        _emit_progress(total)

    bounded_preprocess_s = time.perf_counter() - bounded_start

    return MAPFPrecomputePathsResult(
        paths=tuple(succeeded),
        failed_scenario_indices=tuple(failed),
        no_spatial_path_count=no_spatial_path_count,
        over_horizon_count=over_horizon_count,
        spatially_reachable_count=spatially_reachable_count,
        reachability_component_count=reachability_index.component_count,
        reachability_walkable_cells=reachability_index.walkable_cell_count,
        reachability_preprocess_s=reachability_preprocess_s,
        bounded_preprocess_s=bounded_preprocess_s,
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
    *,
    held_out: bool = False,
) -> str:
    prefix = f"{map_stem}_HO" if held_out else map_stem
    return f"{prefix}_n{agent_count:02d}_{interaction_level.value}_{sequence:03d}"


def benchmark_scenario_set_signature(scenario_indices: Sequence[int]) -> tuple[int, ...]:
    """Unordered scenario-index set identity used for duplicate detection."""
    return _candidate_signature(scenario_indices)


def collect_scenario_set_signatures_by_agent_count(
    instances: Sequence[MAPFBenchmarkInstance],
) -> dict[int, set[tuple[int, ...]]]:
    signatures_by_agent_count: dict[int, set[tuple[int, ...]]] = {}
    for instance in instances:
        signatures_by_agent_count.setdefault(instance.agent_count, set()).add(
            benchmark_scenario_set_signature(instance.scenario_indices)
        )
    return signatures_by_agent_count


def collect_all_scenario_set_signatures(
    instances: Sequence[MAPFBenchmarkInstance],
) -> frozenset[tuple[int, ...]]:
    return frozenset(
        benchmark_scenario_set_signature(instance.scenario_indices)
        for instance in instances
    )


def mapf8_mapf9_signature_overlap(
    mapf9_manifest: MAPFBenchmarkManifest,
    mapf8_manifest: MAPFBenchmarkManifest,
) -> tuple[frozenset[tuple[int, ...]], frozenset[tuple[int, ...]], frozenset[tuple[int, ...]]]:
    """Return (mapf9_signatures, mapf8_signatures, intersection)."""
    mapf9_signatures = collect_all_scenario_set_signatures(mapf9_manifest.instances)
    mapf8_signatures = collect_all_scenario_set_signatures(mapf8_manifest.instances)
    intersection = mapf9_signatures & mapf8_signatures
    return mapf9_signatures, mapf8_signatures, intersection


def validate_mapf8_mapf9_signature_disjointness(
    mapf9_manifest: MAPFBenchmarkManifest,
    mapf8_manifest: MAPFBenchmarkManifest,
) -> None:
    mapf9_signatures, mapf8_signatures, intersection = mapf8_mapf9_signature_overlap(
        mapf9_manifest,
        mapf8_manifest,
    )
    if intersection:
        sample = next(iter(intersection))
        raise ValueError(
            f"MAPF-9 catalogue shares {len(intersection)} scenario-set signature(s) "
            f"with frozen MAPF-8 catalogue on the same map; "
            f"example overlapping signature: {sample}"
        )
    if len(mapf9_signatures) != len(mapf9_manifest.instances):
        raise ValueError("MAPF-9 manifest contains duplicate scenario-set signatures")
    if len(mapf8_signatures) != len(mapf8_manifest.instances):
        raise ValueError("MAPF-8 reference manifest contains duplicate scenario-set signatures")


def instance_agent_position_signature(
    scenarios: Sequence[Scenario],
    scenario_indices: Sequence[int],
) -> tuple[tuple[int, int, int, int], ...]:
    """Ordered (start_row, start_col, goal_row, goal_col) per agent."""
    signature: list[tuple[int, int, int, int]] = []
    for scenario_index in scenario_indices:
        scenario = scenarios[scenario_index]
        signature.append(
            (
                scenario.start.row,
                scenario.start.col,
                scenario.goal.row,
                scenario.goal.col,
            )
        )
    return tuple(signature)


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
    accepted_instance_callback: Callable[[MAPFBenchmarkInstance, int], None] | None = None,
    excluded_signatures: frozenset[tuple[int, ...]] | None = None,
    held_out: bool = False,
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
        if excluded_signatures is not None and signature in excluded_signatures:
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
                held_out=held_out,
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
            accepted_instance_callback(instance, attempts)

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
    accepted_instance_callback: Callable[[MAPFBenchmarkInstance, int], None] | None = None,
    excluded_signatures: frozenset[tuple[int, ...]] | None = None,
    held_out: bool = False,
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
        excluded_signatures=excluded_signatures,
        held_out=held_out,
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
    accepted_instance_callback: Callable[[MAPFBenchmarkInstance, int], None] | None = None,
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
    accepted_instance_callback: Callable[[MAPFBenchmarkInstance, int], None] | None = None,
    generation_start_callback: Callable[[int], None] | None = None,
    excluded_signatures_by_agent_count: dict[int, frozenset[tuple[int, ...]]] | None = None,
    catalogue_role: str = CATALOGUE_ROLE_PRIMARY,
    generation_version: str | None = None,
    primary_manifest_reference: str | None = None,
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

        excluded = None
        if excluded_signatures_by_agent_count is not None:
            excluded = excluded_signatures_by_agent_count.get(agent_count, frozenset())

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
            excluded_signatures=excluded,
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
        catalogue_role=catalogue_role,
        generation_version=generation_version,
        primary_manifest_reference=primary_manifest_reference,
    )
    return MAPFBenchmarkManifestGenerationResult(
        manifest=manifest,
        attempts_by_agent_count=attempts_by_agent_count,
        source_pool=source_pool,
    )


def generate_mapf9_benchmark_manifest(
    grid_map: GridMap,
    scenarios: Sequence[Scenario],
    scenario_name: str,
    mapf8_reference_manifest: MAPFBenchmarkManifest,
    *,
    agent_counts: Sequence[int] = (5, 10, 20),
    instances_per_level: int = 3,
    max_timestep: int = 512,
    seed: int,
    min_reference_length: float = 20.0,
    max_attempts: int = 5000,
    source_pool: MAPFBenchmarkSourcePool | None = None,
    mapf8_manifest_reference: str | None = None,
    precompute_progress_every: int = 0,
    precompute_progress_callback: Callable[[StaticPrecomputeProgress], None] | None = None,
    sampling_progress_every: int = 0,
    sampling_progress_callback: Callable[
        [int, int, int, dict[MAPFInteractionLevel, int], dict[MAPFInteractionLevel, int]],
        None,
    ]
    | None = None,
    accepted_instance_callback: Callable[[MAPFBenchmarkInstance, int], None] | None = None,
    generation_start_callback: Callable[[int], None] | None = None,
) -> MAPFBenchmarkManifestGenerationResult:
    """Generate a MAPF-9 primary catalogue excluding MAPF-8 scenario-set signatures."""
    if seed == mapf8_reference_manifest.seed:
        raise ValueError(
            "MAPF-9 seed must differ from the frozen MAPF-8 catalogue seed "
            f"({mapf8_reference_manifest.seed})."
        )

    excluded_by_agent_count = {
        agent_count: frozenset(signatures)
        for agent_count, signatures in collect_scenario_set_signatures_by_agent_count(
            mapf8_reference_manifest.instances
        ).items()
    }

    return generate_benchmark_manifest(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_name=scenario_name,
        agent_counts=agent_counts,
        instances_per_level=instances_per_level,
        max_timestep=max_timestep,
        seed=seed,
        min_reference_length=min_reference_length,
        max_attempts=max_attempts,
        source_pool=source_pool,
        precompute_progress_every=precompute_progress_every,
        precompute_progress_callback=precompute_progress_callback,
        sampling_progress_every=sampling_progress_every,
        sampling_progress_callback=sampling_progress_callback,
        accepted_instance_callback=accepted_instance_callback,
        generation_start_callback=generation_start_callback,
        excluded_signatures_by_agent_count=excluded_by_agent_count,
        catalogue_role=CATALOGUE_ROLE_MAPF9,
        generation_version=MAPF9_CATALOGUE_GENERATION_VERSION,
        primary_manifest_reference=mapf8_manifest_reference,
    )


def generate_held_out_benchmark_manifest(
    grid_map: GridMap,
    scenarios: Sequence[Scenario],
    scenario_name: str,
    primary_manifest: MAPFBenchmarkManifest,
    *,
    agent_counts: Sequence[int] = (5, 10, 20),
    instances_per_level: int = 3,
    max_timestep: int | None = None,
    seed: int = DEFAULT_HELD_OUT_SEED,
    min_reference_length: float | None = None,
    max_attempts: int = 5000,
    source_pool: MAPFBenchmarkSourcePool | None = None,
    primary_manifest_reference: str | None = None,
    precompute_progress_every: int = 0,
    precompute_progress_callback: Callable[[StaticPrecomputeProgress], None] | None = None,
    accepted_instance_callback: Callable[[MAPFBenchmarkInstance, int], None] | None = None,
    generation_start_callback: Callable[[int], None] | None = None,
) -> MAPFBenchmarkManifestGenerationResult:
    """Generate a held-out validation catalogue excluding primary scenario sets."""
    if seed == primary_manifest.seed:
        raise ValueError(
            "Held-out seed must differ from the primary catalogue seed "
            f"({primary_manifest.seed})."
        )

    resolved_max_timestep = (
        primary_manifest.max_timestep if max_timestep is None else max_timestep
    )
    resolved_min_reference_length = (
        primary_manifest.min_reference_length
        if min_reference_length is None
        else min_reference_length
    )

    if source_pool is None:
        source_pool = prepare_benchmark_source_pool(
            grid_map=grid_map,
            scenarios=scenarios,
            min_reference_length=resolved_min_reference_length,
            max_timestep=resolved_max_timestep,
            precompute_progress_every=precompute_progress_every,
            precompute_progress_callback=precompute_progress_callback,
        )

    primary_signatures_by_agent_count = collect_scenario_set_signatures_by_agent_count(
        primary_manifest.instances
    )

    all_instances: list[MAPFBenchmarkInstance] = []
    attempts_by_agent_count: dict[int, int] = {}

    for agent_count in agent_counts:
        if agent_count <= 0:
            raise ValueError("agent_counts must contain only positive values")

        if generation_start_callback is not None:
            generation_start_callback(agent_count)

        excluded = frozenset(
            primary_signatures_by_agent_count.get(agent_count, set())
        )
        result = generate_benchmark_instances_from_precomputed(
            grid_map=grid_map,
            scenarios=scenarios,
            agent_count=agent_count,
            instances_per_level=instances_per_level,
            max_timestep=resolved_max_timestep,
            seed=seed,
            max_attempts=max_attempts,
            source_pool=source_pool,
            accepted_instance_callback=accepted_instance_callback,
            excluded_signatures=excluded,
            held_out=True,
        )
        all_instances.extend(result.instances)
        attempts_by_agent_count[agent_count] = result.attempts

    manifest = MAPFBenchmarkManifest(
        map_name=grid_map.name,
        scenario_name=scenario_name,
        seed=seed,
        max_timestep=resolved_max_timestep,
        min_reference_length=resolved_min_reference_length,
        instances=tuple(all_instances),
        interaction_low_max_conflicts=primary_manifest.interaction_low_max_conflicts,
        interaction_medium_min_conflicts=primary_manifest.interaction_medium_min_conflicts,
        interaction_medium_max_conflicts=primary_manifest.interaction_medium_max_conflicts,
        interaction_high_min_conflicts=primary_manifest.interaction_high_min_conflicts,
        catalogue_role=CATALOGUE_ROLE_HELD_OUT,
        generation_version=HELD_OUT_CATALOGUE_GENERATION_VERSION,
        primary_manifest_reference=primary_manifest_reference,
    )
    return MAPFBenchmarkManifestGenerationResult(
        manifest=manifest,
        attempts_by_agent_count=attempts_by_agent_count,
        source_pool=source_pool,
    )


def validate_held_out_catalogue(
    held_out_manifest: MAPFBenchmarkManifest,
    primary_manifest: MAPFBenchmarkManifest,
    *,
    scenarios: Sequence[Scenario] | None = None,
    grid_map: GridMap | None = None,
    agent_counts: Sequence[int] = (5, 10, 20),
    instances_per_level: int = 3,
) -> None:
    """Validate held-out catalogue structure without running any MAPF solver."""
    expected_total = len(agent_counts) * len(MAPFInteractionLevel) * instances_per_level
    if held_out_manifest.catalogue_role != CATALOGUE_ROLE_HELD_OUT:
        raise ValueError(
            f"Expected catalogue_role={CATALOGUE_ROLE_HELD_OUT!r}, "
            f"found {held_out_manifest.catalogue_role!r}"
        )
    if len(held_out_manifest.instances) != expected_total:
        raise ValueError(
            f"Expected {expected_total} held-out instances, "
            f"found {len(held_out_manifest.instances)}"
        )

    held_out_ids = [instance.instance_id for instance in held_out_manifest.instances]
    if len(set(held_out_ids)) != len(held_out_ids):
        raise ValueError("Duplicate held-out instance IDs found")

    primary_ids = {instance.instance_id for instance in primary_manifest.instances}
    overlap_ids = set(held_out_ids) & primary_ids
    if overlap_ids:
        raise ValueError(
            f"Held-out instance IDs overlap primary catalogue: {sorted(overlap_ids)}"
        )

    for instance_id in held_out_ids:
        if "_HO_" not in instance_id:
            raise ValueError(
                f"Held-out instance ID missing _HO_ marker: {instance_id}"
            )

    if held_out_manifest.map_name != primary_manifest.map_name:
        raise ValueError("Held-out map_name differs from primary manifest")
    if held_out_manifest.scenario_name != primary_manifest.scenario_name:
        raise ValueError("Held-out scenario_name differs from primary manifest")
    if held_out_manifest.max_timestep != primary_manifest.max_timestep:
        raise ValueError("Held-out max_timestep differs from primary manifest")
    if held_out_manifest.min_reference_length != primary_manifest.min_reference_length:
        raise ValueError("Held-out min_reference_length differs from primary manifest")
    if held_out_manifest.interaction_low_max_conflicts != (
        primary_manifest.interaction_low_max_conflicts
    ):
        raise ValueError("Held-out interaction thresholds differ from primary manifest")
    if held_out_manifest.seed == primary_manifest.seed:
        raise ValueError("Held-out seed must differ from primary catalogue seed")

    primary_signatures_by_agent_count = collect_scenario_set_signatures_by_agent_count(
        primary_manifest.instances
    )
    held_out_position_signatures: set[tuple[tuple[int, int, int, int], ...]] = set()
    primary_position_signatures: set[tuple[tuple[int, int, int, int], ...]] = set()
    seen_held_out_signatures_by_agent: dict[int, set[tuple[int, ...]]] = {}

    if scenarios is not None:
        for instance in primary_manifest.instances:
            primary_position_signatures.add(
                instance_agent_position_signature(scenarios, instance.scenario_indices)
            )

    counts_by_agent_level: dict[tuple[int, MAPFInteractionLevel], list[MAPFBenchmarkInstance]] = {
        (agent_count, level): []
        for agent_count in agent_counts
        for level in MAPFInteractionLevel
    }

    for instance in held_out_manifest.instances:
        if instance.agent_count not in agent_counts:
            raise ValueError(
                f"Unexpected agent_count in held-out manifest: {instance.agent_count}"
            )
        if len(instance.scenario_indices) != instance.agent_count:
            raise ValueError(
                f"Instance {instance.instance_id} has "
                f"{len(instance.scenario_indices)} scenario indices "
                f"for agent_count={instance.agent_count}"
            )
        if len(set(instance.scenario_indices)) != len(instance.scenario_indices):
            raise ValueError(
                f"Duplicate scenario indices in instance {instance.instance_id}"
            )

        signature = benchmark_scenario_set_signature(instance.scenario_indices)
        seen_for_agent = seen_held_out_signatures_by_agent.setdefault(
            instance.agent_count, set()
        )
        if signature in seen_for_agent:
            raise ValueError(
                f"Duplicate held-out scenario set for agent_count="
                f"{instance.agent_count}: {signature}"
            )
        seen_for_agent.add(signature)

        primary_for_agent = primary_signatures_by_agent_count.get(instance.agent_count, set())
        if signature in primary_for_agent:
            raise ValueError(
                f"Held-out instance {instance.instance_id} duplicates primary "
                f"scenario set for agent_count={instance.agent_count}: {signature}"
            )

        if scenarios is not None:
            position_signature = instance_agent_position_signature(
                scenarios, instance.scenario_indices
            )
            if position_signature in held_out_position_signatures:
                raise ValueError(
                    f"Duplicate held-out agent position signature: {instance.instance_id}"
                )
            held_out_position_signatures.add(position_signature)
            if position_signature in primary_position_signatures:
                raise ValueError(
                    f"Held-out instance {instance.instance_id} duplicates primary "
                    "agent (start, goal) assignment"
                )

            if grid_map is not None and not _validate_candidate_indices(
                instance.scenario_indices, scenarios, grid_map
            ):
                raise ValueError(
                    f"Invalid starts/goals for held-out instance {instance.instance_id}"
                )

        if not _interaction_matches_manifest_level(instance):
            raise ValueError(
                f"Instance {instance.instance_id} interaction label "
                f"{instance.interaction_level.name} does not match "
                f"conflict_count={instance.independent_conflict_count}"
            )

        counts_by_agent_level[(instance.agent_count, instance.interaction_level)].append(
            instance
        )

    for agent_count in agent_counts:
        agent_instances = [
            instance
            for instance in held_out_manifest.instances
            if instance.agent_count == agent_count
        ]
        expected_per_agent = len(MAPFInteractionLevel) * instances_per_level
        if len(agent_instances) != expected_per_agent:
            raise ValueError(
                f"Expected {expected_per_agent} held-out instances for "
                f"agent_count={agent_count}, found {len(agent_instances)}"
            )

        for level in MAPFInteractionLevel:
            level_instances = counts_by_agent_level[(agent_count, level)]
            if len(level_instances) != instances_per_level:
                raise ValueError(
                    f"Expected {instances_per_level} {level.name} held-out instances "
                    f"for agent_count={agent_count}, found {len(level_instances)}"
                )


def _interaction_matches_manifest_level(instance: MAPFBenchmarkInstance) -> bool:
    count = instance.independent_conflict_count
    level = instance.interaction_level
    if level == MAPFInteractionLevel.LOW:
        return count == INTERACTION_LOW_MAX_CONFLICTS
    if level == MAPFInteractionLevel.MEDIUM:
        return (
            INTERACTION_MEDIUM_MIN_CONFLICTS
            <= count
            <= INTERACTION_MEDIUM_MAX_CONFLICTS
        )
    if level == MAPFInteractionLevel.HIGH:
        return count >= INTERACTION_HIGH_MIN_CONFLICTS
    return False


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
    payload: dict[str, object] = {
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
    if manifest.catalogue_role != CATALOGUE_ROLE_PRIMARY:
        payload["catalogue_role"] = manifest.catalogue_role
    if manifest.generation_version is not None:
        payload["generation_version"] = manifest.generation_version
    if manifest.primary_manifest_reference is not None:
        payload["primary_manifest_reference"] = manifest.primary_manifest_reference
    return payload


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
        catalogue_role=str(data.get("catalogue_role", CATALOGUE_ROLE_PRIMARY)),
        generation_version=(
            str(data["generation_version"])
            if data.get("generation_version") is not None
            else None
        ),
        primary_manifest_reference=(
            str(data["primary_manifest_reference"])
            if data.get("primary_manifest_reference") is not None
            else None
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
