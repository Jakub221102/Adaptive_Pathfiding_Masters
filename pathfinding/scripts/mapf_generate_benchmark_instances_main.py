from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathfinding.src.experiments.mapf_benchmark_instances import (
    INTERACTION_HIGH_MIN_CONFLICTS,
    INTERACTION_LOW_MAX_CONFLICTS,
    INTERACTION_MEDIUM_MAX_CONFLICTS,
    INTERACTION_MEDIUM_MIN_CONFLICTS,
    MAPFBenchmarkInstance,
    MAPFInteractionLevel,
    StaticPrecomputeProgress,
    generate_benchmark_manifest,
    prepare_benchmark_source_pool,
    save_benchmark_manifest,
)
from pathfinding.src.loaders.map_loader import load_moving_ai_map
from pathfinding.src.loaders.scen_loader import load_moving_ai_scenarios

DEFAULT_MAP_PATH = _REPO_ROOT / "Data" / "bg512-map" / "AR0204SR.map"
DEFAULT_SCEN_PATH = _REPO_ROOT / "Data" / "bg512-scen" / "AR0204SR.map.scen"
DEFAULT_OUTPUT_PATH = (
    _REPO_ROOT / "pathfinding" / "results" / "mapf_instances_ar0204sr.json"
)

DEFAULT_AGENT_COUNTS: tuple[int, ...] = (5, 10, 20)
DEFAULT_INSTANCES_PER_LEVEL = 3
DEFAULT_SEED = 2026
DEFAULT_MAX_TIMESTEP = 512
DEFAULT_MIN_REFERENCE_LENGTH = 20.0
DEFAULT_MAX_ATTEMPTS = 5000
DEFAULT_PRECOMPUTE_PROGRESS_EVERY = 250
DEFAULT_SAMPLING_PROGRESS_EVERY = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic MAPF benchmark instances from MovingAI scenarios.",
    )
    parser.add_argument(
        "--map",
        type=Path,
        default=DEFAULT_MAP_PATH,
        help="Path to a MovingAI .map file.",
    )
    parser.add_argument(
        "--scen",
        type=Path,
        default=DEFAULT_SCEN_PATH,
        help="Path to a MovingAI .scen file.",
    )
    parser.add_argument(
        "--agent-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_AGENT_COUNTS),
        help="Agent counts to generate benchmark instances for.",
    )
    parser.add_argument(
        "--instances-per-level",
        type=int,
        default=DEFAULT_INSTANCES_PER_LEVEL,
        help="Number of instances per interaction level for each agent count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base random seed for deterministic generation.",
    )
    parser.add_argument(
        "--max-timestep",
        type=int,
        default=DEFAULT_MAX_TIMESTEP,
        help="Planning horizon used during independent path evaluation.",
    )
    parser.add_argument(
        "--min-reference-length",
        type=float,
        default=DEFAULT_MIN_REFERENCE_LENGTH,
        help="Minimum MovingAI reference/optimal length for scenario eligibility.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="Maximum candidate sampling attempts per agent count.",
    )
    parser.add_argument(
        "--precompute-progress-every",
        type=int,
        default=DEFAULT_PRECOMPUTE_PROGRESS_EVERY,
        help="Print precompute progress every N scenarios (0 disables).",
    )
    parser.add_argument(
        "--sampling-progress-every",
        type=int,
        default=DEFAULT_SAMPLING_PROGRESS_EVERY,
        help="Print sampling progress every N attempts (0 disables).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output path for the benchmark manifest JSON file.",
    )
    return parser.parse_args()


def _print_level_summary(
    level: MAPFInteractionLevel,
    instances: list,
) -> None:
    if not instances:
        print(f"{level.name}:")
        print("  instances: 0")
        return

    conflict_counts = [instance.independent_conflict_count for instance in instances]
    avg_soc = sum(instance.independent_soc for instance in instances) / len(instances)
    avg_makespan = (
        sum(instance.independent_makespan for instance in instances) / len(instances)
    )

    print(f"{level.name}:")
    print(f"  instances: {len(instances)}")
    print(
        "  conflict range: "
        f"{min(conflict_counts)}"
        + (
            f"-{max(conflict_counts)}"
            if min(conflict_counts) != max(conflict_counts)
            else ""
        )
    )
    print(f"  avg independent SoC: {avg_soc:.1f}")
    print(f"  avg independent makespan: {avg_makespan:.1f}")


def _print_generation_summary(
    manifest,
    attempts_by_agent_count: dict[int, int],
    output_path: Path,
    *,
    eligible_count: int,
    feasible_count: int,
    failed_count: int,
    precompute_time_s: float,
) -> None:
    print()
    print("=" * 60)
    print("MAPF BENCHMARK INSTANCE GENERATION SUMMARY")
    print("=" * 60)
    print()
    print(f"Map: {manifest.map_name}")
    print(f"Scenario: {manifest.scenario_name}")
    print(f"Seed: {manifest.seed}")
    print(f"Max timestep: {manifest.max_timestep}")
    print(f"Min reference length: {manifest.min_reference_length:g}")
    print(
        "Interaction thresholds: "
        f"LOW <= {INTERACTION_LOW_MAX_CONFLICTS}, "
        f"MEDIUM = {INTERACTION_MEDIUM_MIN_CONFLICTS}-{INTERACTION_MEDIUM_MAX_CONFLICTS}, "
        f"HIGH >= {INTERACTION_HIGH_MIN_CONFLICTS}"
    )
    print(f"Eligible scenarios: {eligible_count}")
    print(f"Feasible scenarios: {feasible_count}")
    print(f"Failed precompute: {failed_count}")
    print(f"Precompute time: {precompute_time_s:.1f} s")
    print(f"Total instances: {len(manifest.instances)}")
    print(f"Output manifest: {output_path}")
    print()

    by_agent_count: dict[int, list] = defaultdict(list)
    for instance in manifest.instances:
        by_agent_count[instance.agent_count].append(instance)

    for agent_count in sorted(by_agent_count):
        instances = by_agent_count[agent_count]
        print(f"Agents: {agent_count}")
        print(f"  attempts: {attempts_by_agent_count.get(agent_count, 0)}")
        print()

        for level in MAPFInteractionLevel:
            level_instances = [
                instance
                for instance in instances
                if instance.interaction_level == level
            ]
            _print_level_summary(level, level_instances)
            print()

        print("-" * 40)
        print()


def _print_precompute_progress(progress: StaticPrecomputeProgress) -> None:
    print(f"STATIC PRECOMPUTE {progress.completed} / {progress.total}")
    print(f"  feasible: {progress.feasible}")
    print(f"  no spatial path: {progress.no_spatial_path}")
    print(f"  over horizon: {progress.over_horizon}")


def _print_sampling_progress(
    agent_count: int,
    attempts: int,
    max_attempts: int,
    needed: dict[MAPFInteractionLevel, int],
    found: dict[MAPFInteractionLevel, int],
    *,
    instances_per_level: int,
) -> None:
    print(f"Agents: {agent_count}")
    print(f"Attempt: {attempts} / {max_attempts}")
    print("Accepted:")
    for level in MAPFInteractionLevel:
        print(
            f"  {level.name:6s}: {found[level]:3d} / {instances_per_level}"
        )


def _print_accepted_instance(instance: MAPFBenchmarkInstance, attempt: int) -> None:
    print(f"ACCEPTED {instance.instance_id}")
    print(f"  attempt={attempt}")
    print(f"  conflicts={instance.independent_conflict_count}")
    print(f"  pairs={instance.conflicting_agent_pair_count}")
    print(f"  SoC={instance.independent_soc}")
    print(f"  makespan={instance.independent_makespan}")


def main() -> None:
    args = parse_args()

    grid_map = load_moving_ai_map(args.map)
    scenarios = load_moving_ai_scenarios(args.scen)

    print("Preparing MAPF benchmark source pool...")
    print("Using bounded static 4-connected independent-path precomputation.")
    print("Space-Time A* is NOT used during benchmark source preparation.")
    print()

    precompute_start = time.perf_counter()

    def _precompute_callback(progress: StaticPrecomputeProgress) -> None:
        if args.precompute_progress_every > 0:
            elapsed = time.perf_counter() - precompute_start
            _print_precompute_progress(progress)
            print(f"  elapsed: {elapsed:.1f} s")

    if args.precompute_progress_every > 0:
        print("Static precomputing independent paths:")

    precompute_start = time.perf_counter()
    source_pool = prepare_benchmark_source_pool(
        grid_map=grid_map,
        scenarios=scenarios,
        min_reference_length=args.min_reference_length,
        max_timestep=args.max_timestep,
        precompute_progress_every=args.precompute_progress_every,
        precompute_progress_callback=_precompute_callback
        if args.precompute_progress_every > 0
        else None,
    )
    precompute_time_s = time.perf_counter() - precompute_start

    eligible_count = len(source_pool.eligible_indices)
    feasible_count = len(source_pool.feasible_indices)
    failed_count = len(source_pool.precompute_result.failed_scenario_indices)

    precompute_result = source_pool.precompute_result

    print()
    print(f"Eligible MovingAI scenarios: {eligible_count}")
    print("REACHABILITY PREPROCESSING COMPLETE")
    print(f"  components: {precompute_result.reachability_component_count}")
    print(f"  walkable cells: {precompute_result.reachability_walkable_cells}")
    print(f"  elapsed: {precompute_result.reachability_preprocess_s:.3f} s")
    print("BOUNDED STATIC PRECOMPUTE COMPLETE")
    print(f"  eligible MovingAI sources: {eligible_count}")
    print(f"  sources spatially reachable: {precompute_result.spatially_reachable_count}")
    print(f"  within horizon: {feasible_count}")
    print(f"  over horizon: {precompute_result.over_horizon_count}")
    print(f"  no spatial path: {precompute_result.no_spatial_path_count}")
    print(f"  feasible source pool: {feasible_count}")
    print(f"  bounded precompute elapsed: {precompute_result.bounded_preprocess_s:.1f} s")
    print(f"  total source preparation elapsed: {precompute_time_s:.1f} s")
    if eligible_count:
        print(f"  paths/sec: {eligible_count / precompute_time_s:.1f}")

    sampling_start = time.perf_counter()
    generation_result = generate_benchmark_manifest(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_name=args.scen.name,
        agent_counts=tuple(args.agent_counts),
        instances_per_level=args.instances_per_level,
        max_timestep=args.max_timestep,
        seed=args.seed,
        min_reference_length=args.min_reference_length,
        max_attempts=args.max_attempts,
        source_pool=source_pool,
        sampling_progress_every=args.sampling_progress_every,
        sampling_progress_callback=lambda agent_count, attempts, max_attempts, needed, found: _print_sampling_progress(
            agent_count,
            attempts,
            max_attempts,
            needed,
            found,
            instances_per_level=args.instances_per_level,
        ),
        accepted_instance_callback=_print_accepted_instance,
        generation_start_callback=lambda agent_count: print(
            f"\nGenerating {agent_count}-agent benchmark instances..."
        ),
    )
    sampling_time_s = time.perf_counter() - sampling_start

    save_start = time.perf_counter()
    manifest = generation_result.manifest
    save_benchmark_manifest(manifest, args.output)
    save_time_s = time.perf_counter() - save_start

    _print_generation_summary(
        manifest=manifest,
        attempts_by_agent_count=generation_result.attempts_by_agent_count,
        output_path=args.output,
        eligible_count=eligible_count,
        feasible_count=feasible_count,
        failed_count=failed_count,
        precompute_time_s=precompute_time_s,
    )
    print(f"Sampling time: {sampling_time_s:.2f} s")
    print(f"Manifest save time: {save_time_s:.2f} s")


if __name__ == "__main__":
    main()
