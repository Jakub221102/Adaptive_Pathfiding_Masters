from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathfinding.src.experiments.mapf_benchmark_instances import (
    _eligible_scenario_indices,
    precompute_independent_paths,
)
from pathfinding.src.loaders.map_loader import load_moving_ai_map
from pathfinding.src.loaders.scen_loader import load_moving_ai_scenarios

DEFAULT_MAP_PATH = _REPO_ROOT / "Data" / "bg512-map" / "AR0204SR.map"
DEFAULT_SCEN_PATH = _REPO_ROOT / "Data" / "bg512-scen" / "AR0204SR.map.scen"
DEFAULT_MAX_TIMESTEP = 512
DEFAULT_MIN_REFERENCE_LENGTH = 20.0
DEFAULT_LIMIT = 20
DEFAULT_PROGRESS_EVERY = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute independent Space-Time A* paths for MovingAI scenarios.",
    )
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP_PATH)
    parser.add_argument("--scen", type=Path, default=DEFAULT_SCEN_PATH)
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum number of eligible scenarios to precompute.",
    )
    parser.add_argument("--max-timestep", type=int, default=DEFAULT_MAX_TIMESTEP)
    parser.add_argument(
        "--min-reference-length",
        type=float,
        default=DEFAULT_MIN_REFERENCE_LENGTH,
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Print progress every N scenarios (0 disables).",
    )
    return parser.parse_args()


def _print_progress(completed: int, total: int) -> None:
    print(f"Precomputing independent paths: {completed} / {total}")


def main() -> None:
    args = parse_args()

    if args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.progress_every < 0:
        raise ValueError("progress_every must be non-negative")

    grid_map = load_moving_ai_map(args.map)
    scenarios = load_moving_ai_scenarios(args.scen)
    eligible = _eligible_scenario_indices(
        scenarios=scenarios,
        grid_map=grid_map,
        min_reference_length=args.min_reference_length,
    )
    selected = eligible[: args.limit]

    print(
        f"Eligible scenarios (min_reference_length>={args.min_reference_length:g}): "
        f"{len(eligible)}"
    )
    print(f"Precomputing: {len(selected)}")
    print(f"Max timestep: {args.max_timestep}")
    print()

    total_start = time.perf_counter()
    result = precompute_independent_paths(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_indices=selected,
        max_timestep=args.max_timestep,
        progress_every=args.progress_every,
        progress_callback=_print_progress if args.progress_every > 0 else None,
    )
    total_elapsed = time.perf_counter() - total_start

    succeeded = len(result.paths)
    failed = len(result.failed_scenario_indices)
    mean_per_path = total_elapsed / len(selected) if selected else 0.0

    print()
    print("=" * 60)
    print("PRECOMPUTE SUMMARY")
    print("=" * 60)
    print(f"Requested: {len(selected)}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")
    print(f"Total runtime: {total_elapsed:.3f} s")
    if selected:
        print(f"Mean per scenario: {mean_per_path:.4f} s")
        print(f"Paths/sec: {len(selected) / total_elapsed:.1f}")
    if len(eligible) > 0 and mean_per_path > 0:
        estimate = len(eligible) * mean_per_path
        print(f"Estimated full-pool time ({len(eligible)} eligible): {estimate:.1f} s")
    if failed:
        print(f"Failed indices: {list(result.failed_scenario_indices)}")


if __name__ == "__main__":
    main()
