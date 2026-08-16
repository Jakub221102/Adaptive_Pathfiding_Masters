from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathfinding.src.algorithms.mapf.cbs import build_cbs_root, solve_cbs_with_stats
from pathfinding.src.algorithms.mapf.conflicts import detect_conflicts
from pathfinding.src.algorithms.mapf.demo_scenario import (
    NON_TRIVIAL_DEMO_AGENT_COUNT,
    build_non_trivial_demo_scenario,
)
from pathfinding.src.algorithms.mapf.metrics import makespan, sum_of_costs
from pathfinding.src.algorithms.mapf.prioritized_planning import plan_prioritized
from pathfinding.src.loaders.map_loader import load_moving_ai_map
from pathfinding.src.loaders.scen_loader import load_moving_ai_scenarios

DEFAULT_MAX_TIMESTEP = 512
DEFAULT_MAX_EXPANDED_NODES = 1000
DEFAULT_MAP_PATH = _REPO_ROOT / "Data" / "bg512-map" / "AR0204SR.map"
DEFAULT_SCEN_PATH = _REPO_ROOT / "Data" / "bg512-scen" / "AR0204SR.map.scen"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded PP vs CBS diagnostic comparison on the frozen "
            "AR0204SR non-trivial MAPF demo fixture."
        ),
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
        "--max-timestep",
        type=int,
        default=DEFAULT_MAX_TIMESTEP,
        help="Planning horizon passed to MAPF planners.",
    )
    parser.add_argument(
        "--max-expanded-nodes",
        type=int,
        default=DEFAULT_MAX_EXPANDED_NODES,
        help="Diagnostic CBS CT expansion limit (conflicting nodes expanded).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    grid_map = load_moving_ai_map(args.map)
    scenarios = load_moving_ai_scenarios(args.scen)
    selection = build_non_trivial_demo_scenario(
        scenarios=scenarios,
        grid_map=grid_map,
    )
    scenario = selection.scenario

    print("=" * 60)
    print("CBS DIAGNOSTIC")
    print("=" * 60)
    print()
    print(f"Map: {args.map.stem}")
    print(f"Agents: {NON_TRIVIAL_DEMO_AGENT_COUNT}")
    print(f"Horizon: {args.max_timestep}")
    print(f"Max expanded CT nodes: {args.max_expanded_nodes}")
    print()

    root = build_cbs_root(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=args.max_timestep,
    )
    if root is None:
        print("Root: N/A — individual path infeasible")
    else:
        print("Root:")
        print(f"  SoC: {root.cost}")
        print(f"  Conflicts: {len(root.conflicts)}")
    print()

    print("PRIORITIZED PLANNING")
    print("--------------------")
    pp_start = time.perf_counter()
    pp_result = plan_prioritized(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=args.max_timestep,
    )
    pp_elapsed = time.perf_counter() - pp_start

    print(f"Success: {pp_result.success}")
    print(f"Time: {pp_elapsed:.3f} s")
    if pp_result.success:
        pp_conflicts = detect_conflicts(pp_result.paths)
        print(f"SoC: {sum_of_costs(pp_result.paths)}")
        print(f"Makespan: {makespan(pp_result.paths)}")
        print(f"Conflicts: {len(pp_conflicts)}")
    else:
        print("SoC: N/A")
        print("Makespan: N/A")
        print("Conflicts: N/A")
    print()

    print("CBS")
    print("---")
    cbs_start = time.perf_counter()
    cbs_run = solve_cbs_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=args.max_timestep,
        max_expanded_nodes=args.max_expanded_nodes,
    )
    cbs_elapsed = time.perf_counter() - cbs_start
    stats = cbs_run.stats

    print(f"Termination: {cbs_run.termination_reason}")
    print(f"Time: {cbs_elapsed:.3f} s")
    print(f"Expanded CT nodes: {stats.expanded_ct_nodes}")
    print(f"Generated CT nodes: {stats.generated_ct_nodes}")
    print(f"Low-level replans: {stats.low_level_replans}")
    print(f"Max OPEN size: {stats.max_open_size}")
    print()

    print("Solution:")
    if cbs_run.termination_reason == "expansion_limit":
        print("  N/A — diagnostic expansion limit reached")
    elif cbs_run.result is None or not cbs_run.result.success:
        print("  N/A — CBS reported failure")
    else:
        solution = cbs_run.result
        solution_conflicts = detect_conflicts(solution.paths)
        print(f"  SoC: {sum_of_costs(solution.paths)}")
        print(f"  Makespan: {makespan(solution.paths)}")
        print(f"  Conflicts: {len(solution_conflicts)}")


if __name__ == "__main__":
    main()
