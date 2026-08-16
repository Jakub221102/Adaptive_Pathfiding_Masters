from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathfinding.src.algorithms.mapf.conflicts import detect_conflicts
from pathfinding.src.algorithms.mapf.demo_scenario import build_non_trivial_demo_scenario
from pathfinding.src.algorithms.mapf.prioritized_planning import plan_prioritized
from pathfinding.src.loaders.map_loader import load_moving_ai_map
from pathfinding.src.loaders.scen_loader import load_moving_ai_scenarios
from pathfinding.src.visualization.mapf_viewer import MapfViewerConfig, visualize_mapf_solution

DEFAULT_MAX_TIMESTEP = 512
DEFAULT_MAP_PATH = _REPO_ROOT / "Data" / "bg512-map" / "AR0204SR.map"
DEFAULT_SCEN_PATH = _REPO_ROOT / "Data" / "bg512-scen" / "AR0204SR.map.scen"
DEFAULT_TIMESTEPS_PER_SECOND = 8.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate the frozen AR0204SR non-trivial MAPF demo solution.",
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
        help="Planning horizon passed to plan_prioritized().",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_TIMESTEPS_PER_SECOND,
        help="MAPF timesteps advanced per second during playback.",
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

    result = plan_prioritized(
        grid_map=grid_map,
        scenario=selection.scenario,
        max_timestep=args.max_timestep,
    )

    if not result.success:
        print("MAPF demo planning failed: plan_prioritized returned success=False.")
        raise SystemExit(1)

    conflicts = detect_conflicts(result.paths)
    if conflicts:
        raise AssertionError(
            "MAPF demo planning reported success but detect_conflicts found "
            f"{len(conflicts)} conflict(s); refusing to animate invalid solution."
        )

    visualize_mapf_solution(
        grid_map=grid_map,
        scenario=selection.scenario,
        paths=result.paths,
        config=MapfViewerConfig(timesteps_per_second=args.fps),
    )


if __name__ == "__main__":
    main()
