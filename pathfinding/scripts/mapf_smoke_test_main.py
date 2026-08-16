from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathfinding.src.algorithms.mapf.conflicts import detect_conflicts
from pathfinding.src.algorithms.mapf.metrics import makespan, sum_of_costs
from pathfinding.src.algorithms.mapf.models import AgentPath, MAPFResult, MAPFScenario
from pathfinding.src.algorithms.mapf.prioritized_planning import plan_prioritized
from pathfinding.src.algorithms.mapf.smoke_scenario import build_mapf_scenario
from pathfinding.src.core.models import GridMap
from pathfinding.src.loaders.map_loader import load_moving_ai_map
from pathfinding.src.loaders.scen_loader import load_moving_ai_scenarios

DEFAULT_AGENT_COUNTS: tuple[int, ...] = (5, 10)
DEFAULT_MAX_TIMESTEP = 512

DEFAULT_MAP_PATH = _REPO_ROOT / "Data" / "bg512-map" / "AR0204SR.map"
DEFAULT_SCEN_PATH = _REPO_ROOT / "Data" / "bg512-scen" / "AR0204SR.map.scen"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real-map MAPF smoke test using Prioritized Planning.",
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
    return parser.parse_args()


def _path_cost(path: AgentPath) -> int:
    return len(path.states) - 1


def _format_position(row: int, col: int) -> str:
    return f"({row},{col})"


def _print_agent_selection(scenario: MAPFScenario) -> None:
    print("Selected agents:")
    for agent in scenario.agents:
        print(
            "  "
            f"Agent {agent.agent_id} | "
            f"start={_format_position(agent.start.row, agent.start.col)} | "
            f"goal={_format_position(agent.goal.row, agent.goal.col)}"
        )


def _print_agent_paths(result: MAPFResult) -> None:
    print("Agent paths:")
    for path in result.paths:
        final_state = path.states[-1]
        print(
            "  "
            f"Agent {path.agent_id} | "
            f"start={_format_position(path.states[0].row, path.states[0].col)} | "
            f"goal={_format_position(final_state.row, final_state.col)} | "
            f"cost={_path_cost(path)} | "
            f"arrival_t={final_state.timestep}"
        )


def _run_agent_count(
    grid_map: GridMap,
    scenarios,
    agent_count: int,
    max_timestep: int,
) -> None:
    mapf_scenario = build_mapf_scenario(
        scenarios=scenarios,
        grid_map=grid_map,
        agent_count=agent_count,
    )

    print("=" * 60)
    print(f"Agents: {agent_count}")
    print(f"Max timestep: {max_timestep}")
    print()

    _print_agent_selection(mapf_scenario)

    planning_start = time.perf_counter()
    result = plan_prioritized(
        grid_map=grid_map,
        scenario=mapf_scenario,
        max_timestep=max_timestep,
    )
    planning_time_ms = (time.perf_counter() - planning_start) * 1000.0

    print()
    print(f"Success: {result.success}")
    print(f"Planning time: {planning_time_ms:.3f} ms")

    if not result.success:
        print("Sum of Costs: N/A")
        print("Makespan: N/A")
        print("Final conflicts: N/A")
        return

    conflicts = detect_conflicts(result.paths)
    if conflicts:
        raise AssertionError(
            "plan_prioritized returned success=True but detect_conflicts found "
            f"{len(conflicts)} conflict(s)."
        )

    soc = sum_of_costs(result.paths)
    makespan_value = makespan(result.paths)

    print(f"Sum of Costs: {soc}")
    print(f"Makespan: {makespan_value}")
    print(f"Final conflicts: {len(conflicts)}")
    print()
    _print_agent_paths(result)


def run_smoke_test(
    map_path: Path,
    scen_path: Path,
    max_timestep: int,
    agent_counts: tuple[int, ...] = DEFAULT_AGENT_COUNTS,
) -> None:
    grid_map = load_moving_ai_map(map_path)
    scenarios = load_moving_ai_scenarios(scen_path)

    map_stem = map_path.stem

    print("=" * 60)
    print("MAPF REAL-MAP SMOKE TEST")
    print("=" * 60)
    print()
    print(f"Map: {map_stem}")
    print(f"Map size: {grid_map.height}x{grid_map.width}")
    print("Algorithm: Fixed-Priority Prioritized Planning")
    print()

    for agent_count in agent_counts:
        _run_agent_count(
            grid_map=grid_map,
            scenarios=scenarios,
            agent_count=agent_count,
            max_timestep=max_timestep,
        )
        print()


def main() -> None:
    args = parse_args()
    run_smoke_test(
        map_path=args.map,
        scen_path=args.scen,
        max_timestep=args.max_timestep,
    )


if __name__ == "__main__":
    main()
