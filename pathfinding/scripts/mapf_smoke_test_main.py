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
from pathfinding.src.algorithms.mapf.demo_diagnostics import (
    count_agents_with_changed_path,
    count_agents_with_wait,
    count_total_wait_steps,
)
from pathfinding.src.algorithms.mapf.demo_scenario import (
    NON_TRIVIAL_DEMO_AGENT_COUNT,
    NON_TRIVIAL_DEMO_MIN_OPTIMAL_LENGTH,
    NON_TRIVIAL_DEMO_SCENARIO_INDICES,
    DemoScenarioSelection,
    build_non_trivial_demo_scenario,
)
from pathfinding.src.algorithms.mapf.metrics import makespan, sum_of_costs
from pathfinding.src.algorithms.mapf.models import (
    AgentPath,
    EdgeConflict,
    MAPFAgent,
    MAPFResult,
    MAPFScenario,
    VertexConflict,
)
from pathfinding.src.algorithms.mapf.prioritized_planning import plan_prioritized
from pathfinding.src.algorithms.mapf.smoke_scenario import build_mapf_scenario
from pathfinding.src.algorithms.mapf.space_time_astar import find_path
from pathfinding.src.core.models import GridMap
from pathfinding.src.loaders.map_loader import load_moving_ai_map
from pathfinding.src.loaders.scen_loader import load_moving_ai_scenarios

DEFAULT_AGENT_COUNTS: tuple[int, ...] = (5, 10)
DEFAULT_MAX_TIMESTEP = 512

DEFAULT_MAP_PATH = _REPO_ROOT / "Data" / "bg512-map" / "AR0204SR.map"
DEFAULT_SCEN_PATH = _REPO_ROOT / "Data" / "bg512-scen" / "AR0204SR.map.scen"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-map MAPF smoke tests and optional non-trivial demo.",
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
        "--demo",
        action="store_true",
        help="Run the non-trivial real-map MAPF demo.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run both the basic smoke test and the non-trivial demo.",
    )
    return parser.parse_args()


def _path_cost(path: AgentPath) -> int:
    return len(path.states) - 1


def _format_position(row: int, col: int) -> str:
    return f"({row},{col})"


def _assert_success_without_conflicts(result: MAPFResult) -> None:
    if not result.success:
        return

    conflicts = detect_conflicts(result.paths)
    if conflicts:
        raise AssertionError(
            "plan_prioritized returned success=True but detect_conflicts found "
            f"{len(conflicts)} conflict(s)."
        )


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


def _plan_independent_paths(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
) -> tuple[AgentPath, ...] | None:
    independent_paths: list[AgentPath] = []

    for agent in scenario.agents:
        path = find_path(
            grid_map=grid_map,
            agent=agent,
            max_timestep=max_timestep,
        )
        if path is None:
            return None
        independent_paths.append(path)

    return tuple(independent_paths)


def _format_conflict_summary(conflicts: tuple[VertexConflict | EdgeConflict, ...]) -> None:
    if not conflicts:
        print("Conflict summary: none")
        return

    print("Conflict summary:")
    for conflict in conflicts[:10]:
        if isinstance(conflict, VertexConflict):
            print(
                "  "
                f"vertex t={conflict.timestep} "
                f"agents=({conflict.agent1_id},{conflict.agent2_id}) "
                f"cell={_format_position(conflict.row, conflict.col)}"
            )
            continue

        print(
            "  "
            f"edge t={conflict.timestep} "
            f"agents=({conflict.agent1_id},{conflict.agent2_id}) "
            f"a1={_format_position(conflict.agent1_from_row, conflict.agent1_from_col)}"
            f"->{_format_position(conflict.agent1_to_row, conflict.agent1_to_col)} "
            f"a2={_format_position(conflict.agent2_from_row, conflict.agent2_from_col)}"
            f"->{_format_position(conflict.agent2_to_row, conflict.agent2_to_col)}"
        )

    if len(conflicts) > 10:
        print(f"  ... and {len(conflicts) - 10} more")


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

    _assert_success_without_conflicts(result)

    soc = sum_of_costs(result.paths)
    makespan_value = makespan(result.paths)

    print(f"Sum of Costs: {soc}")
    print(f"Makespan: {makespan_value}")
    print(f"Final conflicts: 0")
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


def _print_demo_agents(selection: DemoScenarioSelection) -> None:
    print("AGENTS")
    print("------")
    for spec in selection.agents:
        reference_length = (
            "N/A"
            if spec.reference_optimal_length is None
            else f"{spec.reference_optimal_length:.8f}"
        )
        print(
            f"Agent {spec.agent_id} | scen_idx={spec.scenario_index} | "
            f"start={_format_position(spec.start.row, spec.start.col)} | "
            f"goal={_format_position(spec.goal.row, spec.goal.col)} | "
            f"reference_length={reference_length}"
        )


def run_non_trivial_demo(
    map_path: Path,
    scen_path: Path,
    max_timestep: int,
) -> None:
    grid_map = load_moving_ai_map(map_path)
    scenarios = load_moving_ai_scenarios(scen_path)
    selection = build_non_trivial_demo_scenario(
        scenarios=scenarios,
        grid_map=grid_map,
    )

    print("=" * 60)
    print("NON-TRIVIAL REAL-MAP MAPF DEMO")
    print("=" * 60)
    print()
    print(f"Map: {map_path.stem}")
    print(f"Agents: {NON_TRIVIAL_DEMO_AGENT_COUNT}")
    print(f"Max timestep: {max_timestep}")
    print(f"Scenario indices: {list(NON_TRIVIAL_DEMO_SCENARIO_INDICES)}")
    print()
    print("Individual path filter:")
    print(f"  Minimum MovingAI optimal length: {NON_TRIVIAL_DEMO_MIN_OPTIMAL_LENGTH:g}")
    print()

    independent_paths = _plan_independent_paths(
        grid_map=grid_map,
        scenario=selection.scenario,
        max_timestep=max_timestep,
    )

    print("INDEPENDENT PATHS")
    print("-----------------")

    if independent_paths is None:
        print("Success: False")
        print("Sum of Costs: N/A")
        print("Makespan: N/A")
        print("Conflicts: N/A")
        return

    independent_conflicts = detect_conflicts(independent_paths)
    independent_soc = sum_of_costs(independent_paths)
    independent_makespan = makespan(independent_paths)

    print("Success: True")
    print(f"Sum of Costs: {independent_soc}")
    print(f"Makespan: {independent_makespan}")
    print(f"Conflicts: {len(independent_conflicts)}")
    _format_conflict_summary(independent_conflicts)
    print()

    if not independent_conflicts:
        raise AssertionError(
            "Non-trivial demo fixture must have at least one independent conflict."
        )

    print("PRIORITIZED PLANNING")
    print("--------------------")

    planning_start = time.perf_counter()
    result = plan_prioritized(
        grid_map=grid_map,
        scenario=selection.scenario,
        max_timestep=max_timestep,
    )
    planning_time_ms = (time.perf_counter() - planning_start) * 1000.0

    print(f"Success: {result.success}")
    print(f"Planning time: {planning_time_ms:.3f} ms")

    if not result.success:
        print("Sum of Costs: N/A")
        print("Makespan: N/A")
        print("Final conflicts: N/A")
        print()
        _print_demo_agents(selection)
        return

    _assert_success_without_conflicts(result)

    pp_soc = sum_of_costs(result.paths)
    pp_makespan = makespan(result.paths)
    final_conflicts = detect_conflicts(result.paths)

    print(f"Sum of Costs: {pp_soc}")
    print(f"Makespan: {pp_makespan}")
    print(f"Final conflicts: {len(final_conflicts)}")
    print()

    changed_agents = count_agents_with_changed_path(independent_paths, result.paths)
    agents_with_wait = count_agents_with_wait(result.paths)
    total_wait_steps = count_total_wait_steps(result.paths)

    print("COORDINATION EFFECT")
    print("-------------------")
    print(f"Conflicts resolved: {len(independent_conflicts)}")
    print(f"Agents with changed path: {changed_agents}")
    print(f"Agents with WAIT: {agents_with_wait}")
    print(f"Total WAIT steps: {total_wait_steps}")
    print(f"SoC overhead: {pp_soc - independent_soc}")
    print(f"Makespan overhead: {pp_makespan - independent_makespan}")
    print()
    _print_demo_agents(selection)


def main() -> None:
    args = parse_args()

    run_basic = not args.demo or args.all
    run_demo = args.demo or args.all

    if run_basic:
        run_smoke_test(
            map_path=args.map,
            scen_path=args.scen,
            max_timestep=args.max_timestep,
        )

    if run_demo:
        if run_basic:
            print()
        run_non_trivial_demo(
            map_path=args.map,
            scen_path=args.scen,
            max_timestep=args.max_timestep,
        )


if __name__ == "__main__":
    main()
