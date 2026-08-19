from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathfinding.src.algorithms.mapf.cbs import (
    CBSRunResult,
    CBSStats,
    build_cbs_root,
    solve_cbs_cardinal_first_with_stats,
    solve_cbs_with_stats,
)
from pathfinding.src.algorithms.mapf.conflicts import detect_conflicts
from pathfinding.src.algorithms.mapf.demo_scenario import (
    NON_TRIVIAL_DEMO_AGENT_COUNT,
    build_non_trivial_demo_scenario,
)
from pathfinding.src.algorithms.mapf.metrics import makespan, sum_of_costs
from pathfinding.src.algorithms.mapf.prioritized_planning import plan_prioritized
from pathfinding.src.algorithms.mapf.models import MAPFResult, MAPFScenario
from pathfinding.src.core.models import GridMap
from pathfinding.src.loaders.map_loader import load_moving_ai_map
from pathfinding.src.loaders.scen_loader import load_moving_ai_scenarios

DEFAULT_MAX_TIMESTEP = 512
DEFAULT_MAX_EXPANDED_NODES = 1000
DEFAULT_COMPARE_MAX_EXPANDED_NODES = 100
DEFAULT_MAP_PATH = _REPO_ROOT / "Data" / "bg512-map" / "AR0204SR.map"
DEFAULT_SCEN_PATH = _REPO_ROOT / "Data" / "bg512-scen" / "AR0204SR.map.scen"
_MAX_FULL_DISTRIBUTION_BUCKETS = 20


@dataclass(frozen=True, slots=True)
class PPReference:
    result: MAPFResult
    elapsed: float


@dataclass(frozen=True, slots=True)
class CBSRunReport:
    label: str
    run: CBSRunResult
    elapsed: float


def _duplicate_rate(duplicates: int, generated: int) -> str:
    if generated == 0:
        return "N/A"
    return f"{duplicates / generated * 100:.1f}%"


def _combined_low_level_searches(stats: CBSStats) -> int:
    return stats.low_level_replans + stats.classification_low_level_searches


def _per_expansion(value: int, expanded: int) -> str:
    if expanded == 0:
        return "N/A"
    return f"{value / expanded:.2f}"


def _selected_percentage(count: int, expanded: int) -> str:
    if expanded == 0:
        return "N/A"
    return f"{count / expanded * 100:.1f}%"


def _print_cost_distribution(
    label: str,
    distribution: tuple[tuple[int, int], ...],
) -> None:
    print(label)
    if not distribution:
        print("  (empty)")
        return

    if len(distribution) <= _MAX_FULL_DISTRIBUTION_BUCKETS:
        for cost, count in distribution:
            print(f"  {cost}: {count}")
        return

    print("  (showing lowest 10 and highest 10 buckets)")
    for cost, count in distribution[:10]:
        print(f"  {cost}: {count}")
    print("  ...")
    for cost, count in distribution[-10:]:
        print(f"  {cost}: {count}")


def _print_soc_plateau_summary(
    label: str,
    distribution: tuple[tuple[int, int], ...],
    expanded: int,
) -> None:
    if expanded == 0 or not distribution:
        print(f"{label}: (empty)")
        return

    if len(distribution) == 1:
        cost, count = distribution[0]
        print(f"{label}: {count} / {expanded} expanded nodes at SoC {cost}")
        return

    parts = ", ".join(f"SoC {cost}: {count}" for cost, count in distribution)
    print(f"{label}: {parts} (total expanded {expanded})")


def _print_cbs_details(report: CBSRunReport) -> None:
    stats = report.run.stats
    combined = _combined_low_level_searches(stats)

    print(f"{report.label.upper()}")
    print("-" * len(report.label))
    print(f"Termination: {report.run.termination_reason}")
    print(f"Time: {report.elapsed:.3f} s")
    print(f"Expanded CT nodes: {stats.expanded_ct_nodes}")
    print(f"Generated CT nodes: {stats.generated_ct_nodes}")
    print(f"Max OPEN size: {stats.max_open_size}")
    print(f"Branch low-level replans: {stats.low_level_replans}")
    print(f"Classification low-level searches: {stats.classification_low_level_searches}")
    print(f"CBS branching/classification low-level searches: {combined}")
    print(
        "  Constraint duplicate rate: "
        f"{_duplicate_rate(stats.duplicate_constraint_signatures, stats.generated_ct_nodes)}"
    )
    print(
        "  Path duplicate rate: "
        f"{_duplicate_rate(stats.duplicate_path_signatures, stats.generated_ct_nodes)}"
    )
    print(f"Classified conflicts: {stats.classified_conflicts}")
    print()
    _print_cost_distribution(
        "Generated-node SoC distribution:",
        stats.generated_cost_distribution,
    )
    print()
    _print_cost_distribution(
        "Expanded-node SoC distribution:",
        stats.expanded_cost_distribution,
    )
    print()


def _print_selected_cardinality(report: CBSRunReport) -> None:
    stats = report.run.stats
    expanded = stats.expanded_ct_nodes

    print("CARDINAL-FIRST SELECTED CONFLICTS")
    print("---------------------------------")
    print(f"CARDINAL: {stats.selected_cardinal_conflicts} ({_selected_percentage(stats.selected_cardinal_conflicts, expanded)})")
    print(
        "SEMI_CARDINAL: "
        f"{stats.selected_semi_cardinal_conflicts} "
        f"({_selected_percentage(stats.selected_semi_cardinal_conflicts, expanded)})"
    )
    print(
        "NON_CARDINAL: "
        f"{stats.selected_non_cardinal_conflicts} "
        f"({_selected_percentage(stats.selected_non_cardinal_conflicts, expanded)})"
    )
    print()


def _print_comparison_table(basic: CBSRunReport, cardinal: CBSRunReport) -> None:
    basic_stats = basic.run.stats
    cardinal_stats = cardinal.run.stats
    basic_combined = _combined_low_level_searches(basic_stats)
    cardinal_combined = _combined_low_level_searches(cardinal_stats)

    rows = [
        ("Termination", basic.run.termination_reason, cardinal.run.termination_reason),
        ("Time (s)", f"{basic.elapsed:.3f}", f"{cardinal.elapsed:.3f}"),
        ("Expanded CT", str(basic_stats.expanded_ct_nodes), str(cardinal_stats.expanded_ct_nodes)),
        ("Generated CT", str(basic_stats.generated_ct_nodes), str(cardinal_stats.generated_ct_nodes)),
        ("Max OPEN", str(basic_stats.max_open_size), str(cardinal_stats.max_open_size)),
        ("Branch replans", str(basic_stats.low_level_replans), str(cardinal_stats.low_level_replans)),
        (
            "Classification searches",
            str(basic_stats.classification_low_level_searches),
            str(cardinal_stats.classification_low_level_searches),
        ),
        ("Combined LL searches", str(basic_combined), str(cardinal_combined)),
        (
            "Path duplicates",
            str(basic_stats.duplicate_path_signatures),
            str(cardinal_stats.duplicate_path_signatures),
        ),
        (
            "Constraint duplicates",
            str(basic_stats.duplicate_constraint_signatures),
            str(cardinal_stats.duplicate_constraint_signatures),
        ),
        (
            "Classified conflicts",
            str(basic_stats.classified_conflicts),
            str(cardinal_stats.classified_conflicts),
        ),
    ]

    label_width = max(len(row[0]) for row in rows)
    print("METRIC".ljust(label_width), "BASIC".rjust(12), "CARDINAL-FIRST".rjust(16))
    print("-" * (label_width + 30))
    for label, basic_value, cardinal_value in rows:
        print(label.ljust(label_width), basic_value.rjust(12), cardinal_value.rjust(16))
    print()


def _print_comparison_interpretation(
    *,
    expansion_budget: int,
    basic: CBSRunReport,
    cardinal: CBSRunReport,
) -> None:
    basic_stats = basic.run.stats
    cardinal_stats = cardinal.run.stats
    basic_combined = _combined_low_level_searches(basic_stats)
    cardinal_combined = _combined_low_level_searches(cardinal_stats)

    print("FIXTURE-LIMITED INTERPRETATION")
    print("------------------------------")
    print(
        f"On this 10-agent AR0204SR fixture and {expansion_budget}-node expansion budget:"
    )

    if basic.run.termination_reason == cardinal.run.termination_reason:
        print(f"- both variants reached {basic.run.termination_reason}")
    else:
        print(
            f"- Basic CBS terminated with {basic.run.termination_reason}; "
            f"Cardinal-First terminated with {cardinal.run.termination_reason}"
        )

    if basic_stats.generated_ct_nodes == cardinal_stats.generated_ct_nodes:
        print(f"- both generated {basic_stats.generated_ct_nodes} CT nodes")
    else:
        print(
            f"- Basic generated {basic_stats.generated_ct_nodes} CT nodes; "
            f"Cardinal-First generated {cardinal_stats.generated_ct_nodes}"
        )

    extra_classification = (
        cardinal_stats.classification_low_level_searches
        - basic_stats.classification_low_level_searches
    )
    if extra_classification > 0:
        print(
            f"- Cardinal-First performed {extra_classification} additional "
            "classification low-level searches"
        )

    print(
        f"- combined low-level searches per expansion: "
        f"Basic {_per_expansion(basic_combined, basic_stats.expanded_ct_nodes)}, "
        f"Cardinal-First {_per_expansion(cardinal_combined, cardinal_stats.expanded_ct_nodes)}"
    )
    if cardinal_stats.expanded_ct_nodes > 0:
        print(
            f"- classification searches per expansion (Cardinal-First): "
            f"{_per_expansion(cardinal_stats.classification_low_level_searches, cardinal_stats.expanded_ct_nodes)}"
        )

    non_cardinal_rate = _selected_percentage(
        cardinal_stats.selected_non_cardinal_conflicts,
        cardinal_stats.expanded_ct_nodes,
    )
    print(
        f"- Cardinal-First selected NON_CARDINAL conflicts on "
        f"{cardinal_stats.selected_non_cardinal_conflicts} / "
        f"{cardinal_stats.expanded_ct_nodes} expansions ({non_cardinal_rate})"
    )
    print()


def _run_pp_reference(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
) -> PPReference:
    pp_start = time.perf_counter()
    pp_result = plan_prioritized(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
    )
    pp_elapsed = time.perf_counter() - pp_start
    return PPReference(result=pp_result, elapsed=pp_elapsed)


def _print_pp_reference(pp: PPReference) -> None:
    print("PRIORITIZED PLANNING")
    print("--------------------")
    print(f"Success: {pp.result.success}")
    print(f"Time: {pp.elapsed:.3f} s")
    if pp.result.success:
        pp_conflicts = detect_conflicts(pp.result.paths)
        print(f"SoC: {sum_of_costs(pp.result.paths)}")
        print(f"Makespan: {makespan(pp.result.paths)}")
        print(f"Conflicts: {len(pp_conflicts)}")
    else:
        print("SoC: N/A")
        print("Makespan: N/A")
        print("Conflicts: N/A")
    print()


def _run_basic_cbs(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
    max_expanded_nodes: int,
) -> CBSRunReport:
    start = time.perf_counter()
    run = solve_cbs_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
        max_expanded_nodes=max_expanded_nodes,
    )
    elapsed = time.perf_counter() - start
    return CBSRunReport(label="Basic CBS", run=run, elapsed=elapsed)


def _run_cardinal_first_cbs(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
    max_expanded_nodes: int,
) -> CBSRunReport:
    start = time.perf_counter()
    run = solve_cbs_cardinal_first_with_stats(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
        max_expanded_nodes=max_expanded_nodes,
    )
    elapsed = time.perf_counter() - start
    return CBSRunReport(label="Cardinal-First CBS", run=run, elapsed=elapsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded PP vs CBS diagnostic comparison on the frozen "
            "AR0204SR non-trivial MAPF demo fixture."
        ),
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run matched Basic CBS vs Cardinal-First CBS comparison.",
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
        default=None,
        help=(
            "Diagnostic CBS CT expansion limit (conflicting nodes expanded). "
            f"Defaults to {DEFAULT_COMPARE_MAX_EXPANDED_NODES} in --compare mode "
            f"and {DEFAULT_MAX_EXPANDED_NODES} otherwise."
        ),
    )
    return parser.parse_args()


def _resolve_max_expanded_nodes(args: argparse.Namespace) -> int:
    if args.max_expanded_nodes is not None:
        return args.max_expanded_nodes
    if args.compare:
        return DEFAULT_COMPARE_MAX_EXPANDED_NODES
    return DEFAULT_MAX_EXPANDED_NODES


def _print_shared_setup(
    *,
    map_stem: str,
    max_timestep: int,
    max_expanded_nodes: int,
    compare: bool,
) -> None:
    title = "CBS COMPARISON DIAGNOSTIC" if compare else "CBS DIAGNOSTIC"
    print("=" * 60)
    print(title)
    print("=" * 60)
    print()
    print(f"Map: {map_stem}")
    print(f"Scenario: {map_stem}")
    print(f"Agents: {NON_TRIVIAL_DEMO_AGENT_COUNT}")
    print(f"Horizon: {max_timestep}")
    print(f"Expansion budget: {max_expanded_nodes}")
    print()


def _print_root_summary(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
) -> None:
    root = build_cbs_root(
        grid_map=grid_map,
        scenario=scenario,
        max_timestep=max_timestep,
    )
    if root is None:
        print("Root: N/A — individual path infeasible")
    else:
        print("Root:")
        print(f"  SoC: {root.cost}")
        print(f"  Conflicts: {len(root.conflicts)}")
    print()


def run_compare_mode(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
    max_expanded_nodes: int,
) -> None:
    _print_pp_reference(_run_pp_reference(grid_map, scenario, max_timestep))

    basic = _run_basic_cbs(grid_map, scenario, max_timestep, max_expanded_nodes)
    cardinal = _run_cardinal_first_cbs(grid_map, scenario, max_timestep, max_expanded_nodes)

    _print_comparison_table(basic, cardinal)
    _print_cbs_details(basic)
    _print_cbs_details(cardinal)
    _print_selected_cardinality(cardinal)

    print("SOC PLATEAU COMPARISON")
    print("----------------------")
    _print_soc_plateau_summary(
        "Basic expanded SoC",
        basic.run.stats.expanded_cost_distribution,
        basic.run.stats.expanded_ct_nodes,
    )
    _print_soc_plateau_summary(
        "Cardinal-First expanded SoC",
        cardinal.run.stats.expanded_cost_distribution,
        cardinal.run.stats.expanded_ct_nodes,
    )
    print()

    _print_comparison_interpretation(
        expansion_budget=max_expanded_nodes,
        basic=basic,
        cardinal=cardinal,
    )


def run_single_cbs_mode(
    grid_map: GridMap,
    scenario: MAPFScenario,
    max_timestep: int,
    max_expanded_nodes: int,
) -> None:
    _print_pp_reference(_run_pp_reference(grid_map, scenario, max_timestep))
    basic = _run_basic_cbs(grid_map, scenario, max_timestep, max_expanded_nodes)
    _print_cbs_details(basic)

    stats = basic.run.stats
    print("CT STRUCTURE ANALYSIS")
    print("---------------------")
    print(f"Generated CT nodes: {stats.generated_ct_nodes}")
    print()
    print("Constraint signatures:")
    print(f"  Unique: {stats.unique_constraint_signatures}")
    print(f"  Duplicate: {stats.duplicate_constraint_signatures}")
    print(
        "  Duplicate rate: "
        f"{_duplicate_rate(stats.duplicate_constraint_signatures, stats.generated_ct_nodes)}"
    )
    print()
    print("Path-set signatures:")
    print(f"  Unique: {stats.unique_path_signatures}")
    print(f"  Duplicate: {stats.duplicate_path_signatures}")
    print(
        "  Duplicate rate: "
        f"{_duplicate_rate(stats.duplicate_path_signatures, stats.generated_ct_nodes)}"
    )


def main() -> None:
    args = parse_args()
    max_expanded_nodes = _resolve_max_expanded_nodes(args)

    grid_map = load_moving_ai_map(args.map)
    scenarios = load_moving_ai_scenarios(args.scen)
    selection = build_non_trivial_demo_scenario(
        scenarios=scenarios,
        grid_map=grid_map,
    )
    scenario = selection.scenario

    _print_shared_setup(
        map_stem=args.map.stem,
        max_timestep=args.max_timestep,
        max_expanded_nodes=max_expanded_nodes,
        compare=args.compare,
    )
    _print_root_summary(grid_map, scenario, args.max_timestep)

    if args.compare:
        run_compare_mode(
            grid_map=grid_map,
            scenario=scenario,
            max_timestep=args.max_timestep,
            max_expanded_nodes=max_expanded_nodes,
        )
    else:
        run_single_cbs_mode(
            grid_map=grid_map,
            scenario=scenario,
            max_timestep=args.max_timestep,
            max_expanded_nodes=max_expanded_nodes,
        )


if __name__ == "__main__":
    main()
