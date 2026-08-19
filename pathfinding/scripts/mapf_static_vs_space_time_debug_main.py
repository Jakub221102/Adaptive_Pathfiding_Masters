"""Local diagnostic: static 4-connected planner vs unconstrained Space-Time A*.

Open in PyCharm and press Run. No command-line arguments required.
Edit the DEBUG CONFIGURATION block below to change experiment settings.

Cursor must NOT run this script on AR0204SR — execute locally only.
"""

from __future__ import annotations

import logging
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathfinding.src.algorithms.mapf.models import MAPFAgent
from pathfinding.src.algorithms.mapf.space_time_astar import find_path
from pathfinding.src.experiments.mapf_benchmark_instances import (
    eligible_scenario_indices,
    evaluate_benchmark_candidate,
)
from pathfinding.src.experiments.mapf_independent_static_path import (
    evaluate_candidate_with_independent_paths,
    find_independent_static_path,
    independent_path_cost,
    select_diagnostic_scenario_indices,
    spatial_trajectory,
    trajectories_equal,
)
from pathfinding.src.loaders.map_loader import load_moving_ai_map
from pathfinding.src.loaders.scen_loader import load_moving_ai_scenarios

# ============================================================
# DEBUG CONFIGURATION
# ============================================================

MAP = _REPO_ROOT / "Data" / "bg512-map" / "AR0204SR.map"
SCEN = _REPO_ROOT / "Data" / "bg512-scen" / "AR0204SR.map.scen"

MAX_TIMESTEP = 512
MIN_REFERENCE_LENGTH = 20.0

EARLY_COUNT = 10
MIDDLE_COUNT = 10
LATE_COUNT = 10

STA_TIMEOUT_S = 10.0
CANDIDATE_GROUP_COUNT = 5
CANDIDATE_AGENT_COUNT = 5

LOG_FILE = _REPO_ROOT / "pathfinding" / "results" / "mapf_static_vs_space_time_debug.log"

# ============================================================
# END DEBUG CONFIGURATION
# ============================================================


@dataclass
class SolverRun:
    success: bool
    cost: int | None
    runtime_s: float
    trajectory: tuple[tuple[int, int], ...] | None = None


@dataclass
class ScenarioComparison:
    scenario_index: int
    reference_length: float | None
    region: str
    sta: SolverRun
    static: SolverRun

    @property
    def cost_equal(self) -> bool:
        return self.sta.success == self.static.success and (
            self.sta.cost == self.static.cost
        )

    @property
    def trajectory_equal(self) -> bool:
        if not self.sta.success or not self.static.success:
            return False
        return self.sta.trajectory == self.static.trajectory


@dataclass
class RuntimeStats:
    label: str
    values: list[float] = field(default_factory=list)

    def report(self) -> list[str]:
        if not self.values:
            return [f"{self.label}: (no data)"]
        return [
            f"{self.label}:",
            f"  min: {min(self.values):.4f} s",
            f"  median: {statistics.median(self.values):.4f} s",
            f"  mean: {statistics.mean(self.values):.4f} s",
            f"  max: {max(self.values):.4f} s",
            f"  total: {sum(self.values):.4f} s",
        ]


class ElapsedTimer:
    def __init__(self) -> None:
        self._start = time.perf_counter()

    def stamp(self) -> str:
        elapsed = time.perf_counter() - self._start
        minutes = int(elapsed // 60)
        seconds = elapsed - minutes * 60
        return f"[{minutes:02d}:{seconds:06.3f}]"


def _setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("mapf_static_vs_space_time_debug")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    original_emit = file_handler.emit

    def flushing_emit(record: logging.LogRecord) -> None:
        original_emit(record)
        file_handler.flush()

    file_handler.emit = flushing_emit  # type: ignore[method-assign]
    logger.addHandler(file_handler)

    return logger


def _log(logger: logging.Logger, timer: ElapsedTimer, message: str) -> None:
    logger.info(f"{timer.stamp()} {message}")


def _region_for_index(
    scenario_index: int,
    early: tuple[int, ...],
    middle: tuple[int, ...],
    late: tuple[int, ...],
) -> str:
    if scenario_index in early:
        return "EARLY"
    if scenario_index in middle:
        return "MIDDLE"
    if scenario_index in late:
        return "LATE"
    return "UNKNOWN"


def _run_sta(
    grid_map,
    agent: MAPFAgent,
    max_timestep: int,
) -> SolverRun:
    start = time.perf_counter()
    path = find_path(
        grid_map=grid_map,
        agent=agent,
        max_timestep=max_timestep,
        constraints=(),
    )
    runtime_s = time.perf_counter() - start
    if path is None:
        return SolverRun(success=False, cost=None, runtime_s=runtime_s)
    return SolverRun(
        success=True,
        cost=independent_path_cost(path),
        runtime_s=runtime_s,
        trajectory=spatial_trajectory(path),
    )


def _run_static(grid_map, agent: MAPFAgent) -> SolverRun:
    start = time.perf_counter()
    path = find_independent_static_path(grid_map, agent)
    runtime_s = time.perf_counter() - start
    if path is None:
        return SolverRun(success=False, cost=None, runtime_s=runtime_s)
    return SolverRun(
        success=True,
        cost=independent_path_cost(path),
        runtime_s=runtime_s,
        trajectory=spatial_trajectory(path),
    )


def _deterministic_candidate_groups(
    selected_indices: tuple[int, ...],
    group_count: int,
    agent_count: int,
) -> list[tuple[int, ...]]:
    groups: list[tuple[int, ...]] = []
    pool = list(selected_indices)
    if len(pool) < agent_count:
        return groups

    for group_id in range(group_count):
        start = (group_id * agent_count) % len(pool)
        chosen: list[int] = []
        offset = 0
        while len(chosen) < agent_count:
            index = pool[(start + offset) % len(pool)]
            if index not in chosen:
                chosen.append(index)
            offset += 1
            if offset > len(pool) * agent_count:
                break
        if len(chosen) == agent_count:
            groups.append(tuple(chosen))
    return groups


def run_diagnostic() -> None:
    log_file = LOG_FILE.resolve()
    logger = _setup_logging(log_file)
    timer = ElapsedTimer()

    _log(logger, timer, "Starting static vs Space-Time A* diagnostic")
    logger.info(f"Log file: {log_file}")
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  MAP: {MAP}")
    logger.info(f"  SCEN: {SCEN}")
    logger.info(f"  MAX_TIMESTEP: {MAX_TIMESTEP}")
    logger.info(f"  MIN_REFERENCE_LENGTH: {MIN_REFERENCE_LENGTH:g}")
    logger.info(f"  EARLY_COUNT: {EARLY_COUNT}")
    logger.info(f"  MIDDLE_COUNT: {MIDDLE_COUNT}")
    logger.info(f"  LATE_COUNT: {LATE_COUNT}")
    logger.info(f"  STA_TIMEOUT_S: {STA_TIMEOUT_S}")
    logger.info("")

    grid_map = load_moving_ai_map(MAP)
    scenarios = load_moving_ai_scenarios(SCEN)
    eligible = eligible_scenario_indices(
        scenarios=scenarios,
        grid_map=grid_map,
        min_reference_length=MIN_REFERENCE_LENGTH,
    )

    early_indices = eligible[:EARLY_COUNT]
    late_indices = eligible[-LATE_COUNT:]
    middle_start = max(0, (len(eligible) - MIDDLE_COUNT) // 2)
    middle_indices = eligible[middle_start : middle_start + MIDDLE_COUNT]
    selected = select_diagnostic_scenario_indices(
        eligible,
        early_count=EARLY_COUNT,
        middle_count=MIDDLE_COUNT,
        late_count=LATE_COUNT,
    )

    logger.info(f"Total MovingAI scenarios: {len(scenarios)}")
    logger.info(f"Eligible source scenarios: {len(eligible)}")
    logger.info(f"Diagnostic selection: {len(selected)} scenarios")
    logger.info(f"  EARLY indices: {early_indices}")
    logger.info(f"  MIDDLE indices: {middle_indices}")
    logger.info(f"  LATE indices: {late_indices}")
    logger.info("")

    comparisons: list[ScenarioComparison] = []
    sta_stopped = False
    sta_stop_index: int | None = None
    sta_stop_runtime: float | None = None

    for scenario_index in selected:
        if sta_stopped:
            break

        scenario = scenarios[scenario_index]
        reference_length = scenario.optimal_length
        region = _region_for_index(
            scenario_index,
            early_indices,
            middle_indices,
            late_indices,
        )

        logger.info("=" * 60)
        logger.info(f"Testing scenario index {scenario_index}")
        logger.info(f"  region: {region}")
        logger.info(f"  reference length: {reference_length}")
        logger.info("")

        agent = MAPFAgent(
            agent_id=0,
            start=scenario.start,
            goal=scenario.goal,
        )

        logger.info("Space-Time A*:")
        sta_run = _run_sta(grid_map, agent, MAX_TIMESTEP)
        logger.info(f"  success: {sta_run.success}")
        logger.info(f"  cost: {sta_run.cost}")
        logger.info(f"  runtime: {sta_run.runtime_s:.4f} s")

        if sta_run.runtime_s > STA_TIMEOUT_S:
            sta_stopped = True
            sta_stop_index = scenario_index
            sta_stop_runtime = sta_run.runtime_s
            logger.info("")
            logger.info("STOPPING: Space-Time A* exceeded timeout threshold")
            logger.info(f"  scenario index: {scenario_index}")
            logger.info(f"  runtime: {sta_run.runtime_s:.4f} s")
            logger.info(f"  threshold: {STA_TIMEOUT_S:.1f} s")
            break

        logger.info("Static 4-connected A*:")
        static_run = _run_static(grid_map, agent)
        logger.info(f"  success: {static_run.success}")
        logger.info(f"  cost: {static_run.cost}")
        logger.info(f"  runtime: {static_run.runtime_s:.4f} s")
        logger.info("")
        logger.info(f"  cost equal: {sta_run.cost == static_run.cost}")
        logger.info(
            f"  exact trajectory equal: "
            f"{sta_run.trajectory == static_run.trajectory if sta_run.success and static_run.success else False}"
        )
        logger.info("")

        comparisons.append(
            ScenarioComparison(
                scenario_index=scenario_index,
                reference_length=reference_length,
                region=region,
                sta=sta_run,
                static=static_run,
            )
        )

    # Runtime summaries
    sta_stats = {
        "EARLY": RuntimeStats("Space-Time A* EARLY"),
        "MIDDLE": RuntimeStats("Space-Time A* MIDDLE"),
        "LATE": RuntimeStats("Space-Time A* LATE"),
        "OVERALL": RuntimeStats("Space-Time A* OVERALL"),
    }
    static_stats = {
        "EARLY": RuntimeStats("Static A* EARLY"),
        "MIDDLE": RuntimeStats("Static A* MIDDLE"),
        "LATE": RuntimeStats("Static A* LATE"),
        "OVERALL": RuntimeStats("Static A* OVERALL"),
    }

    for comparison in comparisons:
        sta_stats[comparison.region].values.append(comparison.sta.runtime_s)
        sta_stats["OVERALL"].values.append(comparison.sta.runtime_s)
        static_stats[comparison.region].values.append(comparison.static.runtime_s)
        static_stats["OVERALL"].values.append(comparison.static.runtime_s)

    logger.info("=" * 60)
    logger.info("RUNTIME SUMMARY")
    logger.info("=" * 60)
    for region in ("EARLY", "MIDDLE", "LATE", "OVERALL"):
        for line in sta_stats[region].report():
            logger.info(line)
        logger.info("")
        for line in static_stats[region].report():
            logger.info(line)
        logger.info("")

    sta_total = sum(sta_stats["OVERALL"].values)
    static_total = sum(static_stats["OVERALL"].values)
    if static_total > 0:
        logger.info(f"Approximate speedup (STA total / static total): {sta_total / static_total:.1f}x")
    logger.info("")

    successful = [c for c in comparisons if c.sta.success and c.static.success]
    cost_equal = sum(1 for c in successful if c.sta.cost == c.static.cost)
    cost_diff = len(successful) - cost_equal
    traj_equal = sum(1 for c in successful if c.trajectory_equal)
    traj_diff_equal_cost = len(successful) - traj_equal

    logger.info("=" * 60)
    logger.info("COST-EQUIVALENCE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Successful comparisons: {len(successful)}")
    logger.info(f"Same optimal cost: {cost_equal} / {len(successful) if successful else 0}")
    logger.info(f"Different cost: {cost_diff} / {len(successful) if successful else 0}")
    logger.info("")

    logger.info("=" * 60)
    logger.info("TRAJECTORY-EQUIVALENCE SUMMARY")
    logger.info("=" * 60)
    logger.info(
        f"Exact same trajectory: {traj_equal} / {len(successful) if successful else 0}"
    )
    logger.info(
        f"Different but equal-cost trajectory: "
        f"{traj_diff_equal_cost} / {len(successful) if successful else 0}"
    )
    logger.info("")

    # Candidate conflict-level comparison
    logger.info("=" * 60)
    logger.info("CANDIDATE INTERACTION COMPARISON")
    logger.info("=" * 60)

    candidate_groups = _deterministic_candidate_groups(
        selected,
        group_count=CANDIDATE_GROUP_COUNT,
        agent_count=CANDIDATE_AGENT_COUNT,
    )

    interaction_matches = 0
    interaction_diffs = 0

    for group_id, scenario_indices in enumerate(candidate_groups, start=1):
        sta_eval = evaluate_benchmark_candidate(
            grid_map=grid_map,
            scenarios=scenarios,
            scenario_indices=scenario_indices,
            max_timestep=MAX_TIMESTEP,
            precomputed_lookup=None,
        )

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
            static_paths.append(path)

        if any(path is None for path in static_paths):
            logger.info(f"Candidate group {group_id}: skipped (missing static path)")
            continue

        static_eval = evaluate_candidate_with_independent_paths(
            grid_map=grid_map,
            scenarios=scenarios,
            scenario_indices=scenario_indices,
            independent_paths=[path for path in static_paths if path is not None],
        )

        logger.info(f"Candidate group {group_id}: indices={scenario_indices}")
        logger.info(f"  STA conflicts: {sta_eval.independent_conflict_count if sta_eval else None}")
        logger.info(
            f"  Static conflicts: "
            f"{static_eval.independent_conflict_count if static_eval else None}"
        )

        if sta_eval is None or static_eval is None:
            logger.info("  metadata match: N/A (evaluation failed)")
            continue

        metadata_match = (
            sta_eval.independent_conflict_count
            == static_eval.independent_conflict_count
            and sta_eval.vertex_conflict_count == static_eval.vertex_conflict_count
            and sta_eval.edge_conflict_count == static_eval.edge_conflict_count
            and sta_eval.conflicting_agent_pair_count
            == static_eval.conflicting_agent_pair_count
            and sta_eval.independent_soc == static_eval.independent_soc
            and sta_eval.independent_makespan == static_eval.independent_makespan
            and sta_eval.interaction_level == static_eval.interaction_level
        )
        logger.info(f"  interaction level STA: {sta_eval.interaction_level.name}")
        logger.info(f"  interaction level static: {static_eval.interaction_level.name}")
        logger.info(f"  metadata match: {metadata_match}")
        logger.info("")

        if metadata_match:
            interaction_matches += 1
        else:
            interaction_diffs += 1

    logger.info(
        f"Candidate metadata matches: {interaction_matches} / "
        f"{interaction_matches + interaction_diffs}"
    )
    logger.info("")

    # Recommendation
    logger.info("=" * 60)
    logger.info("RECOMMENDATION")
    logger.info("=" * 60)

    if sta_stopped:
        logger.info(
            "CASE: Space-Time A* exceeded timeout on at least one diagnostic scenario."
        )
        logger.info(
            f"  Slow scenario index: {sta_stop_index}, runtime: {sta_stop_runtime:.4f} s"
        )
        if cost_diff == 0 and len(successful) > 0:
            logger.info(
                "Static planner shows matching costs on completed scenarios and is "
                "much faster — investigate MAPF-5A.5 integration after full "
                "trajectory/interaction evidence from local run."
            )
        else:
            logger.info(
                "Complete remaining scenarios locally before deciding on MAPF-5A.5."
            )
    elif cost_diff > 0:
        logger.info("CASE C: Cost mismatch detected — do NOT switch to static planner.")
        logger.info("Investigate movement / heuristic / goal semantics first.")
    elif traj_diff_equal_cost > 0 and interaction_diffs > 0:
        logger.info(
            "CASE B: Costs match but trajectories/interaction metadata differ."
        )
        logger.info(
            "Do NOT switch benchmark generation yet; investigate deterministic "
            "tie-breaking so static and STA* choose the same shortest path."
        )
    elif traj_equal == len(successful) and interaction_diffs == 0:
        logger.info(
            "CASE A: Costs, trajectories, and candidate interaction metadata match."
        )
        logger.info(
            "Static planner may replace Space-Time A* for benchmark precompute "
            "(MAPF-5A.5)."
        )
    elif static_total >= sta_total:
        logger.info(
            "CASE D: Static planner is not significantly faster — limited benefit."
        )
    else:
        logger.info(
            "Mixed evidence — review cost/trajectory/interaction summaries above."
        )

    logger.info("")
    _log(logger, timer, "Diagnostic complete")


def main() -> None:
    log_file = LOG_FILE.resolve()
    logger = _setup_logging(log_file)
    timer = ElapsedTimer()

    try:
        run_diagnostic()
    except Exception:
        logger.exception(f"{timer.stamp()} Unexpected error during diagnostic:")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
