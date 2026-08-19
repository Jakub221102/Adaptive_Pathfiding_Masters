"""Static-only horizon verification for MAPF benchmark source scenarios.

Open in PyCharm and press Run. No command-line arguments required.
Edit the DEBUG CONFIGURATION block below to change experiment settings.

This script NEVER calls Space-Time A*. Execute locally only — not via Cursor.
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
from pathfinding.src.experiments.mapf_benchmark_instances import eligible_scenario_indices
from pathfinding.src.experiments.mapf_independent_static_path import (
    find_independent_static_path,
    independent_path_cost,
    independent_path_excess_moves,
    independent_path_fits_horizon,
    merge_diagnostic_scenario_indices,
    select_diagnostic_scenario_indices,
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

SPECIAL_SCENARIO_INDICES = (1125,)

LOG_FILE = _REPO_ROOT / "pathfinding" / "results" / "mapf_static_horizon_debug.log"

# ============================================================
# END DEBUG CONFIGURATION
# ============================================================


@dataclass
class StaticHorizonResult:
    scenario_index: int
    region: str
    reference_length: float | None
    success: bool
    cost: int | None
    runtime_s: float
    fits_horizon: bool
    excess_moves: int | None
    spatial_status: str


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

    logger = logging.getLogger("mapf_static_horizon_debug")
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
    *,
    special_indices: tuple[int, ...],
    early_indices: tuple[int, ...],
    middle_indices: tuple[int, ...],
    late_indices: tuple[int, ...],
) -> str:
    if scenario_index in special_indices:
        return "SPECIAL"
    if scenario_index in early_indices:
        return "EARLY"
    if scenario_index in middle_indices:
        return "MIDDLE"
    if scenario_index in late_indices:
        return "LATE"
    return "UNKNOWN"


def _cost_bucket(cost: int) -> str:
    if cost <= 100:
        return "0-100"
    if cost <= 200:
        return "101-200"
    if cost <= 300:
        return "201-300"
    if cost <= 400:
        return "301-400"
    if cost <= 512:
        return "401-512"
    if cost <= 600:
        return "513-600"
    return "601+"


def _evaluate_static_horizon(
    grid_map,
    scenario,
    scenario_index: int,
    region: str,
    max_timestep: int,
) -> StaticHorizonResult:
    agent = MAPFAgent(
        agent_id=0,
        start=scenario.start,
        goal=scenario.goal,
    )

    start = time.perf_counter()
    path = find_independent_static_path(grid_map, agent)
    runtime_s = time.perf_counter() - start

    if path is None:
        return StaticHorizonResult(
            scenario_index=scenario_index,
            region=region,
            reference_length=scenario.optimal_length,
            success=False,
            cost=None,
            runtime_s=runtime_s,
            fits_horizon=False,
            excess_moves=None,
            spatial_status="NO SPATIAL PATH",
        )

    cost = independent_path_cost(path)
    assert cost is not None
    fits = independent_path_fits_horizon(path, max_timestep)
    excess = independent_path_excess_moves(path, max_timestep)
    spatial_status = (
        "SPATIAL PATH FITS HORIZON"
        if fits
        else "SPATIAL PATH EXISTS BUT EXCEEDS HORIZON"
    )

    return StaticHorizonResult(
        scenario_index=scenario_index,
        region=region,
        reference_length=scenario.optimal_length,
        success=True,
        cost=cost,
        runtime_s=runtime_s,
        fits_horizon=fits,
        excess_moves=excess,
        spatial_status=spatial_status,
    )


def _log_scenario_result(logger: logging.Logger, result: StaticHorizonResult) -> None:
    logger.info("=" * 60)
    logger.info(f"Scenario index: {result.scenario_index}")
    logger.info(f"Region: {result.region}")
    logger.info(f"MovingAI reference length: {result.reference_length}")
    logger.info("")
    logger.info("Static 4-connected A*:")
    logger.info(f"  success: {result.success}")
    logger.info(f"  cost: {result.cost}")
    logger.info(f"  runtime: {result.runtime_s:.4f} s")
    logger.info(f"  spatial status: {result.spatial_status}")
    logger.info("")
    logger.info("Horizon:")
    logger.info(f"  max_timestep: {MAX_TIMESTEP}")
    logger.info(f"  fits_horizon: {result.fits_horizon}")
    if result.excess_moves is None:
        logger.info("  excess_moves: N/A")
    else:
        logger.info(f"  excess_moves: {result.excess_moves}")
    logger.info("=" * 60)
    logger.info("")


def run_diagnostic() -> None:
    log_file = LOG_FILE.resolve()
    logger = _setup_logging(log_file)
    timer = ElapsedTimer()

    _log(logger, timer, "Starting static-only horizon diagnostic")
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
    logger.info(f"  SPECIAL_SCENARIO_INDICES: {SPECIAL_SCENARIO_INDICES}")
    logger.info("")
    logger.info("NOTE: This diagnostic uses ONLY static 4-connected A*.")
    logger.info("Space-Time A* is NOT called.")
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
    base_selected = select_diagnostic_scenario_indices(
        eligible,
        early_count=EARLY_COUNT,
        middle_count=MIDDLE_COUNT,
        late_count=LATE_COUNT,
    )
    selected = merge_diagnostic_scenario_indices(
        base_selected,
        SPECIAL_SCENARIO_INDICES,
    )

    for special_index in SPECIAL_SCENARIO_INDICES:
        if special_index not in eligible:
            raise ValueError(
                f"Special scenario index {special_index} is not eligible "
                f"(min_reference_length={MIN_REFERENCE_LENGTH:g})."
            )

    logger.info(f"Total MovingAI scenarios: {len(scenarios)}")
    logger.info(f"Eligible source scenarios: {len(eligible)}")
    logger.info(f"Selected scenario count: {len(selected)}")
    logger.info(f"  EARLY indices: {early_indices}")
    logger.info(f"  MIDDLE indices: {middle_indices}")
    logger.info(f"  LATE indices: {late_indices}")
    logger.info(f"  SPECIAL indices: {SPECIAL_SCENARIO_INDICES}")
    logger.info(f"  Final selected (deduplicated): {selected}")
    logger.info("")

    results: list[StaticHorizonResult] = []

    for scenario_index in selected:
        scenario = scenarios[scenario_index]
        region = _region_for_index(
            scenario_index,
            special_indices=SPECIAL_SCENARIO_INDICES,
            early_indices=early_indices,
            middle_indices=middle_indices,
            late_indices=late_indices,
        )
        result = _evaluate_static_horizon(
            grid_map=grid_map,
            scenario=scenario,
            scenario_index=scenario_index,
            region=region,
            max_timestep=MAX_TIMESTEP,
        )
        results.append(result)
        _log_scenario_result(logger, result)

        if scenario_index in SPECIAL_SCENARIO_INDICES:
            logger.info("*" * 60)
            logger.info(f"SCENARIO {scenario_index} HORIZON CHECK")
            logger.info("*" * 60)
            logger.info(f"  reference length: {result.reference_length}")
            logger.info(f"  static path cost: {result.cost}")
            logger.info(f"  static runtime: {result.runtime_s:.4f} s")
            logger.info(f"  max timestep: {MAX_TIMESTEP}")
            logger.info(f"  fits horizon: {result.fits_horizon}")
            if result.cost is not None and result.cost > MAX_TIMESTEP:
                logger.info(f"  cost - max_timestep: {result.cost - MAX_TIMESTEP}")
            else:
                logger.info("  cost - max_timestep: 0 or N/A")
            logger.info(f"  spatial status: {result.spatial_status}")
            logger.info("*" * 60)
            logger.info("")

    # Runtime summary
    runtime_by_region = {
        "EARLY": RuntimeStats("Static A* EARLY"),
        "MIDDLE": RuntimeStats("Static A* MIDDLE"),
        "LATE": RuntimeStats("Static A* LATE"),
        "SPECIAL": RuntimeStats("Static A* SPECIAL"),
        "OVERALL": RuntimeStats("Static A* OVERALL"),
    }

    for result in results:
        runtime_by_region[result.region].values.append(result.runtime_s)
        runtime_by_region["OVERALL"].values.append(result.runtime_s)

    slowest = max(results, key=lambda item: item.runtime_s)

    logger.info("=" * 60)
    logger.info("STATIC RUNTIME SUMMARY")
    logger.info("=" * 60)
    for region in ("EARLY", "MIDDLE", "LATE", "SPECIAL", "OVERALL"):
        for line in runtime_by_region[region].report():
            logger.info(line)
        logger.info("")
    logger.info(
        f"Slowest scenario: index={slowest.scenario_index}, "
        f"runtime={slowest.runtime_s:.4f} s"
    )
    logger.info("")

    # Horizon distribution
    spatial_exists = [r for r in results if r.success]
    no_spatial = [r for r in results if not r.success]
    fits = [r for r in spatial_exists if r.fits_horizon]
    exceeds = [r for r in spatial_exists if not r.fits_horizon]
    excess_values = [r.excess_moves for r in exceeds if r.excess_moves is not None]

    logger.info("=" * 60)
    logger.info("HORIZON DISTRIBUTION")
    logger.info("=" * 60)
    logger.info(f"Total tested: {len(results)}")
    logger.info(f"Static path exists: {len(spatial_exists)}")
    logger.info(f"No spatial path: {len(no_spatial)}")
    logger.info(f"Fits horizon <= {MAX_TIMESTEP}: {len(fits)}")
    logger.info(f"Exceeds horizon > {MAX_TIMESTEP}: {len(exceeds)}")
    logger.info("")
    if excess_values:
        logger.info("Excess moves for over-horizon paths:")
        logger.info(f"  minimum excess: {min(excess_values)}")
        logger.info(f"  median excess: {statistics.median(excess_values):.1f}")
        logger.info(f"  maximum excess: {max(excess_values)}")
        logger.info("")
        logger.info("Over-horizon scenarios:")
        for result in exceeds:
            logger.info(f"  {result.scenario_index} -> static_cost={result.cost}")
    logger.info("")

    # Cost distribution
    costs = [r.cost for r in spatial_exists if r.cost is not None]
    buckets = {
        "0-100": 0,
        "101-200": 0,
        "201-300": 0,
        "301-400": 0,
        "401-512": 0,
        "513-600": 0,
        "601+": 0,
    }
    for cost in costs:
        buckets[_cost_bucket(cost)] += 1

    logger.info("=" * 60)
    logger.info("STATIC COST DISTRIBUTION")
    logger.info("=" * 60)
    if costs:
        logger.info(f"min cost: {min(costs)}")
        logger.info(f"median cost: {statistics.median(costs):.1f}")
        logger.info(f"mean cost: {statistics.mean(costs):.1f}")
        logger.info(f"max cost: {max(costs)}")
        logger.info("")
        logger.info("Cost buckets:")
        for label, count in buckets.items():
            logger.info(f"  {label}: {count}")
    else:
        logger.info("(no successful static paths)")
    logger.info("")

    # Reference length vs static cost
    reference_pairs = [
        (r.reference_length, r.cost)
        for r in spatial_exists
        if r.reference_length is not None and r.cost is not None
    ]

    logger.info("=" * 60)
    logger.info("MOVINGAI REFERENCE vs STATIC 4-CONNECTED COST")
    logger.info("=" * 60)
    if reference_pairs:
        paired = [
            (result.scenario_index, result.reference_length, result.cost, result.cost - result.reference_length)
            for result in spatial_exists
            if result.reference_length is not None and result.cost is not None
        ]
        diffs_only = [item[3] for item in paired]
        logger.info(
            f"mean difference (static_cost - reference_length): "
            f"{statistics.mean(diffs_only):.4f}"
        )
        logger.info(f"max difference: {max(diffs_only):.4f}")
        max_scenario_index, max_ref, max_cost, _ = max(paired, key=lambda item: item[3])
        logger.info(
            f"scenario with max difference: {max_scenario_index} "
            f"(reference={max_ref}, static={max_cost})"
        )
        logger.info("")
        logger.info("Per-scenario comparison:")
        for result in spatial_exists:
            if result.reference_length is None or result.cost is None:
                continue
            diff = result.cost - result.reference_length
            logger.info(
                f"  index={result.scenario_index}: "
                f"reference={result.reference_length:.6g}, "
                f"static={result.cost}, diff={diff:.4f}"
            )
    else:
        logger.info("(no comparable reference/static pairs)")
    logger.info("")

    # Interpretation hint
    logger.info("=" * 60)
    logger.info("INTERPRETATION HINT")
    logger.info("=" * 60)
    special_results = [r for r in results if r.scenario_index in SPECIAL_SCENARIO_INDICES]
    for special in special_results:
        if special.success and special.cost is not None and special.cost > MAX_TIMESTEP:
            logger.info(
                f"Scenario {special.scenario_index}: static cost {special.cost} > "
                f"{MAX_TIMESTEP} — prior STA failure may be explained by horizon "
                "infeasibility (Case A)."
            )
        elif special.success and special.cost is not None and special.cost <= MAX_TIMESTEP:
            logger.info(
                f"Scenario {special.scenario_index}: static cost {special.cost} <= "
                f"{MAX_TIMESTEP} — prior STA failure cannot be explained by horizon "
                "length alone (Case B)."
            )
        elif not special.success:
            logger.info(
                f"Scenario {special.scenario_index}: no spatial path — investigate "
                "map/scenario validity."
            )
    logger.info("")
    _log(logger, timer, "Static-only horizon diagnostic complete")


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
