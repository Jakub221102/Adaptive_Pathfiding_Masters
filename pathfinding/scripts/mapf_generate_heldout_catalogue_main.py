"""Generate and freeze the MAPF-7.6 held-out validation catalogue.

Open in PyCharm and press Run. No command-line arguments required.
Edit the HELD-OUT CATALOGUE CONFIGURATION block below to change settings.

This script performs catalogue construction only. It does NOT run CDF-H, CDF-L,
SPF+CD, PP, CBS, or any MAPF priority-strategy benchmark.
"""

from __future__ import annotations

import hashlib
import logging
import sys
import time
import traceback
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathfinding.src.experiments import mapf_benchmark_instances as benchmark_module
from pathfinding.src.experiments.mapf_benchmark_instances import (
    DEFAULT_HELD_OUT_SEED,
    HELD_OUT_CATALOGUE_GENERATION_VERSION,
    generate_held_out_benchmark_manifest,
    load_benchmark_manifest,
    prepare_benchmark_source_pool,
    save_benchmark_manifest,
    validate_held_out_catalogue,
)
from pathfinding.src.loaders.map_loader import load_moving_ai_map
from pathfinding.src.loaders.scen_loader import load_moving_ai_scenarios

# ============================================================
# HELD-OUT CATALOGUE CONFIGURATION (MAPF-7.6 — frozen)
# ============================================================

MAP = _REPO_ROOT / "Data" / "bg512-map" / "AR0204SR.map"
SCEN = _REPO_ROOT / "Data" / "bg512-scen" / "AR0204SR.map.scen"

PRIMARY_MANIFEST = (
    _REPO_ROOT / "pathfinding" / "results" / "mapf_benchmarks" / "AR0204SR_manifest.json"
)
OUTPUT_MANIFEST = (
    _REPO_ROOT
    / "pathfinding"
    / "results"
    / "mapf_benchmarks"
    / "AR0204SR_heldout_manifest.json"
)
DESIGN_DOC = (
    _REPO_ROOT
    / "pathfinding"
    / "results"
    / "mapf_benchmarks"
    / "AR0204SR_heldout_design.md"
)

# Deterministic identifier fixed before any held-out MAPF-7 evaluation.
# Successor to primary catalogue seed 2026; not chosen from candidate outcomes.
HELD_OUT_SEED = DEFAULT_HELD_OUT_SEED

MAX_TIMESTEP = 512
MIN_REFERENCE_LENGTH = 20.0
FINAL_AGENT_COUNTS = (5, 10, 20)
FINAL_INSTANCES_PER_LEVEL = 3
FINAL_MAX_ATTEMPTS = 5000
PRECOMPUTE_PROGRESS_EVERY = 100

LOG_FILE = (
    _REPO_ROOT / "pathfinding" / "results" / "mapf_heldout_catalogue_generation.log"
)

# ============================================================
# END CONFIGURATION
# ============================================================


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
    logger = logging.getLogger("mapf_heldout_catalogue_generation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _primary_manifest_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_held_out_catalogue_generation() -> None:
    log_file = LOG_FILE.resolve()
    logger = _setup_logging(log_file)
    timer = ElapsedTimer()

    logger.info(f"{timer.stamp()} Starting MAPF-7.6 held-out catalogue generation")
    logger.info(f"Generation version: {HELD_OUT_CATALOGUE_GENERATION_VERSION}")
    logger.info(f"Held-out seed: {HELD_OUT_SEED}")
    logger.info(f"Primary manifest: {PRIMARY_MANIFEST.resolve()}")
    logger.info(f"Output manifest: {OUTPUT_MANIFEST.resolve()}")
    logger.info("")

    primary_digest_before = _primary_manifest_digest(PRIMARY_MANIFEST)
    primary_manifest = load_benchmark_manifest(PRIMARY_MANIFEST)

    if primary_manifest.seed == HELD_OUT_SEED:
        raise RuntimeError("Held-out seed must differ from primary catalogue seed")

    logger.info(f"{timer.stamp()} STEP 1 — Load map/scenarios")
    grid_map = load_moving_ai_map(MAP)
    scenarios = load_moving_ai_scenarios(SCEN)
    logger.info(f"Map loaded: {grid_map.name}")
    logger.info(f"Scenarios loaded: {len(scenarios)}")
    logger.info("")

    logger.info(f"{timer.stamp()} STEP 2 — Build source pool (same rules as primary)")
    precompute_start = time.perf_counter()

    def precompute_progress(progress) -> None:
        elapsed = time.perf_counter() - precompute_start
        logger.info(
            f"BOUNDED STATIC PRECOMPUTE {progress.completed} / {progress.total} "
            f"(elapsed {elapsed:.1f} s, feasible {progress.feasible})"
        )

    original_find_path = benchmark_module.find_path

    def guard_sta_find_path(*_args, **_kwargs):
        raise AssertionError(
            "Space-Time A* must not be used for held-out catalogue source precompute"
        )

    benchmark_module.find_path = guard_sta_find_path
    try:
        source_pool = prepare_benchmark_source_pool(
            grid_map=grid_map,
            scenarios=scenarios,
            min_reference_length=MIN_REFERENCE_LENGTH,
            max_timestep=MAX_TIMESTEP,
            precompute_progress_every=PRECOMPUTE_PROGRESS_EVERY,
            precompute_progress_callback=precompute_progress,
        )
    finally:
        benchmark_module.find_path = original_find_path

    logger.info(
        f"Feasible source pool: {len(source_pool.feasible_indices)} scenarios"
    )
    logger.info("")

    logger.info(f"{timer.stamp()} STEP 3 — Generate held-out manifest")
    generation_result = generate_held_out_benchmark_manifest(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_name=SCEN.name,
        primary_manifest=primary_manifest,
        agent_counts=FINAL_AGENT_COUNTS,
        instances_per_level=FINAL_INSTANCES_PER_LEVEL,
        max_timestep=MAX_TIMESTEP,
        seed=HELD_OUT_SEED,
        min_reference_length=MIN_REFERENCE_LENGTH,
        max_attempts=FINAL_MAX_ATTEMPTS,
        source_pool=source_pool,
        primary_manifest_reference=str(PRIMARY_MANIFEST.relative_to(_REPO_ROOT)),
    )
    manifest = generation_result.manifest
    logger.info(f"Generated {len(manifest.instances)} held-out instances")
    for agent_count, attempts in generation_result.attempts_by_agent_count.items():
        logger.info(f"  agent_count={agent_count}: attempts={attempts}")
    logger.info("")

    logger.info(f"{timer.stamp()} STEP 4 — Validate held-out catalogue structure")
    validate_held_out_catalogue(
        manifest,
        primary_manifest,
        scenarios=scenarios,
        grid_map=grid_map,
        agent_counts=FINAL_AGENT_COUNTS,
        instances_per_level=FINAL_INSTANCES_PER_LEVEL,
    )
    logger.info("Held-out catalogue validation passed.")
    logger.info("")

    logger.info(f"{timer.stamp()} STEP 5 — Determinism check")
    repeat_result = generate_held_out_benchmark_manifest(
        grid_map=grid_map,
        scenarios=scenarios,
        scenario_name=SCEN.name,
        primary_manifest=primary_manifest,
        agent_counts=FINAL_AGENT_COUNTS,
        instances_per_level=FINAL_INSTANCES_PER_LEVEL,
        max_timestep=MAX_TIMESTEP,
        seed=HELD_OUT_SEED,
        min_reference_length=MIN_REFERENCE_LENGTH,
        max_attempts=FINAL_MAX_ATTEMPTS,
        source_pool=source_pool,
        primary_manifest_reference=str(PRIMARY_MANIFEST.relative_to(_REPO_ROOT)),
    )
    if repeat_result.manifest.instances != manifest.instances:
        raise RuntimeError("Deterministic held-out regeneration mismatch")
    logger.info("Determinism check passed.")
    logger.info("")

    logger.info(f"{timer.stamp()} STEP 6 — Save manifest")
    output_path = OUTPUT_MANIFEST.resolve()
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    save_benchmark_manifest(manifest, tmp_path)
    loaded = load_benchmark_manifest(tmp_path)
    if loaded != manifest:
        raise RuntimeError("JSON roundtrip mismatch for held-out manifest")
    tmp_path.replace(output_path)
    logger.info(f"Held-out manifest saved to: {output_path}")
    logger.info("")

    primary_digest_after = _primary_manifest_digest(PRIMARY_MANIFEST)
    if primary_digest_after != primary_digest_before:
        raise RuntimeError("Primary manifest was modified during held-out generation")

    logger.info("Primary manifest unchanged.")
    logger.info("")
    logger.info("=" * 72)
    logger.info("MAPF-7.6 HELD-OUT CATALOGUE FROZEN")
    logger.info("=" * 72)
    logger.info(f"Design doc: {DESIGN_DOC.resolve()}")
    logger.info(f"Instances: {len(manifest.instances)}")
    logger.info(f"Seed: {manifest.seed}")
    logger.info(f"Role: {manifest.catalogue_role}")
    for instance in manifest.instances:
        logger.info(f"- {instance.instance_id}")


def main() -> None:
    log_file = LOG_FILE.resolve()
    logger = _setup_logging(log_file)
    timer = ElapsedTimer()
    try:
        run_held_out_catalogue_generation()
    except SystemExit:
        raise
    except Exception:
        logger.exception(f"{timer.stamp()} Unexpected error during held-out generation:")
        logger.error(traceback.format_exc())
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
