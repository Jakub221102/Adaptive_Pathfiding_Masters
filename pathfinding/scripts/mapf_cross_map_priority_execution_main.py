"""Execute MAPF-8 cross-map priority strategy benchmark on a frozen catalogue.

Open in PyCharm and press Run. No command-line arguments required.
Edit the MAPF-8.3 EXECUTION CONFIGURATION block below.

Run AR0400SR first to completion (108/108), review results, then AR0307SR.
Do NOT run both maps concurrently.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathfinding.src.experiments.mapf_cross_map_priority_execution import (
    CrossMapPriorityStrategy,
    mapf8_cross_map_execution_config,
    run_cross_map_priority_benchmark,
)

# ============================================================
# MAPF-8.3 EXECUTION CONFIGURATION
# ============================================================

# Choose exactly one frozen MAPF-8 target per run:
#   "AR0400SR"  (seed 2028, 108 runs)
#   "AR0307SR"  (seed 2029, 108 runs)
# Rejected: AR0204SR, AR0404SR
TARGET_MAP = "AR0307SR"

# Set SMOKE_MODE=True for infrastructure validation only
# (one 5-agent LOW instance × one strategy).
SMOKE_MODE = False

MAX_TIMESTEP: int | None = None

SMOKE_INSTANCE_IDS: tuple[str, ...] = ("AR0400SR_n05_low_000",)
SMOKE_STRATEGIES: tuple[CrossMapPriorityStrategy, ...] = (CrossMapPriorityStrategy.SPF,)

RESUME = False
RESET_RESULTS = True
SAVE_AFTER_EACH_RUN = True
STOP_ON_ERROR = False

# ============================================================
# END CONFIGURATION
# ============================================================


def _effective_instance_ids() -> tuple[str, ...] | None:
    if SMOKE_MODE:
        return SMOKE_INSTANCE_IDS
    return None


def _effective_strategies() -> tuple[CrossMapPriorityStrategy, ...] | None:
    if SMOKE_MODE:
        return SMOKE_STRATEGIES
    return None


def main() -> None:
    config = mapf8_cross_map_execution_config(
        TARGET_MAP,
        _REPO_ROOT,
        resume=RESUME,
        reset_results=RESET_RESULTS,
        save_after_each_run=SAVE_AFTER_EACH_RUN,
        stop_on_error=STOP_ON_ERROR,
        max_timestep=MAX_TIMESTEP,
        instance_ids=_effective_instance_ids(),
        strategies=_effective_strategies(),
    )

    try:
        raise SystemExit(
            run_cross_map_priority_benchmark(config, repo_root=_REPO_ROOT)
        )
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
