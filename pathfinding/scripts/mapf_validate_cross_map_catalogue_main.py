"""Validate and freeze a MAPF-8 cross-map production catalogue manifest.

Open in PyCharm and press Run after manual catalogue generation.
Edit the MAPF-8 VALIDATION CONFIGURATION block below.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathfinding.src.experiments.mapf_cross_map_catalogue_validation import (
    mapf8_cross_map_validation_config,
    run_cross_map_catalogue_validation_main,
)

# ============================================================
# MAPF-8 VALIDATION CONFIGURATION
# ============================================================

# Choose exactly one frozen MAPF-8 target:
#   "AR0400SR"  (seed 2028)
#   "AR0307SR"  (seed 2029)
TARGET_MAP = "AR0307SR"

# Full validation recomputes independent-path metadata via source-pool precompute.
# This may take significant time on real bg512 maps; run manually after generation.
RECOMPUTE_SOURCE_POOL = True

# ============================================================
# END CONFIGURATION
# ============================================================


def main() -> None:
    config = replace(
        mapf8_cross_map_validation_config(TARGET_MAP, _REPO_ROOT),
        recompute_source_pool=RECOMPUTE_SOURCE_POOL,
    )
    raise SystemExit(
        run_cross_map_catalogue_validation_main(
            config,
            repo_root=_REPO_ROOT,
            write_report=True,
        )
    )


if __name__ == "__main__":
    main()
