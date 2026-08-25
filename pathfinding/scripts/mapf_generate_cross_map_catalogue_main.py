"""Generate deterministic MAPF-8 cross-map benchmark catalogues.

Open in PyCharm and press Run. No command-line arguments required.
Edit the MAPF-8 CROSS-MAP CATALOGUE CONFIGURATION block below.

MAPF-8.2 manual production workflow:
  1. Set TARGET_MAP to AR0400SR or AR0307SR (NOT AR0404SR).
  2. Set ALLOW_PRODUCTION_MANIFEST_WRITE = True.
  3. Run this script once per map (sequentially).
  4. Run mapf_validate_cross_map_catalogue_main.py for the same TARGET_MAP.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathfinding.src.experiments.mapf_benchmark_catalogue_generation import (
    mapf8_cross_map_catalogue_config,
    run_benchmark_catalogue_generation_main,
)

# ============================================================
# MAPF-8 CROSS-MAP CATALOGUE CONFIGURATION
# ============================================================

# Choose exactly one frozen MAPF-8 target:
#   "AR0400SR"  (seed 2028)  — NOT AR0404SR
#   "AR0307SR"  (seed 2029)
TARGET_MAP = "AR0307SR"

# Required True for MAPF-8.2 production generation. Default False prevents
# accidental manifest creation during infrastructure development.
ALLOW_PRODUCTION_MANIFEST_WRITE = True

# ============================================================
# END CONFIGURATION
# ============================================================


def main() -> None:
    config = mapf8_cross_map_catalogue_config(
        TARGET_MAP,
        _REPO_ROOT,
        allow_production_manifest_write=ALLOW_PRODUCTION_MANIFEST_WRITE,
    )
    raise SystemExit(run_benchmark_catalogue_generation_main(config))


if __name__ == "__main__":
    main()
