"""Generate deterministic MAPF-8 cross-map benchmark catalogues.

Open in PyCharm and press Run. No command-line arguments required.
Edit the MAPF-8 CROSS-MAP CATALOGUE CONFIGURATION block below.

MAPF-8.1 default: production manifest write is DISABLED.
Enable ALLOW_PRODUCTION_MANIFEST_WRITE in MAPF-8.2 before freezing manifests.
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
#   "AR0400SR"  (seed 2028)
#   "AR0307SR"  (seed 2029)
TARGET_MAP = "AR0400SR"

# MAPF-8.2: set True before manual production catalogue generation.
ALLOW_PRODUCTION_MANIFEST_WRITE = False

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
