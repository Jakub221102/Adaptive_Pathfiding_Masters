"""Validate and freeze MAPF-9 catalogues after generation."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathfinding.src.experiments.mapf9_catalogue_validation import (
    mapf9_catalogue_validation_config,
    run_mapf9_catalogue_validation_main,
)

# ============================================================
# MAPF-9 VALIDATION CONFIGURATION
# ============================================================

TARGET_MAP = "AR0400SR"

# ============================================================
# END CONFIGURATION
# ============================================================


def main() -> None:
    config = mapf9_catalogue_validation_config(TARGET_MAP, _REPO_ROOT)
    raise SystemExit(
        run_mapf9_catalogue_validation_main(
            config,
            repo_root=_REPO_ROOT,
            write_report=True,
        )
    )


if __name__ == "__main__":
    main()
