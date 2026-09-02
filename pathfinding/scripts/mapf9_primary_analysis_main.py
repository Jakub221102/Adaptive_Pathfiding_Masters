"""MAPF-9.6 formal primary analysis entry point."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathfinding.src.experiments.mapf9_primary_analysis import (
    default_mapf9_primary_analysis_config,
    run_mapf9_primary_analysis,
)


def main() -> None:
    config = default_mapf9_primary_analysis_config(_REPO_ROOT)
    raise SystemExit(run_mapf9_primary_analysis(config))


if __name__ == "__main__":
    main()
