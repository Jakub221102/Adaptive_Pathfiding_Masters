"""MAPF-9.7 final thesis artifact generation entry point."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathfinding.src.experiments.mapf9_final_analysis import (
    default_mapf9_final_config,
    run_mapf9_final_analysis,
)


def main() -> None:
    config = default_mapf9_final_config(_REPO_ROOT)
    result = run_mapf9_final_analysis(config)
    print(f"MAPF-9.7 final artifacts written to {result.output_dir}")
    print(f"Figures: {', '.join(result.figures_written)}")


if __name__ == "__main__":
    main()
