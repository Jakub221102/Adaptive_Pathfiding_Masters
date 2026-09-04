"""Generate thesis-ready MAPF figures from frozen analysis CSVs (GRAPHICS-1A).

Open in PyCharm and press Run. No planner or benchmark execution.
Outputs: docs/thesis/img/thesis_ch*.pdf and .png
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathfinding.src.experiments.thesis_mapf_figures import (
    THESIS_FIGURE_BASENAMES,
    default_thesis_mapf_figures_config,
    run_thesis_mapf_figures,
)


def main() -> None:
    config = default_thesis_mapf_figures_config(_REPO_ROOT)
    try:
        result = run_thesis_mapf_figures(config)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1) from None

    print(f"Thesis MAPF figures written to: {result.output_dir}")
    print(f"Figure basenames ({len(THESIS_FIGURE_BASENAMES)}):")
    for basename in THESIS_FIGURE_BASENAMES:
        print(f"  - {basename}.pdf")
        print(f"  - {basename}.png")
    print(f"Total files: {len(result.files_written)}")


if __name__ == "__main__":
    main()
