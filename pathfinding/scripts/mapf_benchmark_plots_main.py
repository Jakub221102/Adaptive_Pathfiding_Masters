"""Generate thesis-ready MAPF benchmark plots (MAPF-5C.2).

Open in PyCharm and press Run. No command-line arguments required.
Edit the MAPF-5C.2 LOCAL PLOTTING CONFIGURATION block below to change paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathfinding.src.experiments.mapf_benchmark_plots import (
    FIGURE_BASENAMES,
    PlotInputs,
    PlotOutputs,
    generate_benchmark_plots,
)

# ============================================================
# MAPF-5C.2 LOCAL PLOTTING CONFIGURATION
# ============================================================

ANALYSIS_DIR = _REPO_ROOT / "pathfinding" / "results" / "mapf_benchmark_analysis"
PLOTS_DIR = ANALYSIS_DIR / "plots"
THESIS_TABLES_PATH = ANALYSIS_DIR / "thesis_tables.md"

# ============================================================
# END CONFIGURATION
# ============================================================


def main() -> None:
    inputs = PlotInputs(analysis_dir=ANALYSIS_DIR)
    outputs = PlotOutputs(
        plots_dir=PLOTS_DIR,
        thesis_tables_path=THESIS_TABLES_PATH,
    )

    print("MAPF-5C.2 benchmark plotting")
    print(f"  analysis dir: {inputs.analysis_dir}")
    print(f"  plots dir:    {outputs.plots_dir}")
    print(f"  tables file:  {outputs.thesis_tables_path}")
    print()

    result = generate_benchmark_plots(inputs, outputs)

    print(f"Wrote thesis tables: {result.thesis_tables_path}")
    print(f"Wrote {len(result.figure_paths)} figure files to {result.plots_dir}:")
    for basename in FIGURE_BASENAMES:
        print(f"  - {basename}.png")
        print(f"  - {basename}.pdf")


if __name__ == "__main__":
    main()
