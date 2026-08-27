"""MAPF-8.6 final thesis figures and interpretation (presentation only).

Open in PyCharm and press Run. No command-line arguments required.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathfinding.src.experiments.mapf8_final_analysis import (
    MANDATORY_FIGURE_BASENAMES,
    default_mapf8_final_config,
    run_mapf8_final_analysis,
)


def main() -> None:
    config = default_mapf8_final_config(_REPO_ROOT)
    try:
        result = run_mapf8_final_analysis(config)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1) from None

    print(f"Output directory: {result.output_dir}")
    print(f"Sensitive instances plotted: {result.sensitive_instance_count}")
    print(f"Figures written: {len(result.figures_written)}")
    for name in result.figures_written:
        print(f"  - figures/{name}")
    print("Markdown written:")
    for name in result.markdown_written:
        print(f"  - {name}")
    print(f"Mandatory figure set: {len(MANDATORY_FIGURE_BASENAMES)} figures")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
