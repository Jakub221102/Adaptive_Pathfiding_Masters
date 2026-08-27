"""MAPF-8.5 formal cross-map analysis (analysis only; no planner execution).

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

from pathfinding.src.experiments.mapf8_cross_map_analysis import (
    OUTPUT_TABLES,
    default_mapf8_config,
    run_mapf8_cross_map_analysis,
)


def main() -> None:
    config = default_mapf8_config(_REPO_ROOT)
    try:
        result = run_mapf8_cross_map_analysis(config)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1) from None

    print(f"Validation: {'PASSED' if result.validation.passed else 'FAILED'}")
    print(f"Output directory: {result.output_dir}")
    print(f"Tables written: {len(result.tables_written)}/{len(OUTPUT_TABLES)}")
    for name in result.tables_written:
        print(f"  - {name}")

    if not result.validation.passed:
        raise SystemExit(1)

    ar0400_sensitive = sum(1 for row in result.ar0400_instance_rows if row["soc_sensitive"])
    ar0307_sensitive = sum(1 for row in result.ar0307_instance_rows if row["soc_sensitive"])
    pooled_sensitive = sum(
        1 for row in result.pooled_instance_rows if row["soc_sensitive"]
    )
    print(
        f"SoC-sensitive (four-strategy): AR0400SR={ar0400_sensitive}/27, "
        f"AR0307SR={ar0307_sensitive}/27, pooled={pooled_sensitive}/54"
    )
    raise SystemExit(0)


if __name__ == "__main__":
    main()
