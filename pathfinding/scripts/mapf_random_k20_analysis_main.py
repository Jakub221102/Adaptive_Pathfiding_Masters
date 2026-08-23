"""Analyze Random K=20 robustness by combining historical K=10 and supplemental runs.

Open in PyCharm and press Run. No command-line arguments required.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathfinding.src.experiments.mapf_random_k20_analysis import (
    RandomK20AnalysisConfig,
    run_random_k20_analysis,
)

# ============================================================
# RANDOM K=20 ANALYSIS CONFIGURATION
# ============================================================

HISTORICAL_K10_JSONL = (
    _REPO_ROOT
    / "pathfinding"
    / "results"
    / "mapf_random_priority_pilot"
    / "results_details.jsonl"
)
SUPPLEMENTAL_JSONL = (
    _REPO_ROOT
    / "pathfinding"
    / "results"
    / "mapf_random_k20_supplemental"
    / "results_details.jsonl"
)
MANIFEST_PATH = (
    _REPO_ROOT / "pathfinding" / "results" / "mapf_benchmarks" / "AR0204SR_manifest.json"
)
OUTPUT_DIR = _REPO_ROOT / "pathfinding" / "results" / "mapf_random_k20_analysis"

# ============================================================
# END CONFIGURATION
# ============================================================


def main() -> None:
    config = RandomK20AnalysisConfig(
        historical_k10_jsonl=HISTORICAL_K10_JSONL,
        supplemental_jsonl=SUPPLEMENTAL_JSONL,
        manifest_path=MANIFEST_PATH,
        output_dir=OUTPUT_DIR,
    )

    print("Random K=20 robustness analysis")
    print(f"  historical k10: {config.historical_k10_jsonl}")
    print(f"  supplemental:   {config.supplemental_jsonl}")
    print(f"  manifest:       {config.manifest_path}")
    print(f"  output dir:     {config.output_dir}")
    print()

    result = run_random_k20_analysis(config)

    print("Validation: PASSED")
    print(f"  K=10 records:        {result.validation.k10_record_count}")
    print(f"  supplemental records:{result.validation.supplemental_record_count}")
    print(f"  combined K=20:       {result.validation.combined_record_count}")
    print(f"  success/failure:     {result.validation.success_count}/"
          f"{result.validation.failure_count}")
    print()
    print("K=10 global summary:")
    print(f"  SoC-sensitive:       {result.k10_global['soc_sensitive_fraction']}")
    print(f"  max SoC range:       {result.k10_global['max_soc_range']}")
    print(f"  mean SoC range:      {result.k10_global['mean_soc_range']:.2f}")
    print()
    print("K=20 global summary:")
    print(f"  SoC-sensitive:       {result.k20_global['soc_sensitive_fraction']}")
    print(f"  max SoC range:       {result.k20_global['max_soc_range']}")
    print(f"  mean SoC range:      {result.k20_global['mean_soc_range']:.2f}")
    print()
    print("Outputs:")
    for name in result.tables_written:
        print(f"  {result.output_dir / name}")
    for name in result.figures_written:
        print(f"  {result.output_dir / (name + '.png')}")
        print(f"  {result.output_dir / (name + '.pdf')}")


if __name__ == "__main__":
    main()
