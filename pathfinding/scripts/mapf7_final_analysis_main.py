"""MAPF-7.9 final primary + held-out generalization analysis.

Open in PyCharm and press Run. No command-line arguments required.
Edit the MAPF-7.9 LOCAL ANALYSIS CONFIGURATION block below to change paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathfinding.src.experiments.mapf7_final_analysis import (
    FINAL_ANALYSIS_DIR,
    OUTPUT_TABLES,
    Mapf7FinalAnalysisConfig,
    run_mapf7_final_analysis,
)
from pathfinding.src.experiments.mapf7_final_analysis_plots import (
    render_mapf7_final_analysis_plots,
)

# ============================================================
# MAPF-7.9 LOCAL ANALYSIS CONFIGURATION
# ============================================================

PRIMARY_CONFLICT_CSV = (
    _REPO_ROOT
    / "pathfinding"
    / "results"
    / "mapf_conflict_aware_priority_execution"
    / "results.csv"
)
PRIMARY_MANIFEST_PATH = (
    _REPO_ROOT / "pathfinding" / "results" / "mapf_benchmarks" / "AR0204SR_manifest.json"
)
PRIMARY_SPF_LPF_CSV = (
    _REPO_ROOT
    / "pathfinding"
    / "results"
    / "mapf_priority_strategy_execution"
    / "results.csv"
)
FIXED_RESULTS_CSV = (
    _REPO_ROOT / "pathfinding" / "results" / "mapf_benchmark_execution" / "results.csv"
)
RANDOM_RESULTS_CSV = (
    _REPO_ROOT / "pathfinding" / "results" / "mapf_random_priority_pilot" / "results.csv"
)
HELDOUT_CONFLICT_CSV = (
    _REPO_ROOT
    / "pathfinding"
    / "results"
    / "mapf_conflict_aware_priority_heldout_execution"
    / "results.csv"
)
HELDOUT_SPF_CSV = (
    _REPO_ROOT / "pathfinding" / "results" / "mapf_spf_heldout_execution" / "results.csv"
)
HELDOUT_MANIFEST_PATH = (
    _REPO_ROOT
    / "pathfinding"
    / "results"
    / "mapf_benchmarks"
    / "AR0204SR_heldout_manifest.json"
)
OUTPUT_DIR = _REPO_ROOT / FINAL_ANALYSIS_DIR

EXPECTED_INSTANCE_COUNT = 27

# ============================================================
# END CONFIGURATION
# ============================================================


def main() -> None:
    config = Mapf7FinalAnalysisConfig(
        primary_conflict_csv=PRIMARY_CONFLICT_CSV,
        primary_manifest_path=PRIMARY_MANIFEST_PATH,
        primary_spf_lpf_csv=PRIMARY_SPF_LPF_CSV,
        fixed_results_csv=FIXED_RESULTS_CSV,
        random_results_csv=RANDOM_RESULTS_CSV,
        heldout_conflict_csv=HELDOUT_CONFLICT_CSV,
        heldout_spf_csv=HELDOUT_SPF_CSV,
        heldout_manifest_path=HELDOUT_MANIFEST_PATH,
        output_dir=OUTPUT_DIR,
        expected_instance_count=EXPECTED_INSTANCE_COUNT,
    )

    print("MAPF-7.9 final analysis (primary + held-out generalization)")
    print(f"  primary mapf-7:  {config.primary_conflict_csv}")
    print(f"  held-out mapf-7: {config.heldout_conflict_csv}")
    print(f"  held-out spf:    {config.heldout_spf_csv}")
    print(f"  output dir:      {config.output_dir}")
    print()

    result = run_mapf7_final_analysis(
        config,
        plot_writer=render_mapf7_final_analysis_plots,
    )

    status = "PASSED" if result.validation.passed else "FAILED"
    print(f"Dataset validation: {status}")
    for message in result.validation.messages:
        print(f"  {message}")

    print()
    if result.validation.passed:
        print(f"Wrote {len(result.tables_written)} artifacts to {result.output_dir}:")
        for filename in result.tables_written:
            print(f"  - {filename}")
        if result.plots_written:
            print()
            print("Wrote figures:")
            for filename in result.plots_written:
                print(f"  - figures/{filename}")
    else:
        print("Validation failed; final analysis tables were not written.")
        print("Expected outputs on success:")
        for filename in OUTPUT_TABLES:
            print(f"  - {filename}")


if __name__ == "__main__":
    main()
