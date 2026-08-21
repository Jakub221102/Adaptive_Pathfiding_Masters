"""Analyze MAPF-7 conflict-aware priority benchmark results (MAPF-7.5).

Open in PyCharm and press Run. No command-line arguments required.
Edit the MAPF-7.5 LOCAL ANALYSIS CONFIGURATION block below to change paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathfinding.src.experiments.mapf_conflict_aware_priority_analysis import (
    OUTPUT_TABLES,
    ConflictAwareAnalysisConfig,
    run_conflict_aware_priority_analysis,
)
from pathfinding.src.experiments.mapf_conflict_aware_priority_plots import (
    render_conflict_aware_priority_plots,
)

# ============================================================
# MAPF-7.5 LOCAL ANALYSIS CONFIGURATION
# ============================================================

CONFLICT_AWARE_RESULTS_CSV = (
    _REPO_ROOT
    / "pathfinding"
    / "results"
    / "mapf_conflict_aware_priority_execution"
    / "results.csv"
)
FIXED_RESULTS_CSV = (
    _REPO_ROOT / "pathfinding" / "results" / "mapf_benchmark_execution" / "results.csv"
)
RANDOM_RESULTS_CSV = (
    _REPO_ROOT
    / "pathfinding"
    / "results"
    / "mapf_random_priority_pilot"
    / "results.csv"
)
SPF_LPF_RESULTS_CSV = (
    _REPO_ROOT
    / "pathfinding"
    / "results"
    / "mapf_priority_strategy_execution"
    / "results.csv"
)
MANIFEST_PATH = (
    _REPO_ROOT / "pathfinding" / "results" / "mapf_benchmarks" / "AR0204SR_manifest.json"
)
OUTPUT_DIR = (
    _REPO_ROOT / "pathfinding" / "results" / "mapf_conflict_aware_priority_analysis"
)

EXPECTED_INSTANCE_COUNT = 27
EXPECTED_RANDOM_K = 10

# ============================================================
# END CONFIGURATION
# ============================================================


def main() -> None:
    config = ConflictAwareAnalysisConfig(
        conflict_aware_results_csv=CONFLICT_AWARE_RESULTS_CSV,
        fixed_results_csv=FIXED_RESULTS_CSV,
        random_results_csv=RANDOM_RESULTS_CSV,
        spf_lpf_results_csv=SPF_LPF_RESULTS_CSV,
        manifest_path=MANIFEST_PATH,
        output_dir=OUTPUT_DIR,
        expected_instance_count=EXPECTED_INSTANCE_COUNT,
        expected_random_k=EXPECTED_RANDOM_K,
    )

    print("MAPF-7.5 conflict-aware priority analysis")
    print(f"  mapf-7 csv:   {config.conflict_aware_results_csv}")
    print(f"  fixed csv:    {config.fixed_results_csv}")
    print(f"  random csv:   {config.random_results_csv}")
    print(f"  spf/lpf csv:  {config.spf_lpf_results_csv}")
    print(f"  manifest:     {config.manifest_path}")
    print(f"  output dir:   {config.output_dir}")
    print()

    result = run_conflict_aware_priority_analysis(
        config,
        plot_writer=render_conflict_aware_priority_plots,
    )

    status = "PASSED" if result.validation.passed else "FAILED"
    print(f"Dataset validation: {status}")
    for message in result.validation.messages:
        print(f"  {message}")

    print()
    if result.validation.passed:
        print(f"Wrote {len(result.tables_written)} tables to {result.output_dir}:")
        for filename in result.tables_written:
            print(f"  - {filename}")
        if result.plots_written:
            print()
            print("Wrote plots:")
            for filename in result.plots_written:
                print(f"  - plots/{filename}")
    else:
        print("Validation failed; authoritative analysis tables were not written.")
        print("Expected outputs on success:")
        for filename in OUTPUT_TABLES:
            print(f"  - {filename}")


if __name__ == "__main__":
    main()
