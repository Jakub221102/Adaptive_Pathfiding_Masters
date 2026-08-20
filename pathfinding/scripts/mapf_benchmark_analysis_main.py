"""Analyze MAPF-5B benchmark results (MAPF-5C.1).

Open in PyCharm and press Run. No command-line arguments required.
Edit the MAPF-5C.1 LOCAL ANALYSIS CONFIGURATION block below to change paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathfinding.src.experiments.mapf_benchmark_analysis import (
    AnalysisConfig,
    OUTPUT_TABLES,
    run_benchmark_analysis,
)

# ============================================================
# MAPF-5C.1 LOCAL ANALYSIS CONFIGURATION
# ============================================================

RESULTS_JSONL = (
    _REPO_ROOT / "pathfinding" / "results" / "mapf_benchmark_execution" / "results_details.jsonl"
)
MANIFEST_PATH = (
    _REPO_ROOT / "pathfinding" / "results" / "mapf_benchmarks" / "AR0204SR_manifest.json"
)
OUTPUT_DIR = _REPO_ROOT / "pathfinding" / "results" / "mapf_benchmark_analysis"

EXPECTED_RECORD_COUNT = 81
EXPECTED_INSTANCE_COUNT = 27
COMMON_BUDGET_MS = 180_000.0

# ============================================================
# END CONFIGURATION
# ============================================================


def main() -> None:
    config = AnalysisConfig(
        results_jsonl_path=RESULTS_JSONL,
        manifest_path=MANIFEST_PATH,
        output_dir=OUTPUT_DIR,
        expected_record_count=EXPECTED_RECORD_COUNT,
        expected_instance_count=EXPECTED_INSTANCE_COUNT,
        common_budget_ms=COMMON_BUDGET_MS,
    )

    print("MAPF-5C.1 benchmark analysis")
    print(f"  input jsonl: {config.results_jsonl_path}")
    print(f"  manifest:    {config.manifest_path}")
    print(f"  output dir:  {config.output_dir}")
    print()

    result = run_benchmark_analysis(config)

    status = "PASSED" if result.validation.passed else "FAILED"
    print(f"Dataset validation: {status}")
    for message in result.validation.messages:
        print(f"  {message}")

    print()
    if result.validation.passed:
        print(f"Wrote {len(result.tables_written)} files to {result.output_dir}:")
        for filename in result.tables_written:
            print(f"  - {filename}")
    else:
        print("Validation failed; authoritative analysis tables were not written.")
        print("Expected outputs on success:")
        for filename in OUTPUT_TABLES:
            print(f"  - {filename}")


if __name__ == "__main__":
    main()
