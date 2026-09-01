"""PyCharm-runnable MAPF-9.4 primary production benchmark entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathfinding.src.experiments.mapf9_primary_execution import (
    MAPF9_ALLOWED_MAPS,
    mapf9_primary_execution_config,
    run_mapf9_primary_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MAPF-9 primary production benchmark harness (SPF/CGLPS/UBLS).",
    )
    parser.add_argument(
        "--map",
        required=True,
        choices=sorted(MAPF9_ALLOWED_MAPS),
        help="Frozen MAPF-9 primary map target (AR0400SR or AR0307SR).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint in the MAPF-9 results directory.",
    )
    parser.add_argument(
        "--reset-results",
        action="store_true",
        help="Clear ONLY this map's MAPF-9 primary results directory before running.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate manifest and print execution plan without planner calls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = mapf9_primary_execution_config(
        args.map,
        _REPO_ROOT,
        resume=args.resume,
        reset_results=args.reset_results,
        plan_only=args.plan_only,
    )
    raise SystemExit(
        run_mapf9_primary_benchmark(config, repo_root=_REPO_ROOT)
    )


if __name__ == "__main__":
    main()
