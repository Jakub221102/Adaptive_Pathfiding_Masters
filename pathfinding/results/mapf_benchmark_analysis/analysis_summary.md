# MAPF Benchmark Analysis Summary (MAPF-5C.1)

## Dataset validation

- Validation status: **PASSED**
- Total runs analyzed: 81
- Unique instances: 27

## Success by algorithm

- `fixed_priority_pp`: 27/27 success
- `cbs_basic`: 22/27 success
- `cbs_cardinal_first`: 21/27 success

## Success by agent count and interaction level

- agents=5, interaction=low: 9/9 success across all algorithms
- agents=5, interaction=medium: 9/9 success across all algorithms
- agents=5, interaction=high: 9/9 success across all algorithms
- agents=10, interaction=low: 9/9 success across all algorithms
- agents=10, interaction=medium: 9/9 success across all algorithms
- agents=10, interaction=high: 4/9 success across all algorithms
- agents=20, interaction=low: 9/9 success across all algorithms
- agents=20, interaction=medium: 9/9 success across all algorithms
- agents=20, interaction=high: 3/9 success across all algorithms

## Non-success termination counts

- time_limit: 10
- expansion_limit: 1
- failure: 0
- error: 0

## Runtime observations

- Successful solve-time statistics exclude censored timeout and expansion-limit runs.
- Observed execution-time summaries include all completed runs regardless of termination.

## Solution quality (paired common-success cases)

- PP vs Basic CBS: 22 paired instances; mean SoC diff = 16.50 (positive means right-hand algorithm lower SoC)
- PP vs Cardinal-First CBS: 21 paired instances; mean SoC diff = 6.76 (positive means right-hand algorithm lower SoC)
- Basic CBS vs Cardinal-First CBS: 21 paired instances; mean SoC diff = 0.00 (positive means right-hand algorithm lower SoC)

## Basic vs Cardinal search effort

- Common-success CBS pairs: 21
- Mean runtime ratio (Cardinal/Basic): 1.252
- Mean CT expansion reduction (Basic - Cardinal): 0.62
- Mean additional combined low-level searches (Cardinal - Basic): 5.43

Cardinal-First may reduce CT expansions while increasing low-level classification/replan work.

## Post-hoc common 180 s budget comparison (secondary)

- Fixed-Priority PP observed success: 27/27
- Fixed-Priority PP within 180 s post-hoc success: 26/27
- CBS algorithms were already executed under the 180 s wall-clock guard.
- Original PP termination labels were not modified for this comparison.

## Caveats

- Interaction level (LOW/MEDIUM/HIGH) reflects independent-path conflict count, not absolute computational difficulty.
- HIGH instances show substantial internal variation in runtime and CBS outcomes.
- This summary is empirical and descriptive; final thesis conclusions are out of scope for MAPF-5C.1.
