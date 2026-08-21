# MAPF-6 Priority Strategy Analysis Summary

## Dataset validation

- Validation status: PASSED
- Manifest: `C:\Users\Jakub\PythonProject\Adaptive_Pathfiding_Masters\pathfinding\results\mapf_benchmarks\AR0204SR_manifest.json`
- Fixed / CBS CSV: `C:\Users\Jakub\PythonProject\Adaptive_Pathfiding_Masters\pathfinding\results\mapf_benchmark_execution\results.csv`
- Random K=10 CSV: `C:\Users\Jakub\PythonProject\Adaptive_Pathfiding_Masters\pathfinding\results\mapf_random_priority_pilot\results.csv`
- SPF / LPF CSV: `C:\Users\Jakub\PythonProject\Adaptive_Pathfiding_Masters\pathfinding\results\mapf_priority_strategy_execution\results.csv`

## Methodological notes

- Random Priority baseline uses K=10 sampled orderings per instance.
- K=20 is deferred as an optional later extension; this analysis infrastructure accepts additional Random rows automatically.
- The MAPF instance is the unit of analysis; Random runs are aggregated per instance before cross-instance summaries.
- Random minimum SoC is labeled as a sampled best-of-K diagnostic, not a fair single-run baseline.
- Random runtime uses measured PP execution time only; permutation overhead was not separately instrumented.
- SPF/LPF runtime uses `total_time_ms = ordering_time_ms + pp_time_ms`, including independent Space-Time A* ordering cost.
- LOW/MEDIUM/HIGH remain interaction-level strata from independent-path conflicts, not absolute difficulty labels.

## Direct empirical findings

- All 27 catalogue instances succeeded for Fixed Priority, Random K=10, SPF, and LPF in the stored datasets.
- 7 of 27 instances show observed priority sensitivity (>1 unique successful Random SoC among K=10 orderings).
- SPF differs from Fixed on 4 instances; LPF differs from Fixed on 5; SPF differs from LPF on 7 instances.
- 6 priority-sensitive instances change SoC while successful Random makespan remains constant.

### Pairwise SoC summaries (diff = LEFT - RIGHT; positive => RIGHT better)

- fixed_vs_spf: common-success=27, mean diff=40.74, median diff=0.00, LEFT better=0, equal=23, RIGHT better=4
- fixed_vs_lpf: common-success=27, mean diff=-3.41, median diff=0.00, LEFT better=2, equal=22, RIGHT better=3
- spf_vs_lpf: common-success=27, mean diff=-44.15, median diff=0.00, LEFT better=5, equal=20, RIGHT better=2

### Runtime

- Mean SPF total runtime: 59625.44 ms (ordering 29621.78 ms, PP 30003.66 ms).
- Mean LPF total runtime: 66861.10 ms (ordering 29694.56 ms, PP 37166.54 ms).
- SPF total faster on 19 instances; LPF total faster on 8 instances.

### CBS quality reference

- Common-success Basic CBS reference available on 22 instances.

## Interpretation

- Priority ordering can change PP solution quality on a subset of instances even when makespan remains unchanged, which supports studying adaptive or conflict-aware priority selection rather than assuming a single fixed order.
- SPF and LPF do not uniformly dominate Fixed Priority or Random K=10; their value depends on instance-level interaction structure.
- The extra independent-path ordering cost of SPF/LPF is non-trivial and must be accounted for when comparing runtime against Fixed or Random PP.
- CBS remains useful as a quality reference on solvable instances, but timeout/expansion-limit records are excluded from quality means.
