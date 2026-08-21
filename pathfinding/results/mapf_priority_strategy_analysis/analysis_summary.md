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
- 1 of 27 instances show observed makespan sensitivity under Random K=10.
- SPF differs from Fixed on 4 instances; LPF differs from Fixed on 5; SPF differs from LPF on 7 instances.
- 6 priority-sensitive instances change SoC while successful Random makespan remains constant.

### Pairwise SoC summaries (diff = LEFT SoC - RIGHT SoC; diff < 0 => LEFT better; diff > 0 => RIGHT better)

- Fixed vs SPF: common-success=27, mean diff=40.74, median diff=0.00, LEFT better=0, equal=23, RIGHT better=4
- Fixed vs LPF: common-success=27, mean diff=-3.41, median diff=0.00, LEFT better=2, equal=22, RIGHT better=3
- SPF vs LPF: common-success=27, mean diff=-44.15, median diff=0.00, LEFT better=5, equal=20, RIGHT better=2

### Runtime

- Mean SPF total runtime: 59.63 s (ordering 29.62 s, 49.7% of total; PP 30.00 s).
- Mean LPF total runtime: 66.86 s (ordering 29.69 s, 44.4% of total; PP 37.17 s).
- Median SPF total runtime: 57.40 s; median LPF total runtime: 57.97 s.
- SPF total faster on 19 instances; LPF total faster on 8 instances.
- Median SPF/LPF total runtimes are similar; the higher LPF mean is influenced by expensive priority-sensitive cases rather than uniformly slower LPF behaviour.

### Basic CBS quality reference

- Basic CBS common-success reference available on 22 instances.
- Fixed vs Basic CBS: mean gap=16.50, median gap=0.00, strategy better=0, equal=18, Basic CBS better=4.
- Random K=10 median vs Basic CBS: mean gap=6.45, median gap=0.00, strategy better=0, equal=19, Basic CBS better=3.
- SPF vs Basic CBS: mean gap=0.18, median gap=0.00, strategy better=0, equal=20, Basic CBS better=2.
- LPF vs Basic CBS: mean gap=16.23, median gap=0.00, strategy better=0, equal=20, Basic CBS better=2.

## Interpretation

- Priority ordering can change PP solution quality on a subset of instances even when makespan remains unchanged, which supports studying adaptive or conflict-aware priority selection rather than assuming a single fixed order.
- SPF and LPF do not uniformly dominate Fixed Priority or Random K=10; their value depends on instance-level interaction structure.
- The extra independent-path ordering cost of SPF/LPF is non-trivial and must be accounted for when comparing runtime against Fixed or Random PP.
- Basic CBS remains useful as a quality reference on solvable instances, but timeout/expansion-limit records are excluded from quality means.
