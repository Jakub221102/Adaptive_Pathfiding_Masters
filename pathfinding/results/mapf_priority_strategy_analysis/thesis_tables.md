# MAPF-6 Thesis Tables

## Table A — Success rates

| Strategy | Successful / total | Success rate |
|---|---:|---:|
| Fixed Priority | 27 / 27 | 100.0% |
| Random K=10 | 270 / 270 runs | 100.0% |
| SPF | 27 / 27 | 100.0% |
| LPF | 27 / 27 | 100.0% |

*Note: 10 nested ordering runs per instance (27 MAPF instances; not 270 independent instances).*

## Table B — Pairwise SoC comparisons

| Comparison | Common success | Mean diff | Median diff | LEFT better | Equal | RIGHT better |
|---|---:|---:|---:|---:|---:|---:|
| Fixed vs SPF | 27 | 40.74 | 0.00 | 0 | 23 | 4 |
| Fixed vs LPF | 27 | -3.41 | 0.00 | 2 | 22 | 3 |
| SPF vs LPF | 27 | -44.15 | 0.00 | 5 | 20 | 2 |

*Sign convention: diff = LEFT SoC - RIGHT SoC. Therefore diff < 0 → LEFT better; diff > 0 → RIGHT better; diff = 0 → equal.*

## Table C — Random priority sensitivity under Random K=10

| Metric | Count / total | Percentage |
|---|---:|---:|
| Observed SoC-sensitive instances | 7 / 27 | 25.9% |
| No observed SoC sensitivity in K=10 | 20 / 27 | 74.1% |
| Observed makespan-sensitive instances | 1 / 27 | 3.7% |
| Maximum sampled Random SoC range | 719 | — |

*Note: K=10 is a sampled subset of possible priority permutations; these counts describe observed sensitivity under that sample, not the full permutation space.*

## Table D — SPF vs LPF runtime cost

| Metric | SPF | LPF |
|---|---:|---:|
| Mean total runtime (s) | 59.63 | 66.86 |
| Median total runtime (s) | 57.40 | 57.97 |
| Mean ordering runtime (s) | 29.62 | 29.69 |
| Mean PP runtime (s) | 30.00 | 37.17 |
| Ordering fraction of mean total runtime | 49.7% | 44.4% |
| Instances where strategy is faster (total) | 19 (SPF) | 8 (LPF) |

*Note: total runtime = ordering time + PP time; ordering includes independent Space-Time A* searches.*

## Table E — Observed priority-sensitive instances under Random K=10

| Instance | Agents | Interaction | Random SoC range | Fixed | SPF | LPF | Random median |
|---|---:|---|---:|---:|---:|---:|---:|
| AR0204SR_n20_high_001 | 20 | high | 719 | 5461 | 4742 | 5461 | 5461.0 |
| AR0204SR_n10_high_001 | 10 | high | 221 | 3183 | 2962 | 3181 | 2962.0 |
| AR0204SR_n10_medium_000 | 10 | medium | 138 | 2935 | 2797 | 2935 | 2935.0 |
| AR0204SR_n10_high_000 | 10 | high | 79 | 2509 | 2487 | 2564 | 2493.0 |
| AR0204SR_n10_high_002 | 10 | high | 43 | 3415 | 3415 | 3458 | 3415.0 |
| AR0204SR_n20_medium_000 | 20 | medium | 2 | 5192 | 5192 | 5190 | 5192.0 |
| AR0204SR_n05_high_002 | 5 | high | 2 | 1670 | 1670 | 1668 | 1670.0 |

*Note: interaction levels LOW/MEDIUM/HIGH are interaction strata, not difficulty labels. All 7 observed SoC-sensitive instances are listed, sorted by Random SoC range descending.*

## Table F — Basic CBS quality reference (aggregate over common-success instances)

*Basic CBS common-success instances: 22. Timeout/expansion-limit cases are excluded from quality averages.*

| Strategy | Common success | Mean SoC gap | Median SoC gap | Strategy better | Equal | Basic CBS better |
|---|---:|---:|---:|---:|---:|---:|
| Fixed | 22 | 16.50 | 0.00 | 0 | 18 | 4 |
| Random K=10 median | 22 | 6.45 | 0.00 | 0 | 19 | 3 |
| SPF | 22 | 0.18 | 0.00 | 0 | 20 | 2 |
| LPF | 22 | 16.23 | 0.00 | 0 | 20 | 2 |
| Random sampled best-of-K (diagnostic) *(diagnostic only)* | 22 | 0.00 | 0.00 | 0 | 22 | 0 |

*Gap definition: strategy SoC - Basic CBS SoC. Negative gap → strategy better; positive gap → Basic CBS better; zero → equal.*
