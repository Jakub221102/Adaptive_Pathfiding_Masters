# MAPF-6 Thesis Tables

## Table A — Success rates

| Strategy | Success rate |
|---|---:|
| Fixed Priority | 1.000 |
| Random K=10 (instance mean) | 1.000 |
| SPF | 1.000 |
| LPF | 1.000 |

## Table B — Pairwise SoC comparisons

| Comparison | Common success | Mean diff | Median diff | LEFT better | Equal | RIGHT better |
|---|---:|---:|---:|---:|---:|---:|
| fixed_vs_spf | 27 | 40.74 | 0.00 | 0 | 23 | 4 |
| fixed_vs_lpf | 27 | -3.41 | 0.00 | 2 | 22 | 3 |
| spf_vs_lpf | 27 | -44.15 | 0.00 | 5 | 20 | 2 |

## Table C — Random sensitivity summary

| Metric | Value |
|---|---:|
| Priority-sensitive instances | 7 / 27 |
| Max Random SoC range | 719 |

## Table D — SPF vs LPF runtime cost

| Metric | SPF | LPF |
|---|---:|---:|
| Mean total runtime (ms) | 59625.44 | 66861.10 |
| Mean ordering runtime (ms) | 29621.78 | 29694.56 |
| Mean PP runtime (ms) | 30003.66 | 37166.54 |
| Median total runtime (ms) | 57401.33 | 57970.90 |
| Instances where strategy is faster (total) | 19 (SPF) | 8 (LPF) |

## Table E — Selected priority-sensitive cases

| Instance | Agents | Interaction | Random SoC range | Fixed | SPF | LPF | Random median |
|---|---:|---|---:|---:|---:|---:|---:|
| AR0204SR_n20_high_001 | 20 | high | 719 | 5461 | 4742 | 5461 | 5461.0 |
| AR0204SR_n10_high_001 | 10 | high | 221 | 3183 | 2962 | 3181 | 2962.0 |
| AR0204SR_n10_medium_000 | 10 | medium | 138 | 2935 | 2797 | 2935 | 2935.0 |
| AR0204SR_n10_high_000 | 10 | high | 79 | 2509 | 2487 | 2564 | 2493.0 |
| AR0204SR_n10_high_002 | 10 | high | 43 | 3415 | 3415 | 3458 | 3415.0 |
| AR0204SR_n05_high_002 | 5 | high | 2 | 1670 | 1670 | 1668 | 1670.0 |
| AR0204SR_n20_medium_000 | 20 | medium | 2 | 5192 | 5192 | 5190 | 5192.0 |

## Table F — CBS quality reference (common success only)

| Instance | CBS SoC | Fixed | SPF | LPF | Random median | Sampled best-of-K |
|---|---:|---:|---:|---:|---:|---:|
| AR0204SR_n05_low_000 | 1378 | 1378 | 1378 | 1378 | 1378.0 | 1378 |
| AR0204SR_n05_low_001 | 868 | 868 | 868 | 868 | 868.0 | 868 |
| AR0204SR_n05_low_002 | 878 | 878 | 878 | 878 | 878.0 | 878 |
| AR0204SR_n05_medium_000 | 1299 | 1299 | 1299 | 1299 | 1299.0 | 1299 |
| AR0204SR_n05_medium_001 | 1291 | 1291 | 1291 | 1291 | 1291.0 | 1291 |
| AR0204SR_n05_medium_002 | 1396 | 1396 | 1396 | 1396 | 1396.0 | 1396 |
| AR0204SR_n05_high_000 | 1263 | 1263 | 1263 | 1263 | 1263.0 | 1263 |
| AR0204SR_n05_high_001 | 1664 | 1664 | 1664 | 1664 | 1664.0 | 1664 |
| AR0204SR_n05_high_002 | 1668 | 1670 | 1670 | 1668 | 1670.0 | 1668 |
| AR0204SR_n10_low_000 | 2594 | 2594 | 2594 | 2594 | 2594.0 | 2594 |
