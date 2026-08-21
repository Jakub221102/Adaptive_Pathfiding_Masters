# MAPF-7 Thesis Tables

## Table A — MAPF-7 success

| Strategy | Successful / total | Success rate |
|---|---:|---:|
| CDF-H (primary) | 27 / 27 | 100.0% |
| CDF-L (ablation) | 27 / 27 | 100.0% |
| SPF+CD (ablation) | 27 / 27 | 100.0% |

## Table B — Pairwise SoC comparisons (selected)

| Comparison | Common success | Mean diff | Median diff | LEFT better | Equal | RIGHT better |
|---|---:|---:|---:|---:|---:|---:|
| CDF-H vs Fixed | 27 | -13.33 | 0.00 | 3 | 22 | 2 |
| CDF-H vs SPF | 27 | 27.41 | 0.00 | 0 | 23 | 4 |
| CDF-H vs CDF-L | 27 | 16.30 | 0.00 | 0 | 22 | 5 |
| CDF-L vs SPF | 27 | 11.11 | 0.00 | 2 | 24 | 1 |
| SPF+CD vs SPF | 27 | 0.00 | 0.00 | 0 | 27 | 0 |

*Sign convention: diff = LEFT SoC − RIGHT SoC.*

## Table C — Priority-sensitive subset (MAPF-6 frozen, n=7)

*Within this subset, MAPF-7 strategies differ in SoC on 5 of 7 instances (derived from stored CDF-H / CDF-L / SPF+CD results).*

| Instance | Agents | Interaction | SPF | CDF-H | CDF-L | SPF+CD | MAPF-7 differ |
|---|---:|---|---:|---:|---:|---:|---|
| AR0204SR_n05_high_002 | 5 | high | 1670 | 1670 | 1668 | 1670 | yes |
| AR0204SR_n10_high_000 | 10 | high | 2487 | 2550 | 2485 | 2487 | yes |
| AR0204SR_n10_medium_000 | 10 | medium | 2797 | 2797 | 2797 | 2797 | no |
| AR0204SR_n10_high_001 | 10 | high | 2962 | 3181 | 2962 | 2962 | yes |
| AR0204SR_n10_high_002 | 10 | high | 3415 | 3458 | 3415 | 3415 | yes |
| AR0204SR_n20_medium_000 | 20 | medium | 5192 | 5192 | 5192 | 5192 | no |
| AR0204SR_n20_high_001 | 20 | high | 4742 | 5157 | 5046 | 4742 | yes |

*CDF-L vs SPF sign convention: diff = CDF-L SoC − SPF SoC; negative → CDF-L better; positive → CDF-L worse.*

## Table D — Interaction-stratified mean SoC diff vs SPF

| Interaction | Strategy | Mean SoC diff vs SPF | Better / equal / worse vs SPF |
|---|---|---:|---|
| LOW | CDF-H | 0.00 | 0 / 9 / 0 |
| LOW | CDF-L | 0.00 | 0 / 9 / 0 |
| LOW | SPF+CD | 0.00 | 0 / 9 / 0 |
| LOW | SPF | 0.00 | 0 / 9 / 0 |
| MEDIUM | CDF-H | 0.00 | 0 / 9 / 0 |
| MEDIUM | CDF-L | 0.00 | 0 / 9 / 0 |
| MEDIUM | SPF+CD | 0.00 | 0 / 9 / 0 |
| MEDIUM | SPF | 0.00 | 0 / 9 / 0 |
| HIGH | CDF-H | 82.22 | 0 / 5 / 4 |
| HIGH | CDF-L | 33.33 | 2 / 6 / 1 |
| HIGH | SPF+CD | 0.00 | 0 / 9 / 0 |
| HIGH | SPF | 0.00 | 0 / 9 / 0 |

## Table E — Runtime comparison (mean, seconds)

| Strategy | Ordering | PP | Total | Ordering fraction |
|---|---:|---:|---:|---:|
| SPF | 29.62 | 30.00 | 59.63 | 49.7% |
| CDF-H | 29.43 | 33.85 | 63.28 | 46.5% |
| CDF-L | 29.56 | 32.38 | 61.94 | 47.7% |
| SPF+CD | 29.52 | 30.08 | 59.60 | 49.5% |

## Table F — Basic CBS quality reference

| Strategy | Common success | Mean gap | Median gap | Strategy better | Equal | Basic CBS better |
|---|---:|---:|---:|---:|---:|---:|
| Fixed | 22 | 16.50 | 0.00 | 0 | 18 | 4 |
| SPF | 22 | 0.18 | 0.00 | 0 | 20 | 2 |
| CDF-H | 22 | 10.14 | 0.00 | 0 | 19 | 3 |
| CDF-L | 22 | 0.09 | 0.00 | 0 | 21 | 1 |
| SPF+CD | 22 | 0.18 | 0.00 | 0 | 20 | 2 |
| Random K=10 median | 22 | 6.45 | 0.00 | 0 | 19 | 3 |
| Random sampled best-of-K (diagnostic) *(diagnostic)* | 22 | 0.00 | 0.00 | 0 | 22 | 0 |

## Table G — MAPF-7 differing instances

| Instance | Agents | Interaction | CDF-H | CDF-L | SPF+CD | SPF | Range |
|---|---:|---|---:|---:|---:|---:|---:|
| AR0204SR_n05_high_002 | 5 | high | 1670 | 1668 | 1670 | 1670 | 2 |
| AR0204SR_n10_high_000 | 10 | high | 2550 | 2485 | 2487 | 2487 | 65 |
| AR0204SR_n10_high_001 | 10 | high | 3181 | 2962 | 2962 | 2962 | 219 |
| AR0204SR_n10_high_002 | 10 | high | 3458 | 3415 | 3415 | 3415 | 43 |
| AR0204SR_n20_high_001 | 20 | high | 5157 | 5046 | 4742 | 4742 | 415 |
