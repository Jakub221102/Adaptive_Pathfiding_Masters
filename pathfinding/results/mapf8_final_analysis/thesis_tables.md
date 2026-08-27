# MAPF-8 Final Thesis Tables

## Table A — Catalogue-level summary (four-strategy SoC range)

| Catalogue | Instances | SoC-sensitive | Makespan-sensitive | Max SoC range | Mean SoC range |
|---|---:|---:|---:|---:|---:|
| AR0204SR primary | 27 | 5/27 (18.5%) | 1/27 (3.7%) | 415 | — |
| AR0204SR held-out | 27 | 3/27 (11.1%) | 0/27 (0.0%) | 264 | — |
| AR0400SR | 27 | 2/27 (7.4%) | 1/27 (3.7%) | 396 | 14.70 |
| AR0307SR | 27 | 2/27 (7.4%) | 1/27 (3.7%) | 22 | 0.85 |

### Aggregate (not an independent catalogue)

| Summary | Instances | SoC-sensitive | Makespan-sensitive | Max SoC range | Mean SoC range |
|---|---:|---:|---:|---:|---:|
| pooled cross-map (AR0400SR + AR0307SR aggregate) | 54 | 4/54 (7.4%) | 2/54 (3.7%) | 396 | 7.78 |

## Table B — Pairwise cross-map comparisons (diff = LEFT − RIGHT)

| Comparison | LEFT better | Equal | RIGHT better | Mean diff | Median diff | Max abs(diff) |
|---|---:|---:|---:|---:|---:|---:|
| CDF-H vs SPF | 0 | 52 | 2 | 0.26 | 0.00 | 13 |
| CDF-L vs SPF | 1 | 51 | 2 | 7.72 | 0.00 | 396 |
| CDF-H vs CDF-L | 2 | 50 | 2 | -7.46 | 0.00 | 396 |
| SPF+CD vs SPF | 0 | 54 | 0 | 0.00 | 0.00 | 0 |

## Table C — Ordering behaviour

| Comparison | Result |
|---|---|
| SPF+CD vs SPF (cross-map) | same order = 54/54 |
| SPF+CD vs SPF (historical AR0204SR) | same order = 54/54 |
| SPF+CD vs SPF (combined four catalogues) | same order = 108/108 |
| CDF-H vs SPF (cross-map) | different order = 36/54; SoC different = 2/54 |
| CDF-L vs SPF (cross-map) | different order = 34/54; SoC different = 3/54 |

## Table D — Sensitive-instance strategy SoC values (supporting detail)

Supporting detail for Figure 2; primary Figure 2 displays four-strategy SoC range.

| Catalogue | instance_id | SPF | CDF-H | CDF-L | SPF+CD | SoC range |
|---|---|---:|---:|---:|---:|---:|
| AR0204SR primary | AR0204SR_n20_high_001 | 4742 | 5157 | 5046 | 4742 | 415 |
| AR0400SR | AR0400SR_n20_high_001 | 6315 | 6315 | 6711 | 6315 | 396 |
| AR0204SR held-out | AR0204SR_HO_n10_high_000 | 2522 | 2522 | 2786 | 2522 | 264 |
| AR0204SR primary | AR0204SR_n10_high_001 | 2962 | 3181 | 2962 | 2962 | 219 |
| AR0204SR primary | AR0204SR_n10_high_000 | 2487 | 2550 | 2485 | 2487 | 65 |
| AR0204SR primary | AR0204SR_n10_high_002 | 3415 | 3458 | 3415 | 3415 | 43 |
| AR0307SR | AR0307SR_n20_high_002 | 4955 | 4968 | 4977 | 4955 | 22 |
| AR0204SR held-out | AR0204SR_HO_n05_medium_000 | 1343 | 1341 | 1343 | 1343 | 2 |
| AR0204SR held-out | AR0204SR_HO_n20_medium_001 | 4591 | 4591 | 4593 | 4591 | 2 |
| AR0204SR primary | AR0204SR_n05_high_002 | 1670 | 1670 | 1668 | 1670 | 2 |
| AR0307SR | AR0307SR_n20_high_000 | 5565 | 5566 | 5565 | 5565 | 1 |
| AR0400SR | AR0400SR_n05_high_000 | 1742 | 1742 | 1741 | 1742 | 1 |
