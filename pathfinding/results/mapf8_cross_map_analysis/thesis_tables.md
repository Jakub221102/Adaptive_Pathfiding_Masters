# MAPF-8 Thesis Tables (draft)

## Table 1 — Catalogue sensitivity summary (four-strategy SoC range)

| Catalogue | Instances | SoC-sensitive | Makespan-sensitive | Max SoC range | Mean SoC range | MAPF-7 3-strategy sensitive (secondary) |
|---|---:|---:|---:|---:|---:|---:|
| AR0204SR primary | 27 | 5 (18.5%) | 1 (3.7%) | 415 | — | 5 |
| AR0204SR held-out | 27 | 3 (11.1%) | 0 (0.0%) | 264 | — | 3 |
| AR0400SR | 27 | 2 (7.4%) | 1 (3.7%) | 396 | 14.70 | 2 (7.4%) |
| AR0307SR | 27 | 2 (7.4%) | 1 (3.7%) | 22 | 0.85 | 2 (7.4%) |
| pooled cross-map (AR0400SR + AR0307SR aggregate) | 54 | 4 (7.4%) | 2 (3.7%) | 396 | 7.78 | 4 (7.4%) |

## Table 2 — Pooled pairwise SoC comparisons (diff = LEFT − RIGHT)

| Comparison | LEFT better | Equal | RIGHT better | Mean diff | Median diff | Max abs(diff) |
|---|---:|---:|---:|---:|---:|---:|
| CDF-H vs SPF | 0 | 52 | 2 | 0.26 | 0.00 | 13 |
| CDF-L vs SPF | 1 | 51 | 2 | 7.72 | 0.00 | 396 |
| CDF-H vs CDF-L | 2 | 50 | 2 | -7.46 | 0.00 | 396 |
| SPF+CD vs SPF | 0 | 54 | 0 | 0.00 | 0.00 | 0 |

## Table 3 — SPF+CD order equality

### Cross-map (AR0400SR + AR0307SR)

- Same order: 54/54
- Different order: 0/54
- CDF-H different order from SPF (cross-map only): 36/54
- CDF-L different order from SPF (cross-map only): 34/54

### Historical AR0204SR (primary + held-out)

- Same order: 54/54
- Different order: 0/54

### Combined four catalogues

- Same order: 108/108
- Different order: 0/108
