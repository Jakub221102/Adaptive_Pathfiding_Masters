# MAPF-9 Final Thesis Tables

## Table 1 — Primary MAPF-9 dataset and execution integrity

| Map | Instances | Seed | SPF success | CGLPS success | UBLS success | Sum A_i | Physical PP |
|---|---:|---:|---:|---:|---:|---:|---:|
| AR0400SR | 27 | 2030 | 27/27 | 27/27 | 27/27 | 33 | 93 |
| AR0307SR | 27 | 2031 | 27/27 | 27/27 | 27/27 | 42 | 111 |
| **Pooled** | **54** | — | **54/54** | **54/54** | **54/54** | **75** | **204** |

## Table 2 — Quality outcomes vs SPF (n = 54)

| Method | SoC improved | SoC equal | SoC worse | Total SoC gain | Makespan-only | Lex better | Lex equal | Lex worse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CGLPS | 2/54 (3.7%) | 52 | 0 | 3 | 3 | 5 | 49 | 0 |
| UBLS | 1/54 (1.9%) | 53 | 0 | 1 | 0 | 1 | 53 | 0 |

## Table 3 — Matched-budget CGLPS vs UBLS (n = 54)

### SoC-only

- CGLPS better: **1**
- Equal: **53**
- UBLS better: **0**

### Frozen lexicographic objective (success → SoC → makespan)

- CGLPS better: **4**
- Equal: **50**
- UBLS better: **0**

### Unique benefit relative to SPF (descriptive, not statistical superiority)

- CGLPS unique SoC: **1**
- CGLPS unique lex: **4**
- UBLS unique SoC: **0**
- UBLS unique lex: **0**

## Table 4 — CGLPS lexicographically improved instances (distinct set)

| Map | Instance | n | Interaction | A_i | Cand. | SPF SoC | CGLPS SoC | ΔSoC | SPF MS | CGLPS MS | Effect | Pair rank | Pair events |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| AR0307SR | AR0307SR_n05_high_000 | 5 | high | 1 | 1 | 1037 | 1037 | 0 | 357 | 356 | Makespan-only | 1 | 241 |
| AR0307SR | AR0307SR_n05_high_002 | 5 | high | 1 | 1 | 1832 | 1832 | 0 | 499 | 498 | Makespan-only | 1 | 367 |
| AR0307SR | AR0307SR_n20_high_000 | 20 | high | 4 | 1 | 4417 | 4415 | 2 | 510 | 510 | SoC | 1 | 1 |
| AR0307SR | AR0307SR_n20_medium_002 | 20 | medium | 2 | 2 | 4518 | 4518 | 0 | 433 | 431 | Makespan-only | 2 | 1 |
| AR0400SR | AR0400SR_n05_high_000 | 5 | high | 1 | 1 | 1295 | 1294 | 1 | 478 | 478 | SoC | 1 | 78 |

*Note:* UBLS matched the SoC improvement on `AR0400SR_n05_high_000`; the four remaining CGLPS-vs-UBLS lex wins occurred on AR0307SR.

## Table 5 — Cost-benefit (logical method accounting, n = 54)

| Method | Logical PP candidates | PP ratio vs SPF | Total logical runtime (ms) | Runtime ratio vs SPF | SoC improved | Makespan-only | Total SoC gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| SPF | 54 | 1.0 | 5985447.1 | 1.000 | 0 | 0 | 0 |
| CGLPS | 129 | 2.389 | 11639447.0 | 1.945 | 2 | 3 | 3 |
| UBLS | 129 | 2.389 | 11574579.0 | 1.934 | 1 | 0 | 1 |

*Physical combined experiment:* 204 PP evaluations total; run.log elapsed AR0400SR = 7848.5 s, AR0307SR = 9387.8 s, combined = 17236.3 s. Physical combined PP count is not directly comparable to a single-method logical-runtime row above.
