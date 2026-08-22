# MAPF-7.9 Final Thesis Tables

*Sign convention: diff = LEFT SoC − RIGHT SoC; diff < 0 → LEFT better.*

## Table 1 — Primary pairwise summary

| Comparison | Mean diff | Median diff | LEFT better | Equal | RIGHT better |
|---|---:|---:|---:|---:|---:|
| CDF-H vs SPF | 27.41 | 0.00 | 0 | 23 | 4 |
| CDF-L vs SPF | 11.11 | 0.00 | 2 | 24 | 1 |
| CDF-H vs CDF-L | 16.30 | 0.00 | 0 | 22 | 5 |
| SPF+CD vs SPF | 0.00 | 0.00 | 0 | 27 | 0 |

## Table 2 — Held-out pairwise summary

| Comparison | Mean diff | Median diff | LEFT better | Equal | RIGHT better |
|---|---:|---:|---:|---:|---:|
| CDF-H vs SPF | -0.07 | 0.00 | 1 | 26 | 0 |
| CDF-L vs SPF | 9.85 | 0.00 | 0 | 25 | 2 |
| CDF-H vs CDF-L | -9.93 | 0.00 | 3 | 24 | 0 |
| SPF+CD vs SPF | 0.00 | 0.00 | 0 | 27 | 0 |

## Table 3 — Primary vs held-out generalization

| Comparison | Primary (L/E/R) | Primary mean diff | Held-out (L/E/R) | Held-out mean diff | Interpretation |
|---|---|---:|---|---:|---|
| cdf_h_vs_spf | 0/23/4 | 27.41 | 1/26/0 | -0.07 | mean effect direction differed between catalogues |
| cdf_l_vs_spf | 2/24/1 | 11.11 | 0/25/2 | 9.85 | partial replication; aggregate counts differ |
| cdf_h_vs_cdf_l | 0/22/5 | 16.30 | 3/24/0 | -9.93 | direction reversed between catalogues |
| spf_cd_vs_spf | 0/27/0 | 0.00 | 0/27/0 | 0.00 | no practical SPF+CD benefit on either catalogue |
| strategy_soc_sensitivity | 5/27 | | 3/27 | | replicated sparse sensitivity |
| makespan_sensitivity | 1/27 | | 0/27 | | makespan less sensitive than SoC on both catalogues |

## Table 4 — SoC-sensitive instances

| Catalogue | Instance | Agents | Interaction | Conflicts | Pairs | Max deg | CDF-H | CDF-L | SPF+CD | SPF | Range |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| primary | AR0204SR_n20_high_001 | 20 | high | 10 | 10 | 4 | 5157 | 5046 | 4742 | 4742 | 415 |
| primary | AR0204SR_n10_high_001 | 10 | high | 3 | 3 | 3 | 3181 | 2962 | 2962 | 2962 | 219 |
| primary | AR0204SR_n10_high_000 | 10 | high | 10 | 10 | 4 | 2550 | 2485 | 2487 | 2487 | 65 |
| primary | AR0204SR_n10_high_002 | 10 | high | 4 | 4 | 3 | 3458 | 3415 | 3415 | 3415 | 43 |
| primary | AR0204SR_n05_high_002 | 5 | high | 3 | 3 | 3 | 1670 | 1668 | 1670 | 1670 | 2 |
| held_out | AR0204SR_HO_n10_high_000 | 10 | high | 5 | 5 | 4 | 2522 | 2786 | 2522 | 2522 | 264 |
| held_out | AR0204SR_HO_n05_medium_000 | 5 | medium | 2 | 2 | 2 | 1341 | 1343 | 1343 | 1343 | 2 |
| held_out | AR0204SR_HO_n20_medium_001 | 20 | medium | 2 | 2 | 2 | 4591 | 4593 | 4591 | 4591 | 2 |

## Table 5 — Interaction-level comparison

| Catalogue | Interaction | Strategy | Mean SoC diff vs SPF | Better / equal / worse |
|---|---|---|---:|---|
| primary | LOW | CDF-H | 0.00 | 0/9/0 |
| primary | LOW | CDF-L | 0.00 | 0/9/0 |
| primary | LOW | SPF+CD | 0.00 | 0/9/0 |
| primary | LOW | SPF | 0.00 | 0/9/0 |
| primary | MEDIUM | CDF-H | 0.00 | 0/9/0 |
| primary | MEDIUM | CDF-L | 0.00 | 0/9/0 |
| primary | MEDIUM | SPF+CD | 0.00 | 0/9/0 |
| primary | MEDIUM | SPF | 0.00 | 0/9/0 |
| primary | HIGH | CDF-H | 82.22 | 0/5/4 |
| primary | HIGH | CDF-L | 33.33 | 2/6/1 |
| primary | HIGH | SPF+CD | 0.00 | 0/9/0 |
| primary | HIGH | SPF | 0.00 | 0/9/0 |
| held_out | LOW | CDF-H | 0.00 | 0/9/0 |
| held_out | LOW | CDF-L | 0.00 | 0/9/0 |
| held_out | LOW | SPF+CD | 0.00 | 0/9/0 |
| held_out | LOW | SPF | 0.00 | 0/9/0 |
| held_out | MEDIUM | CDF-H | -0.22 | 1/8/0 |
| held_out | MEDIUM | CDF-L | 0.22 | 0/8/1 |
| held_out | MEDIUM | SPF+CD | 0.00 | 0/9/0 |
| held_out | MEDIUM | SPF | 0.00 | 0/9/0 |
| held_out | HIGH | CDF-H | 0.00 | 0/9/0 |
| held_out | HIGH | CDF-L | 29.33 | 0/8/1 |
| held_out | HIGH | SPF+CD | 0.00 | 0/9/0 |
| held_out | HIGH | SPF | 0.00 | 0/9/0 |

## Table 6 — Agent-count comparison

| Catalogue | Agents | Strategy | Mean SoC diff vs SPF | Better / equal / worse |
|---|---:|---|---:|---|
| primary | 5 | CDF-H | 0.00 | 0/9/0 |
| primary | 5 | CDF-L | -0.22 | 1/8/0 |
| primary | 5 | SPF+CD | 0.00 | 0/9/0 |
| primary | 10 | CDF-H | 36.11 | 0/6/3 |
| primary | 10 | CDF-L | -0.22 | 1/8/0 |
| primary | 10 | SPF+CD | 0.00 | 0/9/0 |
| primary | 20 | CDF-H | 46.11 | 0/8/1 |
| primary | 20 | CDF-L | 33.78 | 0/8/1 |
| primary | 20 | SPF+CD | 0.00 | 0/9/0 |
| held_out | 5 | CDF-H | -0.22 | 1/8/0 |
| held_out | 5 | CDF-L | 0.00 | 0/9/0 |
| held_out | 5 | SPF+CD | 0.00 | 0/9/0 |
| held_out | 10 | CDF-H | 0.00 | 0/9/0 |
| held_out | 10 | CDF-L | 29.33 | 0/8/1 |
| held_out | 10 | SPF+CD | 0.00 | 0/9/0 |
| held_out | 20 | CDF-H | 0.00 | 0/9/0 |
| held_out | 20 | CDF-L | 0.22 | 0/8/1 |
| held_out | 20 | SPF+CD | 0.00 | 0/9/0 |

## Table 7 — Makespan sensitivity

| Catalogue | Comparison | Differing instances | Mean makespan diff | LEFT / equal / RIGHT |
|---|---|---:|---:|---|
| primary | cdf_h_vs_spf | 0 | 0.00 | 0/27/0 |
| primary | cdf_l_vs_spf | 1 | -0.07 | 1/26/0 |
| primary | spf_cd_vs_spf | 0 | 0.00 | 0/27/0 |
| primary | cdf_h_vs_cdf_l | 1 | 0.07 | 0/26/1 |
| held_out | cdf_h_vs_spf | 0 | 0.00 | 0/27/0 |
| held_out | cdf_l_vs_spf | 0 | 0.00 | 0/27/0 |
| held_out | spf_cd_vs_spf | 0 | 0.00 | 0/27/0 |
| held_out | cdf_h_vs_cdf_l | 0 | 0.00 | 0/27/0 |

## Table 8 — Runtime summary (aggregate rows)

| Catalogue | Strategy | Mean ordering (s) | Mean PP (s) | Mean total (s) | Ordering fraction |
|---|---|---:|---:|---:|---:|
| primary | SPF | 29.62 | 30.00 | 59.63 | 49.7% |
| primary | CDF-H | 29.43 | 33.85 | 63.28 | 46.5% |
| primary | CDF-L | 29.56 | 32.38 | 61.94 | 47.7% |
| primary | SPF+CD | 29.52 | 30.08 | 59.60 | 49.5% |
| held_out | SPF | 33.81 | 34.22 | 68.03 | 49.7% |
| held_out | CDF-H | 35.60 | 36.39 | 71.99 | 49.5% |
| held_out | CDF-L | 35.73 | 37.23 | 72.96 | 49.0% |
| held_out | SPF+CD | 35.80 | 36.23 | 72.03 | 49.7% |

## Table 9 — SPF+CD ablation (aggregate)

| Catalogue | Instances | Identical SoC | Identical makespan | Identical ordering | Differing ordering |
|---|---:|---:|---:|---:|---:|
| primary | 27 | 27 | 27 | 27 | 0 |
| held_out | 27 | 27 | 27 | 27 | 0 |
| combined | 54 | 54 | 54 | 54 | 0 |
