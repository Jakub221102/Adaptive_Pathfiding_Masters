# MAPF-7 Conflict-Aware Priority Analysis Summary

## Dataset validation

- Validation status: PASSED
- MAPF-7 CSV: `pathfinding\results\mapf_conflict_aware_priority_execution\results.csv`
- Manifest: `pathfinding\results\mapf_benchmarks\AR0204SR_manifest.json`

## Methodological notes

- CDF-H remains the pre-declared primary strategy; CDF-L and SPF+CD are required ablations.
- Random sampled best-of-K is diagnostic only, not a fair single-run baseline.
- Basic CBS is a quality reference on common-success instances, not a quality ceiling.
- LOW/MEDIUM/HIGH are independent-path interaction strata, not general difficulty labels.
- Because strata were defined using independent-path conflicts, concentration in HIGH is related to the same underlying interaction signal and is not independent validation.
- SPF runtime comes from MAPF-6; MAPF-7 strategies from a separate benchmark run.

## Confirmed facts

- All 27 instances succeeded for CDF-H, CDF-L, and SPF+CD.
- MAPF-7 strategies differ in SoC on 5 of 27 instances.
- CDF-H vs CDF-L: CDF-H better=0, equal=22, CDF-L better=5; mean diff=16.30, median diff=0.00.
- The pre-declared high-conflict-first direction (CDF-H) was not supported by aggregate SoC counts relative to the CDF-L ablation.
- CDF-H vs SPF: LEFT better=0, equal=23, RIGHT better=4.
- CDF-L vs SPF: LEFT better=2, equal=24, RIGHT better=1.
- SPF+CD vs SPF: identical SoC on 27/27; different SoC on 0/27; different priority order on 0/27.
- MAPF-6 priority-sensitive subset size remains 7 instances (frozen from Random K=10).
- Within that subset, MAPF-7 strategies (CDF-H / CDF-L / SPF+CD) differ in SoC on 5 of 7 instances.

### Runtime

- Mean SPF: ordering 29.62 s, PP 30.00 s, total 59.63 s.
- Mean CDF-H: ordering 29.43 s, PP 33.85 s, total 63.28 s.
- Mean CDF-L: ordering 29.56 s, PP 32.38 s, total 61.94 s.
- Mean SPF+CD: ordering 29.52 s, PP 30.08 s, total 59.60 s.
- Ordering-phase cost is very similar across SPF, CDF-H, CDF-L, and SPF+CD (mean ordering within ~29.43–29.62 s).
- CDF-H and CDF-L have higher mean PP runtime than SPF/SPF+CD because their generated priority orders can make downstream PP more expensive on this catalogue.
- Isolated graph-construction time was not measured separately; comparisons use full ordering phases from separate MAPF-6 (SPF) and MAPF-7 benchmark runs.

### Priority-sensitive subset examples (CDF-L vs SPF, diff = CDF-L SoC − SPF SoC)

- AR0204SR_n05_high_002: SPF=1670, CDF-L=1668, diff=-2 → CDF-L better (lower SoC than SPF).
- AR0204SR_n20_high_001: SPF=4742, CDF-L=5046, diff=304 → CDF-L worse (higher SoC than SPF).

## Interpretation

- Conflict-degree information can change PP outcomes when independent-path conflict structure is heterogeneous, but effects are sparse on this catalogue.
- Direction ablation (CDF-H vs CDF-L) is the primary test of whether high-conflict-first ordering was preferable to low-conflict-first under the frozen methodology.
- SPF+CD isolates whether conflict degree adds signal beyond path length when costs tie.

## Limitations

- Only 27 primary catalogue instances; design was motivated by observations from the same catalogue.
- CDF-H was frozen before MAPF-7 evaluation; CDF-L must not be retroactively promoted to primary.
- Random K=10 samples only 10 permutations per instance.
- Runtime comparisons mix separate historical runs (MAPF-6 vs MAPF-7).
- Isolated graph-construction timing was not recorded; ordering overhead comparisons use full ordering phases.

## MAPF-7 decision support

### A. Stop with CDF-H + ablations and report mixed/negative contribution

**Evidence supporting:**
- MAPF-7 SoC differences occur on only 5/27 instances.
- CDF-H does not clearly dominate SPF on aggregate counts (equal=23, SPF better=4).

**Evidence against:**
- CDF-L may outperform CDF-H on some instances (CDF-L better on 5).
- Effects appear on the frozen priority-sensitive subset (5/7 with MAPF-7 SoC differences).

**Methodological risk:** Accepting a null/mixed result without held-out validation.
**Expected cost:** Analysis/write-up only.

### B. Design one additional frozen conflict-aware variant from observed failure modes

**Evidence supporting:**
- Observed SoC differences on 5 instances show the ordering signal matters on a subset.
- Within the priority-sensitive subset, MAPF-7 SoC differs on 5/7 instances.
- Direction test: CDF-H better 0, CDF-L better 5.
- Graph diagnostics differ descriptively (equal-SoC n=22, differing n=5).

**Evidence against:**
- Additional variants risk post-hoc tuning unless pre-registered and still may not beat SPF.

**Methodological risk:** Overfitting to 27 instances without held-out validation.
**Expected cost:** Small implementation + another 81-run benchmark (or subset ablation).

### C. Proceed to held-out validation of the current frozen family

**Evidence supporting:**
- Current family is fully implemented and benchmarked once on the primary catalogue.
- Held-out catalogue was pre-declared in MAPF-7.1 as preferred over Random K=20.

**Evidence against:**
- If primary-catalogue effect is weak, held-out may confirm null result only.

**Methodological risk:** Low if catalogue generation methodology is unchanged and not used for tuning.
**Expected cost:** Catalogue generation + 81-run benchmark on held-out set.
