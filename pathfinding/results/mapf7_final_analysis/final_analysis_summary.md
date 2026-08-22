# MAPF-7 Final Analysis Summary (MAPF-7.9)

## 1. Data / validation

- Validation status: **PASSED**
- Primary MAPF-7: `pathfinding\results\mapf_conflict_aware_priority_execution\results.csv` (27 instances × 3 strategies = 81 records)
- Held-out MAPF-7: `pathfinding\results\mapf_conflict_aware_priority_heldout_execution\results.csv` (81 records, seed 2027)
- Held-out SPF: `pathfinding\results\mapf_spf_heldout_execution\results.csv` (27 records)
- Held-out validation is **scenario-level on the same MovingAI map/source**, not cross-map validation.
- Primary and held-out catalogues contain **different instances**; no instance-level pairing was performed.

## 2. Primary findings

- CDF-H vs SPF: LEFT better=0, equal=23, RIGHT better=4; mean diff=27.41.
- CDF-L vs SPF: LEFT better=2, equal=24, RIGHT better=1; mean diff=11.11.
- CDF-H vs CDF-L: LEFT better=0, equal=22, RIGHT better=5; mean diff=16.30.
- SPF+CD vs SPF: identical SoC on 27/27; identical ordering on 27/27.
- MAPF-7 strategies differ in SoC on 5/27 primary instances.

## 3. Held-out findings

- CDF-H vs SPF: LEFT better=1, equal=26, RIGHT better=0; mean diff=-0.07.
- CDF-L vs SPF: LEFT better=0, equal=25, RIGHT better=2; mean diff=9.85.
- CDF-H vs CDF-L: LEFT better=3, equal=24, RIGHT better=0; mean diff=-9.93.
- SPF+CD vs SPF: identical SoC on 27/27; identical ordering on 27/27; differing ordering on 0/27.
- MAPF-7 strategies differ in SoC on 3/27 held-out instances.

## 4. Generalization comparison

- **cdf_h_vs_spf**: primary (0/23/4, mean=27.41); held-out (1/26/0, mean=-0.07) → mean effect direction differed between catalogues
- **cdf_l_vs_spf**: primary (2/24/1, mean=11.11); held-out (0/25/2, mean=9.85) → partial replication; aggregate counts differ
- **cdf_h_vs_cdf_l**: primary (0/22/5, mean=16.30); held-out (3/24/0, mean=-9.93) → direction reversed between catalogues
- **spf_cd_vs_spf**: primary (0/27/0, mean=0.00); held-out (0/27/0, mean=0.00) → no practical SPF+CD benefit on either catalogue
- **strategy_soc_sensitivity**: primary=5/27, held-out=3/27 → replicated sparse sensitivity
- **makespan_sensitivity**: primary=1/27, held-out=0/27 → makespan less sensitive than SoC on both catalogues
- **spf_cd_ablation_identical_soc**: primary identical SoC=27/27, held-out identical SoC=27/27, combined identical SoC=54/54 → primary identical ordering=27/27; held-out identical ordering=27/27; combined identical ordering=54/54

## 5. Direction ablation (CDF-H vs CDF-L)

- Primary catalogue: CDF-H better=0, equal=22, CDF-L better=5.
- Held-out catalogue: CDF-H better=3, equal=24, CDF-L better=0.
- The observed direction **reversed between catalogues**: on the primary catalogue CDF-L was better on more instances than CDF-H, whereas on the held-out catalogue CDF-H was better on more instances than CDF-L.

## 6. SPF+CD ablation

- Primary: identical SoC=27/27, identical makespan=27/27, identical ordering=27/27.
- Held-out: identical SoC=27/27, identical makespan=27/27, identical ordering=27/27, differing ordering=0/27.
- Combined (54 instances): identical SoC=54/54, identical ordering=54/54, differing ordering=0/54.
- Across all 54 primary and held-out instances, SPF+CD produced exactly the same priority ordering as plain SPF; the conflict-degree tie-break therefore never changed the resulting order in this experimental sample.

## 7. SoC vs makespan sensitivity

- Primary: SoC-sensitive=5/27, makespan-sensitive=1/27.
- Held-out: SoC-sensitive=3/27, makespan-sensitive=0/27.
- Makespan was substantially less sensitive to tested priority-ordering strategies than SoC within this experimental setting.

### Sensitive instances by interaction stratum (descriptive)

- Primary: LOW=0, MEDIUM=0, HIGH=5.
- Held-out: LOW=0, MEDIUM=2, HIGH=1.
- Interaction strata are derived from independent-path conflicts; HIGH concentration is not independent evidence that conflict information predicts sensitivity.

## 8. Conflict-structure diagnostics

- primary / soc_sensitive (n=5): mean conflicts=6.00, mean pairs=6.00, mean max degree=3.40, mean SoC range=148.80.
- primary / soc_insensitive (n=22): mean conflicts=3.18, mean pairs=1.27, mean max degree=0.95, mean SoC range=0.00.
- held_out / soc_sensitive (n=3): mean conflicts=3.00, mean pairs=3.00, mean max degree=2.67, mean SoC range=89.33.
- held_out / soc_insensitive (n=24): mean conflicts=5.54, mean pairs=1.62, mean max degree=1.17, mean SoC range=0.00.
- Conflict-graph structure may be descriptively associated with priority sensitivity, but degree alone does not reliably exploit that information.

### Case study: AR0204SR_HO_n10_high_001

- Independent conflict events: 97; conflict pairs: 3; max degree: 2.
- SoC (CDF-H / CDF-L / SPF+CD / SPF): 2898 / 2898 / 2898 / 2898.
- Descriptive case study only: high conflict-event frequency with low pair count does not imply MAPF-7 SoC sensitivity (all strategies reached SoC 2898). Orders differ but final cost does not.

## 9. Runtime

- Held-out mean SPF: ordering 33.81 s, PP 34.22 s, total 68.03 s (ordering fraction 49.7%).
- Held-out mean CDF-H: ordering 35.60 s, PP 36.39 s, total 71.99 s (ordering fraction 49.5%).
- Held-out mean CDF-L: ordering 35.73 s, PP 37.23 s, total 72.96 s (ordering fraction 49.0%).
- Held-out mean SPF+CD: ordering 35.80 s, PP 36.23 s, total 72.03 s (ordering fraction 49.7%).
- Ordering phases are of similar overall magnitude across strategies.
- Total runtime changes can also result from the selected priority order making subsequent PP easier or harder.
- Isolated graph-construction overhead was not separately benchmarked.

## 10. Main thesis interpretation

### Supported findings

- **A.** Priority ordering affects SoC only on a minority of tested instances (primary 5/27, held-out 3/27).
- **B.** Makespan is less sensitive than SoC (primary 1/27, held-out 0/27).
- **C.** Simple conflict-degree direction is not a stable general rule across catalogues.
- **D.** SPF remains a strong and stable deterministic baseline.
- **E.** SPF+CD tie-break provides little or no practical benefit when costs rarely tie.
- **F.** Conflict-graph structure may be descriptively associated with sensitivity, but degree alone does not reliably exploit it.
- **G.** Held-out catalogue prevents interpreting primary improvements/regressions as universally generalizing.

## 11. Limitations

1. One MovingAI map/source (AR0204SR).
2. Held-out validation is scenario-level, not cross-map.
3. Only 27 primary + 27 held-out instances.
4. Interaction strata derived from the same independent-path conflict structure used by CDF.
5. Static pre-PP ordering only; no online/dynamic priority adaptation.
6. Degree is a coarse unweighted graph signal.
7. No event multiplicity / conflict type / timing weighting in CDF v1.
8. PP is incomplete and priority-sensitive by nature.
9. Runtime measured from independent runs, not isolated microbenchmarks.
10. Basic CBS cited elsewhere as quality reference, not quality ceiling.

## 12. Final MAPF-7 conclusion

Within this experimental setting on AR0204SR, conflict-degree-based priority ordering produced sparse and catalogue-dependent SoC effects. SPF remained competitive; held-out scenario-level validation did not support treating primary-catalogue observations as universally generalizing. The frozen MAPF-7 family is best reported as a completed negative/mixed methodological result with explicit ablations, not as a reliable improvement over SPF.
