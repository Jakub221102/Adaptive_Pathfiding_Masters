# MAPF-9 Final Interpretation

## Purpose

MAPF-7 showed that static conflict-degree ordering direction was unstable across
priority permutations. MAPF-8 generalized this within bg512: priority order changes
were frequent (36/54 and 34/54 cross-map for CDF-H and CDF-L), while SoC changes
remained sparse (2/54 and 3/54). MAPF-9 therefore tested whether conflict information
is more useful as a **search-guidance signal** for bounded local exploration around
SPF than as a static ordering rule.

## Primary findings

On the fresh 54-instance primary set:

- All methods succeeded on **54/54** instances (SPF, CGLPS, UBLS).
- **CGLPS vs SPF:** SoC improved **2/54**; total SoC gain **3**; makespan-only **3**; lexicographic better **5**.
- **UBLS vs SPF:** SoC improved **1/54**; total SoC gain **1**; makespan-only **0**.
- **CGLPS vs UBLS (matched A_i):** SoC **1/53/0**; lex **4/50/0**.

## Interpretation

Conflict guidance produced a **small observed advantage** over the deterministic
unguided bounded control (UBLS), but improvements remained **sparse**. One CGLPS SoC
improvement (`AR0400SR_n05_high_000`) was **shared** with UBLS; one SoC improvement
(`AR0307SR_n20_high_000`) was **unique** to CGLPS. All **4** CGLPS-vs-UBLS
lexicographic wins occurred on AR0307SR. The result demonstrates **possibility**, not
universal superiority.

## Conflict structure

All observed improvements occurred among **search-active** instances (A_i > 0;
**36/54** search-active overall). LOW-interaction instances had A_i = 0
by construction and therefore identical SPF/CGLPS/UBLS outcomes. Some improved cases
involved many conflict events; many high-conflict cases did not improve. Conflict
magnitude alone did not clearly distinguish improved from non-improved instances
in the observed sample. No causal claim is made.

## Cost

CGLPS used **129** logical PP candidates vs **54** for SPF; pooled logical runtime
ratio ≈ **1.94×** SPF for total SoC
gain **3** over 54 instances. Cost-benefit under
the tested B = 4 transposition neighbourhood is **weak**.

This does **not** imply CGLPS is useless. It shows that conflict-guided local
exploration **can** expose beneficial order changes missed by SPF, but a fixed
B = 4 neighbourhood spends substantial computation on instances where no benefit
is obtained.

## Scope limitations

- bg512 arena family only; two fresh maps (AR0400SR, AR0307SR).
- 54 primary instances; B = 4 frozen, not optimized.
- Single-transposition neighbourhood only.
- Deterministic UBLS control with one frozen seed convention.
- No exhaustive priority permutations; no global optimality claim.
- Descriptive counts only; no inferential significance claim.
- Historical MAPF-7/8 (108 instances) used as context only, not combined denominators.

## Overall conclusion

MAPF-9 provides evidence that conflict information can be useful for directing a
bounded local search over priority orders, but within the tested bg512 setting the
benefit was sparse and modest relative to its computational cost. The result
therefore supports conflict guidance as an **informative search signal** rather
than as a generally superior static prioritization rule.
