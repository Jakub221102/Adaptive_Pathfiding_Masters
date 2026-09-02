# MAPF-9.6 Research Questions

## RQ9.1 — Primary efficacy

On the fresh 54-instance bg512 evaluation set, can CGLPS improve solution quality or recover successful solutions relative to SPF?

- SPF failures: **0/54**
- Recovered by CGLPS: **0**
- Recovered by UBLS: **0**
- Common-success SoC improvements vs SPF:
  - CGLPS: **2/54** improved, **52** equal, **0** structurally worse; total SoC gain **3**
  - UBLS: **1/54** improved, **53** equal, **0** structurally worse; total SoC gain **1**
- Makespan-only improvements vs SPF (SoC equal): CGLPS **3**, UBLS **0**
- Lexicographic (success → SoC → makespan) vs SPF: CGLPS better **5**, equal **49**, SPF better **0**; UBLS better **1**, equal **53**, SPF better **0**

**Conservative answer:** CGLPS showed **sparse** observable quality changes relative to SPF on this closed 54-instance primary set. Success recovery was **not observed** because SPF succeeded on all validated instances.

## RQ9.2 — Guidance value

Under the same actual additional PP-evaluation budget per instance, does conflict-guided CGLPS outperform unguided UBLS?

- SoC-only (CGLPS_soc − UBLS_soc): CGLPS better **1**, equal **53**, UBLS better **0**
- Mean paired SoC diff (CGLPS − UBLS): **-0.037037037037037035**
- Lexicographic CGLPS vs UBLS: CGLPS better **4**, equal **50**, UBLS better **0**

**Conservative answer:** Under matched A_i, conflict-guided CGLPS showed a **small observed matched-budget advantage** over UBLS on this primary sample (1/54 SoC-only, 4/54 lexicographic). The effect remained sparse and modest relative to runtime overhead.

## RQ9.3 — Conditions

Are observed improvements descriptively associated with conflict structure, interaction stratum, or agent count?

- Search-active subset (A_i > 0): **36/54** instances
- Active-subset CGLPS SoC improvements: **2/36**
- Active-subset UBLS SoC improvements: **1/36**

**Conservative answer:** Descriptive summaries are reported in `conflict_structure_summary.csv`. LOW instances with zero conflicting pairs have A_i = 0 by design; identical SPF/CGLPS/UBLS outcomes there do **not** indicate failed local search. No causal or predictive claim is made.

## RQ9.4 — Cost-benefit

What evaluation/runtime overhead was required and how large were the observed quality gains?

- Logical PP candidates: SPF **54**, CGLPS **129**, UBLS **129**
- Physical PP evaluations (combined experiment): **204**
- Total logical runtime (ms): SPF **5985447.1**, CGLPS **11639447.0**, UBLS **11574579.0**
- Physical wall-clock (run.log): AR0400SR **7848.5s**, AR0307SR **9387.8s**, combined **17236.3s**

**Conservative answer:** Additional bounded search added substantial logical and physical runtime relative to sparse SoC gains. See `cost_benefit_summary.csv` for compact ratios.
