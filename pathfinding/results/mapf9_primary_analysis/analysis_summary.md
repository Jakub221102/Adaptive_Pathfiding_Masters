# MAPF-9.6 Primary Analysis Summary

## Validation

Validation passed.

## Primary dataset

- Instances: **54**
- SPF success: **54**
- CGLPS success: **54**
- UBLS success: **54**

## Key outcomes (pooled, recomputed)

- CGLPS vs SPF SoC improved: **2/54**; makespan-only: **3**; lex improved: **5/54**
- UBLS vs SPF SoC improved: **1/54**; makespan-only: **0**; lex improved: **1/54**

### CGLPS vs UBLS (pooled)

- SoC directional: CGLPS better **1**, equal **53**, UBLS better **0**
- Lex directional: CGLPS better **4**, equal **50**, UBLS better **0**

### Unique matched-budget benefit vs SPF (pooled)

- CGLPS unique SoC: **1**; unique lex: **4**
- UBLS unique SoC: **0**; unique lex: **0**

### Per-map CGLPS SoC improvements

- AR0400SR: **1/27** (makespan-only **0**)
- AR0307SR: **1/27** (makespan-only **3**)

### Candidate selection distribution (pooled)

- CGLPS: {0: 49, 1: 4, 2: 1} (non-baseline selected **5/54**)
- UBLS: {0: 53, 1: 1} (non-baseline selected **1/54**)

- Improved-instance audit rows: **6**

## Consistency

All internal consistency invariants passed.

## Interpretation guardrails

- Descriptive counts only; no inferential significance claims.
- Historical MAPF-7/8 (108 instances) remain secondary context only.
- B=4 and transposition neighbourhood remain frozen.
