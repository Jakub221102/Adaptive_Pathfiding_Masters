# MAPF-9 Close

## Status

**MAPF-9 CLOSED**

## Frozen primary evidence

- Fresh primary set: **54** bg512 instances (AR0400SR seed 2030, AR0307SR seed 2031).
- Success: SPF/CGLPS/UBLS **54/54**.
- CGLPS vs SPF: SoC **2/54**, makespan-only **3/54**, lex **5/54**.
- CGLPS vs UBLS (matched budget): lex **4/50/0**.
- Sum A_i = **75**; physical PP = **204**; logical PP candidates SPF/CGLPS/UBLS = **54/129/129**.

## Final interpretation

- Conflict-guided bounded search can occasionally find better priority orders than SPF.
- Advantage over matched UBLS was small and mostly on AR0307SR.
- Quality gains were sparse relative to ≈1.94× logical runtime overhead.
- No success recovery was observed because SPF already solved all primary instances.

## Frozen artifacts

- `pathfinding/docs/mapf9_design_freeze.md`
- `pathfinding/results/mapf_benchmarks/*_mapf9_manifest.json`
- `pathfinding/results/mapf9_primary_execution/`
- `pathfinding/results/mapf9_primary_analysis/` (MAPF-9.6a authoritative analysis)
- `pathfinding/results/mapf9_final_analysis/` (MAPF-9.7 thesis presentation)

## No further tuning

B = 4, CGLPS, UBLS, fresh catalogues, and primary production results are **frozen**.
Any future work on alternative budgets, neighbourhoods, or new execution is **outside MAPF-9**.
