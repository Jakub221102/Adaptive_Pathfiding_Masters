# MAPF-8.5 Cross-Map Analysis Summary

## 1. Dataset validation

- Validation status: **PASSED**
- AR0400SR: expected 108 records, found 108
- AR0400SR: no duplicate (map_name, instance_id, strategy)
- AR0400SR: exactly 27 unique instance IDs
- AR0400SR: instance IDs match manifest
- AR0400SR: strategy set exactly SPF/CDF-H/CDF-L/SPF+CD
- AR0400SR strategy spf: expected 27, found 27
- AR0400SR strategy cdf_h: expected 27, found 27
- AR0400SR strategy cdf_l: expected 27, found 27
- AR0400SR strategy spf_cd: expected 27, found 27
- AR0400SR AR0400SR_n05_high_000: exactly four strategy records
- AR0400SR AR0400SR_n05_low_000: exactly four strategy records
- AR0400SR AR0400SR_n20_medium_002: exactly four strategy records
- AR0400SR AR0400SR_n20_high_002: exactly four strategy records
- AR0400SR AR0400SR_n10_medium_000: exactly four strategy records
- AR0400SR AR0400SR_n05_low_002: exactly four strategy records
- AR0400SR AR0400SR_n20_high_001: exactly four strategy records
- AR0400SR AR0400SR_n20_low_000: exactly four strategy records
- AR0400SR AR0400SR_n05_high_001: exactly four strategy records
- AR0400SR AR0400SR_n10_high_000: exactly four strategy records
- AR0400SR AR0400SR_n10_low_001: exactly four strategy records
- AR0400SR AR0400SR_n05_medium_000: exactly four strategy records
- AR0400SR AR0400SR_n05_low_001: exactly four strategy records
- AR0400SR AR0400SR_n10_high_001: exactly four strategy records
- AR0400SR AR0400SR_n10_low_000: exactly four strategy records
- AR0400SR AR0400SR_n20_high_000: exactly four strategy records
- AR0400SR AR0400SR_n20_medium_000: exactly four strategy records
- AR0400SR AR0400SR_n20_medium_001: exactly four strategy records
- AR0400SR AR0400SR_n10_high_002: exactly four strategy records
- AR0400SR AR0400SR_n20_low_001: exactly four strategy records
- AR0400SR AR0400SR_n10_low_002: exactly four strategy records
- AR0400SR AR0400SR_n10_medium_001: exactly four strategy records
- AR0400SR AR0400SR_n05_medium_001: exactly four strategy records
- AR0400SR AR0400SR_n20_low_002: exactly four strategy records
- AR0400SR AR0400SR_n05_medium_002: exactly four strategy records
- AR0400SR AR0400SR_n05_high_002: exactly four strategy records
- AR0400SR AR0400SR_n10_medium_002: exactly four strategy records
- AR0400SR: success=108, failure=0
- AR0307SR: expected 108 records, found 108
- AR0307SR: no duplicate (map_name, instance_id, strategy)
- AR0307SR: exactly 27 unique instance IDs
- AR0307SR: instance IDs match manifest
- AR0307SR: strategy set exactly SPF/CDF-H/CDF-L/SPF+CD
- AR0307SR strategy spf: expected 27, found 27
- AR0307SR strategy cdf_h: expected 27, found 27
- AR0307SR strategy cdf_l: expected 27, found 27
- AR0307SR strategy spf_cd: expected 27, found 27
- AR0307SR AR0307SR_n10_high_001: exactly four strategy records
- AR0307SR AR0307SR_n05_high_001: exactly four strategy records
- AR0307SR AR0307SR_n20_medium_002: exactly four strategy records
- AR0307SR AR0307SR_n20_medium_000: exactly four strategy records
- AR0307SR AR0307SR_n05_high_002: exactly four strategy records
- AR0307SR AR0307SR_n05_low_002: exactly four strategy records
- AR0307SR AR0307SR_n05_high_000: exactly four strategy records
- AR0307SR AR0307SR_n20_low_002: exactly four strategy records
- AR0307SR AR0307SR_n20_medium_001: exactly four strategy records
- AR0307SR AR0307SR_n10_low_000: exactly four strategy records
- AR0307SR AR0307SR_n20_high_001: exactly four strategy records
- AR0307SR AR0307SR_n05_medium_000: exactly four strategy records
- AR0307SR AR0307SR_n20_high_002: exactly four strategy records
- AR0307SR AR0307SR_n10_high_000: exactly four strategy records
- AR0307SR AR0307SR_n10_low_001: exactly four strategy records
- AR0307SR AR0307SR_n05_medium_002: exactly four strategy records
- AR0307SR AR0307SR_n10_medium_000: exactly four strategy records
- AR0307SR AR0307SR_n05_low_001: exactly four strategy records
- AR0307SR AR0307SR_n20_high_000: exactly four strategy records
- AR0307SR AR0307SR_n10_medium_001: exactly four strategy records
- AR0307SR AR0307SR_n05_low_000: exactly four strategy records
- AR0307SR AR0307SR_n20_low_000: exactly four strategy records
- AR0307SR AR0307SR_n10_low_002: exactly four strategy records
- AR0307SR AR0307SR_n10_medium_002: exactly four strategy records
- AR0307SR AR0307SR_n20_low_001: exactly four strategy records
- AR0307SR AR0307SR_n05_medium_001: exactly four strategy records
- AR0307SR AR0307SR_n10_high_002: exactly four strategy records
- AR0307SR: success=108, failure=0
- Combined MAPF-8: expected 216 records, found 216
- Combined MAPF-8: expected 54 unique instances, found 54

## 2. AR0400SR results

- Instances: 27
- SoC-sensitive: 2 (7.4%)
- Makespan-sensitive: 1 (3.7%)
- Max/mean/median SoC range: 396 / 14.70 / 0.00

## 3. AR0307SR results

- Instances: 27
- SoC-sensitive: 2 (7.4%)
- Makespan-sensitive: 1 (3.7%)
- Max/mean/median SoC range: 22 / 0.85 / 0.00

## 4. Pooled cross-map results

- Instances: 54
- SoC-sensitive: 4 (7.4%)
- Makespan-sensitive: 2 (3.7%)

## 5. Historical AR0204SR comparison

- AR0204SR_primary: SoC-sensitive=5/27 (18.5%), makespan-sensitive=1/27, max SoC range=415.
- AR0204SR_held_out: SoC-sensitive=3/27 (11.1%), makespan-sensitive=0/27, max SoC range=264.
- AR0400SR: SoC-sensitive=2/27 (7.4%), makespan-sensitive=1/27, max SoC range=396.
- AR0307SR: SoC-sensitive=2/27 (7.4%), makespan-sensitive=1/27, max SoC range=22.

## 6. SPF stability

- cdf_h_vs_spf: LEFT better=0, equal=52, RIGHT better=2, mean diff=0.26.
- cdf_l_vs_spf: LEFT better=1, equal=51, RIGHT better=2, mean diff=7.72.
- spf_cd_vs_spf: LEFT better=0, equal=54, RIGHT better=0, mean diff=0.00.

## 7. CDF-H vs CDF-L directionality

- Pooled: LEFT better=2, equal=50, RIGHT better=2, mean diff=-7.46.

## 8. SPF+CD ablation result

- Cross-map same order: 54/54.
- Historical AR0204SR same order: 54/54.
- Combined four catalogues: 108/108.
- No different SPF+CD ordering was observed in any of the 108 tested instances across the four catalogues (54 historical AR0204SR instances and 54 new cross-map instances).
- Limitation: observational only; this does not establish mathematical equivalence between SPF and SPF+CD.

## 9. SoC vs makespan sensitivity

- Pooled SoC-sensitive: 4/54.
- Pooled makespan-sensitive: 2/54.

## 10. Conflict-structure diagnostics

- soc_sensitive (n=4): mean conflicts=19.25, mean pairs=6.50, mean max degree=2.50.
- soc_insensitive (n=50): mean conflicts=6.58, mean pairs=1.22, mean max degree=0.86.

## 11. Answers to RQ8.1–RQ8.6

See `research_questions.md` for explicit RQ answers with evidence and limitations.

## 12. Methodological limitations

- AR0204SR, AR0400SR, and AR0307SR belong to the MovingAI bg512 arena family.
- MAPF-8 demonstrates cross-map validation within bg512, not universal MovingAI generalization or cross-domain generalization.
- Primary sensitivity uses all four strategies (SPF, CDF-H, CDF-L, SPF+CD).
- MAPF-7 three-strategy diagnostic is reported separately as secondary.
- Runtime differences are descriptive; conflict-aware strategies recompute independent paths during ordering.

## 13. Conservative thesis interpretation

### Wnioski (wersja robocza, PL)

W ramach zamrożonych katalogów bg512 (AR0400SR, AR0307SR) oraz historycznych katalogów AR0204SR zaobserwowano rzadką wrażliwość jakościową (SoC) na wybór strategii priorytetu: 4 z 54 nowych instancji cross-map oraz wartości historyczne 5/27 (primary) i 3/27 (held-out) przy definicji czterostrategicznej. Nie stwierdzono spójnej przewagi CDF-H ani CDF-L między mapami; SPF pozostał często baseline'em o równym SoC. W żadnej ze 108 analizowanych instancji (54 historycznych AR0204SR oraz 54 nowych instancjach cross-map) nie zaobserwowano innego porządku priorytetów SPF+CD względem SPF; obserwacja ta nie stanowi dowodu formalnej równoważności obu strategii. Wyniki należy interpretować jako walidację topologii w rodzinie bg512, a nie generalizację na wszystkie mapy MovingAI.

### Sensitive instances (pooled cross-map)

- AR0307SR / AR0307SR_n20_high_002 (n=20, high): SPF=4955, CDF-H=4968, CDF-L=4977, SPF+CD=4955, range=22.
- AR0307SR / AR0307SR_n20_high_000 (n=20, high): SPF=5565, CDF-H=5566, CDF-L=5565, SPF+CD=5565, range=1.
- AR0400SR / AR0400SR_n20_high_001 (n=20, high): SPF=6315, CDF-H=6315, CDF-L=6711, SPF+CD=6315, range=396.
- AR0400SR / AR0400SR_n05_high_000 (n=5, high): SPF=1742, CDF-H=1742, CDF-L=1741, SPF+CD=1742, range=1.
