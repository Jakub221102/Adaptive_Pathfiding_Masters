# MAPF-9 Final Figure Captions

## Figure 1 — `fig01_quality_improvements_over_spf`

**English caption:** Count of instances with SoC improvement or makespan-only improvement relative to SPF for CGLPS and UBLS on the fresh 54-instance primary set (n = 54). SoC improvement requires strictly lower sum of costs; makespan-only improvement requires equal SoC and strictly lower makespan. Categories are reported separately.

**Polish caption:** Liczba instancji z poprawą SoC lub wyłączną poprawą makespan względem SPF dla CGLPS i UBLS w świeżym zbiorze 54 instancji (n = 54). Poprawa SoC oznacza ściśle niższą sumę kosztów; poprawa wyłącznie makespan — równy SoC i niższy makespan. Kategorie podano osobno.

**Interpretation caveat:** Quality changes were sparse (CGLPS: 2 SoC + 3 makespan-only; UBLS: 1 SoC). Descriptive counts only; no inferential significance claim.

---

## Figure 2 — `fig02_matched_budget_cglps_vs_ubls`

**English caption:** Matched-budget comparison of CGLPS and UBLS on n = 54 primary instances. Each instance uses the same actual additional PP-evaluation budget A_i. Stacked bars show SoC-only directional counts (CGLPS better / equal / UBLS better) and frozen lexicographic directional counts under success → SoC → makespan.

**Polish caption:** Porównanie CGLPS i UBLS przy dopasowanym budżecie na n = 54 instancjach pierwotnych. Każda instancja ma ten sam rzeczywisty budżet dodatkowych ewaluacji PP (A_i). Skumulowane słupki pokazują liczności kierunkowe dla samego SoC oraz dla zamrożonego kryterium leksykograficznego (sukces → SoC → makespan).

**Interpretation caveat:** SoC-only counts: 1/53/0. Lexicographic counts: 4/50/0. This compares methods under matched budget; it does not establish statistical significance.

---

## Figure 3 — `fig03_bounded_search_cost`

**English caption:** Total logical method runtime normalized to SPF = 1.00 on the pooled 54-instance primary set. Annotation shows logical PP candidate counts (SPF 54; CGLPS 129; UBLS 129). Values shown: CGLPS ≈ 1.94×, UBLS ≈ 1.93×.

**Polish caption:** Całkowity logiczny czas wykonania metody znormalizowany do SPF = 1,00 dla puli 54 instancji pierwotnych. Adnotacja podaje liczbę logicznych ewaluacji kandydatów PP (SPF 54; CGLPS 129; UBLS 129).

**Interpretation caveat:** Logical runtime accounts per-method preprocessing and candidate evaluation. It is distinct from the physical combined experiment wall-clock (AR0400SR + AR0307SR ≈ 17 236 s combined run.log elapsed with 204 physical PP evaluations shared across methods).
