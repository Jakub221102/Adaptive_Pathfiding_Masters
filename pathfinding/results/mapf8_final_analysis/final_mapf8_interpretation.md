# MAPF-8 Final Interpretation

## 1. Main result

Priority sensitivity remained sparse across all four tested catalogues. Across all **108** MAPF instances, **12/108** were SoC-sensitive under the four-strategy definition, while **3/108** were makespan-sensitive. Sensitivity therefore affects a small minority of instances, but can be large when it occurs.

## 2. Cross-map validation

On the two new bg512 maps, sensitivity was **2/27 (7.4%)** for both AR0400SR and AR0307SR. Together with historical AR0204SR results (**5/27** primary, **3/27** held-out), this supports **persistence of sparse sensitivity across topologies within the bg512 family**. It does **not** support universal MovingAI generalization.

## 3. Magnitude

Sensitivity magnitude is strongly instance-dependent. Max observed SoC ranges include **415** (AR0204SR primary), **264** (held-out), **396** (AR0400SR), and **22** (AR0307SR). Equal catalogue-level sensitivity rates can therefore mask very different effect sizes.

## 4. SPF baseline

SPF remained a stable practical baseline in the tested catalogues: most pairwise comparisons against CDF variants are equal on SoC. This should not be read as “SPF is universally optimal”, only that it frequently matched the best observed cost in this sample.

## 5. Conflict-degree ordering

CDF-H and CDF-L did not show a consistent cross-map SoC advantage. On cross-map instances, priority order often differed from SPF (**36/54** and **34/54**), while SoC differed only **2/54** and **3/54** respectively. Ordering changes are common; quality changes are sparse.

## 6. SPF+CD

No different SPF+CD ordering was observed in **108/108** tested instances (54 historical + 54 cross-map). This is **observational only** and does not establish mathematical equivalence between SPF and SPF+CD.

## 7. Conflict structure

On pooled cross-map data, SoC-sensitive instances (n = 4) showed higher mean conflict-event, pair, and degree summaries than insensitive instances (n = 50). This is a **descriptive association only**; MAPF-8 does not support causal or predictive claims from conflict degree alone.

## 8. Limitation

All maps belong to the MovingAI **bg512 arena family**. MAPF-8 demonstrates **topology-level cross-map validation within bg512**, not universal MovingAI validation or cross-domain validation.
