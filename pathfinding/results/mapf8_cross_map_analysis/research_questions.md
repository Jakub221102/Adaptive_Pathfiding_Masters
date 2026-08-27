# MAPF-8 Research Questions (RQ8.1–RQ8.6)

## RQ8.1 — Does sparse priority sensitivity persist on different map topologies?

- AR0400SR: 2/27 (7.4%) SoC-sensitive under four-strategy range.
- AR0307SR: 2/27 (7.4%) SoC-sensitive.
- Pooled cross-map: 4/54 (7.4%).
- Historical AR0204SR primary: 5/27; held-out: 3/27.
- **Conservative conclusion:** Priority sensitivity remained sparse on both new bg512 topologies; exact sensitive-instance sets differ by map.
- **Limitation:** Only two additional maps within the same bg512 family were tested.

## RQ8.2 — Does SPF remain a stable baseline across maps?

- CDF-H vs SPF (pooled): LEFT better=0, equal=52, RIGHT better=2, mean diff=0.26.
- CDF-L vs SPF (pooled): LEFT better=1, equal=51, RIGHT better=2, mean diff=7.72.
- SPF+CD vs SPF (pooled): LEFT better=0, equal=54, RIGHT better=0, mean diff=0.00.
- **Conservative conclusion:** SPF remained a frequently equal-cost baseline; no map showed a large systematic SoC advantage for CDF variants over SPF.
- **Limitation:** Pairwise means can be outlier-driven on sparse sensitive subsets.

## RQ8.3 — Does either CDF-H or CDF-L show a consistent cross-map advantage?

- Pooled CDF-H vs CDF-L: LEFT better=2, equal=50, RIGHT better=2, mean diff=-7.46.
- AR0204SR primary mean diff=16.30; held-out mean diff=-9.93.
- **Conservative conclusion:** No consistent cross-map advantage for CDF-H or CDF-L was observed; directionality varied by catalogue and map.

## RQ8.4 — Does SPF+CD ever produce a different priority ordering from SPF?

- Cross-map (AR0400SR + AR0307SR): same order=54/54, different order=0/54.
- Historical AR0204SR (primary + held-out): same order=54/54, different order=0/54.
- Combined four catalogues: same order=108/108, different order=0/108.
- **Conservative conclusion:** No different SPF+CD ordering was observed in any of the 108 tested instances across the four catalogues (54 historical AR0204SR instances and 54 new cross-map instances).
- **Limitation:** This is observational and does not establish mathematical equivalence between SPF and SPF+CD.

## RQ8.5 — Does makespan remain less sensitive than SoC?

- Pooled: SoC-sensitive=4/54, makespan-sensitive=2/54.
- **Conservative conclusion:** Makespan sensitivity remained lower than SoC sensitivity within the tested catalogues.

## RQ8.6 — Is conflict-graph structure still descriptively associated with priority sensitivity?

- Pooled sensitive (n=4): mean conflicts=19.25, mean pairs=6.50, mean max degree=2.50.
- Pooled insensitive (n=50): mean conflicts=6.58, mean pairs=1.22, mean max degree=0.86.
- **Conservative conclusion:** Descriptive differences were observed, but MAPF-8 does not support causal or predictive claims from conflict degree alone.
