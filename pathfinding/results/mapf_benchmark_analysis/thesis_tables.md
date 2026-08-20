# MAPF Benchmark Thesis Tables (MAPF-5C.2)

## Table A — Overall algorithm performance

| Algorithm | Observed success | Common-180s success | Timeouts | Expansion limits |
| --- | --- | --- | --- | --- |
| Fixed-Priority PP | 27/27 (100.0%) | 26/27 (96.3%) **POST-HOC** | 0 | 0 |
| Basic CBS | 22/27 (81.5%) | 22/27 (81.5%) | 4 | 1 |
| Cardinal-First CBS | 21/27 (77.8%) | 21/27 (77.8%) | 6 | 0 |

Notes: PP common-180s success is a secondary post-hoc comparison; original PP termination labels are unchanged.

## Table B — Paired solution quality

SoC difference = left-algorithm SoC − right-algorithm SoC; positive values mean the right-hand algorithm achieved lower SoC.

| Comparison | Common-success n | Mean SoC diff | Median SoC diff | Max |SoC diff| | Left better | Equal | Right better |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PP vs Basic CBS | 22 | 16.50 | 0.00 | 221 | 0 | 18 | 4 |
| PP vs Cardinal-First CBS | 21 | 6.76 | 0.00 | 138 | 0 | 18 | 3 |
| Basic CBS vs Cardinal-First CBS | 21 | 0.00 | 0.00 | 0 | 0 | 21 | 0 |

## Table C — Basic vs Cardinal computational trade-off

| Metric | Value |
| --- | ---: |
| Common-success CBS pairs (n) | 21 |
| Mean runtime ratio (Cardinal / Basic) | 1.252 |
| Median runtime ratio (Cardinal / Basic) | 1.097 |
| Cardinal faster (count) | 2 |
| Basic faster (count) | 19 |
| Cases with reduced CT expansions | 2 |
| Mean CT expansion reduction (Basic − Cardinal) | 0.62 |
| Mean additional combined low-level searches (Cardinal − Basic) | 5.43 |

Interpretation: CT expansion reduction alone does not imply runtime improvement; low-level classification/replan overhead must be considered jointly.
