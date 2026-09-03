# Single-Agent and Dynamic Benchmark — Thesis Summary (THESIS-1a)

**Purpose:** Authoritative thesis-facing aggregation of frozen single-agent static, HPA* parameter, and dynamic replanning benchmark CSVs.

**Scope:** Deterministic parsing of existing artifacts only. No planner or benchmark execution.

**Primary static CSV:** `Results/static_algorithms/astar_hpa_jps_cluster_32_scenarios_100.csv`  
**Primary dynamic CSV:** `Results/dynamic_algorithms/astar_vs_dstar_dynamic_seed_42.csv`

**Generated:** 2026-09-03

---

## 1. Static benchmark design

| Property | Value | Source |
|---|---|---|
| Map | `AR0204SR.map` (BG512 / MovingAI) | `pathfinding/scripts/benchmark_main.py` |
| Scenario file | `AR0204SR.map.scen` | same |
| Scenario selection | Filter `optimal_length > min_optimal_length` (100); take first **100** | `base_experiment.py`, `benchmark_main.py` |
| Algorithms | A*, HPA*, JPS | `benchmark_main.py` |
| HPA* parameters (primary) | `cluster_size=32`, `max_entrances_per_cluster_pair=2` | `benchmark_main.py` |
| CSV rows | **300** (3 × 100) | measured |
| Scenarios per algorithm | **100** | measured |
| Movement model (`G_8`) | **8-connected**; orthogonal step cost **1**; diagonal step cost **√2**; diagonal allowed iff destination cell is in-bounds and walkable (no cardinal-adjacency check) | `astar.py` `_get_neighbors`, `_movement_cost` |
| Heuristic (A*, JPS) | Octile: `max(dx,dy) + (√2−1)·min(dx,dy)` | `astar.py` `_heuristic` |
| Path cost convention | Sum of edge costs along returned path (float) | `PathfindingResult.path_cost` |
| MovingAI reference | `optimal_length` column from scenario file | CSV column |
| Cost error (derived) | `path_cost − optimal_length` per found run | `plot_static_algorithms_results.py` `build_summary`; `benchmark_experiment.py` print summary |

**Not primary:** `astar_hpa_jps_cluster_16_scenarios_100.csv` (alternate HPA cluster size).

---

## 2. Static benchmark final table

Aggregated over **found** runs only (`found=True`). All three algorithms: **100/100 found**.

| Algorithm | Found | Mean exec (ms) | Median exec (ms) | Mean path length | Mean path cost | Mean cost error | Median cost error | Mean visited | Other measured |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **A\*** | 100/100 | 8.978 | 6.512 | 101.39 | 119.599 | −0.328 | ≈0 | 936.36 | — |
| **HPA\*** | 100/100 | 8.887 | 6.157 | 110.34 | 128.553 | +8.626 | +8.284 | 15.97 | preproc **818.646 ms** (once); mean query **8.860 ms**; mean abstract nodes **15.97**; mean cache hit ratio **0.00902**; mean cache hits **0.09**; mean cache misses **8.19** |
| **JPS** | 100/100 | 18.326 | 16.569 | 101.39 | 119.599 | −0.328 | ≈0 | 18.81 | mean scanned nodes **6223.96**; median scanned **5650.5** |

**Derived cross-checks:**

- **Observed result:** JPS returned identical `path_cost` to A* on all 100 scenarios (max |Δ| < 1e−13) under the shared movement-cost model.
- **MovingAI reference mismatch (descriptive):** A* path cost **≤ `optimal_length` in 44/100** scenarios and **> `optimal_length` in 56/100** — the implementation octile model does not guarantee equality with the MovingAI reference field. A* optimality under an admissible/consistent heuristic belongs in the theory chapter, not in this empirical summary.

**Column legend:** execution_time_ms, path_length, path_cost, visited_nodes, scanned_nodes = **measured CSV**; cost error = **derived**; HPA preprocessing/query/cache = **measured CSV**.

---

## 3. HPA* parameter studies

Design: same map/scenario filter (100 scenarios, `min_optimal_length=100`), HPA* only.  
Sources: `Results/static_algorithms/cluster_size/`, `max_entrances/`.

### 3.1 Cluster-size sweep (`max_entrances=2` default)

| Cluster size | Found | Found rate | Mean query/exec (ms) | Mean path cost | Mean cost error | Mean visited | Preproc (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 100 | 100% | 5.975 | 127.635 | 7.708 | 80.95 | 638.320 |
| 16 | 100 | 100% | 2.485 | 127.832 | 7.905 | 33.44 | 599.762 |
| 32 | 100 | 100% | 8.850 | 128.553 | 8.626 | 15.97 | 816.428 |
| 64 | 99 | 99% | 28.336 | 127.114 | 7.227 | 8.15 | 679.729 |
| 128 | 89 | **89%** | 77.074 | 124.511 | 4.922 | 4.07 | 841.176 |

### 3.2 Max-entrances sweep (`cluster_size=128`)

| Max entrances | Found | Found rate | Mean query (ms) | Mean path cost | Mean cost error | Mean visited | Preproc (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 78 | 78% | 34.740 | 125.283 | 4.992 | 3.51 | 307.665 |
| 2 | 89 | 89% | 75.449 | 124.511 | 4.922 | 4.07 | 831.398 |
| 4 | 100 | 100% | 99.549 | 124.757 | 4.830 | 4.24 | 1090.101 |
| 8 | 100 | 100% | 99.430 | 124.757 | 4.830 | 4.24 | 1109.438 |
| 16 | 100 | 100% | 99.362 | 124.757 | 4.830 | 4.24 | 1131.716 |

### 3.3 Supported conclusions (retrospective)

| Observation | Label |
|---|---|
| Coarser clusters (128) reduce abstract search work (fewer visited abstract nodes) but **reduce found rate** to 89% on this map/sample | **DESCRIPTIVE / LIMITED CLAIM** |
| Very fine clusters (8–16) increase visited/refinement work; cluster 16 has lowest mean query time among **100% found** configs | **DESCRIPTIVE / LIMITED CLAIM** |
| At cluster 128, increasing entrances from 1→4 restores **100% found**; query time rises but remains descriptive trade-off | **DESCRIPTIVE / LIMITED CLAIM** |
| HPA* path costs exceed A*/JPS on the primary benchmark (mean error +8.63 vs ≈0) — hierarchical abstraction is **not cost-optimal** here | **SAFE PRIMARY CLAIM** (on this CSV) |

**Strongest thesis artifacts:**

| Study | Recommended table/plot | Status |
|---|---|---|
| Cluster size | `pathfinding/plots/plot_cluster_size_results.py` → `Results/plots/static_algorithms/astar_hpa_jps/cluster_size/cluster_size_vs_query_time.png` | **MAIN TEXT** (optional HPA section) |
| Max entrances | `plot_max_entrances_results.py` → `max_entrances_vs_found_rate.png` | **MAIN TEXT** if HPA parameter section included |

---

## 4. Dynamic benchmark design

| Property | Value | Source |
|---|---|---|
| Map / scen | `AR0204SR.map` / `AR0204SR.map.scen` | `dynamic_benchmark_main.py` |
| Scenarios | **50** (first 50 after filter) | script + CSV |
| CSV rows | **100** (50 × 2 algorithms) | measured |
| Obstacle base seed | **42** | `DynamicBenchmarkConfig.obstacle_seed` |
| Per-scenario seed | `42 + scenario_index` (column `seed`) | CSV + `DYNAMIC_BENCHMARK.md` |
| Obstacle count | 4 per scenario | config |
| Obstacle size | 4–10 cells | config |
| Min obstacle spacing | 25 (Chebyshev) | config |
| Placement | Near reference A* path (margin 30) | `DYNAMIC_BENCHMARK.md` |
| Algorithms | A* full replan vs D* Lite incremental | `dynamic_benchmark_experiment.py` |
| Replan trigger | `path_block_lookahead=20` | CSV column + config |
| Waiting policy | `wait_when_no_path=True`, `max_wait_steps=30` | config |
| Max simulation steps | 1500 | config |
| Obstacle prediction during planning | 0 (disabled) | CSV `prediction_steps=0` |
| Collision policy | `PUSH_AGENT` (default) | `DYNAMIC_BENCHMARK.md` |
| Movement / path cost | **8-connected octile** via `AStar._calculate_path_cost` on travelled path | `dynamic_simulation.py` |
| Success (primary) | `final_goal_reached=True` | CSV |
| Algorithm time metric | `total_execution_time_ms` (sum of planning/replan calls) | CSV |
| Wall-clock metric | `wall_clock_s` (full scenario run) | CSV |

---

## 5. Dynamic benchmark final table

| Metric | A* | D* Lite | Source |
|---|---:|---:|---|
| Runs | 50 | 50 | measured |
| `final_goal_reached` | **50/50 (100%)** | **48/50 (96%)** | measured |
| `initial_path_found` | 50/50 | 50/50 | measured |
| Dynamic interaction rate¹ | 35/50 (**70%**) | 34/50 (**68%**) | derived |
| Mean `total_execution_time_ms` | 339.745 | 219.183 | measured |
| Median `total_execution_time_ms` | 18.331 | 114.593 | measured |
| p95 `total_execution_time_ms` | 112.516 | 603.761 | derived |
| Max `total_execution_time_ms` | 15558.320 | 2440.882 | measured |
| Mean `wall_clock_s` | 0.416 | 0.945 | measured |
| Median `wall_clock_s` | 0.080 | 0.754 | measured |
| p95 `wall_clock_s` | 0.205 | 1.518 | derived |
| Max `wall_clock_s` | 15.824 | 4.050 | measured |
| Mean `replanning_count` | 7.06 | 8.16 | measured |
| Mean `waiting_steps` | 0.54 | 4.14 | measured |
| Mean `collision_count` | 0.78 | 0.94 | measured |
| Mean `agent_push_count` | 0.76 | 0.94 | measured |
| Mean `travelled_steps` | 99.34 | 95.46 | measured |
| Mean `total_path_cost` | 115.146 | 110.405 | measured |

¹ Dynamic interaction = `replanning_count > 0` OR `waiting_steps > 0` OR `collision_count > 0`.

**Distribution note (algorithm time):** A* shows a strong right tail: median `total_execution_time_ms` = **18.331** vs mean **339.745** (max **15558.320**, scenario 14). D* Lite median = **114.593** vs mean **219.183** (max **2440.882**). Means are dominated by expensive replanning outliers; medians better represent typical planner-time cost on this CSV.

**Wall-clock vs planner time:** Mean `wall_clock_s` = **0.416** (A*) vs **0.945** (D* Lite) — end-to-end scenario time including simulation overhead, not comparable to `total_execution_time_ms` alone. Do not infer global speed dominance from either metric alone.

**Failed D* Lite goal completions:** 2 scenarios with `final_goal_reached=False` (both had `initial_path_found=True`).

---

## 6. Thesis-safe conclusions

| Conclusion | Label |
|---|---|
| On AR0204SR, 100 filtered scenarios, A*, HPA*, and JPS each find a path in **100/100** cases (cluster-32 primary CSV) | **SAFE PRIMARY CLAIM** |
| **Observed:** JPS returned identical `path_cost` to A* on all 100 scenarios under the shared 8-connected octile cost model | **SAFE PRIMARY CLAIM** |
| HPA* (cluster 32, 2 entrances) trades path quality for fewer abstract nodes visited; mean cost error **+8.626** vs MovingAI `optimal_length` | **SAFE PRIMARY CLAIM** |
| **Descriptive:** MovingAI `optimal_length` differs from implementation path cost in 56/100 strict exceedances (44/100 ≤ reference) — reference/model mismatch, not a proof of suboptimality | **DESCRIPTIVE / LIMITED CLAIM** |
| HPA* cluster/entrance sweeps show found-rate vs query-time trade-offs; cluster 128 without sufficient entrances drops found rate | **DESCRIPTIVE / LIMITED CLAIM** |
| Dynamic seed-42 benchmark: A* reaches goal in **50/50** scenarios; D* Lite in **48/50** | **SAFE PRIMARY CLAIM** (this CSV only) |
| **Distribution-aware:** A* median planner time **18.331 ms** vs D* Lite **114.593 ms**; A* mean **339.745 ms** vs D* Lite **219.183 ms** with A* max **15558.320 ms** (heavy right tail) | **DESCRIPTIVE / LIMITED CLAIM** |
| Mean wall-clock **0.416 s** (A*) vs **0.945 s** (D* Lite) — neither metric establishes global planner dominance | **DESCRIPTIVE / LIMITED CLAIM** |
| D* Lite shows higher mean waiting steps (4.14 vs 0.54) on this configuration | **DESCRIPTIVE / LIMITED CLAIM** |
| Results generalize beyond seed 42 / 50 scenarios / this obstacle config | **Not supported** — **DESCRIPTIVE / LIMITED CLAIM** only |

---

## 7. Recommended thesis figures

### Static (primary benchmark — cluster 32)

| Figure | Generator | Path | Status |
|---|---|---|---|
| Execution time comparison | `plot_static_algorithms_results.py` | `Results/plots/static_algorithms/astar_hpa_jps/summary/algorithms_vs_execution_time.png` | **MAIN TEXT** — regenerate with cluster_32 CSV input |
| Path cost comparison | same | `algorithms_vs_path_cost.png` | **MAIN TEXT** |
| Cost error vs MovingAI | same | `algorithms_vs_cost_error.png` | **OPTIONAL** |
| Visited nodes | same | `algorithms_vs_visited_nodes.png` | **OPTIONAL** |
| JPS scanned vs visited | same | `jps_visited_vs_scanned_nodes.png` | **APPENDIX** |
| A* vs HPA cumulative time | same | `astar_vs_hpa_cumulative_time.png` | **OPTIONAL** |

**DO NOT USE:** `plot_static_algorithms_results.py` as committed — it points to **cluster_16** CSV (`RESULTS_PATH` line 7). Change input to cluster_32 before thesis generation.

### HPA parameters

| Figure | Path | Status |
|---|---|---|
| `cluster_size_vs_query_time.png` | `Results/plots/.../cluster_size/` | **OPTIONAL** (strongest cluster sweep) |
| `max_entrances_vs_found_rate.png` | `Results/plots/.../max_entrances/` | **OPTIONAL** (strongest entrance sweep) |

### Dynamic

| Figure | Generator | Path | Status |
|---|---|---|---|
| Median execution time | `plot_dynamic_benchmark_results.py` | `Results/plots/dynamic_algorithms/algorithms_vs_median_execution_time.png` | **MAIN TEXT** |
| Execution time boxplot | same | `algorithms_vs_execution_time_boxplot.png` | **OPTIONAL** |
| Per-metric bar charts | same | `astar_vs_dstar_<metric>.png` | **APPENDIX** |

---

## 8. Mathematical / implementation notes

### Single-agent static and dynamic (`G_8`)

- **Neighbourhood:** 8-connected (cardinal + diagonal).
- **Move validity rule:** from `(r, c)` to `(r+dr, c+dc)` with `(dr, dc) ∈ {±1,0}² \ {(0,0)}`, the move is allowed iff `(r+dr, c+dc)` is in grid bounds **and** `GridMap.is_walkable(r+dr, c+dc)`. For diagonals, **no** requirement that the two cardinal neighbours `(r+dr, c)` and `(r, c+dc)` are walkable.
- **Costs:** orthogonal = 1.0; diagonal = √2 (`math.sqrt(2)`).
- **Heuristic (A*, JPS):** octile — `max(Δr, Δc) + (√2−1)·min(Δr, Δc)`.
- **JPS:** same destination-only walkability via `_can_move` / `_is_walkable_at` (`jps.py`); pruning differs, geometry matches A*.
- **Dynamic benchmark:** planning uses `AStar` / `DStarLite`; travelled-path cost via `AStar._calculate_path_cost` (`dynamic_simulation.py`). D* Lite neighbours use the same destination-only rule (`dstar_lite.py` `_neighbor_candidates`, `_is_walkable`).
- **Grid encoding:** `GridMap.cells`; walkable iff value 0.

### MAPF (not part of this summary; for cross-reference only)

- **Neighbourhood:** `G_4` — 4-connected + WAIT; unit-cost steps; Manhattan heuristic.
- **Do not** use MAPF motion semantics when interpreting single-agent CSVs.

### Agent identity (MAPF context)

- **Math:** `a_i = (s_i, g_i)` with index `i`.
- **Code:** `MAPFAgent(agent_id, start, goal)` — separate runtime ID.

### CGLPS additional candidate count (MAPF-9; cross-reference)

- **Thesis notation:** `K_x = min(B, |E_C(x)|)` with `B=4`, `E_C` simple conflict-graph edges.
- **Validated:** all 54 MAPF-9 production instances match (`instance_results.csv`).

---

## 9. Limitations

1. **Single map** (AR0204SR) for all reported benchmarks.
2. **Static:** 100 scenarios; MovingAI reference may use a different cost convention than the implementation octile model.
3. **HPA sweeps:** retrospective; cluster 128 run loses 11 scenarios — do not compare found-rate configs naïvely without noting failures.
4. **Dynamic:** **one obstacle seed (42)**; 50 scenarios; no multi-seed replication in committed CSV.
5. **Hardware / Python version** not recorded in CSV metadata.
6. **Plot scripts** may reference non-primary CSV paths — verify before figure generation.
7. **README** sample dynamic statistics are illustrative; this document supersedes them for thesis numbers.

---

*End of THESIS-1a single-agent / dynamic summary.*
