# Thesis Evidence Inventory (THESIS-1)

**Purpose:** Authoritative index of implementation sources, frozen result artifacts, safe numerical claims, figures/tables, superseded material, and chapter mapping for final master's thesis restructuring.

**Scope:** Repository analysis only. MAPF-5 through MAPF-9 are CLOSED. No new experiments.

**Generated:** 2026-09-03

---

## 1. Single-agent evidence

### 1.1 Static MovingAI benchmark (A* / HPA* / JPS)

| Item | Path / value |
|---|---|
| **Authoritative CSV** | `Results/static_algorithms/astar_hpa_jps_cluster_32_scenarios_100.csv` |
| **Alternate CSV (cluster 16)** | `Results/static_algorithms/astar_hpa_jps_cluster_16_scenarios_100.csv` |
| **Execution script** | `pathfinding/scripts/benchmark_main.py` |
| **Experiment class** | `pathfinding/src/experiments/benchmark_experiment.py` |
| **Map** | `AR0204SR.map` (BG512 / MovingAI; local `Data/bg512-map/`) |
| **Scenario file** | `AR0204SR.map.scen` |
| **Scenarios** | 100 (first 100 with `min_optimal_length ≥ 100`) |
| **Algorithms** | A*, HPA*, JPS |
| **HPA config (primary CSV)** | `cluster_size=32`, `max_entrances_per_cluster_pair=2` |
| **Key results (cluster 32, n=100 each)** | A* found 100/100; JPS found 100/100; HPA* found 100/100 |
| **Final summary/table** | `pathfinding/docs/thesis_single_agent_dynamic_summary.md` (THESIS-1a) |
| **Figures (generated, not committed)** | `Results/plots/static_algorithms/` via `pathfinding/plots/plot_static_algorithms_results.py` → `algorithms_vs_execution_time.png`, `algorithms_vs_path_cost.png`, `algorithms_vs_cost_error.png`, `algorithms_vs_visited_nodes.png`, `jps_visited_vs_scanned_nodes.png`, `astar_vs_hpa_cumulative_time.png` |
| **Safe to use?** | **Yes** for path-quality / found-rate claims on this fixed CSV |
| **Superseded?** | cluster_16 CSV is alternate parameter run, not replacement for cluster_32 primary benchmark |

**Implementation sources:** `pathfinding/src/algorithms/astar.py`, `pathfinding/src/algorithms/jps.py`, `pathfinding/src/algorithms/hpa/hpa_star.py`, `pathfinding/src/loaders/map_loader.py`, `pathfinding/src/loaders/scen_loader.py`

### 1.2 HPA* parameter experiments

| Experiment | Authoritative CSV dir | Script | Scenarios | Notes |
|---|---|---|---|---|
| Cluster size sweep | `Results/static_algorithms/cluster_size/hpa_cluster_{8,16,32,64,128}.csv` | `pathfinding/scripts/cluster_size_benchmark_main.py` | 100 each (per README) | Fixed entrances=2 |
| Max entrances sweep | `Results/static_algorithms/max_entrances/hpa_cluster_128_entrances_{1,2,4,8,16}.csv` | `pathfinding/scripts/max_entrances_benchmark_main.py` | 100 each | Fixed cluster=128 |
| **Figures (generated)** | `Results/plots/static_algorithms/` via `plot_cluster_size_results.py`, `plot_max_entrances_results.py` | | | Not committed at inventory time |

### 1.3 Dynamic obstacle / replanning benchmark (A* vs D* Lite)

| Item | Path / value |
|---|---|
| **Authoritative CSV** | `Results/dynamic_algorithms/astar_vs_dstar_dynamic_seed_42.csv` |
| **Script** | `pathfinding/scripts/dynamic_benchmark_main.py` |
| **Experiment class** | `pathfinding/src/experiments/dynamic_benchmark_experiment.py` |
| **Simulation core** | `pathfinding/src/experiments/dynamic_simulation.py` |
| **Map / scen** | `AR0204SR.map` / `AR0204SR.map.scen` |
| **Scenarios** | 50 |
| **Obstacle seed** | 42 (`obstacle_count=4`, spacing/placement per `DynamicBenchmarkConfig`) |
| **Rows** | 100 (50 × 2 algorithms) |
| **Final summary/table** | `pathfinding/docs/thesis_single_agent_dynamic_summary.md` (THESIS-1a) |
| **Documentation** | `docs/DYNAMIC_BENCHMARK.md` |
| **Figures (generated)** | `Results/plots/dynamic_algorithms/` via `pathfinding/plots/plot_dynamic_benchmark_results.py` |
| **Safe to use?** | **Yes** for this frozen CSV; README explicitly marks sample summary as illustrative, not final thesis conclusion |
| **Superseded?** | No newer dynamic CSV in repo |

### 1.4 Single-agent algorithm implementation index (Chapter 3 support)

| Concept | Source | Notes |
|---|---|---|
| Grid map | `pathfinding/src/core/models.py` (`GridMap`) | Implicit grid graph; walkable iff `cell == 0` |
| MovingAI load | `pathfinding/src/loaders/map_loader.py`, `scen_loader.py` | |
| A* | `pathfinding/src/algorithms/astar.py` | 8-connected, octile (static NPC context) |
| HPA* | `pathfinding/src/algorithms/hpa/` | |
| JPS | `pathfinding/src/algorithms/jps.py` | |
| D* Lite | `pathfinding/src/algorithms/dstar_lite.py` | |
| Dynamic map | `pathfinding/src/core/dynamic_models.py` | |

---

## 2. MAPF core implementation evidence

### 2.1 Models

| Concept | Source file | Function/class | Key invariant | Test file(s) |
|---|---|---|---|---|
| MAPFAgent | `pathfinding/src/algorithms/mapf/models.py` | `MAPFAgent` | Unique `agent_id`; start/goal positions | `test_mapf_models.py` |
| TimedState | `models.py` | `TimedState` | Non-negative row/col/timestep | `test_mapf_models.py` |
| MAPFScenario | `models.py` | `MAPFScenario` | Non-empty; unique agent IDs | `test_mapf_models.py` |
| AgentPath | `models.py` | `AgentPath` | Strictly consecutive timesteps | `test_mapf_models.py` |
| MAPFResult | `models.py` | `MAPFResult` | success + paths tuple | `test_mapf_models.py` |

### 2.2 Conflict semantics

| Concept | Source | Function/class | Key invariant | Tests |
|---|---|---|---|---|
| Vertex conflict | `conflicts.py` | `detect_conflicts` → `VertexConflict` | Same cell at same timestep | `test_mapf_conflicts.py` |
| Edge/swap conflict | `conflicts.py` | `EdgeConflict` | Swap on transition; `timestep` = arrival | `test_mapf_conflicts.py` |
| Stay-at-goal | `conflicts.py` | `_position_at` | After final state, position frozen at goal | `test_mapf_conflicts.py` |

### 2.3 Constraints

| Concept | Source | Class | Key invariant | Tests |
|---|---|---|---|---|
| VertexConstraint | `models.py` | `VertexConstraint` | Forbids `(row,col,t)` | `test_space_time_astar.py` |
| EdgeConstraint | `models.py` | `EdgeConstraint` | Arrival-timestep semantics; `timestep > 0` | `test_space_time_astar.py` |

### 2.4 Space-Time A*

| Concept | Source | API | Key invariant | Tests |
|---|---|---|---|---|
| Public API | `space_time_astar.py` | `find_path(...)` | Returns `AgentPath` or `None` | `test_space_time_astar.py` |
| WAIT | `space_time_astar.py` | `_MOVEMENT_DELTAS` includes `(0,0)` | 4-connected + wait | `test_space_time_astar.py` |
| 4-connected motion | `space_time_astar.py` | cardinal deltas only | No diagonal MAPF moves | `test_space_time_astar.py` |
| Manhattan heuristic | `space_time_astar.py` | `_manhattan_heuristic` | Admissible on grid | `test_space_time_astar.py` |
| max_timestep | `space_time_astar.py` | horizon cap | No expansion beyond `max_timestep` | `test_space_time_astar.py` |
| Goal safety | `space_time_astar.py` | `_goal_is_safe` | Goal must be safe through horizon | `test_space_time_astar.py` |
| Constraint indexing | `space_time_astar.py` | `_build_constraint_index` | Per-agent vertex/edge sets | `test_space_time_astar.py` |

### 2.5 Reservation constraints

| Concept | Source | Function | Key invariant | Tests |
|---|---|---|---|---|
| Vertex reservations | `reservations.py` | `build_reservation_constraints` | Occupancy at each planned timestep | `test_mapf_reservations.py` |
| Reverse-edge reservations | `reservations.py` | same | Blocks swap into prior agent position | `test_mapf_reservations.py` |
| Stay-at-goal padding | `reservations.py` | goal loop to `max_timestep` | Post-arrival goal cells reserved | `test_mapf_reservations.py` |

### 2.6 Prioritized Planning

| Concept | Source | Function | Key invariant | Tests |
|---|---|---|---|---|
| Fixed-order PP | `prioritized_planning.py` | `plan_prioritized` | Agents planned in `scenario.agents` order | `test_prioritized_planning.py` |
| No backtracking | `prioritized_planning.py` | single forward pass | Failure returns immediately | `test_prioritized_planning.py` |
| Failure semantics | `prioritized_planning.py` | `termination_reason="failure"` | Partial paths not returned as success | `test_prioritized_planning.py` |
| Stats API | `prioritized_planning.py` | `plan_prioritized_with_stats` | `low_level_searches`, `agents_planned` | `test_prioritized_planning.py` |

### 2.7 Metrics

| Concept | Source | Function | Key invariant | Tests |
|---|---|---|---|---|
| SoC | `metrics.py` | `sum_of_costs` | `Σ (len(path)-1)` | `test_mapf_metrics.py` |
| Makespan | `metrics.py` | `makespan` | `max (len(path)-1)` | `test_mapf_metrics.py` |

### 2.8 Priority ordering & conflict graph

| Concept | Source | Function | Key invariant | Tests |
|---|---|---|---|---|
| SPF | `priority_ordering.py` | `shortest_path_first_order` | `(cost, original_index)` ascending | `test_priority_ordering.py` |
| LPF | `priority_ordering.py` | `longest_path_first_order` | `(-cost, original_index)` | `test_priority_ordering.py` |
| CDF-H | `priority_ordering.py` | `conflict_degree_first_high_order` | `(-degree, cost, index)` | `test_conflict_aware_priority_ordering.py` |
| CDF-L | `priority_ordering.py` | `conflict_degree_first_low_order` | `(degree, cost, index)` | `test_conflict_aware_priority_ordering.py` |
| SPF+CD | `priority_ordering.py` | `shortest_path_first_conflict_degree_order` | `(cost, -degree, index)` | `test_conflict_aware_priority_ordering.py` |
| Conflict graph edges | `priority_ordering.py` | `conflict_graph_edges` | One undirected edge per conflicting pair | `test_conflict_aware_priority_ordering.py` |
| Pair event count | `priority_ordering.py` | `pair_conflict_event_counts` | Multiplicity of conflict events per pair | `test_mapf_bounded_local_priority_search.py` |

### 2.9 CGLPS / UBLS (MAPF-9)

| Concept | Source | Function/constant | Key invariant | Tests |
|---|---|---|---|---|
| CGLPS core | `mapf_bounded_local_priority_search.py` | `evaluate_cglps` | Single SPF neighbourhood; budget B=4 | `test_mapf_bounded_local_priority_search.py`, `test_mapf9_integration.py` |
| UBLS control | same | `evaluate_ubls` | Matched `K_x`; SHA-256 seed | same |
| B = 4 | same | `MAPF9_CGLPS_BUDGET = 4` | Frozen | `test_mapf9_integration.py` |
| Additional candidate count | same | `additional_pp_eval_count` | `K_x = min(B, \|E_C(x)\|)` on simple conflict graph | `mapf9_primary_execution/*/instance_results.csv` |
| Selection objective | same | `select_best_candidate_evaluation` | success → SoC → makespan → candidate_id | same |
| Design freeze doc | `pathfinding/docs/mapf9_design_freeze.md` | — | Authoritative method definition | — |

### 2.10 CBS

| Concept | Source | Function/class | Tests |
|---|---|---|---|
| Conflict splitting | `cbs_splitting.py` | `split_conflict` | `test_cbs_splitting.py` |
| CT root | `cbs.py` | `build_cbs_root` | `test_cbs_root.py` |
| CT node | `cbs.py` | `CBSNode` | `test_cbs_expansion.py` |
| Basic CBS | `cbs.py` | `solve_cbs(..., conflict_selection_mode="basic")` | `test_cbs_solver.py` |
| Cardinal-First CBS | `cbs.py` + `cbs_conflict_selection.py` | `"cardinal_first"` | `test_cbs_conflict_selection.py` |
| OPEN ordering | `cbs.py` | heap `(cost, tie_breaker, node)` | `test_cbs_solver.py` |
| Affected-agent replanning | `cbs.py` | replans only constrained agents | `test_cbs_solver.py` |
| CBS stats | `cbs.py` | `CBSStats` | `test_cbs_stats.py` |
| Duplicate/plateau diagnostics | `cbs.py` | signature counters | `test_cbs_duplicate_analysis.py` |
| Cardinal classification | `cbs_conflict_classification.py` | classify cardinal/semi/non | `test_cbs_conflict_classification.py` |
| Process wall-clock guard | `mapf_cbs_process_runner.py`, `mapf_benchmark_execution.py` | `run_cbs_with_wall_clock_limit` | `test_mapf_cbs_wall_clock_timeout.py` |

### 2.11 Benchmark / catalogue infrastructure

| Component | Source |
|---|---|
| Catalogue generation | `mapf_benchmark_catalogue_generation.py`, `mapf_benchmark_instances.py` |
| Manifest SHA verification | `mapf_cross_map_catalogue_validation.py` → `manifest_file_sha256` |
| MAPF-5 execution | `mapf_benchmark_execution.py` |
| Checkpoint / logging | `pathfinding/results/*_execution/run.log`, `*.log` |

---

## 3. MAPF-5 evidence (Chapter 6 partial — CBS baseline)

### 3.1 Authoritative artifacts

| Role | Path |
|---|---|
| Raw execution CSV | `pathfinding/results/mapf_benchmark_execution/results.csv` |
| Execution log | `pathfinding/results/mapf_benchmark_execution.log` |
| Final analysis | `pathfinding/results/mapf_benchmark_analysis/analysis_summary.md` |
| Final thesis tables | `pathfinding/results/mapf_benchmark_analysis/thesis_tables.md` |
| Manifest | `pathfinding/results/mapf_benchmarks/AR0204SR_manifest.json` |
| Manifest SHA-256 | `5446f42c2995009f1d5b33cb050011d2af076c25bc5eb2b165f9ed7ee5765e60` |
| Seed | 2026 |
| Instances | 27 on AR0204SR |

### 3.2 Safe numerical claims (source: `thesis_tables.md`, `analysis_summary.md`)

| Claim | Value | Source artifact |
|---|---|---|
| PP observed success | 27/27 | `thesis_tables.md` Table A |
| Basic CBS success | 22/27 | same |
| Cardinal-First CBS success | 21/27 | same |
| Basic vs Cardinal common-success | 21 | `thesis_tables.md` Table C |
| Identical SoC on common-success | 21/21 (mean diff 0.00) | same |
| Mean Cardinal/Basic runtime ratio | ≈ 1.252 | same |
| PP vs Basic max \|SoC diff\| | 221 | `thesis_tables.md` Table B |
| PP vs Basic mean SoC diff | 16.50 (22 paired) | same |
| n10_high_001 PP vs Basic | PP SoC 3183, Basic 2962, diff +221 | `paired_quality_comparison.csv` |
| CBS wall-clock guard | 180 s (CBS algorithms) | `mapf_benchmark_execution.py`, `test_mapf_cbs_wall_clock_timeout.py` |
| PP post-hoc 180 s | 26/27 (secondary, post-hoc) | `thesis_tables.md` Table A note |

### 3.3 MAPF-5 figures

Generated by `pathfinding/src/experiments/mapf_benchmark_plots.py` into analysis output `figures/` (not committed):

- `fig01_success_rate_scalability`
- `fig02_runtime_termination_outcomes`
- `fig03_pp_vs_basic_soc_gap`
- `fig04_cardinal_overhead_tradeoff`
- `fig05_high_interaction_outcome_matrix`

---

## 4. MAPF-6 evidence (Chapter 6)

### 4.1 Authoritative artifacts by strategy

| Strategy | Raw CSV / JSONL | Analysis |
|---|---|---|
| Fixed + CBS (from MAPF-5) | `mapf_benchmark_execution/results.csv` | `mapf_priority_strategy_analysis/` |
| Random K=10 | `mapf_random_priority_pilot/results.csv`, `results_details.jsonl` | same |
| SPF / LPF | `mapf_priority_strategy_execution/results.csv` | same |
| Random K=20 supplemental | `mapf_random_k20_supplemental/results_details.jsonl` | `mapf_random_k20_analysis/` |
| **Final tables** | | `mapf_priority_strategy_analysis/thesis_tables.md` |
| **K=20 robustness** | | `mapf_random_k20_analysis/analysis_summary.md` |

### 4.2 Verified numerical claims

| Claim | Value | Source |
|---|---|---|
| Random K=10 evaluations | 270 | `mapf_random_k20_analysis/analysis_summary.md` §1 |
| K=10 success | 270/270 | same |
| SoC-sensitive (K=10) | 7/27 | `mapf_priority_strategy_analysis/thesis_tables.md` Table C |
| Makespan-sensitive (K=10) | 1/27 | same |
| Max SoC range (K=10) | 719 | same; instance `AR0204SR_n20_high_001` |
| Supplemental K=20 records | 270 | `mapf_random_k20_analysis/analysis_summary.md` |
| Combined K=20 records | 540 | same |
| Orderings per instance (combined) | 20 (indices 0..19) | same |
| No duplicate ordering IDs | validated | same §1 |
| No new sensitive instances K=10→K=20 | 0 | same §4 |
| Max range remains | 719 | same §4 |
| Fixed vs SPF | 0/23/4 (L/E/R), mean diff +40.74 | `thesis_tables.md` Table B |
| SPF vs LPF | 5/20/2, mean diff -44.15 | same |
| Fixed vs LPF | 2/22/3, mean diff -3.41 | same |

**Wording (mandatory):** Random K=10/K=20 is **sensitivity analysis** under a finite sample of permutations — NOT exhaustive search, global oracle, global optimum, or global worst ordering.

### 4.3 MAPF-6 figures

From `mapf_priority_strategy_plots.py` (not committed):

- `fig01_priority_sensitivity_by_instance` (**strongest**)
- `fig02_spf_vs_lpf_soc_difference` (**strongest**)
- `fig03_spf_lpf_runtime_cost`
- `fig04_priority_sensitive_case_study`

---

## 5. MAPF-7 evidence (Chapter 6 — static conflict heuristics)

### 5.1 Authoritative hierarchy

| Layer | Path | Role |
|---|---|---|
| Primary execution | `mapf_conflict_aware_priority_execution/results.csv` | Raw (81 rows) |
| Held-out execution | `mapf_conflict_aware_priority_heldout_execution/results.csv` | Raw (81 rows) |
| Held-out SPF | `mapf_spf_heldout_execution/results.csv` | Raw (27 rows) |
| **Interim analysis** | `mapf_conflict_aware_priority_analysis/` | Superseded for thesis numbers |
| **FINAL analysis** | `mapf7_final_analysis/` | **Authoritative** |
| Final summary | `mapf7_final_analysis/final_analysis_summary.md` | |
| Final tables | `mapf7_final_analysis/thesis_tables.md` | |

**Held-out manifest:** `AR0204SR_heldout_manifest.json`, seed **2027**, SHA `8e2f37239ad850b002da933c064541283367ff370d851ff213c0759540a23f25`

### 5.2 Primary AR0204SR (seed 2026) — four-strategy where noted

| Comparison | L / E / R | Source |
|---|---|---|
| CDF-H vs CDF-L | 0 / 22 / 5 | `final_analysis_summary.md`, `thesis_tables.md` Table 1 |
| CDF-H vs SPF | 0 / 23 / 4 | same |
| CDF-L vs SPF | 2 / 24 / 1 | same |
| SPF+CD vs SPF | 0 / 27 / 0 | same |
| SoC-sensitive (3-strategy MAPF-7) | 5/27 | same |
| Makespan-sensitive | 1/27 | same |

### 5.3 Held-out AR0204SR (seed 2027)

| Comparison | L / E / R | Source |
|---|---|---|
| CDF-H vs CDF-L | 3 / 24 / 0 | `thesis_tables.md` Table 2 |
| CDF-H vs SPF | 1 / 26 / 0 | same |
| CDF-L vs SPF | 0 / 25 / 2 | same |
| SPF+CD vs SPF | 0 / 27 / 0 | same |
| SoC-sensitive | 3/27 | same |
| Makespan-sensitive | 0/27 | same |

### 5.4 Combined observational facts

| Claim | Value | Source |
|---|---|---|
| SPF+CD same order as SPF | 54/54 | `final_analysis_summary.md` §6 |
| SPF+CD same SoC as SPF | 54/54 | same |
| Direction CDF-H vs CDF-L | **Reversed** primary vs held-out | `final_analysis_summary.md` §5 |

**Wording:** Held-out is **same-map scenario-level validation**, NOT cross-map validation.

### 5.5 MAPF-7 figures

From `mapf7_final_analysis_plots.py` → `mapf7_final_analysis/figures/` (not committed):

| Figure | Recommendation |
|---|---|
| `fig01_cdf_h_minus_spf_by_catalogue` | **FINAL — strong** |
| `fig02_cdf_h_minus_cdf_l_by_catalogue` | **FINAL — strong** |
| `fig03_soc_on_sensitive_instances` | FINAL |
| `fig04_conflict_structure_vs_priority_effect` | VALID BUT OPTIONAL |
| `fig05_heldout_runtime_composition` | DIAGNOSTIC ONLY |
| `fig06_sensitivity_by_interaction_stratum` | VALID BUT OPTIONAL |

---

## 6. MAPF-8 evidence (Chapter 7 — cross-map within bg512)

### 6.1 Authoritative hierarchy

| Layer | Path |
|---|---|
| Cross-map analysis (audit) | `pathfinding/results/mapf8_cross_map_analysis/` |
| **Final thesis presentation** | `pathfinding/results/mapf8_final_analysis/` |
| Raw execution | `pathfinding/results/mapf8_cross_map_execution/{AR0400SR,AR0307SR}/results.csv` |

### 6.2 Catalogues & SHA

| Map | Seed | Manifest | SHA-256 |
|---|---|---|---|
| AR0400SR | 2028 | `mapf_benchmarks/AR0400SR_manifest.json` | `ab652d77ad99c251d7827684b7b2978a3bd5151609c5b557840eec8522a54259` |
| AR0307SR | 2029 | `mapf_benchmarks/AR0307SR_manifest.json` | `ff6bb64e15d8c507861966e27a00df0df1efcf7b070584b0313fc9448b4f2732` |

Validation: `mapf_benchmarks/AR0400SR_validation.md`, `AR0307SR_validation.md`

### 6.3 Pooled cross-map (54 instances) — verified

| Metric | Value | Source |
|---|---|---|
| Instances | 54 | `mapf8_cross_map_analysis/analysis_summary.md` |
| SoC-sensitive (4-strategy) | 4/54 (7.4%) | same |
| Makespan-sensitive | 2/54 (3.7%) | same |
| Max SoC range | 396 | same, `thesis_tables.md` |
| Mean SoC range | 7.78 | `mapf8_final_analysis/thesis_tables.md` |
| Median SoC range | 0 | same |
| CDF-H differs from SPF (order) | 36/54 | `final_mapf8_interpretation.md` |
| CDF-L differs from SPF (order) | 34/54 | same |
| CDF-H SoC differs from SPF | 2/54 | same |
| CDF-L SoC differs from SPF | 3/54 | same |
| SPF+CD same order as SPF | 54/54 | same |

### 6.4 Historical four-catalogue context

| Catalogue | SoC-sensitive | Makespan-sensitive | Max range | Source |
|---|---|---|---|---|
| AR0204 primary | 5/27 | 1/27 | 415 | `historical_comparison.csv` |
| AR0204 held-out | 3/27 | 0/27 | 264 | same |
| AR0400SR | 2/27 | 1/27 | 396 | same |
| AR0307SR | 2/27 | 1/27 | 22 | same |
| **Combined 108** | **12/108** | **3/108** | — | `final_mapf8_interpretation.md` |

### 6.5 Conflict structure (pooled cross-map, descriptive)

| Group | mean I(x) (conflict events) | mean \|E_C(x)\| (conflicting pairs) | mean max degree | Source |
|---|---|---|---|---|
| soc_sensitive (n=4) | 19.25 | 6.50 | 2.50 | `analysis_summary.md` §10 |
| soc_insensitive (n=50) | 6.58 | 1.22 | 0.86 | same |

### 6.6 Four FINAL MAPF-8 thesis figures

Base dir (generated by `mapf8_final_analysis.py`): `pathfinding/results/mapf8_final_analysis/figures/`

| # | Basename | PNG | PDF |
|---|---|---|---|
| 1 | `fig01_soc_sensitivity_across_catalogues` | `.../fig01_soc_sensitivity_across_catalogues.png` | `.pdf` |
| 2 | `fig02_soc_on_all_sensitive_instances` | `.../fig02_soc_on_all_sensitive_instances.png` | `.pdf` |
| 3 | `fig03_cdf_vs_spf_across_catalogues` | `.../fig03_cdf_vs_spf_across_catalogues.png` | `.pdf` |
| 4 | `fig04_order_change_vs_soc_change` | `.../fig04_order_change_vs_soc_change.png` | `.pdf` |

Captions: `mapf8_final_analysis/figure_captions.md`. **Note:** PNG/PDF not present in git at inventory time; regenerate via `pathfinding/scripts/mapf8_final_analysis_main.py`.

**Wording:** Cross-map = **topology-level validation within bg512** — NOT cross-domain / universal MovingAI generalization.

---

## 7. MAPF-9 evidence (Chapter 8 — CGLPS / UBLS)

### 7.1 Authoritative hierarchy

| Layer | Path | Role |
|---|---|---|
| Design freeze | `pathfinding/docs/mapf9_design_freeze.md` | Method definition |
| Manifests | `mapf_benchmarks/*_mapf9_manifest.json` | Frozen catalogues |
| Primary execution | `mapf9_primary_execution/{AR0400SR,AR0307SR}/` | Audit CSV/JSONL |
| Primary analysis | `mapf9_primary_analysis/` | Authoritative analysis (MAPF-9.6a) |
| **Final presentation** | `mapf9_final_analysis/` | Thesis-facing (MAPF-9.7) |
| Close record | `mapf9_final_analysis/mapf9_close.md` | Stage closure |

### 7.2 Fresh primary set

| Map | Seed | Manifest SHA-256 | Instances |
|---|---|---|---|
| AR0400SR | 2030 | `14964ee449b826c6280299f68a6cfc0cc42d34f3045d87070971027009dbb71c` | 27 |
| AR0307SR | 2031 | `a8bec75ead3a9f18a6d4e99fa05aa5a2f4291ee289b47debf5533ec06dc18403` | 27 |
| **Pooled** | — | — | **54** |

- Zero same-map signature overlap with MAPF-8: validated in `AR0400SR_mapf9_validation.md`, `AR0307SR_mapf9_validation.md`
- MAPF-9 is **CLOSED** — no further tuning

### 7.3 Method (frozen)

| Parameter | Value | Source |
|---|---|---|
| CGLPS | Conflict-guided pair-ranked transpositions of SPF order | `mapf9_design_freeze.md`, `mapf_bounded_local_priority_search.py` |
| UBLS | Unguided random pair transpositions, matched budget | same |
| SPF anchor | Independent-cost ascending | same |
| Neighbourhood | Single transposition only | same |
| Pair ranking key | `-pair_conflict_event_count`, then indices | `generate_cglps_candidate_specs` |
| B | 4 | `MAPF9_CGLPS_BUDGET` |
| K_x | `min(B, \|E_C(x)\|)` — additional CGLPS/UBLS candidate count per instance | `evaluate_cglps` / `evaluate_ubls`; raw artifact fields `actual_a_i`, `expected_a_i_from_manifest` map to thesis K_x; LOW stratum → K_x=0 |
| Selection | success → SoC → makespan → candidate_id | `_selection_sort_key` |
| UBLS seed | SHA-256(`9031:instance_id`) | `ubls_random_generator` |

### 7.4 Final outcomes (source: `mapf9_final_analysis/thesis_tables.md`, `final_mapf9_interpretation.md`)

| Claim | Value |
|---|---|
| Success all methods | 54/54 |
| CGLPS vs SPF SoC improved | 2/54; total gain 3; makespan-only 3; lex better 5 |
| UBLS vs SPF SoC improved | 1/54; makespan-only 0 |
| CGLPS vs UBLS SoC | 1 / 53 / 0 |
| CGLPS vs UBLS lex | 4 / 50 / 0 |
| Unique SoC (vs SPF) | CGLPS 1; UBLS 0 |
| Unique lex (vs SPF) | CGLPS 4; UBLS 0 |
| Search-active instances | 36/54 |
| Sum K_x (raw field `sum A_i`) | 75 |
| Physical PP evaluations | 204 |
| Logical PP candidates | SPF 54; CGLPS 129; UBLS 129 |
| Runtime ratios vs SPF | 1.000 / ~1.945 / ~1.934 |
| Combined wall-clock (run.log) | AR0400 7848.5 s + AR0307 9387.8 s ≈ 17236.3 s |

### 7.5 Five CGLPS lex-improved instances

From `thesis_tables.md` Table 4:

| Map | Instance | Effect | SPF SoC | CGLPS SoC | SPF MS | CGLPS MS |
|---|---|---|---:|---:|---:|---:|
| AR0307SR | n05_high_000 | Makespan-only | 1037 | 1037 | 357 | 356 |
| AR0307SR | n05_high_002 | Makespan-only | 1832 | 1832 | 499 | 498 |
| AR0307SR | n20_high_000 | SoC | 4417 | 4415 | 510 | 510 |
| AR0307SR | n20_medium_002 | Makespan-only | 4518 | 4518 | 433 | 431 |
| AR0400SR | n05_high_000 | SoC | 1295 | 1294 | 478 | 478 |

### 7.6 Three FINAL MAPF-9 thesis figures

Base: `pathfinding/results/mapf9_final_analysis/figures/`

| # | Basename | PNG / PDF paths |
|---|---|---|
| 1 | `fig01_quality_improvements_over_spf` | `.../fig01_quality_improvements_over_spf.{png,pdf}` |
| 2 | `fig02_matched_budget_cglps_vs_ubls` | `.../fig02_matched_budget_cglps_vs_ubls.{png,pdf}` |
| 3 | `fig03_bounded_search_cost` | `.../fig03_bounded_search_cost.{png,pdf}` |

Captions: `mapf9_final_analysis/figure_captions.md`. Regenerate: `pathfinding/scripts/mapf9_final_analysis_main.py`.

---

## 8. Experimental catalogue inventory

| Stage | Map | Seed | Agents | Strata | Inst. | Role | Manifest | SHA-256 (frozen) | Relation | Safe thesis description |
|---|---|---:|---|---|---:|---|---|---|---|---|
| MAPF-5/6 primary | AR0204SR | 2026 | 5,10,20 | LOW/MED/HIGH ×3 | 27 | Primary / development | `AR0204SR_manifest.json` | `5446f42c…765e60` | First frozen MAPF catalogue | Primary AR0204SR development set for PP/CBS/priority sensitivity |
| MAPF-7 held-out | AR0204SR | 2027 | 5,10,20 | same | 27 | Same-map held-out | `AR0204SR_heldout_manifest.json` | `8e2f3723…3f25` | Disjoint scenarios vs 2026 | Scenario-level held-out on AR0204SR; not cross-map |
| MAPF-8 cross-map | AR0400SR | 2028 | 5,10,20 | same | 27 | Cross-map bg512 | `AR0400SR_manifest.json` | `ab652d77…4259` | New map, MAPF-7/8 methods | Cross-map topology validation (CDF family) |
| MAPF-8 cross-map | AR0307SR | 2029 | 5,10,20 | same | 27 | Cross-map bg512 | `AR0307SR_manifest.json` | `ff6bb64e…2732` | New map | same |
| MAPF-9 fresh | AR0400SR | 2030 | 5,10,20 | same | 27 | Fresh algorithm eval | `AR0400SR_mapf9_manifest.json` | `14964ee4…b71c` | 0 overlap with MAPF-8 sig | Fresh CGLPS/UBLS primary set |
| MAPF-9 fresh | AR0307SR | 2031 | 5,10,20 | same | 27 | Fresh algorithm eval | `AR0307SR_mapf9_manifest.json` | `a8bec75e…8403` | 0 overlap with MAPF-8 sig | same |

**Interaction strata:** let **I(x)** = number of independent-path **conflict events** on instance x. LOW: **I(x) = 0**; MEDIUM: **1 ≤ I(x) ≤ 2**; HIGH: **I(x) ≥ 3**. (Distinct from **|E_C(x)|**, the conflicting-agent-pair count on the simple independent-path conflict graph.)

**Other shared parameters:** `max_timestep=512`, `min_reference_length=20.0`.

---

## 9. Figure inventory

**Legend — Status:** FINAL = closed-stage thesis figure; VALID BUT OPTIONAL = support/diagnostic; DIAGNOSTIC ONLY; SUPERSEDED; UNKNOWN = script exists, binary not in repo.

### Single-agent & dynamic

| Chapter | Figure | Exact path (expected) | Source data | Status | Recommendation |
|---|---|---|---|---|---|
| Ch. 5 | algorithms_vs_execution_time | `Results/plots/static_algorithms/algorithms_vs_execution_time.png` | `astar_hpa_jps_cluster_32_scenarios_100.csv` | UNKNOWN | Strong for static benchmark |
| Ch. 5 | algorithms_vs_path_cost | same dir | same CSV | UNKNOWN | Strong |
| Ch. 5 | cluster / entrances sweeps | `Results/plots/static_algorithms/max_entrances_*.png` | HPA CSV dirs | UNKNOWN | Optional parameter study |
| Ch. 5 | dynamic A* vs D* Lite | `Results/plots/dynamic_algorithms/` | `astar_vs_dstar_dynamic_seed_42.csv` | UNKNOWN | Strong for dynamic section |

### MAPF-5

| Ch. 6 | fig01_success_rate_scalability | `mapf_benchmark_analysis/figures/fig01_*.png` | `mapf_benchmark_execution/results.csv` | UNKNOWN | **Strong** |
| Ch. 6 | fig03_pp_vs_basic_soc_gap | same | same | UNKNOWN | **Strong** |
| Ch. 6 | fig04_cardinal_overhead_tradeoff | same | same | UNKNOWN | FINAL for CBS trade-off |
| Ch. 6 | fig05_high_interaction_outcome_matrix | same | same | UNKNOWN | VALID BUT OPTIONAL |

### MAPF-6

| Ch. 6 | fig01_priority_sensitivity_by_instance | `mapf_priority_strategy_analysis/figures/` | random + deterministic CSVs | UNKNOWN | **Strong** |
| Ch. 6 | fig02_spf_vs_lpf_soc_difference | same | same | UNKNOWN | **Strong** |
| Ch. 6 | fig04_priority_sensitive_case_study | same | same | UNKNOWN | FINAL case study |

### MAPF-7

| Ch. 6 | fig01_cdf_h_minus_spf_by_catalogue | `mapf7_final_analysis/figures/` | final analysis CSVs | UNKNOWN | **Strong** |
| Ch. 6 | fig02_cdf_h_minus_cdf_l_by_catalogue | same | same | UNKNOWN | **Strong** |
| Ch. 6 | fig03_soc_on_sensitive_instances | same | same | UNKNOWN | FINAL |
| Ch. 6 | fig04_conflict_structure_vs_priority_effect | same | same | UNKNOWN | OPTIONAL |

### MAPF-8 (four final)

| Ch. 7 | fig01_soc_sensitivity_across_catalogues | `mapf8_final_analysis/figures/fig01_soc_sensitivity_across_catalogues.png` | cross-map + historical tables | UNKNOWN | **Strong — mandatory set** |
| Ch. 7 | fig02_soc_on_all_sensitive_instances | `.../fig02_soc_on_all_sensitive_instances.png` | `sensitive_instances.csv`, Table D | UNKNOWN | **Strong** |
| Ch. 7 | fig03_cdf_vs_spf_across_catalogues | `.../fig03_cdf_vs_spf_across_catalogues.png` | `historical_comparison.csv` | UNKNOWN | **Strong** |
| Ch. 7 | fig04_order_change_vs_soc_change | `.../fig04_order_change_vs_soc_change.png` | pooled cross-map 54 | UNKNOWN | **Strong — key MAPF-8 message** |

### MAPF-9 (three final)

| Ch. 8 | fig01_quality_improvements_over_spf | `mapf9_final_analysis/figures/fig01_quality_improvements_over_spf.png` | `mapf9_final_analysis/thesis_tables.md` | UNKNOWN | **Strong — mandatory set** |
| Ch. 8 | fig02_matched_budget_cglps_vs_ubls | `.../fig02_matched_budget_cglps_vs_ubls.png` | same | UNKNOWN | **Strong** |
| Ch. 8 | fig03_bounded_search_cost | `.../fig03_bounded_search_cost.png` | `cost_benefit_summary.csv` | UNKNOWN | **Strong** |

---

## 10. Table inventory

| Chapter | Table / artifact | Path | Key content | Status | Recommendation |
|---|---|---|---|---|---|
| Ch. 6 MAPF-5 | Thesis tables | `mapf_benchmark_analysis/thesis_tables.md` | Success, paired SoC, CBS overhead | FINAL | Primary MAPF-5 tables |
| Ch. 6 MAPF-6 | Thesis tables | `mapf_priority_strategy_analysis/thesis_tables.md` | Fixed/Random/SPF/LPF | FINAL | Primary MAPF-6 tables |
| Ch. 6 MAPF-6 K20 | K10 vs K20 summary | `mapf_random_k20_analysis/k10_vs_k20_summary.csv` | Robustness | FINAL | Supplementary |
| Ch. 6 MAPF-7 | Final thesis tables | `mapf7_final_analysis/thesis_tables.md` | Primary/held-out/ablation | FINAL | **Replace** `mapf_conflict_aware_priority_analysis/thesis_tables.md` |
| Ch. 7 MAPF-8 cross | Thesis tables | `mapf8_cross_map_analysis/thesis_tables.md` | Per-map + pooled 54 | FINAL | Audit reference |
| Ch. 7 MAPF-8 final | Thesis tables | `mapf8_final_analysis/thesis_tables.md` | Four-catalogue + sensitive detail | FINAL | **Primary for thesis** |
| Ch. 8 MAPF-9 | Thesis tables | `mapf9_final_analysis/thesis_tables.md` | Outcomes, lex instances, cost | FINAL | **Primary for thesis** |
| Ch. 8 MAPF-9 | Primary analysis CSVs | `mapf9_primary_analysis/*.csv` | Audit trail | FINAL | Supporting evidence |
| Ch. 4 catalogues | Validation reports | `mapf_benchmarks/*_validation.md` | SHA, disjointness | FINAL | Methodology |
| Ch. 4 dynamic | Benchmark doc | `docs/DYNAMIC_BENCHMARK.md` | Metric definitions | FINAL | Methodology |

---

## 11. Implementation diagram evidence

### Recommended architecture (data / control flow)

```
MovingAI loaders (map_loader, scen_loader)
    → GridMap + MAPFScenario (core/models, mapf/models)
    → independent Space-Time A*  (space_time_astar.find_path)
    → conflict detection + conflict graph  (conflicts.detect_conflicts, priority_ordering.conflict_graph_edges)
    → ordering strategies  (priority_ordering: SPF, LPF, CDF-*, random)
    → Prioritized Planning  (prioritized_planning.plan_prioritized*)
    → metrics  (metrics.sum_of_costs, makespan)

Parallel CBS branch:
    Space-Time A*  → CBS low-level replans  (cbs.build_cbs_root, expand_cbs_node, solve_cbs)
    conflict split / cardinal classification  (cbs_splitting, cbs_conflict_classification, cbs_conflict_selection)

MAPF-9 branch (on top of SPF + PP):
    prepare_mapf9_instance  →  CGLPS / UBLS bounded transposition search  (mapf_bounded_local_priority_search)
    → benchmark execution / analysis pipelines  (mapf9_primary_execution, mapf9_*_analysis)
```

### Module map for diagram boxes

| Box | Modules |
|---|---|
| Loaders | `pathfinding/src/loaders/map_loader.py`, `scen_loader.py` |
| Grid / scenario | `pathfinding/src/core/models.py`, `pathfinding/src/algorithms/mapf/models.py` |
| Space-Time A* | `pathfinding/src/algorithms/mapf/space_time_astar.py` |
| Conflicts / graph | `pathfinding/src/algorithms/mapf/conflicts.py`, `priority_ordering.py` |
| Orderings | `pathfinding/src/algorithms/mapf/priority_ordering.py` |
| PP | `pathfinding/src/algorithms/mapf/prioritized_planning.py`, `reservations.py` |
| CBS | `pathfinding/src/algorithms/mapf/cbs.py`, `cbs_*.py` |
| CGLPS/UBLS | `pathfinding/src/experiments/mapf_bounded_local_priority_search.py` |
| Benchmark infra | `pathfinding/src/experiments/mapf_benchmark_*.py`, `mapf9_*.py`, `mapf8_*.py` |
| Tests | `pathfinding/tests/unit/test_mapf*.py`, `test_cbs*.py` |

---

## 12. Mathematical-semantics audit

### Recommended graph notation (distinct neighbourhood models)

| Context | Thesis notation | Movement |
|---|---|---|
| Single-agent static/dynamic benchmarks | `G_8 = (V, E_8)` | 8-connected; orthogonal cost 1; diagonal cost √2; diagonal move allowed iff destination cell in-bounds and walkable (no cardinal-adjacency check); octile heuristic |
| MAPF (frozen) | `G_4 = (V, E_4)` | 4-connected + WAIT; unit-cost steps |

Implementation stores walkable cells in `GridMap.cells` and induces edges implicitly — this does **not** invalidate a formal graph model.

### Formula audit

| Formula | Verdict | Notes |
|---|---|---|
| Grid `G = (V,E)` / `G_8`, `G_4` | **MATCH** (with notation split) | Valid as induced grid graphs; use **separate** `G_8` vs `G_4` — not one shared neighbourhood. |
| Agent set `A = {a_1,…,a_n}`, `a_i = (s_i, g_i)` | **MATCH** (mathematical) | Index `i` identifies the agent in the math model. |
| Implementation agent record | **Model vs implementation** | `MAPFAgent(agent_id, start, goal)` — `agent_id` is runtime identity; map to index `i` via scenario ordering when needed. Do not require `agent_id` in the mathematical tuple unless stylistically preferred. |
| Timed path `π_i = (v_i^0,…,v_i^{T_i})` | **MATCH** | `AgentPath.states` as timed `(row,col,t)` sequence |
| Stay-at-goal `π̄_i(t)` = goal for `t > T_i` | **MATCH** | `_position_at` in `conflicts.py`; reservations pad goal to `max_timestep` |
| SoC `Σ_i (len(path_i)-1)` | **MATCH** | `metrics.sum_of_costs` |
| Makespan `max_i (len(path_i)-1)` | **MATCH** | `metrics.makespan` |
| Space-Time state `(row,col,timestep)` | **MATCH** | `TimedState` |
| PP ordering `σ ∈ S_n` | **MATCH** | Permutation encoded as `MAPFScenario.agents` order |
| SPF key `(independent_path_cost, original_index)` | **MATCH** | `shortest_path_first_order` |
| LPF key `(-independent_path_cost, original_index)` | **MATCH** | `longest_path_first_order` |
| Conflict graph: one edge per conflicting pair | **MATCH** | `conflict_graph_edges` canonicalizes `(min_index,max_index)` |
| CDF-H `(-degree, cost, original_index)` | **MATCH** | `cdf_h_sort_key` |
| CDF-L `(degree, cost, original_index)` | **MATCH** | `cdf_l_sort_key` |
| SPF+CD `(cost, -degree, original_index)` | **MATCH** | `spf_cd_sort_key` |
| CGLPS candidate: one transposition of SPF order | **MATCH** | `transpose_order` on SPF order |
| Pair multiplicity = independent conflict events | **MATCH** | `pair_conflict_event_counts` |
| Budget `B = 4` | **MATCH** | `MAPF9_CGLPS_BUDGET` |
| `K_x = min(B, \|E_C(x)\|)` | **MATCH** | `E_C(x)` = simple undirected conflict-graph edge set (one edge per conflicting pair). `B = 4`. Thesis per-instance budget **K_x**; raw artifact fields `actual_a_i`, `expected_a_i_from_manifest` (per instance) and `sum A_i` (aggregate Σ K_x) correspond to thesis notation. Implementation field: `additional_pp_eval_count`. On all **54** MAPF-9 production instances: `expected_a_i_from_manifest == actual_a_i == min(4, conflicting_agent_pair_count)` with `budget_match_ok=True`. Code contains defensive duplicate-order skipping (`seen_orders` in `generate_cglps_candidate_specs`), but **no production instance** required it — distinct pairs on a fixed SPF permutation yield distinct transpositions. |
| Edge constraint arrival timestep | **MATCH** | Documented on `EdgeConstraint` / `EdgeConflict` |
| CBS cardinal-first | **MATCH** | Separate classification + selection modules |

---

## 13. Testing / correctness evidence

**Current total (collect-only):** **906 tests** in `pathfinding/tests` (2026-09-03).

| Category | Test file | Invariant tested |
|---|---|---|
| Conflict detection | `test_mapf_conflicts.py` | Vertex/edge/stay-at-goal; non-conflicting cases |
| Space-Time A* constraints | `test_space_time_astar.py` | Vertex/edge forbiddance, goal safety, WAIT |
| Reservations | `test_mapf_reservations.py` | Vertex, reverse-edge, goal padding |
| PP | `test_prioritized_planning.py` | Fixed order, failure, stats |
| CBS core | `test_cbs_solver.py`, `test_cbs_root.py`, `test_cbs_expansion.py` | Success, conflict-free solutions |
| CBS splitting / cardinal | `test_cbs_splitting.py`, `test_cbs_conflict_classification.py`, `test_cbs_conflict_selection.py` | Split semantics, classification, selection |
| CBS stats / duplicates | `test_cbs_stats.py`, `test_cbs_duplicate_analysis.py` | CT metrics, plateau diagnostics |
| CBS timeout | `test_mapf_cbs_wall_clock_timeout.py` | 180 s process guard |
| Priority ordering | `test_priority_ordering.py`, `test_conflict_aware_priority_ordering.py` | SPF/LPF/CDF keys, graph |
| MAPF-6 analysis | `test_mapf_priority_strategy_analysis.py` | Production CSV consistency |
| MAPF-7 final | `test_mapf7_final_analysis.py`, `test_mapf7_final_analysis_plots.py` | Final table/plot invariants |
| Held-out / cross-map | `test_mapf_heldout_benchmark_catalogue.py`, `test_mapf_cross_map_catalogue_*.py` | Manifest disjointness, structure |
| CGLPS/UBLS | `test_mapf_bounded_local_priority_search.py`, `test_mapf9_integration.py` | Budget, matched K_x, selection |
| MAPF-8/9 final analysis | `test_mapf8_final_analysis.py`, `test_mapf9_final_analysis.py` | Frozen presentation numbers |
| Production-data consistency | `test_mapf_benchmark_analysis.py`, `test_mapf9_primary_analysis.py` | Analysis matches execution CSVs |
| Single-agent regression | `test_static_algorithms_on_movingai.py`, `test_dynamic_replanning.py` | found + path_cost on fixtures |

---

## 14. DO NOT USE AS FINAL THESIS EVIDENCE

| Path | Reason | Authoritative replacement |
|---|---|---|
| `pathfinding/results/mapf_conflict_aware_priority_analysis/thesis_tables.md` | MAPF-7 **interim** (primary-only framing; pre-held-out finalization) | `mapf7_final_analysis/thesis_tables.md` |
| `pathfinding/results/mapf_conflict_aware_priority_analysis/analysis_summary.md` | Superseded by MAPF-7.9 final summary | `mapf7_final_analysis/final_analysis_summary.md` |
| `README.md` §Known limitations ("MAPF out of scope") | **Stale** pre-MAPF roadmap text | This inventory + MAPF final analysis docs |
| `README.md` dynamic benchmark "illustrative" paragraph | Explicitly not final thesis evidence | Re-verify from CSV if citing numerically |
| Any pre-MAPF-8 roadmap / design docs outside `mapf9_design_freeze.md` | May predate cross-map/CGLPS closure | Frozen stage docs in `results/mapf*_final_analysis/` |
| `mapf_benchmark_analysis/analysis_summary.md` header "MAPF-5C.1" | Stage label; thesis should use `thesis_tables.md` | `mapf_benchmark_analysis/thesis_tables.md` |
| `pathfinding/plots/plot_static_algorithms_results.py` (default cluster_16 input) | Wrong CSV wired as default | Regenerate from cluster_32 CSV per `thesis_single_agent_dynamic_summary.md` §7 |
| Exploratory plots not listed in §9 FINAL sets | Not stage-gated | Named `fig*` in `mapf7/8/9_final_analysis` or stage plot modules |
| `pathfinding/results/mapf8_cross_map_analysis/` alone for **thesis prose** | Audit layer; presentation consolidated | `mapf8_final_analysis/` |
| Recomputing MAPF-7 held-out as "cross-map" | Methodological error | Use MAPF-8 for cross-map; MAPF-7 held-out for same-map only |
| Random K=10 **minimum** SoC as fair baseline | Marked diagnostic in MAPF-6 analysis | Use SPF/LPF/Fixed or label explicitly as sampled best-of-K |
| Manifest JSON without SHA check in logs | Integrity | Use SHA from validation.md / run.log / §8 table |
| `*.log` files in gitignore | May be local-only | Prefer committed CSV/MD validation artifacts |

---

## 15. Final evidence matrix

| Thesis section | Implementation evidence | Numerical evidence | Final figure/table | Literature still needed |
|---|---|---|---|---|
| Single-agent static | `astar.py`, `hpa/`, `jps.py`, loaders | `Results/static_algorithms/*.csv`; **`thesis_single_agent_dynamic_summary.md`** | Static plot scripts (cluster_32 input) | A*, HPA*, JPS original papers |
| Single-agent dynamic | `dynamic_simulation.py`, `dstar_lite.py` | `Results/dynamic_algorithms/*.csv`; **`thesis_single_agent_dynamic_summary.md`** | Dynamic plot scripts | D* Lite, replanning surveys |
| MAPF model | `mapf/models.py`, `conflicts.py` | — | — | MAPF definition, conflict types |
| PP | `prioritized_planning.py` | MAPF-5/6 results | MAPF-5 fig03, MAPF-6 fig01 | PP completeness references |
| CBS | `cbs.py`, `cbs_*.py` | MAPF-5 thesis tables | MAPF-5 fig04 | CBS, cardinal conflicts papers |
| Priority sensitivity (Ch.6) | `priority_ordering.py` | MAPF-6 tables + K20 summary | MAPF-6 fig01–02 | — |
| Static conflict heuristics (Ch.6) | CDF-* in `priority_ordering.py` | MAPF-7 final tables | MAPF-7 fig01–03 | Conflict-degree heuristics |
| Cross-map validation (Ch.7) | Same ordering stack | MAPF-8 final + historical CSV | MAPF-8 four figs | bg512 / MovingAI context |
| CGLPS/UBLS (Ch.8) | `mapf_bounded_local_priority_search.py` | MAPF-9 final tables | MAPF-9 three figs | Local search / MAPF heuristics |
| Reproducibility (Ch.4) | Catalogue generators, validators | Manifest SHA table §8 | Validation MD files | — |
| Single-agent + dynamic (Ch.5) | §1, `thesis_single_agent_dynamic_summary.md` | Static + dynamic CSVs | §9 Ch.5 figures | — |

---

## 16. Missing / unresolved

| Gap | Detail |
|---|---|
| Figure binaries in git | MAPF-7/8/9 final PNG/PDF and MAPF-5/6 plot outputs **not committed**; captions and generators exist — regenerate before thesis embed |
| AR0204 primary/held-out validation MD | No `AR0204SR_validation.md` with SHA (unlike MAPF-8/9); SHA computed locally for inventory (§8) but not logged in committed validation artifact |
| Final thesis LaTeX draft | No `.tex` in repo at inventory time — draft location outside this repository |
| Hardware environment | Not recorded in frozen MAPF execution artifacts |
| Exact Python patch version | README recommends 3.13; not pinned in result CSVs |
| Single-agent/dynamic **aggregated summary** | Resolved in `pathfinding/docs/thesis_single_agent_dynamic_summary.md` (THESIS-1a); raw CSVs remain authoritative |
| Dynamic benchmark multi-seed | Only seed 42 CSV committed |
| HPA cluster_16 vs cluster_32 | Two benchmark files; thesis must declare which is primary (cluster_32 per `benchmark_main.py`) |
| Bibliography | Not in repo |

---

## 17. Chapter → evidence mapping (frozen structure)

| Ch. | Title (PL) | Primary evidence |
|---|---|---|
| **1** | Wprowadzenie | — (narrative) |
| **2** | Podstawy teoretyczne i stan wiedzy | Literature external; implementation context §1.4, §2 |
| **3** | Projekt i implementacja systemu | §2, §11, test files §13 |
| **4** | Metodologia badań | §8 catalogue table, `docs/DYNAMIC_BENCHMARK.md`, manifest validation MDs, execution logs |
| **5** | Badania wstępne: single-agent i dynamic replanning | §1, **`thesis_single_agent_dynamic_summary.md`**, static/dynamic CSVs, §9 Ch.5 figures |
| **6** | Wpływ kolejności priorytetów na rozwiązania MAPF | MAPF-5 §3, MAPF-6 §4, MAPF-7 §5 |
| **7** | Walidacja między mapami | MAPF-8 §6 |
| **8** | Konfliktowo sterowane lokalne przeszukiwanie kolejności priorytetów | MAPF-9 §7, `mapf9_design_freeze.md` |
| **9** | Dyskusja | Synthesis across §15 matrix |
| **10** | Zakończenie | — (narrative) |

---

*End of THESIS-1 evidence inventory.*
