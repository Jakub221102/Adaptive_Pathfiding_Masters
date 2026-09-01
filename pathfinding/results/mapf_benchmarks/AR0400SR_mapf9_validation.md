# MAPF-9 Catalogue Validation — AR0400SR

## Identity

- Map: **AR0400SR**
- MAPF-9 seed: **2030**
- Frozen MAPF-8 reference seed: **2028**
- Manifest path: `C:\Users\Jakub\PythonProject\Adaptive_Pathfiding_Masters\pathfinding\results\mapf_benchmarks\AR0400SR_mapf9_manifest.json`
- SHA-256: `14964ee449b826c6280299f68a6cfc0cc42d34f3045d87070971027009dbb71c`
- Validation status: **PASS**

## Generation configuration

- max_timestep: 512
- min_reference_length: 20.0
- max_attempts: 5000
- agent_counts: [5, 10, 20]
- instances_per_interaction_cell: 3
- interaction strata: LOW=0, MEDIUM=1..2, HIGH>=3 independent-path conflicts

## Catalogue structure

- instance_count: 27
- counts_by_agent_count: {5: 9, 10: 9, 20: 9}
- counts_by_interaction_level: {'low': 9, 'medium': 9, 'high': 9}
- cell_counts (agent_count, interaction): {(5, 'low'): 3, (5, 'medium'): 3, (5, 'high'): 3, (10, 'low'): 3, (10, 'medium'): 3, (10, 'high'): 3, (20, 'low'): 3, (20, 'medium'): 3, (20, 'high'): 3}

## Freshness / disjointness

- MAPF-9 scenario-set signature count: 27
- MAPF-8 reference scenario-set signature count: 27
- signature intersection count: **0**
- MAPF-8 reference manifest: `C:\Users\Jakub\PythonProject\Adaptive_Pathfiding_Masters\pathfinding\results\mapf_benchmarks\AR0400SR_manifest.json`

## Source-pool diagnostics

- source_scenario_count: 2890
- eligible_scenario_count: 2840
- feasible_independent_path_count: 964
- over_horizon_count: 1876
- no_spatial_path_count: 0

## Generation paths

- map_path: `C:\Users\Jakub\PythonProject\Adaptive_Pathfiding_Masters\Data\bg512-map\AR0400SR.map`
- scen_path: `C:\Users\Jakub\PythonProject\Adaptive_Pathfiding_Masters\Data\bg512-scen\AR0400SR.map.scen`
- generation_log: `C:\Users\Jakub\PythonProject\Adaptive_Pathfiding_Masters\pathfinding\results\mapf9_catalogue_AR0400SR.log`

## Validation checks

- PASS: MAPF-9 manifest identity
- PASS: instance ID prefixes
- PASS: manifest structure (27-instance 3x3 design)
- PASS: MAPF-9 vs MAPF-8 scenario-set signature disjointness
- PASS: within-manifest scenario-set signatures unique
- PASS: scenario semantics
- PASS: independent-path metadata recomputation

## Methodological note

Catalogue construction used only deterministic scenario sampling and independent-path interaction classification. No SPF, CDF-H, CDF-L, SPF+CD, PP, CGLPS, UBLS, or other strategy outcomes influenced instance selection.
