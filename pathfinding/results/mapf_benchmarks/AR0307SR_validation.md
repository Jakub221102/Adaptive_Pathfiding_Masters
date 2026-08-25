# MAPF-8 Cross-Map Catalogue Validation — AR0307SR

## Identity

- Map: **AR0307SR**
- Expected seed: **2029**
- Manifest path: `C:\Users\Jakub\PythonProject\Adaptive_Pathfiding_Masters\pathfinding\results\mapf_benchmarks\AR0307SR_manifest.json`
- SHA-256: `ff6bb64e15d8c507861966e27a00df0df1efcf7b070584b0313fc9448b4f2732`
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

## Source-pool diagnostics

- source_scenario_count: 3380
- eligible_scenario_count: 3330
- feasible_independent_path_count: 958
- over_horizon_count: 2372
- no_spatial_path_count: 0

## Generation paths

- map_path: `C:\Users\Jakub\PythonProject\Adaptive_Pathfiding_Masters\Data\bg512-map\AR0307SR.map`
- scen_path: `C:\Users\Jakub\PythonProject\Adaptive_Pathfiding_Masters\Data\bg512-scen\AR0307SR.map.scen`
- generation_log: `C:\Users\Jakub\PythonProject\Adaptive_Pathfiding_Masters\pathfinding\results\mapf_cross_map_catalogue_AR0307SR.log`

## Validation checks

- PASS: manifest identity
- PASS: instance ID prefixes
- PASS: manifest structure (27-instance 3x3 design)
- PASS: scenario semantics
- PASS: independent-path metadata recomputation

## Methodological note

Catalogue construction used only deterministic scenario sampling and independent-path interaction classification. No SPF, CDF-H, CDF-L, SPF+CD, PP, or other strategy outcomes influenced instance selection.
