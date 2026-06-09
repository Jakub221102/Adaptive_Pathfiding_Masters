# Project context — Master Path v1

Short status document for supervisors, reviewers, and new contributors.  
Full README: [`README.md`](README.md). Design rules (PL): [`docs/ZASADY_PROJEKTOWE.md`](docs/ZASADY_PROJEKTOWE.md).

---

## Current status

**Static pathfinding — done**

- A\*, HPA\*, JPS implemented and regression-tested.
- Static benchmarks on MovingAI / BG512 with CSV export and plots.
- HPA\* parameter studies: `cluster_size`, `max_entrances_per_cluster_pair`.

**Dynamic environment — done (core scope)**

- `DynamicGridMap`, `DynamicObstacleEvent`, `MovingObstacle`.
- Simulation layer: lookahead replanning, waiting policy, collision policy (`PUSH_AGENT` / `BLOCK_OBSTACLE`), obstacle prediction.
- Seeded moving obstacle generator with spacing and start/goal exclusion.
- A\* replanning and D\* Lite integrated in dynamic simulation and visualization.

**Dynamic benchmark — done (A\* vs D\* Lite)**

- `DynamicBenchmarkExperiment` — 50 scenarios, seed 42, shared obstacle setup per scenario.
- CSV: `Results/dynamic_algorithms/astar_vs_dstar_dynamic_seed_42.csv`
- Plot script: `pathfinding/plots/plot_dynamic_benchmark_results.py`

**Not in scope yet**

- HPA\* in dynamic benchmark.
- Multi-agent / MAPF / CBS.
- Full time-expanded planning.

---

## Implemented modules (key paths)

| Area | Location |
|------|----------|
| Algorithms | `pathfinding/src/algorithms/` |
| Dynamic models & collision | `pathfinding/src/core/dynamic_models.py` |
| Obstacle generator | `pathfinding/src/core/moving_obstacle_generator.py` |
| Dynamic simulation | `pathfinding/src/experiments/dynamic_simulation.py` |
| Dynamic benchmark | `pathfinding/src/experiments/dynamic_benchmark_experiment.py` |
| Static benchmark | `pathfinding/src/experiments/benchmark_experiment.py` |
| Visualization | `pathfinding/src/visualization/` |
| Scripts | `pathfinding/scripts/` |
| Tests | `pathfinding/tests/unit/`, `pathfinding/tests/regression/` |

---

## How to run (quick)

```bash
pytest                                          # from repo root
cd pathfinding/scripts && python benchmark_main.py
cd pathfinding/scripts && python dynamic_benchmark_main.py
cd pathfinding/plots && python plot_dynamic_benchmark_results.py
cd pathfinding/scripts && python dynamic_generated_obstacles_demo.py
```

Maps: local `Data/` (not in git). Results: `Results/` (in git).

---

## Dynamic benchmark status

- Compares A\* full replanning vs D\* Lite incremental replanning.
- Deterministic obstacles: `obstacle_seed + scenario_index`, placed near reference A\* path.
- `total_execution_time_ms` = sum of algorithm planning time; `wall_clock_s` = full scenario wall time.
- Example run (seed 42): A\* 100% goal reach, D\* Lite 96%, ~72% scenarios with dynamic interactions.

Details: [`docs/DYNAMIC_BENCHMARK.md`](docs/DYNAMIC_BENCHMARK.md).

---

## Known limitations

- No time-expanded graph; algorithms see current (optionally predicted) occupancy.
- Collision and waiting behavior are simulation policies, not algorithm internals.
- Generator reduces but does not eliminate MAPF-hard layouts.
- D\* Lite is correct but not performance-tuned for production.

---

## Next steps

1. Threat / local avoidance field.
2. Stronger time-aware obstacle prediction.
3. Dynamic benchmark across multiple seeds and map sets.
4. MAPF / cooperative pathfinding (CBS, reservations).
5. D\* Lite and dynamic simulation performance optimization.
6. HPA\* dynamic replanning benchmark (roadmap).
