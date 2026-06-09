# Adaptive Pathfinding for Dynamic NPC Environments

Master's thesis project: implementation, analysis, and comparison of pathfinding algorithms for NPC agents on grid maps in static and dynamic environments.

Detailed design rules and thesis context (Polish): [`docs/ZASADY_PROJEKTOWE.md`](docs/ZASADY_PROJEKTOWE.md).  
Dynamic benchmark reference: [`docs/DYNAMIC_BENCHMARK.md`](docs/DYNAMIC_BENCHMARK.md).

---

## Project goal

This project implements and compares pathfinding algorithms for NPC agents on grid maps, with support for:

- static maps (MovingAI, Baldur's Gate BG512),
- dynamic obstacles (runtime block/unblock),
- moving obstacles with collision handling,
- replanning under changing conditions.

The focus is on path quality (`path_cost`), computational cost, scalability, and reactive behavior when the environment changes. Collision logic and agent waiting behavior live in the **simulation layer** (`dynamic_simulation.py`), not inside A* or D* Lite themselves.

---

## Requirements

- Python 3.13 (recommended; developed in PyCharm)
- Dependencies: `requirements.txt`

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Repository root must be on `PYTHONPATH` (default in PyCharm: module root = repo root).

**Note:** MovingAI / BG512 maps are **not** in the repository. Place them locally under `Data/` (scripts reference `../../Data/...` relative to `pathfinding/scripts/`).

---

## Implemented algorithms

| Algorithm | Role |
|-----------|------|
| **A\*** | Baseline static pathfinding: 8-directional movement, octile heuristic, corner cutting enabled. |
| **HPA\*** | Hierarchical pathfinding: cluster abstraction, entrances, abstract graph search, local path cache, refinement. |
| **JPS** | Jump Point Search optimization for static grids; path cost matches A\*. |
| **D\* Lite** | Incremental replanning for dynamic maps; used together with A\* replanning in dynamic benchmarks. |

In dynamic scenarios, A\* performs full replanning on each detected blockage; D\* Lite updates costs incrementally and replans from the current agent position.

---

## Static benchmarks

Compares **A\*** vs **HPA\*** vs **JPS** on MovingAI / BG512 scenarios.

- CSV export to `Results/static_algorithms/`
- Plot scripts in `pathfinding/plots/`
- Regression tests assert `found` and `path_cost`; **not** `execution_time_ms`

Run from `pathfinding/scripts/` (paths to `Data/` and `Results/` are relative to this directory):

```bash
cd pathfinding/scripts
python benchmark_main.py
```

Additional HPA\* parameter studies:

```bash
python comparison_main.py
python cluster_size_benchmark_main.py
python max_entrances_benchmark_main.py
```

Generate static plots from `pathfinding/plots/`:

```bash
cd pathfinding/plots
python plot_static_algorithms_results.py
python plot_cluster_size_results.py
python plot_max_entrances_results.py
```

---

## Dynamic environment

Core mechanics (simulation layer):

| Component | Description |
|-----------|-------------|
| `DynamicGridMap` | Wraps a static `GridMap` with runtime `dynamic_blocked` / `dynamic_unblocked` cells. |
| `DynamicObstacleEvent` | Timed `BLOCK` / `UNBLOCK` rectangle events applied during simulation. |
| `MovingObstacle` | Rectangular obstacles that move each simulation step (horizontal/vertical bounce). |
| Seeded generator | `generate_moving_obstacles()` — deterministic placement via `obstacle_seed + scenario_index`. |
| Lookahead replanning | `path_block_lookahead` triggers replan when the planned path ahead is blocked. |
| Waiting policy | `wait_when_no_path` + `max_wait_steps` — agent waits in place when replanning fails. |
| Collision policy | `MovingObstacleCollisionPolicy`: `PUSH_AGENT` (displace agent) or `BLOCK_OBSTACLE` (obstacle bounces). |
| Obstacle prediction | `moving_obstacle_prediction_steps` — temporarily blocks predicted obstacle cells during planning. |
| Spacing constraints | `min_obstacle_spacing` and forbidden start/goal positions reduce pathological bottlenecks. |

Collision resolution, waiting, and obstacle movement are handled in `pathfinding/src/experiments/dynamic_simulation.py` and `pathfinding/src/core/dynamic_models.py`, independent of the pathfinding algorithms.

---

## Dynamic benchmark

Compares **A\* replanning** vs **D\* Lite** on shared dynamic scenarios with generated moving obstacles.

Metrics exported to CSV:

- `final_goal_reached`
- `replanning_count`
- `waiting_steps`
- `collision_count`
- `agent_push_count`
- `obstacle_blocked_count`
- `travelled_steps`
- `total_path_cost`
- `total_execution_time_ms` — sum of planning/replanning algorithm time
- `wall_clock_s` — real elapsed time of the full benchmark scenario run

Run benchmark and plots:

```bash
cd pathfinding/scripts
python dynamic_benchmark_main.py
```

```bash
cd pathfinding/plots
python plot_dynamic_benchmark_results.py
```

Benchmark properties:

- fixed base seed (`obstacle_seed=42` in default config),
- deterministic obstacle generation per scenario,
- identical obstacle setup for A\* and D\* Lite in each scenario,
- obstacles generated near the reference A\* path (with margin), respecting `min_obstacle_spacing` to limit pathological narrow bottlenecks.

See [`docs/DYNAMIC_BENCHMARK.md`](docs/DYNAMIC_BENCHMARK.md) for full metric definitions and interpretation.

### Example dynamic benchmark result

Sample summary from `Results/dynamic_algorithms/astar_vs_dstar_dynamic_seed_42.csv` (seed 42, 50 scenarios, default config):

**A\*:**
- success rate: 100%
- median `total_execution_time_ms`: about 18 ms
- p95: about 108 ms
- max: about 15.5 s

**D\* Lite:**
- success rate: 96%
- median `total_execution_time_ms`: about 115 ms
- p95: about 534 ms
- max: about 2.4 s

**Dynamic interaction rate:** about 72%

*These are illustrative results from a single run and configuration — not final thesis conclusions.*

---

## Visualization

| Mode | Script | Description |
|------|--------|-------------|
| Static viewer | `pathfinding/scripts/main.py` | Single-scenario path display (`STATIC`, `ANIMATED`, `COMPARISON` in `ViewerMode`). |
| Dynamic replanning | `pathfinding/scripts/dynamic_replanning_viewer_main.py` | Step-by-step replanning with moving obstacles. |
| Generated obstacles demo | `pathfinding/scripts/dynamic_generated_obstacles_demo.py` | Seeded moving obstacles on a selected scenario. |
| Moving obstacles demo | `pathfinding/scripts/moving_obstacles_demo.py` | A\* replanning with predefined moving obstacles. |
| Benchmark scenario viewer | `pathfinding/scripts/dynamic_benchmark_scenario_viewer.py` | Inspect a single dynamic benchmark scenario. |

Overlays: HPA\* clusters, performance stats, visit heatmaps (`pathfinding/src/visualization/overlays/`).

```bash
cd pathfinding/scripts
python dynamic_generated_obstacles_demo.py
python dynamic_replanning_viewer_main.py
python main.py
```

---

## Results directory

```
Results/
├── static_algorithms/          # A* / HPA* / JPS CSV (+ cluster_size, max_entrances)
├── dynamic_algorithms/         # A* vs D* Lite dynamic benchmark CSV
└── plots/
    ├── static_algorithms/      # generated by plot_static_*.py
    └── dynamic_algorithms/     # generated by plot_dynamic_benchmark_results.py
```

Example files already in repo:

- `Results/static_algorithms/astar_hpa_jps_cluster_32_scenarios_100.csv`
- `Results/dynamic_algorithms/astar_vs_dstar_dynamic_seed_42.csv`

---

## Tests

From repository root:

```bash
pytest
```

Configuration: `pytest.ini` → `pathfinding/tests`.

Coverage includes:

- unit tests for A\*, JPS, HPA\*, D\* Lite,
- path cost consistency (A\* vs JPS),
- `DynamicGridMap`, moving obstacle generator and collision,
- dynamic replanning simulation,
- dynamic benchmark CSV export and scenario selection.

Regression on MovingAI checks `found` and `path_cost` — not execution times.

---

## Known limitations

- A\* and D\* Lite operate on the **current map state**, not a full time-expanded graph.
- Moving obstacles can create situations that require spatio-temporal planning beyond simple replanning.
- The obstacle generator limits pathological cases via spacing and exclusion rules, but does not solve full MAPF.
- Multi-agent pathfinding, CBS, reservation tables, and full time-aware planning are out of scope.
- D\* Lite implementation is research/educational and not heavily optimized.
- HPA\* dynamic replanning is not yet part of the dynamic benchmark.

---

## Suggested next steps

- threat avoidance / local avoidance field,
- better time-aware obstacle prediction,
- dynamic benchmark over multiple seeds,
- MAPF / cooperative pathfinding as future work,
- optimization of D\* Lite implementation,
- HPA\* in dynamic benchmark (roadmap item).

---

## Repository layout

```
Master_Path_v1/
├── pathfinding/
│   ├── src/
│   │   ├── algorithms/     # A*, JPS, HPA*, D* Lite
│   │   ├── core/           # models, dynamic_models, moving_obstacle_generator
│   │   ├── experiments/    # static & dynamic benchmarks, simulation
│   │   ├── loaders/
│   │   ├── utils/
│   │   └── visualization/
│   ├── scripts/
│   ├── plots/
│   └── tests/
├── Data/                   # maps & scenarios (local, not in git)
├── Results/                # benchmark CSV and plots (in repo)
├── docs/
│   ├── ZASADY_PROJEKTOWE.md
│   └── DYNAMIC_BENCHMARK.md
└── pytest.ini
```

Cursor agent rules: `.cursor/rules/master-thesis.mdc`.
