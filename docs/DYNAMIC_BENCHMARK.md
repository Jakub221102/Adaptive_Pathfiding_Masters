# Dynamic Benchmark

Reference for the A\* replanning vs D\* Lite dynamic benchmark (`DynamicBenchmarkExperiment`).

---

## 1. Purpose

Measure how two replanning strategies perform when an NPC crosses a grid map while rectangular moving obstacles change walkability along the route.

The benchmark answers:

- Does the agent reach the goal (`final_goal_reached`)?
- How often must the planner react (`replanning_count`, `waiting_steps`, collisions)?
- What is the computational cost (`total_execution_time_ms`) vs end-to-end scenario time (`wall_clock_s`)?
- How does path cost compare (`total_path_cost`)?

Collision handling, waiting, and obstacle motion are defined in the **simulation layer** (`dynamic_simulation.py`), not inside A\* or D\* Lite.

---

## 2. Algorithms compared

| Label | Implementation | Behavior in simulation |
|-------|----------------|------------------------|
| **A\*** | `AStar.find_path` / `find_path_with_steps` | Full replan from current position whenever lookahead detects blockage or walkability fails. |
| **D\* Lite** | `DStarLite` | Incremental cost updates on map changes; `replan_with_steps` from current agent position; `move_agent` after each step. |

Both algorithms share the same map, scenario, moving obstacles, and simulation parameters for a given `scenario_index`.

---

## 3. Scenario selection

1. Load map and scenario file (default: BG512 `AR0204SR` in `dynamic_benchmark_main.py`).
2. Filter scenarios with `optimal_length >= min_optimal_length` (default 100).
3. Take the first `scenario_count` scenarios (default 50).
4. For each scenario, generate moving obstacles (see §4). If generation fails after 10 attempts (start/goal blocked), the scenario is skipped with failed stats for both algorithms.

Scenario index and per-scenario seed are recorded in CSV (`scenario_index`, `seed`).

---

## 4. Obstacle generation

Obstacles are created by `generate_valid_scenario_obstacles()`:

1. Compute reference static path with A\* from start to goal.
2. Restrict placement to a bounding box around that path (`path_margin`, default 30 cells).
3. Call `generate_moving_obstacles()` with:
   - `seed = obstacle_seed + scenario_index` (+ retry offset on failed attempts),
   - `obstacle_count`, `min_obstacle_size`, `max_obstacle_size`,
   - `min_obstacle_spacing` (Chebyshev distance between obstacle centers),
   - forbidden start and goal cells.
4. Reject layouts where start or goal is covered.

Default config (`dynamic_benchmark_main.py`): `obstacle_seed=42`, `obstacle_count=4`, size 4–10, `min_obstacle_spacing=25`.

This keeps obstacles **near the intended route** while spacing reduces single-cell bottlenecks and overlapping obstacle stacks.

---

## 5. Replanning policy

Simulation parameters (defaults in `DynamicBenchmarkConfig` / `ExperimentConfig`):

| Parameter | Role |
|-----------|------|
| `path_block_lookahead` | Replan if any cell on the path within N steps ahead is blocked. |
| `wait_when_no_path` | If replan fails, wait in place instead of stopping immediately. |
| `max_wait_steps` | Maximum consecutive waits before giving up. |
| `max_stuck_steps` | Stop if agent cannot progress (D\* Lite / blocked moves). |
| `max_simulation_steps` | Hard cap on simulation iterations. |
| `moving_obstacle_prediction_steps` | During planning, temporarily block cells where obstacles are predicted to move (0 = disabled in default benchmark). |
| `moving_obstacle_collision_policy` | `PUSH_AGENT` (default) or `BLOCK_OBSTACLE` (bounce). |

**Waiting policy:** when replanning returns no path and `wait_when_no_path` is true, the agent increments `waiting_steps` until `max_wait_steps` or a path appears.

**Lookahead:** `is_path_blocked_with_lookahead()` checks the remaining path segment, triggering replan before the agent walks into a blocked cell.

**Obstacle prediction:** `with_predicted_obstacle_blocks()` applies temporary blocks for predicted obstacle positions during each plan/replan call.

---

## 6. Metrics

| Metric | Meaning |
|--------|---------|
| `found` | Initial planning succeeded (legacy flag; see `initial_path_found`). |
| `final_goal_reached` | Agent reached goal within simulation limits. |
| `initial_path_found` | First plan from start found a path. |
| `replanning_count` | Number of replanning invocations. |
| `waiting_steps` | Steps spent waiting because replanning failed (under waiting policy). |
| `collision_count` | Agent–obstacle collision events detected. |
| `agent_push_count` | Times agent was pushed (`PUSH_AGENT` policy). |
| `obstacle_blocked_count` | Times obstacle movement was blocked (`BLOCK_OBSTACLE` policy). |
| `travelled_steps` | Agent movement steps taken. |
| `total_path_cost` | Sum of octile edge costs along travelled route. |
| `total_execution_time_ms` | **Sum of algorithm planning/replanning CPU time** across the scenario. |
| `wall_clock_s` | **Real elapsed wall-clock time** for the entire scenario run (simulation + planning + I/O). |

### `total_execution_time_ms` vs `wall_clock_s`

- **`total_execution_time_ms`** — aggregated time reported by pathfinding calls only (initial plan + each replan). Use this to compare algorithmic work between A\* and D\* Lite.
- **`wall_clock_s`** — measures the full benchmark iteration for that algorithm and scenario, including simulation loop, obstacle updates, collision handling, and Python overhead. Use this for end-to-end latency and outlier detection.

A scenario can have low algorithm time but higher wall-clock time if the simulation runs many steps with many obstacle moves and collision checks.

**Dynamic interaction rate** (printed in summary, not a CSV column): fraction of runs where `replanning_count > 0` or `waiting_steps > 0` or `collision_count > 0`. Combined rate uses scenario-level flags when either algorithm had interactions.

---

## 7. CSV columns

Exported by `export_dynamic_benchmark_results()` (`pathfinding/src/utils/csv_exporter.py`):

```
algorithm
scenario_index
seed
found
final_goal_reached
initial_path_found
replanning_count
waiting_steps
collision_count
agent_push_count
obstacle_blocked_count
travelled_steps
total_path_cost
total_execution_time_ms
wall_clock_s
obstacle_count
prediction_steps
path_block_lookahead
start_row
start_col
goal_row
goal_col
optimal_length
```

Default output: `Results/dynamic_algorithms/astar_vs_dstar_dynamic_seed_42.csv`.

---

## 8. Running the benchmark

From `pathfinding/scripts/`:

```bash
python dynamic_benchmark_main.py
```

Plots from `pathfinding/plots/`:

```bash
python plot_dynamic_benchmark_results.py
```

Debug / inspection:

```bash
python dynamic_benchmark_single_scenario_debug.py
python dynamic_benchmark_scenario_viewer.py
```

Requires map data under `Data/` (see `ExperimentConfig.map_path` in the script).

---

## 9. Interpreting results

**Goal reach rate** — primary success criterion (`final_goal_reached`). Compare A\* vs D\* Lite on the same scenarios.

**Replanning and waiting** — high `replanning_count` or `waiting_steps` indicates a heavily dynamic route. D\* Lite may wait more when incremental repair fails temporarily.

**Path cost** — `total_path_cost` should stay close between algorithms when both reach the goal; large gaps may indicate detours or failed late-stage navigation.

**Execution time** — compare median and p95 of `total_execution_time_ms`. A\* is often faster per call on small replans; D\* Lite amortizes work but has higher per-scenario overhead in the current implementation.

**Wall-clock outliers** — check `wall_clock_s` and the printed slowest-scenario report. A\* can show extreme max algorithm time on hard full replans; D\* Lite max is often lower but success rate may drop.

**Dynamic interaction rate** — if below `min_dynamic_interaction_rate` (default 0.5), the benchmark prints a warning: increase `obstacle_count`, obstacle size, or reduce `path_margin` / spacing constraints.

### Example (seed 42, 50 scenarios, in-repo CSV)

| | A\* | D\* Lite |
|---|-----|----------|
| Goal reach | 100% | 96% |
| Median `total_execution_time_ms` | ~18 ms | ~115 ms |
| p95 | ~108 ms | ~534 ms |
| Max `total_execution_time_ms` | ~15.5 s | ~2.4 s |
| Dynamic interaction rate | ~72% (combined) | |

*Single run, default config — illustrative only.*

---

## 10. Known limitations

- **No time-expanded planning** — planners see current occupancy (plus optional short prediction), not future joint agent–obstacle schedules.
- **Shared generator bias** — obstacles are placed using the static A\* reference path; layouts favor that geometry.
- **Spacing heuristics** — `min_obstacle_spacing` limits pathology but does not guarantee MAPF feasibility.
- **Single agent** — no coordination with other NPCs.
- **D\* Lite** — educational implementation; not tuned for large maps or minimal heap operations.
- **HPA\*** — not included in this benchmark; hierarchical repair under motion is future work.
- **Deterministic seed** — one seed supports reproducibility; multi-seed studies are not yet automated in the default script.
