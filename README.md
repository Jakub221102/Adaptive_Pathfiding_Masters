# Adaptive Pathfinding — Master's Thesis Implementation

Implementation, benchmarks, and analysis for pathfinding and Multi-Agent Path Finding (MAPF) on grid maps.

**Thesis (PL):** *Wpływ kolejności priorytetów i informacji o konfliktach na wieloagentowe wyznaczanie ścieżek*  
**Thesis (EN):** *The Impact of Priority Ordering and Conflict Information on Multi-Agent Path Finding*

The main research focus is how **priority ordering** and **conflict information** affect **Prioritized Planning (PP)** in MAPF. The repository also contains earlier single-agent and dynamic-replanning work used as supporting context.

Additional design notes: [`docs/ZASADY_PROJEKTOWE.md`](docs/ZASADY_PROJEKTOWE.md), [`docs/DYNAMIC_BENCHMARK.md`](docs/DYNAMIC_BENCHMARK.md), [`pathfinding/docs/mapf9_design_freeze.md`](pathfinding/docs/mapf9_design_freeze.md).

---

## Project scope

| Area | Focus |
|------|--------|
| **Single-agent** | A\*, HPA\*, JPS on static MovingAI / BG512 maps |
| **Dynamic** | A\* replanning and D\* Lite under runtime obstacles |
| **MAPF** | Space-Time A\*, constraints, conflict detection, reservations, PP, CBS, priority-order research methods |
| **Evaluation** | Reproducible benchmark catalogues, resumable execution, frozen analysis artifacts |

Collision handling, waiting policies, and moving-obstacle simulation for dynamic scenarios live in the **simulation layer** (`dynamic_simulation.py`), not inside the pathfinding algorithms themselves.

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

**Maps:** MovingAI / BG512 maps are **not** in the repository. Place them locally under `Data/` (scripts reference `../../Data/...` relative to `pathfinding/scripts/`).

---

## Implemented algorithms

### Single-agent (static)

| Algorithm | Role |
|-----------|------|
| **A\*** | Baseline: 8-directional movement, octile heuristic, corner cutting. |
| **HPA\*** | Hierarchical pathfinding: cluster abstraction, entrances, abstract graph search, local path cache, refinement. |
| **JPS** | Jump Point Search; path cost matches A\* on the same grid. |

### Dynamic pathfinding

| Algorithm | Role |
|-----------|------|
| **A\* replanning** | Full replan from current position when the map or lookahead path changes. |
| **D\* Lite** | Incremental cost updates and replanning for dynamic maps. |

### MAPF core (`pathfinding/src/algorithms/mapf/`)

| Component | Role |
|-----------|------|
| **Space-Time A\*** | Low-level planner with vertex and edge constraints. |
| **Conflict detection** | Vertex and edge conflicts between agent paths. |
| **Reservations** | Convert planned paths into constraints for subsequent agents in PP. |
| **Prioritized Planning** | Sequential planning in agent priority order. |
| **Basic CBS** | Conflict-Based Search with standard conflict selection. |
| **Cardinal-First CBS** | CBS with cardinal/semi-cardinal conflict prioritization. |

### Priority strategies and research methods

All conflict-aware orderings (CDF-H, CDF-L, SPF+CD) are **static and instance-adaptive**: computed once from independent Space-Time A\* paths and their conflicts before standard PP.

| Method | Description |
|--------|-------------|
| **Fixed** | Original catalogue agent order (`fixed_priority_pp`). |
| **Random** | Seeded shuffle of agent priorities. |
| **SPF** | Shortest-Path First — sort by independent path cost. |
| **LPF** | Longest-Path First — sort by descending independent path cost. |
| **CDF-H** | Conflict-Degree First (high degree first). |
| **CDF-L** | Conflict-Degree First (low degree first). |
| **SPF+CD** | Shortest path first, then higher conflict degree. |
| **CGLPS** | Conflict-Guided Bounded Local Priority Search — bounded search over conflict-guided local order modifications (MAPF-9). |
| **UBLS** | Uniform-Budget Local Search baseline for MAPF-9 comparison. |

---

## Repository structure

```
Adaptive_Pathfiding_Masters/
├── pathfinding/
│   ├── src/
│   │   ├── algorithms/          # A*, JPS, HPA*, D* Lite, mapf/
│   │   ├── core/                # grid models, dynamic models, obstacle generator
│   │   ├── experiments/         # benchmarks, MAPF harnesses, analysis, thesis figures
│   │   ├── loaders/             # MovingAI map/scenario loaders
│   │   ├── utils/
│   │   └── visualization/       # viewers and overlays
│   ├── scripts/                 # runnable entry points (*_main.py)
│   ├── plots/                   # static/dynamic result plotting
│   ├── tests/                   # unit and regression tests
│   ├── results/                 # MAPF manifests, execution CSV/JSONL, analysis outputs
│   └── docs/                    # MAPF design freeze, thesis evidence notes
├── Results/                     # single-agent and dynamic benchmark CSV/plots
├── Data/                        # maps & scenarios (local, not in git)
├── docs/                        # project rules, dynamic benchmark docs, thesis sources
└── pytest.ini
```

Key packages:

- `pathfinding/src/algorithms/` — algorithm implementations (including `mapf/` subpackage).
- `pathfinding/src/experiments/` — benchmark orchestration, checkpoint/resume logic, analysis pipelines.
- `pathfinding/scripts/` — PyCharm-friendly `*_main.py` entry points.
- `pathfinding/tests/` — unit tests for algorithms and experiment harnesses; regression tests for MovingAI scenarios.

---

## Running the project

### Tests

From repository root:

```bash
pytest
```

Configuration: `pytest.ini` → `pathfinding/tests`.

### Quick / local runs

From `pathfinding/scripts/`:

```bash
cd pathfinding/scripts

# MAPF smoke test (synthetic + MovingAI scenarios, no pygame)
python mapf_smoke_test_main.py

# MAPF demo viewer (requires Data/ maps)
python mapf_demo_main.py

# Single-scenario static path viewer
python main.py

# Dynamic replanning viewer
python dynamic_replanning_viewer_main.py
```

### Single-agent static benchmarks

```bash
cd pathfinding/scripts
python benchmark_main.py
python comparison_main.py              # HPA* parameter studies
python cluster_size_benchmark_main.py
python max_entrances_benchmark_main.py
```

Plots from `pathfinding/plots/`:

```bash
cd pathfinding/plots
python plot_static_algorithms_results.py
python plot_cluster_size_results.py
python plot_max_entrances_results.py
```

CSV output: `Results/static_algorithms/`.

### Dynamic benchmark

```bash
cd pathfinding/scripts
python dynamic_benchmark_main.py
```

CSV output: `Results/dynamic_algorithms/`. See [`docs/DYNAMIC_BENCHMARK.md`](docs/DYNAMIC_BENCHMARK.md) for metric definitions.

### MAPF experiments

Most MAPF harness scripts are configured via constants at the top of each `*_main.py` file (map stem, manifest path, smoke mode). Open in PyCharm and run, or invoke from `pathfinding/scripts/`.

Representative entry points:

| Script | Purpose |
|--------|---------|
| `mapf_benchmark_execution_main.py` | Fixed-Priority PP, Basic CBS, Cardinal-First CBS |
| `mapf_priority_strategy_execution_main.py` | SPF and LPF |
| `mapf_conflict_aware_priority_execution_main.py` | CDF-H, CDF-L, SPF+CD |
| `mapf_random_priority_execution_main.py` | Random priority (seeded) |
| `mapf_cross_map_priority_execution_main.py` | Cross-map SPF/LPF/CDF/SPF+CD execution |
| `mapf9_primary_execution_main.py` | MAPF-9 production benchmark (SPF, CGLPS, UBLS) |
| `mapf9_primary_analysis_main.py` | Analysis of frozen MAPF-9 execution results |
| `thesis_mapf_figures_main.py` | Regenerate thesis MAPF figures from frozen CSVs |

MAPF-9 primary execution CLI (requires `--map`):

```bash
cd pathfinding/scripts
python mapf9_primary_execution_main.py --map AR0400SR
python mapf9_primary_execution_main.py --map AR0400SR --resume
python mapf9_primary_execution_main.py --map AR0400SR --plan-only
```

Allowed maps: `AR0400SR`, `AR0307SR`. Flags: `--resume`, `--reset-results`, `--plan-only`.

Catalogue generation and validation scripts (`mapf_generate_*_main.py`, `mapf_validate_*_main.py`) produce and verify frozen benchmark manifests under `pathfinding/results/mapf_benchmarks/`.

---

## Reproducibility and experiments

The MAPF evaluation pipeline is designed for reproducible, resumable runs:

- **Deterministic seeds** — catalogue generation, random priority orderings, and obstacle placement use fixed seeds.
- **Frozen manifests** — JSON catalogues under `pathfinding/results/mapf_benchmarks/` define instance sets; SHA-256 checksums are validated before production runs (e.g. MAPF-9).
- **Incremental checkpoints** — long benchmarks append to `results_details.jsonl` and rewrite `results.csv`; rerunning skips completed `(instance, method)` keys (`--resume` on MAPF-9).
- **Frozen analysis outputs** — summary CSVs and markdown reports under `pathfinding/results/mapf*_*/` and thesis figure inputs; analysis scripts read existing execution data without re-running planners.
- **Thesis figures** — `thesis_mapf_figures_main.py` renders PDF/PNG into `docs/thesis/img/` from frozen analysis CSVs only.

Single-agent and dynamic benchmarks export CSV to `Results/` with fixed default seeds (e.g. `obstacle_seed=42` for dynamic runs).

---

## Results directories

| Location | Contents |
|----------|----------|
| `Results/static_algorithms/` | A\* / HPA\* / JPS benchmark CSV |
| `Results/dynamic_algorithms/` | A\* vs D\* Lite dynamic benchmark CSV |
| `Results/plots/` | Generated static and dynamic plots |
| `pathfinding/results/mapf_benchmarks/` | Frozen MAPF catalogue manifests and validation reports |
| `pathfinding/results/mapf*_*/` | MAPF execution CSV/JSONL, analysis summaries, thesis tables |

Example files in repo: `Results/static_algorithms/astar_hpa_jps_cluster_32_scenarios_100.csv`, `pathfinding/results/mapf9_primary_execution/AR0400SR/instance_results.csv`.

---

## Known limitations

- A\* and D\* Lite in dynamic scenarios operate on the **current map state**, not a full time-expanded graph.
- Moving obstacles may require spatio-temporal planning beyond simple replanning.
- MAPF evaluation is scoped to the MovingAI **bg512** map family used in the thesis catalogues.
- CBS runs under expansion/time limits in benchmark harnesses; D\* Lite is research/educational and not heavily optimized.
- HPA\* is not part of the dynamic benchmark.

---

## Additional documentation

| Document | Description |
|----------|-------------|
| [`docs/DYNAMIC_BENCHMARK.md`](docs/DYNAMIC_BENCHMARK.md) | Dynamic benchmark metrics and interpretation |
| [`pathfinding/docs/thesis_single_agent_dynamic_summary.md`](pathfinding/docs/thesis_single_agent_dynamic_summary.md) | Frozen single-agent/dynamic evidence summary |
| [`pathfinding/docs/thesis_evidence_inventory.md`](pathfinding/docs/thesis_evidence_inventory.md) | MAPF thesis evidence inventory |
| [`pathfinding/docs/mapf9_design_freeze.md`](pathfinding/docs/mapf9_design_freeze.md) | MAPF-9 CGLPS/UBLS design specification |

Cursor agent rules: `.cursor/rules/master-thesis.mdc`.
