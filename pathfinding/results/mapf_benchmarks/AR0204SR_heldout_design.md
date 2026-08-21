# MAPF-7.6 — Held-Out Validation Catalogue Design (Frozen)

## Held-out purpose

Independent validation after the MAPF-7 conflict-aware priority family
(CDF-H / CDF-L / SPF+CD) was frozen and evaluated once on the 27-instance
primary catalogue. The held-out set tests whether primary-catalogue findings
generalize to unseen scenario selections without using MAPF-7 strategy outcomes
during catalogue construction.

## What remains identical to primary

| Property | Value |
|---|---|
| Map | `AR0204SR.map` |
| Scenario file | `AR0204SR.map.scen` |
| Agent counts | 5, 10, 20 |
| Factorial structure | 3 instances per `agent_count × {LOW, MEDIUM, HIGH}` cell |
| Total instances | 27 |
| `max_timestep` | 512 |
| `min_reference_length` | 20.0 |
| Interaction definition | Independent-path conflict count strata (LOW=0, MEDIUM=1–2, HIGH≥3) |
| Validity rules | Same MovingAI eligibility, bounded independent static paths, duplicate-scenario rejection |
| Selection algorithm | Deterministic seeded sampling; first valid candidate fills each cell |

## What changes

| Property | Primary | Held-out |
|---|---|---|
| Catalogue role | `primary` (implicit) | `held_out_validation` |
| Seed | 2026 | **2027** |
| Instance IDs | `AR0204SR_n{agents}_{level}_{seq}` | `AR0204SR_HO_n{agents}_{level}_{seq}` |
| Manifest | `AR0204SR_manifest.json` | `AR0204SR_heldout_manifest.json` |
| Scenario sets | Primary 27 selections | Different selections; primary sets excluded |

## Held-out seed

**Seed = 2027**

Fixed as the deterministic successor to primary seed 2026. It is an identifier
only — not selected by inspecting candidate catalogues or MAPF-7 outcomes.

Per-agent RNG seeds use the same rule as primary generation:
`agent_count_seed = base_seed + agent_count`.

## Leakage prevention

During held-out generation the procedure:

1. Loads the frozen primary manifest (`AR0204SR_manifest.json`).
2. Builds the set of primary **unordered scenario-index signatures** per agent count.
3. Rejects any candidate whose signature matches a primary instance for the same agent count.
4. Also rejects duplicate signatures within the held-out catalogue.
5. Validates ordered **(start, goal) agent assignments** do not duplicate primary instances.

Duplicate identity (minimum):

- Unordered scenario-index set + agent count (same as primary generator).
- Ordered `(start_row, start_col, goal_row, goal_col)` per agent when scenarios are available.

The procedure does **not**:

- Use CDF-H / CDF-L / SPF+CD / SPF / Fixed / Random / CBS results.
- Tune strata, seeds, or instance counts from MAPF-7 findings.
- Try multiple seeds and pick a favourable catalogue.

## Candidate selection methodology

Mirrors MAPF-5A primary generation (`mapf_generate_final_catalogue_main.py` /
`generate_benchmark_manifest`):

1. Load `AR0204SR.map` and `AR0204SR.map.scen`.
2. Build the feasible source pool via bounded independent static-path precompute.
3. For each agent count (5, 10, 20):
   - Initialize RNG with seed `2027 + agent_count`.
   - Sample unordered scenario-index sets from the feasible pool.
   - Skip invalid candidates, primary duplicates, and held-out duplicates.
   - Classify by frozen independent-path interaction thresholds.
   - Accept the first valid candidate for each unfilled LOW/MEDIUM/HIGH cell.
4. Stop when all 9 cells × 3 instances are filled (27 total).

Implementation entry point:
`generate_held_out_benchmark_manifest()` in
`pathfinding/src/experiments/mapf_benchmark_instances.py`.

PyCharm runner:
`pathfinding/scripts/mapf_generate_heldout_catalogue_main.py`

## Future evaluation protocol

After this manifest is frozen, the **next** stage runs on held-out instances only:

- CDF-H (primary)
- CDF-L (direction ablation)
- SPF+CD (tie-break ablation)

Do not add a new MAPF-7 heuristic before that held-out test. Baseline comparisons
(SPF, Fixed, CBS, Random) may be added later if needed for context — they were
not part of MAPF-7.6 catalogue design.

## Validation checklist (catalogue construction only)

1. Exactly 27 held-out instances.
2. Exactly 9 instances per agent count {5, 10, 20}.
3. Exactly 3 instances per `agent_count × interaction` cell.
4. Unique held-out instance IDs (`_HO_` prefix).
5. No duplicate held-out scenario sets per agent count.
6. No held-out scenario set duplicates a primary instance.
7. Valid starts/goals under existing rules.
8. Deterministic regeneration with seed 2027 reproduces the same manifest.
9. Primary manifest file unchanged.
10. No MAPF-7 priority strategy executed during generation.
