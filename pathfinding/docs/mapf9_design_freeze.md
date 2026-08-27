# MAPF-9.0 Design Freeze — Conflict-Guided Bounded Local Priority Search (CGLPS)

**Status:** Frozen (MAPF-9.0a)  
**Date:** 2026-08-27  
**Scope:** Design only. No implementation, planner execution, or benchmark in this stage.

This document is the authoritative MAPF-9 design reference. MAPF-8 and MAPF-7
results, manifests, and final analysis artifacts are frozen and must not be modified.

---

## 1. Scientific context

MAPF-8 (closed) established that across 108 bg512 instances:

- Four-strategy SoC sensitivity: **12/108** instances.
- Static conflict-degree rules (CDF-H, CDF-L) frequently changed priority order
  relative to SPF but rarely changed final SoC (cross-map: 36/54 and 34/54 order
  changes vs 2/54 and 3/54 SoC changes).
- SPF remained a stable practical baseline; SPF+CD produced identical ordering
  to SPF on all 108 tested instances.
- When sensitivity occurred, magnitude could be large (max observed SoC ranges
  include 415 and 396).

MAPF-9 tests whether a **small, bounded search** over **conflict-guided local
modifications** of the SPF baseline can find better priority orders — without
becoming a large metaheuristic project.

**Limitation:** All evaluation remains within the MovingAI **bg512** arena family.
No universal MovingAI or cross-domain claim.

**Development vs evaluation evidence:** The historical 108 MAPF-7/8 instances
motivated MAPF-9 but must **not** serve as the primary confirmation set. They
may be used only as **secondary retrospective evidence** in MAPF-9.6.

---

## 2. Algorithm name

**Conflict-Guided Bounded Local Priority Search (CGLPS)**

Abbreviation: **CGLPS**.

Control baseline: **Unguided Bounded Local Search (UBLS)**.

---

## 3. Frozen solver stack (unchanged)

- Grid MAPF, 4-connected movement + WAIT.
- Space-Time A* low-level planning.
- Fixed-priority Prioritized Planning (PP).
- Sum of Costs (SoC) primary quality metric; makespan secondary.
- Vertex and edge/swap conflicts on independent paths.
- Deterministic experiment infrastructure.

Do **not** modify: SPF, CDF-H, CDF-L, SPF+CD, PP, Space-Time A*, MAPF-7/8
artifacts.

---

## 4. Initial ordering — SPF baseline

**Frozen baseline:** SPF using the existing MAPF-7/8 definition:

```
sort key = (independent_path_cost, original_scenario_index) ascending
```

Do not redefine SPF.

**Rationale (not universal optimality):** MAPF-8 showed SPF matched the best
observed SoC on most instances; CDF variants changed order frequently without
stable cross-map quality advantage; SPF+CD was observational identical to SPF
on 108/108 instances. SPF is the appropriate **starting point** for a bounded
local search, not a claim of global optimality.

---

## 5. Single-neighborhood search around SPF

CGLPS is **exactly one bounded neighborhood** around the SPF ordering.

Formally, for each instance:

```
O_0 = SPF order
```

Every non-baseline candidate is:

```
O_k = transposition(O_0, agent_a, agent_b)
```

for exactly **one** selected agent pair `(agent_a, agent_b)`.

**Forbidden:**

- Generating candidates from previously improved candidates.
- Iterative hill climbing, greedy descent, or any multi-step local search.
- Recomputing the conflict graph after PP candidate evaluation.
- Expanding a second search neighborhood.

MAPF-9 tests one bounded neighborhood around SPF, not iterative refinement.

---

## 6. Agent-ID vs original-index semantics (correctness-critical)

Conflict objects identify agents by **`agent_id`**. Original scenario indices
and SPF priority positions are **different coordinate systems**. Both must be
handled explicitly.

### 6.1 Mapping

Build an explicit mapping before ranking or swapping:

```
original_index : agent_id → int    // position in input MAPFScenario
agent_id       : original_index → int
```

Use the scenario's agent list order as the original index (0 … n−1).

### 6.2 Canonical pair key (ranking only)

For any agent pair `(agent_a, agent_b)`:

```
i = original_index(agent_a)
j = original_index(agent_b)
canonical_pair = (min(i, j), max(i, j))
```

Pair conflict-event counts and edge ranking use this **canonical original-index
pair**. Tie-breaking on pairs uses lexicographic order on `(min, max)`.

### 6.3 Swap operation (agent identities in SPF order)

The transposition operates on **agent identities**, not on original indices as
SPF positions:

```
O_0 = [agent_id at SPF rank 0, agent_id at SPF rank 1, …]

pos_a = index of agent_a in O_0
pos_b = index of agent_b in O_0
swap positions pos_a and pos_b in O_0
```

**Do not** interpret original scenario indices `(i, j)` as positions in the
SPF order after SPF sorting.

---

## 7. Local move operator

**Frozen operator:** pair transposition swap (Section 5–6).

For each selected conflicting pair `(agent_a, agent_b)`:

1. Start from `O_0` (SPF order).
2. Locate `agent_a` and `agent_b` by identity in `O_0`.
3. Swap their positions; all other agents unchanged.

Properties: deterministic, simple, compatible with existing PP invocation,
scientifically defensible as a minimal single perturbation of SPF.

---

## 8. Conflict-guided candidate set

Candidates originate **only** from agent pairs that conflict on independent
unconstrained Space-Time A* paths, using existing MAPF-7 conflict semantics:

- Vertex conflicts.
- Edge/swap conflicts.
- One conflict-graph edge per conflicting pair (canonical original-index pair).

For each conflict-graph edge, generate **at most one** swap candidate derived
from `O_0`.

**Must not use:** CBS cardinality, PP outcome information, Random K oracle
results, historical best orderings, catalogue interaction labels, or any
information computed after PP evaluation begins.

All candidate generation is computable before any PP candidate evaluation.

### 8.1 Pair conflict-event ranking

When `|edges| > B`, rank edges deterministically:

```
rank_key(edge (agent_a, agent_b)) =
    (-pair_event_count[canonical_pair], min(i,j), max(i,j))
```

Higher independent-path **conflict-event count for the pair** first (vertex +
edge events between exactly those two agents).

**Implementation note (MAPF-9.1):** Add `pair_conflict_event_counts(scenario,
conflicts)` in `priority_ordering.py` — a small helper over existing
`inputs.conflicts`. Per-pair multiplicity is not currently exposed; this is the
minimum required addition.

### 8.2 Candidate deduplication

- Evaluate only **unique** candidate orderings.
- If a swap produces an order already in the candidate set, skip (no PP).
- `B` is a **maximum**, not a required count.

---

## 9. Evaluation budget

**Primary budget (frozen):**

```
B = 4
```

Meaning: at most **four additional** PP candidate evaluations beyond the SPF
baseline per instance.

- SPF baseline = candidate 0.
- `0 ≤ A_i ≤ B` where `A_i` is the actual number of unique additional CGLPS
  candidates evaluated on instance `i`.

**No conditional B=5 test.** MAPF-9.2 smoke tests are **correctness-only**.
No primary B sweep. Any future budget robustness analysis, if ever performed
after the frozen primary experiment, must be separately labelled **secondary**
and is outside the current MAPF-9 roadmap.

---

## 10. Selection objective

Among evaluated candidates (SPF baseline + up to `B` additional), select
**lexicographically**:

1. Successful PP solution preferred over failure.
2. Minimum SoC.
3. Minimum makespan.
4. Lowest candidate index (SPF = candidate 0 wins exact ties).

**Guarantee:** When SPF succeeds, CGLPS cannot return a worse selected result
under this objective — SPF is always evaluated and tie-break favours it.

**Runtime is not** a solution-quality tie-break.

---

## 11. Edge-case behaviour

| Case | Behaviour |
|---|---|
| Zero conflict edges | Only `O_0` evaluated; `A_i = 0` |
| `\|edges\| < B` | One swap per edge (after dedup); no padding |
| `\|edges\| > B` | Top-B edges by ranking; rest ignored |
| Duplicate candidate orders | Evaluate each unique order once |
| Candidate PP fails | Record failure; not selected if any successful candidate exists |
| SPF fails, candidate succeeds | Candidate may be selected (success criterion #1) |
| All candidates fail | Return failure |
| Ordering failure (no independent path) | `termination_reason = ordering_failure`; no PP |

---

## 12. UBLS control baseline

**Unguided Bounded Local Search (UBLS)** — single computation-matched control.

| Dimension | CGLPS | UBLS |
|---|---|---|
| Initial order | SPF (`O_0`) | SPF (`O_0`) |
| Move operator | Pair transposition | Same |
| Additional PP count | `A_i` | **Exactly `A_i`** |
| Pair selection | Conflict-edge ranking | No conflict information |
| Selection objective | Lexicographic (Section 10) | Same |

This is a **local-search computational control**, not a reopening of the
historical Random K priority-order study.

### 12.1 UBLS pair selection

After CGLPS completes instance `i`, record `A_i`. UBLS uses the same swap
operator on `O_0` but selects `A_i` agent pairs **without** conflict guidance.

**Global constant (frozen):**

```
MAPF9_UBLS_GLOBAL_SEED = 9031
```

**Per-instance seed derivation (frozen):**

```python
text = f"{MAPF9_UBLS_GLOBAL_SEED}:{instance_id}"
digest = sha256(text.encode("utf-8")).digest()
seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
rng = random.Random(seed)
```

**Never** use Python's built-in `hash(...)` for experimental seeding.

**Pair pool:** all unordered agent pairs from original scenario indices,
canonically listed as `(min(i,j), max(i,j))`, shuffled deterministically by
`rng`. Iterate shuffled pairs; apply transposition swap on `O_0`; accept
unique orders until `A_i` additional candidates are collected or pairs are
exhausted.

UBLS must not read CGLPS **outcomes** (SoC, makespan, success); only `A_i`.

---

## 13. PP evaluation accounting

**SPF baseline PP is evaluated exactly once per instance** and reused by both
CGLPS and UBLS.

Let:

```
A_i = number of unique additional CGLPS candidates evaluated on instance i
      (0 ≤ A_i ≤ 4)
```

UBLS receives exactly `A_i` additional evaluations on the same instance.

**Physical PP evaluation count over 54 fresh instances:**

```
N_PP = 54 + 2 × Σ A_i
```

| Bound | Value |
|---|---:|
| Minimum (`A_i = 0` ∀i) | **54** |
| Maximum (`A_i = 4` ∀i) | **54 + 2×(54×4) = 486** |

Do **not** independently re-evaluate SPF inside CGLPS and UBLS.

Expected count (design estimate only, based on MAPF-8 cross-map conflict
structure: median pair count ≈ 1, 9/27 zero-pair instances per map):

```
Σ A_i ≈ 80–110   →   N_PP ≈ 210–270   (approximate; not a frozen bound)
```

---

## 14. Preprocessing and runtime accounting

Independent-path preprocessing was expensive in MAPF-8 and must be accounted
for separately from PP evaluation count. **PP count alone is not complete
wall-clock cost.**

### 14.1 Timing components (benchmark harness)

The MAPF-9.4 harness should record:

| Component | Description |
|---|---|
| `independent_path_time` | Space-Time A* for all agents, unconstrained |
| `spf_ordering_time` | SPF sort from independent costs |
| `baseline_spf_pp_time` | Single shared SPF PP run |
| `conflict_detection_and_ranking_time` | Conflict detect + pair counts + edge ranking |
| `cglps_additional_pp_time` | Sum of additional CGLPS PP runs |
| `ubls_candidate_generation_time` | UBLS pair shuffling and order construction |
| `ubls_additional_pp_time` | Sum of additional UBLS PP runs |

Shared independent-path information may be computed once and reused; method-level
logical totals must remain reconstructable.

### 14.2 Logical method totals

```
SPF logical total =
    independent-path/SPF-ordering work
    + baseline SPF PP

CGLPS logical total =
    SPF logical total
    + conflict-guidance/ranking work
    + CGLPS additional PP evaluations

UBLS logical total =
    SPF logical total
    + UBLS candidate-generation work
    + UBLS additional PP evaluations
```

Do **not** count the common SPF PP multiple times merely because three methods
are compared in analysis.

---

## 15. Primary comparisons and metrics

**Main comparison:** SPF vs CGLPS vs UBLS.

### 15.1 Primary metrics

- Success rate.
- SoC (on common-success instances).

### 15.2 Secondary metrics

- Makespan.
- Number of PP candidate evaluations (`1 + A_i` per method per instance;
  physical total `N_PP`).
- Wall-clock runtime (per component and logical totals).

### 15.3 Derived metrics

- Instances improved vs SPF / equal / (see Section 16 for failure cases).
- SoC improvement on common-success instances: `SPF_SOC − method_SOC`
  (positive = method improves).
- Best, mean, median improvement.
- CGLPS vs UBLS under matched `A_i`.
- Instances where CGLPS improves and UBLS does not.

### 15.4 SPF success guarantee

When SPF succeeds, a successful CGLPS/UBLS instance **cannot** produce a worse
selected result under the frozen lexicographic objective. The "worse than SPF"
SoC count on successful-SPF instances is structurally **zero**.

---

## 16. Analysis semantics — SPF failure (MAPF-9.6)

### 16.1 When SPF succeeds

Quality comparison uses:

```
ΔSoC = SPF_SOC − method_SOC
```

- Positive = method improves SoC.
- Zero = equal.

### 16.2 When SPF fails

Do **not** invent an SoC difference. Record separately:

- SPF failure **recovered** by CGLPS.
- SPF failure **recovered** by UBLS.
- **Unrecovered** failure (both fail).

Success recovery is a **separate primary outcome** from SoC improvement on
common-success instances.

---

## 17. Fresh evaluation catalogue

### 17.1 Primary set

| Property | Value |
|---|---|
| Maps | AR0400SR, AR0307SR (bg512) |
| Instances | 2 × 27 = **54** |
| Structure | agent_count ∈ {5, 10, 20} × interaction ∈ {LOW, MEDIUM, HIGH} × 3/cell |
| Generation | Existing `run_benchmark_catalogue_generation` infrastructure |
| Timing | MAPF-9.3, after this design freeze |

### 17.2 Frozen seeds

| Map | Seed |
|---|---:|
| AR0400SR | **2030** |
| AR0307SR | **2031** |

Existing seeds (must not reuse): 2026 (AR0204 primary), 2027 (held-out),
2028 (MAPF-8 AR0400), 2029 (MAPF-8 AR0307).

### 17.3 Manifest names

| File | Role |
|---|---|
| `AR0400SR_mapf9_manifest.json` | MAPF-9 evaluation |
| `AR0307SR_mapf9_manifest.json` | MAPF-9 evaluation |

Do **not** overwrite MAPF-8 manifests (`AR0400SR_manifest.json`,
`AR0307SR_manifest.json`).

### 17.4 Cross-catalogue disjointness (MAPF-9.3 requirement)

A new seed alone is **not** sufficient evidence of a fresh evaluation set.
MAPF-9.3 must explicitly verify scenario-set signature disjointness from
frozen MAPF-8 catalogues on the same map.

Use the existing deterministic convention:

```python
benchmark_scenario_set_signature(scenario_indices)
    = tuple(sorted(scenario_indices))
```

**Required:**

```
signatures(MAPF-9 AR0400SR) ∩ signatures(MAPF-8 AR0400SR) = ∅
signatures(MAPF-9 AR0307SR) ∩ signatures(MAPF-8 AR0307SR) = ∅
```

Also preserve within-manifest duplicate rejection. Fail generation or validation
**loudly** if cross-catalogue overlap exists.

No CGLPS/UBLS outcome may influence catalogue selection.

### 17.5 Historical 108 instances

Secondary retrospective evidence only (MAPF-9.6). Not primary confirmation.

---

## 18. Computational cost summary

Based on MAPF-8 cross-map runtime evidence (same maps, SPF and conflict-aware
ordering):

| Reference | SPF median total | SPF max total |
|---|---:|---:|
| AR0400SR (27 inst.) | ~59 s | ~379 s |
| AR0307SR (27 inst.) | ~111 s | ~239 s |

**PP evaluations (54 fresh instances, frozen bounds):**

| | Count |
|---|---:|
| Minimum | **54** |
| Maximum | **486** |
| Expected (estimate) | ~210–270 |

**Wall-clock (design estimate):** ~6–12 h total for SPF + CGLPS + UBLS with
checkpoint/resume; feasible for manual overnight/local run. Worst single
instances (n=20 HIGH): up to ~15–30 min when `A_i = 4`.

Independent-path preprocessing adds substantial overhead beyond PP count; see
Section 14.

---

## 19. Research questions (frozen)

### RQ9.1 — Primary efficacy

On the fresh 54-instance bg512 evaluation set, can CGLPS improve solution
quality or recover successful solutions relative to SPF?

### RQ9.2 — Guidance value

Under the same actual additional PP-evaluation budget per instance, does
conflict-guided CGLPS outperform unguided UBLS?

### RQ9.3 — Conditions

Are observed CGLPS improvements descriptively associated with independent
conflict structure, interaction stratum, or agent count?

No causal or predictive claims.

### RQ9.4 — Cost-benefit

What PP-evaluation and wall-clock overhead does CGLPS require relative to SPF,
and how does this compare with observed solution-quality gains?

---

## 20. Thesis contribution

### Positive outcome

A controlled comparison of conflict-guided and unguided bounded local priority
search under a matched evaluation budget, demonstrating that small
conflict-targeted transpositions around SPF can improve PP quality without a
new MAPF solver or large metaheuristic.

### Negative/null outcome

Equally valid: documents that even conflict-guided bounded local search does
not reliably improve over SPF in bg512, refining MAPF-8's observation that
priority-order changes rarely translate to quality changes — now tested with an
explicit search budget and matched unguided control.

Do **not** claim this is the first such controlled test in the literature unless
a dedicated literature review establishes that.

Do **not** frame MAPF-9 as guaranteed to beat SPF.

---

## 21. CGLPS algorithm (pseudocode)

```
CGLPS(instance, B=4):

  // --- Preprocessing (shared with SPF logical total) ---
  inputs ← build_conflict_aware_ordering_inputs(grid_map, scenario, max_timestep)
  id_to_index, index_to_id ← agent/original-index maps from scenario

  O_0 ← SPF order as tuple of agent_ids
      (sort agents by (independent_cost, original_index) ascending)

  // --- Conflict-guided candidate generation (single neighborhood) ---
  edges ← conflict_graph_edges(scenario, inputs.conflicts)
  pair_counts ← pair_conflict_event_counts(scenario, inputs.conflicts)

  ranked_edges ← sort edges by (-pair_counts[canonical_pair(i,j)], i, j)
                  where (i,j) = canonical original-index pair

  candidates ← [(0, O_0)]
  seen ← {O_0}
  additional ← 0

  for (agent_a, agent_b) in ranked_edges mapped via index_to_id:
      if additional == B: break
      O_swap ← transposition(O_0, agent_a, agent_b)   // swap by agent identity
      if O_swap ∉ seen:
          append (len(candidates), O_swap) to candidates
          seen.add(O_swap)
          additional += 1

  A_i ← additional

  // --- PP evaluation (SPF PP run once, reused) ---
  baseline ← plan_prioritized(O_0)          // shared; not re-run for UBLS

  results ← [(0, O_0, baseline)]
  for (k, O_k) in candidates[1:]:
      results.append((k, O_k, plan_prioritized(O_k)))

  selected ← lexicographic_min(results, key=success, soc, makespan, k)
  return selected, A_i, diagnostics
```

No iterative expansion. No recomputation of conflicts after PP.

---

## 22. Roadmap

| Stage | Deliverable |
|---|---|
| **MAPF-9.0 / 9.0a** | Design freeze (this document) |
| **MAPF-9.1** | Core CGLPS + UBLS + `pair_conflict_event_counts` |
| **MAPF-9.2** | Unit tests + correctness-only smoke (no production benchmark) |
| **MAPF-9.3** | Generate/freeze MAPF-9 catalogues; verify MAPF-8 disjointness |
| **MAPF-9.4** | Benchmark harness with timing components and PP accounting |
| **MAPF-9.5** | Manual production benchmark (54 × SPF/CGLPS/UBLS) |
| **MAPF-9.6** | Formal analysis + secondary retrospective on historical 108 |
| **MAPF-9.7** | Final thesis artifacts; MAPF-9 close |

---

## 23. Risks

1. **Motivation–evaluation overlap:** Fresh 54-instance set is essential;
   retrospective 108 is exploratory only.
2. **Sparse sensitivity:** Low power to detect improvements in 54 instances;
   report counts and effect sizes.
3. **bg512-only:** Same scope limit as MAPF-8.
4. **Transposition-only neighbourhood:** Null results may reflect operator
   limits, not uselessness of conflict information in general.
5. **UBLS sequential dependency:** `A_i` must be passed without leaking CGLPS
   quality outcomes.

---

## 24. Confirmations (MAPF-9.0a)

- [x] CGLPS single-neighborhood semantics frozen.
- [x] Agent-ID vs original-index semantics frozen.
- [x] B = 4 frozen; no conditional B=5.
- [x] PP bounds: minimum 54, maximum 486.
- [x] SPF PP evaluated once per instance, shared.
- [x] Runtime accounting components defined.
- [x] Catalogue seeds 2030/2031; cross-catalogue disjointness required.
- [x] UBLS seed derivation frozen (SHA-256, not `hash()`).
- [x] RQ9.1–RQ9.4 wording frozen.
- [x] SPF failure analysis semantics frozen.
- [x] No CGLPS/UBLS implementation in this stage.
- [x] No planner or benchmark execution in this stage.
- [x] No manifest generated in this stage.
- [x] MAPF-7/8 artifacts untouched.
