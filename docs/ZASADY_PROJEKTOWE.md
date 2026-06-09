# Zasady projektowe — Master Path v1

Dokument kontekstu pracy magisterskiej. Używaj go jako źródła prawdy przy implementacji, benchmarkach i pisaniu rozdziałów.

---

## Cel projektu

Implementacja, analiza i porównanie algorytmów wyznaczania ścieżek dla agentów w środowiskach siatkowych (grid maps), ze szczególnym uwzględnieniem:

- jakości wyznaczanych ścieżek,
- kosztu obliczeniowego,
- skalowalności,
- zachowania w środowiskach dynamicznych,
- możliwości szybkiego przeliczania tras po zmianach otoczenia.

Benchmarki: **MovingAI** oraz mapy **Baldur's Gate (BG512)**.

---

## Aktualny stan — algorytmy statyczne

### A*

- Ruch 8-kierunkowy, heurystyka octile, koszty diagonalne.
- Trace kroków pod wizualizację.
- Metryki: `execution_time_ms`, `visited_nodes`, `path_length`, `path_cost`.
- **Stan:** stabilny; testy jednostkowe i regresyjne.

### HPA* (Hierarchical Pathfinding A*)

- Podział na klastry, entrances, graf abstrakcyjny, lokalny cache, wyszukiwanie abstrakcyjne, refinacja.
- Dodatkowe statystyki preprocessingu i zapytania (m.in. `cluster_count`, `entrance_count`, `graph_density`, `cache_hit_ratio`, `abstract_search_time_ms`, `refinement_time_ms`, `query_time_ms`).
- **Stan:** stabilny; testy jednostkowe.

### JPS (Jump Point Search)

- Ruch 8-kierunkowy; **corner cutting dozwolony**.
- Koszt ścieżki **zgodny z A\***.
- Statystyki: `visited_nodes`, `scanned_nodes`.
- **Stan:** działa na benchmarkach; **brak** wariantu bez corner cutting.

---

## Benchmarki i eksperymenty

### BenchmarkExperiment

- Scenariusze MovingAI (oraz konfiguracje BG512 w skryptach).
- Metryki wspólne: `found`, `execution_time_ms`, `path_length`, `path_cost`, `visited_nodes`.
- HPA*: m.in. `preprocessing_time_ms`, `query_time_ms`, `cache_hit_ratio`, `abstract_nodes_visited`.
- JPS: `scanned_nodes`.

### Wyniki statyczne

Porównywane: **A\***, **HPA\***, **JPS**.

Analiza: execution time, path cost, cost error, visited/scanned nodes, preprocessing cost.

### Eksperymenty HPA* (zrealizowane)

| Parametr | Wartości badane |
|----------|-----------------|
| `cluster_size` | 8, 16, 32, 64, 128 |
| `max_entrances_per_cluster_pair` | 1, 2, 4, 8, 16 |

Analizowane: query/preprocessing time, path quality, found rate, abstract nodes visited, total cost.

---

## Wizualizacja (pygame)

- Mapy, ścieżki, animacja algorytmów.
- Overlay: klastry HPA*, wydajność, heatmapy odwiedzeń.
- Tryby: `STATIC`, `ANIMATED`, `COMPARISON`.

---

## Testy

### Unit

- A*: prosta ścieżka, diagonalna, brak ścieżki.
- Path cost: poziomy, pionowy, diagonalny.
- JPS: zgodność kosztu z A*, brak ścieżki.
- HPA*: poprawne wyszukiwanie i statystyki preprocessingu.
- D\* Lite: podstawowe replanning i aktualizacja kosztów.
- DynamicGridMap, moving obstacle generator, kolizje ruchomych przeszkód.
- Dynamic benchmark: eksport CSV, wybór scenariuszy.

### Regression (MovingAI, mały scenariusz)

- Znalezienie ścieżki.
- Zgodność kosztu A\* i JPS z A\*.
- Ograniczenie degradacji jakości HPA\*.
- Limity `visited_nodes`.
- Dynamic replanning: A\* i D\* Lite na scenariuszach z przeszkodami.
- **Nie testujemy** czasów wykonania w regresji.

---

## Ważne decyzje projektowe

1. **A\* i JPS** — corner cutting **włączony** (obecna semantyka kosztu).
2. **HPA\*** — optymalizacja i porównanie po **koszcie ścieżki** (`path_cost`), nie po liczbie kroków.
3. Benchmarki oparte o format **MovingAI** (loadery w `pathfinding/src/loaders/`).
4. **Wyniki benchmarków i wykresy** — przechowywane w repozytorium (`Results/`).
5. **Mapy MovingAI** — **nie** w repozytorium; dostarczane lokalnie w `Data/`.
6. Nowe algorytmy dynamiczne powinny **re-używać** istniejące API (`base.py`, `ExperimentConfig`, rejestr algorytmów) i metryki spójne z benchmarkami statycznymi + metryki reakcji na zmianę mapy.
7. Zmiany nie mogą łamać regresji kosztu ścieżki bez świadomej aktualizacji testów i dokumentacji.
8. Przy refaktorze HPA\* zachować rozdzielenie: `cluster_builder`, `entrance_detector`, `abstract_graph_builder`, `abstract_search`, `local_path_cache`.

---

## Stan — środowisko dynamiczne (zrealizowane)

### Modele i symulacja

- `DynamicGridMap` — nakładka na `GridMap` z `dynamic_blocked` / `dynamic_unblocked`.
- `DynamicObstacleEvent` — zdarzenia `BLOCK` / `UNBLOCK` w krokach symulacji.
- `MovingObstacle` — prostokątne przeszkody ruchome (odbicia od ścian).
- `moving_obstacle_generator` — deterministyczna generacja (seed), `min_obstacle_spacing`, wykluczenie start/cel.
- `dynamic_simulation.py` — replanning z lookahead, waiting policy, collision policy (`PUSH_AGENT` / `BLOCK_OBSTACLE`), predykcja przeszkód.

Logika kolizji i oczekiwania agenta jest w **warstwie symulacji**, nie w A\* / D\* Lite.

### Algorytmy dynamiczne

| Algorytm | Status |
|----------|--------|
| **A\* replanning** | Stabilny — pełne przeliczenie przy blokadzie ścieżki |
| **D\* Lite** | Stabilny — replanning przyrostowy; implementacja edukacyjna |

### Benchmark dynamiczny

- `DynamicBenchmarkExperiment`: A\* vs D\* Lite, wspólny setup przeszkód per scenariusz.
- CSV: `Results/dynamic_algorithms/`; wykresy: `pathfinding/plots/plot_dynamic_benchmark_results.py`.
- Szczegóły metryk: [`DYNAMIC_BENCHMARK.md`](DYNAMIC_BENCHMARK.md).

### Kolejne etapy (roadmap)

| Kolejność | Temat |
|-----------|--------|
| 1 | Benchmark dynamiczny na wielu seedach / mapach |
| 2 | Lepsza predykcja czasowa przeszkód |
| 3 | HPA\* w benchmarku dynamicznym |
| 4 | MAPF / CBS / reservation tables |

---

## Konwencje implementacyjne

- Python, `pydantic` w konfiguracji eksperymentów (`ExperimentConfig`).
- Importy: pakiet `pathfinding` z roota repozytorium.
- Skrypty w `pathfinding/scripts/` — ścieżki do `Data/` i `Results/` często względne (`../../Data/...`).
- Nie dodawać zbędnych warstw abstrakcji; rozszerzać istniejące moduły.
- Komentarze tylko tam, gdzie logika biznesowa lub algorytmiczna nie jest oczywista z kodu.

---

## Pliki kluczowe

| Obszar | Ścieżki |
|--------|---------|
| Algorytmy | `pathfinding/src/algorithms/` |
| HPA\* | `pathfinding/src/algorithms/hpa/` |
| Eksperymenty | `pathfinding/src/experiments/` |
| Benchmarki statyczne | `pathfinding/scripts/benchmark_main.py`, `benchmark_experiment.py` |
| Benchmark dynamiczny | `pathfinding/scripts/dynamic_benchmark_main.py`, `dynamic_benchmark_experiment.py` |
| Symulacja dynamiczna | `pathfinding/src/experiments/dynamic_simulation.py` |
| Testy | `pathfinding/tests/unit/`, `pathfinding/tests/regression/` |
| Wykresy | `pathfinding/plots/` |
| Dokumentacja dynamiczna | `docs/DYNAMIC_BENCHMARK.md` |
