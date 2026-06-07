# Master Path v1

**Praca magisterska:** Adaptacyjne wyznaczanie ścieżek dla botów w zróżnicowanych środowiskach z uwzględnieniem dynamicznych warunków i zachowań.

Implementacja, analiza i porównanie algorytmów pathfindingu na mapach siatkowych (MovingAI, Baldur's Gate BG512), z naciskiem na jakość ścieżki, koszt obliczeniowy, skalowalność oraz (w kolejnych etapach) środowiska dynamiczne.

Szczegółowy kontekst projektu i zasady decyzyjne: [`docs/ZASADY_PROJEKTOWE.md`](docs/ZASADY_PROJEKTOWE.md).

---

## Wymagania

- Python 3.13 (zalecane; projekt rozwijany pod PyCharm)
- Zależności: `requirements.txt`

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Katalog główny repozytorium musi być na `PYTHONPATH` (domyślnie w PyCharm: root modułu = repo root).

---

## Struktura repozytorium

```
Master_Path_v1/
├── pathfinding/
│   ├── src/
│   │   ├── algorithms/     # A*, JPS, HPA*
│   │   ├── core/
│   │   ├── experiments/
│   │   ├── loaders/
│   │   ├── utils/
│   │   └── visualization/  # pygame viewer
│   ├── scripts/            # entry pointy benchmarków i wizualizacji
│   ├── plots/              # generowanie wykresów z CSV
│   └── tests/
│       ├── unit/
│       └── regression/
├── Data/                   # mapy i scenariusze (poza git — lokalnie)
├── Results/                # wyniki benchmarków i wykresy (w repo)
├── docs/
│   └── ZASADY_PROJEKTOWE.md
└── pytest.ini
```

**Uwaga:** Mapy MovingAI nie są w repozytorium. Umieść je lokalnie w `Data/` (ścieżki w skryptach wskazują np. `../../Data/...` względem `pathfinding/scripts/`).

---

## Algorytmy (stan aktualny)

| Algorytm | Status | Uwagi |
|----------|--------|--------|
| **A\*** | Stabilny | 8-kierunkowy ruch, heurystyka octile, corner cutting |
| **HPA\*** | Stabilny | Klastry, entrances, graf abstrakcyjny, cache lokalny |
| **JPS** | Stabilny | Koszt zgodny z A\*; bez wariantu bez corner cutting |
| **A\* replanning** | Planowany | — |
| **D\* Lite** | Planowany | — |

---

## Uruchamianie

### Testy

Z katalogu głównego repozytorium:

```bash
pytest
```

Konfiguracja: `pytest.ini` → `pathfinding/tests`.

Regresja używa małego scenariusza MovingAI; **nie** sprawdza czasów wykonania.

### Wizualizacja (pojedynczy scenariusz)

```bash
cd pathfinding/scripts
python main.py
```

Tryby viewer: `STATIC`, `ANIMATED`, `COMPARISON` — w `ExperimentConfig` / `pygame_models.ViewerMode`.

### Benchmark statyczny (A\* / HPA\* / JPS)

```bash
cd pathfinding/scripts
python benchmark_main.py
```

### Porównanie i eksperymenty HPA\*

```bash
python comparison_main.py
python cluster_size_benchmark_main.py
python max_entrances_benchmark_main.py
python run_failed_scenario_analysis.py
```

Wyniki CSV trafiają do `Results/` (ścieżki w `ExperimentConfig.results_dir`).

### Wykresy

```bash
cd pathfinding/plots
python plot_static_algorithms_results.py
python plot_cluster_size_results.py
python plot_max_entrances_results.py
```

---

## Metryki i benchmarki

`BenchmarkExperiment` zbiera m.in.: `found`, `execution_time_ms`, `path_length`, `path_cost`, `visited_nodes`; dla HPA\* także preprocessing/query/cache; dla JPS `scanned_nodes`.

Szczegóły metryk HPA\* i założenia kosztów — w [`docs/ZASADY_PROJEKTOWE.md`](docs/ZASADY_PROJEKTOWE.md).

---

## Kolejne etapy (roadmap)

1. Dynamic obstacles — dodawanie/usuwanie przeszkód w runtime
2. A\* replanning
3. D\* Lite
4. Analiza dynamiczna: A\* replanning vs D\* Lite vs HPA\*

---

## Cursor / AI

Reguły projektu dla asystenta: `.cursor/rules/master-thesis.mdc` (skrót zasad i ograniczeń).
