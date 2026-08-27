# MAPF-8 Final Figure Captions

## Figure 1 — `fig01_soc_sensitivity_across_catalogues`

**English caption:** Four-strategy SoC sensitivity rate for the four independent 27-instance catalogues (AR0204SR primary, AR0204SR held-out, AR0400SR, AR0307SR). Bars show the percentage of instances with non-zero SoC range across SPF, CDF-H, CDF-L, and SPF+CD; exact counts are annotated.

**Polish caption:** Odsetek instancji wrażliwych na SoC w definicji czterech strategii priorytetu (SPF, CDF-H, CDF-L, SPF+CD) dla czterech niezależnych katalogów po 27 instancji. Słupki przedstawiają udział procentowy; wartości dokładne podano w adnotacjach.

**Interpretacja (PL):** Rysunek pokazuje, że wrażliwość jakościowa na wybór strategii priorytetu pozostaje rzadka we wszystkich czterech katalogach (od 7,4% do 18,5%). Nie wskazuje to na uniwersalną dominację jednej strategii, lecz na rzadkie, zależne od instancji efekty. Porównanie obejmuje wyłącznie cztery właściwe katalogi; pulę cross-map traktuje się osobno.

---

## Figure 2 — `fig02_soc_on_all_sensitive_instances`

**English caption:** Horizontal bar chart of four-strategy SoC range (max − min across SPF, CDF-H, CDF-L, SPF+CD) for all twelve sensitive instances across the four catalogues (5 + 3 + 2 + 2). Instances are sorted by range descending; labels use short catalogue codes (AR0204-P, AR0204-H, AR0400, AR0307) and instance identifiers.

**Polish caption:** Poziomy wykres słupkowy zakresu SoC w definicji czterech strategii (max − min dla SPF, CDF-H, CDF-L, SPF+CD) dla wszystkich dwunastu instancji wrażliwych w czterech katalogach (5 + 3 + 2 + 2). Instancje posortowano malejąco według zakresu; etykiety używają skrótów katalogów (AR0204-P, AR0204-H, AR0400, AR0307) oraz identyfikatorów instancji.

**Interpretacja (PL):** Rysunek pokazuje wielkość wrażliwości jakościowej: efekty mogą być bardzo duże (415, 396, 264) albo minimalne (1–2). Szczegółowe wartości SoC poszczególnych strategii podano w Tabeli D. Nie wolno z tego wnioskować o stałej przewadze strategii — wielkość efektu jest silnie zależna od instancji w obrębie rodziny bg512.

---

## Figure 3 — `fig03_cdf_vs_spf_across_catalogues`

**English caption:** Stacked instance counts for CDF-H vs SPF and CDF-L vs SPF by catalogue, using diff = CDF − SPF (negative = CDF better, zero = equal, positive = SPF better). Each catalogue contributes 27 instances.

**Polish caption:** Skumulowana liczba instancji dla CDF-H vs SPF oraz CDF-L vs SPF w podziale na katalogi; konwencja: diff = CDF − SPF (ujemna = CDF lepszy, zero = równy, dodatnia = SPF lepszy). Każdy katalog obejmuje 27 instancji.

**Interpretacja (PL):** We wszystkich katalogach dominuje klasa „equal”, a kierunek przewagi CDF nie jest stabilny między mapami. Pojedyncze odstępstwa mogą silnie wpływać na średnie, dlatego rysunek oparto na liczbach instancji, a nie wyłącznie na średnich różnicach SoC. Nie jest to dowód skuteczności reguły konfliktowej w sensie przyczynowym.

---

## Figure 4 — `fig04_order_change_vs_soc_change`

**English caption:** Cross-map instances only (n = 54): comparison of priority order changes vs SoC changes relative to SPF for CDF-H and CDF-L. Order changes are frequent; final SoC changes are rare.

**Polish caption:** Wyłącznie instancje cross-map (n = 54): porównanie zmian kolejności priorytetów i zmian SoC względem SPF dla CDF-H i CDF-L. Zmiany kolejności są częste; zmiany końcowego SoC — rzadkie.

**Interpretacja (PL):** Rysunek pokazuje kluczowy wynik MAPF-8: modyfikacja kolejności priorytetów jest częsta (36/54 dla CDF-H, 34/54 dla CDF-L), podczas gdy zmiana jakości rozwiązania (SoC) występuje tylko w 2–3 przypadkach na 54. Oznacza to, że zmiana kolejności nie implikuje automatycznie zmiany kosztu końcowego. Nie wolno stąd wnioskować o przyczynowości ani o równoważności strategii.
