# MAPF-8 — Walidacja cross-map w rodzinie bg512 (wersja robocza)

## 1. Cel walidacji cross-map

Celem MAPF-8 było sprawdzenie, czy wnioski z wcześniejszych eksperymentów priorytetowych na mapie AR0204SR utrzymują się na innych topologiach z tej samej rodziny scenariuszy MovingAI bg512. Testowano dwie nowe mapy — AR0400SR i AR0307SR — z zamrożonymi katalogami po 27 instancji MAPF każda. Dla każdej instancji wykonano cztery strategie ustalania kolejności priorytetów: SPF, CDF-H, CDF-L oraz SPF+CD. Walidacja dotyczyła wyłącznie poziomu topologii w obrębie bg512, a nie pełnej generalizacji na wszystkie mapy MovingAI.

## 2. Metodologia

Analiza opierała się na zamrożonych wynikach wykonania MAPF-8.4 oraz formalnej analizie MAPF-8.5. Wrażliwość primary definiowano jako niezerowy zakres SoC między wszystkimi czterema strategiami; analogicznie badano wrażliwość makespan. Porównania parami stosowały konwencję diff = LEFT − RIGHT. Dla SPF+CD porównywano kolejność priorytetów ze SPF; wynik traktowano obserwacyjnie, bez twierdzenia o równoważności formalnej. Historyczne katalogi AR0204SR (primary i held-out) włączono do interpretacji łącznej (108 instancji), zachowując rozdzielenie katalogów.

## 3. Wyniki ogólne

W czterech katalogach łącznie zebrano 108 instancji MAPF. Przy definicji czterech strategii wrażliwych na SoC było 12 instancji (11,1%), a wrażliwych na makespan — 3 (2,8%). Wrażliwość pozostawała rzadka we wszystkich czterech badanych katalogach. Jednocześnie maksymalne obserwowane zakresy SoC sięgały 415, 264, 396 i 22 w poszczególnych katalogach, co pokazuje silną zależność wielkości efektu od instancji.

## 4. Sensitivity

Na mapach cross-map odsetek wrażliwości SoC wyniósł 2/27 (7,4%) dla AR0400SR i AR0307SR — wartość zgodna z rzadkością obserwowaną historycznie (5/27 primary, 3/27 held-out). Wszystkie cztery wrażliwe instancje cross-map należały do warstwy interakcji HIGH; nie wolno jednak z tego wnioskować, że HIGH „powoduje” wrażliwość — jest to opis statystyczny na małej próbie.

## 5. CDF-H i CDF-L

Analiza parami wskazuje, że większość instancji ma równy SoC między CDF a SPF; kierunek ewentualnej przewagi nie jest stabilny między katalogami. Na 54 instancjach cross-map CDF-H zmieniał kolejność względem SPF w 36 przypadkach, a CDF-L w 34, podczas gdy różnica SoC wystąpiła odpowiednio tylko w 2 i 3 przypadkach. Zmiana kolejności priorytetów jest więc częstsza niż zmiana jakości końcowego rozwiązania.

## 6. SPF+CD

We wszystkich 108 analizowanych instancjach (54 historycznych AR0204SR oraz 54 cross-map) nie zaobserwowano innego porządku priorytetów SPF+CD względem SPF. Obserwacja ta nie stanowi dowodu formalnej równoważności obu strategii — w badanej próbie tie-break stopnia konfliktu nie zmienił kolejności.

## 7. Struktura konfliktowa

Instancje wrażliwe w puli cross-map charakteryzowały się wyższymi średnimi liczbami zdarzeń konfliktowych, par konfliktowych i maksymalnego stopnia w grafie konfliktów niż instancje niewrażliwe. MAPF-8 traktuje to wyłącznie jako związek opisowy; sam stopień konfliktu nie wystarczał do jednoznacznego rozróżnienia instancji wrażliwych i niewrażliwych w badanej próbie.

## 8. Ograniczenia

Wszystkie mapy należą do rodziny bg512. MAPF-8 dostarcza dowodu walidacji cross-map/topologii w tym ograniczonym zbiorze, a nie walidacji uniwersalnej dla całego zbioru MovingAI ani dla innych domen. Liczby wrażliwych instancji są małe, więc wnioski należy formułować konserwatywnie.

## 9. Wniosek końcowy

MAPF-8 potwierdza, że wrażliwość priorytetów na jakość SoC pozostaje rzadka także na nowych topologiach bg512, przy silnie instancyjnej wielkości efektu. SPF zachowuje rolę praktycznego baseline'u, a CDF-H/CDF-L nie wykazują spójnej przewagi cross-map. Częste zmiany kolejności CDF nie przekładają się proporcjonalnie na zmiany SoC. Wyniki wspierają ostrożną interpretację wyników MAPF-7 w szerszym, lecz nadal lokalnym kontekście topologii bg512.
