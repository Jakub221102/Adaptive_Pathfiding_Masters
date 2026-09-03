# Konfliktowo sterowane lokalne przeszukiwanie kolejności priorytetów

## Motywacja

W MAPF-7 wykazano niestabilność kierunku statycznego porządkowania według stopnia
konfliktu. MAPF-8 uogólnił ten wniosek w rodzinie bg512: zmiany kolejności priorytetów
były częste, natomiast zmiany sumy kosztów (SoC) pozostawały rzadkie. MAPF-9 bada
zatem, czy informacja o konfliktach jest użyteczniejsza jako sygnał **sterowania
przeszukiwaniem** wokół punktu startowego SPF niż jako stała reguła priorytetu.

## Metoda

Punktem odniesienia pozostaje SPF. Dla każdej instancji budowany jest graf konfliktów
na niezależnych ścieżkach; pary agentów rankowane są według liczby zdarzeń konfliktowych.
CGLPS generuje do B = 4 kandydatów przez pojedynczą transpozycję w kolejności SPF,
wybierając pary o najwyższym rankingu konfliktowym. UBLS stanowi dopasowany kontrolnie
deterministyczny odpowiednik z tym samym budżetem A_i, lecz bez rankingu konfliktowego.
Zamrożone kryterium wyboru rozwiązania: sukces → niższy SoC → niższy makespan.

## Zbiór ewaluacyjny

Główny zbiór ewaluacyjny obejmuje **54 świeże instancje** MAPF na mapach AR0400SR (seed 2030) i
AR0307SR (seed 2031), rozłącznych względem katalogów MAPF-8 na tych samych mapach.
Zakres ewaluacji ograniczono do rodziny map MovingAI **bg512**.

## Wyniki

- Sukces wszystkich metod: **54/54**.
- **CGLPS vs SPF:** poprawa SoC **2/54**; wyłączna poprawa makespan **3/54**; łącznie **5** leksykograficznych usprawnień (przy braku odzyskiwania sukcesu).
- **UBLS vs SPF:** poprawa SoC **1/54**; wyłączna poprawa makespan **0/54**.
- **CGLPS vs UBLS (dopasowany A_i):** SoC **1/53/0**; leksykograficznie **4/50/0**.
- Per mapa (CGLPS SoC): AR0400SR **1/27**, AR0307SR **1/27**.

## Koszt obliczeniowy

Logiczne ewaluacje kandydatów PP: SPF **54**, CGLPS **129**, UBLS **129**. Całkowity
logiczny czas wykonania w puli: CGLPS ≈ **1.94×** SPF,
UBLS ≈ **1.93×** SPF. Fizyczny łączny
eksperyment MAPF-9 wykonał **204** ewaluacje PP (oba mapy, wszystkie metody).

## Interpretacja

W badanej próbie zaobserwowano **niewielką, lecz realną** przewagę CGLPS nad UBLS
przy dopasowanym budżecie, przy jednoczesnej **rzadkości** zmian jakości względem SPF.
Jeden wspólny przypadek poprawy SoC (`AR0400SR_n05_high_000`) pokazuje, że sama lokalna
eksploracja może czasem wystarczyć bez rankingu konfliktowego; pozostałe cztery
leksykograficzne przewagi CGLPS nad UBLS wystąpiły na AR0307SR. Wyniki wskazują na
możliwość korzystnego wykorzystania informacji konfliktowej, ale **nie pozwalają**
wnioskować o ogólnej wyższości CGLPS nad SPF ani UBLS poza badaną konfiguracją.

## Ograniczenia

- tylko bg512; dwa mapy; 54 instancje primary;
- zamrożone B = 4 i sąsiedztwo pojedynczej transpozycji;
- brak optymalizacji budżetu; brak permutacji wyczerpujących;
- UBLS z jedną zamrożoną konwencją seed;
- wyłącznie opisy ilościowe; MAPF-7/8 wyłącznie jako kontekst historyczny.

## Wniosek

Wyniki MAPF-9 wskazują, że informacja o konfliktach może być użyteczna do kierowania
ograniczonym lokalnym przeszukiwaniem kolejności priorytetów, lecz w badanej
konfiguracji korzyść jakościowa była rzadka, a koszt obliczeniowy wysoki. Informacja
konfliktowa pełni więc rolę sygnału przeszukiwania, a nie pewnego deterministycznego
usprawnienia priorytetu globalnego.
