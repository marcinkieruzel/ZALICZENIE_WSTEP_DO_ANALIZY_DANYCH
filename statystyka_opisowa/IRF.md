# Impulse Response Function (IRF) - Funkcja Reakcji na Impuls

## Czym jest IRF?

**Impulse Response Function (IRF)** to **narzędzie analityczne w ekonometrii szeregów czasowych**, które pokazuje jak zmienne w systemie dynamicznym reagują w czasie na jednorazowy szok (impuls) w jednej ze zmiennych. IRF jest fundamentalnym narzędziem w analizie modeli VAR (Vector Autoregression) i pozwala na śledzenie dynamicznych efektów szoków ekonomicznych.

## Intuicja ekonomiczna:

Wyobraź sobie, że:
- Centralny bank nagle podnosi stopy procentowe o 1 punkt procentowy
- **Pytanie:** Jak ten szok wpłynie na inflację, PKB, bezrobocie w kolejnych miesiącach/kwartałach?
- **IRF odpowiada:** Pokazuje trajektorię reakcji każdej zmiennej w czasie po tym szoku

## Model matematyczny:

### Model VAR(p):
IRF wynika z modeli VAR, które można zapisać jako:

$$
\begin{aligned}
Y_t &= c + A_1 Y_{t-1} + A_2 Y_{t-2} + ... + A_p Y_{t-p} + \varepsilon_t
\end{aligned}
$$

Gdzie:
- $Y_t$ = wektor zmiennych w czasie $t$ (np. [PKB, inflacja, stopy procentowe])
- $A_i$ = macierze współczynników
- $\varepsilon_t$ = wektor szoków (innowacji)

### Reprezentacja VMA (Vector Moving Average):
Model VAR można przekształcić do postaci VMA:

$$
\begin{aligned}
Y_t &= \mu + \sum_{i=0}^{\infty} \Phi_i \varepsilon_{t-i}
\end{aligned}
$$

Gdzie:
- $\Phi_i$ = macierze funkcji reakcji na impuls
- $\Phi_0 = I$ (macierz jednostkowa)

### IRF jest zdefiniowane jako:
$$
\begin{aligned}
IRF_{jk}(h) &= \frac{\partial Y_{j,t+h}}{\partial \varepsilon_{k,t}}
\end{aligned}
$$

Interpretacja:
- Jak zmiana zmiennej $j$ w momencie $t+h$ reaguje na jednostkowy szok w zmiennej $k$ w momencie $t$
- $h$ = horyzont czasowy (0, 1, 2, ... okresów po szoku)

## Jak interpretować IRF?

### Podstawowe elementy:
1. **Oś X (horyzont)**: Liczba okresów po szoku (0, 1, 2, 3, ...)
2. **Oś Y (reakcja)**: Wielkość reakcji zmiennej
3. **Linia IRF**: Trajektoria reakcji w czasie

### Typowe wzorce reakcji:

#### 1. **Reakcja pozytywna przejściowa:**
```
  ^
  |    ╱╲
  |   ╱  ╲___
  |__╱_______╲______> czas
  |
```
Zmienna rośnie, osiąga szczyt, potem wraca do równowagi

#### 2. **Reakcja negatywna przejściowa:**
```
  ^
  |_______________> czas
  |  ╲      ╱
  |   ╲____╱
  v
```
Zmienna spada, osiąga dno, potem wraca do równowagi

#### 3. **Reakcja trwała:**
```
  ^
  |      _______
  |     ╱
  |____╱________> czas
  |
```
Zmienna przechodzi na nowy poziom równowagi

#### 4. **Brak reakcji:**
```
  ^
  |_______________> czas
  |
  |
```
Zmienna nie reaguje na szok

## Rodzaje IRF:

### 1. **Orthogonalized IRF (Cholesky):**
- Wymaga uporządkowania zmiennych
- Zakłada rekurencyjną strukturę szoków
- Szoki są ortogonalne (nieskorelowane)
- **Problem:** Wyniki zależą od kolejności zmiennych

### 2. **Generalized IRF (GIRF):**
- Nie wymaga uporządkowania zmiennych
- Uwzględnia korelację między szokami
- Bardziej odporna na specyfikację
- **Zaleta:** Wyniki niezmienne względem kolejności

### 3. **Structural IRF:**
- Oparta na teorii ekonomicznej
- Identyfikacja przez ograniczenia strukturalne
- Najbardziej ekonomicznie interpretowalna
- **Wymaga:** Silnych założeń teoretycznych

## Przykład praktyczny w Python:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

# Generowanie syntetycznych danych makroekonomicznych
np.random.seed(42)
n_obs = 200

# Symulacja VAR(1) z trzema zmiennymi:
# Y1 = PKB, Y2 = Inflacja, Y3 = Stopa procentowa

# Macierz współczynników VAR(1)
A = np.array([
    [0.8, 0.2, -0.1],   # PKB
    [0.1, 0.7,  0.15],  # Inflacja
    [0.05, 0.3, 0.85]   # Stopa procentowa
])

# Macierz wariancji-kowariancji szoków
Sigma = np.array([
    [1.0, 0.2, 0.1],
    [0.2, 0.8, 0.15],
    [0.1, 0.15, 0.6]
])

# Generowanie danych
Y = np.zeros((n_obs, 3))
Y[0] = np.random.multivariate_normal([0, 0, 0], Sigma)

for t in range(1, n_obs):
    epsilon = np.random.multivariate_normal([0, 0, 0], Sigma)
    Y[t] = A @ Y[t-1] + epsilon

# Tworzenie DataFrame
df = pd.DataFrame(Y, columns=['PKB', 'Inflacja', 'Stopa_proc'])

print("=== DANE WEJŚCIOWE ===")
print(df.describe())
print(f"\nKorelacja między zmiennymi:")
print(df.corr())

# Estymacja modelu VAR
model = VAR(df)

# Wybór optymalnej liczby lagów
lag_order = model.select_order(maxlags=8)
print(f"\n=== WYBÓR LICZBY LAGÓW ===")
print(lag_order.summary())

# Wybieramy liczbę lagów na podstawie AIC
optimal_lag = lag_order.aic
print(f"\nOptymalna liczba lagów (AIC): {optimal_lag}")

# Estymacja modelu z optymalną liczbą lagów
results = model.fit(optimal_lag)
print(f"\n=== WYNIKI MODELU VAR({optimal_lag}) ===")
print(results.summary())

# Obliczanie IRF
irf = results.irf(periods=20)

# Wyświetlanie IRF w formie numerycznej
print(f"\n=== IMPULSE RESPONSE FUNCTIONS (pierwsze 10 okresów) ===")
print("\nReakcja PKB na szok w PKB:")
print(irf.irfs[:10, 0, 0])

print("\nReakcja Inflacji na szok w PKB:")
print(irf.irfs[:10, 1, 0])

print("\nReakcja Stopy proc. na szok w Inflacji:")
print(irf.irfs[:10, 2, 1])

# Wizualizacja IRF
fig = irf.plot(orth=True, impulse='PKB', response='Inflacja',
               figsize=(10, 6))
plt.suptitle('IRF: Reakcja Inflacji na szok w PKB', fontsize=14)
plt.tight_layout()
plt.show()

# Kompletna macierz IRF (wszystkie kombinacje)
fig = irf.plot(orth=True, figsize=(15, 12))
plt.suptitle('Macierz funkcji reakcji na impuls (IRF)', fontsize=16)
plt.tight_layout()
plt.show()

# IRF z przedziałami ufności
irf_ci = results.irf(periods=20)
fig = irf_ci.plot(orth=True, impulse='Stopa_proc',
                  figsize=(12, 8), subplot_params={'fontsize': 10})
plt.suptitle('IRF z 95% przedziałami ufności', fontsize=14)
plt.tight_layout()
plt.show()
```

## Zaawansowana analiza IRF:

```python
def analyze_irf_properties(irf_results, variable_names):
    """
    Zaawansowana analiza właściwości IRF
    """
    print("=== ANALIZA WŁAŚCIWOŚCI IRF ===\n")

    n_vars = len(variable_names)
    irfs = irf_results.irfs
    periods = irfs.shape[0]

    for response_idx, response_var in enumerate(variable_names):
        print(f"\nZMIENNA: {response_var}")
        print("-" * 50)

        for impulse_idx, impulse_var in enumerate(variable_names):
            irf_values = irfs[:, response_idx, impulse_idx]

            # Maksymalna reakcja
            max_response = np.max(np.abs(irf_values))
            max_period = np.argmax(np.abs(irf_values))

            # Suma skumulowana (całkowity efekt)
            cumulative_effect = np.sum(irf_values)

            # Czas do połowy dostosowania (half-life)
            if np.abs(irf_values[0]) > 0.01:
                half_value = irf_values[0] / 2
                try:
                    half_life = np.where(np.abs(irf_values) <= np.abs(half_value))[0][0]
                except:
                    half_life = periods
            else:
                half_life = 0

            print(f"\n  Impuls w: {impulse_var}")
            print(f"    • Maksymalna reakcja: {max_response:.4f} w okresie {max_period}")
            print(f"    • Efekt kumulatywny: {cumulative_effect:.4f}")
            print(f"    • Half-life: {half_life} okresów")
            print(f"    • Reakcja początkowa: {irf_values[0]:.4f}")
            print(f"    • Reakcja po 10 okresach: {irf_values[min(10, periods-1)]:.4f}")

# Przykład użycia
analyze_irf_properties(irf, ['PKB', 'Inflacja', 'Stopa_proc'])
```

## Accumulated IRF (AIRF):

Skumulowana funkcja reakcji na impuls pokazuje całkowity efekt szoku w czasie:

```python
def plot_accumulated_irf(irf_results, impulse_var, response_var):
    """
    Wizualizacja skumulowanej IRF
    """
    impulse_idx = irf_results.model.names.index(impulse_var)
    response_idx = irf_results.model.names.index(response_var)

    irf_values = irf_results.irfs[:, response_idx, impulse_idx]
    accumulated_irf = np.cumsum(irf_values)

    periods = len(irf_values)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # IRF
    axes[0].plot(range(periods), irf_values, 'b-', linewidth=2)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].fill_between(range(periods), 0, irf_values, alpha=0.3)
    axes[0].set_title(f'IRF: Reakcja {response_var} na szok w {impulse_var}')
    axes[0].set_xlabel('Okresy')
    axes[0].set_ylabel('Reakcja')
    axes[0].grid(True, alpha=0.3)

    # Accumulated IRF
    axes[1].plot(range(periods), accumulated_irf, 'r-', linewidth=2)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].fill_between(range(periods), 0, accumulated_irf, alpha=0.3, color='red')
    axes[1].set_title(f'Skumulowana IRF: Całkowity efekt na {response_var}')
    axes[1].set_xlabel('Okresy')
    axes[1].set_ylabel('Efekt skumulowany')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\nEfekt długookresowy: {accumulated_irf[-1]:.4f}")

# Przykład użycia
plot_accumulated_irf(irf, 'PKB', 'Inflacja')
```

## Testy istotności IRF:

```python
def test_irf_significance(irf_results, impulse_var, response_var,
                          confidence_level=0.95, n_bootstrap=1000):
    """
    Test istotności IRF metodą bootstrap
    """
    print(f"\n=== TEST ISTOTNOŚCI IRF ===")
    print(f"Impuls: {impulse_var} → Reakcja: {response_var}")
    print(f"Poziom ufności: {confidence_level*100}%")
    print("-" * 50)

    impulse_idx = irf_results.model.names.index(impulse_var)
    response_idx = irf_results.model.names.index(response_var)

    irf_values = irf_results.irfs[:, response_idx, impulse_idx]
    periods = len(irf_values)

    # Bootstrap przedziały ufności (symulacja - w praktyce użyj metody bootstrap z statsmodels)
    # W rzeczywistości: irf_results zawiera już przedziały ufności
    if hasattr(irf_results, 'ci'):
        lower = irf_results.ci[:, response_idx, impulse_idx, 0]
        upper = irf_results.ci[:, response_idx, impulse_idx, 1]

        # Sprawdź istotność
        significant = []
        for t in range(periods):
            if (lower[t] > 0 and upper[t] > 0) or (lower[t] < 0 and upper[t] < 0):
                significant.append(t)

        print(f"\nOkresy z istotną reakcją: {significant[:10]}{'...' if len(significant) > 10 else ''}")
        print(f"Liczba istotnych okresów: {len(significant)} / {periods}")
    else:
        print("Przedziały ufności niedostępne - użyj irf_results.ci")

    # Wyświetl wybrane wartości
    print(f"\n{'Okres':<8} {'IRF':<10} {'Istotny':<10}")
    print("-" * 30)
    for t in [0, 1, 2, 5, 10, 15, periods-1]:
        if t < periods:
            sig_marker = "✓" if hasattr(irf_results, 'ci') and t in significant else "✗"
            print(f"{t:<8} {irf_values[t]:<10.4f} {sig_marker:<10}")

# Przykład użycia
test_irf_significance(irf, 'Stopa_proc', 'PKB')
```

## Interpretacja ekonomiczna - przykłady:

### Przykład 1: Szok polityki monetarnej
```
Szok: Stopa procentowa ↑ 1%

IRF pokazuje:
- PKB: ↓ w okresie 2-8 (recesja z opóźnieniem)
- Inflacja: ↓ w okresie 4-12 (ceny spadają z większym opóźnieniem)
- Bezrobocie: ↑ w okresie 3-10 (rośnie gdy PKB spada)

Wnioski:
- Polityka monetarna działa z opóźnieniem
- Najsilniejszy efekt po 4-6 kwartałach
- Efekty zanikają po 3-4 latach
```

### Przykład 2: Szok cenowy (np. ropa naftowa)
```
Szok: Cena ropy ↑ 10%

IRF pokazuje:
- Inflacja: ↑ natychmiast (okres 0-2)
- PKB: ↓ w okresie 1-6 (wzrost kosztów)
- Stopy proc: ↑ w okresie 2-8 (reakcja banku centralnego)

Wnioski:
- Szok podażowy ma natychmiastowy efekt
- Bank centralny reaguje z opóźnieniem
- Stagflacja (wysoka inflacja + niski PKB)
```

## Porównanie różnych specyfikacji IRF:

```python
def compare_irf_specifications(var_results, impulse, response, periods=20):
    """
    Porównanie różnych specyfikacji IRF
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 1. Orthogonalized IRF (Cholesky)
    irf_orth = var_results.irf(periods=periods)

    impulse_idx = var_results.names.index(impulse)
    response_idx = var_results.names.index(response)

    irf_orth_values = irf_orth.irfs[:, response_idx, impulse_idx]

    axes[0].plot(irf_orth_values, 'b-', linewidth=2, label='Orthogonalized')
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].fill_between(range(periods), 0, irf_orth_values, alpha=0.3)
    axes[0].set_title(f'Orthogonalized IRF\n{impulse} → {response}')
    axes[0].set_xlabel('Okresy')
    axes[0].set_ylabel('Reakcja')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2. Różne porządki zmiennych
    # (Symulacja - w praktyce trzeba zmienić kolejność w DataFrame i przeliczyć VAR)
    irf_alt_values = irf_orth_values * 0.9 + np.random.normal(0, 0.02, periods)

    axes[1].plot(irf_orth_values, 'b-', linewidth=2, label='Kolejność 1')
    axes[1].plot(irf_alt_values, 'r--', linewidth=2, label='Kolejność 2')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_title(f'Wpływ kolejności zmiennych\n{impulse} → {response}')
    axes[1].set_xlabel('Okresy')
    axes[1].set_ylabel('Reakcja')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

# Przykład użycia
compare_irf_specifications(results, 'PKB', 'Inflacja')
```

## Forecast Error Variance Decomposition (FEVD):

FEVD to uzupełnienie IRF - pokazuje jaki procent wariancji błędu prognozy zmiennej wynika z szoków w innych zmiennych:

```python
def analyze_fevd(var_results, periods=20):
    """
    Analiza dekompozycji wariancji błędu prognozy
    """
    fevd = var_results.fevd(periods=periods)

    print("=== FORECAST ERROR VARIANCE DECOMPOSITION ===\n")

    # Wyświetl FEVD dla wybranych okresów
    for period in [1, 5, 10, periods-1]:
        print(f"\nOkres {period}:")
        print("-" * 60)
        print(fevd.summary().tables[period])

    # Wizualizacja
    fig = fevd.plot(figsize=(15, 10))
    plt.suptitle('Dekompozycja wariancji błędu prognozy', fontsize=16)
    plt.tight_layout()
    plt.show()

# Przykład użycia
analyze_fevd(results)
```

## Praktyczne zastosowania IRF:

### 1. **Polityka monetarna:**
- Analiza skutków zmian stóp procentowych
- Ocena transmisji polityki monetarnej
- Timing interwencji banku centralnego

### 2. **Polityka fiskalna:**
- Efekty zmian podatków
- Multiplikatory wydatków rządowych
- Wpływ na PKB i zatrudnienie

### 3. **Analiza szoków:**
- Szoki cenowe (ropa, energia)
- Szoki technologiczne
- Kryzysy finansowe

### 4. **Finanse:**
- Reakcja rynków na informacje
- Spillover effects między rynkami
- Analiza ryzyka systemowego

### 5. **Prognozowanie:**
- Analiza scenariuszowa
- Symulacje "what-if"
- Stress testing

## Ograniczenia i uwagi:

### 1. **Założenia modelu:**
- Liniowość relacji
- Stabilność strukturalna
- Normalność reszt

### 2. **Problem identyfikacji:**
- Kolejność zmiennych (w Cholesky)
- Konieczność ograniczeń strukturalnych
- Nie zawsze jednoznaczna interpretacja ekonomiczna

### 3. **Wrażliwość:**
- Wybór liczby lagów
- Długość próby
- Stacjonarność danych

### 4. **Best practices:**
```python
def irf_best_practices_checklist():
    """
    Lista dobrych praktyk przy analizie IRF
    """
    checklist = [
        "1. Sprawdź stacjonarność zmiennych (test ADF)",
        "2. Wybierz optymalną liczbę lagów (AIC, BIC)",
        "3. Sprawdź diagnostyki modelu (normalność reszt, autokorelacja)",
        "4. Użyj przedziałów ufności (bootstrap)",
        "5. Porównaj różne specyfikacje (Cholesky, GIRF, Structural)",
        "6. Testuj stabilność (rekursywne IRF)",
        "7. Uzupełnij o FEVD",
        "8. Interpretuj ekonomicznie, nie tylko statystycznie",
        "9. Dokumentuj wszystkie wybory specyfikacji"
    ]

    print("=== BEST PRACTICES - IRF ===")
    for item in checklist:
        print(f"  ✓ {item}")

irf_best_practices_checklist()
```

## Przykład kompletnej analizy:

```python
def complete_irf_analysis(data, var_names, maxlags=8, irf_periods=20):
    """
    Kompletna analiza IRF krok po kroku
    """
    print("=" * 70)
    print("KOMPLETNA ANALIZA IMPULSE RESPONSE FUNCTION")
    print("=" * 70)

    # Krok 1: Testy stacjonarności
    print("\n1. TESTY STACJONARNOŚCI (ADF)")
    print("-" * 50)
    for col in var_names:
        result = adfuller(data[col])
        print(f"{col}: ADF = {result[0]:.4f}, p-value = {result[1]:.4f}", end="")
        print(" [Stacjonarny]" if result[1] < 0.05 else " [Niestacjonarny - rozważ różnicowanie]")

    # Krok 2: Wybór lagów
    print("\n2. WYBÓR LICZBY LAGÓW")
    print("-" * 50)
    model = VAR(data[var_names])
    lag_order = model.select_order(maxlags=maxlags)
    print(f"Optymalny lag (AIC): {lag_order.aic}")
    print(f"Optymalny lag (BIC): {lag_order.bic}")

    # Krok 3: Estymacja modelu
    print(f"\n3. ESTYMACJA MODELU VAR({lag_order.aic})")
    print("-" * 50)
    results = model.fit(lag_order.aic)
    print(f"Liczba parametrów: {results.params.size}")
    print(f"Log-Likelihood: {results.llf:.2f}")
    print(f"AIC: {results.aic:.2f}")

    # Krok 4: Diagnostyki
    print("\n4. DIAGNOSTYKI MODELU")
    print("-" * 50)
    # Normalność reszt
    from statsmodels.stats.stattools import jarque_bera
    jb_test = jarque_bera(results.resid)
    print(f"Jarque-Bera test (normalność): p-value = {jb_test[0][1]:.4f}")

    # Krok 5: IRF
    print("\n5. IMPULSE RESPONSE FUNCTIONS")
    print("-" * 50)
    irf = results.irf(periods=irf_periods)

    # Analiza właściwości
    analyze_irf_properties(irf, var_names)

    # Krok 6: FEVD
    print("\n6. FORECAST ERROR VARIANCE DECOMPOSITION")
    print("-" * 50)
    analyze_fevd(results, periods=irf_periods)

    # Krok 7: Wizualizacje
    print("\n7. GENEROWANIE WIZUALIZACJI...")
    fig = irf.plot(orth=True, figsize=(15, 12))
    plt.suptitle('Impulse Response Functions - Kompletna macierz', fontsize=16)
    plt.tight_layout()
    plt.show()

    return results, irf

# Przykład użycia z pełnym workflow
# results, irf = complete_irf_analysis(df, ['PKB', 'Inflacja', 'Stopa_proc'])
```

## Podsumowanie:

| Właściwość | Opis |
|------------|------|
| **Cel** | Analiza dynamicznych reakcji na szoki |
| **Input** | Model VAR, horyzont czasowy |
| **Output** | Trajektoria reakcji w czasie |
| **Interpretacja** | Jak zmienne reagują na jednostkowy szok |
| **Zastosowania** | Polityka monetarna, fiskalna, analiza szoków |
| **Ograniczenia** | Problem identyfikacji, założenie liniowości |

**Impulse Response Function** jest fundamentalnym narzędziem w makroekonomii i ekonometrii finansowej, pozwalającym na zrozumienie dynamicznych relacji między zmiennymi ekonomicznymi i ocenę skutków polityk gospodarczych oraz szoków zewnętrznych.
