
<span style="color:red">**[NIEWYMAGANE]**</span>

# Test Durbina-Watsona (Durbin-Watson Test)

## Kim byli Durbin i Watson?

**James Durbin** (1923-2012) był brytyjskim statystykiem, profesorem na London School of Economics, znanym z prac nad szeregami czasowymi i ekonometrią. **Geoffrey Watson** (1921-1998) był australijskim statystykiem, który również wniósł znaczący wkład w teorię statystyki matematycznej. W 1950 roku wspólnie opracowali test służący do wykrywania autokorelacji w resztach modeli regresji.

## Czym jest test Durbina-Watsona?

Test Durbina-Watsona (DW test) to **test statystyczny służący do wykrywania autokorelacji pierwszego rzędu w resztach modelu regresji liniowej**. Test sprawdza, czy kolejne reszty (błędy) w modelu są ze sobą skorelowane, co narusza jedno z podstawowych założeń klasycznej regresji liniowej.

### Po co testować autokorelację reszt?

W klasycznej regresji liniowej zakładamy, że reszty są:
- niezależne od siebie
- losowe
- nie wykazują żadnego wzorca

Jeśli reszty są skorelowane (autokorelacja), oznacza to że:
- **Model jest źle specyfikowany** - np. brakuje ważnej zmiennej
- **Standardowe błędy są niedoszacowane** - testy istotności są niewiarygodne
- **Prognozy są nieefektywne** - można je poprawić
- **Może istnieć trend lub sezonowość** - nieuwzględnione w modelu

## Wzór matematyczny:

### Statystyka Durbina-Watsona:
$$
DW = \frac{\sum_{t=2}^{n} (e_t - e_{t-1})^2}{\sum_{t=1}^{n} e_t^2}
$$

Gdzie:
- $e_t$ = reszta (błąd) w okresie $t$
- $n$ = liczba obserwacji
- Licznik = suma kwadratów różnic kolejnych reszt
- Mianownik = suma kwadratów reszt

### Alternatywna postać (przybliżona):
$$
DW \approx 2(1 - \rho)
$$

Gdzie:
- $\rho$ = współczynnik autokorelacji pierwszego rzędu reszt
- $\rho = \frac{\text{Cov}(e_t, e_{t-1})}{\text{Var}(e_t)}$

## Interpretacja wartości DW:

### Zakres wartości:
Statystyka DW przyjmuje wartości od **0 do 4**:

```
0                    2                    4
|--------------------|--------------------|
Silna dodatnia       Brak                Silna ujemna
autokorelacja        autokorelacji       autokorelacja
```

### Szczegółowa interpretacja:

| Wartość DW | Interpretacja | Znaczenie |
|------------|---------------|-----------|
| **DW = 2** | Brak autokorelacji | Idealna sytuacja, reszty niezależne |
| **DW < 2** | Dodatnia autokorelacja | Kolejne reszty podobne do siebie |
| **DW > 2** | Ujemna autokorelacja | Kolejne reszty naprzemienne (zig-zag) |
| **DW ≈ 0** | Bardzo silna dodatnia | Reszty "podążają" za sobą |
| **DW ≈ 4** | Bardzo silna ujemna | Reszty zmieniają znak co okres |

### Praktyczne granice (rule of thumb):

**Brak autokorelacji (OK):**
- **1.5 < DW < 2.5** - zwykle akceptowalne
- **1.8 < DW < 2.2** - bardzo dobre

**Problem autokorelacji:**
- **DW < 1.5** - dodatnia autokorelacja (problem!)
- **DW > 2.5** - ujemna autokorelacja (problem!)

## Formalne reguły decyzyjne:

Test Durbina-Watsona używa wartości krytycznych z tablic:

### Wartości krytyczne: $d_L$ (dolna) i $d_U$ (górna)

Zależą od:
- Liczby obserwacji ($n$)
- Liczby zmiennych objaśniających ($k$)
- Poziomu istotności (zwykle $\alpha = 0.05$)

### Reguły decyzyjne:

```
0        d_L       d_U        2      4-d_U    4-d_L      4
|--------|---------|----------|--------|--------|--------|
  Dodatnia  ?    Brak auto-    ?     Ujemna
  autokore-      korelacji          autokore-
  lacja                             lacja

  H₁        ?         H₀         ?        H₁
```

**Interpretacja:**
1. **DW < $d_L$**: Odrzucamy H₀ - jest **dodatnia autokorelacja**
2. **$d_L$ ≤ DW ≤ $d_U$**: **Nieokreślone** - test niekonkluzywny
3. **$d_U$ < DW < 4-$d_U$**: Nie odrzucamy H₀ - **brak autokorelacji**
4. **4-$d_U$ ≤ DW ≤ 4-$d_L$**: **Nieokreślone** - test niekonkluzywny
5. **DW > 4-$d_L$**: Odrzucamy H₀ - jest **ujemna autokorelacja**

### Hipotezy:
- **H₀**: Brak autokorelacji pierwszego rzędu ($\rho = 0$)
- **H₁**: Jest autokorelacja pierwszego rzędu ($\rho \neq 0$)

## Przykład praktyczny w Python:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.stats.stattools import durbin_watson
import seaborn as sns

# Ustawienia
np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')

print("=" * 70)
print("TEST DURBINA-WATSONA - PRZYKŁADY")
print("=" * 70)

# ============================================================================
# PRZYKŁAD 1: Reszty BEZ autokorelacji (prawidłowy model)
# ============================================================================

print("\n### PRZYKŁAD 1: Model BEZ autokorelacji ###\n")

# Generowanie danych
n = 100
X = np.linspace(0, 10, n).reshape(-1, 1)
# Prawdziwa zależność: y = 2 + 3*x + szum losowy (niezależny)
y_true = 2 + 3 * X.flatten()
noise = np.random.normal(0, 1, n)  # Szum niezależny
y = y_true + noise

# Regresja
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
residuals = y - y_pred

# Test Durbina-Watsona
dw_stat = durbin_watson(residuals)

print(f"Współczynniki modelu: a = {model.intercept_:.4f}, b = {model.coef_[0]:.4f}")
print(f"\nStatystyka Durbina-Watsona: {dw_stat:.4f}")

# Interpretacja
if 1.5 < dw_stat < 2.5:
    print("✓ WNIOSEK: BRAK autokorelacji (DW ≈ 2)")
    print("  Model jest poprawnie specyfikowany")
elif dw_stat <= 1.5:
    print("✗ PROBLEM: Dodatnia autokorelacja (DW < 1.5)")
else:
    print("✗ PROBLEM: Ujemna autokorelacja (DW > 2.5)")

# Wizualizacja
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Dopasowanie modelu
axes[0, 0].scatter(X, y, alpha=0.6, label='Dane')
axes[0, 0].plot(X, y_pred, 'r-', linewidth=2, label='Dopasowanie')
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('Y')
axes[0, 0].set_title('Model: Y = a + b*X')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Reszty w czasie
axes[0, 1].plot(residuals, marker='o', linestyle='-', alpha=0.6)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Obserwacja')
axes[0, 1].set_ylabel('Reszta')
axes[0, 1].set_title(f'Reszty w czasie (DW = {dw_stat:.4f})')
axes[0, 1].grid(True, alpha=0.3)

# 3. Reszty vs reszty opóźnione (lag plot)
axes[1, 0].scatter(residuals[:-1], residuals[1:], alpha=0.6)
axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3)
axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.3)
axes[1, 0].set_xlabel('Reszta(t)')
axes[1, 0].set_ylabel('Reszta(t+1)')
axes[1, 0].set_title('Lag Plot - Autokorelacja reszt')
axes[1, 0].grid(True, alpha=0.3)

# 4. ACF (Autocorrelation Function)
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residuals, lags=20, ax=axes[1, 1], alpha=0.05)
axes[1, 1].set_title('Funkcja Autokorelacji (ACF)')

plt.tight_layout()
plt.savefig('durbin_watson_example1.png', dpi=100, bbox_inches='tight')
plt.show()

# ============================================================================
# PRZYKŁAD 2: Reszty Z dodatnią autokorelacją (problem!)
# ============================================================================

print("\n\n### PRZYKŁAD 2: Model Z dodatnią autokorelacją ###\n")

# Generowanie danych z autokorelacją
X2 = np.linspace(0, 10, n).reshape(-1, 1)
y2_true = 2 + 3 * X2.flatten()

# Szum z autokorelacją AR(1): e_t = 0.7*e_{t-1} + ν_t
noise_ar = np.zeros(n)
noise_ar[0] = np.random.normal(0, 1)
for t in range(1, n):
    noise_ar[t] = 0.7 * noise_ar[t-1] + np.random.normal(0, 1)

y2 = y2_true + noise_ar

# Regresja
model2 = LinearRegression()
model2.fit(X2, y2)
y2_pred = model2.predict(X2)
residuals2 = y2 - y2_pred

# Test Durbina-Watsona
dw_stat2 = durbin_watson(residuals2)

print(f"Współczynniki modelu: a = {model2.intercept_:.4f}, b = {model2.coef_[0]:.4f}")
print(f"\nStatystyka Durbina-Watsona: {dw_stat2:.4f}")

# Interpretacja
if 1.5 < dw_stat2 < 2.5:
    print("✓ WNIOSEK: BRAK autokorelacji (DW ≈ 2)")
elif dw_stat2 <= 1.5:
    print("✗ PROBLEM: Dodatnia autokorelacja (DW < 1.5)")
    print("  Reszty są skorelowane - model wymaga poprawy!")
    print("  Możliwe rozwiązania:")
    print("  • Dodaj opóźnioną zmienną zależną")
    print("  • Dodaj brakujące zmienne objaśniające")
    print("  • Użyj metod szeregów czasowych (ARIMA, VAR)")
else:
    print("✗ PROBLEM: Ujemna autokorelacja (DW > 2.5)")

# Wizualizacja
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

# 1. Dopasowanie modelu
axes2[0, 0].scatter(X2, y2, alpha=0.6, label='Dane')
axes2[0, 0].plot(X2, y2_pred, 'r-', linewidth=2, label='Dopasowanie')
axes2[0, 0].set_xlabel('X')
axes2[0, 0].set_ylabel('Y')
axes2[0, 0].set_title('Model z autokorelowanymi resztami')
axes2[0, 0].legend()
axes2[0, 0].grid(True, alpha=0.3)

# 2. Reszty w czasie
axes2[0, 1].plot(residuals2, marker='o', linestyle='-', alpha=0.6, color='red')
axes2[0, 1].axhline(y=0, color='black', linestyle='--')
axes2[0, 1].set_xlabel('Obserwacja')
axes2[0, 1].set_ylabel('Reszta')
axes2[0, 1].set_title(f'Reszty w czasie (DW = {dw_stat2:.4f}) - PROBLEM!')
axes2[0, 1].grid(True, alpha=0.3)

# 3. Lag plot - wyraźna dodatnia korelacja
axes2[1, 0].scatter(residuals2[:-1], residuals2[1:], alpha=0.6, color='red')
axes2[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
axes2[1, 0].axvline(x=0, color='black', linestyle='--', alpha=0.3)
# Dodaj linię trendu
z = np.polyfit(residuals2[:-1], residuals2[1:], 1)
p = np.poly1d(z)
x_line = np.linspace(residuals2[:-1].min(), residuals2[:-1].max(), 100)
axes2[1, 0].plot(x_line, p(x_line), "b--", linewidth=2, label='Trend')
axes2[1, 0].set_xlabel('Reszta(t)')
axes2[1, 0].set_ylabel('Reszta(t+1)')
axes2[1, 0].set_title('Lag Plot - Widoczna dodatnia autokorelacja!')
axes2[1, 0].legend()
axes2[1, 0].grid(True, alpha=0.3)

# 4. ACF
plot_acf(residuals2, lags=20, ax=axes2[1, 1], alpha=0.05)
axes2[1, 1].set_title('ACF - Istotne opóźnienia!')

plt.tight_layout()
plt.savefig('durbin_watson_example2.png', dpi=100, bbox_inches='tight')
plt.show()

# ============================================================================
# PRZYKŁAD 3: Porównanie różnych poziomów autokorelacji
# ============================================================================

print("\n\n### PRZYKŁAD 3: Porównanie różnych poziomów autokorelacji ###\n")

# Różne współczynniki autokorelacji
rho_values = [0.0, 0.3, 0.5, 0.7, 0.9]
dw_results = []

fig3, axes3 = plt.subplots(2, 3, figsize=(16, 10))
axes3 = axes3.flatten()

for idx, rho in enumerate(rho_values):
    # Generowanie danych z różnym ρ
    X_temp = np.linspace(0, 10, n).reshape(-1, 1)
    y_temp_true = 2 + 3 * X_temp.flatten()

    # Szum z autokorelacją
    noise_temp = np.zeros(n)
    noise_temp[0] = np.random.normal(0, 1)
    for t in range(1, n):
        noise_temp[t] = rho * noise_temp[t-1] + np.random.normal(0, 1)

    y_temp = y_temp_true + noise_temp

    # Regresja
    model_temp = LinearRegression()
    model_temp.fit(X_temp, y_temp)
    y_temp_pred = model_temp.predict(X_temp)
    residuals_temp = y_temp - y_temp_pred

    # DW
    dw_temp = durbin_watson(residuals_temp)
    dw_results.append({'rho': rho, 'DW': dw_temp})

    # Teoretyczna wartość: DW ≈ 2(1-ρ)
    dw_theory = 2 * (1 - rho)

    # Wykres reszt
    axes3[idx].plot(residuals_temp, marker='o', linestyle='-', alpha=0.6)
    axes3[idx].axhline(y=0, color='r', linestyle='--')
    axes3[idx].set_title(f'ρ = {rho:.1f}: DW = {dw_temp:.3f}\n(teoria: {dw_theory:.3f})')
    axes3[idx].set_xlabel('Obserwacja')
    axes3[idx].set_ylabel('Reszta')
    axes3[idx].grid(True, alpha=0.3)

# Usuń ostatni pusty subplot
fig3.delaxes(axes3[5])

plt.tight_layout()
plt.savefig('durbin_watson_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

# Podsumowanie wyników
print("\nPodsumowanie różnych poziomów autokorelacji:")
print("-" * 60)
print(f"{'ρ (autok.)':<15} {'DW stat':<15} {'DW teoria':<15} {'Wniosek'}")
print("-" * 60)

for result in dw_results:
    rho = result['rho']
    dw = result['DW']
    dw_theory = 2 * (1 - rho)

    if 1.5 < dw < 2.5:
        conclusion = "OK"
    elif dw <= 1.5:
        conclusion = "Problem (dodatnia)"
    else:
        conclusion = "Problem (ujemna)"

    print(f"{rho:<15.1f} {dw:<15.3f} {dw_theory:<15.3f} {conclusion}")

print("\n💡 WNIOSEK: Im wyższa autokorelacja (ρ), tym niższa statystyka DW")
```

## Funkcja pomocnicza do analizy DW:

```python
def analyze_durbin_watson(residuals, n_vars=1, alpha=0.05, verbose=True):
    """
    Kompleksowa analiza testu Durbina-Watsona

    Parameters:
    -----------
    residuals : array-like
        Reszty z modelu regresji
    n_vars : int
        Liczba zmiennych objaśniających (bez wyrazu wolnego)
    alpha : float
        Poziom istotności
    verbose : bool
        Czy wyświetlać szczegółowe informacje

    Returns:
    --------
    dict : Słownik z wynikami testu
    """
    from statsmodels.stats.stattools import durbin_watson

    # Obliczanie statystyki DW
    dw_stat = durbin_watson(residuals)

    # Przybliżony współczynnik autokorelacji
    rho_approx = 1 - dw_stat / 2

    # Interpretacja
    if 1.5 < dw_stat < 2.5:
        interpretation = "Brak autokorelacji"
        status = "OK"
    elif dw_stat <= 1.5:
        interpretation = "Dodatnia autokorelacja"
        status = "PROBLEM"
    else:
        interpretation = "Ujemna autokorelacja"
        status = "PROBLEM"

    results = {
        'DW_statistic': dw_stat,
        'rho_approx': rho_approx,
        'interpretation': interpretation,
        'status': status,
        'n_obs': len(residuals),
        'n_vars': n_vars
    }

    if verbose:
        print("=" * 60)
        print("ANALIZA DURBINA-WATSONA")
        print("=" * 60)
        print(f"\nStatystyka DW: {dw_stat:.4f}")
        print(f"Przybliżone ρ: {rho_approx:.4f}")
        print(f"\nInterpretacja: {interpretation}")
        print(f"Status: {status}")

        if status == "PROBLEM":
            print("\n⚠ OSTRZEŻENIE: Wykryto autokorelację!")
            print("\nMożliwe rozwiązania:")
            print("  1. Dodaj opóźnioną zmienną zależną: Y_{t-1}")
            print("  2. Dodaj brakujące zmienne objaśniające")
            print("  3. Sprawdź trend lub sezonowość")
            print("  4. Użyj modeli szeregów czasowych (ARIMA, VAR)")
            print("  5. Zastosuj estymację z korektą (Newey-West, HAC)")
        else:
            print("\n✓ Model spełnia założenie o braku autokorelacji")

        print("\nGranice interpretacyjne:")
        print("  • DW ≈ 2.0: Brak autokorelacji")
        print("  • 1.5 < DW < 2.5: Zwykle akceptowalne")
        print("  • DW < 1.5: Dodatnia autokorelacja (problem)")
        print("  • DW > 2.5: Ujemna autokorelacja (problem)")

    return results

# Przykład użycia
# results = analyze_durbin_watson(residuals, n_vars=2)
```

## Ograniczenia testu Durbina-Watsona:

### 1. **Wykrywa tylko autokorelację pierwszego rzędu:**
```python
# Test DW nie wykryje autokorelacji wyższych rzędów
# Dla AR(2): e_t = ρ₁*e_{t-1} + ρ₂*e_{t-2} + ν_t
# Użyj zamiast tego testu Breuscha-Godfreya
```

### 2. **Wymaga stałej macierzy X:**
- Nie działa gdy X zawiera opóźnioną zmienną zależną
- Nie działa w modelach autoregresyjnych

### 3. **Strefy nieokreślone:**
- Między $d_L$ i $d_U$ test nie daje jednoznacznej odpowiedzi

### 4. **Zakłada normalność reszt:**
- W przypadku silnych odchyleń od normalności może dawać błędne wyniki

## Alternatywne testy autokorelacji:

### Test Breuscha-Godfreya (bardziej ogólny):
```python
from statsmodels.stats.diagnostic import acorr_breusch_godfrey

# Test autokorelacji do p-tego rzędu
bg_test = acorr_breusch_godfrey(model, nlags=4)

print("Test Breuscha-Godfreya:")
print(f"LM statistic: {bg_test[0]:.4f}")
print(f"p-value: {bg_test[1]:.4f}")

if bg_test[1] < 0.05:
    print("✗ Odrzucamy H₀: Jest autokorelacja")
else:
    print("✓ Nie odrzucamy H₀: Brak autokorelacji")
```

### Test Ljunga-Boxa (dla szeregów czasowych):
```python
from statsmodels.stats.diagnostic import acorr_ljungbox

# Test dla reszt
lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)

print("\nTest Ljunga-Boxa:")
print(lb_test)
```

## Praktyczne zastosowania:

### 1. **Ekonometria:**
- Weryfikacja modeli regresji ekonomicznych
- Analiza danych panelowych
- Modele makroekonomiczne

### 2. **Szeregi czasowe:**
- Diagnostyka modeli ARIMA
- Weryfikacja założeń VAR
- Analiza prognoz

### 3. **Finanse:**
- Modele wyceny aktywów
- Analiza zwrotów
- Testy efektywności rynku

### 4. **Kontrola jakości:**
- Analiza procesów produkcyjnych
- Monitorowanie stabilności
- Detekcja trendów

## Przykład z rzeczywistymi danymi:

```python
# Przykład z danymi makroekonomicznymi
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.stats.stattools import durbin_watson

# Załóżmy, że mamy dane PKB i konsumpcji
# df = pd.read_csv('dane_makro.csv')

# Prosty model: Konsumpcja = f(PKB)
# X = df[['PKB']].values
# y = df['Konsumpcja'].values

# model = LinearRegression()
# model.fit(X, y)
# y_pred = model.predict(X)
# residuals = y - y_pred

# dw_stat = durbin_watson(residuals)

# Interpretacja w kontekście ekonomicznym
# if dw_stat < 1.5:
#     print("Dodatnia autokorelacja może wskazywać na:")
#     print("  • Brak zmiennej reprezentującej trendy czasowe")
#     print("  • Cykliczność gospodarcza nie ujęta w modelu")
#     print("  • Opóźnione reakcje konsumentów na zmiany PKB")
```

## Podsumowanie:

| Właściwość | Opis |
|------------|------|
| **Cel** | Wykrywanie autokorelacji pierwszego rzędu w resztach |
| **Zakres** | 0 do 4 (ideał: ≈ 2) |
| **H₀** | Brak autokorelacji ($\rho = 0$) |
| **Interpretacja** | DW < 1.5: problem, 1.5-2.5: OK, DW > 2.5: problem |
| **Zastosowanie** | Diagnostyka modeli regresji liniowej |
| **Ograniczenia** | Tylko AR(1), nie działa z opóźnioną zmienną zależną |
| **Alternatywy** | Test Breuscha-Godfreya, test Ljunga-Boxa |

**Test Durbina-Watsona** jest fundamentalnym narzędziem diagnostycznym w ekonometrii, pozwalającym szybko zweryfikować jedno z kluczowych założeń klasycznej regresji liniowej - niezależność reszt. Mimo ograniczeń, pozostaje najpopularniejszym testem autokorelacji ze względu na prostotę interpretacji i powszechną dostępność w oprogramowaniu statystycznym.
