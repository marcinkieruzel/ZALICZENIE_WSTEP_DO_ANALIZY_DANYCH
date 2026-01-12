# Test Przyczynowości Grangera (Granger Causality Test)

## Kim był Clive Granger?

**Sir Clive William John Granger** (1934-2009) to brytyjski ekonomista i laureat Nagrody Nobla w dziedzinie ekonomii z 2003 roku. Granger był profesorem na University of California w San Diego i pionierem w dziedzinie ekonometrii szeregów czasowych. Jego największym wkładem było wprowadzenie koncepcji przyczynowości w analizie szeregów czasowych oraz rozwój teorii kointegracji.

## Czym jest test przyczynowości Grangera?

Test przyczynowości Grangera to **test statystyczny sprawdzający, czy przeszłe wartości jednego szeregu czasowego pomagają przewidywać przyszłe wartości innego szeregu czasowego**. Nie testuje prawdziwej przyczynowości w sensie filozoficznym, ale **przyczynowość w sensie przewidywalności**.

### Kluczowa idea:
Szereg X "powoduje w sensie Grangera" szereg Y, jeśli przeszłe wartości X zawierają informacje pomocne w przewidywaniu Y, których nie ma w przeszłych wartościach samego Y.

## Model matematyczny:

Test opiera się na porównaniu dwóch modeli regresji:

### Model ograniczony (bez przyczynowości):
$$
\begin{aligned}
Y_t &= \alpha_0 + \sum_{i=1}^{p} \alpha_i Y_{t-i} + \varepsilon_t
\end{aligned}
$$

### Model nieograniczony (z przyczynowością):
$$
\begin{aligned}
Y_t &= \beta_0 + \sum_{i=1}^{p} \beta_i Y_{t-i} + \sum_{i=1}^{p} \gamma_i X_{t-i} + \varepsilon_t
\end{aligned}
$$

## Hipotezy testowe:

- **H₀**: $\gamma_1 = \gamma_2 = ... = \gamma_p = 0$ (X nie powoduje Y w sensie Grangera)
- **H₁**: Przynajmniej jeden $\gamma_i \neq 0$ (X powoduje Y w sensie Grangera)

## Jak interpretować wyniki?

### Statystyka testowa:
Test używa **statystyki F** do porównania modeli:
$$
F = \frac{(RSS_{ograniczony} - RSS_{nieograniczony})/p}{RSS_{nieograniczony}/(n-2p-1)}
$$

### Interpretacja:
1. **p-value < α (np. 0.05)**:
   - Odrzucamy H₀
   - X **powoduje** Y w sensie Grangera
   - Przeszłe wartości X pomagają przewidywać Y

2. **p-value > α**:
   - Nie odrzucamy H₀
   - X **nie powoduje** Y w sensie Grangera
   - Przeszłe wartości X nie dodają informacji predykcyjnej

## Rodzaje przyczynowości Grangera:

### 1. Przyczynowość jednostronna:
- X → Y: X powoduje Y, ale Y nie powoduje X

### 2. Przyczynowość dwustronna:
- X ↔ Y: X powoduje Y i Y powoduje X

### 3. Brak przyczynowości:
- X ⊥ Y: ani X nie powoduje Y, ani Y nie powoduje X

### 4. Przyczynowość natychmiastowa:
- Współczesne wartości są skorelowane, ale brak przyczynowości opóźnionej

## Ważne założenia:

1. **Stacjonarność**: Szeregi muszą być stacjonarne (lub skointegrowane)
2. **Optymalna długość opóźnienia**: Wybrana na podstawie kryteriów informacyjnych (AIC, BIC)
3. **Liniowość**: Test zakłada liniowe relacje
4. **Brak pominięcia zmiennych**: Inne istotne zmienne mogą wpływać na wyniki

## Praktyczne znaczenie:

### W ekonomii:
- Czy PKB "powoduje" inflację?
- Czy stopy procentowe "powodują" zmiany kursu walutowego?

### W finansach:
- Czy ceny ropy naftowej "powodują" zmiany w cenach akcji?
- Czy jeden indeks giełdowy "prowadzi" drugi?

### W medycynie/biologii:
- Czy jeden biomarker "poprzedza" wystąpienie choroby?

## Przykład w Python:

```python
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd
import numpy as np

# Generowanie przykładowych danych
np.random.seed(42)
n = 1000

# X nie powoduje Y (brak przyczynowości)
x = np.random.randn(n)
y = np.random.randn(n)

# Tworzenie DataFrame
data = pd.DataFrame({'y': y, 'x': x})

# Test przyczynowości Grangera
# H0: x nie powoduje y w sensie Grangera
result = grangercausalitytests(data[['y', 'x']], maxlag=4, verbose=False)

print("=== TEST PRZYCZYNOWOŚCI GRANGERA ===")
for lag in range(1, 5):
    f_stat = result[lag][0]['ssr_ftest'][0]
    p_value = result[lag][0]['ssr_ftest'][1]
    print(f"Lag {lag}: F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}")
    print(f"Wniosek: {'X powoduje Y' if p_value < 0.05 else 'X nie powoduje Y'}")

# Przykład z prawdziwą przyczynowością
# Y zależy od przeszłych wartości X
y_causal = np.zeros(n)
x_causal = np.random.randn(n)

for t in range(2, n):
    y_causal[t] = 0.5 * y_causal[t-1] + 0.3 * x_causal[t-1] + 0.2 * x_causal[t-2] + np.random.randn()

data_causal = pd.DataFrame({'y': y_causal, 'x': x_causal})
result_causal = grangercausalitytests(data_causal[['y', 'x']], maxlag=4, verbose=False)

print("\n=== TEST Z PRAWDZIWĄ PRZYCZYNOWOŚCIĄ ===")
for lag in range(1, 5):
    f_stat = result_causal[lag][0]['ssr_ftest'][0]
    p_value = result_causal[lag][0]['ssr_ftest'][1]
    print(f"Lag {lag}: F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}")
    print(f"Wniosek: {'X powoduje Y' if p_value < 0.05 else 'X nie powoduje Y'}")
```

### Kluczowe biblioteki Python:
- `statsmodels.tsa.stattools.grangercausalitytests()` - główna funkcja testująca
- `statsmodels.tsa.api.VAR` - modele wektorowej autoregresji
- `pandas` - manipulacja danymi szeregów czasowych
- `numpy` - obliczenia numeryczne

## Ograniczenia i uwagi:

1. **"Correlation is not causation"** - test pokazuje tylko przyczynowość statystyczną
2. **Wrażliwość na długość opóźnienia** - wyniki zależą od wyboru liczby lagów
3. **Założenie liniowości** - może nie wykryć nieliniowych relacji
4. **Problem pominięcia zmiennych** - trzecia zmienna może być prawdziwą przyczyną
5. **Kierunek czasu** - test zakłada, że przyczyna poprzedza skutek w czasie

Test przyczynowości Grangera jest fundamentalnym narzędziem w analizie szeregów czasowych, szczególnie użytecznym w ekonometrii, finansach i innych dziedzinach wymagających analizy relacji czasowych między zmiennymi.