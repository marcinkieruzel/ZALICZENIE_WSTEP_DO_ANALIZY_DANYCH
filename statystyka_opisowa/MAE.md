# Mean Absolute Error (MAE) - Średni Błąd Bezwzględny

## Czym jest MAE?

**Mean Absolute Error (MAE)** to **metryka oceny jakości predykcji**, która mierzy średnią wartość bezwzględną różnic między wartościami przewidywanymi a rzeczywistymi. Jest to jedna z najprostszych i najczęściej używanych metryk w uczeniu maszynowym i analizie szeregów czasowych.

## Wzór matematyczny:

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

Gdzie:
- $n$ = liczba obserwacji
- $y_i$ = rzeczywista wartość dla obserwacji $i$
- $\hat{y}_i$ = przewidywana wartość dla obserwacji $i$
- $|...|$ = wartość bezwzględna

## Interpretacja:

### Zalety MAE:
1. **Łatwość interpretacji** - wynik w tych samych jednostkach co dane
2. **Odporność na outliers** - nie karze mocno za pojedyncze duże błędy
3. **Intuicyjność** - pokazuje "średni błąd" predykcji
4. **Symetryczność** - traktuje jednakowo nadszacowanie i niedoszacowanie

### Wady MAE:
1. **Brak różnicowania** - wszystkie błędy traktowane jednakowo
2. **Nieróżniczkowalność** - problemy optymalizacyjne w niektórych algorytmach
3. **Może maskować** duże błędy (w przeciwieństwie do MSE)

## Porównanie z innymi metrykami:

### MAE vs MSE (Mean Squared Error):
```
MSE = (1/n) * Σ(yi - ŷi)²
```
- **MSE** bardziej karze duże błędy (przez podnoszenie do kwadratu)
- **MAE** traktuje wszystkie błędy równomiernie
- **MSE** w jednostkach kwadratowych, **MAE** w oryginalnych jednostkach

### MAE vs RMSE (Root Mean Squared Error):
```
RMSE = √MSE
```
- **RMSE** w tych samych jednostkach co MAE
- **RMSE** bardziej wrażliwe na outliers
- **MAE** bardziej odporne na wartości odstające

## Kiedy używać MAE?

### Idealne przypadki:
1. **Gdy outliers są problemem** - MAE nie jest tak wrażliwe na skrajne wartości
2. **Gdy potrzebna jest intuicyjna interpretacja** - "średni błąd o X jednostek"
3. **Analiza szeregów czasowych** - szczególnie w prognozowaniu
4. **Dane z szumem** - MAE lepiej radzi sobie z nieregularnymi danymi

### Unikaj MAE gdy:
1. **Duże błędy są krytyczne** - użyj MSE/RMSE
2. **Potrzebujesz różniczkowalności** - w niektórych algorytmach optymalizacji
3. **Chcesz "ukarać" outliers** - MSE lepiej to robi

## Przykłady praktyczne:

### Przykład 1: Prognozowanie cen nieruchomości
```
Rzeczywiste ceny: [300k, 250k, 400k, 350k]
Przewidywane:     [290k, 260k, 380k, 340k]

MAE = |300-290| + |250-260| + |400-380| + |350-340| / 4
MAE = 10 + 10 + 20 + 10 / 4 = 12.5k

Interpretacja: Średnio mylimy się o 12,500 zł
```

### Przykład 2: Prognozowanie sprzedaży
```
Rzeczywiste:   [100, 150, 200, 120]
Przewidywane:  [95,  140, 220, 125]

MAE = |100-95| + |150-140| + |200-220| + |120-125| / 4
MAE = 5 + 10 + 20 + 5 / 4 = 10

Interpretacja: Średnio mylimy się o 10 sztuk
```

## Implementacja w Python:

```python
import numpy as np
from sklearn.metrics import mean_absolute_error
import pandas as pd

# Sposób 1: Własna implementacja
def mae_custom(y_true, y_pred):
    """Własna implementacja MAE"""
    return np.mean(np.abs(y_true - y_pred))

# Sposób 2: Sklearn
from sklearn.metrics import mean_absolute_error

# Przykładowe dane
y_true = np.array([3.0, 2.5, 0.5, 4.0, 5.0])
y_pred = np.array([2.8, 2.7, 0.3, 3.8, 5.2])

# Obliczenie MAE
mae_sklearn = mean_absolute_error(y_true, y_pred)
mae_own = mae_custom(y_true, y_pred)

print(f"MAE (sklearn): {mae_sklearn:.4f}")
print(f"MAE (własne): {mae_own:.4f}")

# Przykład z prognozowaniem szeregów czasowych
import matplotlib.pyplot as plt

# Generowanie danych szeregu czasowego
np.random.seed(42)
time = np.arange(100)
true_values = 10 + 2 * np.sin(time * 0.1) + np.random.normal(0, 0.5, 100)

# Symulacja prostego modelu predykcyjnego
predicted_values = 10 + 1.8 * np.sin(time * 0.1) + np.random.normal(0, 0.3, 100)

# Obliczenie MAE dla całego szeregu
mae_timeseries = mean_absolute_error(true_values, predicted_values)
print(f"\nMAE dla szeregu czasowego: {mae_timeseries:.4f}")

# Wizualizacja
plt.figure(figsize=(12, 6))
plt.plot(time, true_values, label='Wartości rzeczywiste', alpha=0.7)
plt.plot(time, predicted_values, label='Wartości przewidywane', alpha=0.7)
plt.title(f'Porównanie wartości rzeczywistych i przewidywanych\nMAE = {mae_timeseries:.4f}')
plt.xlabel('Czas')
plt.ylabel('Wartość')
plt.legend()
plt.grid(True)
plt.show()

# Analiza błędów w czasie
errors = np.abs(true_values - predicted_values)
plt.figure(figsize=(12, 4))
plt.plot(time, errors, label='Błędy bezwzględne')
plt.axhline(y=mae_timeseries, color='red', linestyle='--', 
           label=f'Średni błąd (MAE = {mae_timeseries:.4f})')
plt.title('Rozkład błędów bezwzględnych w czasie')
plt.xlabel('Czas')
plt.ylabel('Błąd bezwzględny')
plt.legend()
plt.grid(True)
plt.show()
```

## Warianty MAE:

### 1. Weighted MAE (ważony MAE):
$$
WMAE = \frac{\sum_{i=1}^{n} w_i |y_i - \hat{y}_i|}{\sum_{i=1}^{n} w_i}
$$

### 2. Median Absolute Error:
```
MedAE = median(|yi - ŷi|)
```
- Jeszcze bardziej odporna na outliers

### 3. Mean Absolute Percentage Error (MAPE):
$$
MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|
$$

## Praktyczne wskazówki:

### 1. Benchmarking:
- **MAE < 10%** średniej wartości = dobry model
- **MAE > 20%** średniej wartości = słaby model
- Zawsze porównuj z baseline (np. średnia historyczna)

### 2. Kontekst biznesowy:
- W finansach: MAE w tysiącach może być akceptowalne dla milionowych transakcji
- W medycynie: nawet małe MAE może być krytyczne
- W e-commerce: MAE zależy od wartości produktów

### 3. Kombinacja z innymi metrykami:
```python
from sklearn.metrics import mean_squared_error, r2_score

def comprehensive_evaluation(y_true, y_pred):
    """Kompleksowa ocena modelu"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"RMSE/MAE ratio: {rmse/mae:.2f}")
    
    if rmse/mae > 1.5:
        print("⚠️  Duże outliers wykryte (RMSE >> MAE)")
    else:
        print("✅ Błędy równomiernie rozłożone")
```

## Zastosowania MAE w różnych dziedzinach:

### 1. **Finanse:**
- Prognozowanie cen akcji
- Ocena modeli ryzyka kredytowego
- Planowanie budżetu

### 2. **E-commerce:**
- Prognozowanie popytu
- Optymalizacja cen
- Planowanie zapasów

### 3. **Energetyka:**
- Prognozowanie zużycia energii
- Planowanie produkcji
- Optymalizacja sieci

### 4. **Transport:**
- Przewidywanie czasów podróży
- Optymalizacja tras
- Planowanie rozkładów

MAE jest fundamentalną metryką w analizie predykcyjnej - prostą w interpretacji, odporną na outliers i użyteczną w większości zastosowań biznesowych i naukowych.
