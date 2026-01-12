# Root Mean Squared Error (RMSE) - Pierwiastek ze Średniego Błędu Kwadratowego

## Czym jest RMSE?

**Root Mean Squared Error (RMSE)** to **metryka oceny jakości predykcji**, która łączy zalety MSE z łatwością interpretacji MAE. Jest to pierwiastek kwadratowy ze średniego błędu kwadratowego, co sprawia, że wynik jest w tych samych jednostkach co oryginalne dane, zachowując jednocześnie właściwość penalizowania dużych błędów.

## Wzór matematyczny:

$$
RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

Gdzie:
- $n$ = liczba obserwacji
- $y_i$ = rzeczywista wartość dla obserwacji $i$
- $\hat{y}_i$ = przewidywana wartość dla obserwacji $i$
- $MSE$ = Mean Squared Error

## Interpretacja:

### Zalety RMSE:
1. **Intuicyjna interpretacja** - wynik w tych samych jednostkach co dane
2. **Penalizuje duże błędy** - zachowuje właściwość MSE
3. **Różniczkowalność** - nadal łatwe w optymalizacji
4. **Standardowa metryka** - szeroko używana w konkursach ML i publikacjach
5. **Łączy zalety MSE i MAE** - najlepsze z obu światów

### Wady RMSE:
1. **Wrażliwość na outliers** - podobnie jak MSE
2. **Matematyczna złożoność** - pierwiastek może komplikować niektóre obliczenia
3. **Niesymetryczne karanie** - duże błędy nadal dominują
4. **Skala zależna** - trudne porównywanie między różnymi zbiorami danych

## Relacje z innymi metrykami:

### RMSE vs MSE vs MAE:
$$
\begin{aligned}
MAE &= \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \\
MSE &= \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \\
RMSE &= \sqrt{MSE}
\end{aligned}
$$

### Właściwości porównawcze:
- $RMSE \geq MAE$ (zawsze, z równością tylko gdy wszystkie błędy są jednakowe)
- Im większy stosunek $\frac{RMSE}{MAE}$, tym więcej outlierów w danych
- $\frac{RMSE}{MAE} = 1$ → wszystkie błędy jednakowe
- $\frac{RMSE}{MAE} > 1.5$ → znaczące outliers

## Kiedy używać RMSE?

### Idealne przypadki:
1. **Potrzebujesz penalizacji dużych błędów** - ale chcesz interpretację w oryginalnych jednostkach
2. **Konkursy ML i benchmarking** - RMSE to standard branżowy
3. **Raporty biznesowe** - łatwiejsze do wyjaśnienia niż MSE
4. **Modele probabilistyczne** - gdzie duże błędy są szczególnie kosztowne
5. **Porównywanie modeli** - standardowa metryka

### Unikaj RMSE gdy:
1. **Outliers są naturalnie występujące** - użyj MAE
2. **Potrzebujesz absolutnej odporności** - MAE będzie lepszy
3. **Optymalizujesz bezpośrednio funkcję kosztu** - MSE może być wydajniejsze

## Przykłady praktyczne:

### Przykład 1: Prognozowanie sprzedaży (w tysiącach sztuk)
```
Rzeczywiste sprzedaże: [100, 150, 200, 120, 180]
Przewidywane:          [95,  140, 220, 125, 170]

Błędy: [5, 10, -20, -5, 10]
Błędy²: [25, 100, 400, 25, 100]

MSE = (25 + 100 + 400 + 25 + 100) / 5 = 130 (tys.²)
RMSE = √130 = 11.4 tysięcy sztuk

Interpretacja: Średnio mylimy się o 11,400 sztuk
```

### Przykład 2: Prognozowanie cen akcji
```
Rzeczywiste ceny: [$100, $105, $95, $110, $102]
Przewidywane:     [$98,  $108, $90, $115, $100]

Błędy: [2, -3, 5, -5, 2]
Błędy²: [4, 9, 25, 25, 4]

MSE = (4 + 9 + 25 + 25 + 4) / 5 = 13.4 $²
RMSE = √13.4 = $3.66

Interpretacja: Średni błąd około $3.66
```

## Implementacja w Python:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Sposób 1: Własna implementacja
def rmse_custom(y_true, y_pred):
    """Własna implementacja RMSE"""
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)

# Sposób 2: Poprzez sklearn MSE
def rmse_sklearn(y_true, y_pred):
    """RMSE używając sklearn MSE"""
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)

# Sposób 3: Sklearn 0.22+ ma built-in RMSE
try:
    from sklearn.metrics import mean_squared_error
    # W nowszych wersjach sklearn
    rmse_builtin = lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False)
except:
    rmse_builtin = rmse_sklearn

# Przykładowe dane
y_true = np.array([3.0, 2.5, 0.5, 4.0, 5.0])
y_pred = np.array([2.8, 2.7, 0.3, 3.8, 5.2])

# Obliczenie wszystkich metryk
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = rmse_custom(y_true, y_pred)

print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"RMSE/MAE ratio: {rmse/mae:.2f}")

# Analiza wpływu outlierów na RMSE
print("\n" + "="*60)
print("ANALIZA WPŁYWU OUTLIERÓW NA RMSE")
print("="*60)

def analyze_outlier_impact(clean_data, outlier_data):
    """Analizuje wpływ outlierów na różne metryki"""
    y_true_clean, y_pred_clean = clean_data
    y_true_out, y_pred_out = outlier_data
    
    # Metryki dla danych czystych
    mae_clean = mean_absolute_error(y_true_clean, y_pred_clean)
    mse_clean = mean_squared_error(y_true_clean, y_pred_clean)
    rmse_clean = np.sqrt(mse_clean)
    
    # Metryki dla danych z outlierem
    mae_out = mean_absolute_error(y_true_out, y_pred_out)
    mse_out = mean_squared_error(y_true_out, y_pred_out)
    rmse_out = np.sqrt(mse_out)
    
    print(f"Dane czyste    - MAE: {mae_clean:.3f}, MSE: {mse_clean:.3f}, RMSE: {rmse_clean:.3f}")
    print(f"Z outlierem    - MAE: {mae_out:.3f}, MSE: {mse_out:.3f}, RMSE: {rmse_out:.3f}")
    print(f"Wzrost MAE:  {mae_out/mae_clean:.2f}x")
    print(f"Wzrost MSE:  {mse_out/mse_clean:.2f}x")
    print(f"Wzrost RMSE: {rmse_out/rmse_clean:.2f}x")
    print(f"RMSE/MAE (czyste): {rmse_clean/mae_clean:.2f}")
    print(f"RMSE/MAE (outlier): {rmse_out/mae_out:.2f}")

# Dane testowe
clean_data = (
    np.array([10, 12, 11, 13, 9]),
    np.array([11, 11, 12, 12, 10])
)

outlier_data = (
    np.array([10, 12, 11, 13, 9]),
    np.array([11, 11, 12, 25, 10])  # outlier: 25 zamiast 12
)

analyze_outlier_impact(clean_data, outlier_data)

# Wizualizacja różnic między metrykami
def plot_metrics_comparison(y_true, y_pred_list, labels):
    """Porównuje metryki dla różnych predykcji"""
    metrics_data = []
    
    for pred, label in zip(y_pred_list, labels):
        mae = mean_absolute_error(y_true, pred)
        mse = mean_squared_error(y_true, pred)
        rmse = np.sqrt(mse)
        metrics_data.append({'Model': label, 'MAE': mae, 'MSE': mse, 'RMSE': rmse})
    
    df = pd.DataFrame(metrics_data)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # MAE
    axes[0].bar(df['Model'], df['MAE'], color='skyblue')
    axes[0].set_title('Mean Absolute Error (MAE)')
    axes[0].set_ylabel('MAE')
    axes[0].tick_params(axis='x', rotation=45)
    
    # MSE  
    axes[1].bar(df['Model'], df['MSE'], color='lightcoral')
    axes[1].set_title('Mean Squared Error (MSE)')
    axes[1].set_ylabel('MSE')
    axes[1].tick_params(axis='x', rotation=45)
    
    # RMSE
    axes[2].bar(df['Model'], df['RMSE'], color='lightgreen')
    axes[2].set_title('Root Mean Squared Error (RMSE)')
    axes[2].set_ylabel('RMSE')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return df

# Przykład porównania modeli
np.random.seed(42)
n = 100
X = np.random.randn(n, 3)
y = 2*X[:, 0] - X[:, 1] + 0.5*X[:, 2] + np.random.randn(n)*0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Różne modele
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42)
}

predictions = []
model_names = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions.append(y_pred)
    model_names.append(name)

# Dodaj baseline (średnia)
baseline_pred = np.full_like(y_test, np.mean(y_train))
predictions.append(baseline_pred)
model_names.append('Baseline (Mean)')

# Porównanie metryk
print("\n" + "="*60)
print("PORÓWNANIE MODELI")
print("="*60)
metrics_df = plot_metrics_comparison(y_test, predictions, model_names)
print(metrics_df)
```

## RMSE w kontekście różnych problemów:

### 1. Problem regresji czasowej:
```python
def evaluate_time_series_model(y_true, y_pred, periods):
    """Ocena modelu szeregów czasowych z RMSE"""
    rmse_overall = rmse_custom(y_true, y_pred)
    
    # RMSE dla różnych okresów
    rmse_by_period = []
    for i in range(0, len(y_true), periods):
        end_idx = min(i + periods, len(y_true))
        if end_idx > i:
            rmse_period = rmse_custom(y_true[i:end_idx], y_pred[i:end_idx])
            rmse_by_period.append(rmse_period)
    
    return {
        'overall_rmse': rmse_overall,
        'period_rmse': rmse_by_period,
        'rmse_stability': np.std(rmse_by_period)
    }
```

### 2. Cross-validation z RMSE:
```python
from sklearn.model_selection import cross_val_score

def rmse_cv_evaluation(model, X, y, cv=5):
    """Cross-validation z RMSE jako metryką"""
    # Sklearn zwraca ujemne MSE, więc trzeba przekonwertować
    neg_mse_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-neg_mse_scores)
    
    print(f"RMSE CV: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
    print(f"Min RMSE: {rmse_scores.min():.4f}")
    print(f"Max RMSE: {rmse_scores.max():.4f}")
    
    return rmse_scores

# Przykład użycia
model = LinearRegression()
rmse_scores = rmse_cv_evaluation(model, X, y)
```

## Interpretacja RMSE w kontekście biznesowym:

### 1. Benchmark Guidelines:
```python
def interpret_rmse(rmse_value, y_mean, y_std):
    """Interpretuje RMSE w kontekście danych"""
    rmse_vs_mean = rmse_value / y_mean * 100
    rmse_vs_std = rmse_value / y_std
    
    print(f"RMSE: {rmse_value:.4f}")
    print(f"Średnia danych: {y_mean:.4f}")
    print(f"Odchylenie standardowe: {y_std:.4f}")
    print(f"RMSE jako % średniej: {rmse_vs_mean:.1f}%")
    print(f"RMSE / std: {rmse_vs_std:.2f}")
    
    # Interpretacja
    if rmse_vs_mean < 5:
        quality = "Bardzo dobra"
    elif rmse_vs_mean < 10:
        quality = "Dobra"
    elif rmse_vs_mean < 20:
        quality = "Średnia"
    else:
        quality = "Słaba"
    
    print(f"Jakość modelu: {quality}")
    
    return {
        'rmse_percent': rmse_vs_mean,
        'rmse_std_ratio': rmse_vs_std,
        'quality': quality
    }

# Przykład interpretacji
y_example = np.array([100, 120, 90, 110, 105, 95, 115])
y_pred_example = np.array([98, 118, 92, 108, 107, 93, 117])
rmse_ex = rmse_custom(y_example, y_pred_example)

interpret_rmse(rmse_ex, np.mean(y_example), np.std(y_example))
```

## RMSE vs inne metryki - kiedy co wybrać:

### Decyzyjne drzewo wyboru metryki:
```python
def suggest_metric(residuals, business_context):
    """Sugeruje najlepszą metrykę na podstawie danych i kontekstu"""
    abs_residuals = np.abs(residuals)
    
    # Analiza outlierów
    q75, q25 = np.percentile(abs_residuals, [75, 25])
    iqr = q75 - q25
    outlier_threshold = q75 + 1.5 * iqr
    outlier_ratio = np.sum(abs_residuals > outlier_threshold) / len(abs_residuals)
    
    # Stosunek RMSE/MAE
    mae_val = np.mean(abs_residuals)
    rmse_val = np.sqrt(np.mean(residuals**2))
    rmse_mae_ratio = rmse_val / mae_val
    
    print(f"Analiza danych:")
    print(f"- Outliers: {outlier_ratio*100:.1f}%")
    print(f"- RMSE/MAE ratio: {rmse_mae_ratio:.2f}")
    
    # Rekomendacja
    if outlier_ratio > 0.15:
        recommendation = "MAE (dużo outlierów)"
    elif business_context == "cost_sensitive" and rmse_mae_ratio > 1.3:
        recommendation = "MAE (koszty outlierów wysokie)"
    elif business_context == "scientific":
        recommendation = "RMSE (standard naukowy)"
    elif business_context == "ml_competition":
        recommendation = "RMSE (standard konkursów)"
    else:
        recommendation = "RMSE (dobry kompromis)"
    
    print(f"Rekomendacja: {recommendation}")
    return recommendation
```

## RMSE w różnych zastosowaniach:

### 1. **Prognozowanie finansowe:**
```python
def financial_rmse_analysis(returns_true, returns_pred):
    """Analiza RMSE dla zwrotów finansowych"""
    rmse = rmse_custom(returns_true, returns_pred)
    
    # Annualized RMSE
    trading_days = 252
    annualized_rmse = rmse * np.sqrt(trading_days)
    
    print(f"Daily RMSE: {rmse*100:.2f}%")
    print(f"Annualized RMSE: {annualized_rmse*100:.2f}%")
    
    return annualized_rmse
```

### 2. **Kontrola jakości:**
```python
def quality_control_rmse(measurements, targets, tolerance):
    """RMSE w kontekście kontroli jakości"""
    rmse = rmse_custom(measurements, targets)
    
    # Analiza względem tolerancji
    rmse_tolerance_ratio = rmse / tolerance
    
    if rmse_tolerance_ratio < 0.3:
        status = "Excellent"
    elif rmse_tolerance_ratio < 0.5:
        status = "Good"
    elif rmse_tolerance_ratio < 1.0:
        status = "Acceptable"
    else:
        status = "Poor"
    
    print(f"RMSE: {rmse:.4f}")
    print(f"Tolerance: {tolerance:.4f}")
    print(f"RMSE/Tolerance: {rmse_tolerance_ratio:.2f}")
    print(f"Quality Status: {status}")
    
    return status
```

## Zaawansowane techniki z RMSE:

### 1. **Weighted RMSE:**
```python
def weighted_rmse(y_true, y_pred, weights):
    """Ważony RMSE dla różnych ważności predykcji"""
    squared_errors = (y_true - y_pred) ** 2
    weighted_mse = np.average(squared_errors, weights=weights)
    return np.sqrt(weighted_mse)

# Przykład: nowsze obserwacje mają większą wagę
n = len(y_true)
weights = np.linspace(0.5, 1.5, n)  # Rosnące wagi
wrmse = weighted_rmse(y_true, y_pred, weights)
print(f"Weighted RMSE: {wrmse:.4f}")
```

### 2. **Rolling RMSE:**
```python
def rolling_rmse(y_true, y_pred, window_size):
    """RMSE dla kroczącego okna"""
    rolling_rmses = []
    
    for i in range(window_size, len(y_true) + 1):
        window_true = y_true[i-window_size:i]
        window_pred = y_pred[i-window_size:i]
        rmse_window = rmse_custom(window_true, window_pred)
        rolling_rmses.append(rmse_window)
    
    return np.array(rolling_rmses)

# Analiza stabilności modelu w czasie
rolling_rmses = rolling_rmse(y_true, y_pred, window_size=5)
print(f"Średni rolling RMSE: {np.mean(rolling_rmses):.4f}")
print(f"Std rolling RMSE: {np.std(rolling_rmses):.4f}")
```

## Porównanie wszystkich metryk - podsumowanie:

| Metryka | Jednostki | Outliers | Interpretacja | Optymalizacja | Użycie |
|---------|-----------|----------|---------------|---------------|---------|
| **MAE** | Oryginalne | Odporny | Bardzo łatwa | Trudniejsza | Outliers naturalne |
| **MSE** | Kwadratowe | Wrażliwy | Trudna | Bardzo łatwa | Optymalizacja |
| **RMSE** | Oryginalne | Wrażliwy | Łatwa | Łatwa | Standardowe reporting |

### Rekomendacja wyboru:
- **RMSE** - **najlepszy kompromis** dla większości zastosowań
- **Łączy** interpretowalnośc MAE z właściwościami optymalizacyjnymi MSE  
- **Standard w branży** ML i konkursach Kaggle
- **Idealne** do raportowania biznesowego i naukowego

RMSE jest obecnie **najszerzej używaną metryką** w uczeniu maszynowym ze względu na optymalne połączenie interpretowalności i właściwości matematycznych.
