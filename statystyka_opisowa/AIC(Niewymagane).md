
**[NIEWYMAGANE]**

# Akaike Information Criterion (AIC) - Kryterium Informacyjne Akaike

## Kim był Hirotugu Akaike?

**Hirotugu Akaike** (1927-2009) to japoński statystyk i matematyk, który w 1973 roku wprowadził Akaike Information Criterion. Akaike pracował w Institute of Statistical Mathematics w Tokio i był pionierem w dziedzinie selekcji modeli statystycznych. Jego kryterium AIC stało się jednym z najważniejszych narzędzi w modelowaniu statystycznym i uczeniu maszynowym.

## Czym jest AIC?

**Akaike Information Criterion (AIC)** to **kryterium selekcji modeli**, które balansuje dopasowanie modelu do danych z jego złożonością. AIC pomaga wybierać między różnymi modelami, faworyzując te, które dobrze wyjaśniają dane przy użyciu jak najmniejszej liczby parametrów.

## Wzór matematyczny:

$$
AIC = 2k - 2\ln(L)
$$

Gdzie:
- $k$ = liczba parametrów w modelu
- $L$ = maksymalna wartość funkcji wiarygodności (likelihood)
- $\ln(L)$ = logarytm naturalny z wiarygodności

### Alternatywna forma (dla regresji liniowej):
$$
AIC = n \ln\left(\frac{RSS}{n}\right) + 2k
$$

Gdzie:
- $n$ = liczba obserwacji
- $RSS$ = suma kwadratów reszt (Residual Sum of Squares)
- $k$ = liczba parametrów

## Interpretacja AIC:

### Kluczowe zasady:
1. **Mniejsze AIC = lepszy model** - szukamy minimum
2. **Trade-off** między dopasowaniem a złożonością
3. **Relative comparison** - AIC ma znaczenie tylko w porównaniu z innymi modelami
4. **Penalty za parametry** - każdy dodatkowy parametr zwiększa AIC o 2

### Komponenty AIC:
- **-2ln(L)** - "kara" za złe dopasowanie (mniejsze = lepsze)
- **2k** - kara za złożoność modelu (więcej parametrów = większe AIC)

## Jak używać AIC?

### 1. Porównywanie modeli:
```
Model A: AIC = 150.5
Model B: AIC = 148.2
Model C: AIC = 152.1

Najlepszy: Model B (najniższe AIC)
```

### 2. Różnice AIC (ΔAIC):
$$
\Delta AIC_i = AIC_i - AIC_{min}
$$

**Interpretacja różnic:**
- **ΔAIC < 2** - modele równoważne
- **2 < ΔAIC < 7** - model słabszy, ale możliwy
- **4 < ΔAIC < 7** - model znacznie słabszy  
- **ΔAIC > 10** - model praktycznie bez wsparcia

### 3. Akaike Weights:
$$
w_i = \frac{\exp(-\frac{1}{2}\Delta AIC_i)}{\sum_{j=1}^{R} \exp(-\frac{1}{2}\Delta AIC_j)}
$$

Pokazuje względne prawdopodobieństwo, że model jest najlepszy.

## Przykłady praktyczne:

### Przykład 1: Wybór stopnia wielomianu
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

# Funkcja do obliczania AIC dla regresji liniowej
def calculate_aic(y_true, y_pred, n_params):
    """
    Oblicza AIC dla modelu regresji
    """
    n = len(y_true)
    mse = mean_squared_error(y_true, y_pred)
    rss = mse * n
    
    # AIC = n * ln(RSS/n) + 2k
    aic = n * np.log(rss / n) + 2 * n_params
    return aic

# Generowanie danych syntetycznych
np.random.seed(42)
n_samples = 100
x = np.linspace(0, 1, n_samples)
true_function = 2 * x + 0.5 * x**2 - 0.3 * x**3
y = true_function + np.random.normal(0, 0.1, n_samples)

# Testowanie różnych stopni wielomianu
degrees = range(1, 11)
aic_scores = []
models = []

X = x.reshape(-1, 1)

print("PORÓWNANIE MODELI WIELOMIANOWYCH")
print("=" * 50)
print(f"{'Stopień':<8} {'Parametry':<12} {'AIC':<10} {'ΔAIC':<8}")
print("-" * 50)

for degree in degrees:
    # Tworzenie features wielomianowych
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly_features.fit_transform(X)
    
    # Trenowanie modelu
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Predykcja
    y_pred = model.predict(X_poly)
    
    # Liczba parametrów (współczynniki wielomianu + intercept)
    n_params = X_poly.shape[1]
    
    # Obliczanie AIC
    aic = calculate_aic(y, y_pred, n_params)
    aic_scores.append(aic)
    models.append((degree, model, poly_features))

# Znajdowanie najlepszego modelu
best_idx = np.argmin(aic_scores)
min_aic = min(aic_scores)

for i, (degree, aic) in enumerate(zip(degrees, aic_scores)):
    delta_aic = aic - min_aic
    n_params = degree + 1  # Stopień + intercept
    marker = " ←" if i == best_idx else ""
    print(f"{degree:<8} {n_params:<12} {aic:<10.2f} {delta_aic:<8.2f}{marker}")

print(f"\nNajlepszy model: stopień {degrees[best_idx]} (AIC = {min_aic:.2f})")
```

### Przykład 2: Selekcja zmiennych w regresji
```python
from sklearn.datasets import make_regression
from itertools import combinations

# Generowanie danych z wieloma zmiennymi
X_full, y = make_regression(n_samples=200, n_features=8, n_informative=4, 
                           noise=0.1, random_state=42)

feature_names = [f'X{i+1}' for i in range(X_full.shape[1])]

def evaluate_model_subset(X, y, feature_indices):
    """Ocenia model dla podzbioru zmiennych"""
    X_subset = X[:, feature_indices]
    
    # Dodanie interceptu
    n_samples = X_subset.shape[0]
    X_with_intercept = np.column_stack([np.ones(n_samples), X_subset])
    
    # Regresja liniowa (normal equations)
    coeffs = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    y_pred = X_with_intercept @ coeffs
    
    # Liczba parametrów = zmienne + intercept
    n_params = len(feature_indices) + 1
    aic = calculate_aic(y, y_pred, n_params)
    
    return aic, coeffs

print("\n" + "=" * 60)
print("SELEKCJA ZMIENNYCH UŻYWAJĄC AIC")
print("=" * 60)

# Testowanie wszystkich kombinacji zmiennych (do 4 zmiennych)
all_results = []
n_features = X_full.shape[1]

for n_vars in range(1, min(5, n_features + 1)):  # Testuj do 4 zmiennych
    for feature_combo in combinations(range(n_features), n_vars):
        aic, coeffs = evaluate_model_subset(X_full, y, feature_combo)
        
        feature_names_combo = [feature_names[i] for i in feature_combo]
        all_results.append({
            'features': feature_names_combo,
            'n_vars': n_vars,
            'aic': aic,
            'feature_indices': feature_combo
        })

# Sortowanie według AIC
all_results.sort(key=lambda x: x['aic'])

print(f"{'Rank':<5} {'Variables':<20} {'N_vars':<7} {'AIC':<10} {'ΔAIC':<8}")
print("-" * 60)

min_aic = all_results[0]['aic']
for i, result in enumerate(all_results[:10]):  # Pokaż top 10
    delta_aic = result['aic'] - min_aic
    vars_str = ', '.join(result['features'])
    print(f"{i+1:<5} {vars_str:<20} {result['n_vars']:<7} {result['aic']:<10.2f} {delta_aic:<8.2f}")
```

## AIC vs inne kryteria informacyjne:

### 1. BIC (Bayesian Information Criterion):
$$
BIC = k \ln(n) - 2\ln(L)
$$

**Różnice AIC vs BIC:**
- **BIC** bardziej karze złożone modele (gdy n > 7)
- **AIC** lepsze dla predykcji
- **BIC** lepsze dla selekcji "prawdziwego" modelu

### 2. AICc (corrected AIC):
$$
AICc = AIC + \frac{2k(k+1)}{n-k-1}
$$

**Użycie AICc:**
- Małe próby (n/k < 40)
- Korekta dla finite sample bias

### Porównanie kryteriów:
```python
def compare_information_criteria(y_true, y_pred, n_params, n_samples):
    """Porównuje AIC, BIC i AICc"""
    
    # AIC
    aic = calculate_aic(y_true, y_pred, n_params)
    
    # BIC
    mse = mean_squared_error(y_true, y_pred)
    rss = mse * n_samples
    bic = n_samples * np.log(rss / n_samples) + n_params * np.log(n_samples)
    
    # AICc
    if n_samples - n_params - 1 > 0:
        aicc = aic + (2 * n_params * (n_params + 1)) / (n_samples - n_params - 1)
    else:
        aicc = np.inf
    
    return {
        'AIC': aic,
        'BIC': bic,
        'AICc': aicc,
        'n_ratio': n_samples / n_params
    }

# Przykład porównania
results = compare_information_criteria(y, y_pred, 3, len(y))
print(f"\nPorównanie kryteriów (n={len(y)}, k=3):")
for criterion, value in results.items():
    if criterion != 'n_ratio':
        print(f"{criterion}: {value:.2f}")
print(f"n/k ratio: {results['n_ratio']:.1f}")
```

## AIC w szeregach czasowych:

### Model ARIMA selection:
```python
# Przykład selekcji modelu ARIMA używając AIC
def find_best_arima_aic(ts_data, max_p=3, max_d=2, max_q=3):
    """Znajduje najlepszy model ARIMA na podstawie AIC"""
    
    try:
        from statsmodels.tsa.arima.model import ARIMA
        import itertools
        
        # Generowanie kombinacji parametrów
        p_values = range(0, max_p + 1)
        d_values = range(0, max_d + 1) 
        q_values = range(0, max_q + 1)
        
        combinations = list(itertools.product(p_values, d_values, q_values))
        
        results = []
        
        for p, d, q in combinations:
            try:
                model = ARIMA(ts_data, order=(p, d, q))
                fitted = model.fit()
                
                results.append({
                    'order': (p, d, q),
                    'aic': fitted.aic,
                    'bic': fitted.bic,
                    'params': p + q + (1 if fitted.model.trend else 0)
                })
                
            except Exception as e:
                continue
        
        # Sortowanie według AIC
        results.sort(key=lambda x: x['aic'])
        
        print("TOP 5 MODELI ARIMA (według AIC):")
        print("-" * 45)
        print(f"{'Model':<12} {'AIC':<10} {'BIC':<10} {'ΔAIC':<8}")
        print("-" * 45)
        
        min_aic = results[0]['aic'] if results else float('inf')
        
        for i, result in enumerate(results[:5]):
            delta_aic = result['aic'] - min_aic
            order_str = f"ARIMA{result['order']}"
            print(f"{order_str:<12} {result['aic']:<10.2f} {result['bic']:<10.2f} {delta_aic:<8.2f}")
        
        return results[0] if results else None
        
    except ImportError:
        print("Statsmodels nie zainstalowane. Przykład teoretyczny:")
        
        # Symulacja wyników
        mock_results = [
            {'order': (1, 1, 1), 'aic': 245.2, 'bic': 252.1},
            {'order': (2, 1, 0), 'aic': 246.8, 'bic': 251.3},
            {'order': (0, 1, 2), 'aic': 247.5, 'bic': 252.4}
        ]
        
        print("PRZYKŁADOWE WYNIKI:")
        print("-" * 35)
        min_aic = min(r['aic'] for r in mock_results)
        
        for result in mock_results:
            delta_aic = result['aic'] - min_aic
            order_str = f"ARIMA{result['order']}"
            print(f"{order_str:<12} AIC: {result['aic']:.1f} (Δ={delta_aic:.1f})")
        
        return mock_results[0]

# Symulacja szeregu czasowego
np.random.seed(42)
ts_data = np.cumsum(np.random.randn(100)) + 0.1 * np.arange(100)

best_model = find_best_arima_aic(ts_data)
if best_model:
    print(f"\nNajlepszy model: ARIMA{best_model['order']}")
```

## Praktyczne zastosowania AIC:

### 1. **Machine Learning - Feature Selection:**
```python
def stepwise_selection_aic(X, y, direction='forward'):
    """Selekcja krokowa zmiennych na podstawie AIC"""
    n_features = X.shape[1]
    
    if direction == 'forward':
        selected_features = []
        remaining_features = list(range(n_features))
        
        print("FORWARD STEPWISE SELECTION:")
        print("-" * 40)
        
        while remaining_features:
            best_aic = float('inf')
            best_feature = None
            
            for feature in remaining_features:
                test_features = selected_features + [feature]
                aic, _ = evaluate_model_subset(X, y, test_features)
                
                if aic < best_aic:
                    best_aic = aic
                    best_feature = feature
            
            # Sprawdź czy dodanie poprawia model
            if not selected_features:
                # Pierwszy feature zawsze dodajemy
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                print(f"Dodano feature {best_feature}: AIC = {best_aic:.2f}")
            else:
                # Porównaj z aktualnym modelem
                current_aic, _ = evaluate_model_subset(X, y, selected_features)
                if best_aic < current_aic:
                    selected_features.append(best_feature)
                    remaining_features.remove(best_feature)
                    improvement = current_aic - best_aic
                    print(f"Dodano feature {best_feature}: AIC = {best_aic:.2f} (poprawa: {improvement:.2f})")
                else:
                    print("Brak poprawy - kończymy selekcję")
                    break
        
        return selected_features
    
    return []

# Przykład użycia
print("\n" + "=" * 50)
print("STEPWISE FEATURE SELECTION")
print("=" * 50)

selected_vars = stepwise_selection_aic(X_full, y, direction='forward')
print(f"\nWybrane zmienne: {[feature_names[i] for i in selected_vars]}")
```

### 2. **Model Comparison Dashboard:**
```python
def model_comparison_dashboard(models_data):
    """Tworzy dashboard porównawczy modeli z AIC"""
    
    # Oblicz wagi Akaike
    aic_values = [model['aic'] for model in models_data]
    min_aic = min(aic_values)
    
    for model in models_data:
        model['delta_aic'] = model['aic'] - min_aic
        model['akaike_weight'] = np.exp(-0.5 * model['delta_aic'])
    
    # Normalizuj wagi
    total_weight = sum(model['akaike_weight'] for model in models_data)
    for model in models_data:
        model['akaike_weight'] /= total_weight
    
    # Sortuj według AIC
    models_data.sort(key=lambda x: x['aic'])
    
    # Wyświetl dashboard
    print("\n" + "=" * 80)
    print("MODEL COMPARISON DASHBOARD")
    print("=" * 80)
    print(f"{'Rank':<5} {'Model':<20} {'AIC':<10} {'ΔAIC':<8} {'Weight':<8} {'Evidence':<15}")
    print("-" * 80)
    
    for i, model in enumerate(models_data):
        # Klasyfikacja evidence
        delta = model['delta_aic']
        if delta < 2:
            evidence = "Strong"
        elif delta < 4:
            evidence = "Considerable" 
        elif delta < 7:
            evidence = "Less"
        elif delta < 10:
            evidence = "Little"
        else:
            evidence = "None"
        
        print(f"{i+1:<5} {model['name']:<20} {model['aic']:<10.2f} "
              f"{model['delta_aic']:<8.2f} {model['akaike_weight']:<8.3f} {evidence:<15}")
    
    return models_data

# Przykład danych modeli
example_models = [
    {'name': 'Linear Regression', 'aic': 245.2, 'params': 3},
    {'name': 'Polynomial (deg=2)', 'aic': 242.8, 'params': 4}, 
    {'name': 'Polynomial (deg=3)', 'aic': 244.1, 'params': 5},
    {'name': 'Ridge Regression', 'aic': 246.5, 'params': 3},
    {'name': 'Random Forest', 'aic': 248.9, 'params': 10}
]

ranked_models = model_comparison_dashboard(example_models)
```

## Ograniczenia i uwagi o AIC:

### 1. **Założenia AIC:**
- Modele zagnieżdżone lub porównywalne
- Te same dane treningowe
- Duże próby (asymptotyczne właściwości)

### 2. **Kiedy AIC może zawodzić:**
```python
def aic_limitations_demo():
    """Demonstracja ograniczeń AIC"""
    print("\nOGRANICZENIA AIC:")
    print("-" * 30)
    
    # 1. Małe próby
    print("1. MAŁE PRÓBY:")
    print("   - n < 40: użyj AICc zamiast AIC")
    print("   - n/k < 40: korekta na małe próby")
    
    # 2. Overfitting
    print("\n2. OVERFITTING:")
    print("   - AIC może faworyzować złożone modele")
    print("   - Rozważ cross-validation")
    
    # 3. Różne dane
    print("\n3. RÓŻNE ZBIORY DANYCH:")
    print("   - AIC porównuje tylko modele na tych samych danych")
    print("   - Nie można porównać AIC między różnymi zbiorami")
    
    # 4. Non-nested models
    print("\n4. MODELE NIEZAGNIEŻDŻONE:")
    print("   - AIC może być nierzetelne")
    print("   - Lepiej użyć cross-validation")

aic_limitations_demo()
```

### 3. **Best Practices:**
```python
def aic_best_practices():
    """Najlepsze praktyki używania AIC"""
    
    practices = [
        "1. Zawsze porównuj relatywnie (ΔAIC), nigdy absolutnie",
        "2. Użyj AICc dla małych prób (n/k < 40)",
        "3. Łącz z cross-validation dla pewności", 
        "4. Sprawdź założenia modelu, nie tylko AIC",
        "5. Rozważ interpretowalnośc, nie tylko dopasowanie",
        "6. Użyj AIC do pre-selekcji, CV do final validation",
        "7. Dokumentuj cały proces selekcji modeli"
    ]
    
    print("\nBEST PRACTICES AIC:")
    print("-" * 40)
    for practice in practices:
        print(f"  {practice}")

aic_best_practices()
```

## Zastosowania AIC w różnych dziedzinach:

### 1. **Ekonometria:**
- Selekcja modeli ARIMA
- Wybór zmiennych w modelach ekonomicznych
- Testowanie hipotez strukturalnych

### 2. **Biologia/Medycyna:**
- Modele przeżycia
- Analiza dawka-odpowiedź  
- Selekcja biomarkerów

### 3. **Machine Learning:**
- Feature selection
- Hyperparameter tuning
- Model ensembling weights

### 4. **Finanse:**
- Modele ryzyka
- Prognozowanie volatility
- Portfolio optimization

## Podsumowanie AIC:

| Właściwość | Opis |
|------------|------|
| **Cel** | Balansuje dopasowanie vs złożonośc |
| **Interpretacja** | Mniejsze = lepsze (tylko względnie) |
| **Zalety** | Penalizuje overfitting, łatwe w użyciu |
| **Wady** | Asymptotyczne, może faworyzować złożonośc |
| **Użycie** | Selekcja modeli, feature selection |

**AIC jest fundamentalnym narzędziem** w modelowaniu statystycznym - nie podaje "prawdziwej odpowiedzi", ale pomaga w inteligentnej selekcji między alternatywami, balansując złożonośc z jakośćią dopasowania.
