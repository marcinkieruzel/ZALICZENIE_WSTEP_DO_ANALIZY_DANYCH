
<span style="color:red">**[NIEWYMAGANE]**</span>


# Bayesian Information Criterion (BIC) - Bayesowskie Kryterium Informacyjne

## Kim był Gideon Schwarz?

**Gideon Schwarz** (1933-2007) to amerykański statystyk, który w 1978 roku wprowadził Bayesian Information Criterion, czasem nazywane również Schwarz Criterion (SC). Schwarz pracował jako profesor statystyki na Bell Labs i później na Hebrew University. BIC powstało jako bayesowska alternatywa dla AIC Akaike, kładąc większy nacisk na prostotę modelu.

## Czym jest BIC?

**Bayesian Information Criterion (BIC)** to **kryterium selekcji modeli** oparte na podejściu bayesowskim, które jeszcze mocniej niż AIC penalizuje złożonośc modelu. BIC dąży do znalezienia "prawdziwego" modelu generującego dane, podczas gdy AIC koncentruje się na najlepszej predykcji.

## Wzór matematyczny:

$$
BIC = k \ln(n) - 2\ln(L)
$$

Gdzie:
- $k$ = liczba parametrów w modelu
- $n$ = liczba obserwacji
- $L$ = maksymalna wartość funkcji wiarygodności (likelihood)
- $\ln(L)$ = logarytm naturalny z wiarygodności

### Alternatywna forma (dla regresji liniowej):
$$
BIC = n \ln\left(\frac{RSS}{n}\right) + k \ln(n)
$$

Gdzie:
- $RSS$ = suma kwadratów reszt (Residual Sum of Squares)

## Porównanie BIC vs AIC:

### Kluczowe różnice:
$$
\begin{aligned}
AIC &= 2k - 2\ln(L) \\
BIC &= k \ln(n) - 2\ln(L)
\end{aligned}
$$

| Właściwość | AIC | BIC |
|------------|-----|-----|
| **Penalty za parametr** | $2$ | $\ln(n)$ |
| **Kiedy $\ln(n) > 2$** | $n > e^2 \approx 7.4$ | BIC bardziej restrykcyjny |
| **Kiedy $\ln(n) < 2$** | $n < 7.4$ | AIC bardziej restrykcyjny |
| **Cel** | Najlepsza predykcja | "Prawdziwy" model |
| **Podejście** | Teoretyko-informacyjne | Bayesowskie |

### Penalty comparison:
```python
import numpy as np
import matplotlib.pyplot as plt

# Porównanie penalty dla różnych wielkości próby
n_values = np.arange(5, 1000)
aic_penalty = np.full_like(n_values, 2, dtype=float)
bic_penalty = np.log(n_values)

plt.figure(figsize=(10, 6))
plt.plot(n_values, aic_penalty, label='AIC penalty = 2', linewidth=2)
plt.plot(n_values, bic_penalty, label='BIC penalty = ln(n)', linewidth=2)
plt.axhline(y=2, color='red', linestyle='--', alpha=0.7)
plt.axvline(x=np.exp(2), color='red', linestyle='--', alpha=0.7, 
           label=f'n = e² ≈ {np.exp(2):.1f}')
plt.xlabel('Wielkość próby (n)')
plt.ylabel('Penalty za parametr')
plt.title('Porównanie kar AIC vs BIC')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(5, 200)
plt.ylim(0, 8)
plt.show()

print(f"BIC = AIC gdy n = e² ≈ {np.exp(2):.1f}")
print(f"BIC > AIC gdy n > {np.exp(2):.1f} (BIC bardziej restrykcyjny)")
print(f"BIC < AIC gdy n < {np.exp(2):.1f} (AIC bardziej restrykcyjny)")
```

## Interpretacja BIC:

### Kluczowe zasady:
1. **Mniejsze BIC = lepszy model** - szukamy minimum
2. **Silniejsza penalty** za złożonośc niż AIC
3. **Relative comparison** - porównujemy tylko między modelami
4. **Asymptotyczna konsystentność** - wybierze "prawdziwy" model gdy n→∞

### Różnice BIC (ΔBIC):
$$
\Delta BIC_i = BIC_i - BIC_{min}
$$

**Interpretacja różnic Bayesian Evidence:**
- **ΔBIC < 2** - słaba evidence przeciwko modelowi
- **2 < ΔBIC < 6** - pozytywna evidence przeciwko
- **6 < ΔBIC < 10** - silna evidence przeciwko  
- **ΔBIC > 10** - bardzo silna evidence przeciwko

## Przykłady praktyczne:

### Przykład 1: Porównanie AIC vs BIC w selekcji modeli
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

def calculate_ic(y_true, y_pred, n_params, n_samples, criterion='both'):
    """
    Oblicza AIC i/lub BIC dla modelu regresji
    """
    mse = mean_squared_error(y_true, y_pred)
    rss = mse * n_samples
    
    # Logarithmic likelihood (dla regresji liniowej)
    log_likelihood = -0.5 * n_samples * (np.log(2 * np.pi) + np.log(rss/n_samples) + 1)
    
    results = {}
    
    if criterion in ['aic', 'both']:
        aic = 2 * n_params - 2 * log_likelihood
        results['aic'] = aic
        
    if criterion in ['bic', 'both']:
        bic = n_params * np.log(n_samples) - 2 * log_likelihood
        results['bic'] = bic
    
    return results

# Generowanie danych z znaną funkcją
np.random.seed(42)
n_samples = 50  # Średnia próba
x = np.linspace(0, 1, n_samples)
true_function = 2 * x + 1 * x**2  # Prawdziwa funkcja: stopień 2
y = true_function + np.random.normal(0, 0.1, n_samples)

# Testowanie różnych stopni wielomianu
degrees = range(1, 8)
results = []

X = x.reshape(-1, 1)

print("PORÓWNANIE AIC vs BIC - SELEKCJA WIELOMIANU")
print("=" * 60)
print(f"Prawdziwy model: stopień 2")
print(f"Wielkość próby: {n_samples}")
print("=" * 60)
print(f"{'Stopień':<8} {'Param':<6} {'AIC':<10} {'BIC':<10} {'ΔAIC':<8} {'ΔBIC':<8}")
print("-" * 60)

for degree in degrees:
    # Tworzenie features wielomianowych
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly_features.fit_transform(X)
    
    # Trenowanie modelu
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    
    # Liczba parametrów
    n_params = X_poly.shape[1]
    
    # Obliczanie AIC i BIC
    ic_values = calculate_ic(y, y_pred, n_params, n_samples, 'both')
    
    results.append({
        'degree': degree,
        'n_params': n_params,
        'aic': ic_values['aic'],
        'bic': ic_values['bic']
    })

# Znajdowanie najlepszych modeli
best_aic_idx = np.argmin([r['aic'] for r in results])
best_bic_idx = np.argmin([r['bic'] for r in results])

min_aic = results[best_aic_idx]['aic']
min_bic = results[best_bic_idx]['bic']

for i, result in enumerate(results):
    delta_aic = result['aic'] - min_aic
    delta_bic = result['bic'] - min_bic
    
    aic_marker = " ←AIC" if i == best_aic_idx else ""
    bic_marker = " ←BIC" if i == best_bic_idx else ""
    marker = aic_marker + bic_marker
    
    print(f"{result['degree']:<8} {result['n_params']:<6} "
          f"{result['aic']:<10.2f} {result['bic']:<10.2f} "
          f"{delta_aic:<8.2f} {delta_bic:<8.2f}{marker}")

print(f"\nAIC wybiera: stopień {results[best_aic_idx]['degree']}")
print(f"BIC wybiera: stopień {results[best_bic_idx]['degree']}")
print(f"Prawdziwy model: stopień 2")
```

### Przykład 2: Wpływ wielkości próby na wybór AIC vs BIC
```python
def compare_aic_bic_sample_sizes():
    """Porównuje zachowanie AIC vs BIC dla różnych wielkości próby"""
    
    sample_sizes = [20, 50, 100, 500, 1000]
    results_summary = []
    
    for n in sample_sizes:
        # Generowanie danych z znaną funkcją (stopień 2)
        np.random.seed(42)
        x = np.linspace(0, 1, n)
        true_function = 2 * x + 1 * x**2
        y = true_function + np.random.normal(0, 0.1, n)
        
        X = x.reshape(-1, 1)
        
        # Testowanie stopni 1-5
        aic_results = []
        bic_results = []
        
        for degree in range(1, 6):
            poly_features = PolynomialFeatures(degree=degree, include_bias=True)
            X_poly = poly_features.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_poly, y)
            y_pred = model.predict(X_poly)
            
            n_params = X_poly.shape[1]
            ic_values = calculate_ic(y, y_pred, n_params, n, 'both')
            
            aic_results.append(ic_values['aic'])
            bic_results.append(ic_values['bic'])
        
        # Najlepsze modele
        best_aic_degree = np.argmin(aic_results) + 1
        best_bic_degree = np.argmin(bic_results) + 1
        
        results_summary.append({
            'n': n,
            'aic_choice': best_aic_degree,
            'bic_choice': best_bic_degree,
            'penalty_ratio': np.log(n) / 2
        })
    
    # Wyświetlanie wyników
    print("\n" + "=" * 60)
    print("WPŁYW WIELKOŚCI PRÓBY NA WYBÓR AIC vs BIC")
    print("=" * 60)
    print("Prawdziwy model: stopień 2")
    print("-" * 60)
    print(f"{'n':<6} {'ln(n)/2':<8} {'AIC wybór':<10} {'BIC wybór':<10} {'Zgodność':<10}")
    print("-" * 60)
    
    for result in results_summary:
        agreement = "✓" if result['aic_choice'] == result['bic_choice'] else "✗"
        print(f"{result['n']:<6} {result['penalty_ratio']:<8.2f} "
              f"{result['aic_choice']:<10} {result['bic_choice']:<10} {agreement:<10}")
    
    return results_summary

sample_analysis = compare_aic_bic_sample_sizes()
```

## BIC w różnych kontekstach:

### 1. Model Selection w Machine Learning:
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge

def ml_model_comparison_bic(X, y):
    """Porównuje różne modele ML używając BIC"""
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Definicja modeli z różną złożonością
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Tree_depth_3': DecisionTreeRegressor(max_depth=3, random_state=42),
        'Tree_depth_5': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Tree_depth_10': DecisionTreeRegressor(max_depth=10, random_state=42),
        'Forest_10': RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42),
        'Forest_50': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    }
    
    # Estymacja liczby efektywnych parametrów dla różnych modeli
    effective_params = {
        'Linear': X_train.shape[1] + 1,  # Features + intercept
        'Ridge': X_train.shape[1] + 1,   # Similar to linear (regularized)
        'Tree_depth_3': 2**3 - 1,        # Approximate for tree
        'Tree_depth_5': 2**5 - 1,
        'Tree_depth_10': min(2**10 - 1, len(y_train) // 10),  # Cap by sample size
        'Forest_10': 10 * (2**5 - 1) // 10,  # Approximation
        'Forest_50': 50 * (2**5 - 1) // 50   # Approximation
    }
    
    results = []
    
    print("MODEL COMPARISON - BIC ANALYSIS")
    print("=" * 50)
    print(f"Training size: {len(y_train)}")
    print(f"Test size: {len(y_test)}")
    print("-" * 50)
    print(f"{'Model':<15} {'Params':<8} {'Train RMSE':<12} {'Test RMSE':<12} {'BIC':<10}")
    print("-" * 50)
    
    for name, model in models.items():
        # Trenowanie
        model.fit(X_train, y_train)
        
        # Predykcje
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metryki
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # BIC (na zbiorze treningowym)
        n_params = effective_params[name]
        ic_values = calculate_ic(y_train, y_train_pred, n_params, len(y_train), 'bic')
        bic = ic_values['bic']
        
        results.append({
            'model': name,
            'n_params': n_params,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'bic': bic
        })
        
        print(f"{name:<15} {n_params:<8} {train_rmse:<12.4f} {test_rmse:<12.4f} {bic:<10.2f}")
    
    # Najlepszy model według BIC
    best_bic_idx = np.argmin([r['bic'] for r in results])
    best_model = results[best_bic_idx]
    
    print("-" * 50)
    print(f"Najlepszy model (BIC): {best_model['model']}")
    print(f"Test RMSE najlepszego: {best_model['test_rmse']:.4f}")
    
    return results

# Generowanie danych testowych
np.random.seed(42)
X_test = np.random.randn(200, 5)
y_test = 2*X_test[:, 0] - X_test[:, 1] + 0.5*X_test[:, 2]**2 + np.random.randn(200)*0.5

ml_results = ml_model_comparison_bic(X_test, y_test)
```

### 2. BIC w szeregach czasowych:
```python
def time_series_bic_example():
    """Przykład użycia BIC w modelowaniu szeregów czasowych"""
    
    # Symulacja różnych modeli ARIMA
    models_results = [
        {'model': 'ARIMA(1,0,0)', 'params': 2, 'log_lik': -145.2},  # AR(1) + const
        {'model': 'ARIMA(2,0,0)', 'params': 3, 'log_lik': -143.8},  # AR(2) + const  
        {'model': 'ARIMA(1,1,1)', 'params': 2, 'log_lik': -142.5},  # ARIMA + const
        {'model': 'ARIMA(2,1,1)', 'params': 3, 'log_lik': -141.9},  # More complex
        {'model': 'ARIMA(3,1,2)', 'params': 5, 'log_lik': -140.2},  # Very complex
    ]
    
    n_obs = 100
    
    print("\n" + "=" * 50)
    print("BIC W SZEREGACH CZASOWYCH - SELEKCJA ARIMA")
    print("=" * 50)
    print(f"Liczba obserwacji: {n_obs}")
    print("-" * 50)
    print(f"{'Model':<15} {'Params':<8} {'Log-Lik':<10} {'BIC':<10} {'ΔBIC':<8}")
    print("-" * 50)
    
    # Obliczanie BIC
    for model in models_results:
        bic = model['params'] * np.log(n_obs) - 2 * model['log_lik']
        model['bic'] = bic
    
    # Sortowanie według BIC
    models_results.sort(key=lambda x: x['bic'])
    min_bic = models_results[0]['bic']
    
    for i, model in enumerate(models_results):
        delta_bic = model['bic'] - min_bic
        marker = " ← Best" if i == 0 else ""
        
        print(f"{model['model']:<15} {model['params']:<8} "
              f"{model['log_lik']:<10.1f} {model['bic']:<10.2f} {delta_bic:<8.2f}{marker}")
    
    print(f"\nNajlepszy model: {models_results[0]['model']}")

time_series_bic_example()
```

## Interpretacja bayesowska BIC:

### Bayes Factor approximation:
```python
def bic_bayes_factor_interpretation():
    """Interpretacja BIC w kontekście Bayes Factor"""
    
    print("\n" + "=" * 50)
    print("BIC JAKO APROKSYMACJA BAYES FACTOR")
    print("=" * 50)
    
    # Różnice BIC vs siła evidence (Jeffrey's scale)
    evidence_scale = [
        (0, 2, "Not worth mentioning"),
        (2, 6, "Positive evidence"),  
        (6, 10, "Strong evidence"),
        (10, float('inf'), "Very strong evidence")
    ]
    
    print("ΔBIC Interpretation (Jeffrey's Scale):")
    print("-" * 40)
    for min_val, max_val, interpretation in evidence_scale:
        range_str = f"{min_val}-{max_val}" if max_val != float('inf') else f">{min_val}"
        print(f"{range_str:<8} {interpretation}")
    
    # Przykład obliczenia Bayes Factor z BIC
    print(f"\nBayes Factor ≈ exp(-0.5 × ΔBIC)")
    
    delta_bics = [0, 2, 6, 10]
    print(f"\n{'ΔBIC':<6} {'Bayes Factor':<15} {'Evidence against':<20}")
    print("-" * 45)
    
    for delta in delta_bics:
        bf = np.exp(-0.5 * delta)
        if delta == 0:
            evidence = "Equal support"
        elif delta < 2:
            evidence = "Weak"
        elif delta < 6:
            evidence = "Positive" 
        elif delta < 10:
            evidence = "Strong"
        else:
            evidence = "Very strong"
            
        print(f"{delta:<6} {bf:<15.3f} {evidence:<20}")

bic_bayes_factor_interpretation()
```

## Praktyczne wskazówki BIC:

### 1. Kiedy używać BIC zamiast AIC:
```python
def bic_vs_aic_decision_guide():
    """Przewodnik decyzyjny: kiedy używać BIC vs AIC"""
    
    decision_tree = {
        "Cel analizy": {
            "Najlepsza predykcja": "AIC",
            "Znalezienie 'prawdziwego' modelu": "BIC",
            "Exploratory analysis": "AIC",
            "Confirmatory analysis": "BIC"
        },
        "Wielkość próby": {
            "Mała (n < 40)": "AIC lub AICc", 
            "Średnia (40 < n < 200)": "Oba (porównaj)",
            "Duża (n > 200)": "BIC (bardziej konserwatywny)"
        },
        "Dziedzina": {
            "Machine Learning": "AIC (predykcja)",
            "Naukowe modelowanie": "BIC (parsimony)",
            "Finanse/biznes": "AIC (performance)",
            "Medycyna": "BIC (ostrożność)"
        },
        "Typ modelu": {
            "Zagnieżdżone modele": "Oba",
            "Niezagnieżdżone": "Cross-validation lepsze",
            "Szeregi czasowe": "BIC (tradycyjnie)",
            "Deep Learning": "Validation loss"
        }
    }
    
    print("\n" + "=" * 60)
    print("PRZEWODNIK DECYZYJNY: BIC vs AIC")
    print("=" * 60)
    
    for category, options in decision_tree.items():
        print(f"\n{category}:")
        print("-" * len(category))
        for criterion, recommendation in options.items():
            print(f"  {criterion:<25} → {recommendation}")

bic_vs_aic_decision_guide()
```

### 2. Best Practices dla BIC:
```python
def bic_best_practices():
    """Najlepsze praktyki używania BIC"""
    
    practices = [
        {
            "kategoria": "Model Selection",
            "practices": [
                "Zawsze porównuj modele na tych samych danych",
                "Użyj BIC dla sparse/parsimony model selection",
                "Kombinuj z cross-validation dla pewności",
                "Dokumentuj wszystkie testowane modele"
            ]
        },
        {
            "kategoria": "Interpretation", 
            "practices": [
                "Interpretuj ΔBIC, nie absolutne wartości",
                "Użyj Jeffrey's scale dla interpretacji", 
                "Rozważ uncertainty w selekcji (wagi Bayesian)",
                "Sprawdź stabilność wyboru na różnych próbach"
            ]
        },
        {
            "kategoria": "Technical",
            "practices": [
                "Dla małych prób rozważ exact Bayes factors",
                "W ML porównaj z holdout validation",
                "Dla non-nested models użyj cross-validation",
                "Monitoruj overfitting mimo BIC"
            ]
        }
    ]
    
    print("\n" + "=" * 60)
    print("BIC BEST PRACTICES")
    print("=" * 60)
    
    for section in practices:
        print(f"\n{section['kategoria']}:")
        print("-" * len(section['kategoria']))
        for practice in section['practices']:
            print(f"  • {practice}")

bic_best_practices()
```

## Ograniczenia BIC:

### 1. Główne ograniczenia:
```python
def bic_limitations():
    """Demonstracja ograniczeń BIC"""
    
    limitations = [
        {
            "ograniczenie": "Asymptotyczne właściwości",
            "opis": "BIC wymaga dużych prób do optymalnego działania",
            "rozwiązanie": "Użyj exact Bayes factors dla małych prób"
        },
        {
            "ograniczenie": "Assumption о 'true model'",
            "opis": "BIC zakłada istnienie prawdziwego modelu w zbiorze",
            "rozwiązanie": "Rozważ model averaging zamiast selekcji"
        },
        {
            "ograniczenie": "Overly conservative",
            "opis": "Może wybierać zbyt proste modele dla predykcji",
            "rozwiązanie": "Porównaj z cross-validation performance"
        },
        {
            "ograniczenie": "Prior sensitivity",
            "opis": "Implicit uniform priors mogą nie być odpowiednie",
            "rozwiązanie": "Użyj explicit Bayesian analysis"
        }
    ]
    
    print("\n" + "=" * 70)
    print("OGRANICZENIA BIC")
    print("=" * 70)
    
    for lim in limitations:
        print(f"\n{lim['ograniczenie']}:")
        print(f"  Problem: {lim['opis']}")
        print(f"  Rozwiązanie: {lim['rozwiązanie']}")

bic_limitations()
```

## Porównanie wszystkich kryteriów informacyjnych:

### Comprehensive comparison:
```python
def comprehensive_ic_comparison():
    """Kompleksowe porównanie kryteriów informacyjnych"""
    
    criteria_table = {
        'Criterion': ['AIC', 'BIC', 'AICc', 'DIC', 'WAIC'],
        'Penalty': ['2k', 'k×ln(n)', '2k + correction', 'Effective params', 'Effective params'],
        'Best for': ['Prediction', 'Model truth', 'Small samples', 'Hierarchical', 'General Bayesian'],
        'Philosophy': ['Frequentist', 'Bayesian', 'Frequentist', 'Bayesian', 'Bayesian'],
        'Bias': ['Overfits', 'Conservative', 'Balanced', 'Context dependent', 'Balanced']
    }
    
    df = pd.DataFrame(criteria_table)
    
    print("\n" + "=" * 80)
    print("PORÓWNANIE KRYTERIÓW INFORMACYJNYCH")
    print("=" * 80)
    
    # Display as formatted table
    col_widths = [max(len(str(item)) for item in df[col]) + 2 for col in df.columns]
    
    # Header
    header = "".join(col.ljust(width) for col, width in zip(df.columns, col_widths))
    print(header)
    print("-" * len(header))
    
    # Rows
    for _, row in df.iterrows():
        row_str = "".join(str(item).ljust(width) for item, width in zip(row, col_widths))
        print(row_str)
    
    # Recommendations
    print(f"\nREKOMENDACJE:")
    print(f"• Default choice: AIC (predykcja) lub BIC (interpretacja)")
    print(f"• Małe próby (n < 40): AICc")
    print(f"• Duże próby (n > 200): BIC (bardziej stabilny)")
    print(f"• Model uncertainty: Bayesian Model Averaging")

comprehensive_ic_comparison()
```

## Zastosowania BIC w różnych dziedzinach:

### 1. **Genetyka/Bioinformatyka:**
- Phylogenetic model selection
- QTL mapping 
- Population structure analysis

### 2. **Psychologia:**
- Latent class analysis
- Factor model selection
- Cognitive model comparison

### 3. **Ekonometria:**
- Structural break detection
- Regime switching models
- Cointegration testing

### 4. **Signal Processing:**
- Model order selection
- Change point detection
- Spectral analysis

## Podsumowanie BIC:

| Właściwość | BIC |
|------------|-----|
| **Cel** | Znalezienie "prawdziwego" modelu |
| **Penalty** | $k \ln(n)$ (mocniejsza niż AIC gdy $n > 7$) |
| **Zalety** | Konsystentny, konserwatywny, bayesowski |
| **Wady** | Może być zbyt restrykcyjny, asymptotyczny |
| **Idealne użycie** | Duże próby, scientific modeling, parsimony |

**BIC jest kluczowym narzędziem** gdy celem jest znalezienie najbardziej prawdopodobnego modelu generującego dane, szczególnie w kontekście naukowym gdzie prostota i interpretowalność są priorytetami nad maksymalną predykcyjnością.
