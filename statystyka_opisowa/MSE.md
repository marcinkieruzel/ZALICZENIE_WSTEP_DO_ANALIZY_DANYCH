# Mean Squared Error (MSE) - Średni Błąd Kwadratowy

## Czym jest MSE?

**Mean Squared Error (MSE)** to **metryka oceny jakości predykcji**, która mierzy średnią kwadratów różnic między wartościami przewidywanymi a rzeczywistymi. Jest to jedna z najważniejszych metryk w uczeniu maszynowym, szczególnie często używana w regresji liniowej i algorytmach optymalizacji.

## Wzór matematyczny:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Gdzie:
- $n$ = liczba obserwacji
- $y_i$ = rzeczywista wartość dla obserwacji $i$
- $\hat{y}_i$ = przewidywana wartość dla obserwacji $i$
- $(...)^2$ = podnoszenie do kwadratu

## Interpretacja:

### Zalety MSE:
1. **Różniczkowalność** - łatwa optymalizacja w algorytmach gradientowych
2. **Penalizacja dużych błędów** - kwadrowanie wzmacnia wpływ outliers
3. **Matematyczna elegancja** - łączy się z wieloma teoriami statystycznymi
4. **Standardowa metryka** - szeroko akceptowana w literaturze naukowej

### Wady MSE:
1. **Trudność interpretacji** - wynik w jednostkach kwadratowych
2. **Wrażliwość na outliers** - pojedyncze duże błędy dominują wynik
3. **Nierówne traktowanie błędów** - duże błędy nieproporcjonalnie karane
4. **Skala zależna** - trudne porównywanie między różnymi zbiorami danych

## Powiązane metryki:

### RMSE (Root Mean Squared Error):
$$
RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$
- **RMSE** ma te same jednostki co oryginalne dane
- **RMSE** zachowuje właściwości MSE ale jest łatwiejsze do interpretacji

### Porównanie MSE vs MAE:
```
MSE = (1/n) * Σ(yi - ŷi)²
MAE = (1/n) * Σ|yi - ŷi|
```
- **MSE** bardziej karze duże błędy
- **MAE** traktuje wszystkie błędy równomiernie
- **MSE** lepsze gdy outliers są problemem do rozwiązania
- **MAE** lepsze gdy outliers są naturalną częścią danych

## Kiedy używać MSE?

### Idealne przypadki:
1. **Gdy duże błędy są krytyczne** - MSE mocno je karze
2. **Optymalizacja algorytmów** - MSE jest różniczkowalne
3. **Modele probabilistyczne** - MSE związane z maksymalizacją wiarygodności
4. **Porównywanie modeli** - standardowa metryka w konkursach ML

### Unikaj MSE gdy:
1. **Outliers są naturalnie występujące** - użyj MAE
2. **Potrzebujesz intuicyjnej interpretacji** - użyj RMSE lub MAE
3. **Dane mają różne skale** - znormalizuj lub użyj metryk względnych

## Przykłady praktyczne:

### Przykład 1: Prognozowanie temperatury
```
Rzeczywiste temp: [20°C, 25°C, 30°C, 22°C]
Przewidywane:     [22°C, 24°C, 35°C, 21°C]

Błędy: [2, 1, 5, 1]
Błędy²: [4, 1, 25, 1]

MSE = (4 + 1 + 25 + 1) / 4 = 7.75°C²
RMSE = √7.75 = 2.78°C

Interpretacja: Duży błąd 5°C dominuje wynik
```

### Przykład 2: Ceny mieszkań (w tysiącach)
```
Rzeczywiste: [300k, 250k, 400k, 350k]
Przewidywane: [290k, 260k, 420k, 340k]

Błędy: [10k, -10k, -20k, 10k]
Błędy²: [100, 100, 400, 100]k²

MSE = (100 + 100 + 400 + 100) / 4 = 175k²
RMSE = √175 ≈ 13.2k

Interpretacja: Średni błąd około 13,200 zł
```

## Implementacja w Python:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

# Sposób 1: Własna implementacja
def mse_custom(y_true, y_pred):
    """Własna implementacja MSE"""
    return np.mean((y_true - y_pred) ** 2)

def rmse_custom(y_true, y_pred):
    """Własna implementacja RMSE"""
    return np.sqrt(mse_custom(y_true, y_pred))

# Sposób 2: Sklearn
from sklearn.metrics import mean_squared_error

# Przykładowe dane
y_true = np.array([3.0, 2.5, 0.5, 4.0, 5.0])
y_pred = np.array([2.8, 2.7, 0.3, 3.8, 5.2])

# Obliczenie MSE
mse_sklearn = mean_squared_error(y_true, y_pred)
mse_own = mse_custom(y_true, y_pred)
rmse_sklearn = np.sqrt(mse_sklearn)

print(f"MSE (sklearn): {mse_sklearn:.4f}")
print(f"MSE (własne): {mse_own:.4f}")
print(f"RMSE: {rmse_sklearn:.4f}")

# Porównanie MSE vs MAE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.4f}")
print(f"MSE/MAE ratio: {mse_sklearn/mae:.2f}")

# Przykład z outlierami
print("\n" + "="*50)
print("WPŁYW OUTLIERÓW")
print("="*50)

# Dane bez outliera
y_true_clean = np.array([10, 12, 11, 13, 9])
y_pred_clean = np.array([11, 11, 12, 12, 10])

# Dane z outlierem
y_true_outlier = np.array([10, 12, 11, 13, 9])
y_pred_outlier = np.array([11, 11, 12, 25, 10])  # outlier: 25 zamiast 12

mse_clean = mean_squared_error(y_true_clean, y_pred_clean)
mae_clean = mean_absolute_error(y_true_clean, y_pred_clean)

mse_outlier = mean_squared_error(y_true_outlier, y_pred_outlier)
mae_outlier = mean_absolute_error(y_true_outlier, y_pred_outlier)

print(f"Bez outliera - MSE: {mse_clean:.2f}, MAE: {mae_clean:.2f}")
print(f"Z outlierem  - MSE: {mse_outlier:.2f}, MAE: {mae_outlier:.2f}")
print(f"Wzrost MSE: {mse_outlier/mse_clean:.1f}x")
print(f"Wzrost MAE: {mae_outlier/mae_clean:.1f}x")

# Wizualizacja wpływu outlierów
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Wykres bez outliera
ax1.scatter(y_true_clean, y_pred_clean, color='blue', s=100)
ax1.plot([8, 14], [8, 14], 'r--', alpha=0.8, label='Idealna predykcja')
ax1.set_xlabel('Wartości rzeczywiste')
ax1.set_ylabel('Wartości przewidywane')
ax1.set_title(f'Bez outliera\nMSE = {mse_clean:.2f}, MAE = {mae_clean:.2f}')
ax1.legend()
ax1.grid(True)

# Wykres z outlierem
ax2.scatter(y_true_outlier, y_pred_outlier, color='red', s=100)
ax2.plot([8, 14], [8, 14], 'r--', alpha=0.8, label='Idealna predykcja')
# Zaznacz outlier
ax2.scatter([13], [25], color='darkred', s=200, marker='x', label='Outlier')
ax2.set_xlabel('Wartości rzeczywiste')
ax2.set_ylabel('Wartości przewidywane')
ax2.set_title(f'Z outlierem\nMSE = {mse_outlier:.2f}, MAE = {mae_outlier:.2f}')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Przykład praktyczny: regresja liniowa
print("\n" + "="*50)
print("PRZYKŁAD: REGRESJA LINIOWA")
print("="*50)

# Generowanie danych syntetycznych
np.random.seed(42)
X = np.random.randn(100, 1)
y = 2.5 * X.flatten() + 1 + np.random.randn(100) * 0.5

# Podział na zbiory
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Trening modelu
model = LinearRegression()
model.fit(X_train, y_train)

# Predykcja
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Ocena modelu
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

print(f"MSE treningowy:  {mse_train:.4f}")
print(f"MSE testowy:     {mse_test:.4f}")
print(f"RMSE treningowy: {rmse_train:.4f}")
print(f"RMSE testowy:    {rmse_test:.4f}")

# Wizualizacja wyników
plt.figure(figsize=(12, 5))

# Wykres treningowy
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.6, label='Dane treningowe')
plt.plot(X_train, y_pred_train, 'r-', label='Predykcja')
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'Zbiór treningowy\nMSE = {mse_train:.3f}')
plt.legend()
plt.grid(True)

# Wykres testowy
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, alpha=0.6, label='Dane testowe')
plt.plot(X_test, y_pred_test, 'r-', label='Predykcja')
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'Zbiór testowy\nMSE = {mse_test:.3f}')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Warianty MSE:

### 1. Weighted MSE (ważony MSE):
$$
WMSE = \frac{\sum_{i=1}^{n} w_i (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} w_i}
$$

### 2. Normalized MSE:
$$
NMSE = \frac{MSE}{\text{var}(y)} = \frac{MSE}{\frac{1}{n}\sum_{i=1}^{n}(y_i - \bar{y})^2}
$$

### 3. Relative MSE:
$$
RMSE_{rel} = \frac{MSE}{\bar{y}^2}
$$

## Dekompozycja bias-variance:

MSE można rozłożyć na trzy składniki:
$$
MSE = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
$$

```python
def bias_variance_decomposition(y_true, predictions_list):
    """
    Dekompozycja bias-variance dla MSE
    predictions_list: lista predykcji z różnych modeli/bootstrapów
    """
    predictions = np.array(predictions_list)
    mean_prediction = np.mean(predictions, axis=0)
    
    # Bias squared
    bias_squared = np.mean((mean_prediction - y_true) ** 2)
    
    # Variance
    variance = np.mean(np.var(predictions, axis=0))
    
    # Total MSE
    total_mse = np.mean([mean_squared_error(y_true, pred) for pred in predictions])
    
    # Irreducible error (approx)
    irreducible_error = total_mse - bias_squared - variance
    
    return {
        'total_mse': total_mse,
        'bias_squared': bias_squared,
        'variance': variance,
        'irreducible_error': irreducible_error
    }
```

## MSE w różnych algorytmach:

### 1. **Regresja liniowa:**
```python
# MSE jako funkcja kosztu
cost = (1/2m) * Σ(hθ(x) - y)²
```

### 2. **Sieci neuronowe:**
```python
# Backpropagation z MSE
∂MSE/∂w = (2/n) * Σ(y_pred - y_true) * ∂y_pred/∂w
```

### 3. **Gradient Descent:**
```python
def gradient_descent_mse(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    theta = np.random.randn(n)
    
    for epoch in range(epochs):
        predictions = X @ theta
        errors = predictions - y
        gradients = (2/m) * X.T @ errors
        theta -= learning_rate * gradients
    
    return theta
```

## Praktyczne wskazówki:

### 1. Wybór między MSE a MAE:
```python
def choose_metric(residuals):
    """Pomaga wybrać między MSE a MAE"""
    q75, q25 = np.percentile(np.abs(residuals), [75, 25])
    iqr = q75 - q25
    outlier_threshold = q75 + 1.5 * iqr
    outlier_count = np.sum(np.abs(residuals) > outlier_threshold)
    
    if outlier_count / len(residuals) > 0.1:
        return "MAE (dużo outlierów)"
    else:
        return "MSE (outlierów mało)"
```

### 2. Skalowanie MSE:
```python
def normalize_mse(mse, y_true):
    """Normalizuje MSE względem wariancji danych"""
    variance = np.var(y_true)
    return mse / variance

# NMSE < 0.1: Bardzo dobry model
# NMSE < 0.3: Dobry model  
# NMSE > 0.5: Słaby model
```

### 3. Cross-validation z MSE:
```python
from sklearn.model_selection import cross_val_score

def evaluate_model_cv(model, X, y, cv=5):
    """Ocena modelu z cross-validation"""
    mse_scores = -cross_val_score(model, X, y, 
                                 cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(mse_scores)
    
    print(f"MSE: {mse_scores.mean():.4f} ± {mse_scores.std():.4f}")
    print(f"RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
    
    return mse_scores
```

## Zastosowania MSE w różnych dziedzinach:

### 1. **Computer Vision:**
- Ocena jakości rekonstrukcji obrazów
- Porównywanie algorytmów kompresji
- Metryka w autoenkoderach

### 2. **Przetwarzanie sygnałów:**
- Stosunek sygnału do szumu (SNR)
- Ocena filtrów cyfrowych
- Analiza jakości transmisji

### 3. **Ekonometria:**
- Ocena modeli prognozowania ekonomicznego
- Analiza szeregów czasowych
- Walidacja modeli finansowych

### 4. **Inżynieria:**
- Sterowanie procesami
- Optymalizacja systemów
- Kontrola jakości

## Porównanie metryk - podsumowanie:

| Metryka | Jednostki | Outliers | Interpretacja | Optymalizacja |
|---------|-----------|----------|---------------|---------------|
| **MSE** | Kwadratowe | Wrażliwy | Trudna | Łatwa |
| **RMSE** | Oryginalne | Wrażliwy | Średnia | Łatwa |
| **MAE** | Oryginalne | Odporny | Łatwa | Trudniejsza |

MSE jest fundamentalną metryką w uczeniu maszynowym - szczególnie cenną gdy optymalizacja jest kluczowa i gdy duże błędy powinny być mocno penalizowane.
