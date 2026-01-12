# Modele Prognozowania Szeregów Czasowych

## 1. Średnie Kroczące (Moving Average, MA)

**Definicja:** Metoda wygładzania szeregu czasowego przez uśrednianie wartości z określonego okna czasowego.

### Rodzaje średnich kroczących:

#### 1.1 Prosta Średnia Krocząca (Simple Moving Average, SMA)
- Wszystkie obserwacje w oknie mają **równą wagę**
- Wzór:

```math
\text{SMA}_t = \frac{1}{k}(y_t + y_{t-1} + ... + y_{t-k+1})
```
- Najprostsza forma, równe traktowanie wszystkich wartości

#### 1.2 Ważona Średnia Krocząca (Weighted Moving Average, WMA)
- Nowsze obserwacje mają **większą wagę** niż starsze
- Wzór:

```math
\text{WMA}_t = \frac{w_1 y_t + w_2 y_{t-1} + ... + w_k y_{t-k+1}}{w_1 + w_2 + ... + w_k}
```
- Wagi często przypisywane liniowo: `w_i = k - i + 1`
- Przykład (k=3): najnowsza obserwacja waga 3, druga waga 2, trzecia waga 1

#### 1.3 Wykładnicza Średnia Krocząca (Exponential Moving Average, EMA)
- Wagi maleją **wykładniczo** w przeszłość
- Wzór:

```math
\text{EMA}_t = \alpha \cdot y_t + (1-\alpha) \cdot \text{EMA}_{t-1}
```
- α (alpha) - współczynnik wygładzania (0 < α < 1)
- Im większe α, tym większy wpływ najnowszych danych
- Popularna w analizie technicznej na giełdzie

#### 1.4 Centrowana Średnia Krocząca (Centered Moving Average, CMA)
- Używa obserwacji **przed i po** danym punkcie
- Wzór:

```math
\text{CMA}_t = \frac{1}{k}(y_{t-\lfloor k/2 \rfloor} + ... + y_t + ... + y_{t+\lfloor k/2 \rfloor})
```
- Lepsze wygładzanie, ale **nie nadaje się do prognozowania** (wymaga przyszłych danych)
- Stosowana głównie do analizy historycznej i dekompozycji

**Zalety:**
- Bardzo prosta implementacja
- Redukuje szum w danych
- Dobra do wygładzania krótkoterminowych fluktuacji
- Różne typy pozwalają na dostosowanie do potrzeb

**Wady:**
- Nie radzi sobie z trendem i sezonowością
- SMA: wszystkie obserwacje mają równą wagę (opóźnienie)
- Wymaga wystarczającej historii danych
- Opóźnienie (lag) w reagowaniu na zmiany

**Zastosowanie:** Wygładzanie danych, redukcja szumu, analiza krótkoterminowych trendów, analiza techniczna

---

## 2. Wygładzanie Wykładnicze (Exponential Smoothing)

**Definicja:** Metoda prognozowania, która przypisuje wykładniczo malejące wagi starszym obserwacjom.

**Rodzaje:**
- **Simple ES** - dane bez trendu i sezonowości
- **Double ES (Holt)** - dane z trendem
- **Triple ES (Holt-Winters)** - dane z trendem i sezonowością

**Jak działa:**

```math
\hat{y}_{t+1} = \alpha \cdot y_t + (1-\alpha) \cdot \hat{y}_t
```

- α (alpha) - parametr wygładzania (0-1)
- Nowsze obserwacje mają większą wagę

**Zalety:**
- Adaptuje się do zmian w danych
- Prosty do implementacji
- Małe wymagania obliczeniowe
- Holt-Winters obsługuje trend i sezonowość

**Wady:**
- Dobór parametrów może być trudny
- Słaba dla długoterminowych prognoz

**Zastosowanie:** Prognozy krótko- i średnioterminowe, dane z trendem/sezonowością

---

## 3. ARIMA (AutoRegressive Integrated Moving Average)

**Definicja:** Model statystyczny łączący autoregresję (AR), różnicowanie (I) i średnie kroczące (MA).

**Parametry modelu ARIMA(p, d, q):**
- **p** - rząd autoregresji (liczba przeszłych wartości)
- **d** - rząd różnicowania (ile razy różnicujemy, by uzyskać stacjonarność)
- **q** - rząd średniej kroczącej (liczba przeszłych błędów)

**Jak działa:**

```math
y_t = c + \phi_1 y_{t-1} + ... + \phi_p y_{t-p} + \theta_1 \varepsilon_{t-1} + ... + \theta_q \varepsilon_{t-q} + \varepsilon_t
```

**Wymaganie:** Szereg musi być **stacjonarny** (stała średnia, wariancja)

**Zalety:**
- Solidne fundamenty statystyczne
- Dobry dla danych bez wyraźnej sezonowości
- Wiele narzędzi do diagnostyki modelu

**Wady:**
- Wymaga stacjonarności
- Dobór parametrów może być trudny
- Nie radzi sobie z sezonowością (potrzebny SARIMA)
- Słaby dla danych nieliniowych

**Zastosowanie:** Prognozy ekonomiczne, finansowe, dane bez sezonowości

---

## 4. SARIMA (Seasonal ARIMA)

**Definicja:** Rozszerzenie ARIMA uwzględniające sezonowość w danych.

**Parametry modelu SARIMA(p, d, q)(P, D, Q, s):**
- **(p, d, q)** - parametry niesezonowe (jak w ARIMA)
- **(P, D, Q)** - parametry sezonowe
- **s** - długość sezonu (np. 12 dla danych miesięcznych)

**Jak działa:**
- Łączy składnik niesezonowy ARIMA z sezonowym
- Modeluje powtarzające się wzorce

**Zalety:**
- Obsługuje sezonowość
- Elastyczny - wiele konfiguracji
- Dobry dla regularnych wzorców sezonowych

**Wady:**
- Wiele parametrów do doboru
- Wymaga długiej historii danych (min. 2 sezony)
- Czasochłonne obliczeniowo
- Trudniejszy w interpretacji

**Zastosowanie:** Dane ze stabilną sezonowością (sprzedaż, turystyka, energia)

---

## 5. VAR (Vector AutoRegression)

**Definicja:** Model dla **wielu** szeregów czasowych jednocześnie, uwzględniający wzajemne zależności między nimi.

**Jak działa:**
- Każda zmienna jest modelowana jako funkcja przeszłych wartości **wszystkich** zmiennych

```math
\mathbf{y}_t = \mathbf{c} + \mathbf{A}_1 \mathbf{y}_{t-1} + ... + \mathbf{A}_p \mathbf{y}_{t-p} + \mathbf{\varepsilon}_t
```

gdzie **y**_t - wektor zmiennych w czasie t

**Przykład:** Prognozowanie PKB, inflacji i stóp procentowych jednocześnie

**Zalety:**
- Modeluje wzajemne zależności między szeregami
- Nie wymaga określania, która zmienna jest zależna/niezależna
- Dobry do analizy impulsów (jak zmiana jednej zmiennej wpływa na inne)

**Wady:**
- Wymaga dużo danych
- Wiele parametrów do estymacji
- Wszystkie szeregi muszą być stacjonarne
- Trudny w interpretacji dla wielu zmiennych

**Zastosowanie:** Ekonomia, finanse (modelowanie powiązań między zmiennymi makroekonomicznymi)

---

## 6. LSTM (Long Short-Term Memory)

**Definicja:** Typ rekurencyjnej sieci neuronowej (RNN) zaprojektowany do modelowania sekwencji i długoterminowych zależności.

**Jak działa:**
- Sieć neuronowa z pamięcią krótko- i długoterminową
- Mechanizmy "bram" (gates) kontrolują przepływ informacji:
  - **Forget gate** - co zapomnieć
  - **Input gate** - co zapamiętać
  - **Output gate** - co wyprowadzić
- Uczy się nieliniowych wzorców z danych

**Zalety:**
- Radzi sobie z nieliniowościami
- Modeluje długoterminowe zależności
- Elastyczny - różne architektury
- Dobry dla złożonych wzorców
- Może uwzględniać zmienne zewnętrzne (features)

**Wady:**
- Wymaga dużo danych treningowych
- Długi czas trenowania
- "Czarna skrzynka" - trudna interpretacja
- Wymaga doboru hiperparametrów
- Ryzyko przeuczenia (overfitting)

**Zastosowanie:** Złożone szeregi (ceny akcji, prognoza popytu, rozpoznawanie mowy), dane z wieloma zmiennymi

---

## 7. Facebook Prophet

**Definicja:** Model opracowany przez Facebook, zaprojektowany do łatwego prognozowania szeregów czasowych z silną sezonowością.

**Jak działa:**
- Model addytywny:

```math
y(t) = g(t) + s(t) + h(t) + \varepsilon_t
```

  - g(t) - trend (liniowy lub logistyczny)
  - s(t) - sezonowość (roczna, tygodniowa, dzienna)
  - h(t) - efekt świąt/wydarzeń
  - ε_t - błąd

**Zalety:**
- Bardzo prosty w użyciu (automatyczny)
- Doskonale radzi sobie z sezonowością
- Obsługuje brakujące dane
- Można dodać własne święta/wydarzenia
- Odporne na anomalie (outliers)
- Nie wymaga równych odstępów czasowych

**Wady:**
- Mniej elastyczny niż LSTM
- Gorszy dla danych bez wyraźnej sezonowości
- "Czarna skrzynka" (mniej kontroli)
- Może być za uproszczony dla złożonych przypadków

**Zastosowanie:** Biznes (sprzedaż, ruch na stronie), dane z wieloma sezonowościami i świętami

---

## Przykłady Kodu w Python

### 1. Średnie Kroczące

```python
import pandas as pd

# Dane
data = [10, 12, 15, 13, 17, 20, 18, 22]
df = pd.DataFrame({'wartość': data})

# Średnia krocząca z oknem 3
df['MA_3'] = df['wartość'].rolling(window=3).mean()

# Prognoza = ostatnia średnia krocząca
prognoza = df['MA_3'].iloc[-1]
```

**Parametry:**
- `window` - rozmiar okna (ile obserwacji uśredniać)

---

### 2. Wygładzanie Wykładnicze

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Dane z trendem i sezonowością
data = [10, 15, 12, 18, 14, 20, 16, 22, 18, 25, 20, 28]

# Model Holt-Winters
model = ExponentialSmoothing(
    data,
    trend='add',           # trend addytywny
    seasonal='add',        # sezonowość addytywna
    seasonal_periods=4     # długość sezonu (np. kwartały)
)
fit = model.fit()

# Prognoza na 4 kroki
forecast = fit.forecast(steps=4)
```

**Parametry:**
- `trend` - typ trendu: `'add'`, `'mul'`, `None`
- `seasonal` - typ sezonowości: `'add'`, `'mul'`, `None`
- `seasonal_periods` - długość cyklu sezonowego
- `steps` - liczba okresów prognozy

---

### 3. ARIMA

```python
from statsmodels.tsa.arima.model import ARIMA

# Dane
data = [10, 12, 15, 13, 17, 20, 18, 22, 25, 23, 27, 30]

# Model ARIMA(p=1, d=1, q=1)
model = ARIMA(
    data,
    order=(1, 1, 1)  # (p, d, q)
)
fit = model.fit()

# Prognoza na 5 kroków
forecast = fit.forecast(steps=5)

# Podsumowanie modelu
print(fit.summary())
```

**Parametry:**
- `order=(p, d, q)`:
  - `p` - rząd autoregresji AR (przeszłe wartości)
  - `d` - rząd różnicowania (ile razy różnicować)
  - `q` - rząd średniej kroczącej MA (przeszłe błędy)
- `steps` - liczba okresów prognozy

---

### 4. SARIMA

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Dane miesięczne (12 miesięcy sezonowości)
data = [10, 15, 12, 18, 14, 20, 16, 22, 18, 25, 20, 28] * 3  # 3 lata

# Model SARIMA(1,1,1)(1,1,1,12)
model = SARIMAX(
    data,
    order=(1, 1, 1),              # niesezonowe (p, d, q)
    seasonal_order=(1, 1, 1, 12)  # sezonowe (P, D, Q, s)
)
fit = model.fit()

# Prognoza
forecast = fit.forecast(steps=12)
```

**Parametry:**
- `order=(p, d, q)` - parametry niesezonowe (jak ARIMA)
- `seasonal_order=(P, D, Q, s)`:
  - `P` - sezonowa autoregresja
  - `D` - sezonowe różnicowanie
  - `Q` - sezonowa średnia krocząca
  - `s` - długość sezonu (np. 12 dla miesięcy, 4 dla kwartałów)

---

### 5. VAR

```python
from statsmodels.tsa.api import VAR
import pandas as pd

# Dane dla 2 szeregów czasowych
data = pd.DataFrame({
    'PKB': [100, 102, 105, 103, 107, 110, 108, 112],
    'Inflacja': [2.0, 2.1, 2.3, 2.2, 2.4, 2.5, 2.6, 2.7]
})

# Model VAR
model = VAR(data)
fit = model.fit(maxlags=2)  # max 2 opóźnienia

# Prognoza
forecast = fit.forecast(data.values, steps=3)
```

**Parametry:**
- `maxlags` - maksymalna liczba opóźnień do rozważenia
- `steps` - liczba okresów prognozy
- Dane: DataFrame z wieloma kolumnami (szeregami)

---

### 6. LSTM

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Przygotowanie danych (przykład)
data = np.array([10, 12, 15, 13, 17, 20, 18, 22, 25, 23])

# Reshape: (samples, timesteps, features)
X = data[:-1].reshape(-1, 1, 1)  # 9 próbek, 1 timestep, 1 feature
y = data[1:]                      # następne wartości

# Model LSTM
model = Sequential([
    LSTM(50, activation='relu', input_shape=(1, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Trenowanie
model.fit(X, y, epochs=100, verbose=0)

# Prognoza
X_test = np.array([[23]]).reshape(1, 1, 1)
prognoza = model.predict(X_test)
```

**Parametry:**
- `units` (50) - liczba neuronów w warstwie LSTM
- `activation` - funkcja aktywacji (`'relu'`, `'tanh'`)
- `input_shape` - (timesteps, features)
- `epochs` - liczba epok trenowania
- `optimizer` - optymalizator (`'adam'`, `'sgd'`)
- `loss` - funkcja straty (`'mse'`, `'mae'`)

---

### 7. Facebook Prophet

```python
from prophet import Prophet
import pandas as pd

# Dane (wymagane kolumny: 'ds' i 'y')
data = pd.DataFrame({
    'ds': pd.date_range('2020-01-01', periods=100, freq='D'),
    'y': [10 + i*0.5 + np.random.randn() for i in range(100)]
})

# Model Prophet
model = Prophet(
    yearly_seasonality=True,   # sezonowość roczna
    weekly_seasonality=True,   # sezonowość tygodniowa
    daily_seasonality=False    # sezonowość dzienna
)

# Trenowanie
model.fit(data)

# Przyszłe daty
future = model.make_future_dataframe(periods=30)  # 30 dni

# Prognoza
forecast = model.predict(future)

# Wyniki
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
```

**Parametry:**
- `yearly_seasonality` - sezonowość roczna (True/False)
- `weekly_seasonality` - sezonowość tygodniowa (True/False)
- `daily_seasonality` - sezonowość dzienna (True/False)
- `periods` - liczba okresów prognozy
- `freq` - częstotliwość ('D'=dzień, 'W'=tydzień, 'M'=miesiąc)

**Wyniki prognozy:**
- `yhat` - prognozowana wartość
- `yhat_lower` - dolna granica przedziału ufności
- `yhat_upper` - górna granica przedziału ufności

---
