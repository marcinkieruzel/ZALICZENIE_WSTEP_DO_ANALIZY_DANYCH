# Konspekt: Analiza Danych w Systemach Inteligentnych

## 1. Szeregi czasowe

### Definicja
**Szereg czasowy** (ang. *time series*) to uporządkowany ciąg obserwacji pewnej zmiennej, zmierzonych w kolejnych, równo odległych momentach czasu lub w kolejnych okresach czasu.

### Kluczowe cechy:
- **Uporządkowanie czasowe** - kolejność obserwacji ma fundamentalne znaczenie
- **Regularność pomiarów** - dane zbierane w stałych odstępach czasu (np. co godzinę, dziennie, miesięcznie)
- **Zależność czasowa** - wartości mogą być skorelowane z poprzednimi wartościami

### Notacja matematyczna:
- Szereg czasowy oznaczamy: $\{y_1, y_2, y_3, ..., y_t, ..., y_n\}$ lub $\{y_t\}$
- gdzie $t$ - indeks czasu, $n$ - liczba obserwacji

### Przykłady szeregów czasowych:
- Dzienna temperatura w danym mieście
- Miesięczna sprzedaż produktu
- Godzinowe zużycie energii elektrycznej
- Cena akcji na giełdzie (odczyty co minutę)
- Liczba użytkowników odwiedzających stronę WWW (dane dzienne)
- Sygnały medyczne (EKG, EEG)

### Zastosowania:
- **Prognozowanie** - przewidywanie przyszłych wartości
- **Analiza trendów** - wykrywanie długoterminowych zmian
- **Wykrywanie anomalii** - identyfikacja nietypowych wzorców
- **Analiza sezonowości** - badanie powtarzających się wzorców
- **Wspomaganie decyzji** - w finansach, logistyce, medycynie

---

## 2. Komponenty szeregów czasowych

### 2.1 Trend ($T_t$)

**Definicja:** Długoterminowa, systematyczna tendencja zmian wartości szeregu czasowego w określonym kierunku.

**Charakterystyka:**
- Reprezentuje ogólny kierunek rozwoju zjawiska w dłuższym okresie
- Może być rosnący, malejący lub stały (brak trendu)
- Może mieć charakter liniowy lub nieliniowy (wykładniczy, logarytmiczny, etc.)

**Przykłady:**
- Wzrost populacji kraju w ciągu dekad
- Spadek cen pamięci komputerowych w długim okresie
- Wzrost globalnej temperatury (ocieplenie klimatu)
- Rosnąca sprzedaż produktów technologicznych

**Wykrywanie:** Metody wygładzania (średnie kroczące, regresja)

### 2.2 Sezonowość ($S_t$)

**Definicja:** Regularne, powtarzające się wahania wartości szeregu czasowego, które występują w stałych odstępach czasu (cykl roczny, kwartalny, miesięczny).

**Charakterystyka:**
- Wzorce powtarzają się z określoną częstotliwością (np. co rok, co kwartał)
- Długość cyklu sezonowego jest znana i stała
- Związana z kalendarzem, porami roku, zwyczajami

**Przykłady:**
- Wzrost sprzedaży zabawek przed świętami Bożego Narodzenia
- Większe zużycie energii w miesiącach zimowych
- Wzrost ruchu turystycznego w okresie wakacyjnym
- Zwiększona sprzedaż lodów w lecie

**Okres sezonowości:**
- Dane dzienne: tygodniowa sezonowość (7 dni)
- Dane miesięczne: roczna sezonowość (12 miesięcy)
- Dane kwartalne: roczna sezonowość (4 kwartały)

### 2.3 Cykl ($C_t$)

**Definicja:** Długookresowe, nieregularne wahania wokół trendu, które nie mają stałego okresu powtarzalności.

**Charakterystyka:**
- Okresy trwają zazwyczaj dłużej niż rok
- Długość cyklu jest zmienna i trudna do przewidzenia
- Często związane z cyklami ekonomicznymi lub naturalnymi

**Różnica między cyklem a sezonowością:**
- **Sezonowość**: stały okres (np. zawsze 12 miesięcy), przewidywalna
- **Cykl**: zmienny okres (np. 2-10 lat), mniej przewidywalny

**Przykłady:**
- Cykle koniunkturalne w gospodarce (boom → recesja → ożywienie)
- Cykle wyborcze (zmiana zachowań przed/po wyborach)
- Cykle życia produktów na rynku
- Cykle słoneczne (aktywność Słońca co ~11 lat)

### 2.4 Biały szum ($\varepsilon_t$)

**Definicja:** Składnik losowy szeregu czasowego, reprezentujący przypadkowe, nieprzewidywalne wahania bez żadnej struktury czasowej.

**Właściwości matematyczne:**
- Średnia wartość: $E(\varepsilon_t) = 0$
- Stała wariancja: $\text{Var}(\varepsilon_t) = \sigma^2$
- Brak autokorelacji: $\text{Cov}(\varepsilon_t, \varepsilon_s) = 0$ dla $t \neq s$
- Obserwacje są niezależne od siebie

**Charakterystyka:**
- Całkowicie losowy i nieprzewidywalny
- Niemożliwy do modelowania
- Reprezentuje szum, błędy pomiarowe, niewyjaśnione fluktuacje

**Przykłady:**
- Przypadkowe błędy pomiarowe w urządzeniach
- Mikroskopijne wahania cen akcji (niezwiązane z trendami)
- Losowe zakłócenia w systemach transmisji danych

**Znaczenie:** Idealne prognozy osiągamy, gdy reszty (błędy) modelu przypominają biały szum - oznacza to, że wychwyciliśmy wszystkie systematyczne wzorce.

---

### Dekompozycja szeregu czasowego

Szereg czasowy można przedstawić jako kombinację komponentów:

**Model addytywny:**
$$Y_t = T_t + S_t + C_t + \varepsilon_t$$

**Model multiplikatywny:**
$$Y_t = T_t \times S_t \times C_t \times \varepsilon_t$$

**Kiedy stosować:**
- **Addytywny**: gdy amplituda wahań sezonowych jest stała
- **Multiplikatywny**: gdy amplituda wahań sezonowych rośnie z trendem

---

## 3. Pandas - podstawy pracy z danymi

### 3.1 Wczytywanie danych

**Pandas** to biblioteka Python do analizy i manipulacji danymi, szczególnie przydatna do pracy z szeregami czasowymi.

#### Podstawowe metody wczytywania:

**1. Wczytywanie z pliku CSV:**
```python
import pandas as pd

# Podstawowe wczytanie
df = pd.read_csv('dane.csv')

# Z określeniem separatora
df = pd.read_csv('dane.csv', sep=';')

# Z określeniem kolumny z datami jako indeks
df = pd.read_csv('dane.csv', index_col='data', parse_dates=True)

# Z określeniem encoding
df = pd.read_csv('dane.csv', encoding='utf-8')
```

**2. Wczytywanie z pliku Excel:**
```python
# Z pliku Excel
df = pd.read_excel('dane.xlsx', sheet_name='Arkusz1')

# Z określonego arkusza
df = pd.read_excel('dane.xlsx', sheet_name=0)  # pierwszy arkusz
```

**[NIEWYMAGANE]** **3. Wczytywanie z innych źródeł:** **[NIEWYMAGANE]**
```python
# Z JSON
df = pd.read_json('dane.json')

# Z SQL
import sqlite3
conn = sqlite3.connect('baza.db')
df = pd.read_sql('SELECT * FROM tabela', conn)

# Z URL
df = pd.read_csv('https://example.com/dane.csv')
```

**4. Parametry użyteczne przy wczytywaniu:**
```python
df = pd.read_csv('dane.csv',
                 nrows=1000,           # wczytaj tylko pierwsze 1000 wierszy
                 skiprows=5,           # pomiń pierwsze 5 wierszy
                 usecols=['A', 'B'],   # wczytaj tylko kolumny A i B
                 na_values=['?', 'N/A']) # określ wartości traktowane jako brakujące
```

#### Pierwsze spojrzenie na dane:
```python
# Wyświetl pierwsze 5 wierszy
df.head()

# Wyświetl ostatnie 5 wierszy
df.tail()

# Informacje o DataFrame
df.info()

# Podstawowe statystyki
df.describe()

# Kształt danych (liczba wierszy, kolumn)
df.shape

# Nazwy kolumn
df.columns

# Typy danych
df.dtypes
```

---

### 3.2 Sprawdzanie i uzupełnianie danych brakujących

Dane brakujące (ang. *missing data*, *NaN*) to częsty problem w analizie danych. Pandas oferuje narzędzia do ich wykrywania i obsługi.

#### Wykrywanie danych brakujących:

```python
# Sprawdź, które wartości są brakujące (True/False)
df.isnull()
df.isna()  # to samo co isnull()

# Sprawdź, które wartości NIE są brakujące
df.notnull()
df.notna()

# Liczba brakujących wartości w każdej kolumnie
df.isnull().sum()

# Procent brakujących wartości w każdej kolumnie
(df.isnull().sum() / len(df)) * 100

# Całkowita liczba brakujących wartości
df.isnull().sum().sum()

# Wiersze zawierające jakiekolwiek braki
df[df.isnull().any(axis=1)]

# Wizualizacja brakujących danych
**[NIEWYMAGANE]**
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
```

#### Usuwanie danych brakujących:

```python
# Usuń wiersze z jakimikolwiek brakami
df_clean = df.dropna()

# Usuń wiersze, gdzie WSZYSTKIE wartości są brakujące
df_clean = df.dropna(how='all')

# Usuń wiersze z brakami w konkretnych kolumnach
df_clean = df.dropna(subset=['kolumna1', 'kolumna2'])

# Usuń kolumny z brakami
df_clean = df.dropna(axis=1)

# Usuń kolumny, gdzie więcej niż 50% wartości to braki
threshold = len(df) * 0.5
df_clean = df.dropna(thresh=threshold, axis=1)
```

#### Uzupełnianie danych brakujących:

**1. Wypełnienie stałą wartością:**
```python
# Wypełnij zerami
df_filled = df.fillna(0)

# Wypełnij konkretną wartością
df_filled = df.fillna('brak')

# Wypełnij różnymi wartościami dla różnych kolumn
df_filled = df.fillna({'kolumna1': 0, 'kolumna2': 'brak', 'kolumna3': -1})
```

**2. Wypełnienie wartością średnią, medianą, modą:**
```python
# Wypełnij średnią z kolumny
df['kolumna'] = df['kolumna'].fillna(df['kolumna'].mean())

# Wypełnij medianą
df['kolumna'] = df['kolumna'].fillna(df['kolumna'].median())

# Wypełnij modą (najczęstszą wartością)
df['kolumna'] = df['kolumna'].fillna(df['kolumna'].mode()[0])
```

**3. Forward Fill i Backward Fill (dla szeregów czasowych):**
```python
# Forward fill - przepisz ostatnią znaną wartość w przód
df_filled = df.fillna(method='ffill')
# lub
df_filled = df.ffill()

# Backward fill - przepisz następną znaną wartość w tył
df_filled = df.fillna(method='bfill')
# lub
df_filled = df.bfill()

# Forward fill z limitem (max 2 kolejne braki)
df_filled = df.fillna(method='ffill', limit=2)
```

#### Dobre praktyki:

1. **Zawsze analizuj przyczynę braków** - czy są losowe, czy systematyczne?
2. **Dokumentuj decyzje** - zapisz, jakie metody uzupełniania zostały użyte
3. **Weryfikuj wyniki** - sprawdź, czy uzupełnione wartości mają sens
4. **Rozważ różne metody** - dla różnych typów danych (numeryczne vs kategoryczne)
5. **Nie usuwaj pochopnie** - często lepiej uzupełnić niż stracić dane



---

## 4. Praca z datami i czasem w Python

### 4.1 Obiekt `datetime` w Python

Python posiada wbudowany moduł `datetime` do pracy z datami i czasem.

#### Podstawowe klasy modułu datetime:

```python
from datetime import datetime, date, time, timedelta

# 1. datetime - data i czas
dt = datetime(2024, 1, 15, 14, 30, 45)  # rok, miesiąc, dzień, godzina, minuta, sekunda
print(dt)  # 2024-01-15 14:30:45

# 2. date - tylko data
d = date(2024, 1, 15)  # rok, miesiąc, dzień
print(d)  # 2024-01-15

# 3. time - tylko czas
t = time(14, 30, 45)  # godzina, minuta, sekunda
print(t)  # 14:30:45

# 4. timedelta - różnica czasu
td = timedelta(days=7, hours=3, minutes=30)
print(td)  # 7 days, 3:30:00
```

#### Aktualna data i czas:

```python
from datetime import datetime

# Aktualna data i czas
now = datetime.now()
print(now)

# Tylko aktualna data
today = date.today()
print(today)

# Data i czas UTC
utc_now = datetime.utcnow()
print(utc_now)
```

#### Składniki obiektu datetime:

```python
dt = datetime(2024, 1, 15, 14, 30, 45)

# Dostęp do poszczególnych składników
print(dt.year)        # 2024
print(dt.month)       # 1
print(dt.day)         # 15
print(dt.hour)        # 14
print(dt.minute)      # 30
print(dt.second)      # 45
print(dt.weekday())   # 0 (poniedziałek=0, niedziela=6)
print(dt.isoweekday()) # 1 (poniedziałek=1, niedziela=7)
```

#### Operacje arytmetyczne na datach:

```python
from datetime import datetime, timedelta

# Dodawanie czasu
dt = datetime(2024, 1, 15, 14, 30)
nowa_data = dt + timedelta(days=7)  # dodaj 7 dni
print(nowa_data)  # 2024-01-22 14:30:00

# Odejmowanie czasu
wczesniejsza_data = dt - timedelta(hours=5)
print(wczesniejsza_data)  # 2024-01-15 09:30:00

# Różnica między datami
dt1 = datetime(2024, 1, 15)
dt2 = datetime(2024, 1, 22)
roznica = dt2 - dt1
print(roznica)  # 7 days, 0:00:00
print(roznica.days)  # 7
print(roznica.total_seconds())  # 604800.0
```

---

### 4.2 Konwersja stringów na daty w różnych formatach

#### Konwersja string → datetime (parsing):

**[NIEWYMAGANE]** **Metoda `strptime()` - String Parse Time** **[NIEWYMAGANE]**

```python
from datetime import datetime

# Format: YYYY-MM-DD
data_str = "2024-01-15"
dt = datetime.strptime(data_str, "%Y-%m-%d")
print(dt)  # 2024-01-15 00:00:00

# Format: DD/MM/YYYY
data_str = "15/01/2024"
dt = datetime.strptime(data_str, "%d/%m/%Y")
print(dt)  # 2024-01-15 00:00:00

# Format z czasem: YYYY-MM-DD HH:MM:SS
data_str = "2024-01-15 14:30:45"
dt = datetime.strptime(data_str, "%Y-%m-%d %H:%M:%S")
print(dt)  # 2024-01-15 14:30:45

# Format tekstowy: 15 January 2024
data_str = "15 January 2024"
dt = datetime.strptime(data_str, "%d %B %Y")
print(dt)  # 2024-01-15 00:00:00

# Format skrócony: 15/01/24
data_str = "15/01/24"
dt = datetime.strptime(data_str, "%d/%m/%y")
print(dt)  # 2024-01-15 00:00:00
```

#### Konwersja datetime → string (formatting):

**[NIEWYMAGANE]** **Metoda `strftime()` - String Format Time** **[NIEWYMAGANE]**

```python
from datetime import datetime

dt = datetime(2024, 1, 15, 14, 30, 45)

# Różne formaty wyjściowe
print(dt.strftime("%Y-%m-%d"))              # 2024-01-15
print(dt.strftime("%d/%m/%Y"))              # 15/01/2024
print(dt.strftime("%Y-%m-%d %H:%M:%S"))     # 2024-01-15 14:30:45
print(dt.strftime("%d %B %Y"))              # 15 January 2024
print(dt.strftime("%A, %d %B %Y"))          # Monday, 15 January 2024
print(dt.strftime("%d-%m-%Y %H:%M"))        # 15-01-2024 14:30
print(dt.strftime("%I:%M %p"))              # 02:30 PM
```

#### Najważniejsze kody formatowania:

| Kod | Znaczenie | Przykład |
|-----|-----------|----------|
| `%Y` | Rok (4 cyfry) | 2024 |
| `%y` | Rok (2 cyfry) | 24 |
| `%m` | Miesiąc (01-12) | 01 |
| `%B` | Pełna nazwa miesiąca | January |
| `%b` | Skrócona nazwa miesiąca | Jan |
| `%d` | Dzień miesiąca (01-31) | 15 |
| `%A` | Pełna nazwa dnia tygodnia | Monday |
| `%a` | Skrócona nazwa dnia tygodnia | Mon |
| `%H` | Godzina 24h (00-23) | 14 |
| `%I` | Godzina 12h (01-12) | 02 |
| `%M` | Minuta (00-59) | 30 |
| `%S` | Sekunda (00-59) | 45 |
| `%p` | AM/PM | PM |
| `%w` | Dzień tygodnia (0-6) | 1 |
| `%j` | Dzień roku (001-366) | 015 |

---

### 4.3 Praca z datami w Pandas

#### Konwersja stringów na daty w Pandas - `pd.to_datetime()`

**Podstawowa konwersja:**

```python
import pandas as pd

# 1. Automatyczne rozpoznawanie formatu (najłatwiejsze)
dates = ['2024-01-15', '2024-01-16', '2024-01-17']
df = pd.DataFrame({'data': dates})
df['data'] = pd.to_datetime(df['data'])
print(df['data'].dtype)  # datetime64[ns]

# 2. Określenie formatu (szybsze dla dużych zbiorów)
dates = ['15/01/2024', '16/01/2024', '17/01/2024']
df = pd.DataFrame({'data': dates})
df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')

# 3. Format z czasem
dates = ['2024-01-15 14:30:00', '2024-01-16 15:45:00']
df = pd.DataFrame({'data': dates})
df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d %H:%M:%S')
```

**Różne formaty popularnych dat:**

```python
import pandas as pd

# Format ISO 8601: YYYY-MM-DD
df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d')

# Format europejski: DD/MM/YYYY
df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')

# Format amerykański: MM/DD/YYYY
df['data'] = pd.to_datetime(df['data'], format='%m/%d/%Y')

# Format z kropkami: DD.MM.YYYY
df['data'] = pd.to_datetime(df['data'], format='%d.%m.%Y')

# Format długi: 15 January 2024
df['data'] = pd.to_datetime(df['data'], format='%d %B %Y')

# Format krótki: 15 Jan 2024
df['data'] = pd.to_datetime(df['data'], format='%d %b %Y')

# Timestamp Unix (liczba sekund od 1970-01-01)
df['data'] = pd.to_datetime(df['timestamp'], unit='s')
```

**Obsługa błędów - parametr `errors`:**

```python
import pandas as pd

df = pd.DataFrame({
    'data': ['2024-01-15', '2024-02-30', 'invalid', '2024-03-15']  # 30 lutego nie istnieje
})

# errors='raise' (domyślnie) - zgłoś błąd
try:
    df['data'] = pd.to_datetime(df['data'], errors='raise')
except:
    print("Błąd konwersji!")

# errors='coerce' - nieprawidłowe daty → NaT (Not a Time)
df['data'] = pd.to_datetime(df['data'], errors='coerce')
print(df)
# data
# 0   2024-01-15
# 1          NaT  (nieprawidłowa data)
# 2          NaT  (nieprawidłowy format)
# 3   2024-03-15

# errors='ignore' - pozostaw oryginalne wartości przy błędzie
df['data'] = pd.to_datetime(df['data'], errors='ignore')
```

**Konwersja z osobnych kolumn (rok, miesiąc, dzień):**

```python
import pandas as pd

df = pd.DataFrame({
    'rok': [2024, 2024, 2024],
    'miesiąc': [1, 2, 3],
    'dzień': [15, 20, 25],
    'wartość': [100, 200, 300]
})

# Połącz w jedną datę
df['data'] = pd.to_datetime(df[['rok', 'miesiąc', 'dzień']])

# Z godziną, minutą, sekundą
df = pd.DataFrame({
    'rok': [2024], 'miesiąc': [1], 'dzień': [15],
    'godzina': [14], 'minuta': [30], 'sekunda': [45]
})
df['data_czas'] = pd.to_datetime(df[['rok', 'miesiąc', 'dzień', 'godzina', 'minuta', 'sekunda']])
```

**Konwersja podczas wczytywania pliku:**

```python
import pandas as pd

# Automatyczna konwersja przy wczytywaniu CSV
df = pd.read_csv('dane.csv', parse_dates=['data'])

# Konwersja wielu kolumn
df = pd.read_csv('dane.csv', parse_dates=['data_start', 'data_koniec'])

# Konwersja i ustawienie jako indeks
df = pd.read_csv('dane.csv', parse_dates=['data'], index_col='data')

# Określenie formatu podczas wczytywania
df = pd.read_csv('dane.csv', parse_dates=['data'], date_format='%d/%m/%Y')
```

**[NIEWYMAGANE]** **Obsługa mieszanych formatów:** **[NIEWYMAGANE]**

```python
import pandas as pd

df = pd.DataFrame({
    'data': ['2024-01-15', '15/01/2024', '15.01.2024', '2024/01/15']
})

# Metoda 1: Funkcja custom
def parse_mixed_dates(date_str):
    formats = ['%Y-%m-%d', '%d/%m/%Y', '%d.%m.%Y', '%Y/%m/%d']
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue
    return pd.NaT

df['data'] = df['data'].apply(parse_mixed_dates)

# Metoda 2: Automatyczne rozpoznawanie (wolniejsze)
df['data'] = pd.to_datetime(df['data'], infer_datetime_format=True)

# Metoda 3: dayfirst=True dla formatu DD/MM/YYYY
df['data'] = pd.to_datetime(df['data'], dayfirst=True)
```

**[NIEWYMAGANE]** **Parametry `dayfirst` i `yearfirst`:** **[NIEWYMAGANE]**

```python
import pandas as pd

# Niejednoznaczna data: 01/02/2024
# Czy to 1 lutego czy 2 stycznia?

df = pd.DataFrame({'data': ['01/02/2024', '05/03/2024']})

# dayfirst=False (domyślnie) → interpretuje jako MM/DD/YYYY (amerykański)
df['data_us'] = pd.to_datetime(df['data'], dayfirst=False)  # 2 stycznia, 3 maja

# dayfirst=True → interpretuje jako DD/MM/YYYY (europejski)
df['data_eu'] = pd.to_datetime(df['data'], dayfirst=True)   # 1 lutego, 5 marca

print(df)
```

**[NIEWYMAGANE]** **Konwersja timestamp (Unix epoch):** **[NIEWYMAGANE]**

```python
import pandas as pd

# Timestamp w sekundach
df = pd.DataFrame({'timestamp': [1705334400, 1705420800, 1705507200]})
df['data'] = pd.to_datetime(df['timestamp'], unit='s')

# Timestamp w milisekundach
df['data'] = pd.to_datetime(df['timestamp'], unit='ms')

# Timestamp w mikrosekundach
df['data'] = pd.to_datetime(df['timestamp'], unit='us')

# Timestamp w nanosekundach
df['data'] = pd.to_datetime(df['timestamp'], unit='ns')
```

**Praktyczny przykład - czyszczenie i konwersja dat:**

```python
import pandas as pd
import numpy as np

# Dane z różnymi problemami
df = pd.DataFrame({
    'data': ['2024-01-15', '15/01/2024', '2024.01.16', 'invalid', np.nan, '2024-02-30'],
    'wartość': [100, 200, 300, 400, 500, 600]
})

print("Oryginalne dane:")
print(df)

# 1. Konwersja z obsługą błędów
df['data_clean'] = pd.to_datetime(df['data'], errors='coerce')

# 2. Sprawdź, ile dat nie udało się przekonwertować
invalid_dates = df['data_clean'].isna().sum()
print(f"\nLiczba nieprawidłowych dat: {invalid_dates}")

# 3. Wyświetl wiersze z nieprawidłowymi datami
print("\nWiersze z błędnymi datami:")
print(df[df['data_clean'].isna()])

# 4. Usuń wiersze z nieprawidłowymi datami (opcjonalnie)
df_clean = df.dropna(subset=['data_clean'])

# 5. Sortuj po dacie
df_clean = df_clean.sort_values('data_clean')

print("\nOczyszczone dane:")
print(df_clean)
```

#### Wyciąganie składników daty:

```python
import pandas as pd

df = pd.DataFrame({
    'data': pd.date_range('2024-01-15', periods=5, freq='D')
})

# Wyciągnij składniki
df['rok'] = df['data'].dt.year
df['miesiąc'] = df['data'].dt.month
df['dzień'] = df['data'].dt.day
df['dzień_tygodnia'] = df['data'].dt.day_name()
df['dzień_roku'] = df['data'].dt.dayofyear
df['tydzień'] = df['data'].dt.isocalendar().week
df['kwartał'] = df['data'].dt.quarter

print(df)
```

#### Ustawianie daty jako indeks:

```python
import pandas as pd

# Wczytanie z ustawieniem indeksu
df = pd.read_csv('dane.csv', parse_dates=['data'], index_col='data')

# Lub po wczytaniu
df = pd.read_csv('dane.csv')
df['data'] = pd.to_datetime(df['data'])
df = df.set_index('data')

# Sortowanie po dacie
df = df.sort_index()
```

#### Tworzenie zakresów dat:

```python
import pandas as pd

# Zakres dat dziennych
daty = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
print(len(daty))  # 31

# Zakres z określoną liczbą okresów
daty = pd.date_range(start='2024-01-01', periods=10, freq='D')

# Różne częstotliwości
daty_godzinowe = pd.date_range('2024-01-01', periods=24, freq='H')  # godzinowe
daty_miesięczne = pd.date_range('2024-01-01', periods=12, freq='M')  # miesięczne
daty_biznesowe = pd.date_range('2024-01-01', periods=10, freq='B')  # dni robocze
```

#### Przesunięcia czasowe:

```python
import pandas as pd

df = pd.DataFrame({
    'data': pd.date_range('2024-01-01', periods=5, freq='D'),
    'wartość': [10, 20, 30, 40, 50]
})
df = df.set_index('data')

# Przesunięcie o N okresów
df['wartość_wczoraj'] = df['wartość'].shift(1)  # przesunięcie w przód (wcześniejsze dane)
df['wartość_jutro'] = df['wartość'].shift(-1)   # przesunięcie w tył (późniejsze dane)

# Różnica między okresami
df['zmiana'] = df['wartość'].diff()  # różnica względem poprzedniego okresu

print(df)
```
---

*Konspekt będzie rozwijany o kolejne tematy...*
