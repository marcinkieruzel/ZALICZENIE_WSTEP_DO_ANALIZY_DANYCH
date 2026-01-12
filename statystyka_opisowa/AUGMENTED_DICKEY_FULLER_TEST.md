# Test Dickey-Fullera

## Kim byli Dickey i Fuller?

**David A. Dickey** i **Wayne A. Fuller** to amerykańscy statystycy i ekonometrycy, którzy w 1979 roku opublikowali przełomową pracę wprowadzającą test na pierwiastek jednostkowy (unit root test). Dickey był doktorantem, a Fuller jego promotorem na Iowa State University. Ich współpraca zaowocowała jednym z najważniejszych testów w ekonometrii czasowej.

## Czym jest test Dickey-Fullera?

Test Dickey-Fullera (DF) to **test statystyczny służący do sprawdzenia stacjonarności szeregu czasowego**. Konkretnie testuje on obecność pierwiastka jednostkowego w szeregu czasowym, co jest równoważne z testowaniem niestacjonarności.

### Pierwiastek jednostkowy

>Pierwiastek jednostkowy (ang. unit root) – właściwość niektórych procesów stochastycznych (na przykład procesów błądzenia losowego), która może utrudniać wnioskowanie statystyczne przy modelowaniu szeregów czasowych. Liniowy proces stochastyczny ma pierwiastek jednostkowy, gdy pierwiastkiem równania charakterystycznego procesu jest 1. Taki proces jest niestacjonarny, choć niekoniecznie charakteryzuje się trendem.
>Obecność pierwiastka jednostkowego sprawdza się z wykorzystaniem statystycznych testów pierwiastka jednostkowego, np. testu Dickeya-Fullera.

[https://pl.wikipedia.org/wiki/Pierwiastek_jednostkowy](https://pl.wikipedia.org/wiki/Pierwiastek_jednostkowy)

A teraz praktycznie:

Pierwiastek jednostkowy (unit root) oznacza, że szereg czasowy jest niestacjonarny, a jego zachowanie ma „pamięć na zawsze”.


Jeśli szereg ma pierwiastek jednostkowy, to:

- obecna wartość silnie zależy od poprzedniej,
- szoki (losowe zmiany) mają trwały wpływ,
- szereg nie wraca do stałej średniej.

To, co wydarzyło się kiedyś, wpływa na przyszłość na stałe.

Najprostszy przykład matematyczny

Tzw. random walk:

$$
yt​=yt−1​+εt​
$$

Dla AR(p)

$$
yt​=ϕyt−1​+εt​
$$
szereg stacjonarny
$$
∣ϕ∣<1
$$
szereg niestacjonarny (pierwiastek jednostkowy)
$$
ϕ=1
$$

proces niestabilny
$$
∣ϕ∣>1
$$

gdzie: ϕ – współczynnik autoregresyjny

### Wersje testu:
1. **Prosty test Dickey-Fullera (DF)** - podstawowa wersja
2. **Rozszerzony test Dickey-Fullera (ADF)** - uwzględnia autokorelację składnika losowego

### Model matematyczny:
Test opiera się na regresji:

$$
\begin{aligned}
\Delta y_t &= \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \varepsilon_t
\end{aligned}
$$

Gdzie:
- $\Delta y_t = y_t - y_{t-1}$ (pierwsza różnica)
- $\gamma = \rho - 1$ (gdzie $\rho$ to współczynnik autoregresji)

## Jak interpretować wyniki?

### Hipotezy:
- **H₀**: $\gamma = 0$ (szereg ma pierwiastek jednostkowy, jest niestacjonarny)
- **H₁**: $\gamma < 0$ (szereg jest stacjonarny)

### Interpretacja:
1. **p-value < α (np. 0.05)**: 
   - Odrzucamy H₀
   - Szereg jest **stacjonarny**
   - Nie ma pierwiastka jednostkowego

2. **p-value > α**:
   - Nie odrzucamy H₀  
   - Szereg jest **niestacjonarny**
   - Ma pierwiastek jednostkowy

### Statystyka testowa:
Używa się specjalnych wartości krytycznych (tablice Dickey-Fullera), ponieważ standardowe rozkłady t nie mają zastosowania w przypadku pierwiastków jednostkowych.

### Praktyczne znaczenie:
- **Szereg stacjonarny**: można bezpiecznie stosować klasyczne metody ekonometryczne
- **Szereg niestacjonarny**: wymaga różnicowania lub innych technik (kointegracja, modele VAR z korekcją błędu)

Test Dickey-Fullera jest fundamentalnym narzędziem w analizie szeregów czasowych, szczególnie ważnym przed przeprowadzeniem dalszych analiz ekonometrycznych.
