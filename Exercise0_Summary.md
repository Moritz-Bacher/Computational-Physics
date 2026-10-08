# Computational Physics – Exercise 0: First Steps with Agentic AI

**Datum:** WS 2026/27  
**Autoren:** Florian Bruckner, Claas Abert  
**Bearbeitung:** Aufgabe 2 (Monte-Carlo-Berechnung von $\pi$)  

---

## 1. Einleitung & Aufgabenstellung

Gegeben ist das Einheitsquadrat $[0, 1]^2$, in dem ein Viertelkreis mit Radius $r = 1$ liegt. Ein Punkt $(x, y) \in [0, 1]^2$ liegt innerhalb des Viertelkreises, wenn:
$$x^2 + y^2 \le 1$$
Punkte auf dem Rand ($x^2 + y^2 = 1$) zählen per Aufgabenstellung als Punkte, die **innerhalb** liegen.

Das Verhältnis der Fläche des Viertelkreises ($A_{\text{Kreis}} = \frac{\pi}{4}$) zur Fläche des Quadrats ($A_{\text{Quadrat}} = 1$) beträgt $\frac{\pi}{4}$. Zieht man $N$ gleichverteilte Zufallspunkte und zählt, wie viele davon im Viertelkreis liegen ($N_{\text{inside}}$), gilt:
$$\frac{N_{\text{inside}}}{N} \approx \frac{\pi}{4} \implies \hat{\pi} = 4 \cdot \frac{N_{\text{inside}}}{N}$$

---

## 2. Ergebnisse der Teilaufgaben

### Aufgabe 2a: Skript zur Schätzung von $\pi$ für $N = 10^2 \dots 10^7$

Das Skript `estimate_pi.py` zieht $N$ Zufallspunkte für $N \in \{10^2, 10^3, 10^4, 10^5, 10^6, 10^7\}$ und berechnet den Schätzwert $\hat{\pi}$ sowie den absoluten und relativen Fehler zu $\pi$.

#### Numerische Ergebnisse (Einzellauf):

| $N$ | $\hat{\pi}$ | Absoluter Fehler $|\hat{\pi} - \pi|$ | Relativer Fehler |
| :---: | :---: | :---: | :---: |
| $10^2$ | $3{,}3600000$ | $2{,}184 \times 10^{-1}$ | $6{,}95\,\%$ |
| $10^3$ | $3{,}1480000$ | $6{,}407 \times 10^{-3}$ | $0{,}20\,\%$ |
| $10^4$ | $3{,}1488000$ | $7{,}207 \times 10^{-3}$ | $0{,}23\,\%$ |
| $10^5$ | $3{,}1377600$ | $3{,}833 \times 10^{-3}$ | $0{,}12\,\%$ |
| $10^6$ | $3{,}1424360$ | $8{,}433 \times 10^{-4}$ | $0{,}027\,\%$ |
| $10^7$ | $3{,}1413040$ | $2{,}887 \times 10^{-4}$ | $0{,}0092\,\%$ |

#### Frage: *Welche Schritte hat der Agent unaufgefordert ausgeführt?*
1. **Ensemble-Mittelwert über 20 Trials:** Da ein Einzellauf zufallsbedingt stark fluktuiert, hat der Agent zusätzlich 20 unabhängige Wiederholungen pro $N$ berechnet, um den statistischen Erwartungswert des Fehlers sauber zu erfassen.
2. **Theoretische Referenzkurve:** In den Plot wurde die theoretische Standardabweichung $\sigma_N = \sqrt{\frac{\pi(4-\pi)}{N}}$ als $N^{-1/2}$-Referenzgerade eingezeichnet.
3. **Formatierte Ausgabe:** Die Ergebnisse wurden automatisch tabellarisch und formatiert in der Konsole ausgegeben.
4. **Umgebungserkennung:** Der Agent hat die installierten Python-Interpreter auf dem System gesucht und den funktionierenden Anaconda-Pfad identifiziert.

---

### Aufgabe 2b: Skalierung des Fehlers & Verifikation

#### Wie skaliert der Fehler mit $N$ und warum?
Der Fehler skaliert als:
$$\text{Fehler} \propto \frac{1}{\sqrt{N}} = O(N^{-1/2})$$

**Begründung:**
Jeder Zufallspunkt ist ein Bernoulli-Experiment mit Trefferwahrscheinlichkeit $p = \frac{\pi}{4}$. Die Zufallsvariable $N_{\text{inside}}$ folgt einer Binomialverteilung $B(N, p)$ mit Erwartungswert $E[N_{\text{inside}}] = N p$ und Varianz:
$$\operatorname{Var}(N_{\text{inside}}) = N p (1 - p)$$
Für den Schätzer $\hat{\pi} = \frac{4}{N} N_{\text{inside}}$ gilt somit:
$$\operatorname{Var}(\hat{\pi}) = \frac{16}{N^2} \operatorname{Var}(N_{\text{inside}}) = \frac{16 p (1-p)}{N} = \frac{\pi(4-\pi)}{N}$$
Die Standardabweichung des Schätzers ist daher:
$$\sigma_{\hat{\pi}} = \sqrt{\frac{\pi(4-\pi)}{N}} \approx \frac{1{,}642}{\sqrt{N}}$$
Nach dem **Zentralen Grenzwertsatz (CLT)** konvergiert der Fehler asymptotisch mit Rate $N^{-1/2}$. Um eine Dezimalstelle Genauigkeit zu gewinnen, muss $N$ um den Faktor $100$ vergrößert werden.

#### Vertraust du dem Plot? Wie hast du ihn überprüft?
- **Einzellauf vs. Ensemble-Mittelwert:** In einem Einzellauf kann der Fehler rein zufällig bei einem $N$ kleiner ausfallen als bei einem größeren $N$ (z. B. war der Fehler bei $N=10^3$ zufällig kleiner als bei $N=10^4$). Dies ist kein Fehler der Methode, sondern typisches Monte-Carlo-Rauschen.
- **Überprüfung:** Durch die Mittelung über 20 Läufe (`Mean error`) sieht man, dass der gemittelte Fehler exakt parallel zur theoretischen Linie $\propto N^{-1/2}$ im doppelt-logarithmischen Plot verläuft (Steigung $-0{,}5$).

---

### Aufgabe 2c: Vektorisierung vs. Python-Schleife (Laufzeitvergleich)

Die Implementierung wurde um zwei Varianten erweitert:
1. `estimate_pi_vectorized`: NumPy-Arrays (`rng.uniform`, vektorisierter Vergleich `x**2 + y**2 <= 1.0`).
2. `estimate_pi_python_loop`: Reine Python-`for`-Schleife mit dem `random`-Modul.

#### Gemessene Laufzeiten:

| $N$ | Vektorisiert (NumPy) [s] | Python-Schleife [s] | Speedup |
| :---: | :---: | :---: | :---: |
| $10^2$ | $0{,}000335$ | $0{,}000043$ | $0{,}1 \times$ (Overhead) |
| $10^3$ | $0{,}000103$ | $0{,}000166$ | $1{,}6 \times$ |
| $10^4$ | $0{,}000297$ | $0{,}001782$ | $6{,}0 \times$ |
| $10^5$ | $0{,}002452$ | $0{,}016922$ | $6{,}9 \times$ |
| $10^6$ | $0{,}023390$ | $0{,}169300$ | $7{,}2 \times$ |
| $10^7$ | $0{,}230613$ | $1{,}735031$ | **$7{,}5 \times$** |

#### Welche Version ist schneller und warum?
- **NumPy ist ab $N \ge 10^3$ deutlich schneller** und erreicht ab $N = 10^5$ einen konstanten Beschleunigungsfaktor von ca. **$7{,}5 \times$**.
- **Grund:** NumPy führt die Schleifen in optimiertem C-Code direkt im zusammenhängenden Speicherblock aus und nutzt CPU-Vektorregister (SIMD). Die Python-Schleife hat pro Iteration den Overhead des Bytecode-Interpreters, der dynamischen Typprüfung und Objekterzeugung.
- Bei sehr kleinem $N = 100$ ist die Python-Schleife schneller, da der initiale Aufruf-Overhead von NumPy dominiert.

#### Hat der Agent mehr geändert als gefragt?
- Der Agent hat nicht nur die Schleifenversion hinzugefügt, sondern auch eine automatisierte Benchmarking-Funktion `compare_timings()` sowie einen separaten Plot `timing_comparison.png` erstellt.
- Das Überprüfen mit `git diff` ermöglicht es, diese zusätzlichen Blöcke vor dem Akzeptieren gezielt zu kontrollieren.

---

### Aufgabe 2d: Schnellere Konvergenz als $1/\sqrt{N}$

#### Was schlägt der Agent vor?
Standard-Monte-Carlo leidet unter zufälliger Klumpenbildung und Lücken. Um schneller als $O(N^{-1/2})$ zu konvergieren, gibt es zwei fundamentale Ansätze:
1. **Stratified Sampling (Schichtstichprobenverfahren):**
   Das Einheitsquadrat wird in ein $M \times M$ Gitter ($N = M^2$ Zellen) unterteilt. In jeder Zelle wird genau ein Punkt zufällig gewählt. Da vollkommen innere Zellen immer $1$ und vollkommen äußere immer $0$ liefern, entsteht Varianz **nur entlang des Kreisbogens** (Länge $\pi/2$). Die Anzahl der Grenzzellen ist $O(M) = O(\sqrt{N})$. Die Gesamtvarianz sinkt auf $O(N^{-3/2})$, was einer Fehlerskalierung von **$O(N^{-3/4}) = O(N^{-0{,}75})$** entspricht.
2. **Quasi-Monte-Carlo (QMC, z. B. Sobol- oder Halton-Folgen):**
   Verwendung deterministischer Niedrigdiskrepanz-Punktmengen (Low-Discrepancy Sequences), die den Raum gleichmäßiger abdecken als Pseudo-Zufall. Für glatte Randkurven in 2D erreicht QMC typischerweise Raten von **$O(N^{-0{,}75})$ bis $O(N^{-1})$**.

#### Ist die Behauptung wahr? Numerische Verifikation:
Im Skript `faster_convergence.py` wurden alle Verfahren numerisch verglichen und die Konvergenzsteigung $\alpha$ ($\text{Fehler} \propto N^{-\alpha}$) durch lineare Regression im Log-Log-Raum bestimmt:

| Verfahren | Gefittete Rate $\alpha$ | Theoretische Rate | Absoluter Fehler bei $N \approx 10^6$ |
| :--- | :---: | :---: | :---: |
| **Standard-MC (PRNG)** | **$0{,}52$** | $0{,}50$ ($1/\sqrt{N}$) | $1{,}81 \times 10^{-3}$ |
| **Stratified Sampling** | **$0{,}77$** | $0{,}75$ ($N^{-3/4}$) | $3{,}98 \times 10^{-5}$ ($\approx 45\times$ genauer!) |
| **QMC (Sobol)** | **$0{,}72$** | $\approx 0{,}75 \dots 1{,}0$ | $9{,}28 \times 10^{-5}$ |
| **QMC (Halton)** | **$0{,}95$** | $\approx 0{,}75 \dots 1{,}0$ | $2{,}54 \times 10^{-6}$ ($\approx 700\times$ genauer!) |

**Ergebnis:** Die Behauptung ist **vollständig verifiziert**. Sowohl Schichtung als auch Niedrigdiskrepanz-Folgen brechen die $1/\sqrt{N}$-Schranke des reinen Pseudo-Zufalls deutlich.

---

## 3. Reflexion zu Agentic AI (Aufgabe 3)

1. **Zeitersparnis vs. Verifikationsbedarf:**
   - *Zeitersparnis:* Schnelles Aufsetzen von Boilerplate-Code, Vektorisierung, Erstellen von Matplotlib-Plots und automatische Formatierung.
   - *Verifikation:* Man muss physikalische/mathematische Randbedingungen genau prüfen (z. B. Inklusion des Randes $\le$ vs. $<$, Wahl der korrekten theoretischen Rate für Indikatorfunktionen bei QMC, Überprüfung des Plots gegen Einzellauf-Rauschen).
2. **Präzisierung im Prompt (oder Projektdatei `AGENTS.md`):**
   - Konkrete Funktionssignaturen, geforderte Ausgaben, Programmierstile (z. B. "vectorized NumPy", "include boundary points $\le$") explizit spezifizieren.
   - Verbieten unaufgeforderter Refactorings von bestehendem Code, falls Stabilität gewünscht ist.
3. **Erklärbarkeit jeder Codezeile:**
   - Den Code modulare aufbauen lassen, verständliche Kommentare und Typannotationen einfordern, und vor jedem Commit mit `git diff` jede einzelne Änderung nachvollziehen.

---

## 4. Generierte Dateien

- [`estimate_pi.py`](estimate_pi.py): Monte-Carlo-Berechnung, Vektorisierung, Timing-Vergleich.
- [`faster_convergence.py`](faster_convergence.py): Implementierung & numerische Verifikation von Stratified Sampling und QMC.
- [`mc_pi_error.png`](mc_pi_error.png): Log-Log-Plot des Fehlers vs. $N$ mit $1/\sqrt{N}$-Theorielinie.
- [`timing_comparison.png`](timing_comparison.png): Laufzeitvergleich Vektorisierung vs. Python-Schleife.
- [`faster_convergence.png`](faster_convergence.png): Konvergenzvergleich Standard-MC vs. Stratified vs. QMC.
- [`sampling_patterns.png`](sampling_patterns.png): Visueller 2x2-Vergleich der Punktverteilungen (PRNG, Stratified, Sobol, Halton).

