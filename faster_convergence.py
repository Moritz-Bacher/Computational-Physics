"""
Aufgabe 2d: Schnellere Konvergenz als 1/sqrt(N) zur Berechnung von Pi.

Dieses Modul vergleicht vier verschiedene Verfahren zur Monte-Carlo-Integration
eines Viertelkreises im Einheitsquadrat [0, 1]^2:
1. Standard-Monte-Carlo (PRNG, Pseudo-Zufall)          -> Konvergenz: O(N^-0.50)
2. Stratified Sampling (Schichtstichprobe im Gitter)   -> Konvergenz: O(N^-0.75)
3. Quasi-Monte-Carlo mit Sobol-Folge (Low-Discrepancy) -> Konvergenz: O(N^-0.75 ... N^-1.0)
4. Quasi-Monte-Carlo mit Halton-Folge (Low-Discrepancy)-> Konvergenz: O(N^-0.75 ... N^-1.0)

Alle erzeugten Plots werden sowohl interaktiv angezeigt (plt.show()) als auch
als eigenständige .png-Dateien im Arbeitsverzeichnis gespeichert:
- faster_convergence.png: Vergleich der Konvergenzraten (Fehler vs. N auf Log-Log-Skala)
- sampling_patterns.png:  Visueller Vergleich der Punktverteilungen im Einheitsquadrat
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc


def mc_standard(N: int, seed: int = 42) -> float:
    """
    Standard-Monte-Carlo-Verfahren unter Verwendung von Pseudo-Zufallszahlen (PRNG).
    
    FUNKTIONSWEISE:
    1. Erzeugt N unabhängige, identisch gleichverteilte (i.i.d.) Zufallspunkte
       (x_i, y_i) im Einheitsquadrat [0, 1] x [0, 1] mittels NumPy default_rng.
    2. Prüft für jeden Punkt die Viertelkreis-Bedingung: x^2 + y^2 <= 1.0.
       Punkte direkt auf dem Rand (x^2 + y^2 = 1.0) zählen gemäß Vorgabe
       als Punkte, die innerhalb liegen.
    3. Zählt die Anzahl der Treffer (inside) und schätzt Pi über das
       Flächenverhältnis:
           A_Viertelkreis / A_Quadrat = (pi / 4) / 1 = pi / 4
           => pi_est = 4.0 * (Anzahl_Treffer / N).
           
    MATHEMATISCHER HINTERGRUND:
    Da die Punkte rein stochastisch unabhängig gezogen werden, bildet jeder Punkt
    ein Bernoulli-Experiment mit Erfolgswahrscheinlichkeit p = pi / 4.
    Nach dem Zentralen Grenzwertsatz (CLT) skaliert die Standardabweichung des Schätzers
    mit sigma = sqrt(pi * (4 - pi) / N) proportional zu 1 / sqrt(N) = O(N^-0.5).
    Reiner Zufall führt zu lokaler Klumpenbildung (Clustering) und Leerstellen,
    was die Konvergenz auf O(N^-0.5) limitiert.
    
    Parameter:
    ----------
    N : int
        Anzahl der zu generierenden Zufallspunkte.
    seed : int, optional
        Seed für den Pseudo-Zufallszahlengenerator zur Reproduzierbarkeit (Standard: 42).
        
    Rückgabe:
    ---------
    float
        Geschätzter Wert für Pi.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=N)
    y = rng.uniform(0.0, 1.0, size=N)
    
    inside = (x**2 + y**2) <= 1.0
    num_inside = np.sum(inside)
    
    return 4.0 * num_inside / N


def mc_stratified(N_target: int, seed: int = 42) -> tuple[int, float]:
    """
    Stratified Sampling (Schichtstichprobenverfahren / Jittered Grid Sampling).
    
    FUNKTIONSWEISE:
    1. Unterteilt das Einheitsquadrat [0, 1]^2 in ein reguläres Gitter aus
       M x M identischen quadratischen Teilzellen (Strata), wobei M = round(sqrt(N_target))
       ist. Die tatsächliche Gesamtanzahl der Punkte ist N = M^2.
    2. In jeder einzelnen Zelle [i/M, (i+1)/M] x [j/M, (j+1)/M] wird genau EIN
       Zufallspunkt platziert, indem zu den Gitterkoordinaten ein gleichverteilter
       Zufalls-Offset u in [0, 1) / M addiert wird ("Jittering").
    3. Prüft für jeden Punkt die Bedingung x^2 + y^2 <= 1.0.
    4. Berechnet den Schätzwert: pi_est = 4.0 * sum(inside) / N.
    
    MATHEMATISCHER HINTERGRUND:
    Zellen, die vollständig innerhalb des Viertelkreises liegen, liefern immer 1 (Varianz = 0).
    Zellen, die vollständig außerhalb liegen, liefern immer 0 (Varianz = 0).
    Eine stochastische Varianz entsteht AUSSCHLIESSLICH in den Zellen, die von der
    Randkurve x^2 + y^2 = 1 geschnitten werden.
    Die Bogenlänge des Viertelkreises ist pi / 2. Die Anzahl der vom Rand geschnittenen
    Zellen in einem M x M Gitter beträgt O(M) = O(sqrt(N)).
    Die Gesamtvarianz der Summe ist daher:
        Var(pi_est) = O(sqrt(N) * (1 / N^2)) = O(N^-1.5).
    Die Standardabweichung (der Fehler) skaliert folglich als:
        sigma = sqrt(Var) = O(N^-0.75).
    Dies ist asymptotisch deutlich schneller als Standard-Monte-Carlo mit O(N^-0.50)!
    
    Parameter:
    ----------
    N_target : int
        Angestrebte Punktezahl. Wird auf die nächste Quadratzahl N = M^2 gerundet.
    seed : int, optional
        Seed für die Zufallsgenerierung (Standard: 42).
        
    Rückgabe:
    ---------
    tuple[int, float]
        (Tatsächliche Punktanzahl N, Geschätzter Wert für Pi).
    """
    rng = np.random.default_rng(seed)
    M = int(np.round(np.sqrt(N_target)))
    N = M * M
    
    # Gitterindizes i, j in {0, ..., M - 1}
    i = np.arange(M)
    j = np.arange(M)
    ii, jj = np.meshgrid(i, j)
    
    # Gleichverteilter Jitter-Versatz innerhalb jeder Teilzelle
    u_x = rng.uniform(0.0, 1.0, size=(M, M))
    u_y = rng.uniform(0.0, 1.0, size=(M, M))
    
    # Koordinaten der Punkte im Intervall [0, 1]
    x = (ii + u_x) / M
    y = (jj + u_y) / M
    
    inside = (x**2 + y**2) <= 1.0
    pi_est = 4.0 * np.sum(inside) / N
    
    return N, pi_est


def mc_sobol(m: int, scramble: bool = False) -> tuple[int, float]:
    """
    Quasi-Monte-Carlo-Integration (QMC) mit der Sobol-Folge (Low-Discrepancy Sequence).
    
    FUNKTIONSWEISE:
    1. Verwendet scipy.stats.qmc.Sobol zur Generierung von N = 2^m Punkten im [0, 1]^2.
       Sobol-Folgen basieren auf binären Richtungsvektoren und radikaler Inversion zur Basis 2.
       Sie sind deterministisch so konstruiert, dass elementare binäre Intervalle
       (dyadische Boxen) optimal gleichmäßig mit Punkten belegt werden.
    2. Prüft für jeden Punkt x^2 + y^2 <= 1.0 (Rand inklusive).
    3. Berechnet pi_est = 4.0 * sum(inside) / N.
    
    MATHEMATISCHER HINTERGRUND:
    Standard-Monte-Carlo leidet unter Diskrepanz (ungleichmäßige Raumabdeckung).
    Niedrigdiskrepanz-Folgen minimieren die Stern-Diskrepanz D_N^*.
    Nach der Koksma-Hlawka-Ungleichung beträgt der Integrationsfehler für glatte Funktionen
    O((log N)^d / N) ~ O(N^-1).
    Für Indikatorfunktionen mit gekrümmtem Rand (wie unserem Kreisbogen) ist die
    Hardy-Krause-Variation zwar unbeschränkt, die Diskrepanzschranken für konvexe
    Gebiete in 2D garantieren jedoch eine Konvergenzrate von O(N^-0.75) bis O(N^-1.0).
    
    Parameter:
    ----------
    m : int
        Zweierpotenz-Exponent. Die Anzahl der Punkte ist exakt N = 2^m.
        (Sobol-Folgen entfalten ihre optimalen Eigenschaften für N = 2^m).
    scramble : bool, optional
        Falls True, wird Owen-Scrambling angewendet (randomisiertes QMC).
        Standard: False (rein deterministische Sobol-Folge).
        
    Rückgabe:
    ---------
    tuple[int, float]
        (Punktanzahl N = 2^m, Geschätzter Wert für Pi).
    """
    N = 2**m
    sampler = qmc.Sobol(d=2, scramble=scramble, seed=42 if scramble else None)
    pts = sampler.random(N)
    
    x = pts[:, 0]
    y = pts[:, 1]
    
    inside = (x**2 + y**2) <= 1.0
    pi_est = 4.0 * np.sum(inside) / N
    
    return N, pi_est


def mc_halton(N: int) -> float:
    """
    Quasi-Monte-Carlo-Integration (QMC) mit der Halton-Folge (Low-Discrepancy Sequence).
    
    FUNKTIONSWEISE:
    1. Verwendet scipy.stats.qmc.Halton zur Erzeugung von N Punkten im [0, 1]^2.
       Die Halton-Folge verallgemeinert die 1D-Van-der-Corput-Folge auf d Dimensionen,
       indem für jede Dimension eine unterschiedliche Primzahl als Basis gewählt wird.
       Für 2 Dimensionen sind das Basis 2 (für x) und Basis 3 (für y).
    2. Jeder Punkt wird durch radikale Inversion der Primzahlbasis gebildet.
       Dadurch wird der Raum deterministisch ohne Lücken und Klumpen ausgefüllt.
    3. Prüft x^2 + y^2 <= 1.0 (Rand inklusive).
    4. Berechnet pi_est = 4.0 * sum(inside) / N.
    
    MATHEMATISCHER HINTERGRUND:
    Ähnlich wie bei Sobol erreicht die Halton-Folge eine Diskrepanz von O((log N)^2 / N).
    Beim Kreisbogen im Einheitsquadrat ergibt sich empirisch eine Konvergenzrate
    von ca. O(N^-0.8) bis O(N^-1.0), was signifikant schneller ist als 1/sqrt(N).
    
    Parameter:
    ----------
    N : int
        Anzahl der Punkte (muss keine Zweierpotenz sein).
        
    Rückgabe:
    ---------
    float
        Geschätzter Wert für Pi.
    """
    sampler = qmc.Halton(d=2, scramble=False)
    pts = sampler.random(N)
    
    x = pts[:, 0]
    y = pts[:, 1]
    
    inside = (x**2 + y**2) <= 1.0
    return 4.0 * np.sum(inside) / N


def fit_convergence_rate(N_values: list[int], errors: list[float]) -> float:
    """
    Ermittelt die empirische Konvergenzrate alpha aus einem Datensatz.
    
    FUNKTIONSWEISE:
    Nimmt an, dass der Fehler dem Potenzgesetz folgt:
        Fehler(N) = C * N^(-alpha)
    Durch Logarithmieren beider Seiten erhält man eine lineare Gleichung:
        ln(Fehler) = -alpha * ln(N) + ln(C)
    Mittels np.polyfit(ln(N), ln(Fehler), 1) wird die Steigung der Ausgleichsgeraden
    berechnet. Die Konvergenzrate alpha entspricht dem negativen Wert dieser Steigung.
    
    Parameter:
    ----------
    N_values : list[int]
        Liste der ausgewerteten Punktanzahlen N.
    errors : list[float]
        Liste der zugehörigen absoluten Fehler |pi_est - pi|.
        
    Rückgabe:
    ---------
    float
        Empirischer Exponent alpha (z. B. 0.50 für 1/sqrt(N), 0.75 für 1/N^(3/4)).
    """
    n_arr = np.array(N_values, dtype=float)
    err_arr = np.array(errors, dtype=float)
    
    # Filtere eventuelle Nullen heraus, um log(0) zu vermeiden
    valid = err_arr > 0.0
    log_n = np.log(n_arr[valid])
    log_err = np.log(err_arr[valid])
    
    slope, _ = np.polyfit(log_n, log_err, 1)
    return -slope


def visualize_sampling_patterns(N_sample: int = 576):
    """
    Erstellt einen visuellen 2x2-Vergleich der vier Punktverteilungsmuster im Einheitsquadrat.
    
    FUNKTIONSWEISE:
    1. Erzeugt N_sample Punkte für:
       (a) Standard Pseudo-Random (PRNG)
       (b) Stratified Grid Sampling (mit eingezeichneten Gitterlinien)
       (c) Sobol-Folge
       (d) Halton-Folge
    2. Zeichnet in jedem Subplot den Viertelkreisbogen x^2 + y^2 = 1 ein.
    3. Färbt Punkte innerhalb blau und Punkte außerhalb rot ein.
    4. Speichert den Plot als 'sampling_patterns.png' und zeigt ihn mit plt.show() an.
    
    DIDAKTISCHER ZWECK:
    Visualisiert unmittelbar, WARUM Stratified Sampling und QMC schneller konvergieren:
    PRNG zeigt sichtbare Leerstellen und Klumpen.
    Stratified und QMC verteilen die Punkte extrem homogen im Raum.
    
    Parameter:
    ----------
    N_sample : int, optional
        Anzahl der Punkte für die Visualisierung (Standard: 576 = 24^2).
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=130)
    fig.suptitle(f"Vergleich der Sampling-Muster im Einheitsquadrat (N = {N_sample})", fontsize=15, fontweight="bold")
    
    # Kreisbogen zur Visualisierung
    theta = np.linspace(0, np.pi / 2, 200)
    arc_x = np.cos(theta)
    arc_y = np.sin(theta)
    
    # 1. Standard Pseudo-Random
    rng = np.random.default_rng(42)
    x_prng = rng.uniform(0, 1, N_sample)
    y_prng = rng.uniform(0, 1, N_sample)
    
    # 2. Stratified Sampling
    M = int(np.round(np.sqrt(N_sample)))
    i = np.arange(M)
    j = np.arange(M)
    ii, jj = np.meshgrid(i, j)
    x_strat = (ii + rng.uniform(0, 1, (M, M))) / M
    y_strat = (jj + rng.uniform(0, 1, (M, M))) / M
    x_strat = x_strat.flatten()
    y_strat = y_strat.flatten()
    
    # 3. Sobol Sequence (auf nächste Zweierpotenz 512 runden für korrekte Sobol-Eigenschaft)
    m_sobol = int(np.round(np.log2(N_sample)))
    pts_sobol = qmc.Sobol(d=2, scramble=False).random(2**m_sobol)
    x_sobol, y_sobol = pts_sobol[:, 0], pts_sobol[:, 1]
    
    # 4. Halton Sequence
    pts_halton = qmc.Halton(d=2, scramble=False).random(N_sample)
    x_halton, y_halton = pts_halton[:, 0], pts_halton[:, 1]
    
    methods = [
        ("Standard Pseudo-Random (PRNG)\n(Zufällige Klumpen & Leerstellen)", x_prng, y_prng, False),
        ("Stratified Grid Sampling\n(1 Jitter-Punkt pro Zelle)", x_strat, y_strat, True),
        (f"Quasi-Monte-Carlo: Sobol\n(Niedrigdiskrepanz, N = {2**m_sobol})", x_sobol, y_sobol, False),
        ("Quasi-Monte-Carlo: Halton\n(Niedrigdiskrepanz, Basen 2 und 3)", x_halton, y_halton, False)
    ]
    
    for ax, (title, x_pts, y_pts, draw_grid) in zip(axes.flatten(), methods):
        inside = (x_pts**2 + y_pts**2) <= 1.0
        
        # Punkte zeichnen (blau = innen, rot = außen)
        ax.scatter(x_pts[inside], y_pts[inside], s=12, color="#1f77b4", alpha=0.75, label="Innen")
        ax.scatter(x_pts[~inside], y_pts[~inside], s=12, color="#d62728", alpha=0.75, label="Außen")
        
        # Viertelkreisbogen einzeichnen
        ax.plot(arc_x, arc_y, "k-", linewidth=2.5, label=r"$x^2 + y^2 = 1$")
        
        # Optional: Gitterlinien für Stratified Sampling
        if draw_grid:
            grid_lines = np.linspace(0, 1, M + 1)
            for gl in grid_lines:
                ax.axvline(gl, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
                ax.axhline(gl, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
                
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="lower left", fontsize=8)
        
    plt.tight_layout()
    pattern_file = "sampling_patterns.png"
    plt.savefig(pattern_file)
    print(f"Sampling-Muster-Plot gespeichert als: {pattern_file}")
    plt.show()


def verify_faster_convergence():
    """
    Führt die vollständige numerische Untersuchung der Konvergenzraten durch.
    
    FUNKTIONSWEISE:
    1. Definiert eine Reihe von Zweierpotenzen N = 2^6 bis 2^20 (64 bis 1.048.576 Punkte).
    2. Führt für jedes N Berechnungen mit allen vier Verfahren durch:
       - Für stochastische Verfahren (PRNG und Stratified) wird über 15 unabhängige
         Wiederholungen gemittelt, um stochastische Fluktuationen zu glätten.
       - Für QMC (Sobol und Halton) wird der deterministische Fehler berechnet.
    3. Gibt die Fehlerwerte tabellarisch in der Konsole aus.
    4. Berechnet die empirischen Konvergenzraten alpha über fit_convergence_rate().
    5. Erzeugt einen doppelt-logarithmischen Plot (loglog) mit:
       - Den vier Messkurven
       - Den theoretischen Referenzlinien:
           * O(N^-0.50) = 1 / sqrt(N)
           * O(N^-0.75) = 1 / N^(3/4)
           * O(N^-1.00) = 1 / N
    6. Speichert die Grafik als 'faster_convergence.png' ab.
    7. Zeigt den Plot interaktiv mit plt.show() an.
    """
    # Zweierpotenzen von 2^6 (64) bis 2^20 (1.048.576)
    m_values = list(range(6, 21))
    N_values = [2**m for m in m_values]
    
    mc_std_errors = []
    strat_errors = []
    sobol_errors = []
    halton_errors = []
    
    num_trials = 15  # Anzahl der Wiederholungen für stochastische Verfahren
    
    print("=" * 76)
    print("  VERGLEICH DER KONVERGENZRATEN ZUR BERECHNUNG VON PI (Aufgabe 2d)")
    print("=" * 76)
    print(f"{'N':>10} | {'Std-MC (PRNG)':>14} | {'Stratified':>14} | {'Sobol (QMC)':>14} | {'Halton (QMC)':>14}")
    print("-" * 76)
    
    for m, N in zip(m_values, N_values):
        # 1. Standard-MC: Mittelwert über num_trials unabhängige Läufe
        std_errs = [abs(mc_standard(N, seed=seed) - np.pi) for seed in range(num_trials)]
        avg_std_err = float(np.mean(std_errs))
        mc_std_errors.append(avg_std_err)
        
        # 2. Stratified Sampling: Mittelwert über num_trials unabhängige Läufe
        strat_errs = []
        for seed in range(num_trials):
            actual_N, p_strat = mc_stratified(N, seed=seed)
            strat_errs.append(abs(p_strat - np.pi))
        avg_strat_err = float(np.mean(strat_errs))
        strat_errors.append(avg_strat_err)
        
        # 3. Sobol-Folge: Deterministische Auswertung
        _, p_sobol = mc_sobol(m, scramble=False)
        err_sobol = abs(p_sobol - np.pi)
        sobol_errors.append(err_sobol)
        
        # 4. Halton-Folge: Deterministische Auswertung
        p_halton = mc_halton(N)
        err_halton = abs(p_halton - np.pi)
        halton_errors.append(err_halton)
        
        print(f"{N:>10,d} | {avg_std_err:>14.6e} | {avg_strat_err:>14.6e} | {err_sobol:>14.6e} | {err_halton:>14.6e}")
        
    # Gefittete Konvergenzraten (alpha in Fehler ~ N^(-alpha))
    alpha_mc = fit_convergence_rate(N_values, mc_std_errors)
    alpha_strat = fit_convergence_rate(N_values, strat_errors)
    alpha_sobol = fit_convergence_rate(N_values, sobol_errors)
    alpha_halton = fit_convergence_rate(N_values, halton_errors)
    
    print("-" * 76)
    print("EMPIDISCHE KONVERGENZRATEN (Steigung im Log-Log-Plot: Fehler ~ N^-alpha):")
    print(f"  Standard Pseudo-Random MC:  alpha = {alpha_mc:.2f}  (Theorie: 0.50 = 1/sqrt(N))")
    print(f"  Stratified Sampling:        alpha = {alpha_strat:.2f}  (Theorie: 0.75 = 1/N^(3/4))")
    print(f"  Quasi-Monte-Carlo (Sobol):  alpha = {alpha_sobol:.2f}  (Theorie: ~0.75 bis 1.0)")
    print(f"  Quasi-Monte-Carlo (Halton): alpha = {alpha_halton:.2f}  (Theorie: ~0.75 bis 1.0)")
    print("=" * 76)
    
    # Erstellen der Vergleichsgrafik
    plt.figure(figsize=(10, 6.5), dpi=150)
    plt.loglog(N_values, mc_std_errors, 'o-', color='#1f77b4', label=f'Standard MC (fit: $N^{{-{alpha_mc:.2f}}}$)', markersize=5)
    plt.loglog(N_values, strat_errors, 's-', color='#ff7f0e', label=f'Stratified Sampling (fit: $N^{{-{alpha_strat:.2f}}}$)', markersize=5)
    plt.loglog(N_values, sobol_errors, '^-', color='#2ca02c', label=f'QMC Sobol (fit: $N^{{-{alpha_sobol:.2f}}}$)', markersize=5)
    plt.loglog(N_values, halton_errors, 'd-', color='#9467bd', label=f'QMC Halton (fit: $N^{{-{alpha_halton:.2f}}}$)', markersize=5)
    
    # Theoretische Referenzlinien
    ref_N = np.array(N_values, dtype=float)
    plt.loglog(ref_N, 1.6 / np.sqrt(ref_N), 'k--', alpha=0.5, label=r'Theorie-Referenz: $O(N^{-1/2}) = 1/\sqrt{N}$')
    plt.loglog(ref_N, 1.6 / (ref_N**0.75), 'k:', alpha=0.7, label=r'Theorie-Referenz: $O(N^{-3/4}) = 1/N^{3/4}$')
    plt.loglog(ref_N, 2.0 / ref_N, 'k-.', alpha=0.5, label=r'Theorie-Referenz: $O(N^{-1}) = 1/N$')
    
    plt.xlabel('Anzahl der Punkte $N$', fontsize=12)
    plt.ylabel(r'Absoluter Fehler $|\hat{\pi} - \pi|$', fontsize=12)
    plt.title(r'Konvergenzvergleich zur $\pi$-Berechnung: Standard-MC vs. Stratified & QMC', fontsize=13, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    
    # 1. Speichern als eigenständiges .png
    conv_file = "faster_convergence.png"
    plt.savefig(conv_file)
    print(f"Konvergenz-Plot gespeichert als: {conv_file}")
    
    # 2. Interaktiv anzeigen
    plt.show()


if __name__ == "__main__":
    # 1. Visualisierung der Sampling-Muster (interaktiv anzeigen + als .png speichern)
    visualize_sampling_patterns(N_sample=576)
    
    # 2. Numerische Untersuchung der Konvergenzraten (interaktiv anzeigen + als .png speichern)
    verify_faster_convergence()
