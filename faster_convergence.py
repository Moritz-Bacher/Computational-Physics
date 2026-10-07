"""
Task 2d: Faster convergence than 1/sqrt(N) for estimating Pi.

Compares:
1. Standard Pseudo-Random Monte Carlo (PRNG) -> O(N^-0.5)
2. Stratified Sampling (grid-based jittered sampling) -> O(N^-0.75)
3. Quasi-Monte Carlo (Sobol sequence, low-discrepancy) -> O(N^-0.75 to N^-1.0)
4. Quasi-Monte Carlo (Halton sequence) -> O(N^-0.75 to N^-1.0)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

def mc_standard(N: int, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=N)
    y = rng.uniform(0.0, 1.0, size=N)
    inside = (x**2 + y**2) <= 1.0
    return 4.0 * np.sum(inside) / N

def mc_stratified(N_target: int, seed: int = 42) -> tuple[int, float]:
    """
    Stratified sampling: unit square divided into M x M cells.
    Draw exactly one uniform random point per cell.
    Total points N = M^2.
    """
    rng = np.random.default_rng(seed)
    M = int(np.round(np.sqrt(N_target)))
    N = M * M
    
    # Grid coordinates
    i = np.arange(M)
    j = np.arange(M)
    ii, jj = np.meshgrid(i, j)
    
    # Jitter within each cell [ii/M, (ii+1)/M] x [jj/M, (jj+1)/M]
    u_x = rng.uniform(0.0, 1.0, size=(M, M))
    u_y = rng.uniform(0.0, 1.0, size=(M, M))
    
    x = (ii + u_x) / M
    y = (jj + u_y) / M
    
    inside = (x**2 + y**2) <= 1.0
    pi_est = 4.0 * np.sum(inside) / N
    return N, pi_est

def mc_sobol(m: int, scramble: bool = False) -> tuple[int, float]:
    """
    Quasi-Monte Carlo using Sobol sequence with N = 2^m points.
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
    Quasi-Monte Carlo using Halton sequence.
    """
    sampler = qmc.Halton(d=2, scramble=False)
    pts = sampler.random(N)
    x = pts[:, 0]
    y = pts[:, 1]
    inside = (x**2 + y**2) <= 1.0
    return 4.0 * np.sum(inside) / N

def verify_faster_convergence():
    # Power of 2 sample sizes to match Sobol sequences perfectly
    m_values = list(range(6, 21))  # 64 to ~1,048,576
    N_values = [2**m for m in m_values]
    
    mc_std_errors = []
    strat_errors = []
    sobol_errors = []
    halton_errors = []
    
    # Multiple trials for stochastic methods to get smooth expectation curves
    num_trials = 15
    
    print(f"{'N':>10} | {'Std MC Err':>12} | {'Stratified Err':>15} | {'Sobol Err':>12} | {'Halton Err':>12}")
    print("-" * 72)
    
    for m, N in zip(m_values, N_values):
        # 1. Standard MC (average error over trials)
        std_errs = [abs(mc_standard(N, seed=seed) - np.pi) for seed in range(num_trials)]
        avg_std_err = np.mean(std_errs)
        mc_std_errors.append(avg_std_err)
        
        # 2. Stratified Sampling (average error over trials)
        strat_errs = []
        for seed in range(num_trials):
            actual_N, p_strat = mc_stratified(N, seed=seed)
            strat_errs.append(abs(p_strat - np.pi))
        avg_strat_err = np.mean(strat_errs)
        strat_errors.append(avg_strat_err)
        
        # 3. Sobol (deterministic low-discrepancy)
        _, p_sobol = mc_sobol(m, scramble=False)
        err_sobol = abs(p_sobol - np.pi)
        sobol_errors.append(err_sobol)
        
        # 4. Halton (deterministic low-discrepancy)
        p_halton = mc_halton(N)
        err_halton = abs(p_halton - np.pi)
        halton_errors.append(err_halton)
        
        print(f"{N:>10,d} | {avg_std_err:>12.6e} | {avg_strat_err:>15.6e} | {err_sobol:>12.6e} | {err_halton:>12.6e}")
        
    # Fit power law Error = C * N^(-alpha) using linear regression on log-log
    def fit_slope(n_arr, err_arr):
        valid = (np.array(err_arr) > 0)
        x = np.log(np.array(n_arr)[valid])
        y = np.log(np.array(err_arr)[valid])
        slope, intercept = np.polyfit(x, y, 1)
        return -slope  # convergence rate alpha
        
    alpha_mc = fit_slope(N_values, mc_std_errors)
    alpha_strat = fit_slope(N_values, strat_errors)
    alpha_sobol = fit_slope(N_values, sobol_errors)
    alpha_halton = fit_slope(N_values, halton_errors)
    
    print("\n--- Fitted Convergence Rates (Error ~ N^(-alpha)) ---")
    print(f"Standard Pseudo-Random MC:  alpha = {alpha_mc:.2f} (Theory: 0.50 = 1/sqrt(N))")
    print(f"Stratified Sampling:        alpha = {alpha_strat:.2f} (Theory: 0.75 = 1/N^(3/4))")
    print(f"Quasi-Monte Carlo (Sobol):  alpha = {alpha_sobol:.2f} (Theory: ~0.75 to 1.0)")
    print(f"Quasi-Monte Carlo (Halton): alpha = {alpha_halton:.2f} (Theory: ~0.75 to 1.0)")
    
    # Plotting
    plt.figure(figsize=(10, 6.5), dpi=150)
    plt.loglog(N_values, mc_std_errors, 'o-', color='#1f77b4', label=f'Standard MC (fit: $N^{{-{alpha_mc:.2f}}}$)', markersize=5)
    plt.loglog(N_values, strat_errors, 's-', color='#ff7f0e', label=f'Stratified Sampling (fit: $N^{{-{alpha_strat:.2f}}}$)', markersize=5)
    plt.loglog(N_values, sobol_errors, '^-', color='#2ca02c', label=f'QMC Sobol (fit: $N^{{-{alpha_sobol:.2f}}}$)', markersize=5)
    plt.loglog(N_values, halton_errors, 'd-', color='#9467bd', label=f'QMC Halton (fit: $N^{{-{alpha_halton:.2f}}}$)', markersize=5)
    
    # Reference guide lines
    ref_N = np.array(N_values, dtype=float)
    plt.loglog(ref_N, 1.6 / np.sqrt(ref_N), 'k--', alpha=0.5, label=r'Reference: $O(N^{-1/2})$')
    plt.loglog(ref_N, 1.6 / (ref_N**0.75), 'k:', alpha=0.7, label=r'Reference: $O(N^{-3/4})$')
    plt.loglog(ref_N, 2.0 / ref_N, 'k-.', alpha=0.5, label=r'Reference: $O(N^{-1})$')
    
    plt.xlabel('Number of points $N$', fontsize=12)
    plt.ylabel(r'Error $|\hat{\pi} - \pi|$', fontsize=12)
    plt.title('Comparison of Convergence Rates for Pi Estimation', fontsize=14, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    
    out_plot = "faster_convergence.png"
    plt.savefig(out_plot)
    print(f"\nPlot saved to {out_plot}")

if __name__ == "__main__":
    verify_faster_convergence()
