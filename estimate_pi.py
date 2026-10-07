"""
Estimate Pi using the Monte Carlo method (quarter circle in unit square).

Exercise 0 - Task 2a:
Estimates pi for N = 10^2 to 10^7, computes the error |pi_est - pi|,
and plots error versus N on a log-log scale.
"""

import numpy as np
import matplotlib.pyplot as plt

def estimate_pi_monte_carlo(N: int, seed: int = None) -> float:
    """
    Estimate pi by drawing N uniform random points in [0, 1]^2.
    A point (x, y) is inside the quarter circle if x^2 + y^2 <= 1.0.
    Points exactly on the boundary are included.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=N)
    y = rng.uniform(0.0, 1.0, size=N)
    
    # Check quarter circle condition: x^2 + y^2 <= 1
    inside = (x**2 + y**2) <= 1.0
    num_inside = np.sum(inside)
    
    pi_estimate = 4.0 * num_inside / N
    return pi_estimate

def run_experiment():
    # N values from 10^2 to 10^7
    N_values = np.array([10**k for k in range(2, 8)])  # [100, 1000, 10000, 100000, 1000000, 10000000]
    
    print(f"{'N':>12} | {'Pi Estimate':>14} | {'Absolute Error':>16} | {'Relative Error':>16}")
    print("-" * 66)
    
    estimates = []
    errors = []
    
    # Single run evaluation for N = 10^2 ... 10^7
    for N in N_values:
        pi_est = estimate_pi_monte_carlo(N, seed=42)
        err = abs(pi_est - np.pi)
        rel_err = err / np.pi
        estimates.append(pi_est)
        errors.append(err)
        print(f"{N:>12,d} | {pi_est:>14.7f} | {err:>16.7e} | {rel_err:>16.4%}")
        
    # Multiple trials to evaluate the statistical expectation (ensemble average)
    num_trials = 20
    mean_errors = []
    std_errors = []
    N_eval_trials = N_values[:-1]  # Up to 10^6 for multiple trials to keep execution snappy
    
    for N in N_eval_trials:
        trial_errors = []
        for trial_seed in range(num_trials):
            pi_est = estimate_pi_monte_carlo(N, seed=1000 + trial_seed)
            trial_errors.append(abs(pi_est - np.pi))
        mean_errors.append(np.mean(trial_errors))
        std_errors.append(np.std(trial_errors))
        
    # Plotting
    plt.figure(figsize=(9, 6), dpi=150)
    plt.loglog(N_values, errors, 'o-', color='#1f77b4', label=r'Single run error: $|\hat{\pi} - \pi|$', markersize=6)
    plt.loglog(N_eval_trials, mean_errors, 's--', color='#ff7f0e', label=f'Mean error ({num_trials} trials)', markersize=5)
    
    # Theoretical 1/sqrt(N) reference line
    # Standard deviation of the estimator is sqrt(pi * (4 - pi) / N) ~ 1.642 / sqrt(N)
    theoretical_sigma = np.sqrt(np.pi * (4.0 - np.pi) / N_values)
    plt.loglog(N_values, theoretical_sigma, 'k:', linewidth=2, label=r'Theoretical std $\sigma_N = \sqrt{\pi(4-\pi)/N} \propto N^{-1/2}$')
    
    plt.xlabel('Number of points $N$', fontsize=12)
    plt.ylabel(r'Error $|\hat{\pi} - \pi|$', fontsize=12)
    plt.title(r'Monte Carlo Estimation of $\pi$: Error vs $N$', fontsize=14, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    plot_filename = "mc_pi_error.png"
    plt.savefig(plot_filename)
    print(f"\nPlot saved to {plot_filename}")

if __name__ == "__main__":
    run_experiment()
