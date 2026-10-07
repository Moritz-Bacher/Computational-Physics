"""
Estimate Pi using the Monte Carlo method (quarter circle in unit square).

Exercise 0 - Task 2a:
Estimates pi for N = 10^2 to 10^7, computes the error |pi_est - pi|,
and plots error versus N on a log-log scale.
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt

def estimate_pi_vectorized(N: int, seed: int = None) -> float:
    """
    Vectorized NumPy implementation:
    Generates N random points in [0, 1]^2 and checks x^2 + y^2 <= 1.0.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=N)
    y = rng.uniform(0.0, 1.0, size=N)
    inside = (x**2 + y**2) <= 1.0
    return 4.0 * np.sum(inside) / N

# Maintain backward compatibility with initial function name
estimate_pi_monte_carlo = estimate_pi_vectorized

def estimate_pi_python_loop(N: int, seed: int = None) -> float:
    """
    Pure Python loop implementation:
    Generates points one by one in a standard for-loop using the random module.
    """
    if seed is not None:
        random.seed(seed)
    inside_count = 0
    for _ in range(N):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1.0:
            inside_count += 1
    return 4.0 * inside_count / N

def compare_timings():
    """
    Benchmarks runtime of vectorized NumPy vs pure Python loop across various N.
    """
    test_N = [10**2, 10**3, 10**4, 10**5, 10**6, 10**7]
    print("\n--- Timing Comparison: Vectorized NumPy vs Python Loop ---")
    print(f"{'N':>12} | {'Vectorized [s]':>16} | {'Python Loop [s]':>16} | {'Speedup':>10}")
    print("-" * 62)
    
    vec_times = []
    loop_times = []
    bench_N = []
    
    for N in test_N:
        # Time vectorized
        t0 = time.perf_counter()
        _ = estimate_pi_vectorized(N, seed=42)
        t_vec = time.perf_counter() - t0
        vec_times.append(t_vec)
        bench_N.append(N)
        
        # Pure Python loop can be slow for 10^7, but let's test up to 10^7
        if N <= 10**7:
            t0 = time.perf_counter()
            _ = estimate_pi_python_loop(N, seed=42)
            t_loop = time.perf_counter() - t0
            loop_times.append(t_loop)
            speedup = t_loop / t_vec
            print(f"{N:>12,d} | {t_vec:>16.6f} | {t_loop:>16.6f} | {speedup:>9.1f}x")
        else:
            loop_times.append(None)
            print(f"{N:>12,d} | {t_vec:>16.6f} | {'(skipped)':>16} | {'-':>10}")
            
    # Plot timing comparison
    plt.figure(figsize=(8, 5), dpi=150)
    plt.loglog(bench_N, vec_times, 'o-', color='#1f77b4', label='Vectorized (NumPy)', linewidth=2)
    plt.loglog(bench_N[:len(loop_times)], loop_times, 's-', color='#d62728', label='Python Loop', linewidth=2)
    plt.xlabel('Number of points $N$', fontsize=12)
    plt.ylabel('Execution time [seconds]', fontsize=12)
    plt.title('Timing Comparison: Vectorized vs Python Loop', fontsize=13, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    timing_plot = "timing_comparison.png"
    plt.savefig(timing_plot)
    print(f"Timing plot saved to {timing_plot}")


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
    compare_timings()
