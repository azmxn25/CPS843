# ============================================================
# Assignment 3 - Model Fitting & Robust Estimation
# Name: Azman Nazim
# Date: 2026-11-01
# ============================================================
from pathlib import Path
import csv
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT = Path(__file__).resolve().parent
# Data reproduced from Assignment 3, pages 1-3.
x1 = np.arange(1.0, 11.0)
y1 = np.array([2.50, 4.10, 6.80, 10.70, 15.60,
               21.40, 28.30, 36.10, 45.00, 54.80])
x2 = np.arange(1.0, 8.1, 0.5)
y2 = np.array([2.20, 3.10, 14.50, 4.90, 5.80,
               -3.50, 7.90, 8.80, 19.20, 10.90,
               11.80, 1.00, 14.70, 22.50, 16.60])
def normal_equations(X, y):
    """OLS: (X.T X) w = X.T y. Solve, rather than explicitly invert."""
    return np.linalg.solve(X.T @ X, X.T @ y)
def design_linear(x):
    return np.column_stack((x, np.ones_like(x)))
def design_quad(x):
    return np.column_stack((x**2, x, np.ones_like(x)))
def mse(y, yhat):
    return float(np.mean((y - yhat)**2))
def ransac_line(x, y, threshold=1.5, iterations=50, seed=42):
    """2-point RANSAC, vertical residuals, maximum consensus, OLS refit.
    Tie breaker: smaller sum of squared residuals on the consensus set.
    The final inlier mask is determined from the best *hypothesis*, before refit.
    """
    rng = np.random.default_rng(seed)
    best_mask = None
    best_count = -1
    best_error = np.inf
    best_hypothesis = None
    for _ in range(iterations):
        a, b = rng.choice(len(x), size=2, replace=False)
        if np.isclose(x[a], x[b]):
            continue
        slope = (y[b] - y[a]) / (x[b] - x[a])
        intercept = y[a] - slope*x[a]
        distances = np.abs(y - (slope*x + intercept))
        mask = distances < threshold  # strict threshold, as in assignment example
        count = int(mask.sum())
        error = float(np.sum(distances[mask]**2))
        if count > best_count or (count == best_count and error < best_error):
            best_mask, best_count, best_error = mask, count, error
            best_hypothesis = (float(slope), float(intercept))
    if best_mask is None or best_count < 2:
        raise RuntimeError('Insufficient inliers to fit a line')
    m, c = normal_equations(design_linear(x[best_mask]), y[best_mask])
    return {'m': float(m), 'c': float(c), 'mask': best_mask,
            'n_inliers': best_count, 'hypothesis': best_hypothesis,
            'inlier_mse': mse(y[best_mask], m*x[best_mask]+c),
            'all_mse': mse(y, m*x+c)}
def save_rows(filename, header, rows):
    with (OUT / filename).open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
def main():
    # PART 1: fitted linear and quadratic coefficients, MSE and prediction.
    wa = normal_equations(design_linear(x1), y1)
    wb = normal_equations(design_quad(x1), y1)
    pred_a = float((design_linear(np.array([12.0])) @ wa)[0])
    pred_b = float((design_quad(np.array([12.0])) @ wb)[0])
    yhat_a = design_linear(x1) @ wa
    yhat_b = design_quad(x1) @ wb
    mse_a, mse_b = mse(y1, yhat_a), mse(y1, yhat_b)
    save_rows('part1_comparison.csv', ['Model', 'Coefficients (highest degree first)', 'Training MSE', 'Prediction at x=12'], [
        ['Linear', ' / '.join(f'{z:.8f}' for z in wa), mse_a, pred_a],
        ['Quadratic', ' / '.join(f'{z:.8f}' for z in wb), mse_b, pred_b]
    ])
    grid = np.linspace(0, 13, 500)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x1, y1, s=60, c='black', label='Observed data (n=10)', zorder=5)
    ax.plot(grid, design_linear(grid) @ wa, lw=2.3, label='Model A: linear OLS')
    ax.plot(grid, design_quad(grid) @ wb, lw=2.3, label='Model B: quadratic OLS')
    ax.scatter([12], [pred_a], marker='*', s=240, edgecolor='black', linewidth=0.5,
               color='tab:blue', label=f'Linear prediction: {pred_a:.2f}', zorder=6)
    ax.scatter([12], [pred_b], marker='*', s=240, edgecolor='black', linewidth=0.5,
               color='tab:orange', label=f'Quadratic prediction: {pred_b:.2f}', zorder=6)
    ax.axvline(10, color='grey', ls=':', lw=1.2, label='Training boundary (x=10)')
    ax.set(xlabel='Input x', ylabel='Observed / predicted y', title='Figure 1. Least-squares fitting and prediction at x = 12', xlim=(0, 13))
    ax.grid(alpha=0.25); ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(OUT/'figure1_least_squares.png', dpi=300); plt.close(fig)
    # PART 2: complete-data OLS versus hand-coded RANSAC.
    w_ols = normal_equations(design_linear(x2), y2)
    r = ransac_line(x2, y2, threshold=1.5, iterations=50, seed=42)
    save_rows('part2_point_classification.csv', ['Point index', 'x', 'y', 'RANSAC status (t=1.5, S=50)'], [
        [i+1, x, y, 'Inlier' if inc else 'Outlier']
        for i, (x, y, inc) in enumerate(zip(x2, y2, r['mask']))
    ])
    base_ols_mse = mse(y2, design_linear(x2) @ w_ols)
    inlier_ols_mse = mse(y2[r['mask']], design_linear(x2[r['mask']]) @ w_ols)
    save_rows('part2_baseline_comparison.csv', ['Model', 'Slope', 'Intercept', 'Inliers', 'MSE all points', 'MSE RANSAC inliers'], [
        ['OLS (all points)', *w_ols, len(x2), base_ols_mse, inlier_ols_mse],
        ['RANSAC + OLS refit', r['m'], r['c'], r['n_inliers'], r['all_mse'], r['inlier_mse']]
    ])
    grid2 = np.linspace(0.5, 8.5, 400)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x2[r['mask']], y2[r['mask']], s=70, color='tab:green', marker='o', label='RANSAC inliers')
    ax.scatter(x2[~r['mask']], y2[~r['mask']], s=85, color='tab:red', marker='x', label='RANSAC outliers')
    ax.plot(grid2, design_linear(grid2) @ w_ols, color='tab:blue', lw=2.3, label='OLS: all 15 points')
    ax.plot(grid2, r['m']*grid2+r['c'], color='tab:orange', lw=2.3, label='RANSAC: refitted inlier line')
    ax.set(xlabel='Input x', ylabel='Observed y', title='Figure 2. Standard OLS versus robust RANSAC')
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(OUT/'figure2_ransac_vs_ols.png', dpi=300); plt.close(fig)
    # Hyperparameter sweeps, independent fixed seeds for reproducibility.
    configs = [(0.5,50), (1.5,50), (4.0,50), (1.5,5)]
    sweep = []
    results = []
    for t, iterations in configs:
        fit = ransac_line(x2, y2, threshold=t, iterations=iterations, seed=42)
        results.append(fit)
        sweep.append([t, iterations, fit['n_inliers'], fit['m'], fit['c'], fit['inlier_mse']])
    save_rows('part2_parameter_sweep.csv', ['Threshold t', 'Iterations S', 'Inliers', 'Slope', 'Intercept', 'Inlier MSE'], sweep)
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, (t, iters), fit in zip(axs.flat, configs, results):
        ax.scatter(x2[fit['mask']], y2[fit['mask']], c='tab:green', label='Inlier')
        ax.scatter(x2[~fit['mask']], y2[~fit['mask']], c='tab:red', marker='x', label='Outlier')
        ax.plot(grid2, fit['m']*grid2+fit['c'], color='tab:orange')
        ax.set_title(f't={t:g}, S={iters}; {fit["n_inliers"]}/15 inliers')
        ax.grid(alpha=0.2)
        ax.set_xlabel('x'); ax.set_ylabel('y')
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('Figure 3. RANSAC threshold and iteration comparison', y=1.06)
    fig.tight_layout()
    fig.savefig(OUT/'figure3_ransac_sweep.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    # Estimate run-to-run stability over 100 random seeds (additional analysis).
    stability = []
    for iters in (5, 50):
        fits = [ransac_line(x2, y2, 1.5, iters, seed=s) for s in range(100)]
        counts = np.array([f['n_inliers'] for f in fits]); slopes = np.array([f['m'] for f in fits])
        stability.append([iters, float(np.mean(counts)), float(np.std(counts)),
                          float(np.mean(slopes)), float(np.std(slopes)),
                          float(np.mean(counts == 10))])
    save_rows('part2_seed_stability.csv', ['S', 'Mean inliers', 'SD inliers', 'Mean slope', 'SD slope', 'Fraction with 10 inliers'], stability)
    # Week 6 slides p.11: number of trials for target confidence p.
    # Here inlier fraction is 10/15, sample size s=2, p=.99.
    p, inlier_fraction, sample_size = 0.99, 10/15, 2
    required_trials = math.ceil(math.log(1-p) / math.log(1-inlier_fraction**sample_size))
    print('\nPART 1 — OLS (coefficients highest degree first)')
    print(f'Linear     w={wa}; MSE={mse_a:.6f}; y(12)={pred_a:.6f}')
    print(f'Quadratic  w={wb}; MSE={mse_b:.6f}; y(12)={pred_b:.6f}')
    print('\nPART 2 — Robust estimation')
    print(f'Full OLS: slope={w_ols[0]:.6f}, intercept={w_ols[1]:.6f}, all MSE={base_ols_mse:.6f}')
    print(f'RANSAC:   slope={r["m"]:.6f}, intercept={r["c"]:.6f}, inliers={r["n_inliers"]}/15, inlier MSE={r["inlier_mse"]:.6f}')
    print('Inlier indices (1-based):', np.flatnonzero(r['mask'])+1)
    print('Outlier indices (1-based):', np.flatnonzero(~r['mask'])+1)
    print('\nThreshold/iteration sweep [t, S, inliers, slope, intercept, inlier MSE]:')
    for row in sweep: print(row)
    print('\nSeed stability [S, mean inliers, sd inliers, mean slope, sd slope, fraction 10 inliers]:')
    for row in stability: print(row)
    print(f'Week 6 p.11 formula: minimum theoretical trials for p=0.99: {required_trials}')
    print(f'Files written to: {OUT}')
if __name__ == '__main__':
    main()
