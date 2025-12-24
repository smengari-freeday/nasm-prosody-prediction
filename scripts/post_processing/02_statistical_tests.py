#!/usr/bin/env python3
# statistical tests comparing KAN vs MLP performance
#
# tests performed:
# 1. paired t-test (parametric)
# 2. Wilcoxon signed-rank test (non-parametric)
# 3. Cohen's d effect size
# 4. Bootstrap 95% confidence interval
# 5. Bonferroni correction for multiple comparisons
#
# usage:
#   python 02_statistical_tests.py

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# paths (hardcoded)
INPUT_CSV = Path("/Users/s.mengari/Desktop/CODE2/results/model_comparison.csv")
OUTPUT_DIR = Path("/Users/s.mengari/Desktop/CODE2/results/statistical_tests")
ALPHA = 0.05


def paired_cohens_d(x: np.ndarray, y: np.ndarray) -> dict:
    # paired Cohen's d (dz) = mean(diff) / std(diff)
    diff = x - y
    mean_diff = np.nanmean(diff)
    std_diff = np.nanstd(diff, ddof=1)
    
    dz = mean_diff / std_diff if std_diff > 0 else 0.0
    
    abs_dz = abs(dz)
    if abs_dz < 0.2:
        interpretation = 'negligible'
    elif abs_dz < 0.5:
        interpretation = 'small'
    elif abs_dz < 0.8:
        interpretation = 'medium'
    else:
        interpretation = 'large'
    
    return {'dz': dz, 'mean_diff': mean_diff, 'std_diff': std_diff, 'interpretation': interpretation}


def bootstrap_ci(diff: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95) -> dict:
    diff = diff[np.isfinite(diff)]
    
    if len(diff) < 2:
        return {'lower': np.nan, 'upper': np.nan, 'median': np.nan}
    
    np.random.seed(42)
    boot_means = [np.mean(np.random.choice(diff, size=len(diff), replace=True)) 
                  for _ in range(n_bootstrap)]
    
    alpha = 1 - ci
    return {
        'lower': np.percentile(boot_means, 100 * alpha / 2),
        'upper': np.percentile(boot_means, 100 * (1 - alpha / 2)),
        'median': np.percentile(boot_means, 50)
    }


def run_statistical_tests(df: pd.DataFrame, alpha: float = 0.05) -> tuple:
    kan_df = df[df['model'] == 'KAN'].pivot(index='seed', columns='target', values='r2')
    mlp_df = df[df['model'] == 'MLP'].pivot(index='seed', columns='target', values='r2')
    
    n_tests = 3
    bonferroni_alpha = alpha / n_tests
    
    results = []
    report_lines = []
    
    report_lines.append("=" * 70)
    report_lines.append("STATISTICAL TESTS: KAN vs MLP")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    for target in ['f0', 'duration', 'energy']:
        kan_vals = kan_df[target].values
        mlp_vals = mlp_df[target].values
        
        kan_mean, mlp_mean = np.mean(kan_vals), np.mean(mlp_vals)
        kan_std, mlp_std = np.std(kan_vals, ddof=1), np.std(mlp_vals, ddof=1)
        
        report_lines.append(f"=== {target.upper()} ===")
        report_lines.append(f"KAN:  R² = {kan_mean:.4f} ± {kan_std:.4f}")
        report_lines.append(f"MLP:  R² = {mlp_mean:.4f} ± {mlp_std:.4f}")
        report_lines.append(f"Diff: {kan_mean - mlp_mean:+.4f}")
        report_lines.append("")
        
        # paired t-test
        t_stat, t_pval = stats.ttest_rel(kan_vals, mlp_vals)
        report_lines.append(f"Paired t-test: t={t_stat:.3f}, p={t_pval:.4f}")
        
        # Wilcoxon signed-rank test
        try:
            w_stat, w_pval = stats.wilcoxon(kan_vals, mlp_vals)
            report_lines.append(f"Wilcoxon test: W={w_stat:.3f}, p={w_pval:.4f}")
        except:
            w_stat, w_pval = np.nan, np.nan
            report_lines.append("Wilcoxon test: N/A (insufficient variation)")
        
        # Cohen's d
        cd = paired_cohens_d(kan_vals, mlp_vals)
        report_lines.append(f"Cohen's d: {cd['dz']:.3f} ({cd['interpretation']})")
        
        # Bootstrap CI
        diff = kan_vals - mlp_vals
        ci = bootstrap_ci(diff)
        report_lines.append(f"95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
        
        sig = 'YES' if t_pval < bonferroni_alpha else 'NO'
        report_lines.append(f"Significant (alpha={bonferroni_alpha:.4f}): {sig}")
        report_lines.append("")
        
        results.append({
            'target': target,
            'kan_mean': kan_mean, 'kan_std': kan_std,
            'mlp_mean': mlp_mean, 'mlp_std': mlp_std,
            'diff': kan_mean - mlp_mean,
            't_stat': t_stat, 't_pval': t_pval,
            'wilcoxon_stat': w_stat, 'wilcoxon_pval': w_pval,
            'cohens_d': cd['dz'], 'effect_size': cd['interpretation'],
            'ci_lower': ci['lower'], 'ci_upper': ci['upper'],
            'significant_bonferroni': sig
        })
    
    # summary table
    report_lines.append("=" * 70)
    report_lines.append("SUMMARY TABLE (R² comparison)")
    report_lines.append("-" * 50)
    report_lines.append(f"{'Target':<12} {'KAN':<15} {'MLP':<15} {'Diff':<10} {'p-value':<10}")
    report_lines.append("-" * 50)
    for r in results:
        report_lines.append(
            f"{r['target']:<12} {r['kan_mean']:.3f}±{r['kan_std']:.3f}    "
            f"{r['mlp_mean']:.3f}±{r['mlp_std']:.3f}    {r['diff']:+.3f}     {r['t_pval']:.4f}"
        )
    
    return pd.DataFrame(results), "\n".join(report_lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")
    
    results_df, report = run_statistical_tests(df, ALPHA)
    
    results_df.to_csv(OUTPUT_DIR / 'kan_vs_mlp_tests.csv', index=False)
    print(f"Saved: {OUTPUT_DIR / 'kan_vs_mlp_tests.csv'}")
    
    with open(OUTPUT_DIR / 'statistical_report.txt', 'w') as f:
        f.write(report)
    print(f"Saved: {OUTPUT_DIR / 'statistical_report.txt'}")
    
    print()
    print(report)


if __name__ == '__main__':
    main()
