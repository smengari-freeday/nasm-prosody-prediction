#!/usr/bin/env python3
# statistical tests comparing NASM vs MLP performance
#
# reads results from training output directories
# supports both phoneme-level and word-level analysis
#
# performs:
# 1. paired t-test (parametric)
# 2. Wilcoxon signed-rank test (non-parametric)
# 3. Cohen's d effect size
# 4. Bootstrap and t-distribution confidence intervals
# 5. Bonferroni correction for multiple comparisons
#
# variable naming follows GeeksforGeeks conventions:
# - Paired t-test: t, m, s, n (geeksforgeeks.org/data-science/paired-t-test-a-detailed-overview)
# - Effect Size: Cohen's d, M1, M2, SD (geeksforgeeks.org/maths/effect-size)
# - Wilcoxon: stat, p_value (geeksforgeeks.org/machine-learning/wilcoxon-signed-rank-test)
# - Bonferroni: alpha, m, alpha_bonferroni, significant (geeksforgeeks.org/r-language/bonferroni-test)
# - Bootstrap t-test: n_bootstrap, t_statistics, observed_t_statistic (geeksforgeeks.org/r-language/t-test-with-bootstrap-in-r)
# - CI: d, m, s, n, cl (geeksforgeeks.org/python/how-to-calculate-confidence-intervals-in-python)
#
# usage:
#   python 05_statistical_tests.py --level phoneme
#   python 05_statistical_tests.py --level word

import argparse
import json
import numpy as np
from scipy.stats import wilcoxon, ttest_rel, t as t_dist
from pathlib import Path

# paths
RESULTS_DIR = Path("/Users/s.mengari/Desktop/CODE2/results/training")
OUTPUT_DIR = Path("/Users/s.mengari/Desktop/CODE2/results/statistical_tests")
SEEDS = [13, 22, 42, 111, 222, 333]

# significance parameters (GeeksforGeeks Bonferroni naming)
alpha = 0.05  # desired significance level
cl = 1 - alpha  # confidence level (0.95)


def load_results(level: str) -> tuple[dict, dict, list]:
    """
    Load test R² from results.json for each seed.
    
    CRITICAL: Only include seeds where BOTH NASM and MLP results exist.
    This ensures paired alignment for statistical tests.
    """
    nasm_dir = RESULTS_DIR / "kan"
    mlp_dir = RESULTS_DIR / "mlp"
    
    nasm = {'f0': [], 'duration': [], 'energy': []}
    mlp = {'f0': [], 'duration': [], 'energy': []}
    paired_seeds = []  # track which seeds are actually used
    
    for seed in SEEDS:
        nasm_path = nasm_dir / f"{level}_seed{seed}" / "results.json"
        mlp_path = mlp_dir / f"{level}_seed{seed}" / "results.json"
        
        # CRITICAL: only include if BOTH exist (paired alignment)
        if not (nasm_path.exists() and mlp_path.exists()):
            print(f"  Skipping seed {seed}: missing {'NASM' if not nasm_path.exists() else 'MLP'} results")
            continue
        
        # load NASM
        with open(nasm_path) as f:
            nasm_data = json.load(f)
        nasm_metrics = nasm_data.get('test_metrics', nasm_data.get('val_metrics', {}))
        
        # load MLP
        with open(mlp_path) as f:
            mlp_data = json.load(f)
        mlp_metrics = mlp_data.get('test_metrics', mlp_data.get('val_metrics', {}))
        
        # append paired results
        for target in ['f0', 'duration', 'energy']:
            if target in nasm_metrics and target in mlp_metrics:
                nasm[target].append(nasm_metrics[target]['r2'])
                mlp[target].append(mlp_metrics[target]['r2'])
        
        paired_seeds.append(seed)
    
    return nasm, mlp, paired_seeds


def cohens_dz(x, y):
    """
    Cohen's dz (PAIRED effect size) - GeeksforGeeks convention.
    https://www.geeksforgeeks.org/maths/effect-size/
    
    For PAIRED data, we use dz (not d):
    dz = mean(difference) / std(difference)
    
    Note: Classic Cohen's d assumes independent groups.
    For paired/repeated measures, dz is the correct metric.
    """
    # M1, M2 = means of two groups
    M1 = np.mean(x)
    M2 = np.mean(y)
    
    # For paired samples, use difference scores
    diff = np.array(x) - np.array(y)
    SD = np.std(diff, ddof=1)  # standard deviation of differences
    
    # Cohen's dz = (M1 - M2) / SD_diff
    dz = (M1 - M2) / SD if SD > 0 else 0.0
    
    # Effect size interpretation (Cohen's guidelines)
    # Small: |dz| = 0.2, Medium: |dz| = 0.5, Large: |dz| = 0.8
    abs_dz = abs(dz)
    if abs_dz < 0.2:
        interpretation = 'negligible'
    elif abs_dz < 0.5:
        interpretation = 'small'
    elif abs_dz < 0.8:
        interpretation = 'medium'
    else:
        interpretation = 'large'
    
    return dz, interpretation


def bootstrap_ci(d, n_bootstrap=10000):
    """Bootstrap confidence interval for mean difference."""
    np.random.seed(42)
    boot_means = [np.mean(np.random.choice(d, size=len(d), replace=True)) 
                  for _ in range(n_bootstrap)]
    lo = np.percentile(boot_means, 100 * (1 - cl) / 2)
    hi = np.percentile(boot_means, 100 * (1 - (1 - cl) / 2))
    return lo, hi


def bootstrap_t_test(data1, data2, n_bootstrap=1000):
    """
    Bootstrap t-test following GeeksforGeeks convention.
    https://www.geeksforgeeks.org/r-language/t-test-with-bootstrap-in-r/
    
    Resamples data with replacement and computes t-statistic for each resample.
    Returns bootstrap p-value by comparing observed t-statistic to distribution.
    """
    n1 = len(data1)
    n2 = len(data2)
    t_statistics = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        sample1 = np.random.choice(data1, n1, replace=True)
        sample2 = np.random.choice(data2, n2, replace=True)
        # compute t-statistic for this bootstrap sample
        t_stat, _ = ttest_rel(sample1, sample2)
        t_statistics[i] = t_stat
    
    # observed t-statistic from original data
    observed_t_statistic, _ = ttest_rel(data1, data2)
    
    # p-value: proportion of bootstrap t-statistics >= observed (two-tailed)
    p_value = np.mean(np.abs(t_statistics) >= np.abs(observed_t_statistic))
    
    return t_statistics, observed_t_statistic, p_value


def t_interval(d):
    """Confidence interval using t-distribution (GeeksforGeeks method)."""
    m, s, n = np.mean(d), np.std(d, ddof=1), len(d)
    t_val = t_dist.ppf((1 + cl) / 2, df=n-1)
    e = t_val * (s / np.sqrt(n))  # margin of error
    return m - e, m + e


def run_tests(level: str):
    """Run all statistical tests for given level."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # load results from training directories (paired alignment guaranteed)
    nasm, mlp, paired_seeds = load_results(level)
    
    # check if we have data
    n = len(paired_seeds)  # n = number of paired samples
    if n == 0:
        print(f"No {level}-level results found in {RESULTS_DIR}")
        return
    
    if n < 3:
        print(f"WARNING: Only {n} paired seeds. Statistical tests require n >= 3.")
    
    # Bonferroni correction (GeeksforGeeks naming)
    # https://www.geeksforgeeks.org/r-language/bonferroni-test/
    m = 3  # number of tests (F0, duration, energy)
    alpha_bonferroni = alpha / m
    
    print("=" * 70)
    print(f"{level.upper()}-LEVEL STATISTICAL TESTS: NASM vs MLP")
    print(f"Paired seeds used: {paired_seeds} (n={n})")
    print(f"Alpha: {alpha}, Number of tests (m): {m}")
    print(f"Bonferroni-Corrected Significance Level: {alpha_bonferroni}")
    print("=" * 70)
    print()
    
    lines = []
    lines.append(f"{level.upper()}-LEVEL STATISTICAL TESTS: NASM vs MLP")
    lines.append(f"Paired seeds: {paired_seeds} (n={n})")
    lines.append(f"Alpha: {alpha}, Number of tests (m): {m}")
    lines.append(f"Bonferroni-Corrected Significance Level: {alpha_bonferroni}")
    lines.append("=" * 70)
    lines.append("")
    
    # collect p-values for Bonferroni summary
    p_values = []
    results = []
    
    for target in ['f0', 'duration', 'energy']:
        if len(nasm[target]) < 2 or len(mlp[target]) < 2:
            print(f"Skipping {target}: insufficient data")
            continue
        
        # d1=NASM, d2=MLP, d=difference (GeeksforGeeks naming)
        d1 = np.array(nasm[target])
        d2 = np.array(mlp[target])
        d = d1 - d2
        
        # m=mean, s=std (CI naming)
        m1, s1 = np.mean(d1), np.std(d1, ddof=1)
        m2, s2 = np.mean(d2), np.std(d2, ddof=1)
        
        # paired t-test: t = m/(s/√n)
        t, p_t = ttest_rel(d1, d2)
        p_values.append(p_t)
        
        # Wilcoxon signed-rank test: stat, p_value
        try:
            stat, p_value = wilcoxon(d1, d2)
        except Exception:
            stat, p_value = np.nan, np.nan
        
        # Cohen's dz (paired): dz = (M1-M2)/SD_diff
        dz, effect = cohens_dz(d1, d2)
        
        # 95% CI (t-distribution)
        t_lo, t_hi = t_interval(d)
        
        # Bonferroni significance
        significant_bonf = p_t < alpha_bonferroni
        
        results.append({
            'target': target,
            'M1': float(m1), 's1': float(s1),
            'M2': float(m2), 's2': float(s2),
            'delta': float(np.mean(d)),
            't': float(t), 'p_t': float(p_t),
            'w_stat': float(stat) if not np.isnan(stat) else None, 
            'p_value_w': float(p_value) if not np.isnan(p_value) else None,
            'dz': float(dz), 'effect': effect,
            'ci_lo': float(t_lo), 'ci_hi': float(t_hi),
            'significant_bonf': bool(significant_bonf)
        })
    
    # === RESULTS ===
    print()
    print("Results Summary")
    print("-" * 70)
    
    lines.append("")
    lines.append("Results Summary")
    lines.append("-" * 70)
    
    for r in results:
        print(f"\n{r['target'].upper()}:")
        print(f"  NASM mean: {r['M1']:.4f} (SD = {r['s1']:.4f})")
        print(f"  MLP mean:  {r['M2']:.4f} (SD = {r['s2']:.4f})")
        print(f"  Difference: {r['delta']:+.4f}")
        print(f"  t({n-1}) = {r['t']:.3f}, p = {r['p_t']:.4f}")
        print(f"  95% CI: [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
        print(f"  Cohen's dz = {r['dz']:.3f} ({r['effect']} effect)")
        
        if r['significant_bonf']:
            print(f"  --> Significant after Bonferroni correction (p < {alpha_bonferroni:.4f})")
        else:
            print(f"  --> Not significant after Bonferroni correction")
        
        lines.append(f"\n{r['target'].upper()}:")
        lines.append(f"  NASM mean: {r['M1']:.4f} (SD = {r['s1']:.4f})")
        lines.append(f"  MLP mean:  {r['M2']:.4f} (SD = {r['s2']:.4f})")
        lines.append(f"  Difference: {r['delta']:+.4f}")
        lines.append(f"  t({n-1}) = {r['t']:.3f}, p = {r['p_t']:.4f}")
        lines.append(f"  95% CI: [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
        lines.append(f"  Cohen's dz = {r['dz']:.3f} ({r['effect']} effect)")
        if r['significant_bonf']:
            lines.append(f"  --> Significant (p < {alpha_bonferroni:.4f})")
        else:
            lines.append(f"  --> Not significant")
    
    # simple table for thesis
    print()
    print("-" * 70)
    print("Table for thesis:")
    print("-" * 70)
    print(f"{'Target':<12} {'NASM':<12} {'MLP':<12} {'p-value':<10} {'dz':<10} {'Sig?':<8}")
    
    lines.append("")
    lines.append("-" * 70)
    lines.append("Table for thesis:")
    lines.append("-" * 70)
    lines.append(f"{'Target':<12} {'NASM':<12} {'MLP':<12} {'p-value':<10} {'dz':<10} {'Sig?':<8}")
    
    for r in results:
        sig = "Yes" if r['significant_bonf'] else "No"
        row = f"{r['target']:<12} {r['M1']:.3f}        {r['M2']:.3f}        {r['p_t']:.4f}     {r['dz']:.2f}       {sig}"
        print(row)
        lines.append(row)
    
    print("-" * 70)
    print(f"\nNote: Bonferroni-corrected alpha = {alpha_bonferroni:.4f} (0.05 / 3 tests)")
    print(f"n = {n} paired observations (seeds)")
    lines.append("-" * 70)
    lines.append(f"\nNote: Bonferroni-corrected alpha = {alpha_bonferroni:.4f}")
    lines.append(f"n = {n} paired observations")
    
    # save report
    out = OUTPUT_DIR / f'{level}_statistical_report.txt'
    with open(out, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nSaved: {out}")
    
    # save JSON for programmatic access
    json_out = OUTPUT_DIR / f'{level}_statistical_results.json'
    with open(json_out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {json_out}")


def main():
    parser = argparse.ArgumentParser(description='Statistical tests for NASM vs MLP')
    parser.add_argument('--level', type=str, default='phoneme',
                        choices=['phoneme', 'word'],
                        help='Analysis level: phoneme or word')
    args = parser.parse_args()
    
    run_tests(args.level)


if __name__ == '__main__':
    main()

