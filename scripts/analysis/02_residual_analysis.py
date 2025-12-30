#!/usr/bin/env python3
"""
Residual Analysis: Phrase-Final Lengthening
============================================

Analyzes duration residuals after controlling for lexical length to
demonstrate phrase-final lengthening effect.

Method:
1. Fit baseline model: duration ~ phone_count + syllable_count
2. Compute residuals = true_duration - predicted_duration
3. Compare residuals at phrase boundaries vs non-boundaries
4. Statistical test for phrase-final lengthening

Usage:
    python 02_residual_analysis.py

Output:
    - results/residual_analysis.json
    - results/phrase_final_lengthening.png
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json
from sklearn.linear_model import LinearRegression
from scipy import stats

# Add training scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))

# Feature indices
PHONE_COUNT = 3
SYLLABLE_COUNT = 2
PHRASE_BOUNDARY = 11


def load_data():
    """Load word-level data"""
    _dl = {'__file__': str(Path(__file__).parent.parent / 'training' / '01_data_loader.py')}
    exec(open(Path(__file__).parent.parent / 'training' / '01_data_loader.py').read(), _dl)
    
    def collect(split_name):
        phoneme_ds = _dl['PhonemeLevelDataset'](split_name=split_name)
        word_ds = _dl['WordLevelDataset'](phoneme_ds)
        features, targets = [], []
        for i in range(len(word_ds)):
            sample = word_ds[i]
            if sample['targets'] is not None:
                features.append(sample['features'].numpy())
                targets.append(sample['targets'].numpy())
        return np.array(features), np.array(targets)
    
    X_train, y_train = collect('train')
    X_test, y_test = collect('test')
    return X_train, y_train[:, 1], X_test, y_test[:, 1]


def run_residual_analysis():
    """Run phrase-final lengthening analysis"""
    print("=" * 70)
    print("RESIDUAL ANALYSIS: Phrase-Final Lengthening")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    X_train, dur_train, X_test, dur_test = load_data()
    print(f"Test: {len(X_test)} words")
    
    # Extract features
    phone_count = X_test[:, PHONE_COUNT]
    phrase_boundary = X_test[:, PHRASE_BOUNDARY]
    
    print(f"\nPhrase boundary distribution:")
    print(f"  Non-boundary: {np.sum(phrase_boundary == 0)}")
    print(f"  Boundary: {np.sum(phrase_boundary == 1)}")
    
    # Fit lexical length baseline
    X_lexical_train = X_train[:, [PHONE_COUNT, SYLLABLE_COUNT]]
    X_lexical_test = X_test[:, [PHONE_COUNT, SYLLABLE_COUNT]]
    
    model = LinearRegression()
    model.fit(X_lexical_train, dur_train)
    pred = model.predict(X_lexical_test)
    
    # Compute residuals
    residuals = dur_test - pred
    
    baseline_r2 = 1 - np.var(residuals) / np.var(dur_test)
    print(f"\nLexical baseline R²: {baseline_r2:.4f}")
    
    # Analyze by phrase boundary
    resid_non_boundary = residuals[phrase_boundary == 0]
    resid_boundary = residuals[phrase_boundary == 1]
    
    print(f"\n" + "-" * 50)
    print("RESIDUAL DURATION BY PHRASE BOUNDARY")
    print("-" * 50)
    
    print(f"\nNon-boundary words (n={len(resid_non_boundary)}):")
    print(f"  Mean residual: {resid_non_boundary.mean():.4f}")
    print(f"  Std: {resid_non_boundary.std():.4f}")
    
    print(f"\nPhrase-boundary words (n={len(resid_boundary)}):")
    print(f"  Mean residual: {resid_boundary.mean():.4f}")
    print(f"  Std: {resid_boundary.std():.4f}")
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(resid_boundary, resid_non_boundary)
    effect_size = (resid_boundary.mean() - resid_non_boundary.mean()) / np.sqrt(
        (resid_boundary.std()**2 + resid_non_boundary.std()**2) / 2
    )
    
    print(f"\n" + "-" * 50)
    print("STATISTICAL TEST")
    print("-" * 50)
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.2e}")
    print(f"Cohen's d: {effect_size:.4f}")
    
    diff = resid_boundary.mean() - resid_non_boundary.mean()
    if diff > 0 and p_value < 0.001:
        print(f"\n✓ PHRASE-FINAL LENGTHENING CONFIRMED!")
        print(f"  Words at phrase boundaries are {diff:.3f} longer (log-seconds)")
    
    # Save results
    output_dir = Path(__file__).parent.parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'baseline_r2': float(baseline_r2),
        'non_boundary': {
            'n': int(len(resid_non_boundary)),
            'mean': float(resid_non_boundary.mean()),
            'std': float(resid_non_boundary.std())
        },
        'boundary': {
            'n': int(len(resid_boundary)),
            'mean': float(resid_boundary.mean()),
            'std': float(resid_boundary.std())
        },
        'statistical_test': {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(effect_size),
            'difference': float(diff)
        }
    }
    
    with open(output_dir / 'residual_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Boxplot
    ax1 = axes[0]
    bp = ax1.boxplot([resid_non_boundary, resid_boundary], widths=0.6)
    ax1.set_xticklabels(['Non-boundary', 'Phrase boundary'])
    ax1.set_ylabel('Residual Duration (log-seconds)')
    ax1.set_title('Phrase-Final Lengthening Effect')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.scatter([1, 2], [resid_non_boundary.mean(), resid_boundary.mean()], 
                color='red', s=100, zorder=5, marker='D', label='Mean')
    ax1.legend()
    
    # Plot 2: Histogram
    ax2 = axes[1]
    ax2.hist(resid_non_boundary, bins=50, alpha=0.5, label=f'Non-boundary (n={len(resid_non_boundary)})', density=True)
    ax2.hist(resid_boundary, bins=50, alpha=0.5, label=f'Phrase boundary (n={len(resid_boundary)})', density=True)
    ax2.axvline(x=resid_non_boundary.mean(), color='blue', linestyle='--', alpha=0.7)
    ax2.axvline(x=resid_boundary.mean(), color='orange', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Residual Duration')
    ax2.set_ylabel('Density')
    ax2.set_title('Residual Distribution')
    ax2.legend()
    
    # Plot 3: Scatter
    ax3 = axes[2]
    ax3.scatter(phone_count[phrase_boundary == 0], dur_test[phrase_boundary == 0], 
                alpha=0.1, s=5, c='blue', label='Non-boundary')
    ax3.scatter(phone_count[phrase_boundary == 1], dur_test[phrase_boundary == 1], 
                alpha=0.3, s=10, c='red', label='Phrase boundary')
    ax3.set_xlabel('Phone Count')
    ax3.set_ylabel('Duration (log-seconds)')
    ax3.set_title('Duration vs Phone Count')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phrase_final_lengthening.png', dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_dir / 'phrase_final_lengthening.png'}")
    
    return results


if __name__ == '__main__':
    run_residual_analysis()


