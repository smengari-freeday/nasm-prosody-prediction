#!/usr/bin/env python3
# =============================================================================
# Generate Thesis Visualizations for NASM vs MLP Comparison
# =============================================================================
#
# Creates:
# 1. Paired seed plots (slopegraph) - main figure
# 2. Effect size plot (Cohen's dz) with significance markers
# 3. Seed stability boxplots
#
# Statistical methods:
# - Cohen's dz: Cohen (1988), Lakens (2013)
# - CI: t-based confidence intervals (scipy.stats.t)
# - Paired comparison: sign convention is NASM - MLP (positive = NASM better)
#
# Plotting:
# - Matplotlib boxplot: https://www.geeksforgeeks.org/box-plot-in-python-using-matplotlib/
#
# Usage:
#   python 04_generate_visualizations.py [--level phoneme|word]

import json
import argparse
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from scipy import stats as scipy_stats

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path("/Users/s.mengari/Desktop/CODE2")
RESULTS_DIR = PROJECT_ROOT / "results" / "training"
STATS_DIR = PROJECT_ROOT / "results" / "statistical_tests"
OUTPUT_DIR = PROJECT_ROOT / "results" / "visualizations_final"

# Note: Training folders use "kan" but thesis calls it "NASM"
# NASM = Neural Additive Spline Model (thesis terminology)
NASM_FOLDER = "kan"  # folder name in training results

SEEDS = [13, 22, 42, 111, 222, 333]
TARGETS = ['f0', 'duration', 'energy']
TARGET_DISPLAY = {'f0': 'F0', 'duration': 'Duration', 'energy': 'Energy'}

# Style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'nasm': '#2E86AB', 'mlp': '#A23B72'}


# =============================================================================
# Data Loading (with proper seed alignment)
# =============================================================================

def load_results(level: str = 'phoneme'):
    """
    Load R² results with proper seed alignment.
    Returns dict with aligned lists for paired comparison.
    """
    # Store by seed first
    nasm_by_seed = {seed: {} for seed in SEEDS}
    mlp_by_seed = {seed: {} for seed in SEEDS}
    
    for seed in SEEDS:
        # NASM results (folder named 'kan' but model is NASM)
        nasm_path = RESULTS_DIR / NASM_FOLDER / f'{level}_seed{seed}' / 'results.json'
        if nasm_path.exists():
            with open(nasm_path) as f:
                data = json.load(f)
            metrics = data.get('test_metrics', data.get('val_metrics', {}))
            for target in TARGETS:
                if target in metrics:
                    nasm_by_seed[seed][target] = metrics[target]['r2']
        
        # MLP results
        mlp_path = RESULTS_DIR / 'mlp' / f'{level}_seed{seed}' / 'results.json'
        if mlp_path.exists():
            with open(mlp_path) as f:
                data = json.load(f)
            metrics = data.get('test_metrics', data.get('val_metrics', {}))
            for target in TARGETS:
                if target in metrics:
                    mlp_by_seed[seed][target] = metrics[target]['r2']
    
    # Build aligned lists (only include seeds with both models)
    results = {'nasm': {t: [] for t in TARGETS}, 
               'mlp': {t: [] for t in TARGETS},
               'aligned_seeds': []}
    
    for seed in SEEDS:
        # Check if seed has data for both models for all targets
        has_nasm = all(t in nasm_by_seed[seed] for t in TARGETS)
        has_mlp = all(t in mlp_by_seed[seed] for t in TARGETS)
        
        if has_nasm and has_mlp:
            results['aligned_seeds'].append(seed)
            for target in TARGETS:
                results['nasm'][target].append(nasm_by_seed[seed][target])
                results['mlp'][target].append(mlp_by_seed[seed][target])
    
    return results


def load_statistical_results(level: str = 'phoneme'):
    """Load statistical test results from JSON."""
    stats_path = STATS_DIR / f'{level}_statistical_results.json'
    if stats_path.exists():
        with open(stats_path) as f:
            data = json.load(f)
        # Convert list to dict by target (fixed order)
        if isinstance(data, list):
            return {item['target']: item for item in data}
        return data
    return None


def compute_ci(values, confidence=0.95):
    """
    Compute confidence interval using t-distribution.
    Uses scipy.stats.sem for standard error calculation.
    """
    n = len(values)
    if n < 2:
        return 0
    se = scipy_stats.sem(values)  # standard error of the mean
    t_crit = scipy_stats.t.ppf((1 + confidence) / 2, n - 1)
    return t_crit * se


# =============================================================================
# Visualization Functions
# =============================================================================

def create_paired_seed_plot(results, level: str, output_dir: Path):
    """
    Create paired seed plot (slopegraph) - MAIN FIGURE.
    Each line connects the same seed across models.
    Reference: https://www.geeksforgeeks.org/line-chart-in-matplotlib-python/
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    n_seeds = len(results['aligned_seeds'])
    
    for i, target in enumerate(TARGETS):
        ax = axes[i]
        nasm_vals = results['nasm'][target]
        mlp_vals = results['mlp'][target]
        
        if n_seeds == 0:
            ax.text(0.5, 0.5, 'No paired data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Plot paired lines
        for j in range(n_seeds):
            ax.plot([0, 1], [nasm_vals[j], mlp_vals[j]], 
                   color='gray', alpha=0.4, linewidth=1, zorder=1)
        
        # Plot points
        ax.scatter([0]*n_seeds, nasm_vals, color=COLORS['nasm'], s=80, 
                  marker='o', label='NASM', zorder=2, edgecolors='white')
        ax.scatter([1]*n_seeds, mlp_vals, color=COLORS['mlp'], s=80, 
                  marker='s', label='MLP', zorder=2, edgecolors='white')
        
        # Plot means with t-based CI
        nasm_mean, mlp_mean = np.mean(nasm_vals), np.mean(mlp_vals)
        nasm_ci, mlp_ci = compute_ci(nasm_vals), compute_ci(mlp_vals)
        
        ax.errorbar([0], [nasm_mean], yerr=[nasm_ci], color=COLORS['nasm'], 
                   marker='D', markersize=12, capsize=5, capthick=2, linewidth=2, zorder=3)
        ax.errorbar([1], [mlp_mean], yerr=[mlp_ci], color=COLORS['mlp'], 
                   marker='D', markersize=12, capsize=5, capthick=2, linewidth=2, zorder=3)
        
        ax.set_xlim(-0.3, 1.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['NASM', 'MLP'], fontsize=12)
        ax.set_ylabel('$R^2$', fontsize=12)
        ax.set_title(f'{TARGET_DISPLAY[target]}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Dynamic y-axis
        all_vals = nasm_vals + mlp_vals
        ax.set_ylim(max(0, min(all_vals) - 0.1), min(1, max(all_vals) + 0.1))
        
        if i == 0:
            ax.legend(loc='lower left', fontsize=10)
    
    plt.suptitle(f'Paired Seed Comparison ({level.capitalize()} Level, n={n_seeds})\n'
                 'Lines connect same seed; diamonds = mean ± 95% CI (t-based)', fontsize=12, y=1.02)
    plt.tight_layout()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'paired_seeds_{level}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'paired_seeds_{level}.pdf', bbox_inches='tight')
    plt.close()


def create_effect_size_plot(stats, level: str, output_dir: Path):
    """
    Create effect size plot with Cohen's dz.
    Reference: https://www.geeksforgeeks.org/effect-size/
    """
    if stats is None:
        print(f"  No statistical results for {level}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Fixed order
    cohens_dz = []
    significant = []
    
    for target in TARGETS:
        if target not in stats:
            cohens_dz.append(0)
            significant.append(False)
            continue
        item = stats[target]
        dz = item.get('dz', item.get('cohens_dz', 0))
        cohens_dz.append(dz)
        significant.append(item.get('significant_bonf', False))
    
    y_pos = np.arange(len(TARGETS))
    colors = [COLORS['nasm'] if d > 0 else COLORS['mlp'] for d in cohens_dz]
    
    ax.barh(y_pos, cohens_dz, color=colors, alpha=0.7, height=0.6)
    
    # Significance markers
    for i, (dz, sig) in enumerate(zip(cohens_dz, significant)):
        if sig:
            ax.text(dz + (0.1 if dz > 0 else -0.1), i, '*', 
                   fontsize=20, ha='center', va='center', fontweight='bold')
        ax.annotate(f'{dz:.2f}', xy=(dz + (0.05 if dz >= 0 else -0.05), i), 
                   va='center', ha='left' if dz >= 0 else 'right', fontsize=10)
    
    # Reference lines
    ax.axvline(x=0, color='black', linewidth=1)
    for thresh in [0.2, 0.5, 0.8]:
        ax.axvline(x=thresh, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(x=-thresh, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([TARGET_DISPLAY[t] for t in TARGETS], fontsize=12)
    ax.set_xlabel("Cohen's $d_z$ (paired effect size)", fontsize=12)
    ax.set_title(f'Effect Size ({level.capitalize()} Level)\n'
                 'Positive = NASM better; * = significant (Bonferroni p<0.0167)', fontsize=14, fontweight='bold')
    
    ax.text(0.02, 0.02, '|$d_z$|: <0.2=small, 0.5=medium, 0.8=large',
            transform=ax.transAxes, fontsize=9, va='bottom', style='italic', color='gray')
    
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'effect_sizes_{level}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'effect_sizes_{level}.pdf', bbox_inches='tight')
    plt.close()


def create_seed_stability_boxplots(results, level: str, output_dir: Path):
    """
    Create boxplots showing R² distribution across seeds.
    Reference: https://www.geeksforgeeks.org/box-plot-in-python-using-matplotlib/
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    for i, target in enumerate(TARGETS):
        ax = axes[i]
        nasm_vals = results['nasm'][target]
        mlp_vals = results['mlp'][target]
        
        if not nasm_vals or not mlp_vals:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        bp = ax.boxplot([nasm_vals, mlp_vals], tick_labels=['NASM', 'MLP'],
                       patch_artist=True, widths=0.6)
        
        bp['boxes'][0].set_facecolor(COLORS['nasm'])
        bp['boxes'][1].set_facecolor(COLORS['mlp'])
        for box in bp['boxes']:
            box.set_alpha(0.7)
        
        # Jittered points
        for j, (vals, color) in enumerate([(nasm_vals, COLORS['nasm']), (mlp_vals, COLORS['mlp'])]):
            x = np.random.normal(j+1, 0.04, size=len(vals))
            ax.scatter(x, vals, alpha=0.6, color=color, s=50, zorder=3, edgecolors='white')
        
        ax.set_title(f'{TARGET_DISPLAY[target]}', fontsize=14, fontweight='bold')
        ax.set_ylabel('$R^2$', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # SD annotation (sample SD with ddof=1)
        ax.text(0.02, 0.98, f'SD: NASM={np.std(nasm_vals, ddof=1):.4f}, MLP={np.std(mlp_vals, ddof=1):.4f}',
               transform=ax.transAxes, fontsize=8, va='top', style='italic')
    
    n_seeds = len(results['aligned_seeds'])
    plt.suptitle(f'$R^2$ Distribution ({level.capitalize()} Level, n={n_seeds})', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'seed_stability_{level}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'seed_stability_{level}.pdf', bbox_inches='tight')
    plt.close()


# =============================================================================
# Main
# =============================================================================

# =============================================================================
# Spline Visualization (using trained model's extraction API)
# =============================================================================

# Feature names and display names
FEATURE_NAMES = [
    'primary_stress_pos', 'word_frequency', 'syllable_count', 'phoneme_count',
    'pos_noun', 'pos_verb', 'pos_adjective', 'pos_adverb',
    'pos_pronoun', 'pos_preposition', 'pos_determiner', 'pos_auxiliary',
    'syllable_initial', 'syllable_final', 'is_stressed',
    'is_phrase_boundary', 'is_word_boundary',
    'context_sentence_position', 'context_word_position',
    'is_vowel', 'is_voiced', 'is_plosive',
    'is_schwa', 'velar_fricative_next', 'vowel_height_high', 'vowel_height_low', 'vowel_tense',
    'sonority_of_nucleus', 'distance_to_stress_norm', 'stress_pattern_class'
]

DISPLAY_NAMES = {
    'primary_stress_pos': 'Primary Stress Position',
    'word_frequency': 'Word Frequency',
    'syllable_count': 'Syllable Count',
    'phoneme_count': 'Phoneme Count',
    'is_stressed': 'Stressed',
    'is_phrase_boundary': 'Phrase Boundary',
    'is_word_boundary': 'Word Boundary',
    'syllable_final': 'Syllable-Final',
    'context_sentence_position': 'Sentence Position',
    'is_vowel': 'Vowel',
    'vowel_height_high': 'High Vowel',
    'vowel_height_low': 'Low Vowel',
    'sonority_of_nucleus': 'Sonority',
    'distance_to_stress_norm': 'Distance to Stress',
}

# Key features for multipanel plots (Chapter 6)
# Selected for: stable curves, strong shape, literature anchors
KEY_FEATURES = {
    'f0': [17, 0, 28, 24, 14, 18],      # sentence_pos, stress_pos, dist_to_stress, high_vowel, is_stressed, word_pos
    'duration': [3, 2, 15, 17, 18, 1],  # phone_count, syl_count, phrase_boundary, sent_pos, word_pos, word_freq
    'energy': [1, 15, 27, 19, 17, 14]   # word_freq, phrase_boundary, sonority, is_vowel, sent_pos, is_stressed
}


def load_model_class():
    """Load the actual model class from training script."""
    training_dir = PROJECT_ROOT / "scripts" / "training"
    
    # Load TrueKANHead via exec (identical to training script import method)
    # This ensures we use exactly the same model class as training
    _kan = {}
    exec(open(training_dir / "true_kan_heads_vectorized.py").read(), _kan)
    TrueKANHead = _kan['TrueKANHead']
    
    # Define model wrapper (same as training)
    class KANProsodyPredictor(torch.nn.Module):
        def __init__(self, num_basis=8, spline_degree=3):
            super().__init__()
            self.kan = TrueKANHead(
                in_dim=30, out_dim=3, num_basis=num_basis, degree=spline_degree,
                grid_range=(0, 1), enable_interpretability=True, learn_base_linear=False
            )
        
        def forward(self, features, attention_mask=None):
            return self.kan(features)
    
    return KANProsodyPredictor


def load_trained_models(level: str):
    """Load all trained NASM models for a given level."""
    ModelClass = load_model_class()
    models = {}
    
    for seed in SEEDS:
        model_path = RESULTS_DIR / NASM_FOLDER / f'{level}_seed{seed}' / 'best_model.pt'
        if not model_path.exists():
            print(f"    Model not found: {model_path.name}")
            continue
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            model = ModelClass(num_basis=8, spline_degree=3)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            models[seed] = model
        except Exception as e:
            print(f"    Failed to load seed {seed}: {e}")
    
    return models


def extract_spline_curves(models, feature_idx, target_idx, n_points=200, center=True):
    """
    Extract spline curves across all seeds using model's native method.
    
    If center=True, each spline is mean-centered:
        f_centered(x) = f(x) - mean(f(x))
    
    This is valid because additive spline components are identifiable only up to
    a constant offset (absorbed by the bias term). Mean-centering makes shapes
    comparable across seeds without changing predictions.
    """
    curves = []
    x_grid = None
    
    for seed, model in models.items():
        try:
            xs, ys = model.kan.get_curve_for_plotting(feature_idx, target_idx, n_points)
            # Safe tensor -> numpy conversion (handles GPU/grad)
            ys_np = ys.detach().cpu().numpy()
            
            # Mean-center each seed's curve (additive components are defined up to constant)
            if center:
                ys_np = ys_np - np.mean(ys_np)
            
            curves.append(ys_np)
            if x_grid is None:
                x_grid = xs.detach().cpu().numpy()
        except Exception as e:
            pass
    
    if not curves:
        return None, None, None
    
    curves_array = np.array(curves)
    return x_grid, np.mean(curves_array, axis=0), np.std(curves_array, axis=0, ddof=1)


def create_multipanel_splines(models, level: str, output_dir: Path):
    """Create 2×3 multipanel plots for key features (main thesis figures)."""
    spline_dir = output_dir / 'splines'
    spline_dir.mkdir(parents=True, exist_ok=True)
    
    for target_idx, target in enumerate(TARGETS):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()
        
        key_indices = KEY_FEATURES[target]
        
        for panel_idx, feature_idx in enumerate(key_indices):
            ax = axes[panel_idx]
            
            x, mean, std = extract_spline_curves(models, feature_idx, target_idx)
            
            if x is not None:
                ax.fill_between(x, mean - std, mean + std, alpha=0.3, color=COLORS['nasm'], label='±1 SD')
                ax.plot(x, mean, color=COLORS['nasm'], linewidth=2, label='Mean')
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                
                fname = FEATURE_NAMES[feature_idx]
                ax.set_title(DISPLAY_NAMES.get(fname, fname), fontsize=11, fontweight='bold')
                ax.set_xlabel('Feature value (min-max [0,1])', fontsize=9)
                ax.set_ylabel('Mean-centred contribution', fontsize=9)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(FEATURE_NAMES[feature_idx], fontsize=11)
        
        plt.suptitle(f'{TARGET_DISPLAY[target]} - Mean-Centred Additive Contributions ({level.capitalize()} Level)\n'
                     f'Mean ± 1 SD across {len(models)} seeds', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(spline_dir / f'multipanel_{target}_{level}.png', dpi=150, bbox_inches='tight')
        plt.savefig(spline_dir / f'multipanel_{target}_{level}.pdf', bbox_inches='tight')
        plt.close()


def create_summary_grid(models, level: str, output_dir: Path):
    """Create 6×5 grid of all 30 features (appendix figures)."""
    appendix_dir = output_dir / 'splines' / 'appendix'
    appendix_dir.mkdir(parents=True, exist_ok=True)
    
    for target_idx, target in enumerate(TARGETS):
        fig, axes = plt.subplots(6, 5, figsize=(18, 22))
        axes = axes.flatten()
        
        for feature_idx in range(30):
            ax = axes[feature_idx]
            x, mean, std = extract_spline_curves(models, feature_idx, target_idx)
            
            if x is not None:
                ax.fill_between(x, mean - std, mean + std, alpha=0.3, color=COLORS['nasm'])
                ax.plot(x, mean, color=COLORS['nasm'], linewidth=1.5)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
                ax.set_title(FEATURE_NAMES[feature_idx], fontsize=8)
                ax.tick_params(axis='both', labelsize=6)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=8)
                ax.set_title(FEATURE_NAMES[feature_idx], fontsize=8)
        
        plt.suptitle(f'{TARGET_DISPLAY[target]} - All Feature Splines ({level.capitalize()} Level, centred)', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(appendix_dir / f'summary_grid_{target}_{level}.png', dpi=150, bbox_inches='tight')
        plt.savefig(appendix_dir / f'summary_grid_{target}_{level}.pdf', bbox_inches='tight')
        plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate thesis visualizations')
    parser.add_argument('--level', choices=['phoneme', 'word', 'both'], default='both')
    parser.add_argument('--skip-splines', action='store_true', help='Skip spline generation')
    args = parser.parse_args()
    
    levels = ['phoneme', 'word'] if args.level == 'both' else [args.level]
    
    print("=" * 60)
    print("Generate Thesis Visualizations")
    print("=" * 60)
    
    for level in levels:
        print(f"\n[{level.upper()}]")
        
        level_output = OUTPUT_DIR / level
        level_output.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print("  Loading results...")
        results = load_results(level)
        stats = load_statistical_results(level)
        
        n_seeds = len(results['aligned_seeds'])
        print(f"  Aligned seeds: {n_seeds} ({results['aligned_seeds']})")
        
        # Create comparison figures
        print("  Creating paired seed plot...")
        create_paired_seed_plot(results, level, level_output)
        
        print("  Creating effect size plot...")
        create_effect_size_plot(stats, level, level_output)
        
        print("  Creating seed stability boxplots...")
        create_seed_stability_boxplots(results, level, level_output)
        
        # Spline figures (interpretability)
        if not args.skip_splines:
            print("  Loading trained models for splines...")
            models = load_trained_models(level)
            
            if models:
                print(f"    Loaded {len(models)}/{len(SEEDS)} models")
                print("  Creating multipanel spline plots (main figures)...")
                create_multipanel_splines(models, level, level_output)
                
                print("  Creating summary grid (appendix)...")
                create_summary_grid(models, level, level_output)
            else:
                print("    No models loaded, skipping splines")
    
    print(f"\nDone! Saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
